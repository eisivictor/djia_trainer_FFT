import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, LSTM, Dropout, Conv1D, Flatten, Reshape
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
import random
from collections import deque


# Create the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, epsilon_initial, 
                 memory_size, epsilon_final, epsilon_decay_steps,
                 gamma, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon_initial  # exploration rate
        self.epsilon_min = epsilon_final
        self.epsilon_decay = (epsilon_initial - epsilon_final) / epsilon_decay_steps
        self.learning_rate = learning_rate
        self.model = self._build_model_new()
        self.target_model = self._build_model_new()
        self.update_target_model()

    def _build_model_new(self):
        model = Sequential([
            # Explicitly define input layer
            Input(shape=(self.state_size[0], self.state_size[1])),
            
            # First LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.2),

            # Second LSTM layer
            LSTM(64),
            Dropout(0.2),

            # First 2 Conv layers
            Reshape((8, 8)),  # Reshape to proper dimensions for Conv1D
            Conv1D(32, kernel_size=3, padding='same', activation='relu'),
            Conv1D(32, kernel_size=3, padding='same', activation='relu'),

            # Dense layer in the middle
            Flatten(),
            Dense(64, activation='relu'),

            # Second 2 Conv layers
            Reshape((8, 8)),  # Adjust dimensions as needed
            Conv1D(32, kernel_size=3, padding='same', activation='relu'),
            Conv1D(32, kernel_size=3, padding='same', activation='relu'),
            Flatten(),

            # Final dense layers
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])

        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _build_model(self):
        # Neural Net for Deep-Q learning
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.state_size[0], self.state_size[1]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=tf.keras.losses.mean_squared_error(), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def reset_epsilon(self, value):
        self.epsilon = value  # exploration rate

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_training=True):
        if is_training and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        # Reshape state for prediction
        state = np.reshape(state, (1, state.shape[0], state.shape[1]))
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size[0], self.state_size[1]))
        next_states = np.zeros((batch_size, self.state_size[0], self.state_size[1]))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state

        # Predict Q-values for current states and next states
        targets = self.model.predict(states, verbose=0)
        next_target_values = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward if done else reward + self.gamma * np.amax(next_target_values[i])
            targets[i][action] = target

        # Train model
        self.model.fit(states, targets, epochs=1, verbose=0)

        #if self.epsilon > self.epsilon_min:
        #    self.epsilon -= self.epsilon_decay

    def load(self, name):
        #self.model = self._build_model_new()
        #self.model.load_weights(name)  
        self.model = tf.keras.models.load_model(name)


    def save(self, name):
        self.model.save(name)

# Custom environment for stock trading
class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, lookback_window_size=20, use_fft=False):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.use_fft = use_fft  # New flag to control FFT usage

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        self.features = ['Open', 'High', 'Low', 'Close', 'Volume',
                   'sma7', 'sma20', 'ema12', 'ema26', 'rsi',
                   'macd', 'macd_signal', 'macd_hist',
                   'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
                   'adx', 'plus_di', 'minus_di',
                   'stoch_k', 'stoch_d',
                   'vwap', 'atr', 'obv', 'psar',
                   'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']

        # Observation space: OHLCV data + technical indicators + account info
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(lookback_window_size, len(self.features)), dtype=np.float32
        )

        # Episode variables
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.current_price = 0

        # Configure trading fees
        self.buying_fee_pct = 0.01
        self.selling_fee_pct = 0.01

        # Transaction history
        self.transactions = []

        # Data processing
        self.process_data()

    def process_data(self):
        # Add technical indicators
        df = self.df.copy()

        # SMA
        df['sma7'] = df['Close'].rolling(window=7).mean()
        df['sma20'] = df['Close'].rolling(window=20).mean()

        # EMA
        df['ema12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 1. Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # 2. ADX (Average Directional Index)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
        minus_dm = minus_dm.abs().where((minus_dm > 0) & (minus_dm > plus_dm), 0)

        df['plus_di'] = 100 * (plus_dm.rolling(window=14).mean() / df['atr'])
        df['minus_di'] = 100 * (minus_dm.rolling(window=14).mean() / df['atr'])
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=14).mean()

        # 3. Stochastic Oscillator
        n = 14
        df['stoch_k'] = 100 * ((df['Close'] - df['Low'].rolling(window=n).min()) /
                            (df['High'].rolling(window=n).max() - df['Low'].rolling(window=n).min()))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # 4. VWAP (Volume-Weighted Average Price)
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        # 5. ATR (Average True Range)
        # Already calculated for ADX

        # 6. OBV (On-Balance Volume)
        df['obv'] = (df['Volume'] * (df['Close'].diff() > 0).astype(int) -
                     df['Volume'] * (df['Close'].diff() < 0).astype(int)).cumsum()

        # 7. Parabolic SAR
        df['psar'] = df['Close'].copy()  # Placeholder for simplicity
        acceleration_factor = 0.02
        max_acceleration = 0.2

        # Initialize values
        is_uptrend = True
        af = acceleration_factor

        #from IPython.core.debugger import Pdb; Pdb().set_trace()

        ep = df['Low'].iloc[0]  # Extreme point
        psar = df['High'].iloc[0]  # Initial PSAR value

        for i in range(2, len(df)):
            # Update PSAR
            if is_uptrend:
                df.loc[df.index[i], 'psar'] = psar + af * (ep - psar)
                # Ensure PSAR is below low
                df.loc[df.index[i], 'psar'] = min(df.loc[df.index[i], 'psar'],
                                                 df.loc[df.index[i-1], 'Low'],
                                                 df.loc[df.index[i-2], 'Low'])

                # Check for trend reversal
                if df.loc[df.index[i], 'Low'] < df.loc[df.index[i], 'psar']:
                    is_uptrend = False
                    psar = ep
                    ep = df.loc[df.index[i], 'Low']
                    af = acceleration_factor
                else:
                    # Update extreme point and acceleration factor
                    if df.loc[df.index[i], 'High'] > ep:
                        ep = df.loc[df.index[i], 'High']
                        af = min(af + acceleration_factor, max_acceleration)
            else:
                df.loc[df.index[i], 'psar'] = psar - af * (psar - ep)
                # Ensure PSAR is above high
                df.loc[df.index[i], 'psar'] = max(df.loc[df.index[i], 'psar'],
                                                 df.loc[df.index[i-1], 'High'],
                                                 df.loc[df.index[i-2], 'High'])

                # Check for trend reversal
                if df.loc[df.index[i], 'High'] > df.loc[df.index[i], 'psar']:
                    is_uptrend = True
                    psar = ep
                    ep = df.loc[df.index[i], 'High']
                    af = acceleration_factor
                else:
                    # Update extreme point and acceleration factor
                    if df.loc[df.index[i], 'Low'] < ep:
                        ep = df.loc[df.index[i], 'Low']
                        af = min(af + acceleration_factor, max_acceleration)

        # 8. Ichimoku Cloud components
        df['tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
        df['chikou_span'] = df['Close'].shift(-26)

        # Replace NaN values
        df.fillna(0, inplace=True)

        # Normalize data
        self.scaler = MinMaxScaler()

        df[self.features] = self.scaler.fit_transform(df[self.features])

        self.processed_df = df


    def reset(self):
        # Reset environment state
        self.current_step = self.lookback_window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance

        return self._get_observation()

    def _get_observation(self):
        # Get data for current observation window
        start = max(0, self.current_step - self.lookback_window_size)
        end = self.current_step

        obs = self.processed_df[self.features].iloc[start:end].to_numpy()

        # Pad observation if needed
        if len(obs) < self.lookback_window_size:
            padding = np.zeros((self.lookback_window_size - len(obs), len(self.features)))
            obs = np.vstack([padding, obs])

        # If FFT is enabled, transform each feature in the window
        if self.use_fft:
            # obs shape: (lookback_window_size, n_features)
            # We want to apply FFT to each column (feature) over the window
            fft_obs = np.abs(np.fft.fft(obs, axis=0))
            # Only keep the first half of the FFT (real spectrum)
            fft_obs = fft_obs[:self.lookback_window_size // 2, :]
            # Pad if needed to keep shape consistent
            if fft_obs.shape[0] < self.lookback_window_size:
                pad_rows = self.lookback_window_size - fft_obs.shape[0]
                fft_obs = np.vstack([fft_obs, np.zeros((pad_rows, fft_obs.shape[1]))])
            obs = fft_obs

        return obs


    def step(self, action):
        # Get current price from the dataframe
        self.current_price = self.df.iloc[self.current_step]['Close']

        # Get current date
        current_date = self.df.index[self.current_step]

        # Execute action
        if action == 1:  # Buy
            # Calculate max shares that can be bought
            max_possible_shares = self.balance // self.current_price
            if max_possible_shares > 0:
                # Apply buying fee
                buying_fee = max_possible_shares * self.current_price * self.buying_fee_pct
                effective_balance = self.balance - buying_fee

                # Recalculate max shares after fee
                max_shares = effective_balance // self.current_price

                if max_shares > 0:
                    cost = max_shares * self.current_price
                    total_cost = cost + (cost * self.buying_fee_pct)

                    self.shares_held += max_shares
                    self.balance -= total_cost

                    # Record transaction
                    self.transactions.append({
                        'date': current_date,
                        'type': 'BUY',
                        'shares': max_shares,
                        'price': self.current_price,
                        'cost': total_cost,
                        'fee': cost * self.buying_fee_pct
                    })

        elif action == 2:  # Sell
            if self.shares_held > 0:
                sale_value = self.shares_held * self.current_price
                fee = sale_value * self.selling_fee_pct
                net_sale_value = sale_value - fee

                # Record transaction before updating shares
                self.transactions.append({
                    'date': current_date,
                    'type': 'SELL',
                    'shares': self.shares_held,
                    'price': self.current_price,
                    'value': net_sale_value,
                    'fee': fee
                })

                self.balance += net_sale_value
                self.shares_held = 0

        # Update net worth
        self.net_worth = self.balance + self.shares_held * self.current_price

        # Move to next step
        self.current_step += 1

        # Check if done
        done = self.current_step > len(self.df) - 1

        # Calculate reward based on change in net worth
        prev_networth = self.net_worth
        if not done:
            self.current_price = self.df.iloc[self.current_step]['Close']
            current_networth = self.balance + self.shares_held * self.current_price
            reward = (current_networth - prev_networth) / prev_networth  # Percentage change
        else:
            reward = 0

        # Get new observation
        obs = self._get_observation()

        # Additional info
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'transactions': self.transactions
        }

        return obs, reward, done, info


    def step_old(self, action):
        # Get current price from the dataframe
        self.current_price = self.df.iloc[self.current_step]['Close']

        #from IPython.core.debugger import Pdb; Pdb().set_trace()


        # Execute action
        if action == 1:  # Buy
            # Calculate max shares that can be bought
            max_shares = self.balance // self.current_price
            if max_shares > 0:
                self.shares_held += max_shares
                self.balance -= max_shares * self.current_price

        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * self.current_price
                self.shares_held = 0

        # Update net worth
        self.net_worth = self.balance + self.shares_held * self.current_price

        # Move to next step
        self.current_step += 1

        # Check if done
        done = self.current_step >= len(self.df) - 1

        # Calculate reward based on change in net worth
        prev_networth = self.net_worth
        if not done:
            self.current_price = self.df.iloc[self.current_step]['Close']
            current_networth = self.balance + self.shares_held * self.current_price
            reward = (current_networth - prev_networth) / prev_networth  # Percentage change
        else:
            reward = 0

        # Get new observation
        obs = self._get_observation()

        # Additional info
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth
        }

        return obs, reward, done, info

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: ${self.balance:.2f}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net worth: ${self.net_worth:.2f}')
        print(f'Current price: ${self.current_price:.2f}')
        print('-' * 30)


# Function to train the agent
def train_agent(logger, env, agent, episodes, batch_size, update_target_every=10):
    scores = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        info = {}
        info['net_worth'] = [0]      

        while not done:
            # Get action
            action = agent.act(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Store in replay memory
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            # Train the agent
            agent.replay(batch_size)            

        # Update target network
        if e % update_target_every == 0:
            agent.update_target_model()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon -= agent.epsilon_decay

        scores.append(info['net_worth'])

        logger.info(f"Episode: {e+1}/{episodes}, Net Worth: ${info['net_worth']:.2f}, " +
              f"Epsilon: {agent.epsilon:.4f}")

    return scores

# Function to test the agent
def test_agent(logger, env, agent):
    state = env.reset()
    done = False

    balance_history = [env.balance]
    net_worth_history = [env.net_worth]
    transactions = []

    while not done:
        action = agent.act(state, is_training=False)
        #print(f"VICTOR - current_step={env.current_step} data={env.processed_df.iloc[env.current_step]} action={action} balance={env.balance} shares_held={env.shares_held} net_worth={env.net_worth}")
        next_state, reward, done, info = env.step(action)
        state = next_state

        balance_history.append(info['balance'])
        net_worth_history.append(info['net_worth'])

    transactions = info['transactions']

    # Calculate returns
    initial_value = net_worth_history[0]
    final_value = net_worth_history[-1]
    percent_return = ((final_value - initial_value) / initial_value) * 100

    logger.info(f"Starting Balance: ${initial_value:.2f}")
    logger.info(f"Final Net Worth: ${final_value:.2f}")
    logger.info(f"Return: {percent_return:.2f}%")

    return balance_history, net_worth_history, transactions

# Plot results
def plot_results(name, df, balance_history, net_worth_history, transactions):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)  # Use sharex=True to align x-axes

    # Determine the common time period (when net worth history starts)
    test_period = df.index[-len(net_worth_history):]
    start_date = test_period[0]

    # Filter stock data to match the test period
    df_test = df.loc[start_date:]

    # Plot stock price (only for test period)
    ax1.plot(df_test.index, df_test['Close'])
    ax1.set_title('MSFT Stock Price')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True)

    # Add buy/sell markers on stock price graph
    for transaction in transactions:  # Note: accessing first element since transactions is a nested list
        date = transaction['date']
        price = transaction['price']
        # Only add markers if the date is within our test period
        if date >= start_date:
            if transaction['type'] == 'BUY':
                ax1.scatter(date, price, color='green', marker='^', s=100, label='Buy')
            elif transaction['type'] == 'SELL':
                ax1.scatter(date, price, color='red', marker='v', s=100, label='Sell')

    # Add legend without duplicate entries for stock price graph
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='best')

    # Plot net worth over time
    ax2.plot(test_period, net_worth_history)
    ax2.set_title('Agent Net Worth Over Time')
    ax2.set_ylabel('Net Worth ($)')
    ax2.set_xlabel('Date')
    ax2.grid(True)

    # Add buy/sell markers on net worth graph
    test_period_dict = {date: i for i, date in enumerate(test_period)}
    for transaction in transactions:
        date = transaction['date']
        # Check if transaction date is in the test period
        if date in test_period_dict:
            idx = test_period_dict[date]
            if idx < len(net_worth_history):  # Make sure we don't go out of bounds
                net_worth = net_worth_history[idx]
                if transaction['type'] == 'BUY':
                    ax2.scatter(date, net_worth, color='green', marker='^', s=100, label='Buy')
                elif transaction['type'] == 'SELL':
                    ax2.scatter(date, net_worth, color='red', marker='v', s=100, label='Sell')

    # Add legend without duplicate entries for net worth graph
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='best')

    # Format the x-axis to better display dates
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(f'{name}_trading_results.png')
    #plt.show()

# Plot results
def plot_results_old(df, balance_history, net_worth_history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot stock price
    ax1.plot(df.index, df['Close'])
    ax1.set_title('MSFT Stock Price')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True)

    # Plot net worth over time
    test_period = df.index[-len(net_worth_history):]
    ax2.plot(test_period, net_worth_history)
    ax2.set_title('Agent Net Worth Over Time')
    ax2.set_ylabel('Net Worth ($)')
    ax2.set_xlabel('Date')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('msft_trading_results.png')
    plt.show()

def testing_agent(logger, name, test_env, agent, data_frame, loopback_window_size, initial_capital):
    logger.info("\nTesting the agent...")
    agent.epsilon = 0
    
    balance_history, net_worth_history, transactions = test_agent(logger, test_env, agent)

    # Plot results
    plot_results(name, data_frame, balance_history, net_worth_history, transactions)

    # Calculate buy and hold strategy returns for comparison
    buy_and_hold_initial = data_frame.iloc[loopback_window_size]['Close']
    buy_and_hold_final = data_frame.iloc[-1]['Close']
    buy_and_hold_shares = initial_capital / buy_and_hold_initial
    buy_and_hold_value = buy_and_hold_shares * buy_and_hold_final
    buy_and_hold_return = ((buy_and_hold_value - initial_capital) / initial_capital) * 100

    logger.info("\nBuy and Hold Strategy:")
    logger.info(f"Starting Balance: ${initial_capital:.2f}")
    logger.info(f"Final Value: ${buy_and_hold_value:.2f}")
    logger.info(f"Return: {buy_and_hold_return:.2f}%")

    rl_agent_return = ((net_worth_history[-1] - initial_capital) / initial_capital) * 100
    logger.info("\nRL Agent vs Buy and Hold:")
    logger.info(f"RL Agent Return: {rl_agent_return:.2f}%")
    logger.info(f"Buy and Hold Return: {buy_and_hold_return:.2f}%")

    # Return transactions along with other results
    return {
        "balance_history": balance_history,
        "net_worth_history": net_worth_history,
        "transactions": transactions,
        "rl_agent_return": rl_agent_return,
        "buy_and_hold_return": buy_and_hold_return
    }


# Function to fetch historical data
def get_historical_data(ticker, period=5):
    import os
    import pandas as pd
    import yfinance as yf
    
    # Directory for saving/loading historical data
    folder_path = "djia_historical_data"
    filename = os.path.join(folder_path, f"{ticker.replace('.', '_').lower()}_historical_data.csv")
    
    # Check environment variables
    save_to_file = os.environ.get('SAVE_DATA_TO_FILE', 'false').lower() == 'true'
    load_from_file = os.environ.get('LOAD_DATA_FROM_FILE', 'false').lower() == 'true'
    
    # Try to load from file if the environment variable is set and the file exists
    if load_from_file and os.path.exists(filename):
        try:
            data = pd.read_csv(filename, index_col=0, parse_dates=True)
            print(f"Data loaded from {filename}")
            return data
        except Exception as e:
            print(f"Error loading from file: {e}. Falling back to downloading data.")
    
    # Download data if not loading from file or if loading failed
    start_date = pd.Timestamp.now() - pd.DateOffset(years=period)
    #start_date = pd.Timestamp.now() - pd.DateOffset(days=100)

    #data = yf.download(ticker, period=period, end=end_date)
    data = yf.download(ticker)
    
    # Check if data was downloaded successfully (not empty)
    if not data.empty:
        data.columns = [col[0] for col in data.columns]
        # Filter the DataFrame to include only data from the last year
        data = data[data.index >= start_date]
        
        # Save to file if environment variable is set
        if save_to_file:
            # Create the directory if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Save the data
            data.to_csv(filename)
            print(f"Data saved to {filename}")
    
    return data