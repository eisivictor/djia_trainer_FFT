import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import os
import sys
import csv

def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    # Fix MultiIndex columns if present - flatten to single level
    if isinstance(stock_data.columns, pd.MultiIndex):
        print("Converting MultiIndex columns to flat columns")
        # Convert tuples like ('Close', 'AAPL') to just 'Close'
        stock_data.columns = [col[0] for col in stock_data.columns]
    
    return stock_data

def calculate_sma_crossover(data, short_sma=50, long_sma=200):
    """
    Calculate SMAs and identify crossovers
    
    Parameters:
    - data: DataFrame with price data
    - short_sma: Period for the short SMA (default: 50)
    - long_sma: Period for the long SMA (default: 200)
    """
    print(f"Calculating SMAs for data with {len(data)} rows (Short: {short_sma}, Long: {long_sma})")
    
    # Make a deep copy of the data to avoid any reference issues
    data = data.copy(deep=True)
    
    # Print column info for debugging
    print(f"Column types: {type(data.columns)}")
    print(f"Columns before SMA calculation: {data.columns.tolist()}")
    
    # Create column names dynamically
    short_col = f'SMA{short_sma}'
    long_col = f'SMA{long_sma}'
    
    # Calculate the SMAs
    data[short_col] = data['Close'].rolling(window=short_sma).mean()
    data[long_col] = data['Close'].rolling(window=long_sma).mean()
    
    # Print column verification
    print(f"Columns after SMA calculation: {data.columns.tolist()}")
    print(f"SMA columns created: {short_col} and {long_col}")
    
    # Calculate the crossover indicator
    data['SMA_Diff'] = data[short_col] - data[long_col]
    
    # Determine crossover points (where sign of SMA_Diff changes)
    data['SMA_Diff_Prev'] = data['SMA_Diff'].shift(1)
    data['Crossover'] = 0
    
    # Use a safer approach to calculate crossovers - first ensure we have the columns
    try:
        # Use masking approach instead of dropna which seems to cause issues
        mask = ~data[short_col].isna() & ~data[long_col].isna()
        if mask.any():
            # Golden Cross: Short SMA crosses above Long SMA
            data.loc[(data['SMA_Diff'] > 0) & (data['SMA_Diff_Prev'] <= 0) & mask, 'Crossover'] = 1
            # Death Cross: Short SMA crosses below Long SMA
            data.loc[(data['SMA_Diff'] < 0) & (data['SMA_Diff_Prev'] >= 0) & mask, 'Crossover'] = -1
    except Exception as e:
        print(f"Error during crossover calculation: {str(e)}")
        print(f"Data columns after attempt: {data.columns.tolist()}")
    
    # Store SMA periods in the DataFrame for reference
    data.attrs['short_sma'] = short_sma
    data.attrs['long_sma'] = long_sma
    data.attrs['short_col'] = short_col
    data.attrs['long_col'] = long_col
    
    return data

def generate_sma_signals(data):
    """
    Generate buy/sell signals based on SMA crossovers
    - Wait for first Golden Cross before taking any position
    - Buy (1) when SMA50 crosses above SMA200 (Golden Cross)
    - Sell (-1) when SMA50 crosses below SMA200 (Death Cross), but only after first buy
    - Hold (0) otherwise
    """
    # Create a new signal column initialized with zeros (no position)
    data['signal'] = 0
    
    # Find all crossover points
    golden_cross_mask = (data['Crossover'] == 1)
    death_cross_mask = (data['Crossover'] == -1)
    
    if golden_cross_mask.any():
        # Find the index of the first buy signal (Golden Cross)
        first_buy_idx = data[golden_cross_mask].index[0]
        
        # Apply buy signals (all Golden Crosses)
        data.loc[golden_cross_mask, 'signal'] = 1
        
        # Apply sell signals (Death Crosses), but only after the first buy signal
        valid_sell_mask = death_cross_mask & (data.index > first_buy_idx)
        data.loc[valid_sell_mask, 'signal'] = -1
        
        # Forward fill signals to maintain position between crossovers
        # This is essential to hold positions between crossover points
        data.loc[data.index >= first_buy_idx, 'signal'] = data.loc[data.index >= first_buy_idx, 'signal'].ffill()
        
        # Print signal statistics for debugging
        buy_count = (data['signal'] == 1).sum()
        sell_count = (data['signal'] == -1).sum()
        print(f"Signal statistics: Buy/Hold: {buy_count}, Sell/Short: {sell_count}, Total: {len(data)}")
    
    return data

def calculate_returns(data):
    """
    Calculate strategy returns
    """
    # Calculate daily returns (percentage change)
    data['daily_return'] = data['Close'].pct_change()
    
    # Calculate strategy returns based on signals
    data['strategy_return'] = data['signal'].shift(1) * data['daily_return']
    
    # Calculate cumulative returns (starts at 1.0)
    data['cumulative_return'] = (1 + data['daily_return']).cumprod()    
    
    # Add normalized price for comparison (starts at 1.0)
    data['normalized_price'] = data['Close'] / data['Close'].iloc[0]
    
    return data

def calculate_realized_returns(data):
    """
    Calculate strategy returns showing only realized gains after selling
    """
    # Calculate daily returns
    data['daily_return'] = data['Close'].pct_change()
    
    # Create columns for tracking positions and realized returns
    data['position'] = 0  # 0 = no position, 1 = holding position
    data['entry_price'] = np.nan
    data['realized_return'] = 0.0  # Initialize as float instead of int
    data['cumulative_realized_return'] = 1.0
    
    # Track positions and calculate realized returns
    position_open = False
    entry_price = 0
    cumulative_return = 1.0
    
    for i in range(1, len(data)):
        current_signal = data['signal'].iloc[i]
        prev_signal = data['signal'].iloc[i-1]
        
        # Position opened (Buy)
        if current_signal == 1 and prev_signal <= 0:
            position_open = True
            entry_price = data['Close'].iloc[i]
            data.loc[data.index[i], 'position'] = 1
            data.loc[data.index[i], 'entry_price'] = entry_price
        
        # Position maintained
        elif current_signal == 1 and prev_signal == 1:
            data.loc[data.index[i], 'position'] = 1
            data.loc[data.index[i], 'entry_price'] = entry_price
        
        # Position closed (Sell)
        elif current_signal < 0:
            exit_price = data['Close'].iloc[i]
            if position_open:
                # Calculate realized return from this trade
                trade_return = (exit_price / entry_price) - 1
                # Explicitly cast to float to avoid dtype warning
                data.loc[data.index[i], 'realized_return'] = float(trade_return)
                
                # Update cumulative return at sell point only
                cumulative_return *= (1 + trade_return)
                position_open = False
        
        # Update the cumulative_realized_return for all points
        # This will remain flat between sell points
        data.loc[data.index[i], 'cumulative_realized_return'] = cumulative_return
    
    return data

def calculate_metrics(data):
    """
    Calculate performance metrics
    """
    # Annualized return
    days = (data.index[-1] - data.index[0]).days
    ann_return = (data['cumulative_return'].iloc[-1] ** (365/days)) - 1
    
    # Overall return (total return for the entire period)
    overall_return = data['cumulative_return'].iloc[-1] - 1
    
    # Sharpe ratio (annualized)
    ann_volatility = data['strategy_return'].std() * np.sqrt(252)
    sharpe_ratio = ann_return / ann_volatility if ann_volatility != 0 else 0
    
    # Maximum drawdown
    cumulative = data['cumulative_return']
    drawdown = (cumulative / cumulative.cummax()) - 1
    max_drawdown = drawdown.min()
    
    return {
        'Overall Return': overall_return,
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown
    }

def backtest_momentum(ticker, start_date, end_date, window=20, threshold=0):
    """
    Backtest momentum strategy for a given ticker
    """
    # Fetch data
    data = fetch_data(ticker, start_date, end_date)
    
    # Calculate momentum and generate signals
    data = calculate_momentum(data, window)
    data = generate_signals(data, threshold)
    
    # Calculate returns
    data = calculate_returns(data)
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(data['Close'])
    plt.title(f'{ticker} Price')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(data['momentum'])
    plt.axhline(y=threshold, color='g', linestyle='--')
    plt.axhline(y=-threshold, color='r', linestyle='--')
    plt.title(f'{window}-Day Momentum')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(data['cumulative_return'], label='cumulative return')    
    plt.title('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print(f"Backtest Results for {ticker} Momentum Strategy:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    
    # Explain the metrics
    #explain_metrics()
    
    return data, metrics

def backtest_sma_strategy(ticker, start_date, end_date, short_sma=50, long_sma=200):
    """
    Backtest SMA crossover strategy for a given ticker using realized returns
    """
    # Fetch data
    data = fetch_data(ticker, start_date, end_date)
    
    if len(data) < long_sma:
        print(f"Warning: Not enough data points ({len(data)}) for calculating SMA{long_sma}. Need at least {long_sma}.")
        print("Skipping backtest due to insufficient data.")
        return None, {}
    
    # Calculate SMA indicators and generate signals
    data = calculate_sma_crossover(data, short_sma, long_sma)
    
    # Get column names
    short_col = data.attrs.get('short_col', f'SMA{short_sma}')
    long_col = data.attrs.get('long_col', f'SMA{long_sma}')
    
    # Verify SMA columns exist (debugging step)
    missing_columns = []
    if short_col not in data.columns:
        missing_columns.append(short_col)
    if long_col not in data.columns:
        missing_columns.append(long_col)
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print("Available columns:", data.columns.tolist())
        # Add missing columns with NaN values to avoid errors
        for col in missing_columns:
            data[col] = np.nan
    
    # Generate signals
    data = generate_sma_signals(data)
    
    # Calculate returns
    data = calculate_returns(data)
    data = calculate_realized_returns(data)
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    # Create interactive plotly figure with subplots
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=(f'{ticker} Price and SMAs ({short_sma}/{long_sma})', 
                                       'Mark-to-Market Returns (Unrealized)',
                                       'Realized Returns (After Selling)'),
                        vertical_spacing=0.15,
                        shared_xaxes=True)
    
    # Add price to first subplot
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'], 
        name='Price', 
        line=dict(color='black', width=1),
        connectgaps=True
    ), row=1, col=1)
    
    # Safely add SMA lines - check if there are valid data points first
    if not data[short_col].isna().all():
        sma_short_data = data.dropna(subset=[short_col])
        fig.add_trace(go.Scatter(
            x=sma_short_data.index, 
            y=sma_short_data[short_col], 
            name=f'SMA {short_sma}', 
            line=dict(color='blue', width=1.5),
        ), row=1, col=1)
    
    if not data[long_col].isna().all():
        sma_long_data = data.dropna(subset=[long_col])
        fig.add_trace(go.Scatter(
            x=sma_long_data.index, 
            y=sma_long_data[long_col], 
            name=f'SMA {long_sma}', 
            line=dict(color='red', width=1.5),
        ), row=1, col=1)
    
    # Add buy/sell signals as scatter points
    golden_cross = data[data['Crossover'] == 1]
    death_cross = data[data['Crossover'] == -1]
    
    if not golden_cross.empty:
        fig.add_trace(go.Scatter(
            x=golden_cross.index, 
            y=golden_cross[short_col],
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='green'),
            name='Golden Cross (Buy)'
        ), row=1, col=1)
    
    if not death_cross.empty:
        fig.add_trace(go.Scatter(
            x=death_cross.index, 
            y=death_cross[short_col],
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='red'),
            name='Death Cross (Sell)'
        ), row=1, col=1)
    
    # Add mark-to-market returns to second subplot
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['cumulative_return'],
        name='Mark-to-Market Returns',
        line=dict(color='blue', width=1.5)
    ), row=2, col=1)
    
    # Add realized returns to third subplot
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['cumulative_realized_return'],
        name='Realized Returns',
        line=dict(color='green', width=1.5)
    ), row=3, col=1)
    
    # Update layout with improved styling
    fig.update_layout(
        height=1000,  # Taller figure
        width=1200,
        title_text=f"SMA Crossover Strategy Backtest for {ticker}",
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5
        ),
        margin=dict(l=50, r=50, t=80, b=50),  # Adjust margins
        template="plotly_white",  # Use a clean template with grid
    )
    
    # Add grid lines and improve axes
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        rangeslider_visible=False,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray'
    )
    
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray'
    )
    
    # Make y-axes independent for better visualization of each plot
    fig.update_yaxes(autorange=True, fixedrange=False)
    
    # Show the interactive plot
    fig.show()
    
    # Print metrics
    print(f"Backtest Results for {ticker} SMA Crossover Strategy:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    # Explain the metrics
    #explain_metrics()
    
    return data, metrics

def predict_sma_values_tsm(data, buy_prediction_days=7, sell_prediction_days=7, slope_window=5, method='linear'):
    """
    Predict future SMA values using time series models
    
    Parameters:
    - data: DataFrame with price data and SMA columns
    - buy_prediction_days: Days to look ahead for buy signals
    - sell_prediction_days: Days to look ahead for sell signals
    - slope_window: Window used for linear predictions or model training
    - method: Prediction method ('linear', 'arima', or 'exp_smoothing')
    """
    # Get SMA column names from data attributes
    short_sma = data.attrs.get('short_sma', 50)
    long_sma = data.attrs.get('long_sma', 200)
    short_col = data.attrs.get('short_col', f'SMA{short_sma}')
    long_col = data.attrs.get('long_col', f'SMA{long_sma}')
    
    # For storing predictions
    data[f'{short_col}_predicted_buy'] = np.nan
    data[f'{long_col}_predicted_buy'] = np.nan
    data[f'{short_col}_predicted_sell'] = np.nan
    data[f'{long_col}_predicted_sell'] = np.nan
    
    if method == 'linear':
        # Original linear extrapolation method
        data[f'{short_col}_slope'] = data[short_col].diff(slope_window) / slope_window
        data[f'{long_col}_slope'] = data[long_col].diff(slope_window) / slope_window
        
        # Predict future values based on slope
        data[f'{short_col}_predicted_buy'] = data[short_col] + data[f'{short_col}_slope'] * buy_prediction_days
        data[f'{long_col}_predicted_buy'] = data[long_col] + data[f'{long_col}_slope'] * buy_prediction_days
        data[f'{short_col}_predicted_sell'] = data[short_col] + data[f'{short_col}_slope'] * sell_prediction_days
        data[f'{long_col}_predicted_sell'] = data[long_col] + data[f'{long_col}_slope'] * sell_prediction_days
    
    elif method == 'arima':
        # Use ARIMA for time series forecasting
        # Suppress convergence warnings
        warnings.filterwarnings('ignore')
        
        # Only start predictions once we have enough data
        valid_idx = ~data[short_col].isna() & ~data[long_col].isna()
        start_idx = data[valid_idx].index[0]
        valid_data = data.loc[start_idx:].copy()
        
        # Iterate through each data point
        window_size = max(30, slope_window * 3)  # Need sufficient history for modeling
        
        # Define multiple ARIMA models to try in order of complexity
        arima_models = [
            (1,1,0),  # Simple model, often stable
            (2,1,0),  # Slightly more complex
            (0,1,1),  # Moving average model
            (5,1,0),  # Original model (more complex)
        ]
        
        for i in range(window_size, len(valid_data)):
            # Initialize flags
            short_sma_success = False
            long_sma_success = False
            current_idx = valid_data.index[i]
            
            try:
                history_short_sma = valid_data[short_col].iloc[i-window_size:i]
                history_long_sma = valid_data[long_col].iloc[i-window_size:i]
                
                # Try different ARIMA models until one works
                for order in arima_models:
                    if not short_sma_success:
                        try:
                            # Fit ARIMA model for short SMA
                            model_short_sma = ARIMA(history_short_sma, order=order)
                            model_fit_short_sma = model_short_sma.fit(disp=0)  # Suppress output
                            forecast_short_sma_buy = model_fit_short_sma.forecast(steps=buy_prediction_days)
                            forecast_short_sma_sell = model_fit_short_sma.forecast(steps=sell_prediction_days)
                            
                            # Store predictions
                            data.loc[current_idx, f'{short_col}_predicted_buy'] = forecast_short_sma_buy[-1]
                            data.loc[current_idx, f'{short_col}_predicted_sell'] = forecast_short_sma_sell[-1]
                            short_sma_success = True
                            print(f"Successfully fit ARIMA{order} for {short_col} at {current_idx}")
                        except Exception:
                            # Continue to next model if this one fails
                            continue
                    
                    if not long_sma_success:
                        try:
                            # Fit ARIMA model for long SMA
                            model_long_sma = ARIMA(history_long_sma, order=order)
                            model_fit_long_sma = model_long_sma.fit(disp=0)  # Suppress output
                            forecast_long_sma_buy = model_fit_long_sma.forecast(steps=buy_prediction_days)
                            forecast_long_sma_sell = model_fit_long_sma.forecast(steps=sell_prediction_days)
                            
                            # Store predictions
                            data.loc[current_idx, f'{long_col}_predicted_buy'] = forecast_long_sma_buy[-1]
                            data.loc[current_idx, f'{long_col}_predicted_sell'] = forecast_long_sma_sell[-1]
                            long_sma_success = True
                            print(f"Successfully fit ARIMA{order} for {long_col} at {current_idx}")
                        except Exception:
                            # Continue to next model if this one fails
                            continue
                    
                    # If both models succeeded, break the loop
                    if short_sma_success and long_sma_success:
                        break
                
                # If any model failed, fall back to linear prediction
                if not (short_sma_success and long_sma_success):
                    print(f"Falling back to linear prediction at {current_idx}")
                    short_sma_slope = (valid_data[short_col].iloc[i] - valid_data[short_col].iloc[i-slope_window]) / slope_window
                    long_sma_slope = (valid_data[long_col].iloc[i] - valid_data[long_col].iloc[i-slope_window]) / slope_window
                    
                    if not short_sma_success:
                        data.loc[current_idx, f'{short_col}_predicted_buy'] = valid_data[short_col].iloc[i] + short_sma_slope * buy_prediction_days
                        data.loc[current_idx, f'{short_col}_predicted_sell'] = valid_data[short_col].iloc[i] + short_sma_slope * sell_prediction_days
                    
                    if not long_sma_success:
                        data.loc[current_idx, f'{long_col}_predicted_buy'] = valid_data[long_col].iloc[i] + long_sma_slope * buy_prediction_days
                        data.loc[current_idx, f'{long_col}_predicted_sell'] = valid_data[long_col].iloc[i] + long_sma_slope * sell_prediction_days
                
            except Exception as e:
                print(f"ARIMA prediction failed at {current_idx}: {str(e)}")
                # Fall back to linear prediction for this point
                if i > 0:
                    short_sma_slope = (valid_data[short_col].iloc[i] - valid_data[short_col].iloc[i-slope_window]) / slope_window
                    long_sma_slope = (valid_data[long_col].iloc[i] - valid_data[long_col].iloc[i-slope_window]) / slope_window
                    data.loc[current_idx, f'{short_col}_predicted_buy'] = valid_data[short_col].iloc[i] + short_sma_slope * buy_prediction_days
                    data.loc[current_idx, f'{long_col}_predicted_buy'] = valid_data[long_col].iloc[i] + long_sma_slope * buy_prediction_days
                    data.loc[current_idx, f'{short_col}_predicted_sell'] = valid_data[short_col].iloc[i] + short_sma_slope * sell_prediction_days
                    data.loc[current_idx, f'{long_col}_predicted_sell'] = valid_data[long_col].iloc[i] + long_sma_slope * sell_prediction_days
    
    elif method == 'exp_smoothing':
        # Use Exponential Smoothing (Holt-Winters) for forecasting
        warnings.filterwarnings('ignore')
        
        # Only start predictions once we have enough data
        valid_idx = ~data[short_col].isna() & ~data[long_col].isna()
        start_idx = data[valid_idx].index[0]
        valid_data = data.loc[start_idx:].copy()
        
        # Iterate through each data point
        window_size = max(20, slope_window * 2)  # Need sufficient history
        
        for i in range(window_size, len(valid_data)):
            try:
                current_idx = valid_data.index[i]
                history_short_sma = valid_data[short_col].iloc[i-window_size:i]
                history_long_sma = valid_data[long_col].iloc[i-window_size:i]
                
                # Fit Exponential Smoothing model for short SMA
                model_short_sma = ExponentialSmoothing(history_short_sma, trend='add', seasonal=None)
                model_fit_short_sma = model_short_sma.fit()
                forecast_short_sma_buy = model_fit_short_sma.forecast(buy_prediction_days)
                forecast_short_sma_sell = model_fit_short_sma.forecast(sell_prediction_days)
                
                # Fit Exponential Smoothing model for long SMA
                model_long_sma = ExponentialSmoothing(history_long_sma, trend='add', seasonal=None)
                model_fit_long_sma = model_long_sma.fit()
                forecast_long_sma_buy = model_fit_long_sma.forecast(buy_prediction_days)
                forecast_long_sma_sell = model_fit_long_sma.forecast(sell_prediction_days)
                
                # Store predictions
                data.loc[current_idx, f'{short_col}_predicted_buy'] = forecast_short_sma_buy[-1]
                data.loc[current_idx, f'{long_col}_predicted_buy'] = forecast_long_sma_buy[-1]
                data.loc[current_idx, f'{short_col}_predicted_sell'] = forecast_short_sma_sell[-1]
                data.loc[current_idx, f'{long_col}_predicted_sell'] = forecast_long_sma_sell[-1]
            except Exception as e:
                print(f"Exp Smoothing prediction failed at {valid_data.index[i]}: {str(e)}")
                # Fall back to linear prediction
                if i > 0:
                    short_sma_slope = (valid_data[short_col].iloc[i] - valid_data[short_col].iloc[i-slope_window]) / slope_window
                    long_sma_slope = (valid_data[long_col].iloc[i] - valid_data[long_col].iloc[i-slope_window]) / slope_window
                    data.loc[current_idx, f'{short_col}_predicted_buy'] = valid_data[short_col].iloc[i] + short_sma_slope * buy_prediction_days
                    data.loc[current_idx, f'{long_col}_predicted_buy'] = valid_data[long_col].iloc[i] + long_sma_slope * buy_prediction_days
                    data.loc[current_idx, f'{short_col}_predicted_sell'] = valid_data[short_col].iloc[i] + short_sma_slope * sell_prediction_days
                    data.loc[current_idx, f'{long_col}_predicted_sell'] = valid_data[long_col].iloc[i] + long_sma_slope * sell_prediction_days
    
    # Calculate predicted crossover differences
    data['SMA_Diff_Predicted_Buy'] = data[f'{short_col}_predicted_buy'] - data[f'{long_col}_predicted_buy']
    data['SMA_Diff_Predicted_Buy_Prev'] = data['SMA_Diff_Predicted_Buy'].shift(1)
    data['SMA_Diff_Predicted_Sell'] = data[f'{short_col}_predicted_sell'] - data[f'{long_col}_predicted_sell']
    data['SMA_Diff_Predicted_Sell_Prev'] = data['SMA_Diff_Predicted_Sell'].shift(1)
    
    # Initialize predicted crossover columns
    data['Predicted_Golden_Cross'] = 0  # For buy signals
    data['Predicted_Death_Cross'] = 0   # For sell signals
    
    # Identify predicted crossovers
    mask = ~data['SMA_Diff_Predicted_Buy'].isna() & ~data['SMA_Diff_Predicted_Buy_Prev'].isna()
    if mask.any():
        # Predicted Golden Cross: Predicted SMA50 crosses above predicted SMA200 (using buy window)
        data.loc[(data['SMA_Diff_Predicted_Buy'] > 0) & 
                 (data['SMA_Diff_Predicted_Buy_Prev'] <= 0) & 
                 mask, 'Predicted_Golden_Cross'] = 1
    
    mask = ~data['SMA_Diff_Predicted_Sell'].isna() & ~data['SMA_Diff_Predicted_Sell_Prev'].isna()
    if mask.any():
        # Predicted Death Cross: Predicted SMA50 crosses below predicted SMA200 (using sell window)
        data.loc[(data['SMA_Diff_Predicted_Sell'] < 0) & 
                 (data['SMA_Diff_Predicted_Sell_Prev'] >= 0) & 
                 mask, 'Predicted_Death_Cross'] = 1
    
    return data

def backtest_predicted_sma_strategy(ticker, start_date, end_date, short_sma=50, long_sma=200, 
                                   buy_prediction_days=7, sell_prediction_days=7, 
                                   slope_window=5, prediction_method='linear'):
    """
    Backtest predicted SMA crossover strategy with time series forecasting
    
    Parameters:
    - ticker: Stock ticker symbol
    - start_date, end_date: Date range for backtesting
    - short_sma: Period for short SMA (default: 50)
    - long_sma: Period for long SMA (default: 200)
    - buy_prediction_days: Days to look ahead for buy signals
    - sell_prediction_days: Days to look ahead for sell signals
    - slope_window: Window for calculating slopes
    - prediction_method: Method for forecasting ('linear', 'arima', 'exp_smoothing')
    """
    # Fetch data
    data = fetch_data(ticker, start_date, end_date)
    
    if len(data) < long_sma:
        print(f"Warning: Not enough data points ({len(data)}) for calculating SMA{long_sma}. Need at least {long_sma}.")
        print("Skipping backtest due to insufficient data.")
        return None, {}
    
    # Calculate SMA indicators with configurable periods
    data = calculate_sma_crossover(data, short_sma, long_sma)
    
    # Get column names
    short_col = data.attrs.get('short_col', f'SMA{short_sma}')
    long_col = data.attrs.get('long_col', f'SMA{long_sma}')
    
    # Verify SMA columns exist (debugging step)
    missing_columns = []
    if short_col not in data.columns:
        missing_columns.append(short_col)
    if long_col not in data.columns:
        missing_columns.append(long_col)
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print("Available columns:", data.columns.tolist())
        # Add missing columns with NaN values to avoid errors
        for col in missing_columns:
            data[col] = np.nan
            
    # Predict future SMA values using the specified forecasting method
    data = predict_sma_values_tsm(data, buy_prediction_days, sell_prediction_days, slope_window, prediction_method)
    
    # Generate signals based on predicted crossovers
    data = generate_predicted_sma_signals(data)
    
    # Calculate returns
    data = calculate_returns(data)
    data = calculate_realized_returns(data)
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    # Create interactive plotly figure with subplots
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=(f'{ticker} Price and SMAs ({short_sma}/{long_sma}, {prediction_method.capitalize()} Forecasting, Slope Window: {slope_window} days, Buy: {buy_prediction_days} days, Sell: {sell_prediction_days} days)', 
                                       'Realized Returns (After Selling)'),
                        vertical_spacing=0.15,
                        shared_xaxes=True)
    
    # Add price to first subplot
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'], 
        name='Price', 
        line=dict(color='black', width=1),
        connectgaps=True
    ), row=1, col=1)
    
    # Safely add SMA lines
    if not data[short_col].isna().all():
        sma_short_data = data.dropna(subset=[short_col])
        fig.add_trace(go.Scatter(
            x=sma_short_data.index, 
            y=sma_short_data[short_col], 
            name=f'SMA {short_sma}', 
            line=dict(color='blue', width=1.5),
        ), row=1, col=1)
        
        # Add predicted short SMA line for buying
        predicted_buy_col = f'{short_col}_predicted_buy'
        if predicted_buy_col in data.columns and not data[predicted_buy_col].isna().all():
            predicted_data = data.dropna(subset=[predicted_buy_col])
            fig.add_trace(go.Scatter(
                x=predicted_data.index, 
                y=predicted_data[predicted_buy_col], 
                name=f'Buy Predicted SMA {short_sma} (+{buy_prediction_days} days)', 
                line=dict(color='green', width=1.5, dash='dash'),
            ), row=1, col=1)
    
    if not data[long_col].isna().all():
        sma_long_data = data.dropna(subset=[long_col])
        fig.add_trace(go.Scatter(
            x=sma_long_data.index, 
            y=sma_long_data[long_col], 
            name=f'SMA {long_sma}', 
            line=dict(color='red', width=1.5),
        ), row=1, col=1)
        
        # Add predicted long SMA line for buying
        predicted_buy_col = f'{long_col}_predicted_buy'
        if predicted_buy_col in data.columns and not data[predicted_buy_col].isna().all():
            predicted_data = data.dropna(subset=[predicted_buy_col])
            fig.add_trace(go.Scatter(
                x=predicted_data.index, 
                y=predicted_data[predicted_buy_col], 
                name=f'Buy Predicted SMA {long_sma} (+{buy_prediction_days} days)', 
                line=dict(color='green', width=1.5, dash='dash'),
            ), row=1, col=1)
        
        # Add predicted long SMA line for selling
        predicted_sell_col = f'{long_col}_predicted_sell'
        if predicted_sell_col in data.columns and not data[predicted_sell_col].isna().all():
            predicted_data = data.dropna(subset=[predicted_sell_col])
            fig.add_trace(go.Scatter(
                x=predicted_data.index, 
                y=predicted_data[predicted_sell_col], 
                name=f'Sell Predicted SMA {long_sma} (+{sell_prediction_days} days)', 
                line=dict(color='red', width=1.5, dash='dash'),
            ), row=1, col=1)
    
    # Add buy/sell signals as scatter points
    golden_cross = data[data['Predicted_Golden_Cross'] == 1]
    death_cross = data[data['Predicted_Death_Cross'] == 1]
    
    if not golden_cross.empty:
        fig.add_trace(go.Scatter(
            x=golden_cross.index, 
            y=golden_cross[short_col],
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='green'),
            name='Predicted Golden Cross (Buy)'
        ), row=1, col=1)
    
    if not death_cross.empty:
        fig.add_trace(go.Scatter(
            x=death_cross.index, 
            y=death_cross[short_col],
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='red'),
            name='Predicted Death Cross (Sell)'
        ), row=1, col=1)
    
    # Add realized returns directly to second subplot (skipping the mark-to-market subplot)
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['cumulative_realized_return'],
        name='Realized Returns',
        line=dict(color='green', width=1.5)
    ), row=2, col=1)
    
    # Update layout with improved styling
    fig.update_layout(
        height=800,  # Slightly shorter since we removed one subplot
        width=1200,
        title_text=f"Predicted SMA Crossover Strategy Backtest for {ticker}",
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5
        ),
        margin=dict(l=50, r=50, t=80, b=50),  # Adjust margins
        template="plotly_white",  # Use a clean template with grid
    )
    
    # Add grid lines and improve axes
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        rangeslider_visible=False,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray'
    )
    
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray'
    )
    
    # Make y-axes independent for better visualization of each plot
    fig.update_yaxes(autorange=True, fixedrange=False)
    
    # Show the interactive plot
    fig.show()
    
    # Print metrics
    print(f"Backtest Results for {ticker} Predicted SMA Strategy (SMA{short_sma}/SMA{long_sma}, {prediction_method.capitalize()} Forecasting, Slope Window: {slope_window} days, Buy: {buy_prediction_days} days, Sell: {sell_prediction_days} days):")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Explain the metrics
    #explain_metrics()
    
    return data, metrics

def calculate_momentum(data, window):
    """
    Calculate momentum based on price changes over a specified window
    """
    data['momentum'] = data['Close'].pct_change(window)
    return data

def generate_signals(data, momentum_threshold=0):
    """
    Generate buy/sell signals based on momentum
    - Buy (1) when momentum > threshold
    - Sell (-1) when momentum < -threshold
    - Hold (0) otherwise
    """
    data['signal'] = 0
    data.loc[data['momentum'] > momentum_threshold, 'signal'] = 1
    data.loc[data['momentum'] < -momentum_threshold, 'signal'] = -1
    return data

def generate_predicted_sma_signals(data, print_signals=True):
    """
    Generate buy/sell signals based on predicted SMA crossovers with separate windows
    
    Parameters:
    - data: DataFrame with price and SMA data
    - print_signals: If True, print a table of signals with dates and prices
    """
    # Get SMA column names from data attributes
    short_sma = data.attrs.get('short_sma', 50)
    long_sma = data.attrs.get('long_sma', 200)
    short_col = data.attrs.get('short_col', f'SMA{short_sma}')
    long_col = data.attrs.get('long_col', f'SMA{long_sma}')

    # Create a new signal column initialized with zeros (no position)
    data['signal'] = 0
    
    # Find predicted crossover points
    predicted_golden_cross_mask = (data['Predicted_Golden_Cross'] == 1)
    predicted_death_cross_mask = (data['Predicted_Death_Cross'] == 1)
    
    if predicted_golden_cross_mask.any():
        # Find the index of the first buy signal (Predicted Golden Cross)
        first_buy_idx = data[predicted_golden_cross_mask].index[0]
        
        # Apply buy signals (all Predicted Golden Crosses)
        data.loc[predicted_golden_cross_mask, 'signal'] = 1
        
        # Apply sell signals (Predicted Death Crosses), but only after first buy
        valid_sell_mask = predicted_death_cross_mask & (data.index > first_buy_idx)
        data.loc[valid_sell_mask, 'signal'] = -1
        
        # Forward fill signals to maintain position between crossovers
        data.loc[data.index >= first_buy_idx, 'signal'] = data.loc[data.index >= first_buy_idx, 'signal'].ffill()
        
        # Print signal statistics for debugging
        buy_count = (data['signal'] == 1).sum()
        sell_count = (data['signal'] == -1).sum()
        print(f"Predicted Signal statistics: Buy/Hold: {buy_count}, Sell/Short: {sell_count}, Total: {len(data)}")
        
        # Print signal details if requested
        if print_signals:
            print("\n=== SIGNAL DETAILS ===")
            print(f"{'Date':<12} {'Signal Type':<12} {'Close Price':<12} {short_col:<12} {long_col:<12} {'Gain/Loss %':<12}")
            print("-" * 75)
            
            # Get all signal change points
            buy_signals = data[data['signal'] == 1]
            sell_signals = data[data['signal'] == -1]
            
            # Create a combined dataframe with all signals, sorted by date
            all_signals = pd.DataFrame()
            
            # Add buy signals with signal type
            if not buy_signals.empty:
                buy_df = buy_signals.copy()
                buy_df['signal_type'] = 'BUY'
                all_signals = pd.concat([all_signals, buy_df])
            
            # Add sell signals with signal type
            if not sell_signals.empty:
                sell_df = sell_signals.copy()
                sell_df['signal_type'] = 'SELL'
                all_signals = pd.concat([all_signals, sell_df])
            
            # Sort by date
            all_signals = all_signals.sort_index()
            
            # Track running position for calculating gains
            last_buy_price = None
            
            # Print signals in chronological order
            for idx, row in all_signals.iterrows():
                signal_date = idx.strftime('%Y-%m-%d')
                price = row['Close']
                signal_type = row['signal_type']
                
                if signal_type == 'BUY':
                    last_buy_price = price  # Remember buy price for gain calculation
                    print(f"{signal_date:<12} {signal_type:<12} ${price:<10.2f} ${row[short_col]:<10.2f} ${row[long_col]:<10.2f} {'-':<12}")
                else:  # SELL
                    gain_loss = '-'
                    # Calculate gain/loss if we know the last buy price
                    if last_buy_price is not None:
                        gain_loss_pct = ((price / last_buy_price) - 1) * 100
                        gain_loss = f"{gain_loss_pct:.2f}%"
                        last_buy_price = None  # Reset after selling
                    
                    print(f"{signal_date:<12} {signal_type:<12} ${price:<10.2f} ${row[short_col]:<10.2f} ${row[long_col]:<10.2f} {gain_loss:<12}")
            
            print("\n=== CURRENT POSITION ===")
            current_signal = data['signal'].iloc[-1]
            if current_signal > 0:
                # Find the most recent buy signal
                last_buy_idx = buy_signals.index[-1]
                entry_date = last_buy_idx.strftime('%Y-%m-%d')
                entry_price = data.loc[last_buy_idx, 'Close']
                current_price = data['Close'].iloc[-1]
                gain_loss_pct = ((current_price / entry_price) - 1) * 100
                print(f"Currently LONG since {entry_date} at ${entry_price:.2f}")
                print(f"Current price: ${current_price:.2f}, Unrealized gain/loss: {gain_loss_pct:.2f}%")
            elif current_signal < 0:
                print("Currently SHORT/CASH")
            else:
                print("No position")
    
    return data

def has_open_buy_signal(ticker, days=365, strategy='predicted_sma', **strategy_params):
    """
    Check if a ticker has an open buy signal (buy signal without a subsequent sell signal)
    
    Returns:
    - tuple: (has_open_signal, signal_date, signal_price, current_price, gain_loss_pct)
    """
    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()
    
    try:
        # Fetch and process data based on strategy
        data = fetch_data(ticker, start_date, end_date)
        
        if len(data) < 200:
            print(f"Warning: Not enough data for {ticker}. Skipping.")
            return False, None, None, None, None
        
        if strategy == 'sma':
            data = calculate_sma_crossover(data)
            data = generate_sma_signals(data)
        elif strategy == 'predicted_sma':
            data = calculate_sma_crossover(data)
            buy_prediction_days = strategy_params.get('buy_prediction_days', 7)
            sell_prediction_days = strategy_params.get('sell_prediction_days', 7)
            slope_window = strategy_params.get('slope_window', 5)
            prediction_method = strategy_params.get('prediction_method', 'linear')
            data = predict_sma_values_tsm(data, buy_prediction_days, sell_prediction_days, 
                                     slope_window, prediction_method)
            data = generate_predicted_sma_signals(data)
        else:  # momentum
            window = strategy_params.get('window', 20)
            threshold = strategy_params.get('threshold', 0.05)
            data = calculate_momentum(data, window)
            data = generate_signals(data, threshold)
        
        # Check if latest signal is a buy signal
        if len(data) == 0 or 'signal' not in data.columns:
            return False, None, None, None, None
        
        # Find the most recent non-zero signal (ignoring hold signals)
        non_zero_signals = data[data['signal'] != 0]
        
        if len(non_zero_signals) == 0:
            return False, None, None, None, None
            
        # Get the most recent non-zero signal
        latest_signal = non_zero_signals['signal'].iloc[-1]
        
        if latest_signal == 1:  # Open buy signal
            # Find when this buy signal started
            signal_changes = data['signal'].diff().fillna(0)
            buy_signal_idx = signal_changes[signal_changes > 0].index[-1]
            
            signal_date = buy_signal_idx.strftime('%Y-%m-%d')
            signal_price = data.loc[buy_signal_idx, 'Close']
            current_price = data['Close'].iloc[-1]
            gain_loss_pct = (current_price / signal_price - 1) * 100
            
            return True, signal_date, signal_price, current_price, gain_loss_pct
        
        return False, None, None, None, None
    
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return False, None, None, None, None

def scan_tickers(tickers, days=365, strategy='predicted_sma', **strategy_params):
    """
    Scan multiple tickers for open buy signals and return results
    """
    results = []
    
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if not ticker:
            continue
            
        print(f"Scanning {ticker}...")
        has_signal, signal_date, signal_price, current_price, gain_loss_pct = has_open_buy_signal(
            ticker, days, strategy, **strategy_params
        )
        
        if has_signal:
            results.append({
                'ticker': ticker,
                'signal_date': signal_date,
                'signal_price': signal_price,
                'current_price': current_price,
                'gain_loss_pct': gain_loss_pct
            })
            print(f"✅ {ticker}: Open buy signal from {signal_date}, current gain/loss: {gain_loss_pct:.2f}%")
        else:
            print(f"❌ {ticker}: No open buy signal")
    
    return results

def save_scan_results(results, output_file=None):
    """
    Save the scan results to a CSV file
    """
    if not results:
        print("No tickers with open buy signals found.")
        return
    
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"buy_signals_{timestamp}.csv"
    
    # Sort results by signal_date (most recent first)
    sorted_results = sorted(results, key=lambda x: datetime.strptime(x['signal_date'], '%Y-%m-%d'), reverse=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ticker', 'signal_date', 'signal_price', 
                                              'current_price', 'gain_loss_pct'])
        writer.writeheader()
        writer.writerows(sorted_results)
    
    print(f"\nResults saved to {output_file}")
    
    # Also print a summary table to console
    print("\nTickers with open buy signals (sorted by most recent):")
    print("-" * 70)
    print(f"{'Ticker':<8} {'Signal Date':<12} {'Signal Price':<14} {'Current Price':<14} {'Gain/Loss %':<10}")
    print("-" * 70)
    
    for result in sorted_results:
        print(f"{result['ticker']:<8} {result['signal_date']:<12} "
              f"${result['signal_price']:<12.2f} ${result['current_price']:<12.2f} "
              f"{result['gain_loss_pct']:<10.2f}%")

def parse_args():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='Backtest a trading strategy')
    parser.add_argument('-t', '--ticker', type=str, default='AAPL', 
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('-s', '--strategy', type=str, choices=['momentum', 'sma', 'predicted_sma'], default='predicted_sma',
                        help='Trading strategy to use: momentum, sma, or predicted_sma (default: predicted_sma)')
    parser.add_argument('-w', '--window', type=int, default=20, 
                        help='Momentum calculation window in days (default: 20, only for momentum strategy)')
    parser.add_argument('-th', '--threshold', type=float, default=0.05, 
                        help='Momentum threshold for signal generation (default: 0.05, only for momentum strategy)')
    parser.add_argument('-pb', '--buy_prediction_days', type=int, default=7, 
                        help='Number of days to predict ahead for buy signals (default: 7)')
    parser.add_argument('-ps', '--sell_prediction_days', type=int, default=7, 
                        help='Number of days to predict ahead for sell signals (default: 7)')
    parser.add_argument('-d', '--days', type=int, default=365*5, 
                        help='Number of days to backtest (default: 5 years)')
    parser.add_argument('-sw', '--slope_window', type=int, default=5, 
                        help='Window (in days) used to calculate the slope of SMAs (default: 5)')
    parser.add_argument('-pm', '--prediction_method', type=str, choices=['linear', 'arima', 'exp_smoothing'], 
                        default='linear', help='Method for predicting future SMA values (default: linear)')
    
    # Add arguments for SMA periods
    parser.add_argument('--short-sma', type=int, default=50,
                        help='Period for short SMA (default: 50)')
    parser.add_argument('--long-sma', type=int, default=200,
                        help='Period for long SMA (default: 200)')
    
    # Add new arguments for ticker scanning
    parser.add_argument('--scan', action='store_true',
                        help='Scan mode: check multiple tickers for open buy signals')
    parser.add_argument('--tickers', type=str, 
                        help='Comma-separated list of tickers to scan (e.g., "AAPL,MSFT,GOOG")')
    parser.add_argument('--tickers-file', type=str,
                        help='Path to a file containing ticker symbols (one per line)')
    parser.add_argument('--output', type=str,
                        help='Output file for scan results (default: buy_signals_YYYYMMDD_HHMMSS.csv)')
    
    return parser.parse_args()

def explain_metrics():
    """
    Provides detailed explanation of the backtest metrics
    """
    explanations = {
        'Overall Return': """
        The total percentage return for the entire backtest period.
        This shows the absolute performance without annualization.
        For example, 0.75 means a 75% total return over the entire period.
        """,
        
        'Annualized Return': """
        The average yearly return of the strategy, annualized from the total return.
        Higher is better. This tells you how much the strategy would return yearly on average.
        """,
        
        'Annualized Volatility': """
        The standard deviation of returns, annualized to represent yearly risk.
        Lower is generally better. This measures how much the returns fluctuate over time.
        """,
        
        'Sharpe Ratio': """
        The risk-adjusted return, calculated as Annualized Return / Annualized Volatility.
        Higher is better. Generally:
        - < 1: Poor risk-adjusted return
        - 1-2: Acceptable
        - 2-3: Very good
        - > 3: Excellent
        This metric tells you how much return you're getting for each unit of risk taken.
        """,
        
        'Maximum Drawdown': """
        The largest percentage drop from peak to trough in the strategy's equity curve.
        Smaller (closer to zero) is better. This shows the worst-case scenario loss
        that would have occurred historically and helps assess downside risk.
        """
    }
    
    print("\n===== METRICS EXPLANATION =====")
    for metric, explanation in explanations.items():
        print(f"\n{metric}:")
        print(explanation.strip())
    print("\n==============================")

# Example usage
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Check if we're in scan mode
    if args.scan:
        tickers = []
        
        # Get tickers from command line argument
        if args.tickers:
            tickers.extend([t.strip() for t in args.tickers.split(',') if t.strip()])
        
        # Get tickers from file
        if args.tickers_file:
            if os.path.exists(args.tickers_file):
                with open(args.tickers_file, 'r') as f:
                    tickers.extend([line.strip() for line in f if line.strip()])
            else:
                print(f"Error: Tickers file '{args.tickers_file}' not found.")
                sys.exit(1)
        
        if not tickers:
            print("Error: No tickers provided. Use --tickers or --tickers-file.")
            sys.exit(1)
        
        # Run the scan
        print(f"Scanning {len(tickers)} tickers using {args.strategy} strategy...")
        strategy_params = {
            'window': args.window,
            'threshold': args.threshold,
            'buy_prediction_days': args.buy_prediction_days,
            'sell_prediction_days': args.sell_prediction_days,
            'slope_window': args.slope_window,
            'prediction_method': args.prediction_method
        }
        
        results = scan_tickers(tickers, args.days, args.strategy, **strategy_params)
        save_scan_results(results, args.output)
        
    else:
        # Regular single-ticker backtest mode
        ticker = args.ticker
        start_date = datetime.now() - timedelta(days=args.days)
        end_date = datetime.now()
        
        if args.strategy == 'momentum':
            print(f"Running momentum backtest for {ticker} with {args.window}-day window and {args.threshold} threshold...")
            results, metrics = backtest_momentum(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                window=args.window,
                threshold=args.threshold
            )
        elif args.strategy == 'predicted_sma':
            print(f"Running predicted SMA backtest for {ticker} with {args.prediction_method} forecasting (SMAs: {args.short_sma}/{args.long_sma}, slope window: {args.slope_window}, buy: {args.buy_prediction_days}, sell: {args.sell_prediction_days} days)...")
            results, metrics = backtest_predicted_sma_strategy(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                short_sma=args.short_sma,
                long_sma=args.long_sma,
                buy_prediction_days=args.buy_prediction_days,
                sell_prediction_days=args.sell_prediction_days,
                slope_window=args.slope_window,
                prediction_method=args.prediction_method
            )
        else:  # args.strategy == 'sma'
            print(f"Running SMA crossover backtest for {ticker} (SMAs: {args.short_sma}/{args.long_sma})...")
            results, metrics = backtest_sma_strategy(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                short_sma=args.short_sma,
                long_sma=args.long_sma
            )