import os
import time
import argparse
import logging
import random
import shutil
import traceback
import sqlite3  # Add import for SQLite
from datetime import datetime
from ticker_rl import test_model
from list_djia import get_djia_companies

def log_ticker_transactions(db_path, tickers, logger):
    """
    Log the transactions for specific tickers from the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    logger.info("\n--- Ticker Transactions in Database ---")
    for ticker in tickers:
        cursor.execute('''
            SELECT transaction_time, action, price, quantity 
            FROM ticker_transactions 
            WHERE ticker = ?
            ORDER BY transaction_time DESC
        ''', (ticker,))
        
        transactions = cursor.fetchall()
        
        if transactions:
            logger.info(f"\nTransactions for {ticker}:")
            logger.info(f"{'Date':<12} {'Action':<6} {'Price':<10} {'Quantity':<8}")
            logger.info("-" * 40)
            
            for transaction in transactions:
                date, action, price, quantity = transaction    
                logger.info(f"{date:<12} {action:<6} ${price:<9.2f} {quantity:<8}")
        else:
            logger.info(f"\nNo transactions found for {ticker}")
    
    # Add logging for ticker gains
    logger.info("\n--- Ticker Gains in Database ---")
    for ticker in tickers:
        cursor.execute('''
            SELECT execution_date, rl_gain, buy_hold_gain 
            FROM ticker_gains 
            WHERE ticker = ?
            ORDER BY execution_date DESC
        ''', (ticker,))
        
        gains = cursor.fetchall()
        
        if gains:
            logger.info(f"\nGains for {ticker}:")
            logger.info(f"{'Date':<12} {'RL Gain':<12} {'Buy & Hold':<12} {'Diff':<12}")
            logger.info("-" * 50)
            
            for gain in gains:
                exec_date, rl_gain, bh_gain = gain
                diff = rl_gain - bh_gain if bh_gain is not None else "N/A"
                diff_str = f"{diff:.2f}%" if isinstance(diff, float) else diff
                logger.info(f"{exec_date:<12} {rl_gain:.2f}% {bh_gain if bh_gain is not None else 'N/A':<12} {diff_str}")
        else:
            logger.info(f"\nNo gain data found for {ticker}")
    
    conn.close()


def initialize_db(db_path):
    """
    Initialize the SQLite database to store ticker transactions and gains.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ticker_transactions (
            ticker TEXT,
            transaction_time TEXT,
            action TEXT,
            price REAL,
            quantity INTEGER,
            PRIMARY KEY (ticker, transaction_time)
        )
    ''')
    
    # Create gains table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ticker_gains (
            ticker TEXT,
            execution_date TEXT,
            rl_gain REAL,
            buy_hold_gain REAL,
            PRIMARY KEY (ticker, execution_date)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_transactions_to_db(db_path, ticker, flat_transactions, overwrite_all=False):
    """
    Save transactions for a ticker to the database with special handling:
    1. If overwrite_all is True, replace all existing transactions for this ticker
    2. Otherwise:
       - If a transaction for the same date already exists, don't save
       - If transaction types differ on the same date (BUY vs SELL), issue an ERROR
       - Maintain only the 5 most recent transactions per ticker
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # If overwrite_all flag is set, delete all existing transactions for this ticker
    if overwrite_all and flat_transactions:    
        cursor.execute('DELETE FROM ticker_transactions WHERE ticker = ?', (ticker,))
        
        # Take the most recent 5 transactions (or fewer if there aren't 5)
        recent_transactions = flat_transactions[-5:] if len(flat_transactions) > 5 else flat_transactions
        
        # Insert all these transactions
        for transaction in recent_transactions:
            transaction_date = transaction['date'].strftime('%Y-%m-%d')  # Format as YYYY-MM-DD
            
            cursor.execute('''
                INSERT INTO ticker_transactions (ticker, transaction_time, action, price, quantity)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                ticker, 
                transaction_date,
                transaction['type'], 
                transaction['price'], 
                transaction['shares']
            ))
    
    # Normal operation (no overwrite)
    elif flat_transactions:
        last_transaction = flat_transactions[-1]  # Get only the last transaction
        transaction_date = last_transaction['date'].strftime('%Y-%m-%d')  # Format as YYYY-MM-DD
        
        # Check if a transaction for this ticker on this date already exists
        cursor.execute('''
            SELECT action FROM ticker_transactions 
            WHERE ticker = ? AND transaction_time = ?
        ''', (ticker, transaction_date))
        
        existing_record = cursor.fetchone()
        
        if existing_record:
            existing_action = existing_record[0]
            current_action = last_transaction['type']
            
            # If action types differ on the same day (BUY vs SELL)
            if existing_action != current_action:
                print(f"ERROR: CONFLICTING TRANSACTION TYPES FOR {ticker} ON {transaction_date}.")
                print(f"ERROR: DATABASE HAS '{existing_action}' BUT NEW TRANSACTION IS '{current_action}'.")
            # If it's the same date, don't save (silently ignore)
        else:
            # No existing transaction for this date, so save it
            cursor.execute('''
                INSERT OR REPLACE INTO ticker_transactions (ticker, transaction_time, action, price, quantity)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                ticker, 
                transaction_date,
                last_transaction['type'], 
                last_transaction['price'], 
                last_transaction['shares']
            ))
            
            # Check how many transactions exist for this ticker
            cursor.execute('''
                SELECT COUNT(*) FROM ticker_transactions 
                WHERE ticker = ?
            ''', (ticker,))
            
            count = cursor.fetchone()[0]
            
            # If we have more than 5 transactions for this ticker, delete the oldest ones
            if count > 5:
                cursor.execute('''
                    DELETE FROM ticker_transactions 
                    WHERE ticker = ? AND transaction_time IN (
                        SELECT transaction_time FROM ticker_transactions
                        WHERE ticker = ?
                        ORDER BY transaction_time ASC
                        LIMIT ?
                    )
                ''', (ticker, ticker, count - 5))
    
    conn.commit()
    conn.close()

def save_gain_to_db(db_path, ticker, rl_gain, buy_hold_gain=None):
    """
    Save the overall gain of the RL model for a ticker in the database.
    If there's already a gain for the current day, it will be overwritten.
    
    Args:
        db_path: Path to the SQLite database
        ticker: Stock ticker symbol
        rl_gain: Overall gain percentage of the RL model
        buy_hold_gain: Overall gain percentage of the buy & hold strategy (if available)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get current date in YYYY-MM-DD format
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Insert or replace (if exists) the gain for the current day
    cursor.execute('''
        INSERT OR REPLACE INTO ticker_gains (ticker, execution_date, rl_gain, buy_hold_gain)
        VALUES (?, ?, ?, ?)
    ''', (ticker, current_date, rl_gain, buy_hold_gain))
    
    conn.commit()
    conn.close()


def move_png_file(log_dir, ticker):
    """
    Move the PNG file with the log file for reference
    """
    # Look for PNG files matching the pattern in the output directory
    png_files = [f for f in os.listdir(".") if f.lower() == f"{ticker.lower()}_trading_results.png"]
    
    if png_files:
        png_file = png_files[0]
        png_source_path = png_file
        
        # Create a new filename with timestamp
        new_png_filename = f"{ticker.lower()}_trading_results.png"
        png_dest_path = os.path.join(log_dir, new_png_filename)
        
        try:
            # Move the PNG file
            shutil.move(png_source_path, png_dest_path)
            return new_png_filename
        except Exception as e:
            print(f"Error moving PNG file for {ticker}: {e}")
            return None
    return None

def check_ticker_performance(test_results, min_gain_pct=0):
    """
    Check if the ticker passes the performance criteria:
    1. At least 5 transactions
    2. RL performance at least 80% of buy-and-hold strategy (not below 20% less)
    3. RL gain meets or exceeds the minimum gain percentage threshold
    
    Returns: (is_pass, failure_reason)
    """
    # Check for minimum transactions    
    if 'transactions' not in test_results or len(test_results['transactions']) < 5:
        return False, "Insufficient transactions (minimum 5 required)"
    
    # Check for performance against buy-and-hold
    if 'rl_agent_return' in test_results['results'] and 'buy_and_hold_return' in test_results['results']:
        rl_performance = test_results['results']['rl_agent_return']
        buy_hold_performance = test_results['results']['buy_and_hold_return']
                
        performance_diff = rl_performance - buy_hold_performance
        
        # Performance should be at least 80% of buy-and-hold
        if performance_diff < 20:            
            return False, f"RL performance is {performance_diff:.2f}% worse than buy-and-hold strategy (minimum required: 20%)"
        
                  
        if rl_performance < min_gain_pct:
            return False, f"RL gain of {rl_performance:.2f}% is below minimum required gain of {min_gain_pct:.2f}%"
    
    return True, ""

def calculate_gain_percentage(initial_value, final_value):
    """
    Calculate gain percentage from initial to final value
    """
    if initial_value <= 0:
        return 0.0
    
    return ((final_value / initial_value) - 1) * 100

def log_ticker_test_results(log_dir, ticker, test_date, transactions):
    """
    Create a log file for a ticker's test results with last 5 transactions.
    """
    # Create ticker-specific log file
    log_file = os.path.join(log_dir, f"{ticker.lower()}_test_history.log")
    
    # Get last 5 transactions
    last_transactions = transactions[-5:] if len(transactions) > 5 else transactions
    
    with open(log_file, 'a') as f:
        f.write(f"\n=== Test Run: {test_date} ===\n")
        f.write(f"Last {len(last_transactions)} transactions:\n")
        f.write("Date         Action  Price    Shares\n")
        f.write("-" * 40 + "\n")
        
        for trans in last_transactions:
            date = trans['date'].strftime('%Y-%m-%d')
            f.write(f"{date}  {trans['type']:<6}  ${trans['price']:<7.2f}  {trans['shares']}\n")
        f.write("\n")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Test DQN trading models for all Dow Jones Industrial Average companies.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments (matching default training parameters)
    parser.add_argument('--models-dir', type=str, default="djia_models",
                        help='Directory containing trained model weights')
    parser.add_argument('--output-dir', type=str, default="djia_test_results",
                        help='Directory to store testing results')
    parser.add_argument('--log-dir', type=str, default="tester_logs",
                        help='Directory to store testing logs')
    parser.add_argument('--lookback', type=int, default=40,
                        help='Lookback window size')
    parser.add_argument('--initial-capital', type=float, default=10000,
                        help='Initial capital for testing')
    parser.add_argument('--tickers', type=str,
                        help='Comma-separated list of specific tickers to test (if not provided, all DJIA companies will be used)')
    parser.add_argument('--num-tickers', type=int,
                        help='Number of arbitrary tickers to be chosen for testing')
    parser.add_argument('--period', type=int, default=1,
                        help='Data period in years')
    parser.add_argument('--start-date', type=str,
                        help='Start date for test data (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str,
                        help='End date for test data (YYYY-MM-DD format)')
    parser.add_argument('--overwrite-db', action='store_true',
                        help='Overwrite all existing transactions in the database with current calculated transactions')
    parser.add_argument('--min-gain-pct', type=float, default=50.0,
                        help='Minimum required gain percentage for a ticker to pass (default: 50.0)')
    parser.add_argument('--saved-models-dir', type=str, default="djia_models_saved",
                        help='Directory to look for existing saved models')
    parser.add_argument('--prefer-saved', action='store_true',
                        help='Prioritize using models from saved-models-dir if they exist')
    parser.add_argument('--remove-ohlcv', action='store_true',
                        help='Remove OHLCV data from the training dataset to save space')
    parser.add_argument('--min-holding-days', type=int, default=0,
                        help='Minimum holding days for each trade')
    parser.add_argument('--min_days_between_trades', type=int, default=0, 
                        help='Minimum days that must pass between a sell and the next buy (default: 0)')
    parser.add_argument('--buying-fee-pct', type=float, default=0.005,
                        help='Percentage fee for each buy transaction (e.g., 0.005)')
    parser.add_argument('--selling-fee-pct', type=float, default=0.005,
                        help='Percentage fee for each sell transaction (e.g., 0.005)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    specific_tickers_provided = False
    # Get the list of DJIA companies or use specific tickers if provided
    if args.tickers:
        djia_tickers = [ticker.strip() for ticker in args.tickers.split(',')]
        print(f"Using provided list of {len(djia_tickers)} tickers.")
        specific_tickers_provided = True
    else:
        print("Fetching list of Dow Jones Industrial Average companies...")
        # Get the DataFrame of DJIA companies
        #djia_tickers_df = get_djia_companies()
        # Extract the ticker symbols from the 'Ticker' column
        #djia_tickers = djia_tickers_df['Ticker'].tolist()
        #djia_tickers = [ "MMM", "AXP", "AMGN", "AMZN", "AAPL", "CSCO", "KO", "DIS", "GS", "HD", "HON", "IBM", "MCD", "MRK", "MSFT", "NVDA", "PG", "CRM", "TRV", "VZ", "V", "WMT", "AMD", "DKNG", "GOOG", "ORCL", "PINS", "UBER" 
        djia_tickers = [ "AMZN", "AAPL", "MSFT", "NVDA", "AMD", "GOOG", "UBER", "META", "NFLX", "TSLA" ]
        
    # If num-tickers is provided, randomly select that number of tickers
    if args.num_tickers:
        if args.num_tickers <= 0:
            print(f"Error: num-tickers must be positive. Using all {len(djia_tickers)} tickers.")
        elif args.num_tickers > len(djia_tickers):
            print(f"Warning: num-tickers ({args.num_tickers}) exceeds available tickers ({len(djia_tickers)}). Using all available tickers.")
        else:
            djia_tickers = random.sample(djia_tickers, args.num_tickers)
            print(f"Randomly selected {len(djia_tickers)} tickers for testing.")
    
    # Display the tickers we'll be testing
    print(f"Found {len(djia_tickers)} companies to process:")
    print(", ".join(djia_tickers))

    # Log file setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"testing_log_{timestamp}.txt")
    
    # Set up logging to file and stdout
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Test tracking variables
    successful_tickers = []
    failed_tickers = []
    skipped_tickers = []
    performance_failed_tickers = []  # New category for tickers that failed performance criteria
    
    logger.info(f"DJIA Model Testing Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total companies to process: {len(djia_tickers)}")
    logger.info(f"Training parameters: {args}\n")
    logger.info(f"Minimum required gain: {args.min_gain_pct}%\n")
    
    # Initialize the database
    db_path = os.path.join("ticker_transactions.db")  # Move DB to the main root directory
    initialize_db(db_path)

    for i, ticker in enumerate(djia_tickers):

        target_file = f"{ticker.lower()}_trading_model.keras"    
        if args.remove_ohlcv:
            target_file = target_file.replace(".keras", "_no_ohlcv.keras")
    
        if args.prefer_saved:
            model_file = os.path.join(args.saved_models_dir, target_file)
        else:
            model_file = os.path.join(args.models_dir, target_file)
    
        # Check if model exists
        if not model_file:
            logger.info(f"\n[{i+1}/{len(djia_tickers)}] Skipping {ticker} (no trained model found)")
            skipped_tickers.append(ticker)
            continue
        
        start_time = time.time()
        logger.info(f"\n[{i+1}/{len(djia_tickers)}] Testing model for {ticker}...")
        logger.info(f"Model file found: {model_file}")
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Define result filename  
            result_file = os.path.join(args.output_dir, f"{ticker.lower()}_test_results.txt")
            
            # Get current timestamp for the test
            test_date = datetime.now()
            
            # Redirect logging to result file during this test
            result_logger = logging.getLogger(f'test_logger_{ticker}')
            result_logger.setLevel(logging.INFO)
            result_file_handler = logging.FileHandler(result_file)
            result_file_handler.setLevel(logging.INFO)
            result_logger.addHandler(result_file_handler)
            
            # Test the model for this ticker
            # Pass all agent-related arguments to ensure full compatibility
            
            # Calculate start and end dates from period if not provided
            start_date = args.start_date
            end_date = args.end_date
            
            if args.period and not start_date and not end_date:
                from datetime import timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=args.period * 365)).strftime('%Y-%m-%d')
            
            test_results = test_model(
                result_logger, 
                ticker=ticker, 
                lookback=args.lookback,
                initial_capital=args.initial_capital,
                start_date=start_date,
                end_date=end_date,
                model_weights_path=model_file,  # Pass the model weights path
                remove_ohlcv=args.remove_ohlcv,
                min_holding_days= args.min_holding_days,
                min_days_between_trades=args.min_days_between_trades,
                buying_fee_pct=args.buying_fee_pct,
                selling_fee_pct=args.selling_fee_pct
            )
            
            # Add initial_capital to test_results for gain percentage calculation
            test_results['initial_capital'] = args.initial_capital
            
            # Log the ticker's test results with last 5 transactions
            if 'transactions' in test_results:
                log_ticker_test_results(
                    args.log_dir,
                    ticker, 
                    test_date.strftime("%Y-%m-%d %H:%M:%S"),
                    test_results['transactions']
                )
            
            # Check if the ticker passes performance criteria
            is_pass, failure_reason = check_ticker_performance(test_results, args.min_gain_pct)
            
            # Save the last 5 transactions to the database
            if 'transactions' in test_results:
                save_transactions_to_db(db_path, ticker, test_results['transactions'], args.overwrite_db)
            
            # Calculate and save gain percentages
            if 'rl_final_value' in test_results:
                rl_gain = calculate_gain_percentage(args.initial_capital, test_results['rl_final_value'])
                buy_hold_gain = None
                
                if 'buy_hold_final_value' in test_results:
                    buy_hold_gain = calculate_gain_percentage(args.initial_capital, test_results['buy_hold_final_value'])
                
                # Save the gains to the database (will overwrite if already exists for today)
                save_gain_to_db(db_path, ticker, rl_gain, buy_hold_gain)
                
                # Log the saved gain
                logger.info(f"Saved RL gain of {rl_gain:.2f}% to database for {ticker}")
                if buy_hold_gain is not None:
                    logger.info(f"Saved Buy & Hold gain of {buy_hold_gain:.2f}% to database for {ticker}")
            
            # Move the PNG file to logs directory
            moved_png_filename = move_png_file(args.log_dir, ticker)
            
            # Remove the temporary file handler
            result_logger.removeHandler(result_file_handler)
            result_file_handler.close()
            
            if is_pass:
                # Record success
                elapsed_time = time.time() - start_time
                successful_tickers.append(ticker)
                logger.info(f"Status: Success")
                logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
                logger.info(f"[SUCCESS] Successfully tested model for {ticker} in {elapsed_time:.2f} seconds")
                
                # Log performance metrics
                if 'rl_final_value' in test_results and 'buy_hold_final_value' in test_results:
                    rl_perf = test_results['rl_final_value']
                    bh_perf = test_results['buy_hold_final_value']
                    rl_gain_pct = calculate_gain_percentage(args.initial_capital, rl_perf)
                    relative_perf = (rl_perf / bh_perf - 1) * 100 if bh_perf > 0 else 0
                    logger.info(f"RL Final Value: ${rl_perf:.2f} (Gain: {rl_gain_pct:.2f}%)")
                    logger.info(f"Buy & Hold Final Value: ${bh_perf:.2f}")
                    logger.info(f"Relative Performance: {relative_perf:.2f}%")
            else:
                # Record performance failure
                elapsed_time = time.time() - start_time
                performance_failed_tickers.append((ticker, failure_reason))
                logger.info(f"Status: Failed (Performance)")
                logger.info(f"Failure Reason: {failure_reason}")
                logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
                logger.info(f"[FAILED] Model for {ticker} failed performance criteria: {failure_reason}")
                
                # Log transaction count if that's the failure reason
                if 'transactions' in test_results:
                    logger.info(f"Transaction count: {len(test_results['transactions'])}")
                
                # Log performance metrics if that's the failure reason
                if 'rl_final_value' in test_results and 'buy_hold_final_value' in test_results:
                    rl_perf = test_results['rl_final_value']
                    bh_perf = test_results['buy_hold_final_value']
                    rl_gain_pct = calculate_gain_percentage(args.initial_capital, rl_perf)
                    relative_perf = (rl_perf / bh_perf - 1) * 100 if bh_perf > 0 else 0
                    logger.info(f"RL Final Value: ${rl_perf:.2f} (Gain: {rl_gain_pct:.2f}%)")
                    logger.info(f"Buy & Hold Final Value: ${bh_perf:.2f}")
                    logger.info(f"Relative Performance: {relative_perf:.2f}%")
            
            # Log the name of the relevant log file
            logger.info(f"Log file for {ticker}: {result_file}")
            
            # Log the moved PNG file name if it exists
            if moved_png_filename:
                logger.info(f"PNG file for {ticker}: {moved_png_filename}")
            
        except Exception as e:
            # Record failure due to exception
            elapsed_time = time.time() - start_time
            failed_tickers.append((ticker, str(e)))
            logger.info(f"Status: Failed (Error)")
            logger.info(f"Error: {str(e)}")
            
            # Print full traceback
            error_traceback = traceback.format_exc()
            logger.info(f"Full Traceback:\n{error_traceback}")
            
            logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
            logger.info(f"[ERROR] Failed to test model for {ticker}: {str(e)}")            
    
    # Write summary
    logger.info(f"\n\n--- Testing Summary ---")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Successful: {len(successful_tickers)} companies")
    logger.info(f"Failed (Performance): {len(performance_failed_tickers)} companies")
    logger.info(f"Failed (Error): {len(failed_tickers)} companies")
    logger.info(f"Skipped: {len(skipped_tickers)} companies")
    
    if performance_failed_tickers:
        logger.info("\nCompanies that failed performance criteria:")
        for ticker, reason in performance_failed_tickers:
            logger.info(f"- {ticker}: {reason}")
    
    if failed_tickers:
        logger.info("\nCompanies that failed due to errors:")
        for ticker, error in failed_tickers:
            logger.info(f"- {ticker}: {error}")

    # Log the database transactions if specific tickers were provided
    if specific_tickers_provided:
        log_ticker_transactions(db_path, djia_tickers, logger)
    
    # Print summary
    print("\n--- Testing Summary ---")
    print(f"Total companies processed: {len(djia_tickers)}")
    print(f"Successfully tested: {len(successful_tickers)}")
    print(f"Failed (Performance): {len(performance_failed_tickers)}")
    print(f"Failed (Error): {len(failed_tickers)}")
    print(f"Skipped: {len(skipped_tickers)}")
    
    if successful_tickers:
        print(f"\nSuccessful tickers: {', '.join(successful_tickers)}")
    
    if performance_failed_tickers:
        print("\nTickers that failed performance criteria:")
        for ticker, reason in performance_failed_tickers:
            print(f"- {ticker}: {reason}")
    
    if skipped_tickers:
        print(f"\nSkipped tickers: {', '.join(skipped_tickers)}")
    
    if failed_tickers:
        print("\nTickers that failed due to errors:")
        for ticker, error in failed_tickers:
            print(f"- {ticker}: {error}")
    
    print(f"\nAll test results saved to '{args.output_dir}/' directory")
    print(f"All logs and PNG files saved to '{args.log_dir}/' directory")
    print(f"Testing log saved to '{log_file}'")

if __name__ == "__main__":
    main()