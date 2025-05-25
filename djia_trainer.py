import os
import time
import argparse
import shutil
import logging
import random
import traceback
import concurrent.futures
from datetime import datetime
from ticker_rl import train_model
from list_djia import get_djia_companies

def get_parser():
    """Create and return the argument parser for the DJIA model training."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Train DQN trading models for all Dow Jones Industrial Average companies.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('--output-dir', type=str, default="djia_models",
                        help='Directory to store model weights and logs')
    parser.add_argument('--log-dir', type=str, default="trainer_logs",
                        help='Directory to store training logs')
    parser.add_argument('--lookback', type=int, default=40,
                        help='Lookback window size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epsilon-initial', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon-final', type=float, default=0.01,
                        help='Final exploration rate')
    parser.add_argument('--epsilon-decay', type=int, default=10000,
                        help='Epsilon decay steps')
    parser.add_argument('--memory-size', type=int, default=10000,
                        help='Replay memory size')
    parser.add_argument('--episodes', type=int, default=40,
                        help='Training episodes')
    parser.add_argument('--initial-capital', type=float, default=10000,
                        help='Initial capital')
    parser.add_argument('--period', type=int, default=1,
                        help='Data period in years')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip tickers that already have a trained model')
    parser.add_argument('--tickers', type=str,
                        help='Comma-separated list of specific tickers to train (if not provided, all DJIA companies will be used)')
    parser.add_argument('--force-new', action='store_true',
                        help='Force training from scratch, ignoring existing weights')
    parser.add_argument('--num-tickers', type=int,
                        help='Number of arbitrary tickers to be chosen for training')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel training using multiple threads')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads for parallel training')    
    parser.add_argument('--buying-fee-pct', type=float, default=0.005,
                        help='Percentage fee for each buy transaction (e.g., 0.005)')
    parser.add_argument('--selling-fee-pct', type=float, default=0.005,
                        help='Percentage fee for each sell transaction (e.g., 0.005)')
    
    return parser

def setup_logger(logs_dir, timestamp=None):
    """Set up and return the logger for training."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"training_log_{timestamp}.txt")
    
    # Set up logging to file and stdout
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    return logger, log_file

def get_tickers(args):
    """Get the list of tickers to train models for."""
    if args.tickers:
        djia_tickers = [ticker.strip() for ticker in args.tickers.split(',')]
        print(f"Using provided list of {len(djia_tickers)} tickers.")
    else:
        print("Fetching list of Dow Jones Industrial Average companies...")
        # Get the DataFrame of DJIA companies
        #djia_tickers_df = get_djia_companies()
        # Extract the ticker symbols from the 'Ticker' column
        #djia_tickers = djia_tickers_df['Ticker'].tolist()
        #djia_tickers = [ "MMM", "AXP", "AMGN", "AMZN", "AAPL", "CSCO", "KO", "DIS", "GS", "HD", "HON", "IBM", "MCD", "MRK", "MSFT", "NVDA", "PG", "CRM", "TRV", "VZ", "V", "WMT", "AMD", "DKNG", "GOOG", "ORCL", "PINS", "UBER" 
        #djia_tickers = [ "AMZN", "AAPL", "MSFT", "NVDA", "AMD", "GOOG", "UBER", "META", "NFLX", "TSLA" ]
        djia_tickers = [ "AAPL", "NVDA", "GOOG", "META", "NFLX"]
    
    # If num-tickers is provided, randomly select that number of tickers
    if args.num_tickers:
        if args.num_tickers <= 0:
            print(f"Error: num-tickers must be positive. Using all {len(djia_tickers)} tickers.")
        elif args.num_tickers > len(djia_tickers):
            print(f"Warning: num-tickers ({args.num_tickers}) exceeds available tickers ({len(djia_tickers)}). Using all available tickers.")
        else:
            djia_tickers = random.sample(djia_tickers, args.num_tickers)
            print(f"Randomly selected {len(djia_tickers)} tickers for training.")
    
    return djia_tickers

def get_training_params(args):
    """Extract training parameters from command line arguments."""
    return {
        "lookback": args.lookback,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epsilon_initial": args.epsilon_initial,
        "epsilon_final": args.epsilon_final,
        "epsilon_decay": args.epsilon_decay,
        "memory_size": args.memory_size,
        "episodes": args.episodes,
        "initial_capital": args.initial_capital,
        "period": args.period,
        "buying_fee_pct": args.buying_fee_pct,
        "selling_fee_pct": args.selling_fee_pct
    }

def train_ticker(ticker, training_params, models_dir, args, timestamp):
    """Train a model for a specific ticker."""
    # Set up local logger for this thread
    thread_log_dir = os.path.join(args.log_dir, "thread_logs")
    os.makedirs(thread_log_dir, exist_ok=True)
    thread_log_file = os.path.join(thread_log_dir, f"{ticker}_log_{timestamp}.txt")
    
    # Set up thread-specific logger
    thread_logger = logging.getLogger(f"{ticker}_logger")
    thread_logger.setLevel(logging.INFO)
    # Remove any existing handlers to avoid duplicate logging
    for handler in thread_logger.handlers[:]:
        thread_logger.removeHandler(handler)
    thread_logger.propagate = False  # Stop log propagation to parent logger
    
    # Add file handler for this thread with timestamp format
    file_handler = logging.FileHandler(thread_log_file)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    thread_logger.addHandler(file_handler)
    
    # Only add console handler if not in parallel mode
    if not args.parallel:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)  # Use same formatter for console
        console_handler.setLevel(logging.INFO)
        thread_logger.addHandler(console_handler)
    
    # Define model filenames and paths
    model_filename = f"{ticker.lower()}_trading_model.keras"
    target_file = os.path.join(models_dir, model_filename)    

    # Check if model exists in either directory and skip if requested
    if args.skip_existing and os.path.exists(target_file):
        thread_logger.info(f"[{ticker}] Status: Skipped (model already exists)")
        return "skipped", 0, None, thread_log_file
    
    if not args.force_new and not os.path.exists(target_file):
        thread_logger.info(f"[{ticker}] Status: Skipped (model NOT exists)")
        return "skipped", 0, None, thread_log_file

    existing_weights_file = target_file
    resumed = False
    if existing_weights_file and not args.force_new:
        thread_logger.info(f"[{ticker}] Resuming training from existing weights")
        resumed = True
    else:
        # If force_new is True and there's an existing file, remove it
        if args.force_new and os.path.exists(existing_weights_file):
            os.remove(existing_weights_file)
            thread_logger.info(f"[{ticker}] Removed existing weights file, starting fresh")
    
    start_time = time.time()
    thread_logger.info(f"[{ticker}] Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    

    try:
        # Train the model for this ticker
        agent, data = train_model(
            thread_logger,
            ticker=ticker,
            model_filename=existing_weights_file,            
            **training_params
        )        
                
        # Record success
        elapsed_time = time.time() - start_time
        status = "resumed" if resumed else "success"
        thread_logger.info(f"[{ticker}] Status: Success")
        thread_logger.info(f"[{ticker}] Elapsed time: {elapsed_time:.2f} seconds")
        
        if resumed:
            thread_logger.info(f"[{ticker}] ✓ Successfully continued training model in {elapsed_time:.2f} seconds")
        else:
            thread_logger.info(f"[{ticker}] ✓ Successfully trained new model in {elapsed_time:.2f} seconds")
            
        return status, elapsed_time, None, thread_log_file
        
    except Exception as e:
        # Record failure
        elapsed_time = time.time() - start_time
        thread_logger.info(f"[{ticker}] Status: Failed")
        thread_logger.info(f"[{ticker}] Error: {str(e)}")
        thread_logger.info(f"[{ticker}] Elapsed time: {elapsed_time:.2f} seconds")
        thread_logger.info(f"[{ticker}] ✗ Failed to train model: {str(e)}")
        # Print full traceback
        error_traceback = traceback.format_exc()
        thread_logger.info(f"[{ticker}] Full Traceback:\n{error_traceback}")
            
        return "failed", elapsed_time, str(e)

def train_djia_models(args=None):
    """
    Main function to train DQN trading models for DJIA companies.
    
    Args:
        args: Optional command line arguments. If None, they will be parsed from sys.argv.
        
    Returns:
        dict: A dictionary containing the training results.
    """
    # Parse arguments if not provided
    if args is None:
        parser = get_parser()
        args = parser.parse_args()
    
    # Create directories to store model weights and logs
    models_dir = args.output_dir
    logs_dir = args.log_dir
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Get the list of tickers to train
    djia_tickers = get_tickers(args)
    
    # Display the tickers we'll be training on
    print(f"Found {len(djia_tickers)} companies to process:")
    print(", ".join(djia_tickers))
    
    # Get training parameters
    training_params = get_training_params(args)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_file = setup_logger(logs_dir, timestamp)
    
    # Track results
    successful_tickers = []
    failed_tickers = []
    skipped_tickers = []
    resumed_tickers = []
    
    logger.info(f"DJIA Model Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total companies to process: {len(djia_tickers)}")
    logger.info(f"Training parameters: {training_params}")
    if args.parallel:
        logger.info(f"Parallel mode enabled with {args.workers} workers\n")
    else:
        logger.info("Sequential training mode\n")

    # Training mode selection (parallel or sequential)
    if args.parallel:
        logger.info(f"Starting parallel training with {args.workers} workers...")
        start_time = time.time()
        
        # Using ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all training tasks to the executor
            future_to_ticker = {
                executor.submit(
                    train_ticker, ticker, training_params, models_dir, args, timestamp
                ): ticker for ticker in djia_tickers
            }
            
            # Process completed tasks as they finish
            for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
                ticker = future_to_ticker[future]
                try:
                    status, elapsed_time, error, thread_log_file = future.result()
                    
                    # Log the result based on status
                    if status == "success":
                        successful_tickers.append(ticker)
                        logger.info(f"[{i+1}/{len(djia_tickers)}] {ticker}: Success (new) in {elapsed_time:.2f} seconds")
                    elif status == "resumed":
                        successful_tickers.append(ticker)
                        resumed_tickers.append(ticker)
                        logger.info(f"[{i+1}/{len(djia_tickers)}] {ticker}: Success (resumed) in {elapsed_time:.2f} seconds")
                    elif status == "skipped":
                        skipped_tickers.append(ticker)
                        logger.info(f"[{i+1}/{len(djia_tickers)}] {ticker}: Skipped (already exists)")
                    elif status == "failed":
                        failed_tickers.append((ticker, error))
                        logger.info(f"[{i+1}/{len(djia_tickers)}] {ticker}: Failed - {error}")

                    
                    # Read the thread's log file
                    if os.path.exists(thread_log_file):
                        with open(thread_log_file, 'r') as f:
                            thread_log_content = f.read()                
                            # Append to main logger
                            logger.info(f"\n--- Detailed log for {ticker} ---\n")
                            logger.info(thread_log_content)
                            logger.info("--- End of detailed log ---\n")          
            
                        os.remove(thread_log_file)

                except Exception as exc:
                    failed_tickers.append((ticker, str(exc)))
                    logger.info(f"[{i+1}/{len(djia_tickers)}] {ticker}: Failed with exception: {exc}")
        
        total_time = time.time() - start_time
        logger.info(f"\nParallel training completed in {total_time:.2f} seconds")
        
    else:
        # Sequential training (original approach)
        for i, ticker in enumerate(djia_tickers):
            logger.info(f"\n[{i+1}/{len(djia_tickers)}] Training model for {ticker}...")
            
            status, elapsed_time, error, thread_log_file = train_ticker(ticker, training_params, models_dir, args, timestamp)

            # Read the thread's log file
            if os.path.exists(thread_log_file):
                with open(thread_log_file, 'r') as f:
                    thread_log_content = f.read()                
                    # Append to main logger
                    logger.info(f"\n--- Detailed log for {ticker} ---\n")
                    logger.info(thread_log_content)
                    logger.info("--- End of detailed log ---\n")
                
                os.remove(thread_log_file)
            
            if status == "success":
                successful_tickers.append(ticker)
            elif status == "resumed":
                successful_tickers.append(ticker)
                resumed_tickers.append(ticker)
            elif status == "skipped":
                skipped_tickers.append(ticker)
            elif status == "failed":
                failed_tickers.append((ticker, error))
    
    # Write summary to log
    logger.info(f"\n\n--- Training Summary ---")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Successful (new): {len(successful_tickers) - len(resumed_tickers)} companies")
    logger.info(f"Successful (resumed): {len(resumed_tickers)} companies")
    logger.info(f"Failed: {len(failed_tickers)} companies")
    logger.info(f"Skipped: {len(skipped_tickers)} companies")
    
    if failed_tickers:
        logger.info("\nFailed companies:")
        for ticker, error in failed_tickers:
            logger.info(f"- {ticker}: {error}")
    
    # Print summary
    print("\n--- Training Summary ---")
    print(f"Total companies processed: {len(djia_tickers)}")
    print(f"Successfully trained (new): {len(successful_tickers) - len(resumed_tickers)}")
    print(f"Successfully trained (resumed): {len(resumed_tickers)}")
    print(f"Failed: {len(failed_tickers)}")
    print(f"Skipped: {len(skipped_tickers)}")
    
    if successful_tickers:
        print(f"\nSuccessful tickers: {', '.join(successful_tickers)}")
    
    if resumed_tickers:
        print(f"\nResumed tickers: {', '.join(resumed_tickers)}")
    
    if skipped_tickers:
        print(f"\nSkipped tickers: {', '.join(skipped_tickers)}")
    
    if failed_tickers:
        print("\nFailed tickers:")
        for ticker, error in failed_tickers:
            print(f"- {ticker}: {error}")
    
    print(f"\nAll model weights saved to '{models_dir}/' directory")
    print(f"Training log saved to '{log_file}'")
    
    # Return results as a dictionary
    return {
        "successful_tickers": successful_tickers,
        "resumed_tickers": resumed_tickers,
        "failed_tickers": failed_tickers,
        "skipped_tickers": skipped_tickers,
        "log_file": log_file,
        "models_dir": models_dir
    }

def main():
    """Entry point for command-line usage."""
    train_djia_models()

if __name__ == "__main__":
    main()
