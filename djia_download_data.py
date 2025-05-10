# djia_download_data.py
import pandas as pd
import yfinance as yf
import time
import os
import random
import datetime
import logging
import argparse
from functools import wraps

# Initialize logger
logger = None

def setup_logging(log_file="djia_data_fetcher.log"):
    """Set up logging configuration"""
    global logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger

# Function to fetch historical data
def get_historical_data(ticker, period=5):
    """
    Fetch historical stock data for a given ticker
    
    Args:
        ticker (str): Stock ticker symbol
        period (int): Number of years of historical data to fetch
        
    Returns:
        pd.DataFrame: Historical stock data
    """
    # Directory for saving/loading historical data
    folder_path = "djia_historical_data"
    ticker = ticker.lower()
    filename = os.path.join(folder_path, f"{ticker.replace('.', '_')}_historical_data.csv")
    
    # Check environment variables
    save_to_file = os.environ.get('SAVE_DATA_TO_FILE', 'false').lower() == 'true'
    load_from_file = os.environ.get('LOAD_DATA_FROM_FILE', 'false').lower() == 'true'
    
    # Try to load from file if the environment variable is set and the file exists
    if load_from_file and os.path.exists(filename):
        try:
            data = pd.read_csv(filename, index_col=0, parse_dates=True)
            logger.info(f"Data loaded from {filename}")
            return data
        except Exception as e:
            logger.error(f"Error loading from file: {e}. Falling back to downloading data.")
    
    # Download data if not loading from file or if loading failed
    start_date = pd.Timestamp.now() - pd.DateOffset(years=period)
    data = yf.download(ticker)
    
    # Check if data was downloaded successfully (not empty)
    if not data.empty:
        data.columns = [col[0] for col in data.columns]
        # Filter the DataFrame to include only data from the specified period
        data = data[data.index >= start_date]
        
        # Save to file if environment variable is set
        if save_to_file:
            # Create the directory if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Save the data
            data.to_csv(filename)
            logger.info(f"Data saved to {filename}")
    
    return data

def fetch_djia_tickers():
    """
    Fetch the current list of DJIA tickers
    
    Returns:
        list: List of DJIA ticker symbols
    """
    try:
        # Get DJIA components from Wikipedia
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        tables = pd.read_html(url)
        
        for table in tables:
            if 'Symbol' in table.columns:
                # Extract the ticker symbols
                tickers = table['Symbol'].tolist()
                return tickers
        
        # Fallback to a hardcoded list if the above method fails
        logger.warning("Could not fetch DJIA tickers from Wikipedia, using hardcoded list")
        return [
            "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
            "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
            "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
        ]
    except Exception as e:
        logger.error(f"Error fetching DJIA tickers: {e}")
        # Hardcoded backup list
        return [
            "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
            "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
            "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
        ]

def retry_on_error(max_retries=5, backoff_in_seconds=300):
    """
    Decorator for retrying a function with exponential backoff
    
    Args:
        max_retries (int): Maximum number of retry attempts
        backoff_in_seconds (int): Wait time between retries in seconds
        
    Returns:
        function: Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait_time = backoff_in_seconds
                    retries += 1
                    if retries < max_retries:
                        logger.warning(f"Error: {e}. Retry {retries}/{max_retries} in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise
            return None
        return wrapper
    return decorator

def download_stock_data(ticker, period=5, max_retries=10, retry_wait=300):
    """
    Download stock data with retry logic
    
    Args:
        ticker (str): Stock ticker symbol
        period (int): Number of years of historical data to fetch
        max_retries (int): Maximum number of retry attempts
        retry_wait (int): Wait time between retries in seconds
        
    Returns:
        pd.DataFrame: Historical stock data
    """
    # Create a custom retry decorator
    custom_retry = retry_on_error(max_retries=max_retries, backoff_in_seconds=retry_wait)
    
    # Define the download function with retry
    @custom_retry
    def _download(ticker_symbol):
        logger.info(f"Downloading data for {ticker_symbol} (period: {period} years)")
        data = get_historical_data(ticker_symbol, period)
        
        if data is None or data.empty:
            raise Exception(f"Failed to download data for {ticker_symbol}")
        
        logger.info(f"Successfully downloaded data for {ticker_symbol} with {len(data)} rows")
        return data
    
    # Execute the download with retry
    return _download(ticker)

def fetch_all_djia_data(period=5, retry_wait=300, max_retries=10):
    """
    Fetch historical data for all DJIA tickers
    
    Args:
        period (int): Number of years of historical data to fetch
        retry_wait (int): Wait time between retries in seconds
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        tuple: Lists of successful and failed downloads
    """
    # Set up logging if not already done
    if logger is None:
        setup_logging()
        
    logger.info(f"Starting download with period={period} years, retry_wait={retry_wait}s, max_retries={max_retries}")
    
    # Enable saving to file
    os.environ['SAVE_DATA_TO_FILE'] = 'true'
    
    # Create output directory
    os.makedirs("djia_historical_data", exist_ok=True)
    
    # Get DJIA tickers
    tickers = fetch_djia_tickers()
    logger.info(f"Found {len(tickers)} DJIA tickers: {', '.join(tickers)}")
    
    # Track progress
    successful_downloads = []
    failed_downloads = []
    
    # Download data for each ticker
    for ticker in tickers:
        try:
            download_stock_data(ticker, period, max_retries, retry_wait)
            successful_downloads.append(ticker)
            
            # Add a small random delay to avoid rate limiting
            time.sleep(random.uniform(1.0, 3.0))
            
        except Exception as e:
            logger.error(f"Failed to download {ticker} after all retries: {e}")
            failed_downloads.append(ticker)
    
    # Summary
    logger.info(f"Download complete.")
    logger.info(f"Successfully downloaded data for {len(successful_downloads)} tickers: {', '.join(successful_downloads)}")
    
    if failed_downloads:
        logger.warning(f"Failed to download data for {len(failed_downloads)} tickers: {', '.join(failed_downloads)}")
    else:
        logger.info("All downloads completed successfully!")
        
    return successful_downloads, failed_downloads

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Download historical stock data for DJIA tickers')
    parser.add_argument('--period', type=int, default=5, 
                        help='Number of years of historical data to download (default: 5)')
    parser.add_argument('--retry-wait', type=int, default=300,
                        help='Wait time in seconds between retries (default: 300)')
    parser.add_argument('--max-retries', type=int, default=10,
                        help='Maximum number of retry attempts per ticker (default: 10)')
    return parser.parse_args()

def main():
    """
    Main function to run the script from command line
    """
    # Set up logging
    setup_logging()
    
    # Parse command line arguments
    args = parse_arguments()
    
    start_time = datetime.datetime.now()
    logger.info(f"Starting DJIA data fetching at {start_time}")
    
    try:
        fetch_all_djia_data(
            period=args.period,
            retry_wait=args.retry_wait,
            max_retries=args.max_retries
        )
    except Exception as e:
        logger.error(f"Critical error in main program: {e}")
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logger.info(f"Completed DJIA data fetching at {end_time}. Total duration: {duration}")

# This allows the script to be run directly or imported as a module
if __name__ == "__main__":
    main()