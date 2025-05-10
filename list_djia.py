import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import re

def get_djia_companies():
    """
    Get the list of all companies in the Dow Jones Industrial Average (DJIA) and return as a DataFrame.
    
    Returns:
        DataFrame: A pandas DataFrame containing company information.
    """
    try:
        # Method 1: Using Wikipedia (reliable method)
        print("Fetching DJIA companies from Wikipedia...")
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all tables on the page
        tables = soup.find_all('table', {'class': 'wikitable'})
        
        # Look for the table that contains the DJIA components
        component_table = None
        for table in tables:
            caption = table.find('caption')
            if caption and 'component' in caption.text.lower():
                component_table = table
                break
                
        # If we didn't find a table with a caption, try looking for table headers
        if component_table is None:
            for table in tables:
                headers = table.find_all('th')
                header_texts = [header.text.strip().lower() for header in headers]
                # Check if this table has headers that suggest it contains stock components
                if any('symbol' in text for text in header_texts) and any('company' in text for text in header_texts):
                    component_table = table
                    break
        
        if component_table is None:
            raise Exception("Could not find DJIA components table on Wikipedia")
        
        # Extract column indices for company and ticker
        header_row = component_table.find('tr')
        headers = [th.text.strip().lower() for th in header_row.find_all('th')]
        
        # Figure out which column is which
        company_idx = None
        ticker_idx = None
        exchange_idx = None
        
        for i, header in enumerate(headers):
            if 'company' in header:
                company_idx = i
            elif 'symbol' in header or 'ticker' in header:
                ticker_idx = i
            elif 'exchange' in header:
                exchange_idx = i
        
        # Initialize lists to store company data
        tickers = []
        names = []
        
        # Parse rows - skip header
        rows = component_table.find_all('tr')[1:]  # Skip header row
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= max(company_idx, ticker_idx) + 1 if company_idx is not None and ticker_idx is not None else 2:
                company_name = cols[company_idx].text.strip() if company_idx is not None else ""
                ticker_text = cols[ticker_idx].text.strip() if ticker_idx is not None else ""
                
                # Check if ticker contains exchange information
                if "NYSE:" in ticker_text or "NASDAQ:" in ticker_text:
                    ticker = ticker_text.split(":")[-1].strip()
                else:
                    ticker = ticker_text
                
                # Add to lists if we have valid data
                if ticker and company_name:
                    tickers.append(ticker)
                    names.append(company_name)
        
        # If we didn't get data with the header method, try a more manual approach
        if not tickers:
            print("Using alternative parsing method...")
            rows = component_table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    # Look for patterns: NYSE: XXX or NASDAQ: XXX in either column
                    for col in cols:
                        text = col.text.strip()
                        exchange_match = re.search(r'(NYSE|NASDAQ):\s*(\w+)', text)
                        if exchange_match:
                            ticker = exchange_match.group(2)
                            # Find the company name (usually the longest text in the row)
                            company_texts = [c.text.strip() for c in cols if len(c.text.strip()) > 0 and 'NYSE:' not in c.text and 'NASDAQ:' not in c.text]
                            if company_texts:
                                company_name = max(company_texts, key=len)
                                tickers.append(ticker)
                                names.append(company_name)
                                break
        
        # Create a DataFrame
        df = pd.DataFrame({
            'Ticker': tickers,
            'Company Name': names
        })
        
        return df
        
    except Exception as e:
        print(f"Error fetching from Wikipedia: {e}")
        try:
            # Method 2: Manual list of DJIA components
            print("Using fallback method with manual list...")
            
            # Current DJIA components as of last update (30 companies)
            djia_components = [
                ('AAPL', 'Apple Inc.'),
                ('AMGN', 'Amgen Inc.'),
                ('AXP', 'American Express Co.'),
                ('BA', 'Boeing Co.'),
                ('CAT', 'Caterpillar Inc.'),
                ('CRM', 'Salesforce Inc.'),
                ('CSCO', 'Cisco Systems Inc.'),
                ('CVX', 'Chevron Corp.'),
                ('DIS', 'Walt Disney Co.'),
                ('DOW', 'Dow Inc.'),
                ('GS', 'Goldman Sachs Group Inc.'),
                ('HD', 'Home Depot Inc.'),
                ('HON', 'Honeywell International Inc.'),
                ('IBM', 'International Business Machines Corp.'),
                ('INTC', 'Intel Corp.'),
                ('JNJ', 'Johnson & Johnson'),
                ('JPM', 'JPMorgan Chase & Co.'),
                ('KO', 'Coca-Cola Co.'),
                ('MCD', 'McDonald\'s Corp.'),
                ('MMM', '3M Co.'),
                ('MRK', 'Merck & Co. Inc.'),
                ('MSFT', 'Microsoft Corp.'),
                ('NKE', 'Nike Inc.'),
                ('PG', 'Procter & Gamble Co.'),
                ('TRV', 'Travelers Companies Inc.'),
                ('UNH', 'UnitedHealth Group Inc.'),
                ('V', 'Visa Inc.'),
                ('VZ', 'Verizon Communications Inc.'),
                ('WBA', 'Walgreens Boots Alliance Inc.'),
                ('WMT', 'Walmart Inc.')
            ]
            
            # Create DataFrame from manual list
            df = pd.DataFrame(djia_components, columns=['Ticker', 'Company Name'])
            return df
            
        except Exception as e:
            print(f"Error with fallback method: {e}")
            
            # Method 3: Using yfinance if available
            try:
                print("Trying yfinance for DJIA data...")
                # Get DJIA ETF holdings (DIA - SPDR Dow Jones Industrial Average ETF)
                dia = yf.Ticker("DIA")
                holdings = dia.holdings
                
                if holdings is not None and not holdings.empty:
                    # Create a dataframe with just the company symbols and names
                    df = holdings[['symbol', 'name']].rename(columns={'symbol': 'Ticker', 'name': 'Company Name'})
                    return df
                else:
                    raise Exception("Could not retrieve DIA holdings data")
            except Exception as e:
                print(f"Error with yfinance method: {e}")
                
                # Return an empty DataFrame with the correct columns
                return pd.DataFrame(columns=['Ticker', 'Company Name'])

def main():
    # Get DJIA companies as a DataFrame
    djia_df = get_djia_companies()
    
    # Check if we got any data
    if djia_df.empty:
        print("No data was retrieved. Please check your internet connection or try again later.")
        return djia_df
    
    # Display the number of companies
    print(f"\nNumber of companies in the DJIA: {len(djia_df)}")
    
    # Display all companies (since DJIA only has 30)
    print("\nCompanies in the Dow Jones Industrial Average:")
    pd.set_option('display.max_rows', None)  # Show all rows
    print(djia_df)
    pd.reset_option('display.max_rows')  # Reset to default
    
    return djia_df

if __name__ == "__main__":
    # Store the DataFrame in a variable that can be used for further analysis
    djia_companies = main()
    
    # Example of how to use the DataFrame
    print("\nDataFrame is ready for use in your analysis.")
