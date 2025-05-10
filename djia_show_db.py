import sqlite3
import argparse
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib.colors as mcolors
from datetime import datetime, timedelta


def show_db_content(db_path, output_image=None):
    """
    Display the content of the SQLite database in a readable format and optionally save it as a JPG.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ticker_transactions';")
    if not cursor.fetchone():
        print("No data found. The 'ticker_transactions' table does not exist.")
        conn.close()
        return

    # Fetch all rows from the table
    cursor.execute("SELECT * FROM ticker_transactions;")
    rows = cursor.fetchall()

    if not rows:
        print("No transactions found in the database.")
    else:
        # Fetch column names
        column_names = [description[0] for description in cursor.description]

        # Display the data in a tabular format
        print(tabulate(rows, headers=column_names, tablefmt="grid"))

        # If output_image is specified, generate a JPG
        if output_image:
            generate_table_image(rows, column_names, output_image)

    conn.close()

def generate_table_image(rows, column_names, output_image):
    """
    Generate a JPG image of the table with transactions grouped by ticker.
    Highlight transactions from the last 5 days with a dominant background color.
    """
    # Convert rows to a DataFrame for easier manipulation
    df = pd.DataFrame(rows, columns=column_names)

     # Assuming the second column (index 1) is the date column
    date_column = column_names[1]


    # Sort the DataFrame by ticker to ensure all ticker transactions are grouped together    
    df = df.sort_values(by=['ticker', date_column])
    
    # Reset rows from the sorted DataFrame
    rows = df.values.tolist()

    # Create a figure and axis for the table
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5))
    ax.axis('off')

    # Create a table
    table = Table(ax, bbox=[0, 0, 1, 1])
    n_rows, n_cols = df.shape

    # Add header row
    for col_idx, col_name in enumerate(column_names):
        table.add_cell(0, col_idx, width=1/n_cols, height=0.5, text=col_name, loc='center', facecolor='lightgrey')

    # Get the current date and calculate date 3 days ago for highlighting recent transactions
    today = datetime.today()
    x_days_ago = today - timedelta(days=3)

    # Define a more dominant highlight color for recent transactions
    highlight_color = '#FFD700'  # Gold color

    prev_ticker = None
    current_row_idx = 1  # Start from row 1 (row 0 is header)

    for row in rows:
        ticker = row[0]  # Assuming the first column is 'ticker'
        
        if ticker != prev_ticker:
            # Add an empty row with grayed-out background (thinner row)
            for col_idx in range(n_cols):
                table.add_cell(current_row_idx, col_idx, width=1/n_cols, height=0.3, text='', loc='center', facecolor='lightgrey', edgecolor='lightgrey')  # Adjusted height to 0.3 for thinner row
            current_row_idx += 1  # Increment row index for the actual transaction row
        
        # Add the transaction row
        for col_idx, cell in enumerate(row):
            # Default cell color
            color = 'white'
            
            # Highlight recent transactions (date column)
            if col_idx == 1:  # Assuming column index 1 is the date
                try:
                    if datetime.strptime(str(cell), "%Y-%m-%d") > x_days_ago:
                        color = highlight_color
                except ValueError:
                    # If date parsing fails, keep default color
                    pass
                        
            cell = table.add_cell(current_row_idx, col_idx, width=1/n_cols, height=0.5, 
                         text=str(cell), loc='center', 
                         facecolor=color, edgecolor='black')
                        
        prev_ticker = ticker
        current_row_idx += 1  # Increment row index for next row

    # Add the table to the axis
    ax.add_table(table)

    # Save the figure as a JPG
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Table saved as {output_image}")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Display the content of the ticker transactions database.")
    parser.add_argument('--db-path', type=str, default="ticker_transactions.db", 
                        help="Path to the SQLite database file (default: ticker_transactions.db)")
    parser.add_argument('--output-image', type=str, 
                        help="Path to save the table as a JPG image (optional)")

    # Parse arguments
    args = parser.parse_args()

    # Show the database content and optionally save it as an image
    show_db_content(args.db_path, args.output_image)

if __name__ == "__main__":
    main()