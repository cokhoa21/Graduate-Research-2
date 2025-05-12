import pandas as pd
import os
from datetime import datetime
import sys

def remove_duplicate_cookies(input_file, output_file=None):
    """
    Remove duplicate cookies based on name and value
    
    Args:
        input_file (str): Path to input CSV file containing cookies data
        output_file (str, optional): Path to output file. If None, will generate name based on timestamp
    """
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Get initial count
    initial_count = len(df)
    print(f"Initial number of rows: {initial_count}")
    
    # Remove duplicates based on name and value
    df_no_duplicates = df.drop_duplicates(subset=['name', 'value'], keep='first')
    
    # Get final count
    final_count = len(df_no_duplicates)
    removed_count = initial_count - final_count
    print(f"Removed {removed_count} duplicate rows")
    print(f"Final number of rows: {final_count}")
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join('output', f'cookies_no_duplicates_{timestamp}.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to output file
    print(f"Saving results to {output_file}...")
    df_no_duplicates.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python remove_duplicate_cookies.py <input_file_path> <output_file_path>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    remove_duplicate_cookies(input_file, output_file) 