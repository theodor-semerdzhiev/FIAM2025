# This script reads a Parquet file of stock returns and filters for extreme positive and negative returns.

import pandas as pd
import numpy as np

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the path to your Parquet file
    file_path = r"C:\_Files\Personal\Projects\FIAM\FIAM2025\data\ret_sample.parquet"

    print(f"--- Loading data from '{file_path}' ---")

    try:
        # Read the entire Parquet file into a pandas DataFrame
        df = pd.read_parquet(file_path)
        print("Data loaded successfully.")

        # --- Filter for rows where stock_ret is > 500 OR < -1 ---
        print("\n--- Filtering for stock returns > 500 or < -1 ---")
        extreme_returns_df = df[(df['stock_ret'] > 500) | (df['stock_ret'] < -1)]

        # --- Display the results ---
        if not extreme_returns_df.empty:
            print(f"Found {len(extreme_returns_df)} rows with returns greater than 500 or less than -1.")
            print("Displaying filtered rows (id and return columns only):")
            # Select and print only the gvkey, id, and stock_ret columns
            # Assuming 'gvkey' and 'id' are the identifier columns based on user request.
            print(extreme_returns_df[['gvkey', 'id', 'stock_ret']].to_string())
        else:
            print("No rows found with stock returns greater than 500 or less than -1.")

    except FileNotFoundError:
        print(f"Error: The file was not found at '{file_path}'")
    except KeyError:
        print("Error: The DataFrame does not contain the expected columns ('gvkey', 'id', 'stock_ret').")
        print("Please check the column names in your Parquet file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

