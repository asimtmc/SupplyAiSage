import pandas as pd
import numpy as np
import json

def main():
    # Import the data loader function
    from utils.data_loader import load_data_from_database
    
    # Call the function and check what it returns
    data = load_data_from_database()
    
    # Print the type of data returned
    print(f"Type of data returned: {type(data)}")
    
    # If it's a dictionary, check the keys
    if isinstance(data, dict):
        print(f"Keys in data: {list(data.keys())}")
        
        # Check if sales_data exists and its type
        if 'sales_data' in data:
            print(f"Type of sales_data: {type(data['sales_data'])}")
            
            # Check if sales_data has content
            if isinstance(data['sales_data'], pd.DataFrame):
                print(f"Shape of sales_data: {data['sales_data'].shape}")
                print(f"Columns in sales_data: {list(data['sales_data'].columns)}")
                
                # Print a few rows
                print("\nFirst few rows of sales_data:")
                print(data['sales_data'].head())
                
                # Check unique SKUs
                print(f"\nUnique SKUs: {data['sales_data']['sku'].unique()}")
    else:
        print("Data is not a dictionary as expected.")

if __name__ == "__main__":
    main()