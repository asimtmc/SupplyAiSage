import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_sales_data(data, date_col='date', sku_col='sku', quantity_col='quantity'):
    """Process raw sales data into a format suitable for forecasting.
    
    Args:
        data (pd.DataFrame): Raw sales data
        date_col (str): Column name for date
        sku_col (str): Column name for SKU
        quantity_col (str): Column name for quantity
        
    Returns:
        pd.DataFrame: Processed data
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Map columns if using the Sales History Excel format
    if 'FG Code' in df.columns and 'QTY_MONTH' in df.columns:
        column_mapping = {
            'FG Code': 'sku',
            'QTY_MONTH': 'quantity',
            'YR_MONTH_NR': 'date'
        }
        
        # Rename columns to standard format
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
                
        # Update column references
        if date_col != 'date' and date_col == 'YR_MONTH_NR':
            date_col = 'date'
        if sku_col != 'sku' and sku_col == 'FG Code':
            sku_col = 'sku'
        if quantity_col != 'quantity' and quantity_col == 'QTY_MONTH':
            quantity_col = 'quantity'
    
    # Check if required columns exist
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found. Available columns: {list(df.columns)}")
    if sku_col not in df.columns:
        raise ValueError(f"SKU column '{sku_col}' not found. Available columns: {list(df.columns)}")
    if quantity_col not in df.columns:
        raise ValueError(f"Quantity column '{quantity_col}' not found. Available columns: {list(df.columns)}")
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
    # Handle any date parsing errors
    if df[date_col].isna().any():
        print(f"Warning: Some dates could not be parsed. {df[date_col].isna().sum()} rows affected.")
        df = df.dropna(subset=[date_col])
    
    # Ensure quantity column is numeric
    df[quantity_col] = pd.to_numeric(df[quantity_col], errors='coerce')
    df[quantity_col] = df[quantity_col].fillna(0)
    
    # Ensure SKU column is string
    df[sku_col] = df[sku_col].astype(str)
    
    # Sort by date
    df = df.sort_values(by=date_col)
    
    return df

def get_sku_data(data, sku, date_col='date', quantity_col='quantity'):
    """Extract data for a specific SKU.
    
    Args:
        data (pd.DataFrame): Processed sales data
        sku (str): SKU to extract
        date_col (str): Column name for date
        quantity_col (str): Column name for quantity
        
    Returns:
        pd.DataFrame: Data for the specified SKU
    """
    sku_data = data[data['sku'] == sku].copy()
    
    # Select only needed columns and sort by date
    sku_data = sku_data[[date_col, quantity_col]].sort_values(by=date_col)
    
    return sku_data

def check_intermittent_demand(data, threshold=0.3):
    """Check if a time series has intermittent demand.
    
    Args:
        data (pd.DataFrame): Time series with date and quantity columns
        threshold (float): Threshold for proportion of zero values
        
    Returns:
        bool: True if demand is intermittent, False otherwise
    """
    zero_proportion = (data['quantity'] == 0).mean()
    return zero_proportion >= threshold

def prepare_data_for_forecasting(data, frequency='ME'):
    """Prepare data for forecasting by resampling to a regular time frequency.
    
    Args:
        data (pd.DataFrame): Time series with date and quantity columns
        frequency (str): Pandas frequency string (e.g., 'D', 'W', 'ME' for month end)
        
    Returns:
        pd.DataFrame: Resampled data
    """
    # Set date as index
    data_idx = data.set_index('date')
    
    # Map old deprecated frequency codes to newer ones
    freq_mapping = {
        'M': 'ME',  # Month end
        'W': 'W-SUN',  # Week end (Sunday)
        'Q': 'QE',  # Quarter end
        'Y': 'YE'  # Year end
    }
    
    # Apply mapping if needed
    if frequency in freq_mapping:
        freq_to_use = freq_mapping[frequency]
    else:
        freq_to_use = frequency
    
    # Resample to specified frequency
    resampled = data_idx.resample(freq_to_use)['quantity'].sum().reset_index()
    
    return resampled