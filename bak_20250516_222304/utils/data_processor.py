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
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
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

def prepare_data_for_forecasting(data, frequency='M'):
    """Prepare data for forecasting by resampling to a regular time frequency.
    
    Args:
        data (pd.DataFrame): Time series with date and quantity columns
        frequency (str): Pandas frequency string (e.g., 'D', 'W', 'M')
        
    Returns:
        pd.DataFrame: Resampled data
    """
    # Set date as index
    data_idx = data.set_index('date')
    
    # Resample to specified frequency
    resampled = data_idx.resample(frequency)['quantity'].sum().reset_index()
    
    return resampled