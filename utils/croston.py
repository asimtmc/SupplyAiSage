import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def croston_forecast(data, alpha=0.1, h=6, method='original'):
    """
    Implements Croston's method for intermittent demand forecasting.
    
    Args:
        data (pd.Series): Time series data with date index
        alpha (float): Smoothing parameter (0 < alpha < 1)
        h (int): Forecast horizon
        method (str): 'original' or 'sba' (Syntetos-Boylan Approximation)
        
    Returns:
        pd.Series: Forecast values with date index
    """
    if isinstance(data, pd.DataFrame):
        y = data['quantity'].values
    else:
        y = data.values
        
    y = np.array(y)
    
    # Initialization
    y_demand = y[y > 0]  # Demand sizes when demand occurs
    
    if len(y_demand) == 0:
        # All values are zero, return zeros
        if isinstance(data, pd.DataFrame):
            dates = pd.date_range(start=data['date'].iloc[-1] + pd.Timedelta(days=1), periods=h, freq='D')
        else:
            dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=h, freq='D')
        return pd.Series(np.zeros(h), index=dates)
    
    z_t = y_demand[0]  # Initial demand size
    
    # Find initial demand interval
    first_demand = np.where(y > 0)[0][0]
    if first_demand == 0:
        p_t = 1  # If demand occurs at first period
    else:
        p_t = first_demand  # Initial demand interval
    
    # Initialize variables
    z_result = []
    p_result = []
    
    for i in range(len(y)):
        if y[i] > 0:
            # Update demand size estimate
            z_t = alpha * y[i] + (1 - alpha) * z_t
            
            # Calculate periods since last demand
            if i > 0:
                p_t = alpha * (i - np.where(y[:i] > 0)[0][-1]) + (1 - alpha) * p_t
            
            z_result.append(z_t)
            p_result.append(p_t)
    
    # Generate forecast
    if method == 'original':
        # Original Croston's method
        forecast_value = z_result[-1] / p_result[-1]
    else:
        # Syntetos-Boylan Approximation (SBA)
        forecast_value = (1 - alpha/2) * z_result[-1] / p_result[-1]
    
    # Create date range for forecast
    if isinstance(data, pd.DataFrame):
        last_date = data['date'].iloc[-1]
        if pd.api.types.is_datetime64_any_dtype(last_date):
            dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h, freq='D')
        else:
            last_date = pd.to_datetime(last_date)
            dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h, freq='D')
    else:
        dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=h, freq='D')
    
    # Return forecast with dates
    return pd.Series([forecast_value] * h, index=dates)

def croston_optimized(data, parameters=None):
    """
    Run Croston's method with optimized parameters
    
    Args:
        data (pd.DataFrame): Time series data with date and quantity columns
        parameters (dict, optional): Parameters for the model
        
    Returns:
        tuple: (forecast, lower_bound, upper_bound)
    """
    if parameters is None:
        parameters = {
            'alpha': 0.1,
            'method': 'original'
        }
    
    # Extract parameters
    alpha = parameters.get('alpha', 0.1)
    method = parameters.get('method', 'original')
    h = parameters.get('h', 6)  # Forecast horizon
    
    # Generate forecast
    forecast = croston_forecast(data, alpha=alpha, h=h, method=method)
    
    # Calculate prediction intervals
    # For Croston, a simple approach is to use historical variance
    if isinstance(data, pd.DataFrame):
        historical_values = data['quantity'].values
    else:
        historical_values = data.values
    
    # Calculate standard deviation of historical non-zero demand
    non_zero_values = [x for x in historical_values if x > 0]
    if non_zero_values:
        std_dev = float(np.std(non_zero_values))
    else:
        std_dev = 0.1  # Default if no non-zero values
    
    # Create upper and lower bounds (1.96 for 95% confidence)
    lower_bound = forecast - 1.96 * std_dev
    lower_bound = lower_bound.clip(0)  # Ensure non-negative values
    upper_bound = forecast + 1.96 * std_dev
    
    return forecast, lower_bound, upper_bound