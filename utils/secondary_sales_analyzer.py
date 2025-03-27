import pandas as pd
import numpy as np
from scipy import signal, stats
from datetime import datetime
from utils.database import save_secondary_sales, get_secondary_sales

def preprocess_primary_sales_data(sales_data, sku):
    """
    Preprocess sales data for a specific SKU
    
    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Sales data with date, sku, and quantity columns
    sku : str
        SKU identifier
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed sales data for the specific SKU
    """
    # Filter for the specific SKU
    sku_data = sales_data[sales_data['sku'] == sku].copy()
    
    # Ensure data is sorted by date
    sku_data = sku_data.sort_values('date')
    
    # Set date as index for time series analysis
    sku_data.set_index('date', inplace=True)
    
    # Resample to ensure uniform time periods (monthly)
    if 'quantity' in sku_data.columns:
        sku_data = sku_data[['quantity']].resample('MS').sum()
    
    return sku_data

def rolling_average_denoiser(series, window_size=3):
    """
    Apply a rolling average to the time series to smooth out short-term fluctuations
    
    Parameters:
    -----------
    series : pandas.Series
        Time series data
    window_size : int, optional
        Size of the rolling window
    
    Returns:
    --------
    pandas.Series
        Smoothed time series
    """
    return series.rolling(window=window_size, center=True).mean().fillna(series)

def savitzky_golay_denoiser(series, window_size=5, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth the time series
    
    Parameters:
    -----------
    series : pandas.Series
        Time series data
    window_size : int, optional
        Size of the window (must be odd)
    polyorder : int, optional
        Order of the polynomial to fit
    
    Returns:
    --------
    pandas.Series
        Smoothed time series
    """
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Ensure window size is not larger than the series
    if window_size > len(series):
        window_size = min(len(series) - 2, 3)
        if window_size % 2 == 0:
            window_size += 1
    
    # Only apply if we have enough data points
    if len(series) > window_size and window_size > polyorder:
        try:
            smoothed = signal.savgol_filter(series, window_size, polyorder)
            return pd.Series(smoothed, index=series.index)
        except:
            return series
    else:
        return series

def decompose_time_series(series, method='robust'):
    """
    Decompose time series into trend, seasonality, and residual
    
    Parameters:
    -----------
    series : pandas.Series
        Time series data
    method : str, optional
        Method for decomposition ('robust' or 'stl')
    
    Returns:
    --------
    dict
        Dictionary with trend, seasonality, and residual components
    """
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
    
    # Handle short time series
    if len(series) < 12:
        # For very short series, return the series as trend and zeros for others
        return {
            'trend': series,
            'seasonal': pd.Series(0, index=series.index),
            'residual': pd.Series(0, index=series.index)
        }
    
    # Determine period based on data frequency
    period = 12  # Default for monthly data
    
    try:
        if method == 'robust':
            # Use STL decomposition which is more robust to outliers
            stl = STL(series, period=period, robust=True)
            result = stl.fit()
            return {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid
            }
        else:
            # Use classical decomposition
            result = seasonal_decompose(series, model='additive', period=period)
            return {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid
            }
    except:
        # Fallback for very noisy or short series
        return {
            'trend': rolling_average_denoiser(series, window_size=3),
            'seasonal': pd.Series(0, index=series.index),
            'residual': series - rolling_average_denoiser(series, window_size=3)
        }

def identify_sales_anomalies(series, z_threshold=2.0):
    """
    Identify anomalies in sales data that might represent sales target pressure
    
    Parameters:
    -----------
    series : pandas.Series
        Time series data
    z_threshold : float, optional
        Z-score threshold for anomaly detection
    
    Returns:
    --------
    pandas.Series
        Series with True for anomalies, False otherwise
    """
    # Calculate z-scores
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return pd.Series(False, index=series.index)
    
    z_scores = (series - mean) / std
    
    # Identify anomalies
    return abs(z_scores) > z_threshold

def estimate_secondary_sales(primary_sales, algorithm='robust_filter'):
    """
    Estimate secondary sales from primary sales data
    
    Parameters:
    -----------
    primary_sales : pandas.Series
        Primary sales time series data
    algorithm : str, optional
        Algorithm to use for estimation ('robust_filter', 'decomposition', 'arima_smoothing')
    
    Returns:
    --------
    tuple
        (secondary_sales, noise)
    """
    if len(primary_sales) < 3:
        # Not enough data for estimation
        return primary_sales, pd.Series(0, index=primary_sales.index)
    
    if algorithm == 'robust_filter':
        # Apply median filter followed by Savitzky-Golay for more robust smoothing
        median_filtered = primary_sales.rolling(window=3, center=True).median().fillna(primary_sales)
        secondary_sales = savitzky_golay_denoiser(median_filtered)
        
    elif algorithm == 'decomposition':
        # Use time series decomposition to remove irregular components
        decomposition = decompose_time_series(primary_sales)
        seasonal = decomposition['seasonal'].fillna(0)
        trend = decomposition['trend'].fillna(primary_sales)
        # Secondary sales is trend + seasonality
        secondary_sales = trend + seasonal
        
    elif algorithm == 'arima_smoothing':
        # Use ARIMA model to smooth and forecast
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings('ignore')
        
        try:
            # Fit ARIMA model
            model = ARIMA(primary_sales, order=(1, 0, 1))
            model_fit = model.fit()
            # Use fitted values as secondary sales estimate
            secondary_sales = pd.Series(model_fit.fittedvalues, index=primary_sales.index)
        except:
            # Fallback to simpler method if ARIMA fails
            secondary_sales = rolling_average_denoiser(primary_sales)
    
    else:
        # Default to simple moving average
        secondary_sales = rolling_average_denoiser(primary_sales)
    
    # Ensure no negative values
    secondary_sales = secondary_sales.clip(lower=0)
    
    # Calculate noise (difference between primary and estimated secondary)
    noise = primary_sales - secondary_sales
    
    return secondary_sales, noise

def analyze_sku_sales_pattern(sales_data, sku, algorithm='robust_filter'):
    """
    Analyze sales pattern for a specific SKU to estimate secondary sales
    
    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Sales data with date, sku, and quantity columns
    sku : str
        SKU identifier
    algorithm : str, optional
        Algorithm to use for estimation
    
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    # Preprocess data
    sku_data = preprocess_primary_sales_data(sales_data, sku)
    
    if sku_data.empty or len(sku_data) < 3:
        return {
            'sku': sku,
            'status': 'insufficient_data',
            'message': 'Not enough data for analysis'
        }
    
    # Get primary sales
    if 'quantity' in sku_data.columns:
        primary_sales = sku_data['quantity']
    else:
        return {
            'sku': sku,
            'status': 'error',
            'message': 'Quantity column not found in data'
        }
    
    # Estimate secondary sales
    secondary_sales, noise = estimate_secondary_sales(primary_sales, algorithm)
    
    # Create result dataframe
    result_df = pd.DataFrame({
        'date': primary_sales.index,
        'primary_sales': primary_sales.values,
        'secondary_sales': secondary_sales.values,
        'noise': noise.values
    })
    
    # Calculate metrics
    avg_primary = primary_sales.mean()
    avg_secondary = secondary_sales.mean()
    avg_noise = noise.mean()
    noise_percentage = (noise.abs().sum() / primary_sales.sum()) * 100 if primary_sales.sum() > 0 else 0
    
    # Save results to database
    for idx, row in result_df.iterrows():
        # Convert the date index to datetime if it's not already
        if hasattr(idx, 'to_pydatetime'):
            date_obj = idx.to_pydatetime()
        else:
            # If idx is an integer or other non-datetime type, handle accordingly
            # Convert to string and then parse with datetime
            from datetime import datetime
            try:
                date_obj = datetime.fromisoformat(str(idx))
            except:
                # If that fails, use the date as is (database will handle conversion)
                date_obj = idx
            
        save_secondary_sales(
            sku=sku,
            date=date_obj,
            primary_sales=float(row['primary_sales']),
            estimated_secondary_sales=float(row['secondary_sales']),
            noise=float(row['noise']),
            algorithm_used=algorithm
        )
    
    return {
        'sku': sku,
        'status': 'success',
        'algorithm': algorithm,
        'data': result_df,
        'metrics': {
            'avg_primary': avg_primary,
            'avg_secondary': avg_secondary,
            'avg_noise': avg_noise,
            'noise_percentage': noise_percentage
        }
    }

def bulk_analyze_sales(sales_data, selected_skus=None, algorithm='robust_filter', progress_callback=None):
    """
    Analyze sales patterns for multiple SKUs
    
    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Sales data with date, sku, and quantity columns
    selected_skus : list, optional
        List of SKUs to analyze. If None, analyzes all SKUs.
    algorithm : str, optional
        Algorithm to use for estimation
    progress_callback : function, optional
        Callback function for progress reporting
    
    Returns:
    --------
    dict
        Dictionary with analysis results for each SKU
    """
    # Get list of SKUs to analyze
    if selected_skus:
        skus_to_analyze = [sku for sku in selected_skus if sku in sales_data['sku'].unique()]
    else:
        skus_to_analyze = sales_data['sku'].unique().tolist()
    
    results = {}
    
    # Process each SKU
    for i, sku in enumerate(skus_to_analyze):
        # Report progress if callback is provided
        if progress_callback:
            progress_callback(i, sku, len(skus_to_analyze))
        
        # Analyze SKU
        result = analyze_sku_sales_pattern(sales_data, sku, algorithm)
        results[sku] = result
    
    return results