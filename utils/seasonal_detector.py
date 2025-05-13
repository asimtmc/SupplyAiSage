import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

def detect_seasonal_period(time_series, max_lag=24):
    """
    Automatically detect the optimal seasonal period for a time series using autocorrelation.
    
    Parameters:
    -----------
    time_series : pandas.Series
        Time series data to analyze
    max_lag : int, optional (default=24)
        Maximum lag to consider for seasonality detection
        
    Returns:
    --------
    int
        The detected seasonal period, or a default if no clear seasonality is found
    """
    # Ensure we have enough data
    if len(time_series) < 4:
        return 2  # Default minimum
    
    # Limit max_lag to half the series length (statistical significance)
    max_lag = min(max_lag, len(time_series) // 2)
    
    # If we have very limited data, return a conservative estimate
    if max_lag < 4:
        return max(2, len(time_series) // 3)
    
    # Calculate autocorrelation function
    try:
        acf_values = acf(time_series.dropna(), nlags=max_lag, fft=True)
    except:
        # Fallback if ACF calculation fails
        return 12 if len(time_series) >= 24 else max(2, len(time_series) // 4)
    
    # First value (lag 0) is always 1, so we start from index 1
    acf_values = acf_values[1:]
    
    # Find peaks in ACF
    peaks = []
    
    # A point is a peak if it's larger than its neighbors and above a threshold
    threshold = 0.2  # Minimum correlation to consider
    
    for i in range(1, len(acf_values)-1):
        if (acf_values[i] > acf_values[i-1] and 
            acf_values[i] > acf_values[i+1] and 
            acf_values[i] > threshold):
            peaks.append((i+1, acf_values[i]))  # +1 because we skipped lag 0
    
    # If we found peaks, use the highest one as the seasonal period
    if peaks:
        # Sort peaks by correlation value (highest first)
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Common business periods to prioritize if they're close in correlation
        common_periods = [3, 4, 6, 12, 52, 365]
        
        # Get the highest correlation
        max_correlation = peaks[0][1]
        
        # Consider peaks that are close to the maximum correlation
        strong_peaks = [p for p in peaks if p[1] > max_correlation * 0.8]
        
        # Prioritize common business periods if they're among the strong peaks
        for period in common_periods:
            for lag, corr in strong_peaks:
                if abs(lag - period) <= max(1, period * 0.1):  # Allow 10% deviation
                    return period
        
        # If no common periods are found, return the lag with the highest correlation
        return peaks[0][0]
    
    # If no significant peaks, return a reasonable default based on data length
    if len(time_series) >= 24:
        return 12  # Default to annual seasonality for longer series
    elif len(time_series) >= 12:
        return 4   # Default to quarterly for medium series
    else:
        return max(2, len(time_series) // 4)  # Conservative default for short series