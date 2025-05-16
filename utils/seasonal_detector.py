import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL

def detect_seasonal_period(time_series, max_lag=24):
    """Automatically detect seasonal period in a time series.
    
    Args:
        time_series (pd.Series): Time series data
        max_lag (int): Maximum seasonal period to consider
        
    Returns:
        int: Detected seasonal period (0 if no seasonality detected)
    """
    if len(time_series) < max_lag * 2:
        # Not enough data for reliable detection
        return 0
        
    try:
        # Calculate autocorrelation
        acf_values = acf(time_series, nlags=max_lag, fft=True)
        
        # First value is always 1 (correlation with itself)
        acf_values = acf_values[1:]
        
        # Find peaks in autocorrelation
        # A peak is where the value is higher than its neighbors
        peaks = []
        for i in range(1, len(acf_values) - 1):
            if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                peaks.append((i+1, acf_values[i]))  # +1 because we removed the first ACF value
        
        # Sort peaks by correlation value (descending)
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top peak if significant
        if peaks and peaks[0][1] > 0.2:  # Threshold for significance
            return peaks[0][0]
            
        # If no significant peak found, try STL decomposition
        try:
            # For STL, the time series must be regular
            ts = pd.Series(time_series.values)
            
            # Try different seasonal periods
            best_strength = 0
            best_period = 0
            
            # Common seasonal periods to check
            periods_to_check = [3, 4, 6, 12]
            for period in periods_to_check:
                if len(ts) >= period * 2:  # Need enough data
                    try:
                        decomposition = STL(ts, seasonal=period).fit()
                        seasonal = decomposition.seasonal
                        residual = decomposition.resid
                        
                        # Calculate seasonal strength
                        seasonal_strength = 1 - np.var(residual) / np.var(seasonal + residual)
                        
                        if seasonal_strength > best_strength and seasonal_strength > 0.3:
                            best_strength = seasonal_strength
                            best_period = period
                    except:
                        continue
            
            return best_period
            
        except:
            # If all fails, return 0 (no seasonality)
            return 0
            
    except:
        # If any error occurs, return 0
        return 0