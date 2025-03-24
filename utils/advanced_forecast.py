import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.api import VAR, SimpleExpSmoothing
from statsmodels.nonparametric.smoothers_lowess import lowess
from prophet import Prophet
import warnings
from datetime import datetime, timedelta
import json
import math
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from scipy import stats, signal

# Import the database functionality
from utils.database import save_forecast_result, get_forecast_history

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ================================
# 1. ADVANCED DATA PREPROCESSING
# ================================

def detect_outliers(series, method='zscore', threshold=3):
    """
    Detect outliers in time series data using multiple methods
    
    Parameters:
    -----------
    series : pandas.Series
        Time series data with potential outliers
    method : str, optional
        Method to use for outlier detection ('zscore', 'iqr', or 'isolation_forest')
    threshold : float, optional
        Threshold for outlier detection
        
    Returns:
    --------
    pandas.Series
        Boolean mask where True indicates an outlier
    """
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        return pd.Series(z_scores > threshold, index=series.index)
    
    elif method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (series < lower_bound) | (series > upper_bound)
    
    else:
        # Default to Z-score if method not recognized
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        return pd.Series(z_scores > threshold, index=series.index)

def clean_time_series(series, outlier_method='zscore', smoothing=False, smoothing_window=3):
    """
    Clean time series data by handling outliers and applying optional smoothing
    
    Parameters:
    -----------
    series : pandas.Series
        Time series data to clean
    outlier_method : str, optional
        Method to use for outlier detection ('zscore', 'iqr')
    smoothing : bool, optional
        Whether to apply smoothing after outlier treatment
    smoothing_window : int, optional
        Window size for smoothing if applied
        
    Returns:
    --------
    pandas.Series
        Cleaned time series data
    """
    # Make a copy of the series
    cleaned = series.copy()
    
    # Detect outliers
    outliers = detect_outliers(cleaned, method=outlier_method)
    
    # Replace outliers with interpolated values or local median
    if outliers.any():
        # Get non-outlier values for local statistics
        non_outliers = cleaned[~outliers]
        
        if len(non_outliers) > 0:
            # For each outlier, replace with local median or interpolated value
            for idx in cleaned[outliers].index:
                # Try to find nearby non-outlier values
                window = 5  # Look at 5 surrounding points
                
                # Get slice of original series around outlier
                slice_start = max(0, cleaned.index.get_loc(idx) - window//2)
                slice_end = min(len(cleaned), cleaned.index.get_loc(idx) + window//2 + 1)
                local_slice = cleaned.iloc[slice_start:slice_end]
                
                # Filter out other outliers from local slice
                local_non_outliers = local_slice[~outliers.iloc[slice_start:slice_end]]
                
                if len(local_non_outliers) > 0:
                    # Replace with local median if available
                    cleaned.loc[idx] = local_non_outliers.median()
                else:
                    # Otherwise use global statistics or interpolation
                    cleaned.loc[idx] = non_outliers.median()
    
    # Apply smoothing if requested
    if smoothing and smoothing_window > 1:
        # Use centered moving average for smoothing
        cleaned = cleaned.rolling(window=smoothing_window, center=True).mean()
        
        # For the edges where rolling mean produces NaN, use original values
        cleaned = cleaned.fillna(series)
    
    return cleaned

def extract_calendar_features(dates):
    """
    Extract calendar-based features from dates for time series analysis
    
    Parameters:
    -----------
    dates : array-like
        Array of dates
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with extracted calendar features
    """
    dates_df = pd.DataFrame({'date': pd.to_datetime(dates)})
    
    # Extract basic features
    dates_df['month'] = dates_df['date'].dt.month
    dates_df['quarter'] = dates_df['date'].dt.quarter
    dates_df['year'] = dates_df['date'].dt.year
    dates_df['day_of_week'] = dates_df['date'].dt.dayofweek
    dates_df['day_of_month'] = dates_df['date'].dt.day
    dates_df['day_of_year'] = dates_df['date'].dt.dayofyear
    dates_df['week_of_year'] = dates_df['date'].dt.isocalendar().week
    
    # Create cyclical features for month (converts to circular coordinates)
    dates_df['month_sin'] = np.sin(2 * np.pi * dates_df['month'] / 12)
    dates_df['month_cos'] = np.cos(2 * np.pi * dates_df['month'] / 12)
    
    # Create cyclical features for day of week
    dates_df['day_sin'] = np.sin(2 * np.pi * dates_df['day_of_week'] / 7)
    dates_df['day_cos'] = np.cos(2 * np.pi * dates_df['day_of_week'] / 7)
    
    # Create cyclical features for quarter
    dates_df['quarter_sin'] = np.sin(2 * np.pi * dates_df['quarter'] / 4)
    dates_df['quarter_cos'] = np.cos(2 * np.pi * dates_df['quarter'] / 4)
    
    # Drop the original date column
    dates_df = dates_df.drop('date', axis=1)
    
    return dates_df

def detect_change_points(series, method='pelt', penalty=15):
    """
    Detect change points in time series data
    
    Parameters:
    -----------
    series : pandas.Series
        Time series data
    method : str, optional
        Method to use for change point detection ('pelt', 'binary_segmentation', or 'window')
    penalty : float, optional
        Penalty parameter for change point detection algorithms
        
    Returns:
    --------
    list
        List of indices where change points were detected
    """
    try:
        # Simple method using rolling statistics
        if method == 'window':
            # Use rolling standard deviation and mean
            roll_std = series.rolling(window=min(5, len(series) // 3)).std()
            roll_mean = series.rolling(window=min(5, len(series) // 3)).mean()
            
            # Compute rate of change
            roll_std_change = roll_std.pct_change().abs()
            roll_mean_change = roll_mean.pct_change().abs()
            
            # Detect points where changes exceed threshold
            std_changes = np.where(roll_std_change > np.percentile(roll_std_change.dropna(), 90))[0]
            mean_changes = np.where(roll_mean_change > np.percentile(roll_mean_change.dropna(), 90))[0]
            
            # Combine the change points
            change_points = np.unique(np.concatenate([std_changes, mean_changes]))
            return change_points.tolist()
        else:
            # Default to window method if other methods not available
            return detect_change_points(series, method='window', penalty=penalty)
            
    except Exception as e:
        print(f"Change point detection error: {str(e)}")
        return []

def segment_time_series(data, column='quantity'):
    """
    Segment time series into different regimes based on detected change points
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing time series data
    column : str, optional
        Column name containing the time series values
        
    Returns:
    --------
    list
        List of segment boundaries (indices)
    """
    try:
        # Get the series
        series = data[column]
        
        # Detect change points
        change_points = detect_change_points(series)
        
        # Add start and end points to create segments
        segment_boundaries = [0] + change_points + [len(series)]
        
        # Sort and remove duplicates
        segment_boundaries = sorted(list(set(segment_boundaries)))
        
        return segment_boundaries
    
    except Exception as e:
        print(f"Time series segmentation error: {str(e)}")
        return [0, len(data)]

def extract_advanced_features(series):
    """
    Extract advanced statistical features from time series
    
    Parameters:
    -----------
    series : pandas.Series
        Time series data
        
    Returns:
    --------
    dict
        Dictionary of extracted features
    """
    features = {}
    
    # Basic statistics
    features['mean'] = series.mean()
    features['median'] = series.median()
    features['std'] = series.std()
    features['min'] = series.min()
    features['max'] = series.max()
    features['range'] = series.max() - series.min()
    features['iqr'] = series.quantile(0.75) - series.quantile(0.25)
    
    # Shape features
    features['skewness'] = series.skew()
    features['kurtosis'] = series.kurtosis()
    
    # Volatility features
    features['cv'] = features['std'] / features['mean'] if features['mean'] != 0 else np.nan
    features['pct_change_mean'] = series.pct_change().mean()
    features['pct_change_std'] = series.pct_change().std()
    
    # Trend features
    try:
        x = np.arange(len(series))
        trend_coef = np.polyfit(x, series.values, 1)
        features['trend_slope'] = trend_coef[0]
        features['trend_intercept'] = trend_coef[1]
    except:
        features['trend_slope'] = 0
        features['trend_intercept'] = 0
    
    # Stationarity features
    try:
        adf_result = adfuller(series.dropna())
        features['adf_pvalue'] = adf_result[1]
        features['adf_statistic'] = adf_result[0]
    except:
        features['adf_pvalue'] = 1
        features['adf_statistic'] = 0
    
    # Autocorrelation features
    try:
        acf_1 = acf(series.dropna(), nlags=1)[1]
        features['autocorrelation_lag1'] = acf_1
    except:
        features['autocorrelation_lag1'] = 0
    
    try:
        if len(series) >= 12:
            acf_12 = acf(series.dropna(), nlags=12)[12]
            features['autocorrelation_lag12'] = acf_12
        else:
            features['autocorrelation_lag12'] = 0
    except:
        features['autocorrelation_lag12'] = 0
    
    # Intermittency features
    features['zero_count'] = (series == 0).sum()
    features['zero_ratio'] = features['zero_count'] / len(series)
    
    return features

# ================================
# 2. ADVANCED FORECASTING MODELS
# ================================

def train_auto_arima(data, exog=None, max_order=5, seasonal=False, m=12):
    """
    Automatically find the best ARIMA or SARIMA parameters based on AIC
    
    Parameters:
    -----------
    data : pandas.Series
        Time series data for training
    exog : pandas.DataFrame, optional
        Exogenous variables for ARIMA/SARIMA
    max_order : int, optional
        Maximum order to consider for ARIMA parameters
    seasonal : bool, optional
        Whether to include seasonal components
    m : int, optional
        Seasonal period
        
    Returns:
    --------
    tuple
        (best_model, best_order, best_seasonal_order)
    """
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    best_model = None
    
    # Check if data is too short for seasonal analysis
    if seasonal and len(data) < 2 * m:
        seasonal = False
    
    # Limit max_order for short series
    if len(data) < 20:
        max_order = min(2, max_order)
    
    # Cap the search space based on data size to avoid overfitting
    max_p = min(max_order, len(data) // 10)
    max_d = min(2, max_order // 2)  # Differencing is typically 0, 1, or 2
    max_q = min(max_order, len(data) // 10)
    
    # Estimate stationary series
    try:
        # Check for stationarity
        adf_result = adfuller(data)
        kpss_result = kpss(data)
        
        # If clear non-stationarity, suggest d=1
        suggested_d = 1 if (adf_result[1] > 0.05 or kpss_result[1] < 0.05) else 0
    except:
        suggested_d = 1
    
    # Generate combinations more efficiently by focusing on most likely values
    p_values = [0, 1, 2] if max_p >= 2 else list(range(max_p + 1))
    d_values = [suggested_d, 0, 1] if max_d >= 1 else [0]
    q_values = [0, 1, 2] if max_q >= 2 else list(range(max_q + 1))
    
    # For seasonal components, use simpler combinations
    if seasonal:
        seasonal_p_values = [0, 1]
        seasonal_d_values = [0, 1]
        seasonal_q_values = [0, 1]
    
    # Progress tracking
    total_combinations = len(p_values) * len(d_values) * len(q_values)
    if seasonal:
        total_combinations *= len(seasonal_p_values) * len(seasonal_d_values) * len(seasonal_q_values)
    
    print(f"Testing {total_combinations} ARIMA models...")
    
    # For very small datasets, use simplified grid
    if len(data) < 15:
        orders = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
        seasonal_orders = [(0, 0, 0, 0)]
    else:
        # Generate all combinations of p, d, q
        orders = [(p, d, q) for p in p_values for d in d_values for q in q_values]
        
        # For seasonal models, also generate seasonal orders
        if seasonal:
            seasonal_orders = [(sp, sd, sq, m) for sp in seasonal_p_values 
                             for sd in seasonal_d_values 
                             for sq in seasonal_q_values]
        else:
            seasonal_orders = [(0, 0, 0, 0)]
    
    # Try different combinations
    for order in orders:
        for seasonal_order in seasonal_orders:
            # Skip complex models for small datasets to prevent overfitting
            if len(data) < 20 and sum(order) + sum(seasonal_order[:3]) > 3:
                continue
                
            try:
                # For SARIMA
                if seasonal and seasonal_order != (0, 0, 0, 0):
                    model = SARIMAX(data, 
                                   order=order, 
                                   seasonal_order=seasonal_order,
                                   exog=exog,
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
                    
                # For ARIMA
                else:
                    model = ARIMA(data, 
                                 order=order,
                                 exog=exog)
                
                # Fit model with limited iterations for speed
                model_fit = model.fit(disp=0, maxiter=100)
                
                # Check if this model is better
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
                    best_seasonal_order = seasonal_order if seasonal else None
                    best_model = model_fit
            
            except Exception as e:
                continue
    
    # If no model was successfully fit, use a simple fallback
    if best_model is None:
        try:
            model = ARIMA(data, order=(1, 1, 0))
            best_model = model.fit(disp=0)
            best_order = (1, 1, 0)
            best_seasonal_order = None
        except:
            return None, None, None
    
    return best_model, best_order, best_seasonal_order

def create_advanced_lstm_model(input_shape, units_list=[64, 32], dropout_rate=0.2, learning_rate=0.001):
    """
    Create an advanced LSTM model with multiple layers and regularization
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    units_list : list, optional
        List of units for each LSTM layer
    dropout_rate : float, optional
        Dropout rate for regularization
    learning_rate : float, optional
        Learning rate for Adam optimizer
    
    Returns:
    --------
    tensorflow.keras.models.Sequential
        Compiled LSTM model
    """
    model = Sequential()
    
    # First LSTM layer with return sequences if there are more LSTM layers
    if len(units_list) > 1:
        model.add(LSTM(units=units_list[0], 
                     activation='tanh',
                     return_sequences=True,
                     input_shape=input_shape,
                     recurrent_dropout=dropout_rate * 0.5))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Middle LSTM layers if any
        for i in range(1, len(units_list) - 1):
            model.add(LSTM(units=units_list[i], 
                         activation='tanh',
                         return_sequences=True,
                         recurrent_dropout=dropout_rate * 0.5))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Last LSTM layer with no return sequences
        model.add(LSTM(units=units_list[-1], 
                     activation='tanh',
                     recurrent_dropout=dropout_rate * 0.5))
    else:
        # Single LSTM layer
        model.add(LSTM(units=units_list[0], 
                     activation='tanh',
                     input_shape=input_shape,
                     recurrent_dropout=dropout_rate * 0.5))
    
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model with Adam optimizer
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def create_tcn_model(input_shape, filters=64, kernel_size=3, dilations=[1, 2, 4, 8], dropout_rate=0.2):
    """
    Create a Temporal Convolutional Network (TCN) model
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    filters : int, optional
        Number of filters in convolutional layers
    kernel_size : int, optional
        Size of the convolutional kernel
    dilations : list, optional
        Dilation rates for successive layers
    dropout_rate : float, optional
        Dropout rate for regularization
    
    Returns:
    --------
    tensorflow.keras.models.Sequential
        Compiled TCN model
    """
    model = Sequential()
    
    # First convolutional layer
    model.add(Conv1D(filters=filters, 
                   kernel_size=kernel_size, 
                   dilation_rate=dilations[0],
                   padding='causal', 
                   activation='relu', 
                   input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Additional convolutional layers with increasing dilation rates
    for dilation in dilations[1:]:
        model.add(Conv1D(filters=filters, 
                       kernel_size=kernel_size, 
                       dilation_rate=dilation,
                       padding='causal', 
                       activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Global pooling
    model.add(GlobalAveragePooling1D())
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    return model

def train_prophet_model(data, seasonality_mode='additive', changepoint_prior_scale=0.05, 
                       yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                       custom_seasonalities=None):
    """
    Train a Prophet model with customizable parameters
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with 'ds' (dates) and 'y' (values) columns
    seasonality_mode : str, optional
        Seasonality mode ('additive' or 'multiplicative')
    changepoint_prior_scale : float, optional
        Parameter modulating the flexibility of the trend
    yearly_seasonality : bool or int, optional
        Whether to include yearly seasonality
    weekly_seasonality : bool or int, optional
        Whether to include weekly seasonality
    daily_seasonality : bool or int, optional
        Whether to include daily seasonality
    custom_seasonalities : list of dict, optional
        List of custom seasonalities to add (each with 'name', 'period', 'fourier_order')
    
    Returns:
    --------
    Prophet
        Fitted Prophet model
    """
    # Check for enough data for yearly seasonality
    if yearly_seasonality and (data['ds'].max() - data['ds'].min()).days < 365:
        yearly_seasonality = False
    
    # Create model with specified parameters
    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality
    )
    
    # Add custom seasonalities
    if custom_seasonalities:
        for seasonality in custom_seasonalities:
            model.add_seasonality(
                name=seasonality['name'],
                period=seasonality['period'],
                fourier_order=seasonality['fourier_order']
            )
    
    # Fit model
    model.fit(data)
    
    return model

def train_vector_autoregression(data, maxlags=None, ic='aic'):
    """
    Train a Vector Autoregression (VAR) model for multiple related time series
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with multiple time series as columns
    maxlags : int, optional
        Maximum number of lags to check
    ic : str, optional
        Information criterion to use ('aic', 'bic', etc.)
    
    Returns:
    --------
    VARResults
        Fitted VAR model
    """
    # Set maxlags based on data length if not specified
    if maxlags is None:
        maxlags = min(int(len(data) * 0.3), 12)  # Set reasonable maxlags based on data length
    
    # Create and fit VAR model
    model = VAR(data)
    fitted_model = model.fit(maxlags=maxlags, ic=ic)
    
    return fitted_model

def create_ensemble_model(models_dict, weights=None):
    """
    Create an ensemble model from multiple base models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model objects with model names as keys
    weights : dict, optional
        Dictionary of weights for each model
    
    Returns:
    --------
    dict
        Ensemble model configuration
    """
    # Normalize weights if provided
    if weights:
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
    else:
        # Equal weights if not provided
        model_names = list(models_dict.keys())
        normalized_weights = {name: 1.0 / len(model_names) for name in model_names}
    
    # Create ensemble configuration
    ensemble = {
        'base_models': models_dict,
        'weights': normalized_weights,
        'type': 'weighted_average'
    }
    
    return ensemble

def auto_select_best_model(data, models_to_try=None, test_size=0.2, forecast_periods=12,
                          hyperparameter_tuning=True, complex_models_threshold=24, progress_callback=None):
    """
    Automatically select the best forecasting model based on data characteristics
    and test performance.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Time series data with date and quantity columns
    models_to_try : list, optional
        List of model types to evaluate
    test_size : float, optional
        Proportion of data to use for testing
    forecast_periods : int, optional
        Number of periods to forecast
    hyperparameter_tuning : bool, optional
        Whether to tune hyperparameters
    complex_models_threshold : int, optional
        Minimum number of observations required for complex models
    progress_callback : function, optional
        Callback function for progress reporting with parameters (current_idx, current_item, total_items, message, level)
    
    Returns:
    --------
    dict
        Dictionary with best model and evaluation metrics
    """
    # Copy data to avoid modifying original
    data = data.copy()
    
    # Default models to try if none specified
    if models_to_try is None:
        models_to_try = ["auto_arima", "prophet", "ets", "theta", "lstm", "tcn", "ensemble"]
    
    # Filter models based on data length
    if len(data) < complex_models_threshold:
        # For shorter series, use simpler models
        models_to_try = [m for m in models_to_try if m in ["auto_arima", "ets", "theta", "moving_average"]]
        
        # Add moving_average if not already in the list
        if "moving_average" not in models_to_try:
            models_to_try.append("moving_average")
    
    # Split data into train and test sets
    if len(data) <= 6:
        test_size = 1 / len(data)
    
    # Calculate split index
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    # Initialize storage for model results
    model_results = {}
    forecasts = {}
    
    # Function to evaluate a model's forecasts
    def evaluate_forecast(actual, predicted):
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # Calculate MAPE, handling zeros carefully
        if (actual == 0).any():
            # Add small constant to avoid division by zero
            epsilon = 1e-10
            mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
        else:
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape}
    
    # Train and evaluate each model
    for i, model_type in enumerate(models_to_try):
        try:
            # Report progress if callback is provided
            if progress_callback:
                progress_callback(i, model_type, len(models_to_try), 
                                f"Training and evaluating {model_type.upper()} model", "info")
                
            if model_type == "auto_arima":
                # Prepare data for ARIMA
                y_train = train_data['quantity'].values
                
                # Train auto ARIMA model
                arima_model, best_order, best_seasonal_order = train_auto_arima(
                    y_train, 
                    seasonal=(len(train_data) >= 24),
                    m=12
                )
                
                if arima_model is not None:
                    # Make forecasts on test set
                    test_forecast = arima_model.forecast(steps=len(test_data))
                    
                    # Make future forecasts
                    future_forecast = arima_model.forecast(steps=forecast_periods)
                    
                    # Store results
                    forecasts[model_type] = {
                        'test': test_forecast,
                        'future': future_forecast,
                        'params': {
                            'order': best_order,
                            'seasonal_order': best_seasonal_order
                        }
                    }
                    
                    # Calculate metrics
                    model_results[model_type] = evaluate_forecast(
                        test_data['quantity'].values, 
                        test_forecast
                    )
            
            elif model_type == "prophet":
                # Prepare data for Prophet
                prophet_train = pd.DataFrame({
                    'ds': train_data['date'],
                    'y': train_data['quantity']
                })
                
                # Train Prophet model with appropriate parameters
                if hyperparameter_tuning and len(train_data) >= 18:
                    # Multiple seasonality settings to try
                    seasonality_modes = ['additive', 'multiplicative']
                    changepoint_priors = [0.01, 0.05, 0.1]
                    
                    best_prophet_mape = float('inf')
                    best_prophet_model = None
                    best_prophet_params = None
                    
                    for mode in seasonality_modes:
                        for prior in changepoint_priors:
                            # Train model with this configuration
                            prophet_model = train_prophet_model(
                                prophet_train,
                                seasonality_mode=mode,
                                changepoint_prior_scale=prior,
                                yearly_seasonality=(len(train_data) >= 18)
                            )
                            
                            # Make predictions on test set
                            prophet_future = pd.DataFrame({'ds': test_data['date']})
                            prophet_test_forecast = prophet_model.predict(prophet_future)
                            test_predictions = prophet_test_forecast['yhat'].values
                            
                            # Calculate MAPE
                            metrics = evaluate_forecast(test_data['quantity'].values, test_predictions)
                            
                            # Check if this is the best model so far
                            if metrics['mape'] < best_prophet_mape:
                                best_prophet_mape = metrics['mape']
                                best_prophet_model = prophet_model
                                best_prophet_params = {
                                    'seasonality_mode': mode,
                                    'changepoint_prior_scale': prior
                                }
                    
                    prophet_model = best_prophet_model
                    prophet_params = best_prophet_params
                else:
                    # Use default parameters
                    prophet_model = train_prophet_model(
                        prophet_train,
                        yearly_seasonality=(len(train_data) >= 18)
                    )
                    prophet_params = {
                        'seasonality_mode': 'additive',
                        'changepoint_prior_scale': 0.05
                    }
                
                # Make test predictions
                prophet_future = pd.DataFrame({'ds': test_data['date']})
                prophet_test_forecast = prophet_model.predict(prophet_future)
                test_predictions = prophet_test_forecast['yhat'].values
                
                # Make future predictions
                future_dates = pd.date_range(
                    start=data['date'].iloc[-1] + pd.Timedelta(days=30),
                    periods=forecast_periods,
                    freq='MS'  # Month start
                )
                prophet_future = pd.DataFrame({'ds': future_dates})
                prophet_future_forecast = prophet_model.predict(prophet_future)
                future_predictions = prophet_future_forecast['yhat'].values
                
                # Store results
                forecasts[model_type] = {
                    'test': test_predictions,
                    'future': future_predictions,
                    'params': prophet_params
                }
                
                # Calculate metrics
                model_results[model_type] = evaluate_forecast(
                    test_data['quantity'].values, 
                    test_predictions
                )
            
            elif model_type == "ets":
                # Prepare data for ETS (Exponential Smoothing)
                y_train = train_data['quantity'].values
                
                # Determine appropriate parameters based on data characteristics
                if len(train_data) >= 24:
                    # Use Holt-Winters for longer series
                    seasonal_periods = 12  # Annual seasonality
                    
                    # Try both additive and multiplicative seasonality if appropriate
                    if hyperparameter_tuning:
                        # Check if multiplicative seasonality is appropriate
                        use_multiplicative = (train_data['quantity'] > 0).all() and train_data['quantity'].std() / train_data['quantity'].mean() > 0.3
                        
                        if use_multiplicative:
                            ets_model = ExponentialSmoothing(
                                y_train,
                                trend='add',
                                seasonal='mul',
                                seasonal_periods=seasonal_periods,
                                damped=True
                            ).fit(optimized=True, use_brute=False)
                        else:
                            ets_model = ExponentialSmoothing(
                                y_train,
                                trend='add',
                                seasonal='add',
                                seasonal_periods=seasonal_periods,
                                damped=True
                            ).fit(optimized=True, use_brute=False)
                    else:
                        # Use additive as default
                        ets_model = ExponentialSmoothing(
                            y_train,
                            trend='add',
                            seasonal='add',
                            seasonal_periods=seasonal_periods,
                            damped=True
                        ).fit(optimized=True, use_brute=False)
                elif len(train_data) >= 12:
                    # Use simpler Holt-Winters for medium length series
                    seasonal_periods = 4  # Quarterly-like pattern
                    
                    ets_model = ExponentialSmoothing(
                        y_train,
                        trend='add',
                        seasonal='add',
                        seasonal_periods=seasonal_periods,
                        damped=True
                    ).fit(smoothing_level=0.5, smoothing_trend=0.1, smoothing_seasonal=0.1, damping_trend=0.9, optimized=False)
                else:
                    # Use Holt's method (no seasonality) for short series
                    ets_model = ExponentialSmoothing(
                        y_train,
                        trend='add',
                        seasonal=None,
                        damped=True
                    ).fit(smoothing_level=0.5, smoothing_trend=0.1, damping_trend=0.9, optimized=False)
                
                # Make forecasts
                test_forecast = ets_model.forecast(steps=len(test_data))
                future_forecast = ets_model.forecast(steps=forecast_periods)
                
                # Store results
                forecasts[model_type] = {
                    'test': test_forecast,
                    'future': future_forecast,
                    'params': ets_model.params_formatted
                }
                
                # Calculate metrics
                model_results[model_type] = evaluate_forecast(
                    test_data['quantity'].values, 
                    test_forecast
                )
            
            elif model_type == "moving_average":
                # Simple moving average model
                window = min(3, len(train_data) // 2)
                if window < 1:
                    window = 1
                
                # Calculate moving average
                ma_value = train_data['quantity'].rolling(window=window).mean().iloc[-1]
                if pd.isna(ma_value):
                    ma_value = train_data['quantity'].mean()
                
                # Create forecasts
                test_forecast = np.array([ma_value] * len(test_data))
                future_forecast = np.array([ma_value] * forecast_periods)
                
                # Store results
                forecasts[model_type] = {
                    'test': test_forecast,
                    'future': future_forecast,
                    'params': {'window': window}
                }
                
                # Calculate metrics
                model_results[model_type] = evaluate_forecast(
                    test_data['quantity'].values, 
                    test_forecast
                )
            
            elif model_type == "lstm" and len(train_data) >= 12:
                # Prepare data for LSTM
                sequence_length = min(6, len(train_data) // 2)
                
                # Scale data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_train = scaler.fit_transform(train_data[['quantity']])
                
                # Create sequences
                X_train, y_train = [], []
                for i in range(len(scaled_train) - sequence_length):
                    X_train.append(scaled_train[i:i+sequence_length])
                    y_train.append(scaled_train[i+sequence_length])
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # Define callback for early stopping
                early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
                
                # Create and train LSTM model
                lstm_model = create_advanced_lstm_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    units_list=[32, 16],  # Smaller network for speed
                    dropout_rate=0.2
                )
                
                # Train model with appropriate batch size and epochs
                batch_size = min(8, len(X_train) // 2)
                if batch_size < 1:
                    batch_size = 1
                
                lstm_model.fit(
                    X_train, y_train,
                    epochs=50,  # Fewer epochs for speed
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Prepare last sequence for prediction
                last_sequence = scaled_train[-sequence_length:]
                
                # Make test predictions
                test_predictions = []
                current_sequence = last_sequence.copy()
                
                for _ in range(len(test_data)):
                    # Reshape for prediction
                    current_reshaped = current_sequence.reshape(1, sequence_length, 1)
                    
                    # Predict next value
                    next_value = lstm_model.predict(current_reshaped, verbose=0)[0]
                    
                    # Add to predictions
                    test_predictions.append(next_value[0])
                    
                    # Update sequence
                    current_sequence = np.vstack([current_sequence[1:], next_value])
                
                # Make future predictions
                future_predictions = []
                current_sequence = last_sequence.copy()
                
                for _ in range(forecast_periods):
                    # Reshape for prediction
                    current_reshaped = current_sequence.reshape(1, sequence_length, 1)
                    
                    # Predict next value
                    next_value = lstm_model.predict(current_reshaped, verbose=0)[0]
                    
                    # Add to predictions
                    future_predictions.append(next_value[0])
                    
                    # Update sequence
                    current_sequence = np.vstack([current_sequence[1:], next_value])
                
                # Inverse scale predictions
                test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
                
                # Store results
                forecasts[model_type] = {
                    'test': test_predictions,
                    'future': future_predictions,
                    'params': {
                        'sequence_length': sequence_length,
                        'units': [32, 16]
                    }
                }
                
                # Calculate metrics
                model_results[model_type] = evaluate_forecast(
                    test_data['quantity'].values, 
                    test_predictions
                )
            
            elif model_type == "tcn" and len(train_data) >= 12:
                # Skip if TensorFlow is not imported or data is too short
                if 'Conv1D' not in globals() or len(train_data) < 12:
                    continue
                
                # Prepare data for TCN
                sequence_length = min(6, len(train_data) // 2)
                
                # Scale data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_train = scaler.fit_transform(train_data[['quantity']])
                
                # Create sequences
                X_train, y_train = [], []
                for i in range(len(scaled_train) - sequence_length):
                    X_train.append(scaled_train[i:i+sequence_length])
                    y_train.append(scaled_train[i+sequence_length])
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # Define callback for early stopping
                early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
                
                # Create TCN model
                tcn_model = Sequential()
                
                # Add convolutional layers
                tcn_model.add(Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu', input_shape=(sequence_length, 1)))
                tcn_model.add(BatchNormalization())
                tcn_model.add(Dropout(0.2))
                
                tcn_model.add(Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu', dilation_rate=2))
                tcn_model.add(BatchNormalization())
                tcn_model.add(Dropout(0.2))
                
                # Add flatten layer to convert features to vector
                tcn_model.add(Flatten())
                
                # Add output layer
                tcn_model.add(Dense(1))
                
                # Compile model
                tcn_model.compile(optimizer='adam', loss='mse')
                
                # Train model
                batch_size = min(8, len(X_train) // 2)
                if batch_size < 1:
                    batch_size = 1
                
                tcn_model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Prepare last sequence for prediction
                last_sequence = scaled_train[-sequence_length:]
                
                # Make test predictions
                test_predictions = []
                current_sequence = last_sequence.copy()
                
                for _ in range(len(test_data)):
                    # Reshape for prediction
                    current_reshaped = current_sequence.reshape(1, sequence_length, 1)
                    
                    # Predict next value
                    next_value = tcn_model.predict(current_reshaped, verbose=0)[0]
                    
                    # Add to predictions
                    test_predictions.append(next_value[0])
                    
                    # Update sequence
                    current_sequence = np.vstack([current_sequence[1:], next_value])
                
                # Make future predictions
                future_predictions = []
                current_sequence = last_sequence.copy()
                
                for _ in range(forecast_periods):
                    # Reshape for prediction
                    current_reshaped = current_sequence.reshape(1, sequence_length, 1)
                    
                    # Predict next value
                    next_value = tcn_model.predict(current_reshaped, verbose=0)[0]
                    
                    # Add to predictions
                    future_predictions.append(next_value[0])
                    
                    # Update sequence
                    current_sequence = np.vstack([current_sequence[1:], next_value])
                
                # Inverse scale predictions
                test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
                
                # Store results
                forecasts[model_type] = {
                    'test': test_predictions,
                    'future': future_predictions,
                    'params': {
                        'sequence_length': sequence_length,
                        'filters': 32
                    }
                }
                
                # Calculate metrics
                model_results[model_type] = evaluate_forecast(
                    test_data['quantity'].values, 
                    test_predictions
                )
            
            elif model_type == "ensemble" and len(forecasts) >= 2:
                # Create ensemble only if we have at least 2 other models
                ensemble_test_predictions = np.zeros(len(test_data))
                ensemble_future_predictions = np.zeros(forecast_periods)
                
                # Get weights based on inverse MAPE
                weights = {}
                total_weight = 0
                
                for model, metrics in model_results.items():
                    if model != "ensemble" and 'mape' in metrics:
                        weight = 1.0 / (metrics['mape'] + 1e-10)  # Add small constant to avoid division by zero
                        weights[model] = weight
                        total_weight += weight
                
                # Normalize weights
                for model in weights:
                    weights[model] /= total_weight
                
                # Create weighted predictions
                for model, weight in weights.items():
                    ensemble_test_predictions += forecasts[model]['test'] * weight
                    ensemble_future_predictions += forecasts[model]['future'] * weight
                
                # Store results
                forecasts[model_type] = {
                    'test': ensemble_test_predictions,
                    'future': ensemble_future_predictions,
                    'params': {'weights': weights}
                }
                
                # Calculate metrics
                model_results[model_type] = evaluate_forecast(
                    test_data['quantity'].values, 
                    ensemble_test_predictions
                )
            
            elif model_type == "theta" and len(train_data) >= 4:
                # Simple implementation of Theta method
                # This is a simplified version focusing on decomposition and trend
                
                # Get the time series
                y_train = train_data['quantity'].values
                
                # Calculate the trend using linear regression
                x = np.arange(len(y_train))
                trend_coef = np.polyfit(x, y_train, 1)
                trend = trend_coef[1] + trend_coef[0] * x
                
                # Calculate the long-term drift (using the linear trend)
                drift = trend_coef[0]
                
                # Create simple exponential smoothing model for the detrended series
                detrended = y_train - trend
                ses_model = SimpleExpSmoothing(detrended).fit(smoothing_level=0.5)
                
                # Make test forecasts
                test_forecast = []
                for i in range(len(test_data)):
                    # Forecast using trend and SES
                    forecast_idx = len(y_train) + i
                    ses_forecast = ses_model.forecast(1).iloc[0]
                    trend_forecast = trend_coef[1] + trend_coef[0] * forecast_idx
                    
                    # Combine the forecasts
                    test_forecast.append(trend_forecast + ses_forecast)
                
                # Make future forecasts
                future_forecast = []
                for i in range(forecast_periods):
                    # Forecast using trend and SES
                    forecast_idx = len(y_train) + len(test_data) + i
                    ses_forecast = ses_model.forecast(1).iloc[0]
                    trend_forecast = trend_coef[1] + trend_coef[0] * forecast_idx
                    
                    # Combine the forecasts
                    future_forecast.append(trend_forecast + ses_forecast)
                
                # Store results
                forecasts[model_type] = {
                    'test': np.array(test_forecast),
                    'future': np.array(future_forecast),
                    'params': {'drift': drift}
                }
                
                # Calculate metrics
                model_results[model_type] = evaluate_forecast(
                    test_data['quantity'].values, 
                    test_forecast
                )
        
        except Exception as e:
            print(f"Error training {model_type} model: {str(e)}")
            continue
    
    # Select the best model based on MAPE (or other criteria)
    best_model = None
    best_mape = float('inf')
    
    for model, metrics in model_results.items():
        if 'mape' in metrics and metrics['mape'] < best_mape:
            best_mape = metrics['mape']
            best_model = model
    
    # If no models were successfully trained, use a simple fallback
    if best_model is None:
        # Create a simple moving average model
        window = min(3, len(train_data) // 2)
        if window < 1:
            window = 1
        
        ma_value = train_data['quantity'].rolling(window=window).mean().iloc[-1]
        if pd.isna(ma_value):
            ma_value = train_data['quantity'].mean()
        
        test_forecast = np.array([ma_value] * len(test_data))
        future_forecast = np.array([ma_value] * forecast_periods)
        
        best_model = "moving_average"
        forecasts[best_model] = {
            'test': test_forecast,
            'future': future_forecast,
            'params': {'window': window}
        }
        
        model_results[best_model] = evaluate_forecast(
            test_data['quantity'].values, 
            test_forecast
        )
    
    # Return the best model and all model results
    return {
        'best_model': best_model,
        'metrics': model_results,
        'forecasts': forecasts,
        'test_data': test_data,
        'train_data': train_data
    }

# ================================
# 3. ADVANCED FORECASTING WORKFLOW
# ================================

def human_sense_check(forecast_values, historical_data, sense_check_rules=None):
    """
    Perform a "human-like" sense check on forecast values based on historical patterns
    
    Parameters:
    -----------
    forecast_values : array-like
        Array of forecast values to check
    historical_data : pandas.Series
        Historical data series
    sense_check_rules : dict, optional
        Dictionary of custom rules for sense checking
        
    Returns:
    --------
    tuple
        (adjusted_forecast, issues_detected, adjustments_made)
    """
    # Convert inputs to numpy arrays
    forecast = np.array(forecast_values)
    history = np.array(historical_data)
    
    # Initialize return values
    adjusted_forecast = forecast.copy()
    issues_detected = []
    adjustments_made = []
    
    # Default rules if none provided
    if sense_check_rules is None:
        sense_check_rules = {
            'max_growth_rate': 2.0,  # Maximum 200% growth from historical max
            'max_decline_rate': 0.5,  # Maximum 50% decline from historical min
            'min_value': 0,           # Forecasts shouldn't be negative
            'volatility_factor': 2.0, # Maximum 2x historical volatility
            'seasonality_check': True  # Check if seasonal patterns are preserved
        }
    
    # Calculate historical statistics
    hist_max = np.max(history)
    hist_min = np.max([0.1, np.min(history)])  # Avoid issues with zeros
    hist_mean = np.mean(history)
    hist_std = np.std(history)
    
    # Calculate historical volatility (coefficient of variation)
    hist_cv = hist_std / hist_mean if hist_mean > 0 else 0
    
    # Check 1: Extreme growth or decline
    max_allowed = hist_max * sense_check_rules['max_growth_rate']
    min_allowed = hist_min * sense_check_rules['max_decline_rate']
    
    # Apply min/max constraints
    too_high = forecast > max_allowed
    too_low = forecast < min_allowed
    
    if np.any(too_high):
        issues_detected.append(f"Forecast exceeds maximum growth threshold ({sense_check_rules['max_growth_rate']}x historical max)")
        adjusted_forecast[too_high] = max_allowed
        adjustments_made.append(f"Capped forecasts to {sense_check_rules['max_growth_rate']}x historical maximum")
    
    if np.any(too_low):
        issues_detected.append(f"Forecast below minimum decline threshold ({sense_check_rules['max_decline_rate']}x historical min)")
        adjusted_forecast[too_low] = min_allowed
        adjustments_made.append(f"Raised forecasts to {sense_check_rules['max_decline_rate']}x historical minimum")
    
    # Check 2: Negative values
    if np.any(adjusted_forecast < sense_check_rules['min_value']):
        issues_detected.append("Negative forecast values detected")
        adjusted_forecast = np.maximum(adjusted_forecast, sense_check_rules['min_value'])
        adjustments_made.append(f"Adjusted negative values to {sense_check_rules['min_value']}")
    
    # Check 3: Excessive volatility
    forecast_cv = np.std(adjusted_forecast) / np.mean(adjusted_forecast) if np.mean(adjusted_forecast) > 0 else 0
    max_allowed_cv = hist_cv * sense_check_rules['volatility_factor']
    
    if forecast_cv > max_allowed_cv and len(adjusted_forecast) > 3:
        issues_detected.append(f"Forecast volatility ({forecast_cv:.2f}) exceeds historical pattern ({hist_cv:.2f})")
        
        # Smooth the forecast to reduce volatility
        smoothed = pd.Series(adjusted_forecast).rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        # Preserve the overall level by adjusting the smoothed forecast
        level_factor = np.sum(adjusted_forecast) / np.sum(smoothed)
        adjusted_forecast = smoothed * level_factor
        
        adjustments_made.append("Smoothed forecast to match historical volatility patterns")
    
    # Check 4: Seasonality preservation (if applicable and we have enough history)
    if sense_check_rules['seasonality_check'] and len(history) >= 24 and len(adjusted_forecast) >= 12:
        # Extract month-of-year seasonality from history
        try:
            # Simple approach: calculate average for each month position
            monthly_factors = []
            for i in range(12):
                # Get values at positions i, i+12, i+24, etc.
                month_values = history[i::12]
                if len(month_values) > 0:
                    # Calculate this month's average relative to overall average
                    monthly_factors.append(np.mean(month_values) / hist_mean)
                else:
                    monthly_factors.append(1.0)
            
            # Check if forecast follows the seasonal pattern
            for i in range(len(adjusted_forecast)):
                month_pos = i % 12
                expected_factor = monthly_factors[month_pos]
                
                # If forecast doesn't follow seasonal pattern, adjust it
                current_factor = adjusted_forecast[i] / np.mean(adjusted_forecast)
                
                # If the seasonal factor difference is significant
                if abs(current_factor - expected_factor) > 0.3 and expected_factor > 0:
                    adjusted_forecast[i] = np.mean(adjusted_forecast) * expected_factor
            
            # Only report adjustment if significant changes were made
            if not np.allclose(adjusted_forecast, forecast, rtol=0.1):
                issues_detected.append("Forecast doesn't respect historical seasonal patterns")
                adjustments_made.append("Adjusted forecast to preserve seasonal patterns")
        except:
            # Skip seasonality check if it fails
            pass
    
    return adjusted_forecast, issues_detected, adjustments_made

def advanced_generate_forecasts(sales_data, cluster_info=None, forecast_periods=12, 
                              auto_select=True, models_to_evaluate=None, selected_skus=None, 
                              progress_callback=None, hyperparameter_tuning=True,
                              apply_sense_check=True):
    """
    Generate advanced forecasts for SKUs, leveraging improved preprocessing, model selection,
    and post-processing.
    
    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Sales data with date, sku, and quantity columns
    cluster_info : pandas.DataFrame, optional
        Information about SKU clusters
    forecast_periods : int, optional
        Number of periods to forecast (default is 12)
    auto_select : bool, optional
        Whether to automatically select the best model for each SKU
    models_to_evaluate : list, optional
        List of model types to evaluate
    selected_skus : list, optional
        List of specific SKUs to forecast. If None, forecasts all SKUs
    progress_callback : function, optional
        Callback function to report progress (current_index, current_sku, total_skus)
    hyperparameter_tuning : bool, optional
        Whether to tune hyperparameters for models
    apply_sense_check : bool, optional
        Whether to apply human-like sense checking to forecasts
    
    Returns:
    --------
    dict
        Dictionary with forecast results for each SKU
    """
    # Make a copy of the data to avoid modifying the original
    data = sales_data.copy()
    
    # Ensure data is sorted by date
    data = data.sort_values(by=['sku', 'date'])
    
    # Get list of SKUs to forecast
    if selected_skus is not None and len(selected_skus) > 0:
        sku_list = [sku for sku in selected_skus if sku in data['sku'].unique()]
    else:
        sku_list = data['sku'].unique().tolist()
    
    # Check if cluster info is available or needs to be generated
    if cluster_info is None:
        try:
            # Extract features for clustering
            features_df = extract_features(data)
            # Perform clustering
            cluster_info = cluster_skus(features_df)
        except Exception as e:
            print(f"Error generating clusters: {str(e)}")
            # Create dummy cluster info if clustering fails
            dummy_clusters = pd.DataFrame({
                'sku': sku_list,
                'cluster': 0,
                'cluster_name': 'Default Cluster'
            })
            cluster_info = dummy_clusters
    
    # Default models to evaluate if none specified
    if models_to_evaluate is None:
        models_to_evaluate = ["auto_arima", "prophet", "ets", "theta", "lstm", "ensemble"]
    
    # Results dictionary
    forecasts = {}
    
    # Total number of SKUs for progress reporting
    total_skus = len(sku_list)
    
    # Process each SKU
    for idx, sku in enumerate(sku_list):
        # Report progress if callback is provided
        if progress_callback:
            progress_callback(idx, sku, total_skus, f"Starting forecast generation for SKU: {sku}", "info")
        
        try:
            # Filter data for this SKU
            sku_data = data[data['sku'] == sku].copy()
            
            # Skip if not enough data
            if len(sku_data) < 3:
                if progress_callback:
                    progress_callback(idx, sku, total_skus, f"Skipping {sku}: insufficient data (less than 3 points)", "warning")
                print(f"Skipping {sku}: insufficient data (less than 3 points)")
                continue
            
            # Get cluster information for this SKU
            if cluster_info is not None and sku in cluster_info['sku'].values:
                sku_cluster = cluster_info.loc[cluster_info['sku'] == sku, 'cluster'].iloc[0]
                cluster_name = cluster_info.loc[cluster_info['sku'] == sku, 'cluster_name'].iloc[0]
            else:
                sku_cluster = 0
                cluster_name = "Default Cluster"
            
            # 1. Advanced Preprocessing
            # ------------------------
            
            if progress_callback:
                progress_callback(idx, sku, total_skus, "Cleaning time series data and detecting outliers", "info")
                
            # Clean the time series (handle outliers)
            sku_data['quantity_cleaned'] = clean_time_series(sku_data['quantity'])
            
            # Detect significant change points
            segment_boundaries = segment_time_series(sku_data, column='quantity_cleaned')
            
            # Focus on the most recent segment for forecasting
            if len(segment_boundaries) > 1 and segment_boundaries[-1] - segment_boundaries[-2] >= 6:
                # If the last segment has at least 6 points, use only that segment
                start_idx = segment_boundaries[-2]
                sku_data_recent = sku_data.iloc[start_idx:].copy()
            else:
                # Otherwise use full dataset
                sku_data_recent = sku_data.copy()
            
            # 2. Auto Model Selection and Forecasting
            # ---------------------------------
            
            if auto_select:
                # Log model selection process
                if progress_callback:
                    progress_callback(idx, sku, total_skus, "Starting automated model selection process", "info")
                    if hyperparameter_tuning:
                        progress_callback(idx, sku, total_skus, "Hyperparameter tuning enabled - may take longer but produce better results", "info")
                    progress_callback(idx, sku, total_skus, f"Testing models: {', '.join(models_to_evaluate)}", "info")
                
                # Automatically select and train the best model
                model_results = auto_select_best_model(
                    sku_data_recent, 
                    models_to_try=models_to_evaluate,
                    test_size=0.2,
                    forecast_periods=forecast_periods,
                    hyperparameter_tuning=hyperparameter_tuning
                )
                
                best_model_type = model_results['best_model']
                best_forecast = model_results['forecasts'][best_model_type]['future']
                
                # Log best model selection
                if progress_callback:
                    progress_callback(idx, sku, total_skus, f"Selected {best_model_type.upper()} as best model for this SKU", "success")
                    
                    # If metrics are available, log them
                    if 'metrics' in model_results and best_model_type in model_results['metrics']:
                        metrics = model_results['metrics'][best_model_type]
                        if 'mape' in metrics and not np.isnan(metrics['mape']):
                            progress_callback(idx, sku, total_skus, 
                                            f"Model performance - MAPE: {metrics['mape']:.2f}%, RMSE: {metrics.get('rmse', 'N/A')}", "info")
                
                # Get model evaluation metrics
                evaluation_metrics = model_results['metrics']
                
                # Get all models' forecasts
                all_models_forecasts = {}
                for model_name, forecast_data in model_results['forecasts'].items():
                    # Create future dates for the forecast
                    last_date = sku_data['date'].max()
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=30),
                        periods=forecast_periods,
                        freq='MS'  # Month start frequency
                    )
                    
                    # Store forecast as Series with dates
                    all_models_forecasts[model_name] = pd.Series(
                        forecast_data['future'],
                        index=future_dates
                    )
            else:
                # Use the best model from cluster analysis (to be implemented)
                # For now, just use auto_arima as fallback
                # This branch is mostly provided for compatibility with existing code
                
                # Prepare data for auto_arima
                y_train = sku_data_recent['quantity_cleaned'].values
                
                # Train auto ARIMA model
                arima_model, best_order, best_seasonal_order = train_auto_arima(
                    y_train, 
                    seasonal=(len(y_train) >= 24),
                    m=12
                )
                
                if arima_model is not None:
                    # Make future forecasts
                    best_forecast = arima_model.forecast(steps=forecast_periods)
                    best_model_type = "auto_arima"
                else:
                    # Fallback to simple moving average
                    window = min(3, len(sku_data_recent) // 2)
                    if window < 1:
                        window = 1
                    
                    ma_value = sku_data_recent['quantity_cleaned'].rolling(window=window).mean().iloc[-1]
                    if pd.isna(ma_value):
                        ma_value = sku_data_recent['quantity_cleaned'].mean()
                    
                    best_forecast = np.array([ma_value] * forecast_periods)
                    best_model_type = "moving_average"
                
                # Create dummy evaluation metrics
                evaluation_metrics = {
                    best_model_type: {
                        'mape': np.nan,
                        'rmse': np.nan,
                        'mae': np.nan
                    }
                }
                
                # Create dummy all_models_forecasts
                last_date = sku_data['date'].max()
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=30),
                    periods=forecast_periods,
                    freq='MS'  # Month start frequency
                )
                
                all_models_forecasts = {
                    best_model_type: pd.Series(best_forecast, index=future_dates)
                }
            
            # 3. Apply Human-Like Sense Check
            # ---------------------------
            
            if apply_sense_check:
                if progress_callback:
                    progress_callback(idx, sku, total_skus, "Performing human-like sense check on forecast values", "info")
                
                adjusted_forecast, issues, adjustments = human_sense_check(
                    best_forecast,
                    sku_data['quantity'].values
                )
                
                # If significant adjustments were made, use the adjusted forecast
                if len(adjustments) > 0:
                    if progress_callback:
                        progress_callback(idx, sku, total_skus, f"Sense check detected {len(issues)} issues and made {len(adjustments)} adjustments", "warning")
                        for i, issue in enumerate(issues[:3]):  # Show first 3 issues at most
                            progress_callback(idx, sku, total_skus, f"Issue {i+1}: {issue}", "info")
                    
                    best_forecast = adjusted_forecast
                    
                    # Update the forecast in all_models_forecasts
                    all_models_forecasts[best_model_type] = pd.Series(
                        best_forecast,
                        index=all_models_forecasts[best_model_type].index
                    )
                    
                    # Add sense check information
                    sense_check_info = {
                        'issues_detected': issues,
                        'adjustments_made': adjustments
                    }
                else:
                    if progress_callback:
                        progress_callback(idx, sku, total_skus, "Sense check passed - no adjustments needed", "success")
                    sense_check_info = {'issues_detected': [], 'adjustments_made': []}
            else:
                sense_check_info = {'issues_detected': [], 'adjustments_made': []}
            
            # Create future dates for the forecast
            last_date = sku_data['date'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=30),
                periods=forecast_periods,
                freq='MS'  # Month start frequency
            )
            
            # Log forecast completion status
            if progress_callback:
                progress_callback(idx, sku, total_skus, "Finalizing forecast and generating confidence intervals", "info")
                
            # Store the forecasts
            forecasts[sku] = {
                'sku': sku,
                'model': best_model_type,
                'cluster': sku_cluster,
                'cluster_name': cluster_name,
                'forecast': pd.Series(best_forecast, index=future_dates),
                'lower_bound': pd.Series(best_forecast * 0.8, index=future_dates),  # Simple bounds for now
                'upper_bound': pd.Series(best_forecast * 1.2, index=future_dates),
                'model_evaluation': {
                    'metrics': evaluation_metrics,
                    'all_models_forecasts': all_models_forecasts
                },
                'sense_check': sense_check_info,
                'train_set': sku_data_recent
            }
            
            # Log forecast completion status
            if progress_callback:
                # Get forecast average and range for summary
                avg_forecast = np.mean(best_forecast)
                min_forecast = np.min(best_forecast)
                max_forecast = np.max(best_forecast)
                
                progress_callback(idx, sku, total_skus, 
                                f"Forecast complete: Avg={avg_forecast:.1f}, Range=[{min_forecast:.1f}, {max_forecast:.1f}]", 
                                "success")
            
            # Optionally save forecast to database
            try:
                # Prepare forecast data for saving
                forecast_json = json.dumps({
                    'dates': [d.strftime('%Y-%m-%d') for d in future_dates.tolist()],
                    'values': best_forecast.tolist()
                })
                
                # Get metrics for best model
                metrics = evaluation_metrics.get(best_model_type, {})
                mape = metrics.get('mape', np.nan)
                rmse = metrics.get('rmse', np.nan)
                mae = metrics.get('mae', np.nan)
                
                # Save to database
                save_forecast_result(
                    sku=sku,
                    model_type=best_model_type,
                    forecast_periods=forecast_periods,
                    mape=mape,
                    rmse=rmse,
                    mae=mae,
                    forecast_data=forecast_json,
                    model_params=json.dumps({'cluster': int(sku_cluster)})
                )
            except Exception as e:
                error_msg = f"Error saving forecast to database: {str(e)}"
                print(error_msg)
                if progress_callback:
                    progress_callback(idx, sku, total_skus, error_msg, "error")
                # Continue with next SKU even if database save fails
        
        except Exception as e:
            error_msg = f"Error generating forecast for SKU {sku}: {str(e)}"
            print(error_msg)
            if progress_callback:
                progress_callback(idx, sku, total_skus, error_msg, "error")
                # Add more detailed error information if available
                import traceback
                trace_msg = traceback.format_exc()
                progress_callback(idx, sku, total_skus, "See details in console output for full traceback", "error")
            # Continue with next SKU
    
    return forecasts