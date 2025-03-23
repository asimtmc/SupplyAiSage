import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import math

# Import the database functionality
from utils.database import save_forecast_result, get_forecast_history

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def extract_features(sales_data):
    """
    Extract time series features from sales data for clustering

    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Sales data with date, sku, and quantity columns

    Returns:
    --------
    pandas.DataFrame
        DataFrame with extracted features per SKU
    """
    # Ensure data is sorted by date
    sales_data = sales_data.sort_values(by=['sku', 'date'])

    # Aggregate sales data by SKU and month to reduce noise
    monthly_data = sales_data.set_index('date')
    monthly_data = monthly_data.groupby(['sku', pd.Grouper(freq='M')])['quantity'].sum().reset_index()

    # List to store features for each SKU
    features_list = []

    # Get unique SKUs
    skus = monthly_data['sku'].unique()

    for sku in skus:
        # Filter data for this SKU
        sku_data = monthly_data[monthly_data['sku'] == sku].sort_values('date')

        if len(sku_data) < 4:  # Need at least a few months of data
            continue

        # Calculate basic statistics
        mean_sales = sku_data['quantity'].mean()
        std_sales = sku_data['quantity'].std()
        cv = std_sales / mean_sales if mean_sales > 0 else 0

        # Calculate trend
        try:
            sku_data['trend_index'] = np.arange(len(sku_data))
            trend_model = np.polyfit(sku_data['trend_index'], sku_data['quantity'], 1)
            trend_slope = trend_model[0]
        except:
            trend_slope = 0

        # Check for seasonality using autocorrelation
        if len(sku_data) >= 13:  # Need at least a year of data for seasonality
            try:
                autocorr = pd.Series(sku_data['quantity'].values).autocorr(lag=12)  # 12-month lag
            except:
                autocorr = 0
        else:
            autocorr = 0

        # Calculate skewness and kurtosis
        skewness = sku_data['quantity'].skew()
        kurtosis = sku_data['quantity'].kurtosis()

        # Calculate volatility
        pct_change = sku_data['quantity'].pct_change().dropna()
        volatility = pct_change.std() if len(pct_change) > 0 else 0

        # Calculate zero ratio (proportion of months with zero sales)
        zero_ratio = (sku_data['quantity'] == 0).mean()

        # Store features
        features_list.append({
            'sku': sku,
            'mean_sales': mean_sales,
            'cv': cv,
            'trend_slope': trend_slope,
            'seasonality': autocorr,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'volatility': volatility,
            'zero_ratio': zero_ratio
        })

    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)

    return features_df

def cluster_skus(features_df, n_clusters=5):
    """
    Cluster SKUs based on their time series features

    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame with extracted features per SKU
    n_clusters : int, optional
        Number of clusters to create (default is 5)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with cluster assignments per SKU
    """
    # Select features for clustering
    cluster_features = [
        'mean_sales', 'cv', 'trend_slope', 'seasonality', 
        'skewness', 'volatility', 'zero_ratio'
    ]

    # Handle missing values
    for feature in cluster_features:
        if feature in features_df.columns:
            features_df[feature] = features_df[feature].fillna(features_df[feature].median())

    # Subset dataframe to only include features used for clustering
    features_subset = features_df[cluster_features]

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_subset)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)

    # Add cluster labels to original features
    features_df['cluster'] = cluster_labels

    # Determine cluster characteristics
    cluster_profiles = features_df.groupby('cluster')[cluster_features].mean()

    # Assign descriptive names based on characteristics
    cluster_names = {}

    for cluster in range(n_clusters):
        profile = cluster_profiles.loc[cluster]

        # Determine key characteristics of this cluster
        if profile['seasonality'] > 0.3:
            seasonality = "Seasonal"
        else:
            seasonality = "Non-seasonal"

        if profile['trend_slope'] > 0.1:
            trend = "Growing"
        elif profile['trend_slope'] < -0.1:
            trend = "Declining"
        else:
            trend = "Stable"

        if profile['volatility'] > 0.3:
            volatility = "Volatile"
        else:
            volatility = "Steady"

        if profile['zero_ratio'] > 0.3:
            frequency = "Intermittent"
        else:
            frequency = "Regular"

        # Create descriptive name
        name = f"{seasonality} {trend} {volatility} {frequency}"
        cluster_names[cluster] = name

    # Map names to cluster numbers
    features_df['cluster_name'] = features_df['cluster'].map(cluster_names)

    return features_df

def select_best_model(sku_data, forecast_periods=12):
    """
    Select the best forecasting model for a given SKU based on its characteristics

    Parameters:
    -----------
    sku_data : pandas.DataFrame
        Time series data for a specific SKU with date and quantity columns
    forecast_periods : int, optional
        Number of periods to forecast (default is 12)

    Returns:
    --------
    dict
        Dictionary containing model type, parameters, and forecast results
    """
    # Make a copy of the data to avoid modifying the original
    data = sku_data.copy()

    # Ensure data is sorted by date
    data = data.sort_values('date')

    # Calculate features to determine model choice
    if len(data) < 6:
        return {"model": "moving_average", "forecast": pd.Series(), "lower_bound": pd.Series(), "upper_bound": pd.Series()}

    # Check for seasonality using autocorrelation
    seasonality = False
    if len(data) >= 13:
        try:
            autocorr = pd.Series(data['quantity'].values).autocorr(lag=12)
            if autocorr > 0.3:
                seasonality = True
        except:
            pass

    # Check for zeros and intermittent demand
    zero_ratio = (data['quantity'] == 0).mean()
    intermittent = zero_ratio > 0.3

    # Calculate CV to determine if Holt-Winters is appropriate
    cv = data['quantity'].std() / data['quantity'].mean() if data['quantity'].mean() > 0 else float('inf')

    # Check for trend
    try:
        data['trend_index'] = np.arange(len(data))
        trend_model = np.polyfit(data['trend_index'], data['quantity'], 1)
        trend_slope = trend_model[0]
        has_trend = abs(trend_slope) > 0.1 * data['quantity'].mean()
    except:
        has_trend = False

    # Set the model parameters
    if len(data) < 12:
        # For very short series, use simple methods
        model_type = "moving_average"
        window = min(3, len(data) - 1)
        if window < 1:
            window = 1
        forecast = data['quantity'].rolling(window=window).mean().iloc[-1]
        forecast_values = [forecast] * forecast_periods
        lower_bound = [max(0, forecast * 0.7)] * forecast_periods
        upper_bound = [forecast * 1.3] * forecast_periods

    elif intermittent:
        # For intermittent demand, use simple methods
        model_type = "croston"
        # Simple implementation of Croston-like logic
        non_zero_values = data['quantity'][data['quantity'] > 0]
        if len(non_zero_values) > 0:
            avg_non_zero = non_zero_values.mean()
            non_zero_prob = len(non_zero_values) / len(data)
            forecast = avg_non_zero * non_zero_prob
        else:
            forecast = data['quantity'].mean()

        forecast_values = [forecast] * forecast_periods
        lower_bound = [max(0, forecast * 0.6)] * forecast_periods
        upper_bound = [forecast * 1.4] * forecast_periods

    elif seasonality:
        # For seasonal data, try Holt-Winters first
        model_type = "holtwinters"
        try:
            # Determine seasonal period based on data length
            if len(data) >= 24:
                seasonal_periods = 12  # Monthly data, annual seasonality
            else:
                seasonal_periods = 4   # Quarterly-like pattern for shorter data

            # Fit Holt-Winters model
            model = ExponentialSmoothing(
                data['quantity'],
                trend='add',               # Additive trend
                seasonal='add',            # Additive seasonality
                seasonal_periods=seasonal_periods,
                damped=True                # Damped trend to avoid over-forecasting
            )
            results = model.fit(optimized=True, use_brute=False)

            # Generate forecasts
            forecast_values = results.forecast(steps=forecast_periods).values

            # Create confidence intervals (80% by default)
            forecast_std = np.std(data['quantity'])
            z_value = 1.28  # Approximately 80% confidence interval
            lower_bound = forecast_values - z_value * forecast_std
            upper_bound = forecast_values + z_value * forecast_std
        except:
            # Fall back to SARIMA if Holt-Winters fails
            model_type = "sarima"
            try:
                # Simple SARIMA model
                model = SARIMAX(
                    data['quantity'],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
                forecast_obj = results.get_forecast(steps=forecast_periods)
                forecast_values = forecast_obj.predicted_mean.values
                conf_int = forecast_obj.conf_int(alpha=0.1)
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
            except:
                # Fall back to ARIMA if SARIMA fails
                model_type = "arima"
                try:
                    model = ARIMA(data['quantity'], order=(1, 1, 1))
                    results = model.fit()
                    forecast_obj = results.get_forecast(steps=forecast_periods)
                    forecast_values = forecast_obj.predicted_mean.values
                    conf_int = forecast_obj.conf_int(alpha=0.1)
                    lower_bound = conf_int.iloc[:, 0].values
                    upper_bound = conf_int.iloc[:, 1].values
                except:
                    # Fall back to moving average if ARIMA fails
                    model_type = "moving_average"
                    window = min(6, len(data) - 1)
                    forecast = data['quantity'].rolling(window=window).mean().iloc[-1]
                    forecast_values = [forecast] * forecast_periods
                    lower_bound = [max(0, forecast * 0.7)] * forecast_periods
                    upper_bound = [forecast * 1.3] * forecast_periods
    else:
        # For non-seasonal data, try ARIMA or Prophet
        if len(data) >= 24:
            model_type = "prophet"
            try:
                # Prepare data for Prophet
                prophet_data = pd.DataFrame({
                    'ds': data['date'],
                    'y': data['quantity']
                })

                # Initialize and fit Prophet model
                m = Prophet(
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=0.1,
                    yearly_seasonality=True if len(data) >= 24 else False,
                    weekly_seasonality=False,
                    daily_seasonality=False
                )
                m.fit(prophet_data)

                # Create future dataframe and make predictions
                last_date = data['date'].max()
                future_dates = [last_date + timedelta(days=30*i) for i in range(1, forecast_periods+1)]
                future = pd.DataFrame({'ds': future_dates})

                forecast_df = m.predict(future)
                forecast_values = forecast_df['yhat'].values
                lower_bound = forecast_df['yhat_lower'].values
                upper_bound = forecast_df['yhat_upper'].values
            except:
                # Fall back to ARIMA if Prophet fails
                model_type = "arima"
                try:
                    model = ARIMA(data['quantity'], order=(1, 1, 1))
                    results = model.fit()
                    forecast_obj = results.get_forecast(steps=forecast_periods)
                    forecast_values = forecast_obj.predicted_mean.values
                    conf_int = forecast_obj.conf_int(alpha=0.1)
                    lower_bound = conf_int.iloc[:, 0].values
                    upper_bound = conf_int.iloc[:, 1].values
                except:
                    # Fall back to moving average if ARIMA fails
                    model_type = "moving_average"
                    window = min(6, len(data) - 1)
                    forecast = data['quantity'].rolling(window=window).mean().iloc[-1]
                    forecast_values = [forecast] * forecast_periods
                    lower_bound = [max(0, forecast * 0.7)] * forecast_periods
                    upper_bound = [forecast * 1.3] * forecast_periods
        else:
            # For medium-length non-seasonal series, use ARIMA
            model_type = "arima"
            try:
                model = ARIMA(data['quantity'], order=(1, 1, 1))
                results = model.fit()
                forecast_obj = results.get_forecast(steps=forecast_periods)
                forecast_values = forecast_obj.predicted_mean.values
                conf_int = forecast_obj.conf_int(alpha=0.1)
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
            except:
                # Fall back to moving average if ARIMA fails
                model_type = "moving_average"
                window = min(6, len(data) - 1)
                forecast = data['quantity'].rolling(window=window).mean().iloc[-1]
                forecast_values = [forecast] * forecast_periods
                lower_bound = [max(0, forecast * 0.7)] * forecast_periods
                upper_bound = [forecast * 1.3] * forecast_periods

    # Ensure all forecasts are non-negative
    forecast_values = np.maximum(0, forecast_values)
    lower_bound = np.maximum(0, lower_bound)

    # Create output dict
    last_date = data['date'].max()
    forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, forecast_periods+1)]

    result = {
        "model": model_type,
        "forecast": pd.Series(forecast_values, index=forecast_dates),
        "lower_bound": pd.Series(lower_bound, index=forecast_dates),
        "upper_bound": pd.Series(upper_bound, index=forecast_dates)
    }

    return result

def create_lstm_model(sequence_length):
    """
    Create and compile an LSTM model for time series forecasting

    Parameters:
    -----------
    sequence_length : int
        Length of input sequences

    Returns:
    --------
    tensorflow.keras.models.Sequential
        Compiled LSTM model
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Create model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def prepare_lstm_data(data, sequence_length=12):
    """
    Prepare data for LSTM model by creating sequences

    Parameters:
    -----------
    data : numpy.ndarray
        Time series data
    sequence_length : int, optional
        Length of input sequences (default is 12)

    Returns:
    --------
    tuple
        (X_train, y_train) where X_train contains sequences and y_train contains next values
    """
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])

    return np.array(X), np.array(y)

def train_lstm_model(data, test_size=0.2, sequence_length=12, epochs=50):
    """
    Train an LSTM model on time series data

    Parameters:
    -----------
    data : numpy.ndarray
        Time series data
    test_size : float, optional
        Proportion of data to use for testing (default is 0.2)
    sequence_length : int, optional
        Length of input sequences (default is 12)
    epochs : int, optional
        Number of training epochs (default is 50)

    Returns:
    --------
    tuple
        (model, scaler, test_mape, test_rmse) where model is the trained LSTM model
    """
    # Check if data is sufficient
    if len(data) <= sequence_length:
        raise ValueError(f"Data length ({len(data)}) must be greater than sequence_length ({sequence_length})")

    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Prepare data
    X, y = prepare_lstm_data(data_scaled, sequence_length)

    # If no test data requested, use all data for training
    if test_size == 0 or len(X) < 3:  # Ensure we have enough data to split
        X_train, y_train = X, y
        X_test, y_test = np.array([]), np.array([])
    else:
        # Split into train and test sets
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Create and train model
    model = create_lstm_model(sequence_length)

    # Training parameters
    fit_params = {
        'epochs': epochs,
        'batch_size': min(32, len(X_train)),
        'verbose': 0
    }

    # Add validation data if available
    if len(X_test) > 0:
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        fit_params['validation_data'] = (X_test, y_test)
        fit_params['callbacks'] = [early_stopping]

    # Train the model
    model.fit(X_train, y_train, **fit_params)

    # Calculate metrics if we have test data
    test_rmse = float('nan')
    test_mape = float('nan')

    if len(X_test) > 0:
        # Evaluate model
        y_pred = model.predict(X_test, verbose=0)

        # Inverse transform predictions and actual values
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred).flatten()

        # Calculate metrics
        test_rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        # Avoid division by zero in MAPE calculation
        valid_indices = y_test_inv > 0
        if np.any(valid_indices):
            test_mape = np.mean(np.abs((y_test_inv[valid_indices] - y_pred_inv[valid_indices]) / y_test_inv[valid_indices])) * 100

    return model, scaler, test_mape, test_rmse

def forecast_with_lstm(model, scaler, last_sequence, forecast_periods=12):
    """
    Generate forecasts using a trained LSTM model

    Parameters:
    -----------
    model : tensorflow.keras.models.Sequential
        Trained LSTM model
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted scaler for data normalization
    last_sequence : numpy.ndarray
        Last sequence of values to use as input
    forecast_periods : int, optional
        Number of periods to forecast (default is 12)

    Returns:
    --------
    numpy.ndarray
        Forecasted values
    """
    # Make a copy of the last sequence
    curr_sequence = last_sequence.copy()

    # List to hold forecast
    forecast = []

    # Generate forecasts
    for _ in range(forecast_periods):
        # Reshape for prediction [samples, time steps, features]
        curr_sequence_reshaped = curr_sequence.reshape(1, curr_sequence.shape[0], 1)

        # Make prediction
        pred = model.predict(curr_sequence_reshaped, verbose=0)
        pred_value = pred[0][0]  # Extract the predicted value (single scalar)

        # Add prediction to forecast
        forecast.append(pred_value)

        # Update sequence: remove first element and add prediction at the end
        curr_sequence = np.append(curr_sequence[1:], pred_value)

    # Inverse transform forecast
    forecast = np.array(forecast).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast).flatten()

    # Ensure non-negative values
    forecast = np.maximum(0, forecast)

    return forecast

def evaluate_models(sku_data, models_to_evaluate=None, test_size=0.2, forecast_periods=12):
    """
    Evaluate multiple forecasting models on a test set and select the best one

    Parameters:
    -----------
    sku_data : pandas.DataFrame
        Time series data for a specific SKU with date and quantity columns
    models_to_evaluate : list, optional
        List of model types to evaluate (default is None, which evaluates all models)
    test_size : float, optional
        Proportion of data to use for testing (default is 0.2)
    forecast_periods : int, optional
        Number of periods to forecast (default is 12)

    Returns:
    --------
    dict
        Dictionary with best model information and metrics for all evaluated models
    """
    # Make a copy of the data
    data = sku_data.copy()

    # Ensure we have enough data
    if len(data) < 12:
        return {"best_model": "moving_average", "metrics": {}}

    # Default models to evaluate if none specified
    if models_to_evaluate is None or len(models_to_evaluate) == 0:
        models_to_evaluate = ["arima", "sarima", "prophet", "lstm", "holtwinters"]

    # Print list of models being evaluated (for debugging)
    print(f"Evaluating models: {models_to_evaluate}")

    # Always use the last 6 months as test data regardless of test_size parameter
    if len(data) <= 6:
        # If we have 6 or fewer data points, use at least one for testing
        test_size = 1 / len(data)
        train_size = int(len(data) * (1 - test_size))
    else:
        # Use last 6 months for testing
        train_size = len(data) - 6

    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()

    # Print information about the split
    print(f"Training data: {len(train_data)} points, Test data: {len(test_data)} points (last 6 months)")

    # Metrics to store results
    metrics = {}
    all_models_test_pred = {}  # Store test predictions for visualization
    all_models_forecasts = {}  # Store future forecasts for visualization

    # Create date range for future forecasting (after the last data point)
    last_date = data['date'].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=30),  # Assuming monthly data
        periods=forecast_periods,
        freq='MS'  # Month start frequency
    )

    # Evaluate each model
    for model_type in models_to_evaluate:
        try:
            if model_type == "arima":
                # Train ARIMA model on training data
                model = ARIMA(train_data['quantity'], order=(1, 1, 1))
                model_fit = model.fit()

                # Generate test forecasts
                forecast_obj = model_fit.get_forecast(steps=len(test_data))
                y_pred = forecast_obj.predicted_mean.values

                # Store test predictions
                all_models_test_pred[model_type] = pd.Series(y_pred, index=test_data['date'])

                # Train on all data for future forecast
                full_model = ARIMA(data['quantity'], order=(1, 1, 1))
                full_model_fit = full_model.fit()

                # Generate future forecasts
                future_forecast = full_model_fit.get_forecast(steps=forecast_periods)
                future_values = future_forecast.predicted_mean.values

                # Store future forecasts
                all_models_forecasts[model_type] = pd.Series(future_values, index=future_dates)

            elif model_type == "sarima":
                # Adjust seasonal period based on data length
                seasonal_period = 12  # Default for annual seasonality

                # For shorter data series, use a smaller seasonal period or no seasonality
                if len(train_data) >= 24:
                    # Full annual seasonality for data with 2+ years
                    seasonal_order = (1, 1, 1, 12)
                elif len(train_data) >= 12:
                    # Reduced seasonal component for data with 1+ year
                    seasonal_order = (1, 1, 0, 4)  # Quarterly-like pattern
                else:
                    # No seasonality for short data
                    seasonal_order = (0, 0, 0, 0)

                try:
                    # Train SARIMA model with adjusted seasonality
                    model = SARIMAX(
                        train_data['quantity'],
                        order=(1, 1, 1),
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    model_fit = model.fit(disp=False, maxiter=50, method='powell')

                    # Generate test forecasts
                    forecast_obj = model_fit.get_forecast(steps=len(test_data))
                    y_pred = forecast_obj.predicted_mean.values

                    # Store test predictions
                    all_models_test_pred[model_type] = pd.Series(y_pred, index=test_data['date'])

                    # Train on all data for future forecast
                    full_model = SARIMAX(
                        data['quantity'],
                        order=(1, 1, 1),
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    full_model_fit = full_model.fit(disp=False, maxiter=50, method='powell')

                    # Generate future forecasts
                    future_forecast = full_model_fit.get_forecast(steps=forecast_periods)
                    future_values = future_forecast.predicted_mean.values

                    # Store future forecasts
                    all_models_forecasts[model_type] = pd.Series(future_values, index=future_dates)
                except Exception as e:
                    print(f"SARIMA model failed: {str(e)}")
                    continue

            elif model_type == "prophet":
                # Prepare data for Prophet
                prophet_train = pd.DataFrame({
                    'ds': train_data['date'],
                    'y': train_data['quantity']
                })

                # Train Prophet model
                m = Prophet(
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=0.1,
                    yearly_seasonality=True if len(train_data) >= 24 else False,
                    weekly_seasonality=False,
                    daily_seasonality=False
                )
                m.fit(prophet_train)

                # Create future dataframe for test period
                test_future = pd.DataFrame({'ds': test_data['date']})

                # Generate test forecasts
                test_forecast = m.predict(test_future)
                y_pred = test_forecast['yhat'].values

                # Store test predictions
                all_models_test_pred[model_type] = pd.Series(y_pred, index=test_data['date'])

                # Train on all data for future forecast
                prophet_full = pd.DataFrame({
                    'ds': data['date'],
                    'y': data['quantity']
                })

                full_m = Prophet(
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=0.1,
                    yearly_seasonality=True if len(data) >= 24 else False,
                    weekly_seasonality=False,
                    daily_seasonality=False
                )
                full_m.fit(prophet_full)

                # Create future dataframe for forecast period
                future = pd.DataFrame({'ds': future_dates})

                # Generate future forecasts
                future_forecast = full_m.predict(future)
                future_values = future_forecast['yhat'].values

                # Store future forecasts
                all_models_forecasts[model_type] = pd.Series(future_values, index=future_dates)

            elif model_type == "lstm":
                # Prepare data for LSTM
                sequence_length = min(12, len(train_data) // 2)

                # Train LSTM model
                model, scaler, _, _ = train_lstm_model(
                    train_data['quantity'].values,
                    test_size=0,  # Use all train data
                    sequence_length=sequence_length,
                    epochs=50
                )

                # Prepare last sequence for forecasting
                last_sequence = train_data['quantity'].values[-sequence_length:]
                last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()

                # Generate test forecasts
                y_pred = forecast_with_lstm(
                    model,
                    scaler,
                    last_sequence_scaled,
                    forecast_periods=len(test_data)
                )

                # Store test predictions
                all_models_test_pred[model_type] = pd.Series(y_pred, index=test_data['date'])

                # Train on all data for future forecast
                full_model, full_scaler, _, _ = train_lstm_model(
                    data['quantity'].values,
                    test_size=0,  # Use all data
                    sequence_length=sequence_length,
                    epochs=50
                )

                # Prepare last sequence for future forecasting
                full_last_sequence = data['quantity'].values[-sequence_length:]
                full_last_sequence_scaled = full_scaler.transform(full_last_sequence.reshape(-1, 1)).flatten()

                # Generate future forecasts
                future_values = forecast_with_lstm(
                    full_model,
                    full_scaler,
                    full_last_sequence_scaled,
                    forecast_periods=forecast_periods
                )

                # Store future forecasts
                all_models_forecasts[model_type] = pd.Series(future_values, index=future_dates)

            elif model_type == "holtwinters":
                # Check if we have enough data
                if len(train_data) >= 12:
                    try:
                        # Determine seasonal period if possible
                        if len(train_data) >= 24:
                            seasonal_periods = 12  # Monthly data, annual seasonality
                        else:
                            seasonal_periods = 4   # Quarterly-like pattern for shorter data

                        # Train Holt-Winters Exponential Smoothing model
                        model = ExponentialSmoothing(
                            train_data['quantity'],
                            trend='add',               # Additive trend
                            seasonal='add',            # Additive seasonality
                            seasonal_periods=seasonal_periods,
                            damped=True                # Damped trend to avoid over-forecasting
                        )
                        model_fit = model.fit(
                            optimized=True,            # Find optimal parameters
                            use_brute=False            # Use gradient descent for speed
                        )

                        # Generate test forecasts
                        y_pred = model_fit.forecast(steps=len(test_data))

                        # Convert to NumPy array for consistency
                        y_pred = y_pred.values

                        # Store test predictions
                        all_models_test_pred[model_type] = pd.Series(y_pred, index=test_data['date'])

                        # Train on all data for future forecast
                        full_model = ExponentialSmoothing(
                            data['quantity'],
                            trend='add',
                            seasonal='add',
                            seasonal_periods=seasonal_periods,
                            damped=True
                        )
                        full_model_fit = full_model.fit(
                            optimized=True,
                            use_brute=False
                        )

                        # Generate future forecasts
                        future_values = full_model_fit.forecast(steps=forecast_periods)

                        # Store future forecasts
                        all_models_forecasts[model_type] = pd.Series(future_values, index=future_dates)
                    except Exception as e:
                        print(f"Holt-Winters model failed: {str(e)}")
                        continue
                else:
                    # Skip if not enough data
                    continue

            else:
                # Skip unknown model type
                continue

            # Calculate metrics
            y_true = test_data['quantity'].values

            # Ensure predictions are non-negative
            y_pred = np.maximum(0, y_pred)

            # Calculate RMSE
            rmse = math.sqrt(mean_squared_error(y_true, y_pred))

            # Calculate MAPE, handling zero values
            mask = y_true > 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan

            # Calculate MAE
            mae = mean_absolute_error(y_true, y_pred)

            # Store metrics
            metrics[model_type] = {
                'rmse': rmse,
                'mape': mape,
                'mae': mae
            }

        except Exception as e:
            # Skip failed models
            print(f"Model {model_type} failed: {str(e)}")
            continue

    # Select best model based on RMSE, but only from models requested by user
    if metrics:
        if models_to_evaluate:
            # Filter to only include models that were requested
            valid_models = {k: v for k, v in metrics.items() if k in models_to_evaluate}
            if valid_models:
                best_model = min(valid_models.items(), key=lambda x: x[1]['rmse'])[0]
            else:
                best_model = min(metrics.items(), key=lambda x: x[1]['rmse'])[0]
        else:
            best_model = min(metrics.items(), key=lambda x: x[1]['rmse'])[0]
    else:
        best_model = "moving_average"
        # Add moving average as fallback
        window = min(3, len(data) // 2)

        # Calculate moving average for test period
        ma_pred = np.array([train_data['quantity'].rolling(window=window).mean().iloc[-1]] * len(test_data))
        all_models_test_pred["moving_average"] = pd.Series(ma_pred, index=test_data['date'])

        # Calculate moving average for future
        ma_future = np.array([data['quantity'].rolling(window=window).mean().iloc[-1]] * forecast_periods)
        all_models_forecasts["moving_average"] = pd.Series(ma_future, index=future_dates)

        # Calculate basic metrics for moving average
        y_true = test_data['quantity'].values

        # Avoid errors if we have no test data
        if len(y_true) > 0:
            rmse = math.sqrt(mean_squared_error(y_true, ma_pred))
            mae = mean_absolute_error(y_true, ma_pred)

            # Calculate MAPE, handling zero values
            mask = y_true > 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - ma_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan
        else:
            rmse = np.nan
            mae = np.nan
            mape = np.nan

        metrics["moving_average"] = {
            'rmse': rmse,
            'mape': mape,
            'mae': mae
        }

    # Always calculate forecasts for ALL selected models (no fallback values)
    if models_to_evaluate:
        print(f"Ensuring forecasts for all selected models: {models_to_evaluate}")
        for model in models_to_evaluate:
            model_lower = model.lower()
            if model_lower not in all_models_forecasts or all_models_forecasts[model_lower].isnull().all():
                # Try to actually calculate a forecast for the missing model
                try:
                    if model_lower == "arima":
                        # Train ARIMA model on all data
                        full_model = ARIMA(data['quantity'], order=(1, 1, 1))
                        full_model_fit = full_model.fit()
                        
                        # Generate future forecasts
                        future_forecast = full_model_fit.get_forecast(steps=forecast_periods)
                        future_values = future_forecast.predicted_mean.values
                        
                        # Store forecast
                        all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                        
                    elif model_lower == "sarima":
                        # Determine seasonal period based on data length
                        if len(data) >= 24:
                            seasonal_order = (1, 1, 1, 12)  # Annual seasonality
                        elif len(data) >= 12:
                            seasonal_order = (1, 1, 0, 4)   # Quarterly-like pattern
                        else:
                            seasonal_order = (0, 0, 0, 0)   # No seasonality
                            
                        # Train SARIMA model
                        full_model = SARIMAX(
                            data['quantity'],
                            order=(1, 1, 1),
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        full_model_fit = full_model.fit(disp=False, maxiter=50, method='powell')
                        
                        # Generate future forecasts
                        future_forecast = full_model_fit.get_forecast(steps=forecast_periods)
                        future_values = future_forecast.predicted_mean.values
                        
                        # Store forecast
                        all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                        
                    elif model_lower == "prophet":
                        # Prepare data for Prophet
                        prophet_full = pd.DataFrame({
                            'ds': data['date'],
                            'y': data['quantity']
                        })
                        
                        # Train Prophet model
                        full_m = Prophet(
                            changepoint_prior_scale=0.05,
                            seasonality_prior_scale=0.1,
                            yearly_seasonality=True if len(data) >= 24 else False,
                            weekly_seasonality=False,
                            daily_seasonality=False
                        )
                        full_m.fit(prophet_full)
                        
                        # Create future dataframe
                        future = pd.DataFrame({'ds': future_dates})
                        
                        # Generate forecasts
                        future_forecast = full_m.predict(future)
                        future_values = future_forecast['yhat'].values
                        
                        # Store forecast
                        all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                        
                    elif model_lower == "lstm":
                        # Train LSTM model if data is sufficient
                        if len(data) >= sequence_length + 2:
                            sequence_length = min(12, len(data) // 3)
                            
                            # Train model
                            full_model, full_scaler, _, _ = train_lstm_model(
                                data['quantity'].values,
                                test_size=0,  # Use all data
                                sequence_length=sequence_length,
                                epochs=50
                            )
                            
                            # Prepare last sequence
                            full_last_sequence = data['quantity'].values[-sequence_length:]
                            full_last_sequence_scaled = full_scaler.transform(full_last_sequence.reshape(-1, 1)).flatten()
                            
                            # Generate forecasts
                            future_values = forecast_with_lstm(
                                full_model,
                                full_scaler,
                                full_last_sequence_scaled,
                                forecast_periods=forecast_periods
                            )
                            
                            # Store forecast
                            all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                        else:
                            # For LSTM, if we have insufficient data, use a simple moving average
                            window = min(3, len(data) // 2)
                            avg_value = data['quantity'].rolling(window=window).mean().iloc[-1]
                            if pd.isna(avg_value):
                                avg_value = data['quantity'].mean() if len(data) > 0 else 0
                            future_values = [avg_value] * forecast_periods
                            all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                            
                    elif model_lower == "holtwinters":
                        # Try Holt-Winters if data is sufficient
                        if len(data) >= 12:
                            # Determine seasonal period
                            if len(data) >= 24:
                                seasonal_periods = 12  # Annual seasonality
                            else:
                                seasonal_periods = 4   # Quarterly-like pattern
                                
                            # Train model
                            full_model = ExponentialSmoothing(
                                data['quantity'],
                                trend='add',
                                seasonal='add',
                                seasonal_periods=seasonal_periods,
                                damped=True
                            )
                            full_model_fit = full_model.fit(
                                optimized=True,
                                use_brute=False
                            )
                            
                            # Generate forecasts
                            future_values = full_model_fit.forecast(steps=forecast_periods)
                            
                            # Store forecast
                            all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                        else:
                            # For Holt-Winters, if we have insufficient data, use a simple moving average
                            window = min(3, len(data) // 2)
                            avg_value = data['quantity'].rolling(window=window).mean().iloc[-1]
                            if pd.isna(avg_value):
                                avg_value = data['quantity'].mean() if len(data) > 0 else 0
                            future_values = [avg_value] * forecast_periods
                            all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                            
                    else:
                        # For other models, use a simple moving average
                        window = min(3, len(data) // 2)
                        avg_value = data['quantity'].rolling(window=window).mean().iloc[-1]
                        if pd.isna(avg_value):
                            avg_value = data['quantity'].mean() if len(data) > 0 else 0
                        future_values = [avg_value] * forecast_periods
                        all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                        
                except Exception as e:
                    # If calculation fails, use a simple average (but log the error)
                    print(f"Error calculating {model_lower} forecast: {str(e)}")
                    window = min(3, len(data) // 2)
                    avg_value = data['quantity'].rolling(window=window).mean().iloc[-1]
                    if pd.isna(avg_value):
                        avg_value = data['quantity'].mean() if len(data) > 0 else 0
                    future_values = [avg_value] * forecast_periods
                    all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                
                # Also ensure there's an entry in metrics for this model
                if model_lower not in metrics:
                    metrics[model_lower] = {
                        'rmse': np.nan,
                        'mape': np.nan,
                        'mae': np.nan
                    }

    # Return complete evaluation results
    return {
        "best_model": best_model,
        "metrics": metrics,
        "train_set": pd.Series(train_data['quantity'].values, index=train_data['date']),
        "test_set": pd.Series(test_data['quantity'].values, index=test_data['date']),
        "test_predictions": all_models_test_pred.get(best_model, pd.Series([])),
        "all_models_forecasts": all_models_forecasts,
        "all_models_test_pred": all_models_test_pred
    }

def generate_forecasts(sales_data, cluster_info, forecast_periods=12, evaluate_models_flag=False, models_to_evaluate=None, selected_skus=None, progress_callback=None):
    """
    Generate forecasts for SKUs based on their clusters

    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Sales data with date, sku, and quantity columns
    cluster_info : pandas.DataFrame
        Information about SKU clusters
    forecast_periods : int, optional
        Number of periods to forecast (default is 12)
    evaluate_models_flag : bool, optional
        Whether to evaluate models on test data (default is False)
    models_to_evaluate : list, optional
        List of model types to evaluate (default is None, which evaluates all models)
    selected_skus : list, optional
        List of specific SKUs to forecast. If None, forecasts all SKUs (default is None)
    progress_callback : function, optional
        Callback function to report progress (current_index, current_sku, total_skus)

    Returns:
    --------
    dict
        Dictionary with forecast results for each SKU
    """
    # Ensure data is sorted
    sales_data = sales_data.sort_values(by=['sku', 'date'])

    # List to hold forecast results
    forecasts = {}

    # Get unique SKUs, filtered by selected_skus if provided
    if selected_skus is not None and len(selected_skus) > 0:
        skus = [sku for sku in selected_skus if sku in sales_data['sku'].unique()]
    else:
        skus = sales_data['sku'].unique()

    # Track progress for the callback
    total_skus = len(skus)

    for i, sku in enumerate(skus):
        try:
            # Update progress if callback is provided
            if progress_callback:
                progress_callback(i+1, sku, total_skus)

            # Filter data for this SKU
            sku_data = sales_data[sales_data['sku'] == sku].copy()

            # Ensure we have enough data points
            if len(sku_data) < 3:
                continue

            # Get cluster for this SKU if available
            if sku in cluster_info['sku'].values:
                sku_cluster = cluster_info[cluster_info['sku'] == sku]['cluster'].iloc[0]
                sku_cluster_name = cluster_info[cluster_info['sku'] == sku]['cluster_name'].iloc[0]
            else:
                sku_cluster = -1
                sku_cluster_name = "Unclassified"

            # Evaluate models if requested
            model_evaluation = None
            if evaluate_models_flag and len(sku_data) >= 12:
                model_evaluation = evaluate_models(
                    sku_data,
                    models_to_evaluate=models_to_evaluate,
                    test_size=0.2,
                    forecast_periods=forecast_periods
                )

                # Use best model for forecasting
                best_model = model_evaluation["best_model"]
            else:
                best_model = None

            # Select model and generate forecast
            if best_model == "lstm" and len(sku_data) >= 24:
                try:
                    # Train LSTM with full data
                    sequence_length = min(12, len(sku_data) // 3)  # Reduce sequence length to ensure enough training samples

                    # Ensure we have enough data points for the LSTM sequence
                    if len(sku_data) > sequence_length + 2:
                        model, scaler, _, _ = train_lstm_model(
                            sku_data['quantity'].values,
                            test_size=0.1,  # Small validation set
                            sequence_length=sequence_length,
                            epochs=50
                        )

                        # Prepare last sequence for forecasting
                        last_sequence = sku_data['quantity'].values[-sequence_length:]
                        last_sequence = scaler.transform(last_sequence.reshape(-1, 1)).flatten()

                        # Generate forecasts
                        forecast_values = forecast_with_lstm(
                            model,
                            scaler,
                            last_sequence,
                            forecast_periods=forecast_periods
                        )

                        # Create confidence intervals (wider for LSTM to reflect uncertainty)
                        lower_bound = forecast_values * 0.75
                        upper_bound = forecast_values * 1.25

                        # Create dates for forecast periods
                        last_date = sku_data['date'].max()
                        forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, forecast_periods+1)]

                        # Create forecast result
                        forecast_result = {
                            "model": "lstm",
                            "forecast": pd.Series(forecast_values, index=forecast_dates),
                            "lower_bound": pd.Series(lower_bound, index=forecast_dates),
                            "upper_bound": pd.Series(upper_bound, index=forecast_dates)
                        }
                    else:
                        # Fall back to another model if sequence length doesn't work
                        forecast_result = select_best_model(sku_data, forecast_periods)
                except Exception as e:
                    # If LSTM fails, fall back to another model
                    print(f"LSTM failed for SKU {sku}: {str(e)}")
                    forecast_result = select_best_model(sku_data, forecast_periods)
            else:
                # Use select_best_model for other cases
                forecast_result = select_best_model(sku_data, forecast_periods)

            # Add metadata to result
            forecast_result['sku'] = sku
            forecast_result['cluster'] = sku_cluster
            forecast_result['cluster_name'] = sku_cluster_name

            # Add model evaluation results if available
            if model_evaluation:
                forecast_result['model_evaluation'] = model_evaluation

            # Store in dictionary
            forecasts[sku] = forecast_result

        except Exception as e:
            print(f"Error forecasting SKU {sku}: {str(e)}")
            # Continue with the next SKU rather than failing the entire process
            continue

    return forecasts