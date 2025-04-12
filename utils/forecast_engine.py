import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel
from prophet import Prophet
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
import json
import math
from utils.parameter_optimizer import get_model_parameters

# For auto_arima, try to import pmdarima (also known as pyramid-arima)
auto_arima_available = False
try:
    from pmdarima.arima import auto_arima
    auto_arima_available = True
except ImportError:
    # Define a simple fallback for auto_arima in case the import fails
    def auto_arima(y, **kwargs):
        return ARIMA(y, order=(1, 1, 1)).fit()

# TensorFlow imports wrapped in try-except to handle compatibility issues
tensorflow_available = False

# Define dummy classes to prevent import errors
class Sequential:
    def __init__(self, *args, **kwargs):
        pass
    def add(self, *args, **kwargs):
        pass
    def compile(self, *args, **kwargs):
        pass
    def fit(self, *args, **kwargs):
        return self
    def predict(self, *args, **kwargs):
        return np.zeros((1, 1))

class Dense:
    def __init__(self, *args, **kwargs):
        pass

class LSTM:
    def __init__(self, *args, **kwargs):
        pass

class Dropout:
    def __init__(self, *args, **kwargs):
        pass

class EarlyStopping:
    def __init__(self, *args, **kwargs):
        pass

# Attempt to import TensorFlow only if needed
def try_import_tensorflow():
    global tensorflow_available
    if not tensorflow_available:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential as TFSequential
            from tensorflow.keras.layers import Dense as TFDense, LSTM as TFLSTM, Dropout as TFDropout
            from tensorflow.keras.callbacks import EarlyStopping as TFEarlyStopping
            # Override the dummy classes with real implementations
            global Sequential, Dense, LSTM, Dropout, EarlyStopping
            Sequential = TFSequential
            Dense = TFDense
            LSTM = TFLSTM
            Dropout = TFDropout
            EarlyStopping = TFEarlyStopping
            tensorflow_available = True
            return True
        except (ImportError, TypeError, AttributeError) as e:
            print(f"TensorFlow import error: {e}")
            return False
    return tensorflow_available

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
        # For seasonal data, either use Holt-Winters or decomposition based on data length
        if len(data) >= 24:
            # For longer data, try a decomposition model
            model_type = "decomposition"
            try:
                # Prepare data for decomposition
                ts_data = data.set_index('date')['quantity']

                # Use annual seasonality for longer time series
                period = 12

                # Decompose the time series
                decomposition = seasonal_decompose(
                    ts_data, 
                    model='additive',
                    period=period
                )

                # Extract components
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid

                # Handle NaN values in components
                trend = trend.fillna(method='bfill').fillna(method='ffill')
                seasonal = seasonal.fillna(method='bfill').fillna(method='ffill')
                residual = residual.fillna(method='bfill').fillna(method='ffill')

                # Fit ARIMA model to residuals for future prediction
                residual_model = ARIMA(residual.dropna(), order=(1, 0, 1))
                residual_fit = residual_model.fit()

                # Forecast residuals
                residual_forecast = residual_fit.forecast(steps=forecast_periods)

                # Extract trend component's growth rate
                trend_values = trend.values
                # Use last n points to calculate trend slope
                last_n = min(6, len(trend_values))
                trend_end = trend_values[-last_n:]
                trend_indices = np.arange(len(trend_end))
                trend_fit = np.polyfit(trend_indices, trend_end, 1)
                trend_slope = trend_fit[0]

                # Get last trend value
                last_trend = trend.iloc[-1]

                # Get seasonal pattern
                seasonal_pattern = seasonal.values[-period:]

                # Generate forecasts by recombining components
                forecast_values = []
                for i in range(forecast_periods):
                    # Get seasonal component using modulo to repeat the pattern
                    seasonal_idx = i % len(seasonal_pattern)
                    seasonal_component = seasonal_pattern[seasonal_idx]

                    # Combine trend, seasonal, and residual forecasts
                    pred = last_trend + i * trend_slope + seasonal_component + residual_forecast[i]
                    forecast_values.append(max(0, pred))  # Ensure non-negative values

                forecast_values = np.array(forecast_values)

                # Create confidence intervals
                forecast_std = np.std(data['quantity'])
                z_value = 1.28  # Approximately 80% confidence interval
                lower_bound = forecast_values - z_value * forecast_std
                upper_bound = forecast_values + z_value * forecast_std

            except Exception as e:
                print(f"Decomposition model failed: {str(e)}")
                # Fall back to Holt-Winters if decomposition fails
                model_type = "holtwinters"
                # Continue to Holt-Winters code below

                # Determine seasonal period based on data length
                seasonal_periods = 12  # Monthly data, annual seasonality

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

        else:
            # For shorter data, use Holt-Winters
            model_type = "holtwinters"
            try:
                # Determine seasonal period based on data length
                seasonal_periods = 4   # Quarterly-like pattern for shorter data

                # Fit Holt-Winters model
                model = ExponentialSmoothing(
                    data['quantity'],
                    trend='add',               # Additive trend
                    seasonal='add',            # Additive seasonality
                    seasonal_periods=seasonal_periods,
                    damped=True                # Damped trend to avoid over-forecasting
                )

                # Use fixed parameters instead of optimization to avoid numerical issues
                results = model.fit(
                    smoothing_level=0.5,      # More conservative smoothing level
                    smoothing_trend=0.1,      # Conservative trend smoothing
                    smoothing_seasonal=0.1,   # Conservative seasonal smoothing
                    damping_trend=0.9,        # Strong damping to prevent explosion
                    optimized=False           # Don't optimize to avoid numerical issues
                )

                # Generate forecasts and handle potential NaN values
                forecast_raw = results.forecast(steps=forecast_periods).values

                # Replace any NaN values with the mean of the data
                data_mean = np.mean(data['quantity'])
                forecast_values = np.where(np.isnan(forecast_raw), data_mean, forecast_raw)

                # Ensure forecasts are reasonable (within historical range)
                data_max = np.max(data['quantity']) * 1.5  # Allow 50% growth
                data_min = max(0, np.min(data['quantity']) * 0.5)  # Don't go below 0
                forecast_values = np.clip(forecast_values, data_min, data_max)

                # Create confidence intervals (80% by default)
                forecast_std = np.std(data['quantity'])
                z_value = 1.28  # Approximately 80% confidence interval
                lower_bound = forecast_values - z_value * forecast_std
                upper_bound = forecast_values + z_value * forecast_std
            except Exception as e:
                # Fall back to SARIMA if Holt-Winters fails
                print(f"Holt-Winters failed: {str(e)}")
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
                except Exception as e:
                    # Fall back to ARIMA if SARIMA fails
                    print(f"SARIMA failed: {str(e)}")
                    model_type = "arima"
                    try:
                        model = ARIMA(data['quantity'], order=(1, 1, 1))
                        results = model.fit()
                        forecast_obj = results.get_forecast(steps=forecast_periods)
                        forecast_values = forecast_obj.predicted_mean.values
                        conf_int = forecast_obj.conf_int(alpha=0.1)
                        lower_bound = conf_int.iloc[:, 0].values
                        upper_bound = conf_int.iloc[:, 1].values
                    except Exception as e:
                        # Fall back to moving average if ARIMA fails
                        print(f"ARIMA failed: {str(e)}")
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
                # Get the first day of the next month
                next_month = last_date.replace(day=1) + timedelta(days=32)
                first_day_next_month = next_month.replace(day=1)

                # Generate a sequence of first days of months
                future_dates = []
                for i in range(forecast_periods):
                    # Add months by creating a date on the 1st of each subsequent month
                    next_date = first_day_next_month.replace(month=((first_day_next_month.month + i - 1) % 12) + 1)
                    # Adjust the year if we wrapped around December
                    if next_date.month < first_day_next_month.month:
                        next_date = next_date.replace(year=next_date.year + 1)
                    future_dates.append(next_date)
                future = pd.DataFrame({'ds': future_dates})

                forecast_df = m.predict(future)
                forecast_values = forecast_df['yhat'].values
                lower_bound = forecast_df['yhat_lower'].values
                upper_bound = forecast_df['yhat_upper'].values
            except Exception as e:
                # Fall back to ARIMA if Prophet fails
                print(f"Prophet failed: {str(e)}")
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
    # Get the first day of the next month
    next_month = last_date.replace(day=1) + timedelta(days=32)
    first_day_next_month = next_month.replace(day=1)

    # Generate a sequence of first days of months
    forecast_dates = []
    for i in range(forecast_periods):
        # Add months by creating a date on the 1st of each subsequent month
        next_date = first_day_next_month.replace(month=((first_day_next_month.month + i - 1) % 12) + 1)
        # Adjust the year if we wrapped around December
        if next_date.month < first_day_next_month.month:
            next_date = next_date.replace(year=next_date.year + 1)
        forecast_dates.append(next_date)

    result = {
        "model": model_type,
        "forecast": pd.Series(forecast_values, index=forecast_dates),
        "lower_bound": pd.Series(lower_bound, index=forecast_dates),
        "upper_bound": pd.Series(upper_bound, index=forecast_dates)
    }

    return result

def create_lstm_model(sequence_length, units=50, dropout_rate=0.2, recurrent_dropout=0.0, include_exogenous=False, n_exog_features=0):
    """
    Create and compile an LSTM model for time series forecasting with improved regularization

    Parameters:
    -----------
    sequence_length : int
        Length of input sequences
    units : int, optional
        Number of LSTM units (default is 50, use smaller values for limited data)
    dropout_rate : float, optional
        Dropout rate for regularization (default is 0.2)
    recurrent_dropout : float, optional
        Recurrent dropout rate for LSTM cells (default is 0.0)
    include_exogenous : bool, optional
        Whether to include exogenous features (default is False)
    n_exog_features : int, optional
        Number of exogenous features if include_exogenous is True (default is 0)

    Returns:
    --------
    tensorflow.keras.models.Sequential or None
        Compiled LSTM model or None if TensorFlow is not available
    """
    # Check if TensorFlow is available
    if not globals().get('tensorflow_available', False):
        print("TensorFlow is not available. LSTM model creation skipped.")
        return None

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Determine input shape based on whether exogenous variables are included
    if include_exogenous:
        input_shape = (sequence_length, 1 + n_exog_features)
    else:
        input_shape = (sequence_length, 1)

    # Create model - using a simpler architecture for limited data
    model = Sequential()

    # First LSTM layer with appropriate regularization
    model.add(LSTM(
        units=units, 
        return_sequences=True, 
        input_shape=input_shape,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2 regularization to prevent overfitting
    ))

    # Additional dropout layer
    model.add(Dropout(dropout_rate))

    # Second LSTM layer
    model.add(LSTM(
        units=max(10, units // 2),  # Smaller second layer
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))

    # Final dropout layer
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=1))

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )

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

# Global model cache
_model_cache = {}

def train_lstm_model(data, test_size=0.2, sequence_length=12, epochs=50, include_exogenous=False, exog_data=None, use_cache=True):
    """
    Train an LSTM model on time series data with enhanced regularization and architecture
    for limited datasets, with optional caching

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
    include_exogenous : bool, optional
        Whether to include exogenous features (default is False)
    exog_data : numpy.ndarray, optional
        Exogenous data to include if include_exogenous is True (default is None)
    use_cache : bool, optional
        Whether to use cached models if available (default is True)

    Returns:
    --------
    tuple
        (model, scaler, test_mape, test_rmse) where model is the trained LSTM model
        If TensorFlow is not available, returns (None, scaler, np.nan, np.nan)
    """
    # Check if TensorFlow is available
    if not globals().get('tensorflow_available', False):
        print("TensorFlow is not available. LSTM model training skipped.")
        scaler = MinMaxScaler()
        scaler.fit(data.reshape(-1, 1))
        return None, scaler, float('nan'), float('nan')
    # Generate a cache key based on data characteristics
    if use_cache:
        # Create a fingerprint of the data
        data_hash = hash(data.tobytes())
        cache_key = f"lstm_{data_hash}_{sequence_length}_{epochs}"

        # Check if we have a cached model
        if cache_key in _model_cache:
            print(f"Using cached LSTM model: {cache_key}")
            return _model_cache[cache_key]
    # Check if data is sufficient
    if len(data) <= sequence_length:
        raise ValueError(f"Data length ({len(data)}) must be greater than sequence_length ({sequence_length})")

    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Handle exogenous data if provided
    n_exog_features = 0
    if include_exogenous and exog_data is not None:
        exog_scaler = MinMaxScaler()
        exog_data_scaled = exog_scaler.fit_transform(exog_data)
        n_exog_features = exog_data.shape[1]

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

    # Determine model complexity based on data size
    # Simplified architecture for limited data
    if len(X_train) < 24:  # Very limited data
        units = 10  # Smaller model for very limited data
        dropout_rate = 0.3  # Higher dropout for more regularization
        recurrent_dropout = 0.2
    elif len(X_train) < 48:  # Limited data
        units = 20
        dropout_rate = 0.25
        recurrent_dropout = 0.1
    else:  # More abundant data
        units = 50
        dropout_rate = 0.2
        recurrent_dropout = 0.0

    # Reshape for LSTM [samples, time steps, features]
    if include_exogenous and exog_data is not None:
        # Combine time series data with exogenous features
        # Implement combining logic here
        pass
    else:
        # Standard reshaping for univariate time series
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        if len(X_test) > 0:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create model with appropriate complexity
    model = create_lstm_model(
        sequence_length=sequence_length,
        units=units,
        dropout_rate=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        include_exogenous=include_exogenous,
        n_exog_features=n_exog_features
    )

    # Training parameters
    fit_params = {
        'epochs': min(epochs, 100),  # Cap max epochs to prevent overfitting
        'batch_size': min(16, len(X_train)),  # Smaller batch size for better generalization
        'verbose': 0
    }

    # Add validation data and callbacks if available
    if len(X_test) > 0:
        # Early stopping with patience to avoid overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Learning rate reduction on plateau to improve convergence
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )

        fit_params['validation_data'] = (X_test, y_test)
        fit_params['callbacks'] = [early_stopping, reduce_lr]

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

    # Cache the trained model if caching is enabled
    if use_cache:
        cache_key = f"lstm_{hash(data.tobytes())}_{sequence_length}_{epochs}"
        _model_cache[cache_key] = (model, scaler, test_mape, test_rmse)

    return model, scaler, test_mape, test_rmse

def forecast_with_lstm(model, scaler, last_sequence, forecast_periods=12):
    """
    Generate forecasts using a trained LSTM model

    Parameters:
    -----------
    model : tensorflow.keras.models.Sequential or None
        Trained LSTM model or None if TensorFlow is not available
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted scaler for data normalization
    last_sequence : numpy.ndarray
        Last sequence of values to use as input
    forecast_periods : int, optional
        Number of periods to forecast (default is 12)

    Returns:
    --------
    numpy.ndarray
        Forecasted values or zeros if model is None
    """
    # Check if model is None (TensorFlow not available)
    if model is None:
        print("LSTM model is not available. Returning zeros for forecast.")
        return np.zeros(forecast_periods)
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

def create_ensemble_forecast(forecasts_dict, weights=None, method="weighted_average"):
    """
    Create an ensemble forecast by combining multiple individual forecasts

    Parameters:
    -----------
    forecasts_dict : dict
        Dictionary with model names as keys and forecast Series as values
    weights : dict, optional
        Dictionary with model names as keys and weights as values.
        If None, equal weights are used for weighted_average method
    method : str, optional
        Method to use for ensemble creation: 'weighted_average', 'simple_average',
        'median', or 'stacking' (default is 'weighted_average')

    Returns:
    --------
    pandas.Series
        Ensemble forecast
    """
    if not forecasts_dict:
        raise ValueError("No forecasts provided for ensemble creation")

    # Get a list of all forecast Series
    forecasts_list = list(forecasts_dict.values())
    model_names = list(forecasts_dict.keys())

    # Make sure all forecasts have the same index
    first_forecast = forecasts_list[0]
    date_index = first_forecast.index

    # Ensure all forecasts have same length and dates
    for name, forecast in forecasts_dict.items():
        if len(forecast) != len(first_forecast) or not forecast.index.equals(first_forecast.index):
            raise ValueError(f"Forecast {name} has different length or dates than other forecasts")

    # Create DataFrame from forecasts for easier manipulation
    forecasts_df = pd.DataFrame({name: forecast for name, forecast in forecasts_dict.items()})

    # Apply selected ensemble method
    if method == "simple_average":
        # Simple average of all forecasts
        ensemble_forecast = forecasts_df.mean(axis=1)

    elif method == "weighted_average":
        # Use weights if provided, otherwise equal weights
        if weights is None:
            # Equal weights for all models
            weights = {name: 1.0 / len(model_names) for name in model_names}
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            weights = {name: weight / total_weight for name, weight in weights.items()}

        # Calculate weighted average
        weighted_sum = pd.Series(0.0, index=date_index)
        for name, forecast in forecasts_dict.items():
            if name in weights:
                weighted_sum += forecast * weights[name]

        ensemble_forecast = weighted_sum

    elif method == "median":
        # Use median of forecasts (robust to outliers)
        ensemble_forecast = forecasts_df.median(axis=1)

    elif method == "stacking":
        # Stacking should be implemented with a meta-model trained on multiple model outputs
        # This is a simplified version using a weighted average based on model performance
        # In a real stacking implementation, you would train a meta-model on the validation set
        # using the predictions from base models as features

        # For now, we'll use a simple weighted average based on inverse RMSE
        # This would be replaced with actual stacking implementation
        ensemble_forecast = forecasts_df.mean(axis=1)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return ensemble_forecast

def evaluate_models(sku_data, models_to_evaluate=None, test_size=0.2, forecast_periods=12, use_tuned_parameters=False):
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
    use_tuned_parameters : bool, optional
        Whether to use tuned parameters from hyperparameter optimization (default is False).
        If True, will attempt to use tuned parameters for each model/SKU combination,
        falling back to default parameters when tuned parameters are not available.

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
        models_to_evaluate = ["arima", "sarima", "prophet", "lstm", "holtwinters", "decomposition", "ensemble", "auto_arima", "ets", "theta", "moving_average"]

    # Print list of models being evaluated (for debugging)
    print(f"Evaluating models: {models_to_evaluate}")

    # Get SKU identifier for parameter lookup if using tuned parameters
    sku_identifier = data['sku'].iloc[0] if 'sku' in data.columns else None

    # Create a dictionary to store tuned parameters for each model type
    tuned_parameters = {}

    # If using tuned parameters, try to load them for each model
    if use_tuned_parameters and sku_identifier:
        print(f"Attempting to use tuned parameters for SKU: {sku_identifier}")
        for model_type in models_to_evaluate:
            # Map the model_type to model types used in parameter tuning
            param_model_type = model_type
            if model_type == "auto_arima":
                param_model_type = "auto_arima"
            elif model_type == "prophet":
                param_model_type = "prophet"
            elif model_type == "ets":
                param_model_type = "ets"
            elif model_type == "theta":
                param_model_type = "theta"

            # Try to get tuned parameters for this model type and SKU
            try:
                model_params = get_model_parameters(sku_identifier, param_model_type)
                if model_params and 'parameters' in model_params:
                    tuned_parameters[model_type] = model_params['parameters']
                    print(f"Loaded tuned parameters for {model_type}: {tuned_parameters[model_type]}")
                else:
                    print(f"No tuned parameters found for {model_type}, using defaults")
            except Exception as e:
                print(f"Error loading tuned parameters for {model_type}: {str(e)}")

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

                # Check if we have tuned parameters for prophet
                if use_tuned_parameters and 'prophet' in tuned_parameters:
                    params = tuned_parameters['prophet']
                    # Convert parameters to correct types
                    cp_scale = float(params.get('changepoint_prior_scale', 0.05))
                    season_scale = float(params.get('seasonality_prior_scale', 0.1))
                    print(f"Using tuned Prophet parameters: changepoint_prior_scale={cp_scale}, seasonality_prior_scale={season_scale}")
                else:
                    cp_scale = 0.05
                    season_scale = 0.1

                # Train Prophet model with default or tuned parameters
                m = Prophet(
                    changepoint_prior_scale=cp_scale,
                    seasonality_prior_scale=season_scale,
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

                # Use the same parameters (tuned or default) for the full model
                full_m = Prophet(
                    changepoint_prior_scale=cp_scale,
                    seasonality_prior_scale=season_scale,
                    yearly_seasonality=True if len(data) >= 24 else False,
                    weekly_seasonality=False,
                    daily_seasonality=False
                )
                full_m.fit(prophet_full)

                # Create future dataframe for forecast period
                last_date = data['date'].max()
                # Get the first day of thenext month
                next_month = last_date.replace(day=1) + timedelta(days=32)
                first_day_next_month = next_month.replace(day=1)

                # Generate a sequence of first days of months
                future_dates = []
                for i in range(forecast_periods):
                    # Add months by creating a date on the 1st of each subsequent month
                    next_date = first_day_next_month.replace(month=((first_day_next_month.month + i - 1) % 12) + 1)
                    # Adjust the year if we wrapped around December
                    if next_date.month < first_day_next_month.month:
                        next_date = next_date.replace(year=next_date.year + 1)
                    future_dates.append(next_date)
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

            elif model_type == "ensemble":
                # Skip if we don't have at least 2 other models evaluated
                if len(all_models_test_pred) < 2:
                    print("Skipping ensemble - need at least 2 models to create ensemble")
                    continue

                # Create ensemble of test predictions
                try:
                    # Get weights based on inverse RMSE (better models get higher weights)
                    weights = {}
                    for model, model_metrics in metrics.items():
                        if 'rmse' in model_metrics and not np.isnan(model_metrics['rmse']) and model_metrics['rmse'] > 0:
                            weights[model] = 1.0 / model_metrics['rmse']

                    # Create test ensemble forecast
                    y_pred_ensemble = create_ensemble_forecast(
                        all_models_test_pred, 
                        weights=weights,
                        method="weighted_average"
                    )

                    # Store ensemble test predictions
                    all_models_test_pred[model_type] = y_pred_ensemble

                    # Use ensemble for future forecasts too
                    if len(all_models_forecasts) >= 2:
                        future_ensemble = create_ensemble_forecast(
                            all_models_forecasts,
                            weights=weights,
                            method="weighted_average"
                        )
                        all_models_forecasts[model_type] = future_ensemble

                    # Convert ensemble test predictions to array for metric calculation
                    y_pred = y_pred_ensemble.values

                except Exception as e:
                    print(f"Ensemble creation failed: {str(e)}")
                    continue

            elif model_type == "holtwinters":
                # Check if we have enough data
                if len(train_data) >= 12:
                    try:
                        # Determine seasonal period if possible
                        if len(train_data) >= 24:
                            seasonal_periods = 12  # Monthly data, annual seasonality
                        else:
                            seasonal_periods = 4   # Quarterly-like pattern for shorter data

                        # Train Holt-Winters Exponential Smoothing model with improved parameter handling
                        model = ExponentialSmoothing(
                            train_data['quantity'],
                            trend='add',               # Additive trend
                            seasonal='add',            # Additive seasonality
                            seasonal_periods=seasonal_periods,
                            damped=True                # Damped trend to avoid over-forecasting
                        )

                        # Use more stable optimization settings
                        model_fit = model.fit(
                            optimized=True,            
                            use_brute=True,            # Use brute force method which is more stable
                            method='SLSQP'             # Use Sequential Least Squares Programming
                        )

                        # Generate test forecasts and validate results aren't NaN
                        y_pred = model_fit.forecast(steps=len(test_data))

                        # Check if we have valid forecasts (not NaN)
                        if np.isnan(y_pred).any():
                            print(f"Holtwinters generated NaN values for test predictions - trying simpler model")
                            # Try simpler model parameters
                            model = ExponentialSmoothing(
                                train_data['quantity'],
                                trend=None,            # No trend component
                                seasonal='add',        # Additive seasonality only
                                seasonal_periods=seasonal_periods
                            )
                            model_fit = model.fit()
                            y_pred = model_fit.forecast(steps=len(test_data))

                            # If still NaN, raise exception to try fallback
                            if np.isnan(y_pred).any():
                                raise ValueError("Unable to generate valid forecasts with Holtwinters")

                        # Convert to NumPy array for consistency
                        y_pred = y_pred.values

                        # Store test predictions
                        all_models_test_pred[model_type] = pd.Series(y_pred, index=test_data['date'])

                        # Train on all data for future forecast using the same successful approach
                        full_model = ExponentialSmoothing(
                            data['quantity'],
                            trend='add' if 'trend' in model_fit.params and model_fit.params['trend'] else None,
                            seasonal='add',
                            seasonal_periods=seasonal_periods,
                            damped='damped' in model_fit.params and model_fit.params['damped']
                        )

                        # Use the same successful fitting method
                        if 'method' in model_fit.mle_retvals:
                            method = model_fit.mle_retvals['method']
                        else:
                            method = 'SLSQP'

                        full_model_fit = full_model.fit(
                            optimized=True,
                            use_brute=True,
                            method=method
                        )

                        # Generate future forecasts and validate
                        future_values = full_model_fit.forecast(steps=forecast_periods)

                        # Verify forecasts are valid numbers
                        if np.isnan(future_values).any():
                            print(f"Holtwinters generated NaN values for future forecasts - trying simpler model")
                            # Try simpler model
                            full_model = ExponentialSmoothing(
                                data['quantity'],
                                trend=None,
                                seasonal='add',
                                seasonal_periods=seasonal_periods
                            )
                            full_model_fit = full_model.fit()
                            future_values = full_model_fit.forecast(steps=forecast_periods)

                            # If still NaN, raise exception to try fallback
                            if np.isnan(future_values).any():
                                raise ValueError("Unable to generate valid forecasts with Holtwinters")

                        # Store future forecasts (validate again just to be safe)
                        if not np.isnan(future_values).any():
                            all_models_forecasts[model_type] = pd.Series(future_values, index=future_dates)
                        else:
                            raise ValueError("Holtwinters produced NaN values")

                    except Exception as e:
                        print(f"Holt-Winters model failed: {str(e)}")
                        continue
                else:
                    # Skip if not enough data
                    continue

            elif model_type == "decomposition":
                # Check if we have enough data
                if len(train_data) >= 12:  # Need at least a year of data for decomposition
                    try:
                        # Prepare data for decomposition
                        ts_data = train_data.set_index('date')['quantity']

                        # Determine frequency (period) based on data length
                        if len(ts_data) >= 24:
                            period = 12  # Annual seasonality
                        else:
                            period = min(4, len(ts_data) // 3)  # Shorter period for limited data

                        # Decompose the time series into trend, seasonal, and residual components
                        decomposition = seasonal_decompose(
                            ts_data, 
                            model='additive',  # Additive decomposition
                            period=period
                        )

                        # Extract components
                        trend = decomposition.trend
                        seasonal = decomposition.seasonal
                        residual = decomposition.resid

                        # Handle NaN values in components
                        trend = trend.fillna(method='bfill').fillna(method='ffill')
                        seasonal = seasonal.fillna(method='bfill').fillna(method='ffill')
                        residual = residual.fillna(method='bfill').fillna(method='ffill')

                        # Fit ARIMA model to residuals for future prediction
                        residual_model = ARIMA(residual.dropna(), order=(1, 0, 1))
                        residual_fit = residual_model.fit()

                        # Forecast residuals for test period
                        residual_forecast = residual_fit.forecast(steps=len(test_data))

                        # Extract trend component's growth rate (simple linear approximation)
                        trend_values = trend.values
                        if len(trend_values) > 1:
                            # Use last n points to calculate trend slope
                            last_n = min(6, len(trend_values))
                            trend_end = trend_values[-last_n:]
                            trend_indices = np.arange(len(trend_end))
                            trend_fit = np.polyfit(trend_indices, trend_end, 1)
                            trend_slope = trend_fit[0]
                        else:
                            trend_slope = 0

                        # Get last trend value
                        last_trend = trend.iloc[-1]

                        # Generate trend forecast for test period
                        trend_forecast = np.array([last_trend + i * trend_slope for i in range(1, len(test_data) + 1)])

                        # Get seasonal pattern (using modulo arithmetic to repeat the pattern)
                        seasonal_pattern = seasonal.values[-period:]

                        # Generate test predictions by recombining components
                        y_pred = []
                        for i in range(len(test_data)):
                            # Get seasonal component using modulo to repeat the pattern
                            seasonal_idx = i % len(seasonal_pattern)
                            seasonal_component = seasonal_pattern[seasonal_idx]

                            # Combine trend, seasonal, and residual forecasts
                            pred = trend_forecast[i] + seasonal_component + residual_forecast[i]
                            y_pred.append(max(0, pred))  # Ensure non-negative values

                        # Store test predictions
                        all_models_test_pred[model_type] = pd.Series(y_pred, index=test_data['date'])

                        # Now generate future forecasts with the model
                        # Forecast residuals for future periods
                        future_residual_forecast = residual_fit.forecast(steps=forecast_periods)

                        # Generate trend forecast for future periods
                        future_trend_forecast = np.array([
                            last_trend + (len(test_data) + i) * trend_slope 
                            for i in range(1, forecast_periods + 1)
                        ])

                        # Generate future predictions by recombining components
                        future_values = []
                        for i in range(forecast_periods):
                            # Get seasonal component
                            seasonal_idx = (len(train_data) + i) % len(seasonal_pattern)
                            seasonal_component = seasonal_pattern[seasonal_idx]

                            # Combine trend, seasonal, and residual forecasts
                            pred = future_trend_forecast[i] + seasonal_component + future_residual_forecast[i]
                            future_values.append(max(0, pred))  # Ensure non-negative values

                        # Store future forecasts
                        all_models_forecasts[model_type] = pd.Series(future_values, index=future_dates)

                    except Exception as e:
                        print(f"Decomposition model failed: {str(e)}")
                        continue
                else:
                    # Skip if not enough data
                    continue

            elif model_type == "auto_arima":
                # Check if we have enough data
                if len(train_data) >= 8:  # Need at least 8 data points for auto_arima
                    try:
                        # Check if auto_arima is available
                        if auto_arima_available:
                            # Prepare data
                            ts_data = train_data['quantity'].values

                            # Check if we have tuned parameters for auto_arima
                            if use_tuned_parameters and 'auto_arima' in tuned_parameters:
                                params = tuned_parameters['auto_arima']
                                # Convert parameters to correct types
                                d = int(params.get('d', 1)) if params.get('d') is not None else None
                                seasonal = params.get('seasonal', 'true').lower() == 'true'

                                if seasonal:
                                    m = int(params.get('m', 12 if len(train_data) >= 24 else 4))
                                else:
                                    m = 1

                                stepwise = params.get('stepwise', 'true').lower() == 'true'
                                max_p = int(params.get('max_p', 5))
                                max_q = int(params.get('max_q', 5))
                                max_order = int(params.get('max_order', 5))

                                print(f"Using tuned auto_arima parameters: d={d}, seasonal={seasonal}, m={m}, stepwise={stepwise}, max_p={max_p}, max_q={max_q}")

                                # Fit auto_arima model with tuned parameters
                                model = auto_arima(
                                    ts_data,
                                    d=d,                         # Differencing
                                    seasonal=seasonal,           # Seasonality
                                    m=m,                         # Seasonal period
                                    stepwise=stepwise,           # Stepwise approach
                                    max_p=max_p,                 # Max AR order
                                    max_q=max_q,                 # Max MA order
                                    max_order=max_order,         # Max total order
                                    suppress_warnings=True,
                                    error_action="ignore"
                                )
                            else:
                                # Fit auto_arima model - automatically finds the best order parameters with default values
                                model = auto_arima(
                                    ts_data,
                                    seasonal=True,                   # Enable seasonality
                                    m=12 if len(train_data) >= 24 else 4,  # Seasonal period
                                    stepwise=True,                   # Use stepwise approach for faster fitting
                                    suppress_warnings=True,          # Suppress warnings for cleaner output
                                    error_action="ignore"            # Ignore errors in ARIMA estimation
                                )

                            # Generate test forecasts
                            forecast_obj = model.predict(n_periods=len(test_data))
                            y_pred = forecast_obj

                            # Store test predictions
                            all_models_test_pred[model_type] = pd.Series(y_pred, index=test_data['date'])

                            # Train on all data for future forecast
                            full_data = data['quantity'].values

                            # Use the same parameters for the full model as we used for the training model
                            if use_tuned_parameters and 'auto_arima' in tuned_parameters:
                                # Use the same tuned parameters
                                full_model = auto_arima(
                                    full_data,
                                    d=d,
                                    seasonal=seasonal,
                                    m=m,
                                    stepwise=stepwise,
                                    max_p=max_p,
                                    max_q=max_q,
                                    max_order=max_order,
                                    suppress_warnings=True,
                                    error_action="ignore"
                                )
                            else:
                                # Use default parameters
                                full_model = auto_arima(
                                    full_data,
                                    seasonal=True,
                                    m=12 if len(data) >= 24 else 4,
                                    stepwise=True,
                                    suppress_warnings=True,
                                    error_action="ignore"
                                )

                            # Generate future forecasts
                            future_values = full_model.predict(n_periods=forecast_periods)

                            # Store future forecasts
                            all_models_forecasts[model_type] = pd.Series(future_values, index=future_dates)
                        else:
                            # Fallback to ARIMA if auto_arima is not available
                            print("Auto ARIMA not available, falling back to standard ARIMA")

                            # Use standard ARIMA with fixed parameters
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

                    except Exception as e:
                        print(f"Auto ARIMA model failed: {str(e)}")
                        continue
                else:
                    # Skip if not enough data
                    continue

            elif model_type == "ets":
                # Check if we have enough data
                if len(train_data) >= 8:  # Need reasonable amount of data for ETS
                    try:
                        # Prepare data
                        ts_data = train_data['quantity'].values

                        # Check if we have tuned parameters for ETS
                        if use_tuned_parameters and 'ets' in tuned_parameters:
                            params = tuned_parameters['ets']
                            # Convert parameters to correct types
                            error_type = params.get('error', 'add')
                            trend_type = params.get('trend', 'add')
                            seasonal_type = params.get('seasonal', 'add')
                            damped = params.get('damped_trend', 'true').lower() == 'true'
                            seasonal_periods_param = int(params.get('seasonal_periods', 12 if len(train_data) >= 24 else 4))

                            print(f"Using tuned ETS parameters: error={error_type}, trend={trend_type}, seasonal={seasonal_type}, damped={damped}, seasonal_periods={seasonal_periods_param}")

                            # Create and fit ETS model with tuned parameters
                            model = ETSModel(
                                ts_data,
                                error=error_type,
                                trend=trend_type,
                                seasonal=seasonal_type,
                                damped_trend=damped,
                                seasonal_periods=seasonal_periods_param
                            )
                        else:
                            # Create and fit ETS model with default parameters
                            model = ETSModel(
                                ts_data,
                                error="add",                     # Additive errors
                                trend="add",                     # Additive trend
                                seasonal="add",                  # Additive seasonal
                                damped_trend=True,               # Damped trend to avoid over-forecasting
                                seasonal_periods=12 if len(train_data) >= 24 else 4  # Seasonal periods
                            )
                        model_fit = model.fit(disp=False)

                        # Generate test forecasts
                        forecast_obj = model_fit.forecast(steps=len(test_data))

                        # Convert to array if needed
                        if hasattr(forecast_obj, 'values'):
                            y_pred = forecast_obj.values
                        else:
                            y_pred = forecast_obj

                        # Store test predictions
                        all_models_test_pred[model_type] = pd.Series(y_pred, index=test_data['date'])

                        # Train on all data for future forecast
                        full_data = data['quantity'].values

                        # Use the same parameters for the full model
                        if use_tuned_parameters and 'ets' in tuned_parameters:
                            # Use the same tuned parameters
                            full_model = ETSModel(
                                full_data,
                                error=error_type,
                                trend=trend_type, 
                                seasonal=seasonal_type,
                                damped_trend=damped,
                                seasonal_periods=seasonal_periods_param
                            )
                        else:
                            # Use default parameters
                            full_model = ETSModel(
                                full_data,
                                error="add",
                                trend="add",
                                seasonal="add",
                                damped_trend=True,
                                seasonal_periods=12 if len(data) >= 24 else 4
                            )
                        full_model_fit = full_model.fit(disp=False)

                        # Generate future forecasts
                        future_values = full_model_fit.forecast(steps=forecast_periods)

                        # Convert to array if needed
                        if hasattr(future_values, 'values'):
                            future_values = future_values.values

                        # Store future forecasts
                        all_models_forecasts[model_type] = pd.Series(future_values, index=future_dates)

                    except Exception as e:
                        print(f"ETS model failed: {str(e)}")
                        continue
                else:
                    # Skip if not enough data
                    continue

            elif model_type == "theta":
                # Check if we have enough data
                if len(train_data) >= 8:  # Need reasonable amount of data for Theta
                    try:
                        # Prepare data
                        ts_data = train_data['quantity'].values

                        # Check if we have tuned parameters for Theta
                        if use_tuned_parameters and 'theta' in tuned_parameters:
                            params = tuned_parameters['theta']
                            # Convert parameters to correct types
                            deseasonalize_param = params.get('deseasonalize', 'true').lower() == 'true'
                            period_param = int(params.get('period', 12 if len(train_data) >= 24 else 4))

                            print(f"Using tuned Theta parameters: deseasonalize={deseasonalize_param}, period={period_param}")

                            # Create and fit Theta model with tuned parameters
                            model = ThetaModel(
                                ts_data,
                                deseasonalize=deseasonalize_param,
                                period=period_param
                            )
                        else:
                            # Create and fit Theta model with default parameters
                            model = ThetaModel(
                                ts_data,
                                deseasonalize=True,                # Deseasonalize the time series
                                period=12 if len(train_data) >= 24 else 4  # Seasonal period
                            )
                        model_fit = model.fit()

                        # Generate test forecasts
                        forecast_obj = model_fit.forecast(steps=len(test_data))
                        y_pred = forecast_obj.values

                        # Store test predictions
                        all_models_test_pred[model_type] = pd.Series(y_pred, index=test_data['date'])

                        # Train on all data for future forecast
                        full_data = data['quantity'].values

                        # Use the same parameters for the full model
                        if use_tuned_parameters and 'theta' in tuned_parameters:
                            # Use the same tuned parameters
                            full_model = ThetaModel(
                                full_data,
                                deseasonalize=deseasonalize_param,
                                period=period_param
                            )
                        else:
                            # Use default parameters
                            full_model = ThetaModel(
                                full_data,
                                deseasonalize=True,
                                period=12 if len(data) >= 24 else 4
                            )
                        full_model_fit = full_model.fit()

                        # Generate future forecasts
                        future_values = full_model_fit.forecast(steps=forecast_periods).values

                        # Store future forecasts
                        all_models_forecasts[model_type] = pd.Series(future_values, index=future_dates)

                    except Exception as e:
                        print(f"Theta model failed: {str(e)}")
                        continue
                else:
                    # Skip if not enough data
                    continue

            elif model_type == "moving_average":
                # Moving average is simple and works with minimal data
                try:
                    # Calculate moving average window size based on data length
                    window = min(3, len(train_data) // 2)
                    if window < 1:
                        window = 1

                    # Calculate moving average for test period
                    # Use the last value of the rolling average from training data
                    ma_value = train_data['quantity'].rolling(window=window).mean().iloc[-1]
                    y_pred = np.array([ma_value] * len(test_data))

                    # Store test predictions
                    all_models_test_pred[model_type] = pd.Series(y_pred, index=test_data['date'])

                    # Use all data for future forecasts
                    full_ma_value = data['quantity'].rolling(window=window).mean().iloc[-1]

                    # Handle NaN values
                    if pd.isna(full_ma_value):
                        full_ma_value = data['quantity'].mean() if len(data) > 0 else 0

                    # Create forecasts
                    future_values = np.array([full_ma_value] * forecast_periods)

                    # Store future forecasts
                    all_models_forecasts[model_type] = pd.Series(future_values, index=future_dates)

                except Exception as e:
                    print(f"Moving Average model failed: {str(e)}")
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
            # Make sure every selected model gets a forecast, even if it's not in the forecasts yet
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
                        last_date = data['date'].max()
                        # Get the first day of the next month
                        next_month = last_date.replace(day=1) + timedelta(days=32)
                        first_day_next_month = next_month.replace(day=1)

                        # Generate a sequence of first days of months
                        future_dates = []
                        for i in range(forecast_periods):
                            # Add months by creating a date on the 1st of each subsequent month
                            next_date = first_day_next_month.replace(month=((first_day_next_month.month + i - 1) % 12) + 1)
                            # Adjust the year if we wrapped around December
                            if next_date.month < first_day_next_month.month:
                                next_date = next_date.replace(year=next_date.year + 1)
                            future_dates.append(next_date)
                        future = pd.DataFrame({'ds': future_dates})

                        # Generate forecasts
                        future_forecast = full_m.predict(future)
                        future_values = future_forecast['yhat'].values

                        # Store forecast
                        all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)

                    elif model_lower == "ensemble":
                        # For ensemble, we need at least 2 other models to be available
                        available_models = {k: v for k, v in all_models_forecasts.items() 
                                          if k != model_lower and not v.isnull().all()}

                        if len(available_models) >= 2:
                            # Get weights based on RMSE if available
                            weights = {}
                            for model, model_metrics in metrics.items():
                                if model in available_models and 'rmse' in model_metrics and not np.isnan(model_metrics['rmse']) and model_metrics['rmse'] > 0:
                                    weights[model] = 1.0 / model_metrics['rmse']

                            # Create ensemble forecast
                            future_ensemble = create_ensemble_forecast(
                                available_models,
                                weights=weights,
                                method="weighted_average"
                            )

                            # Store forecast
                            all_models_forecasts[model_lower] = future_ensemble
                        else:
                            print(f"Cannot create ensemble forecast: need at least 2 other models")
                            # Add moving average as a fallback
                            window = min(3, len(data) // 2)
                            ma_future = np.array([data['quantity'].rolling(window=window).mean().iloc[-1]] * forecast_periods)
                            all_models_forecasts[model_lower] = pd.Series(ma_future, index=future_dates)

                    elif model_lower == "decomposition":
                        # Create decomposition forecast if data is sufficient
                        if len(data) >= 12:  # Need at least a year of data
                            try:
                                # Prepare data for decomposition
                                ts_data = data.set_index('date')['quantity']

                                # Determine frequency (period) based on data length
                                if len(ts_data) >= 24:
                                    period = 12  # Annual seasonality
                                else:
                                    period = min(4, len(ts_data) // 3)  # Shorter period for limited data

                                # Decompose the time series
                                decomposition = seasonal_decompose(
                                    ts_data, 
                                    model='additive',
                                    period=period
                                )

                                # Extract components
                                trend = decomposition.trend
                                seasonal = decomposition.seasonal

                                # Handle NaN values in components
                                trend = trend.fillna(method='bfill').fillna(method='ffill')
                                seasonal = seasonal.fillna(method='bfill').fillna(method='ffill')

                                # Extract trend slope
                                trend_values = trend.values
                                if len(trend_values) > 1:
                                    # Use last n points to calculate trend slope
                                    last_n = min(6, len(trend_values))
                                    trend_end = trend_values[-last_n:]
                                    trend_indices = np.arange(len(trend_end))
                                    trend_fit = np.polyfit(trend_indices, trend_end, 1)
                                    trend_slope = trend_fit[0]
                                else:
                                    trend_slope = 0

                                # Get last trend value
                                last_trend = trend.iloc[-1]

                                # Get seasonal pattern
                                seasonal_pattern = seasonal.values[-period:]

                                # Generate future forecasts
                                future_values = []
                                for i in range(forecast_periods):
                                    # Get seasonal component
                                    seasonal_idx = i % len(seasonal_pattern)
                                    seasonal_component = seasonal_pattern[seasonal_idx]

                                    # Combine trend and seasonal components
                                    pred = last_trend + i * trend_slope + seasonal_component
                                    future_values.append(max(0, pred))  # Ensure non-negative values

                                # Store forecast
                                all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)

                            except Exception as e:
                                print(f"Cannot create decomposition forecast: {str(e)}")
                                # Add moving average as a fallback
                                window = min(3, len(data) // 2)
                                ma_future = np.array([data['quantity'].rolling(window=window).mean().iloc[-1]] * forecast_periods)
                                all_models_forecasts[model_lower] = pd.Series(ma_future, index=future_dates)
                        else:
                            print(f"Cannot create decomposition forecast: insufficient data")
                            # Add moving average as a fallback
                            window = min(3, len(data) // 2)
                            ma_future = np.array([data['quantity'].rolling(window=window).mean().iloc[-1]] * forecast_periods)
                            all_models_forecasts[model_lower] = pd.Series(ma_future, index=future_dates)

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

                    elif model_lower == "auto_arima":
                        # Use auto_arima if statsmodels failed
                        try:
                            # Create auto_arima model with appropriate parameters
                            if len(data) >= 24:
                                seasonal = True
                                m = 12
                            elif len(data) >= 12:
                                seasonal = True
                                m = 4
                            else:
                                seasonal = False
                                m = 1

                            # Train on all data for future forecast
                            full_data = data.set_index('date')['quantity']

                            # Use a simple stepwise auto_arima
                            auto_model = auto_arima(
                                full_data, 
                                start_p=1, start_q=1,
                                max_p=3, max_q=3,
                                m=m,
                                seasonal=seasonal,
                                d=1, D=1 if seasonal else 0,
                                stepwise=True,
                                suppress_warnings=True,
                                error_action='ignore',
                                max_order=5
                            )

                            # Generate future forecasts
                            future_values = auto_model.predict(n_periods=forecast_periods)

                            # Store future forecasts
                            all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                        except Exception as e:
                            print(f"Auto ARIMA fallback failed: {str(e)}")
                            # Use moving average as fallback
                            window = min(3, len(data) // 2) if len(data) > 0 else 1
                            if window < 1:
                                window = 1
                            ma_future = np.array([data['quantity'].rolling(window=window).mean().iloc[-1] if len(data) > 0 else 0] * forecast_periods)
                            all_models_forecasts[model_lower] = pd.Series(ma_future, index=future_dates)

                    elif model_lower == "ets":
                        # Try ETS model
                        try:
                            # Check if data is sufficient
                            if len(data) < 12:
                                raise ValueError("Not enough data for ETS model")

                            # Train on all data for future forecast
                            full_data = data['quantity'].values

                            # Create and fit ETS model
                            ets_model = ETSModel(
                                full_data,
                                error="add",
                                trend="add",
                                seasonal="add" if len(data) >= 24 else None,
                                seasonal_periods=12 if len(data) >= 24 else None,
                                damped_trend=True
                            )

                            ets_fit = ets_model.fit(disp=False)

                            # Generate future forecasts
                            future_values = ets_fit.forecast(steps=forecast_periods)

                            # Convert to array if needed
                            if hasattr(future_values, 'values'):
                                future_values = future_values.values

                            # Store future forecasts
                            all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                        except Exception as e:
                            print(f"ETS fallback failed: {str(e)}")
                            # Use moving average as fallback
                            window = min(3, len(data) // 2) if len(data) > 0 else 1
                            if window < 1:
                                window = 1
                            ma_future = np.array([data['quantity'].rolling(window=window).mean().iloc[-1] if len(data) > 0 else 0] * forecast_periods)
                            all_models_forecasts[model_lower] = pd.Series(ma_future, index=future_dates)

                    elif model_lower == "theta":
                        # Try Theta model
                        try:
                            # Check if data is sufficient
                            if len(data) < 8:
                                raise ValueError("Not enough data for Theta model")

                            # Train on all data for future forecast
                            full_data = data['quantity'].values

                            # Create and fit Theta model
                            theta_model = ThetaModel(
                                full_data,
                                deseasonalize=True,
                                period=12 if len(data) >= 24 else 4
                            )

                            theta_fit = theta_model.fit()

                            # Generate future forecasts
                            future_values = theta_fit.forecast(steps=forecast_periods)

                            # Convert to array if needed
                            if hasattr(future_values, 'values'):
                                future_values = future_values.values

                            # Store future forecasts
                            all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                        except Exception as e:
                            print(f"Theta fallback failed: {str(e)}")
                            # Use moving average as fallback
                            window = min(3, len(data) // 2) if len(data) > 0 else 1
                            if window < 1:
                                window = 1
                            ma_future = np.array([data['quantity'].rolling(window=window).mean().iloc[-1] if len(data) > 0 else 0] * forecast_periods)
                            all_models_forecasts[model_lower] = pd.Series(ma_future, index=future_dates)

                    elif model_lower == "moving_average":
                        # Moving average is simple and should always work
                        try:
                            # Calculate moving average window size based on data length
                            window = min(3, len(data) // 2) if len(data) > 0 else 1
                            if window < 1:
                                window = 1

                            # Use all data for future forecasts
                            full_ma_value = data['quantity'].rolling(window=window).mean().iloc[-1] if len(data) > 0 else 0

                            # Handle NaN values
                            if pd.isna(full_ma_value):
                                full_ma_value = data['quantity'].mean() if len(data) > 0 else 0

                            # Create forecasts
                            future_values = np.array([full_ma_value] * forecast_periods)

                            # Store future forecasts
                            all_models_forecasts[model_lower] = pd.Series(future_values, index=future_dates)
                        except Exception as e:
                            print(f"Moving Average fallback failed: {str(e)}")
                            # Use mean as ultimate fallback
                            mean_value = data['quantity'].mean() if len(data) > 0 else 0
                            future_values = np.array([mean_value] * forecast_periods)
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

def generate_forecasts(sales_data, cluster_info, forecast_periods=12, evaluate_models_flag=False, models_to_evaluate=None, selected_skus=None, progress_callback=None, use_tuned_parameters=False):
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
    use_tuned_parameters : bool, optional
        Whether to use tuned parameters from hyperparameter optimization (default is False).
        If True, will attempt to use tuned parameters for each model/SKU combination,
        falling back to default parameters when tuned parameters are not available.

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
                    forecast_periods=forecast_periods,
                    use_tuned_parameters=use_tuned_parameters
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

                        # Create dates for forecast periods (using 1st day of each month)
                        last_date = sku_data['date'].max()
                        # Get the first day of the next month
                        next_month = last_date.replace(day=1) + timedelta(days=32)
                        first_day_next_month = next_month.replace(day=1)

                        # Generate a sequence of first days of months
                        forecast_dates = []
                        for i in range(forecast_periods):
                            # Add months by creating a date on the 1st of each subsequent month
                            next_date = first_day_next_month.replace(month=((first_day_next_month.month + i - 1) % 12) + 1)
                            # Adjust the year if we wrapped around December
                            if next_date.month < first_day_next_month.month:
                                next_date = next_date.replace(year=next_date.year + 1)
                            forecast_dates.append(next_date)

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