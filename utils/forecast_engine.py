import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import math

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
        # For seasonal data, try SARIMA
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

def generate_forecasts(sales_data, cluster_info, forecast_periods=12):
    """
    Generate forecasts for all SKUs based on their clusters
    
    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Sales data with date, sku, and quantity columns
    cluster_info : pandas.DataFrame
        Information about SKU clusters
    forecast_periods : int, optional
        Number of periods to forecast (default is 12)
    
    Returns:
    --------
    dict
        Dictionary with forecast results for each SKU
    """
    # Ensure data is sorted
    sales_data = sales_data.sort_values(by=['sku', 'date'])
    
    # List to hold forecast results
    forecasts = {}
    
    # Get unique SKUs
    skus = sales_data['sku'].unique()
    
    for sku in skus:
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
        
        # Select model and generate forecast
        forecast_result = select_best_model(sku_data, forecast_periods)
        
        # Add metadata to result
        forecast_result['sku'] = sku
        forecast_result['cluster'] = sku_cluster
        forecast_result['cluster_name'] = sku_cluster_name
        
        # Store in dictionary
        forecasts[sku] = forecast_result
    
    return forecasts
