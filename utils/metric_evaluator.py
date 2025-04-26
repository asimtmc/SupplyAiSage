import pandas as pd
import numpy as np
from datetime import datetime
import json
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_default_parameters(model_type):
    """Get default parameters for a model type"""
    if model_type in ['arima', 'auto_arima']:
        return {'p': 1, 'd': 1, 'q': 0}
    elif model_type == 'prophet':
        return {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'additive'
        }
    elif model_type == 'ets':
        return {
            'trend': 'add',
            'seasonal': None,
            'seasonal_periods': 1,
            'damped_trend': False
        }
    elif model_type == 'theta':
        return {
            'deseasonalize': True,
            'period': 12,
            'method': 'auto'
        }
    else:
        return {}

def calculate_model_improvement(sku, model_type, sample_size=None):
    """
    Calculate the improvement of optimized parameters over default parameters
    
    Parameters:
    -----------
    sku : str
        SKU identifier
    model_type : str
        Model type (e.g., 'auto_arima', 'prophet')
    sample_size : int, optional
        Number of data points to use for evaluation
    
    Returns:
    --------
    dict
        Dictionary with improvement metrics
    """
    try:
        # Get data for this SKU
        import streamlit as st
        from utils.data_loader import load_data_from_database
        
        # Load sales data into session state
        load_success = load_data_from_database()
        
        # Check if data was loaded successfully
        if not load_success or not hasattr(st.session_state, 'sales_data') or st.session_state.sales_data is None:
            logger.error(f"No sales data available for {sku}")
            return None
        
        # Filter data for this SKU
        sku_data = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku]
        if sku_data.empty:
            logger.error(f"No data found for SKU {sku}")
            return None
        
        # Sort by date and prepare for modeling
        sku_data = sku_data.sort_values('date')
        
        # Check if we have enough data for a meaningful split
        if len(sku_data) < 5:
            logger.warning(f"Not enough data for {sku} (only {len(sku_data)} points)")
            return {
                'default_rmse': None,
                'optimized_rmse': None,
                'improvement': 0.0,
                'error': 'Insufficient data for evaluation'
            }
            
        # Split into train/test sets - use 80% for training, 20% for testing
        train_size = int(len(sku_data) * 0.8)
        if train_size < 3:
            train_size = len(sku_data) - 2  # Ensure at least 2 points for testing
            
        train_data = sku_data.iloc[:train_size]
        test_data = sku_data.iloc[train_size:]
        
        if len(test_data) < 2:
            logger.warning(f"Not enough test data for {sku} (only {len(test_data)} points)")
            # Use last 2 points for testing if available
            if len(sku_data) > 2:
                train_data = sku_data.iloc[:-2]
                test_data = sku_data.iloc[-2:]
            else:
                # Not enough data to properly evaluate
                return {
                    'default_rmse': None,
                    'optimized_rmse': None,
                    'improvement': 0.0,
                    'error': 'Insufficient data for evaluation'
                }
        
        # Get optimized parameters
        from utils.database import get_model_parameters
        optimized_params_obj = get_model_parameters(sku, model_type)
        
        if not optimized_params_obj or 'parameters' not in optimized_params_obj:
            logger.warning(f"No optimized parameters found for {sku}, {model_type}")
            return {
                'default_rmse': None,
                'optimized_rmse': None,
                'improvement': 0.0,
                'error': 'No optimized parameters found'
            }
        
        optimized_params = optimized_params_obj['parameters']
        
        # Get default parameters
        default_params = get_default_parameters(model_type)
        
        # Evaluate models with both parameter sets
        default_metrics = evaluate_model(model_type, train_data, test_data, default_params)
        optimized_metrics = evaluate_model(model_type, train_data, test_data, optimized_params)
        
        if not default_metrics or not optimized_metrics:
            return {
                'default_rmse': None,
                'optimized_rmse': None,
                'improvement': 0.0,
                'error': 'Error evaluating models'
            }
        
        # Calculate improvement - handle None values
        if default_metrics['rmse'] is not None and optimized_metrics['rmse'] is not None and default_metrics['rmse'] > 0:
            improvement = (default_metrics['rmse'] - optimized_metrics['rmse']) / default_metrics['rmse']
        else:
            improvement = 0.0
        
        # Return results
        return {
            'default_rmse': default_metrics['rmse'],
            'optimized_rmse': optimized_metrics['rmse'],
            'default_mape': default_metrics['mape'],
            'optimized_mape': optimized_metrics['mape'],
            'default_mae': default_metrics['mae'],
            'optimized_mae': optimized_metrics['mae'],
            'improvement': improvement
        }
    
    except Exception as e:
        logger.error(f"Error calculating improvement for {sku}, {model_type}: {str(e)}")
        return {
            'default_rmse': None,
            'optimized_rmse': None,
            'improvement': 0.0,
            'error': str(e)
        }

def evaluate_model(model_type, train_data, test_data, parameters):
    """
    Evaluate a model with given parameters on test data
    
    Parameters:
    -----------
    model_type : str
        Model type (e.g., 'auto_arima', 'prophet')
    train_data : pandas.DataFrame
        Training data
    test_data : pandas.DataFrame
        Test data
    parameters : dict
        Model parameters
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    try:
        # Check if data is valid
        if train_data is None or test_data is None or len(train_data) == 0 or len(test_data) == 0:
            logger.error(f"Invalid data for {model_type} evaluation")
            return {'rmse': None, 'mape': None, 'mae': None, 'error': 'Invalid data'}
            
        # Make sure date and quantity columns exist
        required_columns = ['date', 'quantity']
        for col in required_columns:
            if col not in train_data.columns or col not in test_data.columns:
                logger.error(f"Missing {col} column in data for {model_type} evaluation")
                return {'rmse': None, 'mape': None, 'mae': None, 'error': f'Missing {col} column'}
                
        # Import forecast modules based on model type
        if model_type in ['arima', 'auto_arima']:
            # Use statsmodels ARIMA
            from statsmodels.tsa.arima.model import ARIMA
            
            # Prepare data - ensure quantity is numeric
            train_data['quantity'] = pd.to_numeric(train_data['quantity'], errors='coerce')
            train_data = train_data.dropna(subset=['quantity'])
            if len(train_data) == 0:
                return {'rmse': None, 'mape': None, 'mae': None, 'error': 'No valid numeric data in training set'}
                
            train_series = pd.Series(train_data['quantity'].values, index=pd.DatetimeIndex(train_data['date']))
            
            # Fit model with given parameters
            p = parameters.get('p', 1)
            d = parameters.get('d', 1)
            q = parameters.get('q', 0)
            
            model = ARIMA(train_series, order=(p, d, q))
            model_fit = model.fit()
            
            # Generate forecast for test period
            forecast = model_fit.forecast(steps=len(test_data))
            
            # Align forecast with test data
            forecast_aligned = forecast.values if len(forecast) == len(test_data) else forecast.values[:len(test_data)]
            
            # Calculate metrics
            return calculate_metrics(test_data['quantity'].values, forecast_aligned)
            
        elif model_type == 'prophet':
            # Use Prophet
            try:
                from prophet import Prophet
            except ImportError:
                return {'rmse': None, 'mape': None, 'mae': None, 'error': 'Prophet not installed'}
            
            # Prepare data
            train_prophet = train_data[['date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'})
            
            # Configure Prophet model
            model = Prophet(
                changepoint_prior_scale=parameters.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=parameters.get('seasonality_prior_scale', 10.0),
                seasonality_mode=parameters.get('seasonality_mode', 'additive')
            )
            
            # Fit model
            model.fit(train_prophet)
            
            # Create future dataframe for prediction
            future = pd.DataFrame({'ds': test_data['date']})
            
            # Make forecast
            forecast = model.predict(future)
            
            # Calculate metrics
            return calculate_metrics(test_data['quantity'].values, forecast['yhat'].values)
            
        elif model_type == 'ets':
            # Use statsmodels ETS
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Prepare data
            train_series = pd.Series(train_data['quantity'].values, index=pd.DatetimeIndex(train_data['date']))
            
            # Configure model
            trend = parameters.get('trend', None)
            seasonal = parameters.get('seasonal', None)
            seasonal_periods = parameters.get('seasonal_periods', 1)
            damped_trend = parameters.get('damped_trend', False)
            
            # Only include seasonal if seasonal periods > 1
            if seasonal_periods <= 1:
                seasonal = None
            
            model = ExponentialSmoothing(
                train_series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend
            )
            
            # Fit model
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=len(test_data))
            
            # Calculate metrics
            return calculate_metrics(test_data['quantity'].values, forecast.values)
            
        elif model_type == 'theta':
            # Use sktime Theta method
            try:
                from sktime.forecasting.theta import ThetaForecaster
            except ImportError:
                return {'rmse': None, 'mape': None, 'mae': None, 'error': 'sktime not installed'}
            
            # Prepare data
            train_series = pd.Series(train_data['quantity'].values, index=pd.DatetimeIndex(train_data['date']))
            
            # Configure model
            deseasonalize = parameters.get('deseasonalize', True)
            period = parameters.get('period', 12)
            
            # Create model
            model = ThetaForecaster(deseasonalize=deseasonalize, sp=period)
            
            # Fit model
            model.fit(train_series)
            
            # Generate forecast
            forecast = model.predict(fh=np.arange(1, len(test_data) + 1))
            
            # Calculate metrics
            return calculate_metrics(test_data['quantity'].values, forecast.values)
            
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None
    
    except Exception as e:
        logger.error(f"Error evaluating {model_type} model: {str(e)}")
        return {'rmse': None, 'mape': None, 'mae': None, 'error': str(e)}

def calculate_metrics(actual, predicted):
    """
    Calculate standard forecast evaluation metrics
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    
    Returns:
    --------
    dict
        Dictionary with metrics
    """
    try:
        # Check if inputs are None
        if actual is None or predicted is None:
            return {'rmse': None, 'mape': None, 'mae': None, 'error': 'Input data is None'}
            
        # Convert inputs to numpy arrays if they aren't already
        try:
            actual = np.array(actual, dtype=float)
            predicted = np.array(predicted, dtype=float)
        except:
            return {'rmse': None, 'mape': None, 'mae': None, 'error': 'Could not convert inputs to numpy arrays'}
        
        # Handle NaN values
        valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[valid_mask]
        predicted = predicted[valid_mask]
        
        # If no valid data points remain, return None for all metrics
        if len(actual) == 0 or len(predicted) == 0:
            return {'rmse': None, 'mape': None, 'mae': None, 'error': 'No valid data points after filtering NaNs'}
        
        # Ensure arrays are same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # Handle negative predictions
        predicted = np.maximum(predicted, 0)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Calculate MAE
        mae = np.mean(np.abs(actual - predicted))
        
        # Calculate MAPE - avoid division by zero
        nonzero_idx = actual != 0
        if np.sum(nonzero_idx) > 0:
            mape = np.mean(np.abs((actual[nonzero_idx] - predicted[nonzero_idx]) / actual[nonzero_idx])) * 100
        else:
            mape = None
        
        # Check for NaN or Infinity in results
        if np.isnan(rmse) or np.isinf(rmse):
            rmse = None
        if mape is not None and (np.isnan(mape) or np.isinf(mape)):
            mape = None
        if np.isnan(mae) or np.isinf(mae):
            mae = None
        
        return {'rmse': rmse, 'mape': mape, 'mae': mae}
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {'rmse': None, 'mape': None, 'mae': None, 'error': str(e)}

def get_all_model_improvements(skus=None, models=None):
    """
    Get improvement metrics for all SKU-model combinations
    
    Parameters:
    -----------
    skus : list, optional
        List of SKUs to include, defaults to all
    models : list, optional
        List of models to include, defaults to all
    
    Returns:
    --------
    dict
        Dictionary mapping (sku, model_type) to improvement metrics
    """
    # Connect to database
    conn = sqlite3.connect('data/supply_chain.db')
    cursor = conn.cursor()
    
    # Get all SKU-model combinations
    if skus and models:
        # Filter by both SKUs and models
        placeholders = ','.join(['?'] * len(skus))
        model_placeholders = ','.join(['?'] * len(models))
        cursor.execute(
            f"SELECT DISTINCT sku, model_type FROM model_parameter_cache WHERE sku IN ({placeholders}) AND model_type IN ({model_placeholders})",
            skus + models
        )
    elif skus:
        # Filter by SKUs only
        placeholders = ','.join(['?'] * len(skus))
        cursor.execute(
            f"SELECT DISTINCT sku, model_type FROM model_parameter_cache WHERE sku IN ({placeholders})",
            skus
        )
    elif models:
        # Filter by models only
        placeholders = ','.join(['?'] * len(models))
        cursor.execute(
            f"SELECT DISTINCT sku, model_type FROM model_parameter_cache WHERE model_type IN ({placeholders})",
            models
        )
    else:
        # Get all combinations
        cursor.execute("SELECT DISTINCT sku, model_type FROM model_parameter_cache")
    
    combinations = cursor.fetchall()
    conn.close()
    
    # Calculate improvements for each combination
    improvements = {}
    for sku, model_type in combinations:
        improvements[(sku, model_type)] = calculate_model_improvement(sku, model_type)
    
    return improvements