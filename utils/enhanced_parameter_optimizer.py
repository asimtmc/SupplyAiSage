import pandas as pd
import numpy as np
import os
import time
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for parameter ranges and validation
MIN_IMPROVEMENT_THRESHOLD = 0.10  # Require at least 10% improvement to accept new parameters
MAX_WORSENING_ALLOWED = 0.05     # Allow at most 5% worsening for parameters with other benefits
MAX_VALIDATION_ATTEMPTS = 3      # Number of validation attempts before falling back to defaults

# Caching for optimization results
MODEL_PARAMETER_CACHE = {}

def log_optimization_result(sku_id, model_type, parameters, metrics, baseline_metrics=None):
    """
    Log optimization result to file for offline analysis.
    
    Parameters:
    -----------
    sku_id : str
        SKU identifier
    model_type : str
        Model type (e.g., 'prophet', 'arima', 'ets', 'theta')
    parameters : dict
        Optimized parameters
    metrics : dict
        Performance metrics
    baseline_metrics : dict, optional
        Baseline metrics for comparison
    """
    try:
        log_dir = "logs/optimization"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{model_type}_optimization_log.jsonl")
        
        log_entry = {
            "timestamp": time.time(),
            "sku_id": sku_id,
            "model_type": model_type,
            "parameters": parameters,
            "metrics": metrics
        }
        
        if baseline_metrics:
            log_entry["baseline_metrics"] = baseline_metrics
            
            # Calculate improvement percentages
            if "mape" in metrics and "mape" in baseline_metrics:
                log_entry["mape_improvement_pct"] = (
                    (baseline_metrics["mape"] - metrics["mape"]) / baseline_metrics["mape"] * 100
                    if baseline_metrics["mape"] > 0 else 0
                )
            
            if "rmse" in metrics and "rmse" in baseline_metrics:
                log_entry["rmse_improvement_pct"] = (
                    (baseline_metrics["rmse"] - metrics["rmse"]) / baseline_metrics["rmse"] * 100
                    if baseline_metrics["rmse"] > 0 else 0
                )
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        return True
    except Exception as e:
        logger.error(f"Error logging optimization result: {str(e)}")
        return False

def validate_arima_parameters(params):
    """
    Validate ARIMA parameters, ensuring they're within reasonable ranges.
    
    Parameters:
    -----------
    params : dict
        Parameters to validate
        
    Returns:
    --------
    dict
        Validated parameters
    """
    valid_params = {}
    
    # Put reasonable limits on p, d, q values
    valid_params['p'] = min(max(params.get('p', 1), 0), 5)
    valid_params['d'] = min(max(params.get('d', 1), 0), 2)
    valid_params['q'] = min(max(params.get('q', 1), 0), 5)
    
    # Validate seasonal parameters
    valid_params['seasonal'] = params.get('seasonal', False)
    if valid_params['seasonal']:
        valid_params['m'] = min(max(params.get('m', 12), 2), 365)
    
    return valid_params

def validate_prophet_parameters(params):
    """
    Validate Prophet parameters, ensuring they're within reasonable ranges.
    
    Parameters:
    -----------
    params : dict
        Parameters to validate
        
    Returns:
    --------
    dict
        Validated parameters
    """
    valid_params = {}
    
    # Prior scales should be positive and not extreme
    valid_params['changepoint_prior_scale'] = min(max(params.get('changepoint_prior_scale', 0.05), 0.001), 0.5)
    valid_params['seasonality_prior_scale'] = min(max(params.get('seasonality_prior_scale', 10.0), 0.01), 100.0)
    
    # Seasonality mode should be valid
    if 'seasonality_mode' in params and params['seasonality_mode'] in ['additive', 'multiplicative']:
        valid_params['seasonality_mode'] = params['seasonality_mode']
    else:
        valid_params['seasonality_mode'] = 'additive'
    
    return valid_params

def validate_ets_parameters(params):
    """
    Validate ETS parameters, ensuring they're valid options.
    
    Parameters:
    -----------
    params : dict
        Parameters to validate
        
    Returns:
    --------
    dict
        Validated parameters
    """
    valid_params = {}
    
    # Validate trend type
    if 'trend' in params and params['trend'] in ['add', 'mul', None]:
        valid_params['trend'] = params['trend']
    else:
        valid_params['trend'] = 'add'  # Default to additive trend
    
    # Validate seasonal type
    if 'seasonal' in params and params['seasonal'] in ['add', 'mul', None]:
        valid_params['seasonal'] = params['seasonal']
    else:
        valid_params['seasonal'] = None  # Default to no seasonality
    
    # Validate seasonal periods
    if 'seasonal_periods' in params:
        valid_params['seasonal_periods'] = min(max(params.get('seasonal_periods', 1), 1), 52)
    else:
        valid_params['seasonal_periods'] = 1
    
    # Validate damped trend
    valid_params['damped_trend'] = params.get('damped_trend', False)
    
    return valid_params

def validate_theta_parameters(params):
    """
    Validate Theta parameters, ensuring they're within reasonable ranges.
    
    Parameters:
    -----------
    params : dict
        Parameters to validate
        
    Returns:
    --------
    dict
        Validated parameters
    """
    valid_params = {}
    
    # Validate deseasonalize flag (should be boolean)
    valid_params['deseasonalize'] = bool(params.get('deseasonalize', True))
    
    # Validate period (should be positive integer)
    valid_params['period'] = min(max(int(params.get('period', 12)), 1), 52)
    
    # Method can be 'auto', 'additive', or 'multiplicative'
    if 'method' in params and params['method'] in ['auto', 'additive', 'multiplicative']:
        valid_params['method'] = params['method']
    else:
        valid_params['method'] = 'auto'
    
    return valid_params

def calculate_performance_metrics(actual, predicted):
    """
    Calculate performance metrics for time series forecasting.
    
    Parameters:
    -----------
    actual : numpy.ndarray or pandas.Series
        Actual values
    predicted : numpy.ndarray or pandas.Series
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Ensure we have valid data
    if len(actual) == 0 or len(predicted) == 0:
        return {"mape": float('inf'), "rmse": float('inf')}
    
    # Handle NaN values
    valid_indices = ~(np.isnan(actual) | np.isnan(predicted))
    actual_valid = np.array(actual)[valid_indices]
    predicted_valid = np.array(predicted)[valid_indices]
    
    # If no valid points remain, return infinite error
    if len(actual_valid) == 0:
        return {"mape": float('inf'), "rmse": float('inf')}
    
    # Calculate RMSE
    rmse = mean_squared_error(actual_valid, predicted_valid, squared=False)
    
    # Calculate MAPE, handling zero values appropriately
    # We use a small epsilon to avoid division by zero
    epsilon = 1e-10
    mape = np.mean(np.abs((actual_valid - predicted_valid) / (np.abs(actual_valid) + epsilon))) * 100
    
    return {"mape": mape, "rmse": rmse}

def calculate_arima_baseline_metrics(train_series, val_series):
    """
    Calculate baseline metrics for ARIMA model using default parameters.
    
    Parameters:
    -----------
    train_series : pandas.Series
        Training time series data
    val_series : pandas.Series
        Validation time series data
    
    Returns:
    --------
    dict
        Baseline metrics
    """
    try:
        # Import ARIMA modules here to avoid import loop
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.arima.model import ARIMA
        
        # Default parameters for ARIMA
        default_params = {'p': 1, 'd': 1, 'q': 0}
        
        # Fit ARIMA with default parameters
        model = ARIMA(train_series, order=(default_params['p'], default_params['d'], default_params['q']))
        model_fit = model.fit()
        
        # Generate forecast for validation period
        forecast = model_fit.forecast(steps=len(val_series))
        
        # Calculate metrics
        metrics = calculate_performance_metrics(val_series, forecast)
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating ARIMA baseline metrics: {str(e)}")
        return {"mape": float('inf'), "rmse": float('inf')}

def calculate_prophet_baseline_metrics(train_df, val_df):
    """
    Calculate baseline metrics for Prophet model using default parameters.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data with 'ds' and 'y' columns
    val_df : pandas.DataFrame
        Validation data with 'ds' and 'y' columns
    
    Returns:
    --------
    dict
        Baseline metrics
    """
    try:
        # Import Prophet here to avoid import loop
        from prophet import Prophet
        
        # Default parameters for Prophet
        default_params = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}
        
        # Fit Prophet with default parameters
        model = Prophet(
            changepoint_prior_scale=default_params['changepoint_prior_scale'],
            seasonality_prior_scale=default_params['seasonality_prior_scale'],
            seasonality_mode=default_params['seasonality_mode']
        )
        model.fit(train_df)
        
        # Generate forecast for validation period
        future = model.make_future_dataframe(periods=len(val_df), freq='D')
        forecast = model.predict(future)
        
        # Extract forecast for validation period
        pred_df = forecast.tail(len(val_df))
        
        # Calculate metrics
        metrics = calculate_performance_metrics(val_df['y'], pred_df['yhat'])
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating Prophet baseline metrics: {str(e)}")
        return {"mape": float('inf'), "rmse": float('inf')}

def calculate_ets_baseline_metrics(train_series, val_series):
    """
    Calculate baseline metrics for ETS model using default parameters.
    
    Parameters:
    -----------
    train_series : pandas.Series
        Training time series data
    val_series : pandas.Series
        Validation time series data
    
    Returns:
    --------
    dict
        Baseline metrics
    """
    try:
        # Import statsmodels here to avoid import loop
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        
        # Default parameters for ETS
        default_params = {'trend': 'add', 'seasonal': None, 'seasonal_periods': 1, 'damped_trend': False}
        
        # Fit ETS with default parameters
        model = ETSModel(
            train_series,
            trend=default_params['trend'],
            seasonal=default_params['seasonal'],
            seasonal_periods=default_params['seasonal_periods'],
            damped_trend=default_params['damped_trend']
        )
        model_fit = model.fit(disp=False)
        
        # Generate forecast for validation period
        forecast = model_fit.forecast(steps=len(val_series))
        
        # Calculate metrics
        metrics = calculate_performance_metrics(val_series, forecast)
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating ETS baseline metrics: {str(e)}")
        return {"mape": float('inf'), "rmse": float('inf')}

def calculate_theta_baseline_metrics(train_series, val_series):
    """
    Calculate baseline metrics for Theta model using default parameters.
    
    Parameters:
    -----------
    train_series : pandas.Series
        Training time series data
    val_series : pandas.Series
        Validation time series data
    
    Returns:
    --------
    dict
        Baseline metrics
    """
    try:
        # Import statsmodels here to avoid import loop
        from statsmodels.tsa.forecasting.theta import ThetaModel
        
        # Default parameters for Theta
        default_params = {'deseasonalize': True}
        
        # Fit Theta with default parameters
        model = ThetaModel(
            train_series,
            deseasonalize=default_params['deseasonalize']
        )
        model_fit = model.fit()
        
        # Generate forecast for validation period
        forecast = model_fit.forecast(steps=len(val_series))
        
        # Calculate metrics
        metrics = calculate_performance_metrics(val_series, forecast)
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating Theta baseline metrics: {str(e)}")
        return {"mape": float('inf'), "rmse": float('inf')}

def optimize_arima_parameters_enhanced(train_series, val_series):
    """
    Optimize ARIMA parameters with robust validation.
    
    Parameters:
    -----------
    train_series : pandas.Series
        Training time series data
    val_series : pandas.Series
        Validation time series data
    
    Returns:
    --------
    dict
        Optimized parameters and metrics
    """
    try:
        import optuna
        from statsmodels.tsa.arima.model import ARIMA
        
        # Calculate baseline metrics with default parameters
        baseline_metrics = calculate_arima_baseline_metrics(train_series, val_series)
        logger.info(f"ARIMA baseline metrics: {baseline_metrics}")
        
        # Define objective function for Optuna
        def objective(trial):
            # Sample parameters from predefined ranges
            p = trial.suggest_int('p', 0, 5)
            d = trial.suggest_int('d', 0, 2)
            q = trial.suggest_int('q', 0, 5)
            
            # Have at least one non-zero term
            if p == 0 and d == 0 and q == 0:
                p = 1  # Force at least AR(1) if all zeros
            
            try:
                # Fit ARIMA model
                model = ARIMA(train_series, order=(p, d, q))
                model_fit = model.fit()
                
                # Generate forecast for validation period
                forecast = model_fit.forecast(steps=len(val_series))
                
                # Calculate metrics
                metrics = calculate_performance_metrics(val_series, forecast)
                
                return metrics['mape']  # Optimize for MAPE
            except Exception as e:
                logger.warning(f"Error in ARIMA optimization: {str(e)}")
                return float('inf')  # Penalize errors
        
        # Create and run Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)
        
        # Get best parameters
        best_params = study.best_params
        
        # Validate parameters
        validated_params = validate_arima_parameters(best_params)
        
        # Evaluate with best parameters
        try:
            model = ARIMA(train_series, order=(validated_params['p'], validated_params['d'], validated_params['q']))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(val_series))
            best_metrics = calculate_performance_metrics(val_series, forecast)
        except Exception:
            # If best parameters fail, fall back to default
            validated_params = {'p': 1, 'd': 1, 'q': 0}
            model = ARIMA(train_series, order=(validated_params['p'], validated_params['d'], validated_params['q']))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(val_series))
            best_metrics = calculate_performance_metrics(val_series, forecast)
        
        return {
            'parameters': validated_params,
            'metrics': best_metrics,
            'baseline_metrics': baseline_metrics
        }
    except Exception as e:
        logger.error(f"Error in ARIMA optimization: {str(e)}")
        # Return default parameters
        return {
            'parameters': {'p': 1, 'd': 1, 'q': 0},
            'metrics': {'mape': float('inf'), 'rmse': float('inf')},
            'baseline_metrics': {'mape': float('inf'), 'rmse': float('inf')}
        }

def optimize_prophet_parameters_enhanced(train_df, val_df):
    """
    Optimize Prophet parameters with robust validation.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data with 'ds' and 'y' columns
    val_df : pandas.DataFrame
        Validation data with 'ds' and 'y' columns
    
    Returns:
    --------
    dict
        Optimized parameters and metrics
    """
    try:
        import optuna
        from prophet import Prophet
        
        # Calculate baseline metrics with default parameters
        baseline_metrics = calculate_prophet_baseline_metrics(train_df, val_df)
        logger.info(f"Prophet baseline metrics: {baseline_metrics}")
        
        # Define objective function for Optuna
        def objective(trial):
            # Sample parameters from predefined ranges
            changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
            seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 100, log=True)
            seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
            
            try:
                # Fit Prophet model
                model = Prophet(
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    seasonality_mode=seasonality_mode
                )
                model.fit(train_df)
                
                # Generate forecast for validation period
                future = model.make_future_dataframe(periods=len(val_df), freq='D')
                forecast = model.predict(future)
                
                # Extract forecast for validation period
                pred_df = forecast.tail(len(val_df))
                
                # Calculate metrics
                metrics = calculate_performance_metrics(val_df['y'], pred_df['yhat'])
                
                return metrics['mape']  # Optimize for MAPE
            except Exception as e:
                logger.warning(f"Error in Prophet optimization: {str(e)}")
                return float('inf')  # Penalize errors
        
        # Create and run Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        # Get best parameters
        best_params = study.best_params
        
        # Validate parameters
        validated_params = validate_prophet_parameters(best_params)
        
        # Evaluate with best parameters
        try:
            model = Prophet(
                changepoint_prior_scale=validated_params['changepoint_prior_scale'],
                seasonality_prior_scale=validated_params['seasonality_prior_scale'],
                seasonality_mode=validated_params['seasonality_mode']
            )
            model.fit(train_df)
            
            future = model.make_future_dataframe(periods=len(val_df), freq='D')
            forecast = model.predict(future)
            
            pred_df = forecast.tail(len(val_df))
            best_metrics = calculate_performance_metrics(val_df['y'], pred_df['yhat'])
        except Exception:
            # If best parameters fail, fall back to default
            validated_params = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}
            model = Prophet(
                changepoint_prior_scale=validated_params['changepoint_prior_scale'],
                seasonality_prior_scale=validated_params['seasonality_prior_scale'],
                seasonality_mode=validated_params['seasonality_mode']
            )
            model.fit(train_df)
            
            future = model.make_future_dataframe(periods=len(val_df), freq='D')
            forecast = model.predict(future)
            
            pred_df = forecast.tail(len(val_df))
            best_metrics = calculate_performance_metrics(val_df['y'], pred_df['yhat'])
        
        return {
            'parameters': validated_params,
            'metrics': best_metrics,
            'baseline_metrics': baseline_metrics
        }
    except Exception as e:
        logger.error(f"Error in Prophet optimization: {str(e)}")
        # Return default parameters
        return {
            'parameters': {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            'metrics': {'mape': float('inf'), 'rmse': float('inf')},
            'baseline_metrics': {'mape': float('inf'), 'rmse': float('inf')}
        }

def optimize_ets_parameters_enhanced(train_series, val_series):
    """
    Optimize ETS parameters with robust validation.
    
    Parameters:
    -----------
    train_series : pandas.Series
        Training time series data
    val_series : pandas.Series
        Validation time series data
    
    Returns:
    --------
    dict
        Optimized parameters and metrics
    """
    try:
        import optuna
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        
        # Calculate baseline metrics with default parameters
        baseline_metrics = calculate_ets_baseline_metrics(train_series, val_series)
        logger.info(f"ETS baseline metrics: {baseline_metrics}")
        
        # Define objective function for Optuna
        def objective(trial):
            # Sample parameters from predefined ranges
            trend = trial.suggest_categorical('trend', ['add', 'mul', None])
            if trend is not None:
                damped_trend = trial.suggest_categorical('damped_trend', [True, False])
            else:
                damped_trend = False
                
            # For small datasets, be conservative with seasonality
            if len(train_series) < 24:
                seasonal = None
                seasonal_periods = 1
            else:
                seasonal = trial.suggest_categorical('seasonal', ['add', 'mul', None])
                if seasonal is not None:
                    seasonal_periods = trial.suggest_int('seasonal_periods', 4, min(12, len(train_series) // 3))
                else:
                    seasonal_periods = 1
            
            try:
                # Fit ETS model
                model = ETSModel(
                    train_series,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods,
                    damped_trend=damped_trend
                )
                model_fit = model.fit(disp=False)
                
                # Generate forecast for validation period
                forecast = model_fit.forecast(steps=len(val_series))
                
                # Calculate metrics
                metrics = calculate_performance_metrics(val_series, forecast)
                
                return metrics['mape']  # Optimize for MAPE
            except Exception as e:
                logger.warning(f"Error in ETS optimization: {str(e)}")
                return float('inf')  # Penalize errors
        
        # Create and run Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        # Get best parameters
        best_params = study.best_params
        
        # Validate parameters
        validated_params = validate_ets_parameters(best_params)
        
        # Evaluate with best parameters
        try:
            model = ETSModel(
                train_series,
                trend=validated_params['trend'],
                seasonal=validated_params['seasonal'],
                seasonal_periods=validated_params['seasonal_periods'],
                damped_trend=validated_params['damped_trend']
            )
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=len(val_series))
            best_metrics = calculate_performance_metrics(val_series, forecast)
        except Exception:
            # If best parameters fail, fall back to default
            validated_params = {'trend': 'add', 'seasonal': None, 'seasonal_periods': 1, 'damped_trend': False}
            model = ETSModel(
                train_series,
                trend=validated_params['trend'],
                seasonal=validated_params['seasonal'],
                seasonal_periods=validated_params['seasonal_periods'],
                damped_trend=validated_params['damped_trend']
            )
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=len(val_series))
            best_metrics = calculate_performance_metrics(val_series, forecast)
        
        return {
            'parameters': validated_params,
            'metrics': best_metrics,
            'baseline_metrics': baseline_metrics
        }
    except Exception as e:
        logger.error(f"Error in ETS optimization: {str(e)}")
        # Return default parameters
        return {
            'parameters': {'trend': 'add', 'seasonal': None, 'seasonal_periods': 1, 'damped_trend': False},
            'metrics': {'mape': float('inf'), 'rmse': float('inf')},
            'baseline_metrics': {'mape': float('inf'), 'rmse': float('inf')}
        }

def optimize_theta_parameters_enhanced(train_series, val_series):
    """
    Optimize Theta parameters with robust validation.
    
    Parameters:
    -----------
    train_series : pandas.Series
        Training time series data
    val_series : pandas.Series
        Validation time series data
    
    Returns:
    --------
    dict
        Optimized parameters and metrics
    """
    try:
        import optuna
        from statsmodels.tsa.forecasting.theta import ThetaModel
        
        # Calculate baseline metrics with default parameters
        baseline_metrics = calculate_theta_baseline_metrics(train_series, val_series)
        logger.info(f"Theta baseline metrics: {baseline_metrics}")
        
        # Define objective function for Optuna
        def objective(trial):
            # Sample parameters from predefined ranges
            deseasonalize = trial.suggest_categorical('deseasonalize', [True, False])
            period = trial.suggest_int('period', 4, min(12, len(train_series) // 3)) if deseasonalize else 12
            method = trial.suggest_categorical('method', ['auto', 'additive', 'multiplicative'])
            
            try:
                # Fit Theta model
                model = ThetaModel(
                    train_series,
                    deseasonalize=deseasonalize,
                    period=period,
                    method=method
                )
                model_fit = model.fit()
                
                # Generate forecast for validation period
                forecast = model_fit.forecast(steps=len(val_series))
                
                # Calculate metrics
                metrics = calculate_performance_metrics(val_series, forecast)
                
                return metrics['mape']  # Optimize for MAPE
            except Exception as e:
                logger.warning(f"Error in Theta optimization: {str(e)}")
                return float('inf')  # Penalize errors
        
        # Create and run Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        # Get best parameters
        best_params = study.best_params
        
        # Validate parameters
        validated_params = validate_theta_parameters(best_params)
        
        # Evaluate with best parameters
        try:
            model = ThetaModel(
                train_series,
                deseasonalize=validated_params['deseasonalize'],
                period=validated_params['period'],
                method=validated_params['method']
            )
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(val_series))
            best_metrics = calculate_performance_metrics(val_series, forecast)
        except Exception:
            # If best parameters fail, fall back to default
            validated_params = {'deseasonalize': True, 'period': 12, 'method': 'auto'}
            model = ThetaModel(
                train_series,
                deseasonalize=validated_params['deseasonalize'],
                period=validated_params['period'] if 'period' in validated_params else 12
            )
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(val_series))
            best_metrics = calculate_performance_metrics(val_series, forecast)
        
        return {
            'parameters': validated_params,
            'metrics': best_metrics,
            'baseline_metrics': baseline_metrics
        }
    except Exception as e:
        logger.error(f"Error in Theta optimization: {str(e)}")
        # Return default parameters
        return {
            'parameters': {'deseasonalize': True, 'period': 12, 'method': 'auto'},
            'metrics': {'mape': float('inf'), 'rmse': float('inf')},
            'baseline_metrics': {'mape': float('inf'), 'rmse': float('inf')}
        }

def verify_optimization_result(optimization_result, model_type, sku_id):
    """
    Verify optimization result and decide whether to use optimal or default parameters.
    
    Parameters:
    -----------
    optimization_result : dict
        Optimization result with parameters, metrics, and baseline metrics
    model_type : str
        Model type (e.g., 'prophet', 'arima', 'ets', 'theta')
    sku_id : str
        SKU identifier
    
    Returns:
    --------
    dict
        Verified optimization result
    """
    if not optimization_result or 'parameters' not in optimization_result:
        logger.warning(f"Invalid optimization result for {model_type} on {sku_id}")
        return get_default_parameters(model_type)
    
    parameters = optimization_result.get('parameters', {})
    metrics = optimization_result.get('metrics', {'mape': float('inf'), 'rmse': float('inf')})
    baseline_metrics = optimization_result.get('baseline_metrics', {'mape': float('inf'), 'rmse': float('inf')})
    
    # Skip verification if baseline metrics are not available
    if not baseline_metrics or baseline_metrics.get('mape', float('inf')) == float('inf'):
        return {'parameters': parameters, 'score': metrics.get('mape', float('inf'))}
    
    # Calculate improvement percentage
    baseline_mape = baseline_metrics.get('mape', float('inf'))
    optimized_mape = metrics.get('mape', float('inf'))
    
    if baseline_mape > 0 and optimized_mape > 0:
        improvement_pct = (baseline_mape - optimized_mape) / baseline_mape
    else:
        improvement_pct = -1  # Invalid improvement
    
    # Log the verification process
    logger.info(f"Verifying {model_type} optimization for {sku_id}")
    logger.info(f"Baseline MAPE: {baseline_mape:.2f}%, Optimized MAPE: {optimized_mape:.2f}%")
    logger.info(f"Improvement: {improvement_pct * 100:.2f}%")
    
    # Decision logic
    if improvement_pct >= MIN_IMPROVEMENT_THRESHOLD:
        # If significant improvement, use optimized parameters
        logger.info(f"Using optimized parameters for {model_type} on {sku_id} (improvement: {improvement_pct * 100:.2f}%)")
        return {'parameters': parameters, 'score': optimized_mape}
    elif -MAX_WORSENING_ALLOWED <= improvement_pct < MIN_IMPROVEMENT_THRESHOLD:
        # If slight worsening or insufficient improvement, use baseline parameters
        logger.info(f"Using default parameters for {model_type} on {sku_id} (insufficient improvement: {improvement_pct * 100:.2f}%)")
        return {'parameters': get_default_parameters(model_type)['parameters'], 'score': baseline_mape}
    else:
        # If significant worsening, definitely use baseline parameters
        logger.warning(f"Using default parameters for {model_type} on {sku_id} (significant worsening: {improvement_pct * 100:.2f}%)")
        return {'parameters': get_default_parameters(model_type)['parameters'], 'score': baseline_mape}

def get_default_parameters(model_type):
    """
    Get default parameters for a specific model type.
    
    Parameters:
    -----------
    model_type : str
        Model type (e.g., 'prophet', 'arima', 'ets', 'theta')
    
    Returns:
    --------
    dict
        Default parameters and a placeholder score
    """
    if model_type == 'arima' or model_type == 'auto_arima':
        return {'parameters': {'p': 1, 'd': 1, 'q': 0}, 'score': float('inf')}
    elif model_type == 'prophet':
        return {'parameters': {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}, 'score': float('inf')}
    elif model_type == 'ets':
        return {'parameters': {'trend': 'add', 'seasonal': None, 'seasonal_periods': 1, 'damped_trend': False}, 'score': float('inf')}
    elif model_type == 'theta':
        return {'parameters': {'deseasonalize': True, 'period': 12, 'method': 'auto'}, 'score': float('inf')}
    else:
        logger.warning(f"Unknown model type: {model_type}, returning empty parameters")
        return {'parameters': {}, 'score': float('inf')}