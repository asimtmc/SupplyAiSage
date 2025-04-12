import pandas as pd
import numpy as np
import traceback
import time
import uuid
import concurrent.futures
import threading
import json
import logging
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import database functions for parameter caching
from utils.database import save_model_parameters, get_model_parameters, get_parameters_update_required, ModelParameterCache, SessionFactory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('parameter_optimizer')

# Global dictionary to track optimization tasks
_active_optimization_tasks = {}

# Maximum number of concurrent optimization tasks
MAX_CONCURRENT_TASKS = 6

# Background task management
def get_active_tasks_count():
    """Returns the number of currently active optimization tasks"""
    return len(_active_optimization_tasks)

def cleanup_stale_tasks():
    """Remove stale tasks that have been running for too long"""
    current_time = datetime.now()
    stale_keys = []

    for task_key, task_info in _active_optimization_tasks.items():
        # If task has been running for more than 10 minutes, consider it stale
        time_diff = current_time - task_info['start_time'] 
        if time_diff.total_seconds() > 600:  # 10 minutes
            stale_keys.append(task_key)

    # Remove stale tasks
    for key in stale_keys:
        logger.warning(f"Removing stale optimization task: {key}")
        del _active_optimization_tasks[key]

    return len(stale_keys)

def calculate_metrics(y_true, y_pred):
    """Calculate forecast error metrics"""
    try:
        metrics = {}

        # Ensure numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Check for invalid values
        if np.isnan(y_true).any() or np.isnan(y_pred).any() or np.isinf(y_true).any() or np.isinf(y_pred).any():
            logger.warning("Input data for metrics calculation contains NaN or Inf values")
            return {
                'mae': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf'),
                'weighted_error': float('inf')
            }

        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(y_true, y_pred)

        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

        # Mean Absolute Percentage Error, handling zeros
        mask = y_true > 0
        if mask.sum() > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.nan

        # Custom weighted error (for intermittent demand)
        mape_component = 0 if np.isnan(metrics['mape']) else metrics['mape'] * 0.01
        metrics['weighted_error'] = metrics['rmse'] * 0.4 + metrics['mae'] * 0.3 + mape_component * 0.3

        return metrics
    except Exception as e:
        logger.error(f"Error in calculate_metrics: {str(e)}")
        return {
            'mae': float('inf'),
            'rmse': float('inf'),
            'mape': float('inf'),
            'weighted_error': float('inf')
        }

def objective_function_arima(params, train_data, val_data):
    """
    Objective function for ARIMA parameter optimization

    Parameters:
    -----------
    params : dict
        Dictionary of parameters to optimize: p, d, q
    train_data : pandas.Series
        Training data
    val_data : pandas.Series
        Validation data

    Returns:
    --------
    float
        Error metric to minimize
    """
    try:
        import pandas as pd
        from statsmodels.tsa.arima.model import ARIMA

        # Extract parameters
        p = params['p']
        d = params['d']
        q = params['q']

        # Make sure data is properly formatted for ARIMA
        if not isinstance(train_data, pd.Series):
            logger.warning("Converting train_data to Series for ARIMA")
            if hasattr(train_data, 'values'):
                train_data = pd.Series(train_data.values)
            else:
                train_data = pd.Series(train_data)

        if not isinstance(val_data, pd.Series):
            logger.warning("Converting val_data to Series for ARIMA")
            if hasattr(val_data, 'values'):
                val_data = pd.Series(val_data.values)
            else:
                val_data = pd.Series(val_data)

        # Make sure we have sufficient data for the specified order
        min_data_needed = max(5, p + d + q + 1)  # At least p+d+q+1 points needed
        if len(train_data) < min_data_needed:
            logger.warning(f"Not enough training data for ARIMA order ({p},{d},{q}). Need at least {min_data_needed}")
            return float('inf')

        # Train model with detailed logging enabled
        logger.info(f"Fitting ARIMA model with order=(p={p}, d={d}, q={q})")
        model = ARIMA(train_data, order=(p, d, q))

        # Use a try block specifically for model fitting
        try:
            fitted_model = model.fit(disp=0, maxiter=100)  # Increase max iterations and disable convergence messages
            logger.info(f"ARIMA fitting complete. AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")
        except Exception as fit_error:
            logger.error(f"ARIMA fitting failed for order ({p},{d},{q}): {str(fit_error)}")
            return float('inf')

        # Make predictions on validation set
        try:
            predictions = fitted_model.forecast(steps=len(val_data))

            # Check for invalid predictions
            if np.isnan(predictions).any() or np.isinf(predictions).any():
                logger.warning(f"ARIMA model produced invalid predictions for order ({p},{d},{q})")
                return float('inf')

            # Calculate metrics
            metrics = calculate_metrics(val_data.values, predictions)

            # Return weighted error
            return metrics['weighted_error']

        except Exception as pred_error:
            logger.error(f"Error in ARIMA predictions for order ({p},{d},{q}): {str(pred_error)}")
            return float('inf')

    except Exception as e:
        logger.error(f"Error in ARIMA objective function: {str(e)}")
        return float('inf')  # Return a very large error

def objective_function_prophet(params, train_data, val_data):
    """
    Objective function for Prophet parameter optimization

    Parameters:
    -----------
    params : dict
        Dictionary of parameters to optimize
    train_data : pandas.DataFrame
        Training data with 'ds' and 'y' columns
    val_data : pandas.DataFrame
        Validation data with 'ds' and 'y' columns

    Returns:
    --------
    float
        Error metric to minimize
    """
    try:
        from prophet import Prophet

        # Extract parameters
        changepoint_prior_scale = params['changepoint_prior_scale']
        seasonality_prior_scale = params['seasonality_prior_scale']
        seasonality_mode = params['seasonality_mode']

        # Create and train model with detailed logging
        logger.info(f"Fitting Prophet model with parameters: changepoint_prior_scale={changepoint_prior_scale}, " +
                   f"seasonality_prior_scale={seasonality_prior_scale}, seasonality_mode={seasonality_mode}")

        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode,
            interval_width=0.95,  # Add this to see prediction intervals
            mcmc_samples=0  # Using MAP estimation for speed
        )

        # Add yearly seasonality if enough data
        if len(train_data) >= 365:
            logger.info("Adding yearly seasonality with period=365, fourier_order=5")
            model.add_seasonality(name='yearly', period=365, fourier_order=5)

        model.fit(train_data)
        logger.info("Prophet model fitting complete")

        # Make predictions
        future = pd.DataFrame({'ds': val_data['ds']})
        forecast = model.predict(future)

        # Calculate metrics
        metrics = calculate_metrics(val_data['y'].values, forecast['yhat'].values)

        # Return weighted error
        return metrics['weighted_error']

    except Exception as e:
        logger.error(f"Error in Prophet objective function: {str(e)}")
        return float('inf')

def objective_function_ets(params, train_data, val_data):
    """
    Objective function for ETS parameter optimization

    Parameters:
    -----------
    params : dict
        Dictionary of parameters to optimize
    train_data : pandas.Series
        Training data
    val_data : pandas.Series
        Validation data

    Returns:
    --------
    float
        Error metric to minimize
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        # Extract parameters
        trend = params['trend']
        seasonal = params['seasonal']
        seasonal_periods = params['seasonal_periods']
        damped_trend = params['damped_trend']

        # Create and train model with detailed logging
        logger.info(f"Fitting ETS model with parameters: trend={trend}, seasonal={seasonal}, " +
                   f"seasonal_periods={seasonal_periods}, damped_trend={damped_trend}")

        if seasonal is not None:
            model = ExponentialSmoothing(
                train_data,
                trend=trend, 
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend
            )
            logger.info(f"ETS model with seasonality (period={seasonal_periods})")
        else:
            model = ExponentialSmoothing(
                train_data,
                trend=trend,
                damped_trend=damped_trend
            )
            logger.info("ETS model without seasonality")

        fitted_model = model.fit(optimized=True)

        # Log the model parameters - safely handle non-float parameter values
        try:
            params = fitted_model.params
            param_items = []
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    param_items.append(f"{key}={value:.4f}")
                else:
                    param_items.append(f"{key}={value}")
            param_str = ", ".join(param_items)
            logger.info(f"ETS fitted parameters: {param_str}")
            logger.info(f"ETS model fit statistics - AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")
        except Exception as e:
            logger.warning(f"Could not format ETS parameters: {str(e)}")

        # Make predictions
        predictions = fitted_model.forecast(steps=len(val_data))

        # Calculate metrics
        metrics = calculate_metrics(val_data.values, predictions)

        # Return weighted error
        return metrics['weighted_error']

    except Exception as e:
        logger.error(f"Error in ETS objective function: {str(e)}")
        return float('inf')

def objective_function_theta(params, train_data, val_data):
    """
    Objective function for Theta method parameter optimization

    Parameters:
    -----------
    params : dict
        Dictionary of parameters to optimize (theta value, deseasonalize flag)
    train_data : pandas.Series
        Training data
    val_data : pandas.Series
        Validation data

    Returns:
    --------
    float
        Error metric to minimize
    """
    try:
        import pandas as pd
        from statsmodels.tsa.forecasting.theta import ThetaModel

        # Extract parameters
        # Note: statsmodels ThetaModel has a fixed theta=2.0, it's not configurable
        deseasonalize = params.get('deseasonalize', True)
        period = params.get('period', 12)
        method = params.get('method', 'auto')

        # Log parameters
        logger.info(f"Fitting Theta model with parameters: deseasonalize={deseasonalize}, period={period}, method={method}")

        # Make sure data is clean
        if train_data.isna().any() or np.isinf(train_data.values).any():
            logger.warning("Training data contains NaN or Inf values, cleaning...")
            train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()

        # Check if we have enough data points
        if len(train_data) < 4:
            logger.warning(f"Not enough training data for Theta model: {len(train_data)} points")
            return float('inf')

        # Fit the model
        try:
            # Use statsmodels' implementation - ThetaModel doesn't accept theta parameter
            model = ThetaModel(
                train_data,
                period=period if deseasonalize else None,
                deseasonalize=deseasonalize,
                use_test=False,  # Don't use test data
                method=method   # Use the specified method
            )

            # Fit and get forecasts - don't pass theta to fit()
            results = model.fit()
            predictions = results.forecast(steps=len(val_data))

            # Calculate accuracy metrics
            metrics = calculate_metrics(val_data.values, predictions)
            logger.info(f"Theta model fit completed successfully (statsmodels uses fixed theta=2)")

            return metrics['weighted_error']

        except Exception as model_error:
            logger.error(f"Error in Theta model fitting: {str(model_error)}")
            return float('inf')

    except Exception as e:
        logger.error(f"Error in Theta objective function: {str(e)}")
        return float('inf')

def optimize_arima_parameters(train_data, val_data, n_trials=30):
    """
    Optimize ARIMA parameters using grid search

    Parameters:
    -----------
    train_data : pandas.Series
        Training data
    val_data : pandas.Series
        Validation data
    n_trials : int, optional
        Number of parameter combinations to try

    Returns:
    --------
    dict
        Optimized parameters and score
    """
    import pandas as pd

    # Initial validation
    if len(train_data) < 5 or len(val_data) < 1:
        logger.warning(f"Not enough data for ARIMA optimization: train={len(train_data)}, val={len(val_data)}")
        return {'parameters': {'p': 1, 'd': 1, 'q': 0}, 'score': float('inf')}

    # Ensure the data is clean (no NaN, Inf)
    if train_data.isna().any() or np.isinf(train_data.values).any():
        logger.warning("Training data contains NaN or Inf values, cleaning...")
        train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()

    if val_data.isna().any() or np.isinf(val_data.values).any():
        logger.warning("Validation data contains NaN or Inf values, cleaning...")
        val_data = val_data.replace([np.inf, -np.inf], np.nan).dropna()

    # Check again after cleaning
    if len(train_data) < 5 or len(val_data) < 1:
        logger.warning(f"Not enough data after cleaning: train={len(train_data)}, val={len(val_data)}")
        return {'parameters': {'p': 1, 'd': 1, 'q': 0}, 'score': float('inf')}

    best_params = {'p': 1, 'd': 1, 'q': 0}  # Default values
    best_score = float('inf')

    # Grid of parameters to try - prioritize simpler models
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]

    # Log optimization parameters
    logger.info(f"ARIMA optimization starting with {len(p_values) * len(d_values) * len(q_values)} combinations")
    logger.info(f"Training data: {len(train_data)} points, Validation data: {len(val_data)} points")

    # Try stationarity test to guess good d value
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(train_data)
        if result[1] > 0.05:  # Not stationary
            d_values = [1, 0]  # Prioritize d=1
            logger.info(f"Data is not stationary (p-value={result[1]:.4f}), prioritizing d=1")
        else:
            d_values = [0, 1]  # Prioritize d=0
            logger.info(f"Data is stationary (p-value={result[1]:.4f}), prioritizing d=0")
    except Exception as e:
        logger.warning(f"Stationarity test failed: {str(e)}")

    # Try combinations - adding a counter
    tried_combinations = 0
    successful_fits = 0

    for p in p_values:
        for d in d_values:
            for q in q_values:
                if p + d + q > 3:
                    continue  # Skip overly complex models (reduced from 4 to 3)

                tried_combinations += 1

                # Log current attempt
                logger.info(f"Trying ARIMA({p},{d},{q}) - combination {tried_combinations}")

                params = {'p': p, 'd': d, 'q': q}
                score = objective_function_arima(params, train_data, val_data)

                # If score is not infinite, it's a successful fit
                if score < float('inf'):
                    successful_fits += 1

                if score < best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"New best model: ARIMA({p},{d},{q}) with score {score:.4f}")

    # Log optimization summary
    logger.info(f"ARIMA optimization complete. Tried {tried_combinations} combinations, {successful_fits} successful fits")
    logger.info(f"Best model: ARIMA({best_params['p']},{best_params['d']},{best_params['q']}) with score {best_score:.4f}")

    # If we couldn't find any working model, try a simpler approach
    if best_score == float('inf'):
        logger.warning("No suitable ARIMA model found, trying simple first difference model")
        best_params = {'p': 1, 'd': 1, 'q': 0}  # Simple AR(1) with first differencing

    return {'parameters': best_params, 'score': best_score}

def optimize_prophet_parameters(train_data, val_data, n_trials=20):
    """
    Optimize Prophet parameters

    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training data with 'ds' and 'y' columns
    val_data : pandas.DataFrame
        Validation data with 'ds' and 'y' columns
    n_trials : int, optional
        Number of parameter combinations to try

    Returns:
    --------
    dict
        Optimized parameters and score
    """
    # Try to use Optuna if available
    try:
        import optuna

        def objective(trial):
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
            }
            return objective_function_prophet(params, train_data, val_data)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_score = study.best_value

    except ImportError:
        # Fallback to grid search if Optuna is not available
        best_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10,
            'seasonality_mode': 'additive'
        }
        best_score = float('inf')

        # Grid of parameters to try
        changepoint_prior_scales = [0.001, 0.01, 0.05, 0.1, 0.5]
        seasonality_prior_scales = [0.01, 0.1, 1.0, 10.0]
        seasonality_modes = ['additive', 'multiplicative']

        # Try combinations
        for cps in changepoint_prior_scales:
            for sps in seasonality_prior_scales:
                for sm in seasonality_modes:
                    params = {
                        'changepoint_prior_scale': cps,
                        'seasonality_prior_scale': sps,
                        'seasonality_mode': sm
                    }
                    score = objective_function_prophet(params, train_data, val_data)

                    if score < best_score:
                        best_score = score
                        best_params = params

    return {'parameters': best_params, 'score': best_score}

def optimize_ets_parameters(train_data, val_data, n_trials=15):
    """
    Optimize ETS parameters

    Parameters:
    -----------
    train_data : pandas.Series
        Training data
    val_data : pandas.Series
        Validation data
    n_trials : int, optional
        Number of parameter combinations to try

    Returns:
    --------
    dict
        Optimized parameters and score
    """
    best_params = {
        'trend': 'add',
        'seasonal': None,
        'seasonal_periods': 1,
        'damped_trend': False
    }
    best_score = float('inf')

    # Grid of parameters to try
    trend_types = ['add', 'mul', None]
    seasonal_types = ['add', 'mul', None]

    # Determine possible seasonal periods
    if len(train_data) >= 24:  # At least 2 years of monthly data
        seasonal_periods_list = [4, 12]  # Quarterly and yearly
    elif len(train_data) >= 14:  # At least 14 days of daily data
        seasonal_periods_list = [7]  # Weekly
    else:
        seasonal_periods_list = [1]  # No seasonality

    # Try combinations
    for trend in trend_types:
        for seasonal in seasonal_types:
            for seasonal_periods in seasonal_periods_list:
                for damped_trend in [True, False]:
                    # Skip invalid combinations
                    if seasonal is not None and seasonal_periods == 1:
                        continue
                    if trend is None and damped_trend:
                        continue

                    params = {
                        'trend': trend,
                        'seasonal': seasonal,
                        'seasonal_periods': seasonal_periods,
                        'damped_trend': damped_trend
                    }

                    score = objective_function_ets(params, train_data, val_data)

                    if score < best_score:
                        best_score = score
                        best_params = params

    return {'parameters': best_params, 'score': best_score}

def optimize_theta_parameters(train_data, val_data, n_trials=10):
    """
    Optimize Theta method parameters

    Parameters:
    -----------
    train_data : pandas.Series
        Training data
    val_data : pandas.Series
        Validation data
    n_trials : int, optional
        Number of parameter combinations to try

    Returns:
    --------
    dict
        Optimized parameters and score
    """
    import pandas as pd

    # Make sure data is clean
    if train_data.isna().any() or np.isinf(train_data.values).any():
        logger.warning("Training data contains NaN or Inf values, cleaning...")
        train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()

    if val_data.isna().any() or np.isinf(val_data.values).any():
        logger.warning("Validation data contains NaN or Inf values, cleaning...")
        val_data = val_data.replace([np.inf, -np.inf], np.nan).dropna()

    # Check if we have enough data
    if len(train_data) < 4:
        logger.warning(f"Not enough data for Theta optimization: {len(train_data)} points")
        return {'parameters': {'deseasonalize': True, 'period': 12, 'method': 'auto'}, 'score': float('inf')}

    best_params = {'deseasonalize': True, 'period': 12, 'method': 'auto'}
    best_score = float('inf')

    # Parameters to try - Note: statsmodels ThetaModel has a fixed theta=2
    deseasonalize_options = [True, False]
    method_options = ['auto', 'additive', 'multiplicative']
    period_values = [4, 12] if len(train_data) >= 24 else [min(len(train_data) // 2, 12)]

    # Log optimization parameters
    logger.info(f"Theta optimization starting with {len(deseasonalize_options) * len(method_options) * len(period_values)} combinations")
    logger.info(f"Training data: {len(train_data)} points, Validation data: {len(val_data)} points")

    # Try combinations
    tried_combinations = 0
    successful_fits = 0

    for deseasonalize in deseasonalize_options:
        for method in method_options:
            for period in period_values:
                tried_combinations += 1

                # Skip unnecessary combinations
                if not deseasonalize and period != period_values[0]:
                    continue

                # Log current attempt
                logger.info(f"Trying Theta model with deseasonalize={deseasonalize}, period={period}, method={method} - combination {tried_combinations}")

                params = {
                    'deseasonalize': deseasonalize,
                    'period': period,
                    'method': method
                }

                score = objective_function_theta(params, train_data, val_data)

                # If score is not infinite, it's a successful fit
                if score < float('inf'):
                    successful_fits += 1

                if score < best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"New best Theta model: deseasonalize={deseasonalize}, period={period}, method={method} with score {score:.4f}")

    # Log optimization summary
    logger.info(f"Theta optimization complete. Tried {tried_combinations} combinations, {successful_fits} successful fits")

    if best_score == float('inf'):
        logger.warning("No suitable Theta model found, using default parameters")
        best_params = {'deseasonalize': True, 'period': 12, 'method': 'auto'}

    return {'parameters': best_params, 'score': best_score}

def run_optimization_task(sku, model_type, data, cross_validation=True, n_trials=None, progress_callback=None):
    """
    Run parameter optimization for a specific SKU and model

    Parameters:
    -----------
    sku : str
        SKU identifier
    model_type : str
        Type of forecasting model
    data : pandas.DataFrame
        Time series data with date and quantity columns
    cross_validation : bool, optional
        Whether to use cross-validation for optimization
    n_trials : int, optional
        Number of parameter combinations to try
    progress_callback : function, optional
        Callback function for progress reporting

    Returns:
    --------
    dict
        Optimized parameters and score
    """
    try:
        # Prepare data
        data = data.copy()

        # Filter data for this specific SKU if provided
        if 'sku' in data.columns:
            sku_data = data[data['sku'] == sku].copy()
            if len(sku_data) > 0:
                data = sku_data
                if progress_callback:
                    progress_callback(sku, model_type, f"Using {len(data)} data points specific to SKU {sku}")
            else:
                if progress_callback:
                    progress_callback(sku, model_type, f"No data found for SKU {sku} in dataset", level="warning")

        data.sort_values('date', inplace=True)

        # Set default number of trials based on model type
        if n_trials is None:
            if model_type == 'arima':
                n_trials = 30
            elif model_type == 'prophet':
                n_trials = 20
            elif model_type == 'ets':
                n_trials = 15
            else:
                n_trials = 10

        # Cross-validation approach
        if cross_validation and len(data) >= 12:
            # Use time series cross-validation with expanding window
            from sklearn.model_selection import TimeSeriesSplit

            # Determine number of splits based on data size
            n_splits = min(5, max(3, len(data) // 6))

            tscv = TimeSeriesSplit(n_splits=n_splits)

            # Maps for collecting parameter votes
            parameters_scores = []

            for train_idx, val_idx in tscv.split(data):
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]

                # Optimize for specific model type
                if model_type == 'arima':
                    train_series = train_data.set_index('date')['quantity']
                    val_series = val_data.set_index('date')['quantity']
                    fold_result = optimize_arima_parameters(train_series, val_series, n_trials=max(10, n_trials // n_splits))

                elif model_type == 'prophet':
                    # Format data for Prophet
                    train_prophet = pd.DataFrame({
                        'ds': train_data['date'],
                        'y': train_data['quantity']
                    })
                    val_prophet = pd.DataFrame({
                        'ds': val_data['date'],
                        'y': val_data['quantity']
                    })
                    fold_result = optimize_prophet_parameters(train_prophet, val_prophet, n_trials=max(8, n_trials // n_splits))

                elif model_type == 'ets':
                    train_series = train_data.set_index('date')['quantity']
                    val_series = val_data.set_index('date')['quantity']
                    fold_result = optimize_ets_parameters(train_series, val_series, n_trials=max(6, n_trials // n_splits))

                elif model_type == 'theta':
                    train_series = train_data.set_index('date')['quantity']
                    val_series = val_data.set_index('date')['quantity']
                    fold_result = optimize_theta_parameters(train_series, val_series, n_trials=max(5, n_trials // n_splits))

                else:
                    # Default to ARIMA for unknown model types
                    train_series = train_data.set_index('date')['quantity']
                    val_series = val_data.set_index('date')['quantity']
                    fold_result = optimize_arima_parameters(train_series, val_series, n_trials=max(10, n_trials // n_splits))

                parameters_scores.append(fold_result)

                if progress_callback:
                    progress_callback(sku, model_type, f"Completed CV fold {len(parameters_scores)}/{n_splits}")

            # Select best parameters based on average score
            best_score = float('inf')
            best_params = None

            for result in parameters_scores:
                if result['score'] < best_score:
                    best_score = result['score']
                    best_params = result['parameters']

            final_result = {'parameters': best_params, 'score': best_score}

        else:
            # Simple train-validation split for smaller datasets
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            val_data = data.iloc[split_idx:]

            # Optimize for specific model type
            if model_type == 'arima':
                train_series = train_data.set_index('date')['quantity']
                val_series = val_data.set_index('date')['quantity']
                final_result = optimize_arima_parameters(train_series, val_series, n_trials=n_trials)

            elif model_type == 'prophet':
                # Format data for Prophet
                train_prophet = pd.DataFrame({
                    'ds': train_data['date'],
                    'y': train_data['quantity']
                })
                val_prophet = pd.DataFrame({
                    'ds': val_data['date'],
                    'y': val_data['quantity']
                })
                final_result = optimize_prophet_parameters(train_prophet, val_prophet, n_trials=n_trials)

            elif model_type == 'ets':
                train_series = train_data.set_index('date')['quantity']
                val_series = val_data.set_index('date')['quantity']
                final_result = optimize_ets_parameters(train_series, val_series, n_trials=n_trials)

            elif model_type == 'theta':
                train_series = train_data.set_index('date')['quantity']
                val_series = val_data.set_index('date')['quantity']
                final_result = optimize_theta_parameters(train_series, val_series, n_trials=n_trials)

            else:
                # Default to ARIMA for unknown model types
                train_series = train_data.set_index('date')['quantity']
                val_series = val_data.set_index('date')['quantity']
                final_result = optimize_arima_parameters(train_series, val_series, n_trials=n_trials)

        # Save parameters to database with detailed info and error handling
        try:
            save_success = save_model_parameters(
                sku=sku,
                model_type=model_type,
                parameters=final_result['parameters'],
                best_score=final_result['score'],
                tuning_iterations=n_trials
            )

            if progress_callback:
                if save_success:
                    progress_callback(sku, model_type, f"Parameters successfully saved to database", level='success')
                else:
                    progress_callback(sku, model_type, f"Failed to save parameters to database", level='warning')
        except Exception as db_error:
            if progress_callback:
                progress_callback(sku, model_type, f"Error saving to database: {str(db_error)}", level='error')
            logger.error(f"Database error saving parameters for {sku}_{model_type}: {str(db_error)}")

        if progress_callback:
            progress_callback(sku, model_type, f"Optimization complete, score: {final_result['score']:.4f}")

        return final_result

    except Exception as e:
        logger.error(f"Error optimizing parameters for {sku}, {model_type}: {str(e)}")
        logger.error(traceback.format_exc())

        if progress_callback:
            progress_callback(sku, model_type, f"Optimization failed: {str(e)}", level='error')

        return {'parameters': None, 'score': float('inf'), 'error': str(e)}

    finally:
        # Remove from active tasks
        task_key = f"{sku}_{model_type}"
        if task_key in _active_optimization_tasks:
            del _active_optimization_tasks[task_key]

def optimize_parameters_async(sku, model_type, data, cross_validation=True, n_trials=None, progress_callback=None, priority=False):
    """
    Start parameter optimization in a background thread with concurrency control

    Parameters:
    -----------
    sku : str
        SKU identifier
    model_type : str
        Type of forecasting model
    data : pandas.DataFrame
        Time series data with date and quantity columns
    cross_validation : bool, optional
        Whether to use cross-validation for optimization
    n_trials : int, optional
        Number of parameter combinations to try
    progress_callback : function, optional
        Callback function for progress reporting
    priority : bool, optional
        Whether this task has priority and should run regardless of concurrency limits

    Returns:
    --------
    bool
        True if optimization started, False otherwise
    """
    task_key = f"{sku}_{model_type}"

    # Check if task is already running
    if task_key in _active_optimization_tasks:
        if progress_callback:
            progress_callback(sku, model_type, "Optimization already in progress", level='warning')
        return False

    # Clean up any stale tasks
    cleanup_stale_tasks()

    # Check concurrency limits if not a priority task
    if not priority and get_active_tasks_count() >= MAX_CONCURRENT_TASKS:
        if progress_callback:
            progress_callback(sku, model_type, 
                             f"Optimization skipped - maximum concurrent tasks ({MAX_CONCURRENT_TASKS}) reached", 
                             level='warning')
        return False

    # Create thread for optimization
    thread = threading.Thread(
        target=run_optimization_task,
        args=(sku, model_type, data, cross_validation, n_trials, progress_callback),
        daemon=True
    )

    # Store thread in active tasks
    _active_optimization_tasks[task_key] = {
        'thread': thread,
        'start_time': datetime.now(),
        'status': 'running',
        'priority': priority
    }

    # Start the thread
    thread.start()

    if progress_callback:
        progress_callback(sku, model_type, 
                         f"Optimization started in background (Active tasks: {get_active_tasks_count()}/{MAX_CONCURRENT_TASKS})", 
                         level='info')

    return True

def get_optimization_status(sku=None, model_type=None):
    """
    Get status of optimization tasks

    Parameters:
    -----------
    sku : str, optional
        SKU identifier to filter tasks
    model_type : str, optional
        Model type to filter tasks

    Returns:
    --------
    dict
        Status of optimization tasks
    """
    result = {}

    for task_key, task_info in _active_optimization_tasks.items():
        task_sku, task_model = task_key.split('_', 1)

        # Apply filters
        if sku is not None and task_sku != sku:
            continue
        if model_type is not None and task_model != model_type:
            continue

        # Calculate runtime
        runtime = (datetime.now() - task_info['start_time']).total_seconds()

        result[task_key] = {
            'sku': task_sku,
            'model_type': task_model,
            'status': task_info['status'],
            'runtime_seconds': runtime,
            'start_time': task_info['start_time'].strftime('%Y-%m-%d %H:%M:%S')
        }

    return result

def batch_optimize_parameters(sku_data_dict, model_types=None, max_workers=4, progress_callback=None):
    """
    Run optimization for multiple SKUs and models in parallel

    Parameters:
    -----------
    sku_data_dict : dict
        Dictionary mapping SKU to time series data
    model_types : list, optional
        List of model types to optimize
    max_workers : int, optional
        Maximum number of concurrent optimization tasks
    progress_callback : function, optional
        Callback function for progress reporting

    Returns:
    --------
    dict
        Results of optimization
    """
    if model_types is None:
        model_types = ['arima', 'prophet', 'ets']

    # Create list of all SKU-model combinations
    combinations = []
    for sku, data in sku_data_dict.items():
        for model_type in model_types:
            # Check if parameters need updating
            if not get_parameters_update_required(sku, model_type, days_threshold=7):
                if progress_callback:
                    progress_callback(sku, model_type, "Using cached parameters (less than 7 days old)", level='info')
                continue

            combinations.append((sku, model_type, data))

    if not combinations:
        if progress_callback:
            progress_callback(None, None, "No parameters need updating", level='info')
        return {}

    # Function to run optimization for a single combination
    def process_combination(combo):
        sku, model_type, data = combo
        return (sku, model_type, run_optimization_task(sku, model_type, data, progress_callback=progress_callback))

    # Run optimizations in parallel
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_combo = {executor.submit(process_combination, combo): combo for combo in combinations}

        for future in concurrent.futures.as_completed(future_to_combo):
            try:
                sku, model_type, result = future.result()
                results[f"{sku}_{model_type}"] = result
            except Exception as e:
                combo = future_to_combo[future]
                sku, model_type, _ = combo
                logger.error(f"Error optimizing {sku}_{model_type}: {str(e)}")

                if progress_callback:
                    progress_callback(sku, model_type, f"Optimization failed: {str(e)}", level='error')

    return results

def schedule_parameter_updates(sku_data_dict, model_types=None, days_threshold=7):
    """
    Schedule periodic parameter updates

    Parameters:
    -----------
    sku_data_dict : dict
        Dictionary mapping SKU to time series data function (callable that returns the data)
    model_types : list, optional
        List of model types to optimize
    days_threshold : int, optional
        Number of days after which parameters are considered stale

    Returns:
    --------
    dict
        Scheduled update information
    """
    if model_types is None:
        model_types = ['arima', 'prophet', 'ets']

    def update_parameters():
        while True:
            try:
                # Get latest data for each SKU
                current_data = {}
                for sku, data_func in sku_data_dict.items():
                    try:
                        current_data[sku] = data_func()
                    except Exception as e:
                        logger.error(f"Error getting data for {sku}: {str(e)}")

                # Run optimization for SKUs with stale parameters
                for sku, data in current_data.items():
                    for model_type in model_types:
                        if get_parameters_update_required(sku, model_type, days_threshold=days_threshold):
                            logger.info(f"Updating parameters for {sku}_{model_type}")
                            optimize_parameters_async(
                                sku=sku,
                                model_type=model_type,
                                data=data,
                                cross_validation=True
                            )

                # Wait 1 day before checking again
                time.sleep(86400)  # 24 hours

            except Exception as e:
                logger.error(f"Error in parameter update scheduler: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(3600)  # 1 hour

    # Start the scheduler in a background thread
    scheduler_thread = threading.Thread(target=update_parameters, daemon=True)
    scheduler_thread.start()

    return {
        'status': 'scheduled',
        'models': model_types,
        'days_threshold': days_threshold,
        'skus': list(sku_data_dict.keys())
    }

def save_model_parameters(sku, model_type, parameters, best_score=None, tuning_iterations=1):
    """
    Save optimized model parameters to the cache

    Parameters:
    -----------
    sku : str
        SKU identifier
    model_type : str
        Type of forecasting model (e.g., 'arima', 'prophet', 'xgboost')
    parameters : dict
        Dictionary of optimized parameters
    best_score : float, optional
        Best score achieved during tuning (lower is better)
    tuning_iterations : int, optional
        Number of tuning iterations performed

    Returns:
    --------
    bool
        True if save successful, False otherwise
    """
    try:
        session = SessionFactory()

        # Skip if parameters is None
        if parameters is None:
            print(f"Cannot save None parameters for {sku}, {model_type}")
            return False

        # Check if entry already exists
        existing = session.query(ModelParameterCache).filter(
            ModelParameterCache.sku == sku,
            ModelParameterCache.model_type == model_type
        ).first()

        # Convert parameters to JSON string
        import json
        if not isinstance(parameters, str):
            parameters_json = json.dumps(parameters)
        else:
            parameters_json = parameters

        # Log the parameters being saved
        print(f"Saving parameters for {sku}, {model_type}: {parameters_json}")

        if existing:
            # Update existing entry
            existing.parameters = parameters_json
            existing.last_updated = datetime.now()
            existing.tuning_iterations += tuning_iterations

            # Only update best_score if it's better (lower) than the existing one
            if best_score is not None:
                if existing.best_score is None or best_score < existing.best_score:
                    existing.best_score = best_score

            print(f"Updated existing parameters entry for {sku}, {model_type}")
        else:
            # Create new entry
            cache_id = str(uuid.uuid4())
            new_cache = ModelParameterCache(
                id=cache_id,
                sku=sku,
                model_type=model_type,
                parameters=parameters_json,
                last_updated=datetime.now(),
                tuning_iterations=tuning_iterations,
                best_score=best_score
            )
            session.add(new_cache)
            print(f"Created new parameters entry for {sku}, {model_type}")

        # Commit with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                session.commit()
                print(f"Successfully committed parameters for {sku}, {model_type}")
                return True
            except Exception as commit_error:
                if attempt < max_retries - 1:
                    print(f"Commit failed, retrying: {str(commit_error)}")
                    time.sleep(0.5)  # Wait before retrying
                    session.rollback()
                else:
                    raise commit_error

    except Exception as e:
        print(f"Error saving model parameters: {str(e)}")
        if session:
            session.rollback()
        return False
    finally:
        if session:
            session.close()

def get_model_parameters(sku, model_type):
    """
    Retrieve optimized model parameters from the cache

    Parameters:
    -----------
    sku : str
        SKU identifier
    model_type : str
        Type of forecasting model (e.g., 'arima', 'prophet', 'xgboost')

    Returns:
    --------
    dict or None
        Dictionary with parameters or None if not found
    """
    try:
        session = SessionFactory()

        # Query the database
        cache_entry = session.query(ModelParameterCache).filter(
            ModelParameterCache.sku == sku,
            ModelParameterCache.model_type == model_type
        ).first()

        if cache_entry:
            # Convert parameters from JSON string to dict
            import json

            # Check if parameters is a string and try to parse it
            if isinstance(cache_entry.parameters, str):
                try:
                    parameters = json.loads(cache_entry.parameters)
                except json.JSONDecodeError as e:
                    print(f"Error parsing parameters JSON for {sku}, {model_type}: {str(e)}")
                    print(f"Raw parameters: {cache_entry.parameters}")
                    parameters = cache_entry.parameters  # Keep as string if can't parse
            else:
                parameters = cache_entry.parameters

            # Print debug info
            print(f"Retrieved parameters for {sku}, {model_type}: {parameters}")

            # Check for valid parameter structure
            if isinstance(parameters, dict) or isinstance(parameters, str):
                return {
                    'parameters': parameters,
                    'last_updated': cache_entry.last_updated,
                    'best_score': cache_entry.best_score
                }
            else:
                print(f"Invalid parameter structure for {sku}, {model_type}")
                return None
        else:
            print(f"No parameters found for {sku}, {model_type}")
            return None

    except Exception as e:
        import traceback
        print(f"Error retrieving model parameters: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        if session:
            session.close()