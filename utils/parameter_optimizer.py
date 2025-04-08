import pandas as pd
import numpy as np
import json
import time
import threading
import concurrent.futures
from datetime import datetime, timedelta
import traceback
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import database functions for parameter caching
from utils.database import save_model_parameters, get_model_parameters, get_parameters_update_required

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('parameter_optimizer')

# Global dictionary to track optimization tasks
_active_optimization_tasks = {}

# Maximum number of concurrent optimization tasks
MAX_CONCURRENT_TASKS = 2

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
    metrics = {}

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
    metrics['weighted_error'] = metrics['rmse'] * 0.4 + metrics['mae'] * 0.3 + (metrics['mape'] * 0.01 if not np.isnan(metrics['mape']) else 0) * 0.3

    return metrics

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
        from statsmodels.tsa.arima.model import ARIMA

        # Extract parameters
        p = params['p']
        d = params['d']
        q = params['q']

        # Train model with detailed logging enabled
        logger.info(f"Fitting ARIMA model with order=(p={p}, d={d}, q={q})")
        model = ARIMA(train_data, order=(p, d, q))
        fitted_model = model.fit(disp=1, maxiter=50)  # Set disp=1 to show convergence messages
        logger.info(f"ARIMA fitting complete. AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")

        # Make predictions on validation set
        predictions = fitted_model.forecast(steps=len(val_data))

        # Calculate metrics
        metrics = calculate_metrics(val_data.values, predictions)

        # Return weighted error
        return metrics['weighted_error']

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

        # Log the model parameters
        params = fitted_model.params
        param_str = ", ".join([f"{key}={value:.4f}" for key, value in params.items()])
        logger.info(f"ETS fitted parameters: {param_str}")
        logger.info(f"ETS model fit statistics - AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")

        # Make predictions
        predictions = fitted_model.forecast(steps=len(val_data))

        # Calculate metrics
        metrics = calculate_metrics(val_data.values, predictions)

        # Return weighted error
        return metrics['weighted_error']

    except Exception as e:
        logger.error(f"Error in ETS objective function: {str(e)}")
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
    best_params = {'p': 1, 'd': 1, 'q': 0}  # Default values
    best_score = float('inf')

    # Grid of parameters to try - prioritize simpler models
    p_values = [0, 1, 2, 3]
    d_values = [0, 1, 2]
    q_values = [0, 1, 2, 3]

    # Try stationarity test to guess good d value
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(train_data)
        if result[1] > 0.05:  # Not stationary
            d_values = [1, 2, 0]  # Prioritize d=1
        else:
            d_values = [0, 1, 2]  # Prioritize d=0
    except:
        pass

    # Try combinations
    for p in p_values:
        for d in d_values:
            for q in q_values:
                if p + d + q > 4:
                    continue  # Skip overly complex models

                params = {'p': p, 'd': d, 'q': q}
                score = objective_function_arima(params, train_data, val_data)

                if score < best_score:
                    best_score = score
                    best_params = params

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