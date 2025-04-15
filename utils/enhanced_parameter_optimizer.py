import pandas as pd
import numpy as np
import logging
import traceback
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import original optimizer for base functions
from utils.parameter_optimizer import (
    calculate_metrics, objective_function_arima, objective_function_prophet,
    objective_function_ets, objective_function_theta
)

# Import database functions for parameter caching
from utils.database import save_model_parameters, get_model_parameters

# Set up logging
logger = logging.getLogger('enhanced_parameter_optimizer')

def optimize_arima_parameters_enhanced(train_data, val_data, n_trials=30):
    """
    Enhanced ARIMA parameters optimization with validation and sanity checks

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
    # Default parameters to use as a baseline
    default_params = {'p': 1, 'd': 1, 'q': 0}  # Default to AR(1) with first differencing

    # Initial validation
    if len(train_data) < 5 or len(val_data) < 1:
        logger.warning(f"Not enough data for ARIMA optimization: train={len(train_data)}, val={len(val_data)}")
        return {'parameters': default_params, 'score': float('inf')}

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
        return {'parameters': default_params, 'score': float('inf')}
    
    # Get baseline score with default parameters
    default_score = objective_function_arima(default_params, train_data, val_data)
    
    # If default parameters give infinity as score, try an even simpler model
    if default_score == float('inf'):
        simpler_default = {'p': 0, 'd': 1, 'q': 0}  # Simple random walk with drift
        default_score = objective_function_arima(simpler_default, train_data, val_data)
        
        if default_score == float('inf'):
            logger.warning("Even simpler ARIMA model failed. Data may be problematic.")
            return {'parameters': default_params, 'score': float('inf')}
        else:
            default_params = simpler_default
    
    logger.info(f"Baseline ARIMA score with default parameters: {default_score:.4f}")
    
    # Initialize best parameters and score with defaults
    best_params = default_params.copy()
    best_score = default_score

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

    # Try combinations
    tried_combinations = 0
    successful_fits = 0

    for p in p_values:
        for d in d_values:
            for q in q_values:
                if p + d + q > 3:
                    continue  # Skip overly complex models

                tried_combinations += 1

                # Log current attempt
                logger.info(f"Trying ARIMA({p},{d},{q}) - combination {tried_combinations}")

                params = {'p': p, 'd': d, 'q': q}
                score = objective_function_arima(params, train_data, val_data)

                # Only consider valid scores
                if score != float('inf') and not np.isnan(score) and score < 1000:
                    successful_fits += 1
                    
                    # Only use new parameters if they provide significant improvement (at least 5% better)
                    if score < best_score * 0.95:
                        best_score = score
                        best_params = params
                        logger.info(f"New best model: ARIMA({p},{d},{q}) with score {score:.4f}, improvement: {(default_score - score) / default_score * 100:.2f}%")

    # Log optimization summary
    logger.info(f"ARIMA optimization complete. Tried {tried_combinations} combinations, {successful_fits} successful fits")
    
    # Add sanity check - if optimization made things worse, revert to defaults
    if best_score > default_score * 1.5:  # If more than 50% worse
        logger.warning(f"Optimization resulted in worse score ({best_score:.4f} vs {default_score:.4f}). Reverting to default parameters.")
        best_params = default_params
        best_score = default_score

    return {'parameters': best_params, 'score': best_score}


def optimize_prophet_parameters_enhanced(train_data, val_data, n_trials=20):
    """
    Enhanced Prophet parameters optimization with validation and sanity checks

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
    # Default parameters to use as a baseline
    default_params = {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'additive'
    }
    
    # Get baseline score using default parameters
    default_score = objective_function_prophet(default_params, train_data, val_data)
    
    # If default parameters give infinity as score, there's an issue with the data
    if default_score == float('inf'):
        logger.warning("Default Prophet parameters resulted in infinite score. Data may be problematic.")
        return {'parameters': default_params, 'score': default_score}
    
    logger.info(f"Baseline Prophet score with default parameters: {default_score:.4f}")
    
    # Initialize best parameters and score with defaults
    best_params = default_params.copy()
    best_score = default_score
    
    # Try to use Optuna if available
    try:
        import optuna

        def objective(trial):
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
            }
            score = objective_function_prophet(params, train_data, val_data)
            
            # Verify the score is reasonable (not extreme or NaN)
            if score != float('inf') and not np.isnan(score) and score < 1000:
                return score
            else:
                # Return a high but not infinite score to allow optimization to continue
                return 1000.0

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        candidate_params = study.best_params
        candidate_score = study.best_value
        
        # Only use the new parameters if they provide significant improvement (at least 10% better)
        if candidate_score < best_score * 0.9 and candidate_score != float('inf') and not np.isnan(candidate_score):
            best_params = candidate_params
            best_score = candidate_score
            logger.info(f"Found better Prophet parameters, improvement: {(default_score - candidate_score) / default_score * 100:.2f}%")

    except ImportError:
        # Fallback to grid search if Optuna is not available
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

                    # Only consider valid scores
                    if score != float('inf') and not np.isnan(score) and score < 1000:
                        if score < best_score * 0.9:  # Must be at least 10% better
                            best_score = score
                            best_params = params
                            logger.info(f"Found better Prophet parameters: {params} with score {score:.4f}")

    # Add sanity check - if optimization made things significantly worse, revert to defaults
    if best_score > default_score * 5:  # If more than 5 times worse
        logger.warning(f"Optimization resulted in significantly worse score ({best_score:.4f} vs {default_score:.4f}). Reverting to default parameters.")
        best_params = default_params
        best_score = default_score

    logger.info(f"Final Prophet parameters: {best_params} with score {best_score:.4f}")
    return {'parameters': best_params, 'score': best_score}


def optimize_ets_parameters_enhanced(train_data, val_data, n_trials=15):
    """
    Enhanced ETS parameters optimization with validation and sanity checks

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
    # Default parameters to use as a baseline
    default_params = {
        'trend': 'add',
        'seasonal': None,
        'seasonal_periods': 1,
        'damped_trend': False
    }
    
    # Get baseline score using default parameters
    default_score = objective_function_ets(default_params, train_data, val_data)
    
    # If default parameters give infinity as score, try a simpler model
    if default_score == float('inf'):
        simpler_default = {
            'trend': None,
            'seasonal': None,
            'seasonal_periods': 1,
            'damped_trend': False
        }
        default_score = objective_function_ets(simpler_default, train_data, val_data)
        
        if default_score == float('inf'):
            logger.warning("Even simpler ETS model failed. Data may be problematic.")
            return {'parameters': default_params, 'score': float('inf')}
        else:
            default_params = simpler_default
    
    logger.info(f"Baseline ETS score with default parameters: {default_score:.4f}")
    
    # Initialize best parameters and score with defaults
    best_params = default_params.copy()
    best_score = default_score
    
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
    tried_combinations = 0
    successful_fits = 0
    
    for trend in trend_types:
        for seasonal in seasonal_types:
            for seasonal_periods in seasonal_periods_list:
                for damped_trend in [True, False]:
                    # Skip invalid combinations
                    if seasonal is not None and seasonal_periods == 1:
                        continue
                    if trend is None and damped_trend:
                        continue
                    
                    tried_combinations += 1

                    params = {
                        'trend': trend,
                        'seasonal': seasonal,
                        'seasonal_periods': seasonal_periods,
                        'damped_trend': damped_trend
                    }

                    score = objective_function_ets(params, train_data, val_data)

                    # Only consider valid scores
                    if score != float('inf') and not np.isnan(score) and score < 1000:
                        successful_fits += 1
                        
                        # Only use if significantly better (at least 5% improvement)
                        if score < best_score * 0.95:
                            best_score = score
                            best_params = params
                            logger.info(f"Found better ETS parameters: {params} with score {score:.4f}")

    # Log optimization summary
    logger.info(f"ETS optimization complete. Tried {tried_combinations} combinations, {successful_fits} successful fits")
    
    # Add sanity check - if optimization made things worse, revert to defaults
    if best_score > default_score * 2:  # If more than 2 times worse
        logger.warning(f"Optimization resulted in worse score ({best_score:.4f} vs {default_score:.4f}). Reverting to default parameters.")
        best_params = default_params
        best_score = default_score

    return {'parameters': best_params, 'score': best_score}


def optimize_theta_parameters_enhanced(train_data, val_data, n_trials=10):
    """
    Enhanced Theta method parameters optimization with validation and sanity checks

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

    # Default parameters
    default_params = {'deseasonalize': True, 'period': 12, 'method': 'auto'}
    
    # Get baseline score using default parameters
    default_score = objective_function_theta(default_params, train_data, val_data)
    
    # If default parameters give infinity as score, try a simpler model
    if default_score == float('inf'):
        simpler_default = {'deseasonalize': False, 'period': 1, 'method': 'auto'}
        default_score = objective_function_theta(simpler_default, train_data, val_data)
        
        if default_score == float('inf'):
            logger.warning("Even simpler Theta model failed. Data may be problematic.")
            return {'parameters': default_params, 'score': float('inf')}
        else:
            default_params = simpler_default
    
    logger.info(f"Baseline Theta score with default parameters: {default_score:.4f}")
    
    # Initialize best parameters and score with defaults
    best_params = default_params.copy()
    best_score = default_score

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

                # Only consider valid scores
                if score != float('inf') and not np.isnan(score) and score < 1000:
                    successful_fits += 1
                    
                    # Only use if significantly better (at least 5% improvement)
                    if score < best_score * 0.95:
                        best_score = score
                        best_params = params
                        logger.info(f"New best Theta model: deseasonalize={deseasonalize}, period={period}, method={method} with score {score:.4f}")

    # Log optimization summary
    logger.info(f"Theta optimization complete. Tried {tried_combinations} combinations, {successful_fits} successful fits")

    # Add sanity check - if optimization made things worse, revert to defaults
    if best_score > default_score * 2:  # If more than 2 times worse
        logger.warning(f"Optimization resulted in worse score ({best_score:.4f} vs {default_score:.4f}). Reverting to default parameters.")
        best_params = default_params
        best_score = default_score

    return {'parameters': best_params, 'score': best_score}


# Helper function to verify optimization results
def verify_optimization_result(result, model_type, sku_identifier=None):
    """
    Verify optimization results and ensure they are valid

    Parameters:
    -----------
    result : dict
        Optimization result with parameters and score
    model_type : str
        Model type (e.g., 'prophet', 'arima')
    sku_identifier : str, optional
        SKU identifier for logging

    Returns:
    --------
    dict
        Verified parameters or defaults if invalid
    """
    if not result or 'parameters' not in result:
        logger.warning(f"Invalid optimization result for {model_type}" + 
                      (f" (SKU: {sku_identifier})" if sku_identifier else ""))
        # Return appropriate defaults based on model type
        if model_type == 'arima' or model_type == 'auto_arima':
            return {'parameters': {'p': 1, 'd': 1, 'q': 0}, 'score': float('inf')}
        elif model_type == 'prophet':
            return {'parameters': {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}, 'score': float('inf')}
        elif model_type == 'ets':
            return {'parameters': {'trend': 'add', 'seasonal': None, 'seasonal_periods': 1, 'damped_trend': False}, 'score': float('inf')}
        elif model_type == 'theta':
            return {'parameters': {'deseasonalize': True, 'period': 12, 'method': 'auto'}, 'score': float('inf')}
        else:
            return {'parameters': {}, 'score': float('inf')}
    
    # Check for extreme scores
    if 'score' in result and (result['score'] == float('inf') or np.isnan(result['score']) or result['score'] > 1000):
        logger.warning(f"Extreme score in optimization result for {model_type}" + 
                      (f" (SKU: {sku_identifier})" if sku_identifier else ""))
        # Keep parameters but mark score as extreme
        return {'parameters': result['parameters'], 'score': float('inf'), 'extreme_score': True}
    
    # Result appears valid
    return result