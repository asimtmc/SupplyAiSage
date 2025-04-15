import pandas as pd
import numpy as np
import logging
from datetime import datetime
import traceback

# Import parameter optimizers
from utils.parameter_optimizer import (
    optimize_prophet_parameters, optimize_arima_parameters,
    optimize_ets_parameters, optimize_theta_parameters
)

# Use our enhanced optimizer's log function
from utils.enhanced_parameter_optimizer import log_optimization_result

# Import enhanced optimizers
from utils.enhanced_parameter_optimizer import (
    optimize_prophet_parameters_enhanced, optimize_arima_parameters_enhanced,
    optimize_ets_parameters_enhanced, optimize_theta_parameters_enhanced,
    verify_optimization_result
)

# Import database functions
from utils.database import save_model_parameters, get_model_parameters

# Set up logging
logger = logging.getLogger('advanced_forecast')

def optimize_parameters_with_validation(data, model_type, sku_id, use_enhanced=True):
    """
    Optimize model parameters with robust validation and safety checks
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Time series data with 'date' and 'quantity' columns
    model_type : str
        Model type (e.g., 'prophet', 'arima', 'ets', 'theta')
    sku_id : str
        SKU identifier for parameter caching
    use_enhanced : bool, optional
        Whether to use enhanced optimization functions (default is True)
        
    Returns:
    --------
    dict
        Optimized parameters and metrics
    """
    logger.info(f"Starting parameter optimization for {model_type} model (SKU: {sku_id})")
    
    # Split data into train and validation sets (80% train, 20% validation)
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size].copy()
    val_data = data.iloc[train_size:].copy()
    
    # Skip optimization if not enough data
    if len(train_data) < 5 or len(val_data) < 1:
        logger.warning(f"Not enough data to optimize {model_type} parameters for SKU {sku_id}")
        return None
    
    try:
        # Choose optimization function based on model type and enhancement flag
        if model_type == 'prophet':
            # Convert data to Prophet format
            prophet_train = pd.DataFrame({
                'ds': train_data['date'],
                'y': train_data['quantity']
            })
            prophet_val = pd.DataFrame({
                'ds': val_data['date'],
                'y': val_data['quantity']
            })
            
            if use_enhanced:
                result = optimize_prophet_parameters_enhanced(prophet_train, prophet_val)
            else:
                result = optimize_prophet_parameters(prophet_train, prophet_val)
                
        elif model_type == 'arima' or model_type == 'auto_arima':
            # Convert to series for ARIMA
            train_series = pd.Series(train_data['quantity'].values, index=train_data['date'])
            val_series = pd.Series(val_data['quantity'].values, index=val_data['date'])
            
            if use_enhanced:
                result = optimize_arima_parameters_enhanced(train_series, val_series)
            else:
                result = optimize_arima_parameters(train_series, val_series)
                
        elif model_type == 'ets':
            # Convert to series for ETS
            train_series = pd.Series(train_data['quantity'].values, index=train_data['date'])
            val_series = pd.Series(val_data['quantity'].values, index=val_data['date'])
            
            if use_enhanced:
                result = optimize_ets_parameters_enhanced(train_series, val_series)
            else:
                result = optimize_ets_parameters(train_series, val_series)
                
        elif model_type == 'theta':
            # Convert to series for Theta
            train_series = pd.Series(train_data['quantity'].values, index=train_data['date'])
            val_series = pd.Series(val_data['quantity'].values, index=val_data['date'])
            
            if use_enhanced:
                result = optimize_theta_parameters_enhanced(train_series, val_series)
            else:
                result = optimize_theta_parameters(train_series, val_series)
                
        else:
            logger.warning(f"Unsupported model type: {model_type}")
            return None
        
        # Verify the result
        verified_result = verify_optimization_result(result, model_type, sku_id)
        
        # Save parameters to database
        if verified_result and 'parameters' in verified_result and verified_result['score'] != float('inf'):
            try:
                save_model_parameters(
                    sku_id, 
                    model_type, 
                    verified_result['parameters'], 
                    verified_result['score']
                )
                logger.info(f"Successfully saved optimized parameters for {model_type} (SKU: {sku_id})")
            except Exception as e:
                logger.error(f"Failed to save optimized parameters: {str(e)}")
        
        return verified_result
        
    except Exception as e:
        logger.error(f"Error during parameter optimization for {model_type}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def get_optimized_parameters(sku_id, model_type, data=None, force_reoptimize=False, use_enhanced=True):
    """
    Get optimized parameters from database or optimize them if needed
    
    Parameters:
    -----------
    sku_id : str
        SKU identifier
    model_type : str
        Model type (e.g., 'prophet', 'arima', 'ets', 'theta')
    data : pandas.DataFrame, optional
        Time series data with 'date' and 'quantity' columns (required if parameters need to be optimized)
    force_reoptimize : bool, optional
        Whether to force reoptimization even if parameters exist in the database
    use_enhanced : bool, optional
        Whether to use enhanced optimization functions (default is True)
        
    Returns:
    --------
    dict
        Optimized parameters
    """
    # Try to get parameters from database first (unless forced to reoptimize)
    if not force_reoptimize:
        cached_params = get_model_parameters(sku_id, model_type)
        if cached_params:
            logger.info(f"Found cached parameters for {model_type} (SKU: {sku_id})")
            return cached_params
    
    # Need to optimize parameters
    if data is not None:
        logger.info(f"Optimizing parameters for {model_type} (SKU: {sku_id})")
        result = optimize_parameters_with_validation(data, model_type, sku_id, use_enhanced)
        if result and 'parameters' in result:
            return result['parameters']
    
    # Return default parameters if optimization fails or data is None
    logger.warning(f"Using default parameters for {model_type} (SKU: {sku_id})")
    if model_type == 'prophet':
        return {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}
    elif model_type == 'arima' or model_type == 'auto_arima':
        return {'p': 1, 'd': 1, 'q': 0}
    elif model_type == 'ets':
        return {'trend': 'add', 'seasonal': None, 'seasonal_periods': 1, 'damped_trend': False}
    elif model_type == 'theta':
        return {'deseasonalize': True, 'period': 12, 'method': 'auto'}
    else:
        return {}