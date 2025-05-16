import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

def calculate_material_requirements(forecasts, bom_data, supplier_data):
    """
    Calculate material requirements based on forecasts, BOM, and supplier data
    
    Parameters:
    -----------
    forecasts : dict
        Dictionary with forecast results for each SKU
    bom_data : pandas.DataFrame
        Bill of materials data linking SKUs to materials
    supplier_data : pandas.DataFrame
        Supplier information including lead times and MOQs
    
    Returns:
    --------
    pandas.DataFrame
        Material requirements plan with quantities and timing
    """
    # Create empty list to store material requirements
    requirements = []
    
    # Get the forecast dates (same for all SKUs)
    first_sku = next(iter(forecasts))
    forecast_dates = forecasts[first_sku]['forecast'].index
    
    # Calculate requirements for each period
    for date in forecast_dates:
        # For each SKU in the forecast
        for sku, forecast_data in forecasts.items():
            # Get the forecasted quantity for this period
            forecast_qty = forecast_data['forecast'].get(date, 0)
            
            # Round up to nearest integer
            forecast_qty = np.ceil(forecast_qty)
            
            # Skip if forecast is zero
            if forecast_qty == 0:
                continue
            
            # Get the BOM for this SKU
            sku_bom = bom_data[bom_data['sku'] == sku]
            
            # Calculate material needs for each component
            for _, bom_row in sku_bom.iterrows():
                material_id = bom_row['material_id']
                qty_required = bom_row['quantity_required']
                
                # Calculate total material needed
                total_qty = forecast_qty * qty_required
                
                # Get supplier info for this material
                supplier_info = supplier_data[supplier_data['material_id'] == material_id]
                
                if len(supplier_info) > 0:
                    # Get lead time and MOQ
                    lead_time = supplier_info['lead_time_days'].iloc[0]
                    moq = supplier_info['moq'].iloc[0] if 'moq' in supplier_info.columns else 1
                    
                    # Calculate order date (accounting for lead time)
                    order_date = date - timedelta(days=lead_time)
                    
                    # Ensure meeting MOQ
                    order_qty = max(total_qty, moq)
                    
                    # Add to requirements list
                    requirements.append({
                        'sku': sku,
                        'material_id': material_id,
                        'forecast_date': date,
                        'order_date': order_date,
                        'quantity_required': total_qty,
                        'order_quantity': order_qty,
                        'lead_time_days': lead_time,
                        'moq': moq
                    })
    
    # Convert to DataFrame
    if requirements:
        requirements_df = pd.DataFrame(requirements)
        
        # Aggregate by material and order date
        aggregated = requirements_df.groupby(['material_id', 'order_date']).agg({
            'quantity_required': 'sum',
            'lead_time_days': 'first',
            'moq': 'first'
        }).reset_index()
        
        # Recalculate order quantity based on aggregated requirements
        aggregated['order_quantity'] = aggregated.apply(
            lambda x: max(x['quantity_required'], x['moq']), axis=1
        )
        
        return aggregated
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            'material_id', 'order_date', 'quantity_required', 
            'lead_time_days', 'moq', 'order_quantity'
        ])

def generate_production_plan(forecasts, bom_data, supplier_data):
    """
    Generate a production plan based on forecasts and material availability
    
    Parameters:
    -----------
    forecasts : dict
        Dictionary with forecast results for each SKU
    bom_data : pandas.DataFrame
        Bill of materials data linking SKUs to materials
    supplier_data : pandas.DataFrame
        Supplier information including lead times and MOQs
    
    Returns:
    --------
    pandas.DataFrame
        Production plan with quantities and timing
    """
    # Create empty list to store production plan
    production_plan = []
    
    # Get the forecast dates (same for all SKUs)
    first_sku = next(iter(forecasts))
    forecast_dates = forecasts[first_sku]['forecast'].index
    
    # Calculate production for each period and SKU
    for date in forecast_dates:
        month_name = date.strftime('%B %Y')
        
        for sku, forecast_data in forecasts.items():
            # Get the forecasted quantity for this period
            forecast_qty = forecast_data['forecast'].get(date, 0)
            lower_bound = forecast_data['lower_bound'].get(date, 0)
            upper_bound = forecast_data['upper_bound'].get(date, 0)
            
            # Round to nearest integer
            forecast_qty = np.ceil(forecast_qty)
            lower_bound = np.ceil(lower_bound)
            upper_bound = np.ceil(upper_bound)
            
            # Skip if forecast is zero
            if forecast_qty == 0:
                continue
            
            # Add to production plan
            production_plan.append({
                'sku': sku,
                'period': month_name,
                'date': date,
                'production_quantity': forecast_qty,
                'min_quantity': lower_bound,
                'max_quantity': upper_bound,
                'confidence': (forecast_data.get('model', 'unknown') != 'moving_average')
            })
    
    # Convert to DataFrame
    if production_plan:
        production_df = pd.DataFrame(production_plan)
        
        # Sort by date and SKU
        production_df = production_df.sort_values(by=['date', 'sku'])
        
        return production_df
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            'sku', 'period', 'date', 'production_quantity', 
            'min_quantity', 'max_quantity', 'confidence'
        ])

def run_what_if_scenario(forecasts, bom_data, supplier_data, scenario):
    """
    Run a what-if scenario analysis on the supply chain
    
    Parameters:
    -----------
    forecasts : dict
        Dictionary with forecast results for each SKU
    bom_data : pandas.DataFrame
        Bill of materials data linking SKUs to materials
    supplier_data : pandas.DataFrame
        Supplier information including lead times and MOQs
    scenario : dict
        Scenario parameters to model
    
    Returns:
    --------
    dict
        Results of the scenario analysis
    """
    # Make copies of the data to avoid modifying originals
    supplier_data_copy = supplier_data.copy()
    
    # Deep copy of forecasts dictionary including pandas Series objects
    forecasts_copy = {}
    for k, v in forecasts.items():
        sku_forecast = v.copy()
        # Create deep copies of the pandas Series objects
        if 'forecast' in sku_forecast and hasattr(sku_forecast['forecast'], 'copy'):
            sku_forecast['forecast'] = sku_forecast['forecast'].copy()
        if 'upper_bound' in sku_forecast and hasattr(sku_forecast['upper_bound'], 'copy'):
            sku_forecast['upper_bound'] = sku_forecast['upper_bound'].copy()
        if 'lower_bound' in sku_forecast and hasattr(sku_forecast['lower_bound'], 'copy'):
            sku_forecast['lower_bound'] = sku_forecast['lower_bound'].copy()
        forecasts_copy[k] = sku_forecast
    
    # Apply scenario modifications
    if scenario.get('type') == 'supplier_delay':
        # Increase lead times for specified supplier or all suppliers
        supplier_id = scenario.get('supplier_id', None)
        delay_days = scenario.get('delay_days', 14)  # Default 2 weeks
        
        if supplier_id:
            # Apply to specific supplier
            supplier_mask = supplier_data_copy['supplier_id'] == supplier_id
            supplier_data_copy.loc[supplier_mask, 'lead_time_days'] += delay_days
        else:
            # Apply to all suppliers
            supplier_data_copy['lead_time_days'] += delay_days
    
    elif scenario.get('type') == 'demand_increase':
        # Increase demand for specified SKU or all SKUs
        sku = scenario.get('sku', None)
        increase_percent = scenario.get('increase_percent', 30)  # Default 30%
        
        if sku and sku in forecasts_copy:
            # Apply to specific SKU
            if hasattr(forecasts_copy[sku]['forecast'], 'multiply'):
                forecasts_copy[sku]['forecast'] = forecasts_copy[sku]['forecast'].multiply(1 + increase_percent/100)
                forecasts_copy[sku]['upper_bound'] = forecasts_copy[sku]['upper_bound'].multiply(1 + increase_percent/100)
                forecasts_copy[sku]['lower_bound'] = forecasts_copy[sku]['lower_bound'].multiply(1 + increase_percent/100)
            else:
                # Fallback for non-Series objects
                forecasts_copy[sku]['forecast'] = forecasts_copy[sku]['forecast'] * (1 + increase_percent/100)
                forecasts_copy[sku]['upper_bound'] = forecasts_copy[sku]['upper_bound'] * (1 + increase_percent/100)
                forecasts_copy[sku]['lower_bound'] = forecasts_copy[sku]['lower_bound'] * (1 + increase_percent/100)
        else:
            # Apply to all SKUs
            for sku_id in forecasts_copy:
                if hasattr(forecasts_copy[sku_id]['forecast'], 'multiply'):
                    forecasts_copy[sku_id]['forecast'] = forecasts_copy[sku_id]['forecast'].multiply(1 + increase_percent/100)
                    forecasts_copy[sku_id]['upper_bound'] = forecasts_copy[sku_id]['upper_bound'].multiply(1 + increase_percent/100)
                    forecasts_copy[sku_id]['lower_bound'] = forecasts_copy[sku_id]['lower_bound'].multiply(1 + increase_percent/100)
                else:
                    # Fallback for non-Series objects
                    forecasts_copy[sku_id]['forecast'] = forecasts_copy[sku_id]['forecast'] * (1 + increase_percent/100)
                    forecasts_copy[sku_id]['upper_bound'] = forecasts_copy[sku_id]['upper_bound'] * (1 + increase_percent/100)
                    forecasts_copy[sku_id]['lower_bound'] = forecasts_copy[sku_id]['lower_bound'] * (1 + increase_percent/100)
    
    elif scenario.get('type') == 'material_shortage':
        # Simulate material shortage by reducing availability
        material_id = scenario.get('material_id', None)
        if material_id:
            # Create a flag for SKUs that use this material
            affected_skus = bom_data[bom_data['material_id'] == material_id]['sku'].unique()
            
            # Reduce forecasts for affected SKUs
            reduction_percent = scenario.get('reduction_percent', 50)  # Default 50%
            for sku_id in affected_skus:
                if sku_id in forecasts_copy:
                    if hasattr(forecasts_copy[sku_id]['forecast'], 'multiply'):
                        forecasts_copy[sku_id]['forecast'] = forecasts_copy[sku_id]['forecast'].multiply(1 - reduction_percent/100)
                        forecasts_copy[sku_id]['upper_bound'] = forecasts_copy[sku_id]['upper_bound'].multiply(1 - reduction_percent/100)
                        forecasts_copy[sku_id]['lower_bound'] = forecasts_copy[sku_id]['lower_bound'].multiply(1 - reduction_percent/100)
                    else:
                        # Fallback for non-Series objects
                        forecasts_copy[sku_id]['forecast'] = forecasts_copy[sku_id]['forecast'] * (1 - reduction_percent/100)
                        forecasts_copy[sku_id]['upper_bound'] = forecasts_copy[sku_id]['upper_bound'] * (1 - reduction_percent/100)
                        forecasts_copy[sku_id]['lower_bound'] = forecasts_copy[sku_id]['lower_bound'] * (1 - reduction_percent/100)
    
    # Generate new production plan and material requirements
    new_prod_plan = generate_production_plan(forecasts_copy, bom_data, supplier_data_copy)
    new_materials = calculate_material_requirements(forecasts_copy, bom_data, supplier_data_copy)
    
    # Return results
    return {
        'production_plan': new_prod_plan,
        'material_requirements': new_materials,
        'scenario': scenario
    }
