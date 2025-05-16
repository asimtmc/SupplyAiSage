import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def plot_forecast(historical_data, forecast_data, upper_bound=None, lower_bound=None, model_name=None, sku=None):
    """Create an interactive forecast plot with historical data and forecast.
    
    Args:
        historical_data (pd.DataFrame): Historical data with date and quantity columns
        forecast_data (pd.Series): Forecast data with date index
        upper_bound (pd.Series, optional): Upper confidence bound for forecast
        lower_bound (pd.Series, optional): Lower confidence bound for forecast
        model_name (str, optional): Name of the model used for forecasting
        sku (str, optional): SKU identifier
        
    Returns:
        plotly.graph_objects.Figure: Interactive forecast plot
    """
    # Create a new figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['quantity'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue'),
            marker=dict(size=6)
        )
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data.values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash'),
            marker=dict(size=6)
        )
    )
    
    # Add confidence bounds if provided
    if upper_bound is not None and lower_bound is not None:
        fig.add_trace(
            go.Scatter(
                x=upper_bound.index,
                y=upper_bound.values,
                mode='lines',
                name='Upper Bound',
                line=dict(color='rgba(255, 0, 0, 0.2)'),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=lower_bound.index,
                y=lower_bound.values,
                mode='lines',
                name='Lower Bound',
                line=dict(color='rgba(255, 0, 0, 0.2)'),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                showlegend=False
            )
        )
    
    # Set layout
    title = f"Demand Forecast"
    if sku:
        title += f" for {sku}"
    if model_name:
        title += f" using {model_name} model"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Quantity",
        legend_title="Data Series",
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_model_comparison(historical_data, forecasts_dict, sku=None):
    """Create a plot to compare forecasts from different models.
    
    Args:
        historical_data (pd.DataFrame): Historical data with date and quantity columns
        forecasts_dict (dict): Dictionary mapping model names to forecast Series
        sku (str, optional): SKU identifier
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot comparing models
    """
    # Create a new figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['quantity'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue'),
            marker=dict(size=6)
        )
    )
    
    # Color scheme for different models
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    
    # Add each model's forecast
    for i, (model_name, forecast) in enumerate(forecasts_dict.items()):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines+markers',
                name=f"{model_name}",
                line=dict(color=color, dash='dash'),
                marker=dict(size=6)
            )
        )
    
    # Set layout
    title = "Model Comparison"
    if sku:
        title += f" for {sku}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Quantity",
        legend_title="Models",
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_parameter_importance(parameters, metrics, model_type):
    """Create a plot showing parameter importance based on metrics.
    
    Args:
        parameters (dict): Model parameters
        metrics (dict): Performance metrics
        model_type (str): Type of model
        
    Returns:
        plotly.graph_objects.Figure: Bar chart of parameter importance
    """
    # This is a simplified version just to demonstrate the function
    # In a real implementation, this would compute actual importance scores
    
    # Create dummy importance scores for demonstration
    importance = {}
    for param, value in parameters.items():
        # Skip non-numeric parameters
        if not isinstance(value, (int, float)):
            continue
        # Assign random importance score
        importance[param] = np.random.random()
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame({
        'Parameter': list(importance.keys()),
        'Importance': list(importance.values())
    })
    df = df.sort_values('Importance', ascending=False)
    
    # Create plot
    fig = px.bar(
        df, 
        x='Parameter', 
        y='Importance',
        title=f"Parameter Importance for {model_type} Model",
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Parameter",
        yaxis_title="Relative Importance",
        template="plotly_white",
        height=400
    )
    
    return fig