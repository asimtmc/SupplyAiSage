import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from datetime import datetime, timedelta
import numpy as np
import streamlit as st

@st.cache_data(ttl=1800, show_spinner=False)  # 30 minute cache for visualizations
def plot_forecast(forecast_result, sales_data=None, forecast_data=None, sku=None, selected_models=None, show_anomalies=False, confidence_interval=0.95):
    """
    Create interactive forecast visualization with caching for better performance
    
    This function supports two calling patterns:
    1. plot_forecast(forecast_result) - single parameter from the Advanced Demand Forecasting page
    2. plot_forecast(sales_data, forecast_data, sku, selected_models) - original calling pattern
    
    Parameters:
    -----------
    forecast_result : dict
        Forecast result dictionary containing historical data and forecast data
    sales_data : pandas.DataFrame, optional
        Historical sales data (used in original calling pattern)
    forecast_data : dict, optional
        Forecast information (used in original calling pattern)
    sku : str, optional
        SKU identifier (used in original calling pattern)
    selected_models : list, optional
        List of models to display (used in original calling pattern)
    show_anomalies : bool, optional
        Whether to highlight anomalies in the data
    confidence_interval : float, optional
        Confidence interval level for forecast bounds (0-1)
    """
    # Create a default figure
    fig = go.Figure()
    
    # Handle the case where forecast_result is passed as a single parameter
    if sales_data is None and forecast_data is None and sku is None:
        # Check if forecast_result is None or not a dictionary
        if forecast_result is None or not isinstance(forecast_result, dict):
            fig.update_layout(
                title="No forecast data available",
                annotations=[{
                    'text': "Please run a forecast first to see results",
                    'showarrow': False,
                    'font': {'size': 16},
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5
                }]
            )
            return fig
            
        # Extract data from forecast_result
        # Get historical data
        sku_data = pd.DataFrame()
        if 'historical_data' in forecast_result and isinstance(forecast_result['historical_data'], pd.DataFrame):
            sku_data = forecast_result['historical_data'].copy()
        elif 'data' in forecast_result and isinstance(forecast_result['data'], pd.DataFrame):
            sku_data = forecast_result['data'].copy()
        elif 'history' in forecast_result and isinstance(forecast_result['history'], pd.DataFrame):
            sku_data = forecast_result['history'].copy()
            
        # Get the SKU from the forecast_result if available
        if 'sku' in forecast_result:
            sku = forecast_result['sku']
        else:
            sku = "Unknown SKU"
        
        # Add historical data if available
        if not sku_data.empty and 'date' in sku_data.columns and 'quantity' in sku_data.columns:
            sku_data = sku_data.sort_values('date')
            fig.add_trace(
                go.Scatter(
                    x=sku_data['date'],
                    y=sku_data['quantity'],
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='blue'),
                    marker=dict(size=8, symbol='circle'),
                    hovertemplate='%{x|%b %Y}: %{y:,.0f} units'
                )
            )
            
            # Add anomalies if requested
            if show_anomalies and 'anomalies' in forecast_result:
                anomalies = forecast_result['anomalies']
                if isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=anomalies['date'],
                            y=anomalies['quantity'],
                            mode='markers',
                            name='Anomalies',
                            marker=dict(
                                size=12, 
                                symbol='x',
                                color='red',
                                line=dict(width=2, color='black')
                            ),
                            hovertemplate='Anomaly: %{x|%b %Y}: %{y:,.0f} units'
                        )
                    )
        
        # Add forecast
        forecast_df = pd.DataFrame()
        
        # Check all possible forecast data keys
        if 'forecast_data' in forecast_result and isinstance(forecast_result['forecast_data'], pd.DataFrame) and not forecast_result['forecast_data'].empty:
            forecast_df = forecast_result['forecast_data'].copy()
        elif 'forecast' in forecast_result and isinstance(forecast_result['forecast'], pd.DataFrame) and not forecast_result['forecast'].empty:
            forecast_df = forecast_result['forecast'].copy()
            
        # If we have forecast data, add it to the plot
        if not forecast_df.empty:
            # Check column formats and normalize if needed
            if 'ds' in forecast_df.columns and 'date' not in forecast_df.columns:
                forecast_df['date'] = forecast_df['ds']
            elif 'date' in forecast_df.columns and 'ds' not in forecast_df.columns:
                forecast_df['ds'] = forecast_df['date']
                
            # Get the forecast values based on column naming convention
            if 'yhat' in forecast_df.columns:
                # Prophet-style output with yhat column
                x_col = 'ds'
                y_col = 'yhat'
                lower_col = 'yhat_lower'
                upper_col = 'yhat_upper'
            elif 'forecast' in forecast_df.columns:
                # ARIMA/SARIMA style with forecast column
                x_col = 'date' if 'date' in forecast_df.columns else 'ds'
                y_col = 'forecast'
                lower_col = 'lower_bound' if 'lower_bound' in forecast_df.columns else None
                upper_col = 'upper_bound' if 'upper_bound' in forecast_df.columns else None
            elif 'predicted' in forecast_df.columns:
                # Generic ML model style with predicted column
                x_col = 'date' if 'date' in forecast_df.columns else 'ds'
                y_col = 'predicted'
                lower_col = 'lower_bound' if 'lower_bound' in forecast_df.columns else None
                upper_col = 'upper_bound' if 'upper_bound' in forecast_df.columns else None
            elif 'mean' in forecast_df.columns:
                # statsmodels style with mean column
                x_col = 'date' if 'date' in forecast_df.columns else 'ds'
                y_col = 'mean'
                lower_col = 'mean_ci_lower' if 'mean_ci_lower' in forecast_df.columns else None
                upper_col = 'mean_ci_upper' if 'mean_ci_upper' in forecast_df.columns else None
            else:
                # Default case - use first datetime column as x and first numeric column as y
                datetime_cols = [col for col in forecast_df.columns if pd.api.types.is_datetime64_any_dtype(forecast_df[col])]
                numeric_cols = [col for col in forecast_df.columns if pd.api.types.is_numeric_dtype(forecast_df[col])]
                
                if datetime_cols and numeric_cols:
                    x_col = datetime_cols[0]
                    y_col = numeric_cols[0]
                    lower_col = None
                    upper_col = None
                else:
                    # Unable to determine columns
                    fig.update_layout(
                        title="Unknown forecast data format",
                        annotations=[{
                            'text': "Forecast data format not recognized",
                            'showarrow': False,
                            'font': {'size': 16},
                            'xref': 'paper',
                            'yref': 'paper',
                            'x': 0.5,
                            'y': 0.5
                        }]
                    )
                    return fig
                
            # Add the forecast trace
            if x_col in forecast_df.columns and y_col in forecast_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df[x_col],
                        y=forecast_df[y_col],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', dash='solid'),
                        marker=dict(size=8, symbol='circle'),
                        hovertemplate='%{x|%b %Y}: %{y:,.0f} units'
                    )
                )
                
                # Add confidence intervals if available
                if lower_col and upper_col and lower_col in forecast_df.columns and upper_col in forecast_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_df[x_col].tolist() + forecast_df[x_col].tolist()[::-1],
                            y=forecast_df[upper_col].tolist() + forecast_df[lower_col].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(255, 0, 0, 0.2)',
                            line=dict(color='rgba(255, 255, 255, 0)'),
                            hoverinfo='skip',
                            name='Confidence Interval',
                            showlegend=True
                        )
                    )
            
            # Update layout
            max_date = sku_data['date'].max() if not sku_data.empty and 'date' in sku_data.columns else None
            
            fig.update_layout(
                title=f"<b>Sales Forecast for {sku}</b>",
                xaxis_title="Date",
                yaxis_title="Units Sold",
                template="plotly_white",
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add vertical line separating history and forecast if possible
            if max_date is not None:
                # Convert to string representation to avoid timestamp math issues
                fig.add_shape(
                    type="line",
                    x0=max_date,
                    y0=0,
                    x1=max_date,
                    y1=1,
                    yref="paper",
                    line=dict(
                        color="gray",
                        width=2,
                        dash="dash",
                    )
                )
                
                # Add annotation for the forecast start
                fig.add_annotation(
                    x=max_date,
                    y=1,
                    yref="paper",
                    text="Forecast Start",
                    showarrow=False,
                    xanchor="right",
                    yanchor="top"
                )
        else:
            # If no forecast data but we have historical data
            if not sku_data.empty:
                fig.update_layout(
                    title=f"<b>Historical Sales for {sku}</b>",
                    xaxis_title="Date",
                    yaxis_title="Units Sold",
                    template="plotly_white",
                    height=500
                )
            else:
                # Truly empty data case
                fig.update_layout(
                    title=f"No data available for {sku}",
                    annotations=[{
                        'text': "No data available for this SKU",
                        'showarrow': False,
                        'font': {'size': 16},
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5
                    }]
                )
        
        return fig
    
    # Original implementation for the old calling pattern
    # Filter sales data for this SKU and ensure valid filtering
    if not isinstance(sku, str) or sku not in sales_data['sku'].unique():
        # If sku is invalid, return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text="No valid SKU selected or data unavailable",
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=400)
        return fig
        
    sku_data = sales_data[sales_data['sku'] == sku].copy()
    sku_data = sku_data.sort_values('date')

    # Create figure
    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=sku_data['date'],
            y=sku_data['quantity'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='blue'),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='%{x|%b %Y}: %{y:,.0f} units'
        )
    )

    # Define model colors
    model_colors = {
        'arima': 'green',
        'sarima': 'purple',
        'prophet': 'orange',
        'lstm': 'brown',
        'holtwinters': 'teal',
        'moving_average': 'gray'
    }

    # Check if forecast_data is available
    if forecast_data is None:
        # No forecast data available, just return the historical data chart
        fig.update_layout(
            title=f"<b>Historical Sales for {sku}</b>",
            height=400
        )
        return fig
    
    # Add forecasts for all selected models (no primary model, all are equal)
    if selected_models and forecast_data and isinstance(forecast_data, dict) and 'model_evaluation' in forecast_data and 'all_models_forecasts' in forecast_data['model_evaluation']:
        all_models = forecast_data['model_evaluation']['all_models_forecasts']

        # Add each selected model
        models_added = False
        for model in selected_models:
            if model.lower() in all_models:
                model_forecast = all_models[model.lower()]
                if model_forecast is None or len(model_forecast) == 0:
                    continue
                    
                models_added = True
                # Choose a color for this model
                color = model_colors.get(model.lower(), 'red')

                fig.add_trace(
                    go.Scatter(
                        x=model_forecast.index,
                        y=model_forecast.values,
                        mode='lines+markers',
                        name=f"{model.upper()} Forecast",
                        line=dict(color=color, dash='solid'),
                        marker=dict(size=8, symbol='circle'),
                        hovertemplate='%{x|%b %Y}: %{y:,.0f} units'
                    )
                )
        
        # If no models were added, show a message
        if not models_added:
            fig.add_annotation(
                text="No forecast models available for selected options",
                showarrow=False,
                font=dict(size=16),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )
            
    elif forecast_data and isinstance(forecast_data, dict) and 'forecast' in forecast_data:
        # Fallback to using the model's forecast if no other models are available
        forecast_values = forecast_data['forecast']
        if forecast_values is not None and len(forecast_values) > 0:
            model_name = forecast_data.get('model', 'Default').upper()
            fig.add_trace(
                go.Scatter(
                    x=forecast_values.index,
                    y=forecast_values.values,
                    mode='lines+markers',
                    name=f"Forecast ({model_name})",
                    line=dict(color='red', dash='solid'),
                    marker=dict(size=8, symbol='circle'),
                    hovertemplate='%{x|%b %Y}: %{y:,.0f} units'
                )
            )
        else:
            # No forecast values available
            fig.add_annotation(
                text="Forecast data available but no values to display",
                showarrow=False,
                font=dict(size=16),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )
    else:
        # No forecast data available
        fig.add_annotation(
            text="No forecast data available",
            showarrow=False,
            font=dict(size=16),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5
        )

    # Add confidence intervals if available
    if isinstance(forecast_data, dict) and 'lower_bound' in forecast_data and 'upper_bound' in forecast_data and 'forecast' in forecast_data:
        lower_bound = forecast_data['lower_bound']
        upper_bound = forecast_data['upper_bound']
        
        # Check that all required elements are available and not None
        if lower_bound is not None and upper_bound is not None and forecast_data['forecast'] is not None:
            # Use the right x values for the confidence interval
            try:
                x_values = forecast_data['forecast'].index
                
                # Add shaded area for confidence interval
                fig.add_trace(
                    go.Scatter(
                        x=x_values.tolist() + x_values.tolist()[::-1],
                        y=upper_bound.values.tolist() + lower_bound.values.tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(color='rgba(255, 255, 255, 0)'),
                        hoverinfo='skip',
                        showlegend=False
                    )
                )
            except (AttributeError, TypeError) as e:
                # Unable to add confidence intervals, skip this part
                pass

    # Add test predictions if available and requested
    try:
        # Only attempt to add test predictions if all required data exists
        if isinstance(forecast_data, dict) and 'show_test_predictions' in forecast_data and forecast_data['show_test_predictions']:
            if 'model_evaluation' in forecast_data:
                # Get test data
                test_set = None
                if 'test_set' in forecast_data['model_evaluation']:
                    test_set = forecast_data['model_evaluation']['test_set']
                elif forecast_data.get('test_set') is not None:
                    test_set = forecast_data.get('test_set')
    
                # Get test predictions for the selected models
                if selected_models and 'all_models_test_pred' in forecast_data['model_evaluation']:
                    for model in selected_models:
                        if model.lower() in forecast_data['model_evaluation']['all_models_test_pred']:
                            test_predictions = forecast_data['model_evaluation']['all_models_test_pred'][model.lower()]
    
                            # Choose a color for this model that matches the forecast
                            color = model_colors.get(model.lower(), 'orange')
    
                            # Make sure test_predictions is valid data
                            if test_predictions is None or not isinstance(test_predictions, pd.Series) or test_predictions.empty:
                                continue
    
                            fig.add_trace(
                                go.Scatter(
                                    x=test_predictions.index,
                                    y=test_predictions.values,
                                    mode='lines+markers',
                                    name=f"{model.upper()} Test Predictions",
                                    line=dict(color=color, dash='dot'),
                                    marker=dict(size=6, symbol='circle'),
                                    hovertemplate='%{x|%b %Y}: %{y:,.0f} units'
                                )
                            )
                elif 'test_predictions' in forecast_data['model_evaluation']:
                    # Fall back to the best model's predictions
                    test_predictions = forecast_data['model_evaluation']['test_predictions']
    
                    # Make sure test_predictions is valid before plotting
                    if test_predictions is not None and isinstance(test_predictions, pd.Series) and not test_predictions.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=test_predictions.index,
                                y=test_predictions.values,
                                mode='lines+markers',
                                name='Test Predictions',
                                line=dict(color='orange', dash='dot'),
                                marker=dict(size=6, symbol='circle'),
                                hovertemplate='%{x|%b %Y}: %{y:,.0f} units'
                            )
                        )
    
                # Add test actuals if available
                if test_set is not None and isinstance(test_set, pd.Series) and not test_set.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=test_set.index,
                            y=test_set.values,
                            mode='markers',
                            name='Test Actuals',
                            marker=dict(size=8, symbol='square', color='blue'),
                            hovertemplate='%{x|%b %Y}: %{y:,.0f} units'
                        )
                    )
    except Exception as e:
        # If anything goes wrong during test predictions plotting, just continue without them
        pass

    # Create a better layout - need to handle case when forecast data might be missing
    try:
        # Only attempt to get forecast date range if forecast data exists and has valid index
        if isinstance(forecast_data, dict) and 'forecast' in forecast_data and forecast_data['forecast'] is not None and len(forecast_data['forecast']) > 0:
            forecast_end = forecast_data['forecast'].index.max()
            months = pd.date_range(start=sku_data['date'].min(), end=forecast_end, freq='MS')
            has_forecast = True
        else:
            # Only use historical data for date range
            months = pd.date_range(start=sku_data['date'].min(), end=sku_data['date'].max(), freq='MS')
            has_forecast = False
            
        fig.update_layout(
            title=f"<b>{'Sales Forecast' if has_forecast else 'Historical Sales'} for {sku}</b>",
            xaxis=dict(
                title="Date",
                tickvals=months,
                tickformat="%b %Y",
                tickangle=45,
                showgrid=True,
                gridcolor='rgba(220, 220, 220, 0.8)'
            ),
            yaxis=dict(
                title="Units Sold",
                showgrid=True,
                gridcolor='rgba(220, 220, 220, 0.8)'
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            height=500,
            template="plotly_white"
        )
        
        # Only add forecast-related visual elements if we have forecast data
        if has_forecast:
            # Add a vertical line separating historical and forecast periods
            # Use a direct shape instead of add_vline to avoid type issues
            # Convert to string format to avoid type issues
            max_date = sku_data['date'].max()
            if isinstance(max_date, pd.Timestamp):
                max_date = max_date.strftime('%Y-%m-%d')

            forecast_start_line = dict(
                type="line",
                xref="x",
                yref="paper",
                x0=max_date,
                y0=0,
                x1=max_date,
                y1=1,
                line=dict(
                    color="gray",
                    dash="dash",
                )
            )

            # Add annotation for the line
            # Use the same string-formatted date to ensure consistency
            forecast_annotation = dict(
                x=max_date,
                y=1,
                xref="x",
                yref="paper",
                text="Forecast Start",
                showarrow=False,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.8)",
                font=dict(size=12)
            )

            # Update layout to include the shape and annotation
            if 'shapes' in fig.layout:
                fig.layout.shapes = list(fig.layout.shapes) + [forecast_start_line]
            else:
                fig.layout.shapes = [forecast_start_line]

            if 'annotations' in fig.layout:
                fig.layout.annotations = list(fig.layout.annotations) + [forecast_annotation]
            else:
                fig.layout.annotations = [forecast_annotation]

            # Indicate forecast area with light shading
            forecast_shade = dict(
                type="rect",
                xref="x", yref="paper",
                x0=sku_data['date'].max(), y0=0,
                x1=forecast_end, y1=1,
                fillcolor="rgba(200, 200, 200, 0.1)",
                layer="below",
                line_width=0,
            )

            fig.update_layout(shapes=[forecast_shade])
    except Exception as e:
        # If there's any error in layout creation, use a simple default layout
        fig.update_layout(
            title=f"<b>Sales Data for {sku}</b>",
            height=400
        )

    return fig

def plot_cluster_summary(cluster_info):
    """
    Create a plotly figure showing cluster distribution and characteristics

    Parameters:
    -----------
    cluster_info : pandas.DataFrame
        Information about SKU clusters

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the cluster summary
    """
    if len(cluster_info) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="No cluster data available")
        return fig

    # Count SKUs in each cluster
    cluster_counts = cluster_info['cluster_name'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']

    # Create figure
    fig = px.bar(
        cluster_counts, 
        x='Cluster', 
        y='Count',
        title="SKU Distribution by Cluster",
        color='Count',
        color_continuous_scale=px.colors.sequential.Blues
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Number of SKUs",
        template="plotly_white"
    )

    return fig

def plot_material_requirements(material_requirements, top_n=10):
    """
    Create a plotly figure showing material requirements over time

    Parameters:
    -----------
    material_requirements : pandas.DataFrame
        Material requirements data
    top_n : int, optional
        Number of top materials to display (default is 10)

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the material requirements plot
    """
    if len(material_requirements) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="No material requirements data available")
        return fig

    # Group by material and calculate total requirements
    material_totals = material_requirements.groupby('material_id')['order_quantity'].sum().reset_index()

    # Get top N materials by quantity
    top_materials = material_totals.sort_values('order_quantity', ascending=False).head(top_n)['material_id'].values

    # Filter data for top materials
    filtered_data = material_requirements[material_requirements['material_id'].isin(top_materials)]

    # Convert to long format for plotting
    filtered_data['order_date'] = pd.to_datetime(filtered_data['order_date'])

    # Create figure
    fig = px.line(
        filtered_data, 
        x='order_date', 
        y='order_quantity',
        color='material_id',
        title=f"Material Requirements Over Time (Top {top_n})",
        markers=True
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Order Date",
        yaxis_title="Order Quantity",
        legend_title="Material ID",
        template="plotly_white"
    )

    return fig

def plot_production_plan(production_plan):
    """
    Create a plotly figure showing the production plan over time

    Parameters:
    -----------
    production_plan : pandas.DataFrame
        Production plan data

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the production plan plot
    """
    if len(production_plan) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="No production plan data available")
        return fig

    # Group by period and aggregate
    period_summary = production_plan.groupby(['period', 'date']).agg({
        'production_quantity': 'sum',
        'min_quantity': 'sum',
        'max_quantity': 'sum'
    }).reset_index()

    # Sort by date
    period_summary = period_summary.sort_values('date')

    # Create figure
    fig = go.Figure()

    # Add production quantity
    fig.add_trace(go.Bar(
        x=period_summary['period'],
        y=period_summary['production_quantity'],
        name='Production Quantity',
        marker_color='darkblue'
    ))

    # Add min and max range
    fig.add_trace(go.Scatter(
        x=period_summary['period'],
        y=period_summary['min_quantity'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=period_summary['period'],
        y=period_summary['max_quantity'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.2)',
        name='Production Range'
    ))

    # Update layout
    fig.update_layout(
        title="Production Plan by Period",
        xaxis_title="Period",
        yaxis_title="Quantity",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )

    return fig

def plot_forecast_accuracy(actuals, forecasts):
    """
    Create a plotly figure showing forecast accuracy metrics

    Parameters:
    -----------
    actuals : pandas.DataFrame
        Actual sales data
    forecasts : dict
        Dictionary of forecast results

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the forecast accuracy plot
    """
    # If no actual data available for comparison
    if len(actuals) == 0 or len(forecasts) == 0:
        fig = go.Figure()
        fig.update_layout(title="No forecast accuracy data available")
        return fig

    # Create a list to store accuracy metrics
    accuracy_data = []

    # For each SKU in the forecast
    for sku, forecast_data in forecasts.items():
        # Get actual data for this SKU
        sku_actuals = actuals[actuals['sku'] == sku].copy()

        if len(sku_actuals) == 0:
            continue

        # Get forecast data
        forecast = forecast_data['forecast']

        # Find overlapping dates
        actual_dates = set(sku_actuals['date'])
        forecast_dates = set(forecast.index)
        common_dates = actual_dates.intersection(forecast_dates)

        if not common_dates:
            continue

        # Calculate accuracy metrics for common dates
        mape_sum = 0
        count = 0

        for date in common_dates:
            actual = sku_actuals[sku_actuals['date'] == date]['quantity'].iloc[0]
            predicted = forecast.get(date, 0)

            if actual > 0:  # Avoid division by zero
                mape = abs(actual - predicted) / actual
                mape_sum += mape
                count += 1

        if count > 0:
            mape_avg = mape_sum / count
            accuracy = max(0, 100 * (1 - mape_avg))  # Convert MAPE to accuracy percentage

            accuracy_data.append({
                'sku': sku,
                'accuracy': accuracy,
                'model': forecast_data.get('model', 'unknown'),
                'cluster': forecast_data.get('cluster_name', 'unknown')
            })

    # If no accuracy data could be calculated
    if not accuracy_data:
        fig = go.Figure()
        fig.update_layout(title="No forecast accuracy data available for comparison")
        return fig

    # Convert to DataFrame
    accuracy_df = pd.DataFrame(accuracy_data)

    # Create figure
    fig = px.bar(
        accuracy_df.sort_values('accuracy', ascending=False), 
        x='sku', 
        y='accuracy',
        color='model',
        title="Forecast Accuracy by SKU",
        labels={'accuracy': 'Accuracy (%)'},
        hover_data=['cluster']
    )

    # Add threshold line at 80% accuracy
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Good Accuracy (80%)")

    # Update layout
    fig.update_layout(
        xaxis_title="SKU",
        yaxis_title="Accuracy (%)",
        template="plotly_white",
        yaxis=dict(range=[0, 100])  # Set y-axis range from 0 to 100%
    )

    return fig

def plot_inventory_health(sales_data, forecast_data):
    """
    Create a plotly figure showing inventory health metrics

    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Historical sales data
    forecast_data : dict
        Dictionary of forecast results

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the inventory health plot
    """
    if len(sales_data) == 0 or len(forecast_data) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="No inventory health data available")
        return fig

    # Create a list to store inventory metrics
    inventory_data = []

    # For each SKU
    for sku in sales_data['sku'].unique():
        if sku not in forecast_data:
            continue

        # Get sales data for this SKU
        sku_sales = sales_data[sales_data['sku'] == sku].copy()

        if len(sku_sales) < 3:  # Need at least a few data points
            continue

        # Calculate average monthly sales
        sku_sales['month'] = sku_sales['date'].dt.to_period('M')
        monthly_sales = sku_sales.groupby('month')['quantity'].sum().reset_index()
        avg_monthly_sales = monthly_sales['quantity'].mean()

        if avg_monthly_sales == 0:
            continue

        # Get forecast data
        next_month_forecast = forecast_data[sku]['forecast'].iloc[0] if len(forecast_data[sku]['forecast']) > 0 else 0

        # Calculate months of supply (ratio of next month forecast to average monthly sales)
        months_of_supply = next_month_forecast / avg_monthly_sales if avg_monthly_sales > 0 else float('inf')

        # Calculate coefficient of variation (measure of demand variability)
        cv = monthly_sales['quantity'].std() / monthly_sales['quantity'].mean() if monthly_sales['quantity'].mean() > 0 else 0

        # Calculate growth trend (comparing last 3 months to previous 3 months)
        if len(monthly_sales) >= 6:
            recent_avg = monthly_sales.iloc[-3:]['quantity'].mean()
            previous_avg = monthly_sales.iloc[-6:-3]['quantity'].mean() if len(monthly_sales) >= 6 else avg_monthly_sales
            growth_rate = ((recent_avg / previous_avg) - 1) * 100 if previous_avg > 0 else 0
        else:
            growth_rate = 0

        # Determine inventory status
        if months_of_supply < 0.5:
            status = "Stockout Risk"
            color = "red"
        elif months_of_supply > 3:
            status = "Overstocked"
            color = "orange"
        else:
            status = "Healthy"
            color = "green"

        # Calculate risk score based on:
        # - Demand variability (higher = more risk)
        # - Months of supply (extremes = more risk)
        # - Growth rate (higher volatility = more risk)
        supply_risk_factor = abs(months_of_supply - 1.75) / 1.75  # 1.75 months is ideal (0-3.5 range)
        variability_risk_factor = min(cv, 1.0)  # Cap at 1.0
        growth_risk_factor = min(abs(growth_rate) / 50, 1.0)  # Cap at 1.0, 50% growth is high volatility

        risk_score = (0.4 * supply_risk_factor + 0.4 * variability_risk_factor + 0.2 * growth_risk_factor) * 100
        risk_score = min(max(risk_score, 0), 100)  # Ensure between 0-100

        # Calculate importance score based on volume and value
        importance_score = avg_monthly_sales

        # Add to data
        inventory_data.append({
            'sku': sku,
            'avg_monthly_sales': avg_monthly_sales,
            'next_month_forecast': next_month_forecast,
            'months_of_supply': min(months_of_supply, 12),  # Cap at 12 months for visualization
            'status': status,
            'color': color,
            'demand_variability': cv,
            'growth_rate': growth_rate,
            'risk_score': risk_score,
            'importance': importance_score
        })

    # If no inventory data could be calculated
    if not inventory_data:
        fig = go.Figure()
        fig.update_layout(title="No inventory health data available")
        return fig

    # Convert to DataFrame
    inventory_df = pd.DataFrame(inventory_data)

    # Create figure
    fig = px.scatter(
        inventory_df,
        x='demand_variability',
        y='months_of_supply',
        color='status',
        size='avg_monthly_sales',
        hover_name='sku',
        hover_data=['growth_rate', 'risk_score'],
        title="Inventory Health Matrix",
        color_discrete_map={"Stockout Risk": "red", "Overstocked": "orange", "Healthy": "green"}
    )

    # Add reference zones
    fig.add_hrect(y0=0, y1=0.5, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0.5, y1=3, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=3, y1=12, fillcolor="orange", opacity=0.1, line_width=0)

    # Update hover template
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br><br>" +
                      "Months of Supply: %{y:.1f}<br>" +
                      "Demand Variability (CV): %{x:.2f}<br>" +
                      "Average Monthly Sales: %{marker.size:.0f}<br>" +
                      "Growth Rate: %{customdata[0]:.1f}%<br>" +
                      "Risk Score: %{customdata[1]:.0f}/100<br>" +
                      "<extra></extra>"
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Demand Variability (CV)",
        yaxis_title="Months of Supply",
        template="plotly_white",
        yaxis=dict(range=[0, 12])  # Set y-axis range from 0 to 12 months
    )

    return fig

def plot_model_comparison(model_comparison, selected_model=None):
    """
    Create a plotly figure comparing different forecasting models for a specific SKU
    with detailed error metrics visualization - using a clustered bar chart with a line for MAPE

    Parameters:
    -----------
    model_comparison : dict
        Dictionary containing model evaluation data with metrics for each model
    selected_model : str, optional
        The name of the selected (best) model to highlight

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the model comparison
    """
    # Create default figure
    fig = go.Figure()
    
    # Check if model_comparison is valid
    if not isinstance(model_comparison, dict) or not model_comparison:
        fig.update_layout(
            title="No model comparison data available",
            annotations=[{
                'text': "No model comparison data available",
                'showarrow': False,
                'font': {'size': 16},
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return fig

    # Get metrics for each model
    metrics = {}
    for model_name, model_data in model_comparison.items():
        if isinstance(model_data, dict):
            metrics[model_name] = model_data
    
    # If no valid metrics found
    if not metrics:
        fig.update_layout(
            title="Invalid model comparison data",
            annotations=[{
                'text': "Model comparison data is not in the expected format",
                'showarrow': False,
                'font': {'size': 16},
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return fig

    # If selected_model is not provided or not in metrics, determine the best model
    if selected_model is None or selected_model not in metrics:
        # Find model with lowest MAPE (or RMSE if MAPE not available)
        selected_model = min(
            metrics.keys(), 
            key=lambda m: metrics[m].get('mape', float('inf')) 
            if not pd.isna(metrics[m].get('mape', float('inf'))) else metrics[m].get('rmse', float('inf'))
        )

    # Create data for bar chart
    models = sorted(list(metrics.keys()))  # Sort alphabetically for consistent ordering
    rmse_values = [metrics[m].get('rmse', 0) for m in models]
    mape_values = [metrics[m].get('mape', 0) if not pd.isna(metrics[m].get('mape', 0)) else 0 for m in models]
    mae_values = [metrics[m].get('mae', 0) for m in models]

    # Create figure with multiple traces
    fig = go.Figure()

    # Add RMSE bars
    fig.add_trace(go.Bar(
        x=models,
        y=rmse_values,
        name='RMSE',
        marker_color='#1f77b4',
        text=[f"{v:.2f}" for v in rmse_values],
        textposition='auto',
    ))

    # Add MAE bars (clustered with RMSE)
    fig.add_trace(go.Bar(
        x=models,
        y=mae_values,
        name='MAE',
        marker_color='#2ca02c',
        text=[f"{v:.2f}" for v in mae_values],
        textposition='auto',
    ))

    # Add MAPE as a line chart on secondary y-axis
    fig.add_trace(go.Scatter(
        x=models,
        y=mape_values,
        name='MAPE (%)',
        mode='lines+markers',
        marker=dict(size=10, color='#ff7f0e'),
        line=dict(width=3, color='#ff7f0e', shape='linear'),  # Use linear shape for straight lines
        text=[f"{v:.2f}%" for v in mape_values],
        yaxis='y2'
    ))

    # Highlight the best model if it exists
    for i, model in enumerate(models):
        if model == best_model:
            # Only add annotation if there are RMSE values
            if rmse_values and max(rmse_values) > 0:
                y_pos = max(rmse_values) * 1.1
                fig.add_annotation(
                    x=model,
                    y=y_pos,
                    text="Best Model",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#d62728",
                    font=dict(
                        family="Arial",
                        size=12,
                        color="#d62728"
                    ),
                    bordercolor="#d62728",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="white",
                    opacity=0.8
                )

    # Update layout with dual y-axis
    fig.update_layout(
        title={
            'text': f"Model Comparison for SKU: {sku}",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        barmode='group',
        xaxis=dict(
            title="Model",
            tickangle=-45,
            categoryorder='total descending'
        ),
        yaxis=dict(
            title="RMSE / MAE",
            side='left',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211,211,211,0.5)'
        ),
        yaxis2=dict(
            title="MAPE (%)",
            side='right',
            overlaying='y',
            showgrid=False,
            rangemode='nonnegative'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        margin=dict(l=50, r=100, t=80, b=100)
    )

    # Add explanation of metrics
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        text="<b>Lower values indicate better model performance</b>",
        showarrow=False,
        font=dict(
            family="Arial",
            size=10,
            color="#666666"
        ),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#c7c7c7",
        borderwidth=1,
        borderpad=4
    )

    return fig

def plot_forecast_error_distribution(sku, forecast_data):
    """
    Create a plotly figure showing the distribution of forecast errors for a specific SKU

    Parameters:
    -----------
    sku : str
        SKU identifier
    forecast_data : dict
        Forecast information for the SKU containing test data or forecast errors

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the error distribution
    """
    # Check if test data exists
    if ('test_set' not in forecast_data or 
        'test_predictions' not in forecast_data or 
        len(forecast_data['test_set']) == 0):
        fig = go.Figure()
        fig.update_layout(title=f"No test data available for error analysis on SKU: {sku}")
        return fig

    # Get test data and predictions
    actuals = forecast_data['test_set'].values
    predictions = forecast_data['test_predictions'].values

    # Calculate errors
    errors = actuals - predictions
    percentage_errors = np.where(actuals > 0, (errors / actuals) * 100, 0)

    # Create figure with two subplots: histogram and QQ plot
    fig = go.Figure()

    # Add histogram of errors
    fig.add_trace(go.Histogram(
        x=errors,
        name='Error Distribution',
        marker_color='#1f77b4',
        opacity=0.7,
        nbinsx=20,
        histnorm='probability'
    ))

    # Calculate descriptive statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)

    # Add vertical lines for mean and median
    fig.add_vline(x=mean_error, line_dash="solid", line_color="#d62728", 
                  annotation_text=f"Mean: {mean_error:.2f}", 
                  annotation_position="top right")

    fig.add_vline(x=median_error, line_dash="dash", line_color="#2ca02c", 
                  annotation_text=f"Median: {median_error:.2f}", 
                  annotation_position="top left")

    # Add normal distribution curve
    x = np.linspace(min(errors), max(errors), 100)
    y = (1 / (std_error * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_error) / std_error) ** 2)

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='#ff7f0e', width=2)
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': f"Forecast Error Distribution for SKU: {sku}",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        xaxis_title="Forecast Error (Actual - Predicted)",
        yaxis_title="Probability",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Add annotation with error statistics
    stats_text = f"""
    <b>Error Statistics:</b><br>
    Mean Error: {mean_error:.2f}<br>
    Median Error: {median_error:.2f}<br>
    Std Dev: {std_error:.2f}<br>
    Min Error: {min(errors):.2f}<br>
    Max Error: {max(errors):.2f}<br>
    """

    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(
            family="Arial",
            size=12,
            color="#000000"
        ),
        align="right",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#c7c7c7",
        borderwidth=1,
        borderpad=4
    )

    return fig

def plot_forecast_accuracy_trend(actuals, forecasts, periods=6):
    """
    Create a plotly figure showing forecast accuracy trend over time

    Parameters:
    -----------
    actuals : pandas.DataFrame
        Actual sales data
    forecasts : dict
        Dictionary of forecast results
    periods : int, optional
        Number of periods to analyze (default is 6)

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the forecast accuracy trend
    """
    if len(actuals) == 0 or len(forecasts) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="No forecast accuracy data available")
        return fig

    # Get alldates in chronological order
    all_dates = sorted(actuals['date'].unique())

    # Limit to the most recent periods
    if len(all_dates) > periods:
        analysis_dates = all_dates[-periods:]
    else:
        analysis_dates = all_dates

    # Create data structure to store accuracy by period
    period_accuracy = []

    # For each period
    for date in analysis_dates:
        period_name = date.strftime('%b %Y')

        # Calculate accuracy for this period across all SKUs
        total_mape = 0
        count = 0

        for sku, forecast_data in forecasts.items():
            # Get actual data for this SKU and period
            sku_actuals = actuals[(actuals['sku'] == sku) & (actuals['date'] == date)]

            if len(sku_actuals) == 0:
                continue

            # Get forecast for this period
            if date not in forecast_data['forecast'].index:
                continue

            actual = sku_actuals['quantity'].iloc[0]
            predicted = forecast_data['forecast'].get(date, 0)

            if actual > 0:  # Avoid division by zero
                mape = abs(actual - predicted) / actual
                total_mape += mape
                count += 1

        # Calculate average accuracy for this period
        if count > 0:
            period_mape = total_mape / count
            period_accuracy.append({
                'period': period_name,
                'date': date,
                'mape': period_mape * 100,
                'accuracy': max(0, 100 - period_mape * 100)
            })

    # If no accuracy data could be calculated
    if not period_accuracy:
        fig = go.Figure()
        fig.update_layout(title="No forecast accuracy trend data available")
        return fig

    # Convert to DataFrame and sort chronologically
    accuracy_df = pd.DataFrame(period_accuracy)
    accuracy_df = accuracy_df.sort_values('date')

    # Create figure
    fig = go.Figure()

    # Add accuracy line
    fig.add_trace(go.Scatter(
        x=accuracy_df['period'],
        y=accuracy_df['accuracy'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='green', width=3),
        marker=dict(size=10)
    ))

    # Add MAPE line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=accuracy_df['period'],
        y=accuracy_df['mape'],
        mode='lines+markers',
        name='MAPE',
        yaxis='y2',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=8)
    ))

    # Add target line at 80% accuracy
    fig.add_hline(y=80, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Target Accuracy (80%)")

    # Update layout with dual y-axis
    fig.update_layout(
        title="Forecast Accuracy Trend Over Time",
        xaxis_title="Period",
        yaxis=dict(
            title="Accuracy (%)",
            range=[0, 100]
        ),
        yaxis2=dict(
            title="MAPE (%)",
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )

    return fig

def plot_inventory_risk_matrix(inventory_data):
    """
    Create a risk-impact matrix for inventory management

    Parameters:
    -----------
    inventory_data : pandas.DataFrame
        Dataframe with inventory metrics including risk_score and importance

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the risk-impact matrix
    """
    if len(inventory_data) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="No inventory risk data available")
        return fig

    # Normalize importance to 0-100 scale for visualization
    max_importance = inventory_data['importance'].max()
    inventory_data = inventory_data.copy()
    if max_importance > 0:
        inventory_data['importance_norm'] = inventory_data['importance'] / max_importance * 100
    else:
        inventory_data['importance_norm'] = 0

    # Create labels for quadrants
    def risk_label(row):
        if row['risk_score'] >= 50 and row['importance_norm'] >= 50:
            return "Critical Focus"
        elif row['risk_score'] >= 50 and row['importance_norm'] < 50:
            return "Proactive Monitor"
        elif row['risk_score'] < 50 and row['importance_norm'] >= 50:
            return "Active Manage"
        else:
            return "Routine Review"

    inventory_data['risk_category'] = inventory_data.apply(risk_label, axis=1)

    # Create color map
    color_map = {
        "Critical Focus": "red",
        "Proactive Monitor": "orange",
        "Active Manage": "blue",
        "Routine Review": "green"
    }

    # Create figure
    fig = px.scatter(
        inventory_data,
        x='risk_score',
        y='importance_norm',
        color='risk_category',
        hover_name='sku',
        size='avg_monthly_sales',
        title="Inventory Risk-Impact Matrix",
        labels={
            'risk_score': 'Risk Score (0-100)',
            'importance_norm': 'Business Impact (0-100)'
        },
        color_discrete_map=color_map
    )

    # Add quadrant lines
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.add_vline(x=50, line_dash="dash", line_color="gray")

    # Add quadrant labels
    fig.add_annotation(x=75, y=75, text="Critical Focus", showarrow=False, font=dict(size=14, color="red"))
    fig.add_annotation(x=75, y=25, text="Proactive Monitor", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=25, y=75, text="Active Manage", showarrow=False, font=dict(size=14, color="blue"))
    fig.add_annotation(x=25, y=25, text="Routine Review", showarrow=False, font=dict(size=14, color="green"))

    # Update layout
    fig.update_layout(
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 100]),
        template="plotly_white"
    )

    return fig

def plot_what_if_comparison(base_scenario, what_if_scenario):
    """
    Create a plotly figure comparing base scenario with what-if scenario

    Parameters:
    -----------
    base_scenario : dict
        Base scenario results
    what_if_scenario : dict
        What-if scenario results

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the scenario comparison
    """
    # Extract data from scenarios
    if 'production_plan' not in base_scenario or 'production_plan' not in what_if_scenario:
        # Return empty figure if missing required data
        fig = go.Figure()
        fig.update_layout(title="Missing production plan data in scenarios")
        return fig

    base_prod = base_scenario['production_plan']
    what_if_prod = what_if_scenario['production_plan']

    if len(base_prod) == 0 or len(what_if_prod) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="No scenario comparison data available")
        return fig

    # Ensure date column is in datetime format
    if 'date' in base_prod.columns and 'date' in what_if_prod.columns:
        if not pd.api.types.is_datetime64_any_dtype(base_prod['date']):
            try:
                base_prod['date'] = pd.to_datetime(base_prod['date'])
            except:
                pass
        if not pd.api.types.is_datetime64_any_dtype(what_if_prod['date']):
            try:
                what_if_prod['date'] = pd.to_datetime(what_if_prod['date'])
            except:
                pass

    # Ensure we have a 'period' column
    if 'period' not in base_prod.columns and 'date' in base_prod.columns:
        base_prod['period'] = base_prod['date'].dt.strftime('%B %Y')

    if 'period' not in what_if_prod.columns and 'date' in what_if_prod.columns:
        what_if_prod['period'] = what_if_prod['date'].dt.strftime('%B %Y')

    # Group by period
    base_summary = base_prod.groupby('period')['production_quantity'].sum().reset_index()
    what_if_summary = what_if_prod.groupby('period')['production_quantity'].sum().reset_index()

    # Merge the data
    comparison = pd.merge(base_summary, what_if_summary, on='period', suffixes=('_base', '_what_if'), how='outer').fillna(0)

    # Calculate percentage change (handle division by zero)
    comparison['percent_change'] = comparison.apply(
        lambda row: ((row['production_quantity_what_if'] - row['production_quantity_base']) / row['production_quantity_base'] * 100) 
        if row['production_quantity_base'] > 0 else 
        (100 if row['production_quantity_what_if'] > 0 else 0), 
        axis=1
    )

    # Sort by period chronologically if possible
    try:
        # Try to convert periods to datetime for sorting
        temp_dates = pd.to_datetime(comparison['period'], format='%B %Y')
        comparison['sort_key'] = temp_dates
        comparison = comparison.sort_values('sort_key')
        comparison = comparison.drop('sort_key', axis=1)
    except:
        # If conversion fails, keep original order
        pass

    # Create figure
    fig = go.Figure()

    # Add base scenario
    fig.add_trace(go.Bar(
        x=comparison['period'],
        y=comparison['production_quantity_base'],
        name='Base Scenario',
        marker_color='blue'
    ))

    # Add what-if scenario
    fig.add_trace(go.Bar(
        x=comparison['period'],
        y=comparison['production_quantity_what_if'],
        name='What-If Scenario',
        marker_color='red'
    ))

    # Add percentage change as a line
    fig.add_trace(go.Scatter(
        x=comparison['period'],
        y=comparison['percent_change'],
        mode='lines+markers',
        name='% Change',
        yaxis='y2',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))

    # Update layout with dual y-axis
    fig.update_layout(
        title="Base vs. What-If Scenario Comparison",
        xaxis_title="Period",
        yaxis_title="Production Quantity",
        yaxis2=dict(
            title="Percent Change (%)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )

    return fig