import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from datetime import datetime, timedelta
import numpy as np

def plot_forecast(sales_data, forecast_data, sku_or_skus, selected_models=None):
    """
    Create a plotly figure showing historical sales and forecast for one or multiple SKUs
    with enhanced visualization and error metrics. Can show multiple models if selected.
    
    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Historical sales data
    forecast_data : dict or None
        Forecast information for a single SKU. Set to None for multi-SKU mode.
    sku_or_skus : str or list
        SKU identifier or list of SKU identifiers
    selected_models : list, optional
        List of model names to include in the plot. If None, only the best model is shown.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the forecast plot
    """
    # Create figure
    fig = go.Figure()
    
    # Import streamlit for accessing session_state
    import streamlit as st
    
    # Convert single SKU to list for consistent processing
    skus = [sku_or_skus] if isinstance(sku_or_skus, str) else sku_or_skus
    
    # Check if we have multiple SKUs or just one
    is_multi_sku = len(skus) > 1
    
    # Define color palette for visualization
    # Using distinct colors for train/test data and forecasts
    forecast_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Validate that selected SKUs have forecast data
    valid_skus = []
    for sku in skus:
        if 'forecasts' in st.session_state and sku in st.session_state.forecasts:
            valid_skus.append(sku)
    
    # Return empty figure if no valid data
    if not valid_skus:
        fig.update_layout(title="No forecast data available for the selected SKUs")
        return fig
            
            # Add test data with distinct color if it exists
            if 'test_set' in current_forecast_data and len(current_forecast_data['test_set']) > 0:
                test_dates = current_forecast_data['test_set'].index
                test_values = current_forecast_data['test_set'].values
                
                # Create hover text with test data values
                test_hover = [f"<b>{sku}</b><br><b>Historical Data (Test)</b><br>" +
                             f"<b>Date:</b> {date.strftime('%Y-%m-%d')}<br>" +
                             f"<b>Value:</b> {int(value) if not np.isnan(value) else 'N/A'}" 
                             for date, value in zip(test_dates, test_values)]
                
                fig.add_trace(go.Scatter(
                    x=test_dates,
                    y=test_values,
                    mode='lines+markers',
                    name=f'{name_prefix}Test Data (Actual)',
                    line=dict(color=test_color, width=2, dash='dot'),
                    marker=dict(size=8, color=test_color, symbol='circle'),
                    hoverinfo='text',
                    hovertext=test_hover,
                    hoverlabel=dict(bgcolor=test_color, font=dict(color='white')),
                    showlegend=True
                ))
        else:
            # If no train/test split, add all historical data
            sku_sales = sales_data[sales_data['sku'] == sku].copy()
            
            if len(sku_sales) > 0:
                sku_sales = sku_sales.sort_values('date')
                
                # Create hover text for historical data
                hist_hover = [f"<b>{sku}</b><br><b>Historical Data</b><br>" +
                             f"<b>Date:</b> {date.strftime('%Y-%m-%d')}<br>" +
                             f"<b>Value:</b> {int(value) if not np.isnan(value) else 'N/A'}" 
                             for date, value in zip(sku_sales['date'], sku_sales['quantity'])]
                
                fig.add_trace(go.Scatter(
                    x=sku_sales['date'],
                    y=sku_sales['quantity'],
                    mode='lines+markers',
                    name=f'{name_prefix}Historical Sales',
                    line=dict(color=train_color, width=2),
                    marker=dict(size=6),
                    hoverinfo='text',
                    hovertext=hist_hover,
                    hoverlabel=dict(bgcolor=train_color, font=dict(color='white')),
                ))
    
    # Now add forecast lines for each SKU
    for idx, sku in enumerate(skus):
        # Skip if no data for this SKU
        if sku not in st.session_state.forecasts:
            continue
            
        # Get this SKU's forecast data
        current_forecast_data = st.session_state.forecasts[sku]
        
        # Add SKU name prefix for multi-SKU display
        name_prefix = f"{sku} - " if is_multi_sku else ""
        
        # Define a list of colors with better contrast for multiple models
        model_colors = ['#d62728', '#0099ff', '#9467bd', '#02bc77', '#e377c2', '#ffb74d', '#4d4dff', '#17becf']
        
        # Define line styles for better differentiation
        line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
        
        # If no models selected, use the best model
        if selected_models is None or len(selected_models) == 0:
            current_selected_models = [current_forecast_data['model']]
        else:
            current_selected_models = selected_models
        
        # Get forecast data for the primary (best) model
        primary_model = current_forecast_data['model']
        forecast = current_forecast_data['forecast']
        
        # Get lower/upper bounds if they exist
        lower_bound = current_forecast_data.get('lower_bound', None)
        upper_bound = current_forecast_data.get('upper_bound', None)
    
        # Check if we should show multiple models for this SKU
        show_multiple_models = (current_selected_models is not None and 
                               len(current_selected_models) > 1 and 
                               'model_evaluation' in current_forecast_data and 
                               'all_models_forecasts' in current_forecast_data['model_evaluation'])
        
        # Select a base forecast color for this SKU - different for each SKU
        base_forecast_color = model_colors[idx % len(model_colors)]
        
        if show_multiple_models:
            # Add each selected model's forecast for this SKU
            for i, model_name in enumerate(current_selected_models):
                # Check if the model exists in model_evaluation
                if ('model_evaluation' in current_forecast_data and 
                    'all_models_forecasts' in current_forecast_data['model_evaluation'] and 
                    model_name in current_forecast_data['model_evaluation']['all_models_forecasts']):
                    
                    model_forecast = current_forecast_data['model_evaluation']['all_models_forecasts'][model_name]
                    
                    # Choose color - primary model gets SKU color, others get variations
                    color_idx = idx % len(model_colors) if model_name == primary_model else ((idx + i) % len(model_colors))
                    
                    # Calculate dash pattern index (use different patterns for different models)
                    dash_idx = i % len(line_styles)
                    dash_pattern = line_styles[dash_idx]
                    
                    # Different marker symbols for different models
                    marker_symbols = ['diamond', 'circle', 'square', 'triangle-up', 'star', 'x']
                    marker_symbol = marker_symbols[i % len(marker_symbols)]
                    
                    # Create hover text with SKU, model name and exact values
                    hover_text = [f"<b>SKU:</b> {sku}<br><b>Model:</b> {model_name.upper()}<br>" +
                                 f"<b>Date:</b> {date.strftime('%Y-%m-%d')}<br>" +
                                 f"<b>Value:</b> {int(value) if not np.isnan(value) else 'N/A'}" 
                                 for date, value in zip(model_forecast.index, model_forecast.values)]
                    
                    fig.add_trace(go.Scatter(
                        x=model_forecast.index,
                        y=model_forecast.values,
                        mode='lines+markers',
                        name=f"{name_prefix}{model_name.upper()} Forecast",
                        line=dict(
                            color=model_colors[color_idx], 
                            width=3 if model_name == primary_model else 2,
                            dash=dash_pattern
                        ),
                        marker=dict(
                            size=8 if model_name == primary_model else 6, 
                            symbol=marker_symbol,
                            color=model_colors[color_idx]
                        ),
                        hoverinfo='text',
                        hovertext=hover_text,
                        hoverlabel=dict(bgcolor=model_colors[color_idx], font=dict(color='white')),
                    ))
                    
                    # Confidence intervals removed as requested
                elif model_name == primary_model:
                    # If primary model isn't in all_models_forecasts, still show it
                    fig.add_trace(go.Scatter(
                        x=forecast.index,
                        y=forecast.values,
                        mode='lines+markers',
                        name=f"{name_prefix}{primary_model.upper()} Forecast",
                        line=dict(color=base_forecast_color, width=3, dash='dash'),
                        marker=dict(size=8, symbol='diamond'),
                    ))
        else:
            # Only add the primary model forecast for this SKU
            # Create hover text with SKU, model name and exact values
            hover_text = [f"<b>SKU:</b> {sku}<br><b>Model:</b> {primary_model.upper()}<br>" +
                         f"<b>Date:</b> {date.strftime('%Y-%m-%d')}<br>" +
                         f"<b>Value:</b> {int(value) if not np.isnan(value) else 'N/A'}" 
                         for date, value in zip(forecast.index, forecast.values)]
                    
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines+markers',
                name=f"{name_prefix}{primary_model.upper()} Forecast",
                line=dict(color=base_forecast_color, width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hoverinfo='text',
                hovertext=hover_text,
                hoverlabel=dict(bgcolor=base_forecast_color, font=dict(color='white')),
            ))
        
        # Confidence interval removed as requested
    
    # We'll omit test predictions for multi-SKU view since they would make the chart too cluttered
    # For single SKU view, show test predictions if available 
    if not is_multi_sku and len(skus) == 1 and skus[0] in st.session_state.forecasts:
        current_data = st.session_state.forecasts[skus[0]]
        
        # Check if we should show test predictions
        has_train_test_split = 'train_set' in current_data and 'test_set' in current_data
        show_test_predictions = has_train_test_split and (
            'show_test_predictions' in current_data or 
            ('test_predictions' in current_data and len(current_data['test_predictions']) > 0)
        )
        
        if show_test_predictions and 'test_predictions' in current_data and 'test_set' in current_data:
            test_dates = current_data['test_set'].index if 'test_set' in current_data else None
            if test_dates is not None and len(test_dates) > 0:
                test_pred = current_data['test_predictions'].values
        
        # Get the primary model color for consistency
        test_pred_color = '#ff7f0e'  # Default orange
        
        # Create hover text for test predictions - only if test_dates exists
        test_pred_hover = []
        if test_dates is not None and len(test_dates) > 0 and len(test_pred) > 0:
            test_pred_hover = [f"<b>Test Prediction</b><br>" +
                             f"<b>Date:</b> {date.strftime('%Y-%m-%d')}<br>" +
                             f"<b>Predicted:</b> {int(value) if not np.isnan(value) else 'N/A'}" 
                             for date, value in zip(test_dates, test_pred)]
        
        # Add test predictions with better styling - ensuring test_dates is valid first
        if test_dates is not None and len(test_dates) > 0 and len(test_pred) > 0:
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_pred,
                mode='lines+markers',
                name='Test Period Forecast',
                line=dict(
                    color=test_pred_color, 
                    width=3, 
                    dash='dot'
                ),
                marker=dict(
                    size=8, 
                    color=test_pred_color, 
                    symbol='star',
                    line=dict(color='#ffffff', width=1)  # Add white border to markers
                ),
                hoverinfo='text',
                hovertext=test_pred_hover,
                hoverlabel=dict(bgcolor=test_pred_color, font=dict(color='white')),
                showlegend=True
            ))
        
        # Add area between actual and predicted for test period to highlight error
        if test_dates is not None and len(test_dates) > 0 and len(test_pred) > 0 and len(test_dates) == len(test_pred):
            test_actual = current_data['test_set'].values
            
            # Convert to lists to ensure we can concatenate even if different types
            test_dates_list = test_dates.tolist() if hasattr(test_dates, 'tolist') else list(test_dates)
            test_pred_list = test_pred.tolist() if hasattr(test_pred, 'tolist') else list(test_pred)
            test_actual_list = test_actual.tolist() if hasattr(test_actual, 'tolist') else list(test_actual)
            
            # Create a filled area between actual and predicted
            fig.add_trace(go.Scatter(
                x=test_dates_list + test_dates_list[::-1],
                y=test_pred_list + test_actual_list[::-1],
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.1)',  # Light orange fill
                line=dict(color='rgba(255, 127, 14, 0)'),  # Transparent line
                name='Forecast Error',
                showlegend=True
            ))
            
            # Annotate the test period
            if len(test_dates) > 1:  # Need at least 2 points to safely get midpoint
                mid_point_idx = len(test_dates) // 2
                max_pred = max(test_pred_list) if test_pred_list else 0
                max_actual = max(test_actual_list) if test_actual_list else 0
                max_y = max(max_pred, max_actual)
                
                fig.add_annotation(
                    x=test_dates_list[mid_point_idx],
                    y=max_y * 1.05,
                    text="Test Period",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#888888",
                    font=dict(size=10, color="#888888"),
                    align="center"
                )
    
    # Add model performance summary if available
    # For multi-SKU view, we'll skip this to avoid clutter
    if not is_multi_sku and len(skus) == 1 and skus[0] in st.session_state.forecasts:
        current_data = st.session_state.forecasts[skus[0]]
        if 'model_evaluation' in current_data and current_data['model_evaluation'].get('metrics'):
            model_name = current_data['model'].upper()
            metrics = current_data['model_evaluation']['metrics'].get(current_data['model'], {})
            
            # Prepare annotation text
            metrics_text = f"<b>Model: {model_name}</b><br>"
            if 'rmse' in metrics:
                metrics_text += f"RMSE: {metrics['rmse']:.2f}<br>"
            if 'mape' in metrics and not np.isnan(metrics['mape']):
                metrics_text += f"MAPE: {metrics['mape']:.2f}%<br>"
            if 'mae' in metrics:
                metrics_text += f"MAE: {metrics['mae']:.2f}<br>"
            
            # Add annotation
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=metrics_text,
                showarrow=False,
                font=dict(
                    family="Arial",
                    size=12,
                    color="#000000"
                ),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="#c7c7c7",
                borderwidth=1,
                borderpad=4
            )
    
    # Update layout with more modern style
    # Different title for multi-SKU vs single SKU
    title_text = f"Sales Forecast Comparison for Multiple SKUs" if is_multi_sku else f"Sales Forecast for SKU: {skus[0]}"
    
    fig.update_layout(
        title={
            'text': title_text,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=22)
        },
        xaxis_title="Date",
        yaxis_title="Quantity",
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
    
    # Add grid lines for better readability
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211,211,211,0.5)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211,211,211,0.5)'
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

def plot_model_comparison(sku, forecast_data):
    """
    Create a plotly figure comparing different forecasting models for a specific SKU
    with detailed error metrics visualization - using a clustered bar chart with a line for MAPE
    
    Parameters:
    -----------
    sku : str
        SKU identifier
    forecast_data : dict
        Forecast information for the SKU containing model evaluation data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the model comparison
    """
    # Check if model evaluation data exists
    if 'model_evaluation' not in forecast_data or not forecast_data['model_evaluation'].get('metrics'):
        fig = go.Figure()
        fig.update_layout(title=f"No model comparison data available for SKU: {sku}")
        return fig
    
    # Get model evaluation metrics
    metrics = forecast_data['model_evaluation']['metrics']
    best_model = forecast_data['model_evaluation']['best_model']
    
    # Create data for bar chart
    models = list(metrics.keys())
    rmse_values = [metrics[m].get('rmse', 0) for m in models]
    mape_values = [metrics[m].get('mape', 0) if not np.isnan(metrics[m].get('mape', 0)) else 0 for m in models]
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
        line=dict(width=3, color='#ff7f0e'),
        text=[f"{v:.2f}%" for v in mape_values],
        yaxis='y2'
    ))
    
    # Highlight the best model
    for i, model in enumerate(models):
        if model == best_model:
            fig.add_annotation(
                x=model,
                y=max(rmse_values) * 1.1,
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
        
    # Get all dates in chronological order
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
