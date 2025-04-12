import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from datetime import datetime, timedelta
import numpy as np
import streamlit as st

@st.cache_data(ttl=1800, show_spinner=False, max_entries=100)  # 30 minute cache for visualizations
def plot_forecast(sales_data, forecast_data, sku=None, selected_models=None, show_anomalies=False, confidence_interval=0.95):
    """
    Create interactive forecast visualization with caching for better performance

    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Historical sales data
    forecast_data : dict
        Forecast information including model data
    sku : str
        SKU identifier
    selected_models : list
        List of models to display on the chart
    show_anomalies : bool, optional
        Whether to highlight anomalies in the data
    confidence_interval : float, optional
        Confidence interval level for forecast bounds (0-1)
    """
    # Create a default figure
    fig = go.Figure()

    # Filter sales data for this SKU and ensure valid filtering
    if not isinstance(sku, str) or sku not in sales_data['sku'].unique():
        # If sku is invalid, return empty figure with error message
        fig.add_annotation(
            text="No valid SKU selected or data unavailable",
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=400)
        return fig

    sku_data = sales_data[sales_data['sku'] == sku].copy()
    sku_data = sku_data.sort_values('date')

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

    # Define model colors using a colorblind-friendly palette
    # Based on ColorBrewer and Okabe-Ito colorblind-safe schemes
    model_colors = {
        'arima': '#2b8cbe',         # Blue
        'sarima': '#7b3294',        # Purple
        'prophet': '#d95f0e',       # Orange
        'lstm': '#e7298a',          # Pink
        'holtwinters': '#1b9e77',   # Teal
        'decomposition': '#66a61e', # Green
        'ensemble': '#e6ab02',      # Yellow
        'moving_average': '#666666',# Gray
        'auto_arima': '#88419d',    # Dark purple
        'ets': '#8c96c6',           # Light purple
        'theta': '#fc8d59'          # Salmon
    }

    # Check if forecast_data is available
    if forecast_data is None:
        # No forecast data available, just return the historical data chart
        fig.update_layout(
            title=f"<b>Historical Sales for {sku}</b>",
            height=400
        )
        return fig

    # Add forecasts for all selected models
    models_added = False

    # First check if we have model_evaluation data with multiple model forecasts
    if (selected_models and forecast_data and 
        isinstance(forecast_data, dict) and 
        'model_evaluation' in forecast_data and 
        'all_models_forecasts' in forecast_data['model_evaluation']):

        all_models_data = forecast_data['model_evaluation']['all_models_forecasts']

        # If we have selected models, use them
        if selected_models:
            for model in selected_models:
                model_key = model.lower()  # Always use lowercase for model keys

                # Check if this model exists in the data
                if model_key in all_models_data:
                    model_forecast = all_models_data[model_key]

                    # Skip if no forecast data available for this model
                    if model_forecast is None or len(model_forecast) == 0:
                        continue

                    # Add this model's forecast to the chart
                    models_added = True
                    color = model_colors.get(model_key, 'red')  # Default to red if color not defined

                    # Assign different line patterns based on model type
                    dash_patterns = {
                        'arima': 'solid',
                        'sarima': 'dash',
                        'prophet': 'dot',
                        'lstm': 'dashdot',
                        'holtwinters': 'longdash',
                        'decomposition': 'longdashdot',
                        'ensemble': 'solid',
                        'moving_average': 'dash',
                        'auto_arima': 'dot',
                        'ets': 'dashdot',
                        'theta': 'longdash'
                    }

                    # Assign different marker symbols for additional differentiation
                    marker_symbols = {
                        'arima': 'circle',
                        'sarima': 'square',
                        'prophet': 'diamond',
                        'lstm': 'triangle-up',
                        'holtwinters': 'pentagon',
                        'decomposition': 'star',
                        'ensemble': 'hexagon',
                        'moving_average': 'cross',
                        'auto_arima': 'circle-open',
                        'ets': 'square-open',
                        'theta': 'diamond-open'
                    }

                    dash_pattern = dash_patterns.get(model_key, 'solid')
                    marker_symbol = marker_symbols.get(model_key, 'circle')

                    fig.add_trace(
                        go.Scatter(
                            x=model_forecast.index,
                            y=model_forecast.values,
                            mode='lines+markers',
                            name=f"{model_key.upper()} Forecast",
                            line=dict(color=color, dash=dash_pattern, width=2),
                            marker=dict(size=8, symbol=marker_symbol),
                            hovertemplate='%{x|%b %Y}: %{y:,.0f} units'
                        )
                    )

    # If no models from the selection were added, fallback to the best model
    if not models_added and isinstance(forecast_data, dict) and 'forecast' in forecast_data:
        forecast_values = forecast_data['forecast']
        if forecast_values is not None and len(forecast_values) > 0:
            model_name = forecast_data.get('model', 'Default').upper()
            model_key = model_name.lower()
            color = model_colors.get(model_key, 'red')

            # Get dash pattern and marker symbol for this model
            dash_patterns = {
                'arima': 'solid',
                'sarima': 'dash',
                'prophet': 'dot',
                'lstm': 'dashdot',
                'holtwinters': 'longdash',
                'decomposition': 'longdashdot',
                'ensemble': 'solid',
                'moving_average': 'dash',
                'auto_arima': 'dot',
                'ets': 'dashdot',
                'theta': 'longdash'
            }

            marker_symbols = {
                'arima': 'circle',
                'sarima': 'square',
                'prophet': 'diamond',
                'lstm': 'triangle-up',
                'holtwinters': 'pentagon',
                'decomposition': 'star',
                'ensemble': 'hexagon',
                'moving_average': 'cross',
                'auto_arima': 'circle-open',
                'ets': 'square-open',
                'theta': 'diamond-open'
            }

            dash_pattern = dash_patterns.get(model_key, 'solid')
            marker_symbol = marker_symbols.get(model_key, 'circle')

            fig.add_trace(
                go.Scatter(
                    x=forecast_values.index,
                    y=forecast_values.values,
                    mode='lines+markers',
                    name=f"{model_name} Forecast",
                    line=dict(color=color, dash=dash_pattern, width=2),
                    marker=dict(size=8, symbol=marker_symbol),
                    hovertemplate='%{x|%b %Y}: %{y:,.0f} units'
                )
            )
            models_added = True

    # If still no models were added, show a message
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
                if 'test_set' in forecast_data:
                    test_set = forecast_data['test_set']

                # Get test predictions for the selected models
                if selected_models and 'all_models_test_pred' in forecast_data['model_evaluation']:
                    for model in selected_models:
                        model_key = model.lower()
                        if model_key in forecast_data['model_evaluation']['all_models_test_pred']:
                            test_predictions = forecast_data['model_evaluation']['all_models_test_pred'][model_key]

                            # Choose a color for this model that matches the forecast color
                            color = model_colors.get(model_key, 'orange')

                            # Make sure test_predictions is valid data
                            if test_predictions is None or not isinstance(test_predictions, pd.Series) or test_predictions.empty:
                                continue

                            # Get appropriate dash pattern and marker for this model
                            dash_patterns = {
                                'arima': 'solid',
                                'sarima': 'dash',
                                'prophet': 'dot',
                                'lstm': 'dashdot',
                                'holtwinters': 'longdash',
                                'decomposition': 'longdashdot',
                                'ensemble': 'solid',
                                'moving_average': 'dash',
                                'auto_arima': 'dot',
                                'ets': 'dashdot',
                                'theta': 'longdash'
                            }

                            marker_symbols = {
                                'arima': 'circle',
                                'sarima': 'square',
                                'prophet': 'diamond',
                                'lstm': 'triangle-up',
                                'holtwinters': 'pentagon',
                                'decomposition': 'star',
                                'ensemble': 'hexagon',
                                'moving_average': 'cross',
                                'auto_arima': 'circle-open',
                                'ets': 'square-open',
                                'theta': 'diamond-open'
                            }

                            # For test predictions, use the same pattern but lighter
                            dash_pattern = dash_patterns.get(model_key, 'dot')
                            marker_symbol = marker_symbols.get(model_key, 'circle')

                            fig.add_trace(
                                go.Scatter(
                                    x=test_predictions.index,
                                    y=test_predictions.values,
                                    mode='lines+markers',
                                    name=f"{model_key.upper()} Test Predictions",
                                    line=dict(color=color, dash='dot', width=1.5),
                                    marker=dict(size=6, symbol=marker_symbol),
                                    hovertemplate='%{x|%b %Y}: %{y:,.0f} units'
                                )
                            )
                elif 'test_predictions' in forecast_data['model_evaluation']:
                    # Fall back to the best model's predictions
                    test_predictions = forecast_data['model_evaluation']['test_predictions']

                    # Make sure test_predictions is valid before plotting
                    if test_predictions is not None and isinstance(test_predictions, pd.Series) and not test_predictions.empty:
                        # Use the color of the best model
                        best_model = forecast_data.get('model', 'default').lower()
                        color = model_colors.get(best_model, 'orange')

                        fig.add_trace(
                            go.Scatter(
                                x=test_predictions.index,
                                y=test_predictions.values,
                                mode='lines+markers',
                                name=f"{best_model.upper()} Test Predictions",
                                line=dict(color=color, dash='dot'),
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

    # Setup figure layout
    try:
        # Determine date range for the chart
        if isinstance(forecast_data, dict) and 'forecast' in forecast_data and forecast_data['forecast'] is not None and len(forecast_data['forecast']) > 0:
            forecast_end = forecast_data['forecast'].index.max()
            months = pd.date_range(start=sku_data['date'].min(), end=forecast_end, freq='MS')
            has_forecast = True
        else:
            # Only use historical data for date range
            months = pd.date_range(start=sku_data['date'].min(), end=sku_data['date'].max(), freq='MS')
            has_forecast = False

        # Create a comprehensive layout
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

        # Add forecast divider line if we have forecast data
        if has_forecast:
            # Get the last historical date
            max_date = sku_data['date'].max()

            # Add a vertical line separating historical and forecast periods
            fig.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=max_date,
                y0=0,
                x1=max_date,
                y1=1,
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
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.8)",
                font=dict(size=12)
            )

            # Add light shading to the forecast area
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=max_date,
                y0=0,
                x1=forecast_end,
                y1=1,
                fillcolor="rgba(200, 200, 200, 0.1)",
                layer="below",
                line_width=0,
            )
    except Exception as e:
        # If layout creation fails, use a simple default layout
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

def plot_model_comparison(selected_sku=None, forecast_data=None, models_to_show=None):
    """
    Create a plotly figure comparing different forecasting models for a specific SKU
    with detailed error metrics visualization - using a clustered bar chart with a line for MAPE

    Parameters:
    -----------
    selected_sku: str, optional
        The SKU identifier to display in the chart title
    forecast_data : dict
        Dictionary containing model evaluation data with metrics for each model
    models_to_show : list, optional
        List of model names to include in the visualization

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the model comparison
    """
    # Create default figure
    fig = go.Figure()

    # Check if forecast_data is valid and has model_evaluation
    if not isinstance(forecast_data, dict) or 'model_evaluation' not in forecast_data or not forecast_data['model_evaluation']:
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

    # Get model metrics from model_evaluation
    model_comparison = {}
    if 'metrics' in forecast_data['model_evaluation']:
        model_comparison = forecast_data['model_evaluation']['metrics']

    # Check if we have valid model metrics
    if not model_comparison:
        fig.update_layout(
            title="No model metrics available",
            annotations=[{
                'text': "No model metrics data available for comparison",
                'showarrow': False,
                'font': {'size': 16},
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return fig

    # Get the selected (best) model
    selected_model = forecast_data.get('model_evaluation', {}).get('best_model', None)
    if not selected_model and 'model' in forecast_data:
        selected_model = forecast_data['model']

    # Get metrics for each model
    metrics = {}
    for model_name, model_metrics in model_comparison.items():
        if isinstance(model_metrics, dict):
            metrics[model_name] = model_metrics

    # If no valid metrics found
    if not metrics:
        fig.update_layout(
            title="Invalid model metrics data",
            annotations=[{
                'text': "Model metrics data is not in the expected format",
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

    # Get SKU name for title
    sku_name = selected_sku if selected_sku else "Unknown SKU"

    # Define a consistent model order based on common forecasting methodologies
    model_order = [
        'auto_arima', 'arima', 'sarima', 'prophet', 'decomposition', 
        'holtwinters', 'ets', 'theta', 'moving_average', 'lstm', 'ensemble'
    ]

    # Determine which models to display
    available_models = list(metrics.keys())

    if models_to_show and isinstance(models_to_show, list) and len(models_to_show) > 0:
        # Filter models by those in models_to_show that are also available
        filtered_models = [m for m in models_to_show if m.lower() in available_models]
        models_to_use = filtered_models if filtered_models else available_models
    else:
        models_to_use = available_models

    # Sort models based on the predefined order
    sorted_models = []
    for model in model_order:
        if model in models_to_use:
            sorted_models.append(model)

    # Add any remaining models not in the predefined order
    for model in models_to_use:
        if model not in sorted_models:
            sorted_models.append(model)

    # Extract metrics for the sorted models
    rmse_values = [metrics[m].get('rmse', 0) for m in sorted_models]
    mape_values = [metrics[m].get('mape', 0) if not pd.isna(metrics[m].get('mape', 0)) else 0 for m in sorted_models]
    mae_values = [metrics[m].get('mae', 0) for m in sorted_models]

    # Prettify model names for display
    display_models = [m.upper() for m in sorted_models]

    # Create figure with multiple traces
    fig = go.Figure()

    # Add RMSE bars
    fig.add_trace(go.Bar(
        x=display_models,
        y=rmse_values,
        name='RMSE',
        marker_color='#1f77b4',
        text=[f"{v:.2f}" for v in rmse_values],
        textposition='auto',
    ))

    # Add MAE bars (clustered with RMSE)
    fig.add_trace(go.Bar(
        x=display_models,
        y=mae_values,
        name='MAE',
        marker_color='#2ca02c',
        text=[f"{v:.2f}" for v in mae_values],
        textposition='auto',
    ))

    # Add MAPE as a line chart on secondary y-axis
    # Use smoothed line with better color and thickness
    fig.add_trace(go.Scatter(
        x=display_models,
        y=mape_values,
        name='MAPE (%)',
        mode='lines+markers',
        marker=dict(
            size=10, 
            color='#ff7f0e',
            symbol='circle',
            line=dict(width=2, color='#e65c00')
        ),
        line=dict(
            width=4, 
            color='#ff7f0e', 
            shape='spline',  # Use spline for smoother curves
            smoothing=0.3    # Add smoothing for a nicer curve
        ),
        text=[f"{v:.2f}%" for v in mape_values],
        yaxis='y2'
    ))

    # Highlight the best model if it exists
    for i, model in enumerate(sorted_models):
        if model.lower() == selected_model.lower():
            # Only add annotation if there are RMSE values
            if rmse_values and max(rmse_values) > 0:
                y_pos = max(rmse_values) * 1.1
                fig.add_annotation(
                    x=display_models[i],
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
            'text': f"Model Comparison for SKU: {sku_name}",
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
            # Use the provided order instead of auto-ordering
            categoryorder='array',
            categoryarray=display_models
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

#Data processing and visualization functions
def highlight_data_columns(row):
    # Define a color map for highlighting the SKU information columns
    color_map = {
        'sku_code': 'lightblue',
        'sku_name': 'lightgreen',
        'model': 'lightyellow',
        'best_model': 'lightcoral'
    }

    # Style the row based on the column name
    styled_row = []
    for col, val in row.items():
        if col in color_map:
            styled_row.append(f'<td style="background-color: {color_map[col]}; position: sticky; left: 0; z-index: 1;">{val}</td>')
        else:
            styled_row.append(f'<td>{val}</td>')
    return "".join(styled_row)

def get_table_styles():
    """
    Returns custom CSS styles for dataframe tables with frozen columns
    """
    return [
        {'selector': 'thead th:nth-child(-n+4)', 'props': 'position: sticky; left: 0; z-index: 3; background-color: white; box-shadow: 2px 0px 3px rgba(0,0,0,0.1);'},
        {'selector': 'tbody td:nth-child(-n+4)', 'props': 'position: sticky; left: 0; z-index: 2; background-color: white; box-shadow: 2px 0px 3px rgba(0,0,0,0.1);'},
        {'selector': 'thead th', 'props': 'position: sticky; top: 0; z-index: 1; background-color: white; box-shadow: 0px 1px 3px rgba(0,0,0,0.1);'},
        {'selector': 'thead th:nth-child(-n+4)', 'props': 'position: sticky; top: 0; left: 0; z-index: 4; background-color: white; box-shadow: 2px 2px 3px rgba(0,0,0,0.1);'}
    ]

# Remove page config and demo UI elements that interfere with importing this module

# Example function for demo/standalone usage
def create_demo_dashboard(title="Demand Forecasting Dashboard"):
    st.title(title)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        sales_data = pd.read_csv(uploaded_file)
        sales_data['date'] = pd.to_datetime(sales_data['date'])

        # --- DATA PREPROCESSING ---
        # Convert date column to datetime
        if 'date' in sales_data.columns and not pd.api.types.is_datetime64_any_dtype(sales_data['date']):
            try:
                sales_data['date'] = pd.to_datetime(sales_data['date'])
            except ValueError:
                st.error("Invalid date format. Please ensure your 'date' column is in a valid format (e.g., YYYY-MM-DD).")
                st.stop()

    # Check for required columns in sales data
    required_columns = ['sku', 'date', 'quantity']
    if not all(col in sales_data.columns for col in required_columns):
        st.error(f"Sales data must contain columns: {', '.join(required_columns)}")
        st.stop()

    # --- DATA EXPLORATION AND VISUALIZATION ---
    # Select SKU for analysis
    selected_sku = st.selectbox("Select SKU", sales_data['sku'].unique())

    # Filter sales data for the selected SKU
    selected_sku_data = sales_data[sales_data['sku'] == selected_sku]

    # --- FORECASTING ---
    # Placeholder for forecast results (replace with your forecasting logic)
    forecast_results = {
        selected_sku: {
            'forecast': pd.Series([100, 110, 120, 130, 140], index=pd.date_range(start='2024-01-01', periods=5, freq='M')),
            'lower_bound': pd.Series([90, 100, 110, 120, 130], index=pd.date_range(start='2024-01-01', periods=5, freq='M')),
            'upper_bound': pd.Series([110, 120, 130, 140, 150], index=pd.date_range(start='2024-01-01', periods=5, freq='M')),
            'model': 'ExampleModel'
        }
    }

    # --- VISUALIZATIONS ---
    # Forecast plot
    forecast_plot = plot_forecast(sales_data, forecast_results, sku=selected_sku)
    st.plotly_chart(forecast_plot, use_container_width=True)



    # --- MODEL COMPARISON ---
    # Placeholder for model comparison results
    model_comparison_results = {
        'model_evaluation': {
            'metrics': {
                'model1': {'rmse': 10, 'mape': 5, 'mae': 8},
                'model2': {'rmse': 12, 'mape': 7, 'mae': 9},
                'model3': {'rmse': 8, 'mape': 3, 'mae': 6}
            },
            'best_model': 'model3'
        }
    }

    model_comparison_plot = plot_model_comparison(selected_sku, model_comparison_results)
    st.plotly_chart(model_comparison_plot, use_container_width=True)

    # --- ERROR ANALYSIS ---
    # Placeholder for error analysis results
    error_analysis_results = {
        'test_set': pd.Series([100, 110, 120, 130, 140]),
        'test_predictions': pd.Series([98, 105, 125, 135, 138])
    }
    error_analysis_plot = plot_forecast_error_distribution(selected_sku, error_analysis_results)
    st.plotly_chart(error_analysis_plot, use_container_width=True)


    # --- FORECAST ACCURACY ---
    # Placeholder for forecast accuracy results
    forecast_accuracy_results = {
        selected_sku: {'forecast': pd.Series([100, 110, 120, 130, 140], index=pd.date_range(start='2024-01-01', periods=5, freq='M'))}
    }
    forecast_accuracy_plot = plot_forecast_accuracy(selected_sku_data, forecast_accuracy_results)
    st.plotly_chart(forecast_accuracy_plot, use_container_width=True)


    # --- INVENTORY MANAGEMENT ---
    # Placeholder for inventory data (replace with your actual data)
    inventory_data = pd.DataFrame({
        'sku': ['A', 'B', 'C', 'D'],
        'avg_monthly_sales': [100, 50, 200, 150],
        'months_of_supply': [1, 2, 0.5, 3],
        'demand_variability': [0.2, 0.5, 0.1, 0.3],
        'growth_rate': [5, -2, 10, 0],
        'risk_score': [20, 60, 10, 40],
        'importance': [1000, 500, 2000, 1500]
    })
    inventory_health_plot = plot_inventory_health(sales_data, forecast_results)
    st.plotly_chart(inventory_health_plot, use_container_width=True)

    inventory_risk_matrix_plot = plot_inventory_risk_matrix(inventory_data)
    st.plotly_chart(inventory_risk_matrix_plot, use_container_width=True)

    # --- WHAT-IF ANALYSIS ---
    base_scenario = {'production_plan': pd.DataFrame({'period': ['Jan 2024', 'Feb 2024', 'Mar 2024'], 'production_quantity': [100, 120, 110]})}
    what_if_scenario = {'production_plan': pd.DataFrame({'period': ['Jan 2024', 'Feb 2024', 'Mar 2024'], 'production_quantity': [110, 130, 120]})}
    what_if_comparison_plot = plot_what_if_comparison(base_scenario, what_if_scenario)
    st.plotly_chart(what_if_comparison_plot, use_container_width=True)

    # --- SKU SUMMARY ---
    # Create a sample DataFrame for SKU summary
    all_sku_df = pd.DataFrame({
        'sku_code': ['SKU1', 'SKU2', 'SKU3', 'SKU4'],
        'sku_name': ['Product A', 'Product B', 'Product C', 'Product D'],
        'model': ['ARIMA', 'Prophet', 'SARIMA', 'LSTM'],
        'best_model': ['True', 'False', 'False', 'True'],
        'rmse': [10, 12, 8, 6],
        'mape': [5, 7, 3, 4],
        'mae': [8, 9, 6, 5]
    })

    # Use styling to highlight data column types with frozen columns till model name
    styled_df = all_sku_df.style.apply(highlight_data_columns, axis=None)

    # Add frozen panes - freeze the first 4 columns and the header row
    styled_df = styled_df.set_sticky(axis="index", levels=[0])  # Freeze header row
    styled_df = styled_df.set_sticky(axis="columns", levels=list(range(4)))  # Freeze first 4 columns

    # Add enhanced styling for better visibility of frozen areas
    styled_df = styled_df.set_table_styles(get_table_styles(), overwrite=True)

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=600,  # Increased height for better visibility
        column_config={
            # Configure the info columns (SKU code, SKU name, model, best model)
            "sku_code": st.column_config.TextColumn(
                "SKU Code",
                width="medium",
                help="Unique identifier for the SKU"
            ),
            "sku_name": st.column_config.TextColumn(
                "SKU Name",
                width="medium", 
                help="Name of the SKU"
            ),
            "model": st.column_config.TextColumn(
                "Model",
                width="medium",
                help="Forecasting model used"
            ),
            "best_model": st.column_config.TextColumn(
                "Best",
                width="small",
                help="Check mark indicates best performing model"
            )
        },
        hide_index=True
    )

else:
    st.write("No CSV file uploaded.")