import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from datetime import datetime, timedelta
import numpy as np

def plot_forecast(sales_data, forecast_data, sku):
    """
    Create a plotly figure showing historical sales and forecast for a specific SKU
    
    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Historical sales data
    forecast_data : dict
        Forecast information for the SKU
    sku : str
        SKU identifier
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the forecast plot
    """
    # Filter sales data for the specified SKU
    sku_sales = sales_data[sales_data['sku'] == sku].copy()
    
    if len(sku_sales) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title=f"No historical data available for SKU: {sku}")
        return fig
    
    # Ensure data is sorted by date
    sku_sales = sku_sales.sort_values('date')
    
    # Get forecast data
    forecast = forecast_data['forecast']
    lower_bound = forecast_data['lower_bound']
    upper_bound = forecast_data['upper_bound']
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=sku_sales['date'],
        y=sku_sales['quantity'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='blue'),
        marker=dict(size=6),
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red'),
        marker=dict(size=6),
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=list(upper_bound.index) + list(lower_bound.index)[::-1],
        y=list(upper_bound.values) + list(lower_bound.values)[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 0, 0, 0)'),
        name='95% Confidence Interval',
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Sales Forecast for SKU: {sku}",
        xaxis_title="Date",
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
        
        # Add to data
        inventory_data.append({
            'sku': sku,
            'avg_monthly_sales': avg_monthly_sales,
            'next_month_forecast': next_month_forecast,
            'months_of_supply': min(months_of_supply, 12),  # Cap at 12 months for visualization
            'status': status,
            'color': color,
            'demand_variability': cv
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
        title="Inventory Health Matrix",
        color_discrete_map={"Stockout Risk": "red", "Overstocked": "orange", "Healthy": "green"}
    )
    
    # Add reference zones
    fig.add_hrect(y0=0, y1=0.5, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0.5, y1=3, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=3, y1=12, fillcolor="orange", opacity=0.1, line_width=0)
    
    # Update layout
    fig.update_layout(
        xaxis_title="Demand Variability (CV)",
        yaxis_title="Months of Supply",
        template="plotly_white",
        yaxis=dict(range=[0, 12])  # Set y-axis range from 0 to 12 months
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
    base_prod = base_scenario['production_plan']
    what_if_prod = what_if_scenario['production_plan']
    
    if len(base_prod) == 0 or len(what_if_prod) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="No scenario comparison data available")
        return fig
    
    # Group by period
    base_summary = base_prod.groupby('period')['production_quantity'].sum().reset_index()
    what_if_summary = what_if_prod.groupby('period')['production_quantity'].sum().reset_index()
    
    # Merge the data
    comparison = pd.merge(base_summary, what_if_summary, on='period', suffixes=('_base', '_what_if'))
    
    # Calculate percentage change
    comparison['percent_change'] = (comparison['production_quantity_what_if'] - comparison['production_quantity_base']) / comparison['production_quantity_base'] * 100
    
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
