import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta
from utils.visualization import (
    plot_forecast_accuracy, 
    plot_inventory_health, 
    plot_forecast_accuracy_trend,
    plot_inventory_risk_matrix
)

# Set page config
st.set_page_config(
    page_title="KPI Dashboard",
    page_icon="📊",
    layout="wide"
)

# Check if data is loaded in session state
if 'sales_data' not in st.session_state or st.session_state.sales_data is None:
    st.warning("Please upload sales data on the main page first.")
    st.stop()

if 'forecasts' not in st.session_state or not st.session_state.forecasts:
    st.warning("Please run forecast analysis on the Demand Forecasting page first.")
    st.stop()

# Page title
st.title("Supply Chain KPI Dashboard")
st.markdown("""
This dashboard provides real-time insights into your key performance indicators (KPIs).
Monitor forecast accuracy, inventory health, and supplier performance to make data-driven decisions.
""")

# Create metrics summary at the top
st.header("Supply Chain Overview")

# Calculate key metrics
total_skus = len(st.session_state.sales_data['sku'].unique())
total_materials = len(st.session_state.bom_data['material_id'].unique()) if 'bom_data' in st.session_state and st.session_state.bom_data is not None else 0
total_suppliers = len(st.session_state.supplier_data['supplier_id'].unique()) if 'supplier_data' in st.session_state and st.session_state.supplier_data is not None and 'supplier_id' in st.session_state.supplier_data.columns else 0

# Get forecast accuracy if we have actual data to compare with
forecast_accuracy = 0
forecasted_skus = len(st.session_state.forecasts)

# Create metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total SKUs",
        value=total_skus,
        delta=f"{forecasted_skus} forecasted"
    )

with col2:
    st.metric(
        label="Total Materials",
        value=total_materials,
        delta=None
    )

with col3:
    st.metric(
        label="Total Suppliers",
        value=total_suppliers,
        delta=None
    )

with col4:
    # Calculate the date of the latest data
    if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
        last_update = st.session_state.sales_data['date'].max().strftime('%Y-%m-%d')
    else:
        last_update = "N/A"
    
    st.metric(
        label="Last Data Update",
        value=last_update,
        delta=None
    )

# Create tabs for different KPI categories
tab1, tab2, tab3 = st.tabs(["Forecast Accuracy", "Inventory Health", "Supplier Performance"])

with tab1:
    st.header("Forecast Accuracy Analysis")
    
    # Instructions for evaluating forecast accuracy
    st.markdown("""
    Forecast accuracy compares predicted values against actual outcomes. To evaluate:
    1. Upload recent sales data that covers periods you previously forecasted
    2. The system will automatically calculate accuracy metrics
    """)
    
    # Option to upload actual data for comparison
    st.subheader("Upload Actual Sales Data")
    actuals_file = st.file_uploader("Upload recent sales data (Excel)", type=["xlsx", "xls"], key="actuals_upload")
    
    if actuals_file is not None:
        try:
            # Process the actuals data
            from utils.data_processor import process_sales_data
            actuals_data = process_sales_data(actuals_file)
            
            # Display sample of the data
            st.success(f"Successfully loaded actual sales data with {len(actuals_data)} records!")
            st.write("Preview of actuals data:")
            st.dataframe(actuals_data.head(), use_container_width=True)
            
            # Create accuracy analysis tabs
            acc_tab1, acc_tab2, acc_tab3 = st.tabs(["Accuracy by SKU", "Accuracy Trend", "Model Performance"])
            
            with acc_tab1:
                # Generate forecast accuracy chart
                st.subheader("Forecast Accuracy by SKU")
                accuracy_fig = plot_forecast_accuracy(actuals_data, st.session_state.forecasts)
                st.plotly_chart(accuracy_fig, use_container_width=True)
                
                # Calculate aggregate accuracy metrics
                st.subheader("Accuracy Metrics")
                
                # Add MAPE and MAE for each model and SKU
                st.subheader("Model Accuracy by SKU")
                
                # Calculate accuracy metrics for each model/SKU combination
                model_metrics = []
                
                for sku, forecast_data in st.session_state.forecasts.items():
                    sku_actuals = actuals_data[actuals_data['sku'] == sku]
                    if len(sku_actuals) == 0:
                        continue
                    
                    model_type = forecast_data.get('model', 'unknown')
                    forecast = forecast_data['forecast']
                    
                    # Find overlapping dates
                    actual_dates = set(sku_actuals['date'])
                    forecast_dates = set(forecast.index)
                    common_dates = actual_dates.intersection(forecast_dates)
                    
                    if not common_dates:
                        continue
                    
                    # Calculate MAPE and MAE
                    total_error = 0
                    total_abs_error = 0
                    total_abs_pct_error = 0
                    count = 0
                    
                    for date in common_dates:
                        actual = sku_actuals[sku_actuals['date'] == date]['quantity'].iloc[0]
                        predicted = forecast.get(date, 0)
                        
                        if actual > 0:  # Avoid division by zero
                            error = predicted - actual
                            abs_error = abs(error)
                            abs_pct_error = abs_error / actual
                            
                            total_error += error
                            total_abs_error += abs_error
                            total_abs_pct_error += abs_pct_error
                            count += 1
                    
                    if count > 0:
                        mae = total_abs_error / count
                        mape = (total_abs_pct_error / count) * 100
                        
                        model_metrics.append({
                            'SKU': sku,
                            'Model': model_type,
                            'MAPE (%)': f"{mape:.2f}%",
                            'MAE': f"{mae:.2f}"
                        })
                
                if model_metrics:
                    # Convert to DataFrame and display
                    metrics_df = pd.DataFrame(model_metrics)
                    st.dataframe(metrics_df, use_container_width=True)
                else:
                    st.info("No data available to calculate model metrics")
                
                # Function to calculate MAPE
                def calculate_mape(actuals, forecasts):
                    mape_sum = 0
                    count = 0
                    
                    for sku, forecast_data in forecasts.items():
                        sku_actuals = actuals[actuals['sku'] == sku]
                        if len(sku_actuals) == 0:
                            continue
                        
                        forecast = forecast_data['forecast']
                        
                        # Find overlapping dates
                        actual_dates = set(sku_actuals['date'])
                        forecast_dates = set(forecast.index)
                        common_dates = actual_dates.intersection(forecast_dates)
                        
                        for date in common_dates:
                            actual = sku_actuals[sku_actuals['date'] == date]['quantity'].iloc[0]
                            predicted = forecast.get(date, 0)
                            
                            if actual > 0:  # Avoid division by zero
                                mape_sum += abs(actual - predicted) / actual
                                count += 1
                    
                    return (mape_sum / count) * 100 if count > 0 else 0
                
                mape = calculate_mape(actuals_data, st.session_state.forecasts)
                accuracy = max(0, 100 - mape)  # Convert MAPE to accuracy percentage
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Average Forecast Accuracy",
                        value=f"{accuracy:.1f}%",
                        delta=f"{accuracy - 80:.1f}%" if accuracy > 80 else f"{accuracy - 80:.1f}%",
                        delta_color="normal" if accuracy >= 80 else "inverse"
                    )
                
                with col2:
                    st.metric(
                        label="MAPE (Mean Absolute Percentage Error)",
                        value=f"{mape:.1f}%",
                        delta=f"{20 - mape:.1f}%" if mape < 20 else f"{20 - mape:.1f}%",
                        delta_color="normal" if mape <= 20 else "inverse"
                    )
                
                with col3:
                    # Count SKUs with good accuracy (>80%)
                    accuracy_data = []
                    for sku, forecast_data in st.session_state.forecasts.items():
                        sku_actuals = actuals_data[actuals_data['sku'] == sku]
                        if len(sku_actuals) == 0:
                            continue
                        
                        forecast = forecast_data['forecast']
                        
                        # Find overlapping dates
                        actual_dates = set(sku_actuals['date'])
                        forecast_dates = set(forecast.index)
                        common_dates = actual_dates.intersection(forecast_dates)
                        
                        if not common_dates:
                            continue
                        
                        # Calculate accuracy for this SKU
                        mape_sum = 0
                        count = 0
                        
                        for date in common_dates:
                            actual = sku_actuals[sku_actuals['date'] == date]['quantity'].iloc[0]
                            predicted = forecast.get(date, 0)
                            
                            if actual > 0:  # Avoid division by zero
                                mape_sum += abs(actual - predicted) / actual
                                count += 1
                        
                        if count > 0:
                            sku_mape = mape_sum / count
                            sku_accuracy = max(0, 100 - sku_mape * 100)
                            accuracy_data.append((sku, sku_accuracy))
                    
                    good_accuracy_count = sum(1 for _, acc in accuracy_data if acc >= 80)
                    st.metric(
                        label="SKUs with >80% Accuracy",
                        value=f"{good_accuracy_count}",
                        delta=f"{good_accuracy_count / len(accuracy_data) * 100:.1f}%" if accuracy_data else "N/A"
                    )
            
            with acc_tab2:
                # Show accuracy trend over time
                st.subheader("Forecast Accuracy Trend")
                trend_fig = plot_forecast_accuracy_trend(actuals_data, st.session_state.forecasts)
                st.plotly_chart(trend_fig, use_container_width=True)
                
                # Add explanation
                st.markdown("""
                **Understanding the Accuracy Trend:**
                
                - The green line shows forecast accuracy over time (higher is better)
                - The red dotted line shows MAPE over time (lower is better)
                - The horizontal green line indicates the target accuracy of 80%
                
                **Key Insights:**
                - Consistent trends above the target line indicate reliable forecasts
                - Downward trends may indicate changing market conditions
                - Seasonal patterns in accuracy suggest the need for seasonal model adjustments
                """)
            
            with acc_tab3:
                # Show model performance comparison
                st.subheader("Model Performance Comparison")
                
                # Calculate accuracy by model type
                model_performance = []
                model_types = set()
                
                for sku, forecast_data in st.session_state.forecasts.items():
                    sku_actuals = actuals_data[actuals_data['sku'] == sku]
                    if len(sku_actuals) == 0:
                        continue
                    
                    model_type = forecast_data.get('model', 'unknown')
                    model_types.add(model_type)
                    forecast = forecast_data['forecast']
                    
                    # Find overlapping dates
                    actual_dates = set(sku_actuals['date'])
                    forecast_dates = set(forecast.index)
                    common_dates = actual_dates.intersection(forecast_dates)
                    
                    if not common_dates:
                        continue
                    
                    # Calculate accuracy for this SKU
                    mape_sum = 0
                    count = 0
                    
                    for date in common_dates:
                        actual = sku_actuals[sku_actuals['date'] == date]['quantity'].iloc[0]
                        predicted = forecast.get(date, 0)
                        
                        if actual > 0:  # Avoid division by zero
                            mape_sum += abs(actual - predicted) / actual
                            count += 1
                    
                    if count > 0:
                        sku_mape = mape_sum / count
                        sku_accuracy = max(0, 100 - sku_mape * 100)
                        model_performance.append({
                            'sku': sku,
                            'model': model_type,
                            'accuracy': sku_accuracy,
                            'mape': sku_mape * 100
                        })
                
                if model_performance:
                    # Convert to DataFrame
                    model_perf_df = pd.DataFrame(model_performance)
                    
                    # Calculate average accuracy by model type
                    model_avg = model_perf_df.groupby('model').agg({
                        'accuracy': ['mean', 'min', 'max', 'count'],
                        'mape': 'mean'
                    }).reset_index()
                    
                    model_avg.columns = ['Model', 'Avg Accuracy', 'Min Accuracy', 'Max Accuracy', 'SKU Count', 'Avg MAPE']
                    model_avg = model_avg.sort_values('Avg Accuracy', ascending=False)
                    
                    # Format percentages
                    model_avg['Avg Accuracy'] = model_avg['Avg Accuracy'].round(1).astype(str) + '%'
                    model_avg['Min Accuracy'] = model_avg['Min Accuracy'].round(1).astype(str) + '%'
                    model_avg['Max Accuracy'] = model_avg['Max Accuracy'].round(1).astype(str) + '%'
                    model_avg['Avg MAPE'] = model_avg['Avg MAPE'].round(1).astype(str) + '%'
                    
                    # Display table
                    st.dataframe(model_avg, use_container_width=True)
                    
                    # Create accuracy distribution by model chart
                    fig = px.box(
                        model_perf_df,
                        x='model',
                        y='accuracy',
                        color='model',
                        title='Accuracy Distribution by Model Type',
                        labels={
                            'model': 'Forecasting Model',
                            'accuracy': 'Accuracy (%)'
                        }
                    )
                    
                    # Add reference line at 80%
                    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target Accuracy (80%)")
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Model Type",
                        yaxis_title="Accuracy (%)",
                        yaxis=dict(range=[0, 100]),
                        template="plotly_white"
                    )
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data to compare model performance")
                
        except Exception as e:
            st.error(f"Error processing actuals data: {str(e)}")
    else:
        # If no actuals data is uploaded, show a placeholder
        st.info("Please upload actual sales data to calculate forecast accuracy metrics")
        
        # Show recommended accuracy targets
        st.subheader("Recommended Accuracy Targets")
        
        targets_data = {
            'SKU Category': ['A (High Value)', 'B (Medium Value)', 'C (Low Value)'],
            'Target Accuracy': ['95%', '85%', '75%'],
            'Acceptable MAPE': ['5%', '15%', '25%']
        }
        
        st.table(pd.DataFrame(targets_data))

with tab2:
    st.header("Inventory Health Monitor")
    
    # Create inventory analysis tabs
    inv_tab1, inv_tab2, inv_tab3 = st.tabs(["Inventory Status", "Risk Assessment", "Actionable Insights"])
    
    with inv_tab1:
        # Show inventory health matrix
        st.subheader("Inventory Health Matrix")
        inventory_fig = plot_inventory_health(st.session_state.sales_data, st.session_state.forecasts)
        st.plotly_chart(inventory_fig, use_container_width=True)
        
        # Add explanation of the chart
        st.markdown("""
        **Understanding the Inventory Health Matrix:**
        
        - **Horizontal Axis:** Demand Variability (CV) - Higher values mean less predictable demand
        - **Vertical Axis:** Months of Supply - Inventory coverage based on forecasted demand
        - **Bubble Size:** Average Monthly Sales Volume
        - **Color:**
            - 🟢 **Green:** Healthy inventory levels (0.5-3 months supply)
            - 🔴 **Red:** Stockout Risk (< 0.5 months supply)
            - 🟠 **Orange:** Overstocked (> 3 months supply)
        """)
    
    # Calculate detailed inventory metrics
    if 'bom_data' in st.session_state and st.session_state.bom_data is not None:
        # Prepare inventory data
        inventory_data = []
        
        for sku in st.session_state.sales_data['sku'].unique():
            if sku not in st.session_state.forecasts:
                continue
                
            # Get sales data for this SKU
            sku_sales = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku].copy()
            
            if len(sku_sales) < 3:  # Need at least a few data points
                continue
            
            # Calculate average monthly sales
            sku_sales['month'] = sku_sales['date'].dt.to_period('M')
            monthly_sales = sku_sales.groupby('month')['quantity'].sum().reset_index()
            avg_monthly_sales = monthly_sales['quantity'].mean()
            
            if avg_monthly_sales == 0:
                continue
            
            # Get forecast data
            next_month_forecast = st.session_state.forecasts[sku]['forecast'].iloc[0] if len(st.session_state.forecasts[sku]['forecast']) > 0 else 0
            
            # Calculate months of supply
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
                'status': status,
                'months_of_supply': months_of_supply,
                'avg_monthly_sales': avg_monthly_sales,
                'demand_variability': cv,
                'growth_rate': growth_rate,
                'risk_score': risk_score,
                'importance': importance_score
            })
        
        # Convert to DataFrame
        if inventory_data:
            inventory_df = pd.DataFrame(inventory_data)
            
            with inv_tab1:
                st.subheader("Inventory Status Summary")
                
                # Get counts by status
                status_counts = inventory_df['status'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                
                # Calculate percentages
                total = status_counts['Count'].sum()
                status_counts['Percentage'] = (status_counts['Count'] / total * 100).round(1)
                status_counts['Percentage'] = status_counts['Percentage'].astype(str) + '%'
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                healthy_count = status_counts[status_counts['Status'] == 'Healthy']['Count'].iloc[0] if 'Healthy' in status_counts['Status'].values else 0
                stockout_count = status_counts[status_counts['Status'] == 'Stockout Risk']['Count'].iloc[0] if 'Stockout Risk' in status_counts['Status'].values else 0
                overstock_count = status_counts[status_counts['Status'] == 'Overstocked']['Count'].iloc[0] if 'Overstocked' in status_counts['Status'].values else 0
                
                with col1:
                    st.metric(
                        label="Healthy Inventory Items",
                        value=healthy_count,
                        delta=f"{(healthy_count / total * 100):.1f}%"
                    )
                
                with col2:
                    st.metric(
                        label="Stockout Risk Items",
                        value=stockout_count,
                        delta=f"{(stockout_count / total * 100):.1f}%",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        label="Overstocked Items",
                        value=overstock_count,
                        delta=f"{(overstock_count / total * 100):.1f}%",
                        delta_color="inverse"
                    )
                
                # Display chart
                fig = px.pie(
                    status_counts,
                    values='Count',
                    names='Status',
                    title='Inventory Status Distribution',
                    color='Status',
                    color_discrete_map={'Healthy': 'green', 'Stockout Risk': 'red', 'Overstocked': 'orange'},
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display items at risk
                st.subheader("Items at Risk")
                at_risk = inventory_df[inventory_df['status'] == 'Stockout Risk'].sort_values('avg_monthly_sales', ascending=False)
                
                if len(at_risk) > 0:
                    st.dataframe(at_risk[['sku', 'months_of_supply', 'avg_monthly_sales']], use_container_width=True)
                else:
                    st.info("No items at stockout risk detected")
            
            with inv_tab2:
                st.subheader("Inventory Risk-Impact Matrix")
                
                # Create risk-impact matrix
                risk_fig = plot_inventory_risk_matrix(inventory_df)
                st.plotly_chart(risk_fig, use_container_width=True)
                
                # Add explanation
                st.markdown("""
                **Understanding the Risk-Impact Matrix:**
                
                This matrix helps prioritize inventory management actions based on risk and business impact:
                
                - **Horizontal Axis:** Risk Score - Based on demand variability, supply levels, and growth volatility
                - **Vertical Axis:** Business Impact - Based on sales volume and value
                - **Bubble Size:** Average Monthly Sales Volume
                - **Quadrants:**
                    - 🔴 **Critical Focus:** High risk and high impact items require immediate attention
                    - 🔶 **Proactive Monitor:** High risk but lower impact items need regular monitoring
                    - 🔵 **Active Manage:** Lower risk but high impact items should be actively managed
                    - 🟢 **Routine Review:** Lower risk and lower impact items need only routine oversight
                """)
                
                # Show high risk items
                st.subheader("High Risk Items")
                high_risk = inventory_df.sort_values('risk_score', ascending=False).head(10)
                
                if not high_risk.empty:
                    # Format for display
                    display_cols = ['sku', 'status', 'risk_score', 'months_of_supply', 'avg_monthly_sales', 'growth_rate']
                    high_risk_display = high_risk[display_cols].copy()
                    high_risk_display['risk_score'] = high_risk_display['risk_score'].round(1)
                    high_risk_display['months_of_supply'] = high_risk_display['months_of_supply'].round(2)
                    high_risk_display['growth_rate'] = high_risk_display['growth_rate'].round(1).astype(str) + '%'
                    
                    st.dataframe(high_risk_display, use_container_width=True)
                    
                    # Risk statistics
                    avg_risk = inventory_df['risk_score'].mean()
                    high_risk_pct = (len(inventory_df[inventory_df['risk_score'] > 50]) / len(inventory_df) * 100)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Average Risk Score",
                            value=f"{avg_risk:.1f}/100",
                            delta=f"{avg_risk - 50:.1f}" if avg_risk <= 50 else f"{avg_risk - 50:.1f}",
                            delta_color="normal" if avg_risk <= 50 else "inverse"
                        )
                    
                    with col2:
                        st.metric(
                            label="Items with High Risk (>50)",
                            value=f"{high_risk_pct:.1f}%",
                            delta=None
                        )
                else:
                    st.info("No high risk items found")
            
            with inv_tab3:
                st.subheader("Actionable Inventory Insights")
                
                # Generate actionable insights
                st.markdown("### Recommended Actions")
                
                # For stockout risk items
                stockout_items = inventory_df[inventory_df['status'] == 'Stockout Risk'].sort_values('importance', ascending=False)
                if len(stockout_items) > 0:
                    st.markdown("#### 🔴 Stockout Risk Items")
                    st.markdown(f"**{len(stockout_items)}** items are at risk of stockout. Immediate action is needed:")
                    
                    # Create markdown bullets for top 5 items
                    top_stockout = stockout_items.head(5)
                    for _, item in top_stockout.iterrows():
                        st.markdown(f"- **{item['sku']}**: Increase inventory immediately. Current supply: **{item['months_of_supply']:.2f}** months, Sales: **{item['avg_monthly_sales']:.0f}** units/month")
                    
                    if len(stockout_items) > 5:
                        st.markdown(f"*... and {len(stockout_items) - 5} more items*")
                
                # For overstocked items
                overstock_items = inventory_df[inventory_df['status'] == 'Overstocked'].sort_values(['months_of_supply', 'importance'], ascending=[False, False])
                if len(overstock_items) > 0:
                    st.markdown("#### 🟠 Overstocked Items")
                    st.markdown(f"**{len(overstock_items)}** items are overstocked. Consider reducing inventory:")
                    
                    # Create markdown bullets for top 5 items
                    top_overstock = overstock_items.head(5)
                    for _, item in top_overstock.iterrows():
                        st.markdown(f"- **{item['sku']}**: Reduce replenishment. Current supply: **{item['months_of_supply']:.2f}** months, Sales: **{item['avg_monthly_sales']:.0f}** units/month")
                    
                    if len(overstock_items) > 5:
                        st.markdown(f"*... and {len(overstock_items) - 5} more items*")
                
                # Items with high variability
                high_var_items = inventory_df[inventory_df['demand_variability'] > 0.5].sort_values('importance', ascending=False)
                if len(high_var_items) > 0:
                    st.markdown("#### 🔄 High Variability Items")
                    st.markdown(f"**{len(high_var_items)}** items have highly variable demand. Consider safety stock adjustments:")
                    
                    # Create markdown bullets for top 5 items
                    top_var = high_var_items.head(5)
                    for _, item in top_var.iterrows():
                        st.markdown(f"- **{item['sku']}**: Implement safety stock. Variability: **{item['demand_variability']:.2f}** CV, Sales: **{item['avg_monthly_sales']:.0f}** units/month")
                    
                    if len(high_var_items) > 5:
                        st.markdown(f"*... and {len(high_var_items) - 5} more items*")
                
                # Growing items
                growing_items = inventory_df[inventory_df['growth_rate'] > 20].sort_values('growth_rate', ascending=False)
                if len(growing_items) > 0:
                    st.markdown("#### 📈 High Growth Items")
                    st.markdown(f"**{len(growing_items)}** items are showing strong growth. Plan for increased demand:")
                    
                    # Create markdown bullets for top 5 items
                    top_growing = growing_items.head(5)
                    for _, item in top_growing.iterrows():
                        st.markdown(f"- **{item['sku']}**: Plan for growth. Rate: **{item['growth_rate']:.1f}%**, Current sales: **{item['avg_monthly_sales']:.0f}** units/month")
                    
                    if len(growing_items) > 5:
                        st.markdown(f"*... and {len(growing_items) - 5} more items*")
        else:
            with inv_tab1:
                st.info("No inventory data available for analysis")
            
            with inv_tab2:
                st.info("No inventory data available for risk assessment")
                
            with inv_tab3:
                st.info("No inventory data available for actionable insights")
    else:
        with inv_tab1:
            st.info("Please upload BOM data to calculate detailed inventory metrics")
        
        with inv_tab2:
            st.info("Please upload BOM data to perform risk assessment")
            
        with inv_tab3:
            st.info("Please upload BOM data to generate actionable insights")

with tab3:
    st.header("Supplier Performance Scorecard")
    
    if 'supplier_data' in st.session_state and st.session_state.supplier_data is not None:
        # Option to upload supplier performance data
        st.subheader("Upload Supplier Performance Data")
        supplier_perf_file = st.file_uploader("Upload supplier performance data (Excel)", type=["xlsx", "xls"], key="supplier_perf_upload")
        
        if supplier_perf_file is not None:
            try:
                # Read supplier performance data
                supplier_perf = pd.read_excel(supplier_perf_file)
                
                # Display sample of the data
                st.success(f"Successfully loaded supplier performance data with {len(supplier_perf)} records!")
                st.write("Preview:")
                st.dataframe(supplier_perf.head(), use_container_width=True)
                
                # Check if the required columns are present
                required_cols = ['supplier_id', 'on_time_delivery', 'quality_score']
                
                if all(col in supplier_perf.columns for col in required_cols):
                    # Calculate supplier performance metrics
                    st.subheader("Supplier Performance Metrics")
                    
                    # Aggregate metrics by supplier
                    supplier_metrics = supplier_perf.groupby('supplier_id').agg({
                        'on_time_delivery': 'mean',
                        'quality_score': 'mean'
                    }).reset_index()
                    
                    # Calculate overall score (50% on-time, 50% quality)
                    supplier_metrics['overall_score'] = (supplier_metrics['on_time_delivery'] * 50 + 
                                                       supplier_metrics['quality_score'] * 50) / 100
                    
                    # Sort by overall score
                    supplier_metrics = supplier_metrics.sort_values('overall_score', ascending=False)
                    
                    # Format percentages
                    supplier_metrics['on_time_delivery'] = (supplier_metrics['on_time_delivery'] * 100).round(1).astype(str) + '%'
                    supplier_metrics['quality_score'] = (supplier_metrics['quality_score'] * 100).round(1).astype(str) + '%'
                    supplier_metrics['overall_score'] = (supplier_metrics['overall_score']).round(1).astype(str) + '%'
                    
                    # Display table
                    st.dataframe(supplier_metrics, use_container_width=True)
                    
                    # Create performance distribution chart
                    st.subheader("Supplier Performance Distribution")
                    
                    # Convert back to numeric for plotting
                    supplier_perf['on_time_delivery'] = pd.to_numeric(supplier_perf['on_time_delivery'], errors='coerce')
                    supplier_perf['quality_score'] = pd.to_numeric(supplier_perf['quality_score'], errors='coerce')
                    
                    # Create scatter plot
                    fig = px.scatter(
                        supplier_perf,
                        x='on_time_delivery',
                        y='quality_score',
                        color='supplier_id',
                        title='Supplier Performance Matrix',
                        labels={
                            'on_time_delivery': 'On-Time Delivery Rate',
                            'quality_score': 'Quality Score'
                        },
                        hover_name='supplier_id'
                    )
                    
                    # Add quadrant lines
                    fig.add_hline(y=0.9, line_dash="dash", line_color="gray")
                    fig.add_vline(x=0.9, line_dash="dash", line_color="gray")
                    
                    # Update layout
                    fig.update_layout(
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1])
                    )
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add quadrant explanation
                    st.markdown("""
                    **Supplier Performance Quadrants:**
                    
                    - **Top Right:** Strategic Partners (High Quality, High On-Time Delivery)
                    - **Top Left:** Quality Focused but Unreliable (High Quality, Low On-Time Delivery)
                    - **Bottom Right:** Fast but Inconsistent (Low Quality, High On-Time Delivery)
                    - **Bottom Left:** Problematic Suppliers (Low Quality, Low On-Time Delivery)
                    """)
                else:
                    st.error(f"Supplier performance data is missing required columns. Please include: {', '.join(required_cols)}")
            except Exception as e:
                st.error(f"Error processing supplier performance data: {str(e)}")
        else:
            # If no supplier performance data is uploaded, show a placeholder
            st.info("Please upload supplier performance data to analyze supplier metrics")
            
            # Show sample supplier scorecard structure
            st.subheader("Supplier Scorecard Template")
            
            sample_data = {
                'Metric': ['On-Time Delivery', 'Quality Score', 'Lead Time Reliability', 'Communication', 'Price Competitiveness'],
                'Target': ['95%', '98%', '90%', '85%', '80%'],
                'Weight': ['30%', '25%', '20%', '15%', '10%']
            }
            
            st.table(pd.DataFrame(sample_data))
    else:
        st.info("Please upload supplier data first to enable supplier performance analysis")

# Export KPI dashboard
st.header("Export KPI Dashboard")

if st.button("Prepare KPI Report"):
    # Create a summary report in DataFrame format
    report_data = []
    
    # Add forecast metrics
    if 'forecasts' in st.session_state and st.session_state.forecasts:
        report_data.append({
            'Category': 'Forecasting',
            'Metric': 'Number of SKUs Forecasted',
            'Value': len(st.session_state.forecasts)
        })
        
        # Count models used
        model_counts = {}
        for sku, forecast in st.session_state.forecasts.items():
            model = forecast.get('model', 'unknown')
            model_counts[model] = model_counts.get(model, 0) + 1
        
        for model, count in model_counts.items():
            report_data.append({
                'Category': 'Forecasting',
                'Metric': f'SKUs using {model.upper()} model',
                'Value': count
            })
    
    # Add inventory metrics if available
    if 'bom_data' in st.session_state and st.session_state.bom_data is not None:
        report_data.append({
            'Category': 'Inventory',
            'Metric': 'Total Materials',
            'Value': total_materials
        })
    
    # Add supplier metrics if available
    if 'supplier_data' in st.session_state and st.session_state.supplier_data is not None:
        report_data.append({
            'Category': 'Suppliers',
            'Metric': 'Total Suppliers',
            'Value': total_suppliers
        })
        
        if 'lead_time_days' in st.session_state.supplier_data.columns:
            avg_lead_time = st.session_state.supplier_data['lead_time_days'].mean()
            report_data.append({
                'Category': 'Suppliers',
                'Metric': 'Average Lead Time (days)',
                'Value': round(avg_lead_time, 1)
            })
    
    # Convert to DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Display report preview
    st.subheader("KPI Report Preview")
    st.dataframe(report_df, use_container_width=True)
    
    # Convert to Excel for download
    excel_buffer = io.BytesIO()
    report_df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
    excel_buffer.seek(0)
    
    # Create download button
    st.download_button(
        label="Download KPI Report as Excel",
        data=excel_buffer,
        file_name=f"supply_chain_kpi_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.ms-excel"
    )
