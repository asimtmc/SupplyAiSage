import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta
from utils.visualization import plot_forecast_accuracy, plot_inventory_health

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
            
            # Generate forecast accuracy chart
            st.subheader("Forecast Accuracy by SKU")
            accuracy_fig = plot_forecast_accuracy(actuals_data, st.session_state.forecasts)
            st.plotly_chart(accuracy_fig, use_container_width=True)
            
            # Calculate aggregate accuracy metrics
            st.subheader("Accuracy Metrics")
            
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
    
    # Calculate inventory metrics
    if 'bom_data' in st.session_state and st.session_state.bom_data is not None:
        st.subheader("Inventory Status Summary")
        
        # Count SKUs in each status category
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
            
            # Determine inventory status
            if months_of_supply < 0.5:
                status = "Stockout Risk"
            elif months_of_supply > 3:
                status = "Overstocked"
            else:
                status = "Healthy"
            
            inventory_data.append({
                'sku': sku,
                'status': status,
                'months_of_supply': months_of_supply,
                'avg_monthly_sales': avg_monthly_sales
            })
        
        # Convert to DataFrame
        if inventory_data:
            inventory_df = pd.DataFrame(inventory_data)
            
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
    else:
        st.info("Please upload BOM data to calculate detailed inventory metrics")

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
