import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime
from utils.planning import generate_production_plan
from utils.visualization import plot_production_plan

# Set page config
st.set_page_config(
    page_title="Production Planning",
    page_icon="🏭",
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
st.title("Production Planning")
st.markdown("""
This module converts your demand forecasts into optimized production plans.
Prioritize high-accuracy SKUs, minimize changeovers, and ensure on-time delivery.
""")

# Initialize session state variables for this page
if 'production_plan' not in st.session_state:
    st.session_state.production_plan = None
if 'run_production_planning' not in st.session_state:
    st.session_state.run_production_planning = False

# Create sidebar for settings
with st.sidebar:
    st.header("Production Settings")
    
    # Production buffer percentage
    buffer_percent = st.slider(
        "Safety Stock (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        help="Percentage of additional production to account for uncertainties"
    )
    
    # Priority settings
    st.subheader("Production Prioritization")
    
    priority_method = st.radio(
        "Prioritize Production By:",
        options=["Forecast Accuracy", "Sales Volume", "Equal Priority"],
        index=0
    )
    
    # Generate production plan button
    if st.button("Generate Production Plan"):
        st.session_state.run_production_planning = True
        with st.spinner("Generating production plan..."):
            # Adjust forecasts with buffer
            adjusted_forecasts = {}
            for sku, forecast in st.session_state.forecasts.items():
                adjusted_forecast = forecast.copy()
                adjusted_forecast['forecast'] = forecast['forecast'] * (1 + buffer_percent / 100)
                adjusted_forecasts[sku] = adjusted_forecast
            
            # Generate production plan
            st.session_state.production_plan = generate_production_plan(
                adjusted_forecasts,
                st.session_state.bom_data if 'bom_data' in st.session_state else pd.DataFrame(),
                st.session_state.supplier_data if 'supplier_data' in st.session_state else pd.DataFrame()
            )
            
            # Apply prioritization if needed
            if priority_method == "Forecast Accuracy" and 'production_plan' in st.session_state:
                # Sort by confidence (prioritize high confidence SKUs)
                st.session_state.production_plan = st.session_state.production_plan.sort_values(
                    by=['date', 'confidence', 'production_quantity'],
                    ascending=[True, False, False]
                )
            
            elif priority_method == "Sales Volume" and 'production_plan' in st.session_state:
                # Sort by production quantity (prioritize high volume SKUs)
                st.session_state.production_plan = st.session_state.production_plan.sort_values(
                    by=['date', 'production_quantity'],
                    ascending=[True, False]
                )
            
            st.success("Production plan generated successfully!")

# Main content
if st.session_state.run_production_planning and st.session_state.production_plan is not None:
    # Show production plan summary
    st.header("Production Plan Summary")
    
    # Create chart
    production_fig = plot_production_plan(st.session_state.production_plan)
    st.plotly_chart(production_fig, use_container_width=True)
    
    # Production plan details
    st.header("Production Schedule")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["By Period", "By SKU", "Daily Production Plan"])
    
    with tab1:
        # Group by period
        period_plan = st.session_state.production_plan.groupby(['period', 'date']).agg({
            'production_quantity': 'sum',
            'min_quantity': 'sum',
            'max_quantity': 'sum'
        }).reset_index().sort_values('date')
        
        st.dataframe(period_plan, use_container_width=True)
        
        # Highlight key insights
        total_production = st.session_state.production_plan['production_quantity'].sum()
        peak_period = period_plan.loc[period_plan['production_quantity'].idxmax()]['period']
        peak_quantity = period_plan['production_quantity'].max()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Production", f"{int(total_production):,} units")
        with col2:
            st.metric("Peak Production Period", peak_period)
        with col3:
            st.metric("Peak Period Quantity", f"{int(peak_quantity):,} units")
    
    with tab2:
        # Group by SKU
        sku_plan = st.session_state.production_plan.groupby('sku').agg({
            'production_quantity': 'sum'
        }).reset_index().sort_values('production_quantity', ascending=False)
        
        # Add percentage of total
        sku_plan['percentage'] = (sku_plan['production_quantity'] / sku_plan['production_quantity'].sum() * 100).round(1)
        sku_plan['percentage'] = sku_plan['percentage'].astype(str) + '%'
        
        st.dataframe(sku_plan, use_container_width=True)
        
        # Create a pie chart of top SKUs
        top_skus = sku_plan.head(10).copy()
        others_sum = sku_plan['production_quantity'].sum() - top_skus['production_quantity'].sum()
        
        if others_sum > 0:
            others_row = pd.DataFrame({
                'sku': ['Others'],
                'production_quantity': [others_sum],
                'percentage': [f"{(others_sum / sku_plan['production_quantity'].sum() * 100).round(1)}%"]
            })
            top_skus = pd.concat([top_skus, others_row])
        
        fig = px.pie(
            top_skus, 
            values='production_quantity', 
            names='sku',
            title='Production Distribution by SKU',
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Daily Production Plan")
        
        # Allow selecting a specific period for the daily plan
        selected_month = st.selectbox(
            "Select Month for Daily Planning",
            options=sorted(st.session_state.production_plan['period'].unique()),
            key="daily_plan_month"
        )
        
        # Filter the production plan for the selected month
        month_plan = st.session_state.production_plan[
            st.session_state.production_plan['period'] == selected_month
        ].copy()
        
        if not month_plan.empty:
            # Get the date range for this month
            start_date = month_plan['date'].min()
            end_date = month_plan['date'].max()
            
            # Create a daily date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Get unique SKUs
            skus = sorted(month_plan['sku'].unique())
            
            # Create daily production data
            # First, get the base monthly production
            sku_monthly_production = month_plan.groupby('sku')['production_quantity'].sum()
            
            # Next, extract forecast data for the selected month to determine distribution
            if 'forecasts' in st.session_state:
                # Create a daily production plan
                daily_data = []
                
                # Get the number of working days in the month (excluding weekends)
                working_days = len([d for d in date_range if d.weekday() < 5])  # 0-4 are Monday-Friday
                
                for sku in skus:
                    if sku in sku_monthly_production:
                        # Get monthly production for this SKU
                        total_production = sku_monthly_production[sku]
                        
                        # Calculate daily production (evenly distributed across working days)
                        daily_prod = total_production / working_days
                        
                        # Get forecast for this SKU if available
                        daily_forecast = 0
                        if sku in st.session_state.forecasts:
                            forecast_data = st.session_state.forecasts[sku]
                            # Find forecast for this month
                            for date, value in forecast_data['forecast'].items():
                                if date.strftime('%Y-%m') == selected_month[:7]:
                                    daily_forecast = value / 30  # Approximate daily forecast
                                    break
                        
                        # Create initial inventory (use 2x daily production as a placeholder)
                        initial_inventory = daily_prod * 2
                        
                        # Create records for each working day
                        current_inventory = initial_inventory
                        for date in date_range:
                            if date.weekday() < 5:  # Monday-Friday
                                # Create a record for this day
                                row = {
                                    'date': date,
                                    'sku': sku,
                                    'opening_stock': current_inventory,
                                    'production': daily_prod,
                                    'forecast_sales': daily_forecast,
                                    'closing_stock': current_inventory + daily_prod - daily_forecast
                                }
                                daily_data.append(row)
                                
                                # Update inventory for next day
                                current_inventory = row['closing_stock']
                
                # Convert to DataFrame
                daily_df = pd.DataFrame(daily_data)
                
                # Show tables and charts by SKU
                st.write("Daily Production and Inventory Plan")
                
                # Allow selecting a specific SKU to view
                selected_sku = st.selectbox("Select SKU", options=skus)
                
                # Filter for selected SKU
                sku_daily = daily_df[daily_df['sku'] == selected_sku].copy()
                
                # Format dates and round numbers for display
                sku_daily['date'] = sku_daily['date'].dt.strftime('%Y-%m-%d')
                for col in ['opening_stock', 'production', 'forecast_sales', 'closing_stock']:
                    sku_daily[col] = sku_daily[col].round(0).astype(int)
                
                # Display table
                st.dataframe(sku_daily, use_container_width=True)
                
                # Create chart
                chart_data = daily_df[daily_df['sku'] == selected_sku].copy()
                
                fig = go.Figure()
                
                # Add opening stock line
                fig.add_trace(go.Scatter(
                    x=chart_data['date'],
                    y=chart_data['opening_stock'],
                    mode='lines+markers',
                    name='Opening Stock',
                    line=dict(color='blue')
                ))
                
                # Add production as bars
                fig.add_trace(go.Bar(
                    x=chart_data['date'],
                    y=chart_data['production'],
                    name='Production',
                    marker_color='green'
                ))
                
                # Add forecast sales as bars
                fig.add_trace(go.Bar(
                    x=chart_data['date'],
                    y=chart_data['forecast_sales'] * -1,  # Negative to show below axis
                    name='Forecast Sales',
                    marker_color='red'
                ))
                
                # Add closing stock line
                fig.add_trace(go.Scatter(
                    x=chart_data['date'],
                    y=chart_data['closing_stock'],
                    mode='lines+markers',
                    name='Closing Stock',
                    line=dict(color='purple')
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'Daily Production Plan for {selected_sku}',
                    barmode='relative',
                    xaxis_title='Date',
                    yaxis_title='Quantity',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("Forecast data is not available to generate daily plans.")
        else:
            st.warning("No production plan data found for the selected month.")
    
    # Detailed view
    st.header("Detailed Production Schedule")
    
    # Allow filtering by period
    periods = sorted(st.session_state.production_plan['period'].unique())
    selected_period = st.selectbox("Select Period", options=periods)
    
    # Filter data
    filtered_plan = st.session_state.production_plan[st.session_state.production_plan['period'] == selected_period]
    
    # Sort by priority
    filtered_plan = filtered_plan.sort_values(
        by=['production_quantity'],
        ascending=False
    )
    
    # Show detailed table
    st.dataframe(filtered_plan[['sku', 'production_quantity', 'min_quantity', 'max_quantity']], use_container_width=True)
    
    # Export option
    st.header("Export Production Plan")
    
    if st.button("Prepare Production Plan Export"):
        # Prepare export data
        export_data = st.session_state.production_plan.copy()
        export_data['date'] = export_data['date'].dt.strftime('%Y-%m-%d')
        
        # Display preview
        st.subheader("Export Preview")
        st.dataframe(export_data.head(10), use_container_width=True)
        
        # Convert to Excel for download
        excel_buffer = io.BytesIO()
        export_data.to_excel(excel_buffer, index=False, engine='xlsxwriter')
        excel_buffer.seek(0)
        
        # Create download button
        st.download_button(
            label="Download Production Plan as Excel",
            data=excel_buffer,
            file_name=f"production_plan_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.ms-excel"
        )

else:
    # Show instructions when no production plan has been generated
    st.info("👈 Please configure and generate a production plan using the sidebar.")
    
    # Show some useful information
    if 'forecasts' in st.session_state and st.session_state.forecasts:
        st.subheader("Forecast Summary")
        
        # Calculate total forecasted demand
        total_demand = 0
        skus_with_forecasts = 0
        
        for sku, forecast in st.session_state.forecasts.items():
            if len(forecast['forecast']) > 0:
                total_demand += forecast['forecast'].sum()
                skus_with_forecasts += 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Forecasted Demand", f"{int(total_demand):,} units")
        
        with col2:
            st.metric("SKUs with Forecasts", skus_with_forecasts)
        
        st.markdown("""
        ### Production Planning Process
        
        1. The system starts with the demand forecasts you've already generated
        2. Adjusts quantities based on your safety stock percentage
        3. Applies production prioritization based on your selected method
        4. Generates a detailed production schedule by period and SKU
        
        Click the "Generate Production Plan" button in the sidebar to start.
        """)
