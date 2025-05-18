"""
Single-page demand forecasting application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sqlite3
import io
import plotly.express as px

# Import utility functions
from utils.data_processor import process_sales_data, get_sku_data, check_intermittent_demand, prepare_data_for_forecasting
from utils.seasonal_detector import detect_seasonal_period
from utils.croston import croston_optimized
from utils.parameter_optimizer import optimize_parameters, get_model_parameters_with_fallback
from utils.database import save_forecast_result
from utils.visualization import plot_forecast, plot_model_comparison, plot_parameter_importance

# Page configuration
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title
st.title("📊 Intermittent Demand Forecasting")
st.subheader("Specialized for inventory with irregular patterns")

# Load sample data function
def load_sample_data():
    """Load sample data from the samples directory"""
    try:
        # Check for sample data file
        sample_path = 'data/samples/sample_data.xlsx'
        if os.path.exists(sample_path):
            data = pd.read_excel(sample_path)
            
            # Map the custom Excel file format to standard columns if needed
            if 'FG Code' in data.columns and 'QTY_MONTH' in data.columns:
                # Map necessary columns to standard format
                data.rename(columns={
                    'FG Code': 'sku',
                    'QTY_MONTH': 'quantity',
                    'YR_MONTH_NR': 'date'
                }, inplace=True)
            
            # Process data if needed - check date column format, etc.
            if 'date' in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data['date']):
                    data['date'] = pd.to_datetime(data['date'])
            
            return data
        else:
            # Fallback to CSV if Excel file doesn't exist
            csv_path = 'data/samples/sales_sample.csv'
            if os.path.exists(csv_path):
                data = pd.read_csv(csv_path)
                
                # Process data if needed - check date column format, etc.
                if 'date' in data.columns:
                    if not pd.api.types.is_datetime64_any_dtype(data['date']):
                        data['date'] = pd.to_datetime(data['date'])
                
                return data
        
        return None
    except Exception as e:
        print(f"Error loading sample data: {str(e)}")
        return None

# Auto-load sample data if not already loaded
if 'sales_data' not in st.session_state or st.session_state.sales_data.empty:
    try:
        sample_data = load_sample_data()
        if sample_data is not None:
            st.session_state.sales_data = sample_data
    except Exception as e:
        print(f"Error auto-loading sample data: {e}")

# Create tabs for different sections
tabs = st.tabs(["Data Management", "Forecasting"])

# Data Management Tab
with tabs[0]:
    st.subheader("Data Management")
    
    # Data management buttons in a row
    col1, col2, col3 = st.columns(3)
    
    # Load Sample Data button
    if col1.button("📊 Load Sample Data"):
        with st.spinner("Loading sample data..."):
            sample_data = load_sample_data()
            if sample_data is not None:
                st.session_state.sales_data = sample_data
                st.success(f"Sample data loaded successfully! {len(sample_data)} records loaded.")
            else:
                st.error("Failed to load sample data.")
    
    # Upload Data button/widget
    uploaded_file = col2.file_uploader("📤 Upload Sales Data", type=['xlsx'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_excel(uploaded_file)
            
            # Check required columns
            required_columns = ['date', 'sku', 'quantity']
            if not all(col in df.columns for col in required_columns):
                # Try to map columns if using specific format
                if 'FG Code' in df.columns and 'QTY_MONTH' in df.columns:
                    df.rename(columns={
                        'FG Code': 'sku',
                        'QTY_MONTH': 'quantity',
                        'YR_MONTH_NR': 'date'
                    }, inplace=True)
                
                # Check again after mapping
                if not all(col in df.columns for col in required_columns):
                    st.error("The uploaded file must contain 'date', 'sku', and 'quantity' columns.")
                else:
                    st.success(f"Successfully uploaded and mapped {uploaded_file.name}")
                    st.session_state.sales_data = df
            else:
                # Display preview
                st.success(f"Successfully uploaded {uploaded_file.name}")
                st.session_state.sales_data = df
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # View Data button
    if col3.button("👁️ View Sales Data"):
        # Check if data is loaded
        if 'sales_data' in st.session_state and not st.session_state.sales_data.empty:
            data = st.session_state.sales_data
            st.write("### Sales Data Preview")
            st.dataframe(data.head(10))
            
            # Display summary statistics
            st.write("### Data Summary")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", str(len(data)))
            
            if 'sku' in data.columns:
                col2.metric("Unique SKUs", str(data['sku'].nunique()))
            else:
                col2.metric("Unique SKUs", "N/A")
            
            if 'date' in data.columns:
                date_range = f"{data['date'].min().date()} to {data['date'].max().date()}"
                col3.metric("Date Range", date_range)
            else:
                col3.metric("Date Range", "N/A")
            
            # Show distribution of quantity 
            st.write("### Quantity Distribution")
            if 'quantity' in data.columns:
                fig = px.histogram(data, x='quantity', nbins=20, title="Distribution of Demand Quantities")
                st.plotly_chart(fig, use_container_width=True)
            elif 'QTY_MONTH' in data.columns:
                fig = px.histogram(data, x='QTY_MONTH', nbins=20, title="Distribution of Demand Quantities")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Find any numeric column for histogram
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    fig = px.histogram(data, x=numeric_cols[0], nbins=20, title=f"Distribution of {numeric_cols[0]}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No suitable numeric columns found for histogram visualization.")
        else:
            st.info("No data available. Please load sample data or upload your own file.")

# Forecasting Tab
with tabs[1]:
    st.subheader("Demand Forecasting")
    
    # Check if data is available
    if 'sales_data' not in st.session_state or st.session_state.sales_data.empty:
        st.info("No sales data available. Please go to the Data Management tab to load data.")
        st.stop()
    
    # Process the data for forecasting
    try:
        sales_data = process_sales_data(st.session_state.sales_data)
        
        # Get unique SKUs and sort them
        skus = sorted(sales_data['sku'].unique())
        
        # Main forecasting area
        col1, col2 = st.columns([3, 1])
        
        # SKU selection in column 2 (sidebar replacement)
        with col2:
            st.write("### Settings")
            selected_sku = st.selectbox("Select SKU to Forecast", skus)
            
            # Data preprocessing options
            st.write("#### Data Preprocessing")
            
            # Frequency selection
            frequency = st.selectbox(
                "Data Frequency",
                options=['D', 'W', 'ME'],
                format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'ME': 'Monthly'}[x],
                index=2  # Default to monthly
            )
            
            # Croston model parameters
            st.write("#### Model Parameters")
            
            # Alpha parameter (smoothing)
            alpha = st.slider("Alpha (Smoothing)", 0.01, 0.5, 0.1, 0.01)
            
            # Method selection
            method = st.radio(
                "Croston Variant",
                options=['original', 'sba'],
                format_func=lambda x: {
                    'original': 'Original Croston',
                    'sba': 'SBA Variant'
                }[x]
            )
            
            # Forecast horizon
            horizon = st.slider("Forecast Horizon", 1, 24, 6)
            
            # Buttons for actions
            optimize_button = st.button("🔄 Optimize Parameters")
            forecast_button = st.button("📈 Generate Forecast")
        
        # Main content area in column 1
        with col1:
            # Get data for the selected SKU
            sku_data = get_sku_data(sales_data, selected_sku)
            
            # Check if this SKU has intermittent demand
            is_intermittent = check_intermittent_demand(sku_data)
            
            if is_intermittent:
                st.success(f"✅ SKU {selected_sku} has intermittent demand (many zero values). Croston method is recommended.")
            else:
                st.info(f"ℹ️ SKU {selected_sku} does not have highly intermittent demand. Standard forecasting methods may also work well.")
            
            # Preprocess data
            resampled_data = prepare_data_for_forecasting(sku_data, frequency=frequency)
            
            # Detect seasonality automatically
            detected_period = detect_seasonal_period(resampled_data['quantity'])
            seasonal_period = detected_period  # Use detected value
            
            # Display data
            st.write("### Historical Data")
            st.line_chart(resampled_data.set_index('date')['quantity'])
            
            # Data quality assessment
            zero_percent = (resampled_data['quantity'] == 0).mean() * 100
            
            # Create metrics display
            met1, met2, met3, met4 = st.columns(4)
            met1.metric("Records", str(len(resampled_data)))
            met2.metric("Average Demand", f"{resampled_data['quantity'].mean():.2f}")
            met3.metric("Zero Values %", f"{zero_percent:.1f}%")
            met4.metric("Seasonal Period", str(detected_period))
            
            # Data quality warning if needed
            if len(resampled_data) < 12:
                st.warning("⚠️ Limited data available. Forecast may be less reliable.")
            elif zero_percent > 70:
                st.warning("⚠️ Very high proportion of zeros. Consider using SBA Croston variant.")
        
        # Parameter optimization logic
        if optimize_button:
            with st.spinner("Optimizing parameters..."):
                # Optimize parameters
                result = optimize_parameters(
                    selected_sku,
                    'croston',
                    resampled_data,
                    cross_validation=True
                )
                
                # Get optimized parameters
                optimized_params = result['parameters']
                
                # Update values
                alpha = optimized_params.get('alpha', alpha)
                method = optimized_params.get('method', method)
                
                st.success("Parameters optimized!")
                
                # Show metrics
                if 'metrics' in result:
                    st.write("Optimization Metrics:")
                    for metric, value in result['metrics'].items():
                        st.write(f"- {metric.upper()}: {value:.4f}")
        
        # Forecast generation logic
        if forecast_button:
            with st.spinner("Generating forecast..."):
                # Set parameters
                params = {
                    'alpha': alpha,
                    'method': method,
                    'h': horizon
                }
                
                # Generate forecast
                forecast, lower_bound, upper_bound = croston_optimized(
                    resampled_data,
                    parameters=params
                )
                
                # Store forecast data for comparison
                if 'forecasts' not in st.session_state:
                    st.session_state.forecasts = {}
                
                model_name = 'Croston' if method == 'original' else 'SBA-Croston'
                st.session_state.forecasts[model_name] = forecast
                
                # Plot forecast
                st.write("### Forecast Results")
                forecast_plot = plot_forecast(
                    resampled_data,
                    forecast,
                    upper_bound,
                    lower_bound,
                    model_name,
                    selected_sku
                )
                
                st.plotly_chart(forecast_plot, use_container_width=True)
                
                # Confidence interval explanation
                ci_explanation = """
                **Understanding Confidence Intervals:**
                - The shaded area represents the 95% prediction interval.
                - Wider intervals indicate higher uncertainty in the forecast.
                - For intermittent demand, these intervals help account for the irregular pattern of demand.
                """
                st.info(ci_explanation)
                
                # Format forecast as table
                forecast_df = pd.DataFrame({
                    'Date': forecast.index,
                    'Forecast': forecast.values,
                    'Lower Bound': lower_bound.values,
                    'Upper Bound': upper_bound.values
                })
                
                st.write("### Forecast Data")
                st.dataframe(forecast_df)
                
                # Compare with previous forecasts
                if len(st.session_state.forecasts) > 1:
                    st.write("### Model Comparison")
                    
                    comparison_plot = plot_model_comparison(
                        resampled_data,
                        st.session_state.forecasts,
                        selected_sku
                    )
                    
                    st.plotly_chart(comparison_plot, use_container_width=True)
                
                # Show parameter importance
                st.write("### Parameter Importance")
                
                # Create simple metrics from the forecast
                simple_metrics = {
                    'mae': np.mean(np.abs(resampled_data['quantity'].iloc[-min(6, len(resampled_data)):].values - 
                                          forecast.iloc[:min(6, len(forecast))].values)),
                    'forecast_mean': float(forecast.mean())
                }
                
                importance_plot = plot_parameter_importance(
                    params,
                    simple_metrics,
                    'Croston'
                )
                st.plotly_chart(importance_plot, use_container_width=True)
                
                # Success message
                st.success(f"Forecast generated successfully!")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

# Add footer
st.markdown("---")
st.markdown("© 2025 Intermittent Demand Forecasting | Optimized for Heroku Deployment")