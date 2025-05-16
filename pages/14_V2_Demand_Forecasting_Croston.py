import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import io
import plotly.express as px

# Import utility functions
from utils.data_processor import process_sales_data, get_sku_data, check_intermittent_demand, prepare_data_for_forecasting
from utils.seasonal_detector import detect_seasonal_period
from utils.croston import croston_optimized
from utils.parameter_optimizer import optimize_parameters, get_model_parameters_with_fallback
from utils.database import save_forecast_result
from utils.visualization import plot_forecast, plot_model_comparison, plot_parameter_importance
from utils.session_data import load_data_if_needed

# Page configuration
st.set_page_config(
    page_title="Intermittent Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# Page title
st.title("🔄 Intermittent Demand Forecasting with Croston Method")

# Sidebar for controls
st.sidebar.header("Settings")

# Load data
data_loaded = load_data_if_needed()

if not data_loaded:
    st.info("No sales data found in session state. Please upload data on the home page first.")
    st.stop()

# Process the data
try:
    sales_data = process_sales_data(st.session_state.sales_data)
    
    # Get unique SKUs and sort them
    skus = sorted(sales_data['sku'].unique())
    
    # SKU selection
    selected_sku = st.sidebar.selectbox("Select SKU to Forecast", skus)
    
    # Get data for the selected SKU
    sku_data = get_sku_data(sales_data, selected_sku)
    
    # Check if this SKU has intermittent demand
    is_intermittent = check_intermittent_demand(sku_data)
    
    if is_intermittent:
        st.success(f"✅ SKU {selected_sku} has intermittent demand (many zero values). Croston method is recommended.")
    else:
        st.info(f"ℹ️ SKU {selected_sku} does not have highly intermittent demand. Standard forecasting methods may work well.")
    
    # Data preprocessing options
    st.sidebar.subheader("Data Preprocessing")
    
    # Frequency selection
    frequency = st.sidebar.selectbox(
        "Data Frequency",
        options=['D', 'W', 'M'],
        format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[x],
        index=2  # Default to monthly
    )
    
    # Preprocess data
    resampled_data = prepare_data_for_forecasting(sku_data, frequency=frequency)
    
    # Display data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Historical Data")
        st.line_chart(resampled_data.set_index('date')['quantity'])
    
    with col2:
        st.subheader("Data Summary")
        st.write(f"Total Records: {len(resampled_data)}")
        st.write(f"Date Range: {resampled_data['date'].min().date()} to {resampled_data['date'].max().date()}")
        st.write(f"Average Demand: {resampled_data['quantity'].mean():.2f}")
        st.write(f"Zero Values: {(resampled_data['quantity'] == 0).sum()} ({(resampled_data['quantity'] == 0).mean()*100:.1f}%)")
        
        data_quality = st.empty()  # Placeholder for data quality assessment
        
        # Data quality assessment
        if len(resampled_data) < 12:
            data_quality.warning("⚠️ Limited data available. Forecast may be less reliable.")
        elif (resampled_data['quantity'] == 0).mean() > 0.7:
            data_quality.warning("⚠️ Very high proportion of zeros. Consider using SBA Croston variant.")
        else:
            data_quality.success("✅ Sufficient data for forecasting.")
    
    # Model parameters
    st.sidebar.subheader("Croston Model Parameters")
    
    # Detect seasonality automatically
    with st.spinner("Detecting seasonal patterns..."):
        detected_period = detect_seasonal_period(resampled_data['quantity'])
    
    # Allow user to override detected seasonality
    seasonal_period = st.sidebar.slider(
        "Seasonal Period",
        min_value=0,
        max_value=12,
        value=detected_period,
        help="0 means no seasonality. The system detected a period of " + str(detected_period)
    )
    
    # Alpha parameter (smoothing)
    alpha = st.sidebar.slider("Alpha (Smoothing)", 0.01, 0.5, 0.1, 0.01)
    
    # Method selection
    method = st.sidebar.radio(
        "Croston Variant",
        options=['original', 'sba'],
        format_func=lambda x: {
            'original': 'Original Croston',
            'sba': 'Syntetos-Boylan Approximation (SBA)'
        }[x]
    )
    
    # Forecast horizon
    horizon = st.sidebar.slider("Forecast Horizon", 1, 24, 6)
    
    # Parameter tuning
    st.sidebar.subheader("Parameter Tuning")
    
    # Button to optimize parameters
    optimize_button = st.sidebar.button("Optimize Parameters")
    
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
            
            # Update sliders with optimized values
            alpha = optimized_params.get('alpha', alpha)
            method = optimized_params.get('method', method)
            
            st.sidebar.success("Parameters optimized!")
            
            # Show metrics
            if 'metrics' in result:
                st.sidebar.write("Optimization Metrics:")
                for metric, value in result['metrics'].items():
                    st.sidebar.write(f"- {metric.upper()}: {value:.4f}")
    
    # Create forecast button
    forecast_button = st.button("Generate Forecast")
    
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
            
            # Save to database
            forecast_data = {
                'forecast': forecast.to_dict(),
                'lower_bound': lower_bound.to_dict(),
                'upper_bound': upper_bound.to_dict()
            }
            
            metadata = {
                'sku': selected_sku,
                'frequency': frequency,
                'parameters': params,
                'seasonal_period': seasonal_period,
                'is_intermittent': is_intermittent
            }
            
            forecast_id = save_forecast_result(
                selected_sku,
                model_name,
                forecast_data,
                metadata
            )
            
            # Plot forecast
            forecast_plot = plot_forecast(
                resampled_data,
                forecast,
                upper_bound,
                lower_bound,
                model_name,
                selected_sku
            )
            
            st.plotly_chart(forecast_plot, use_container_width=True)
            
            # Display forecast details
            st.subheader("Forecast Details")
            
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
            
            st.write(forecast_df)
            
            # Compare with previous forecasts
            if len(st.session_state.forecasts) > 1:
                st.subheader("Model Comparison")
                
                comparison_plot = plot_model_comparison(
                    resampled_data,
                    st.session_state.forecasts,
                    selected_sku
                )
                
                st.plotly_chart(comparison_plot, use_container_width=True)
            
            # Show parameter importance
            st.subheader("Parameter Importance")
            
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
            st.success(f"Forecast generated successfully! Forecast ID: {forecast_id}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)