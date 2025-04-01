import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
import json
from datetime import datetime, timedelta
from utils.data_processor import process_sales_data
from utils.forecast_engine import extract_features, cluster_skus
from utils.advanced_forecast import (
    advanced_generate_forecasts, 
    detect_outliers, 
    clean_time_series,
    extract_advanced_features,
    detect_change_points
)
from utils.visualization import plot_forecast, plot_cluster_summary, plot_model_comparison
from utils.parameter_optimizer import optimize_parameters_async, get_optimization_status
from utils.error_analysis import analyze_forecast_errors, analyze_model_performance, generate_error_report
from utils.secondary_sales_analyzer import analyze_sku_sales_pattern, bulk_analyze_sales
from utils.database import get_secondary_sales

# Initialize session state variables
if 'advanced_forecast_periods' not in st.session_state:
    st.session_state.advanced_forecast_periods = 12  # default 12 months
    
# Helper function for formatting model parameters
def format_parameters(params, model_type):
    """Format model parameters for display"""
    if not params:
        return "No parameters available"

    formatted = []
    if model_type == "auto_arima":
        formatted.append(f"Order: (p={params.get('p', '?')}, d={params.get('d', '?')}, q={params.get('q', '?')})")
    elif model_type == "prophet":
        formatted.append(f"Changepoint prior scale: {params.get('changepoint_prior_scale', '?')}")
        formatted.append(f"Seasonality prior scale: {params.get('seasonality_prior_scale', '?')}")
        formatted.append(f"Seasonality mode: {params.get('seasonality_mode', '?')}")
    elif model_type == "ets":
        formatted.append(f"Trend: {params.get('trend', 'None')}")
        formatted.append(f"Seasonal: {params.get('seasonal', 'None')}")
        formatted.append(f"Damped trend: {params.get('damped_trend', 'False')}")
    elif model_type == "theta":
        formatted.append(f"Theta: {params.get('theta', '?')}")
    elif model_type == "lstm":
        formatted.append(f"Units: {params.get('units', '?')}")
        formatted.append(f"Dropout: {params.get('dropout', '?')}")
        formatted.append(f"Epochs: {params.get('epochs', '?')}")
    
    return "; ".join(formatted)
if 'run_advanced_forecast' not in st.session_state:
    st.session_state.run_advanced_forecast = False
if 'advanced_selected_sku' not in st.session_state:
    st.session_state.advanced_selected_sku = None
if 'advanced_selected_skus' not in st.session_state:
    st.session_state.advanced_selected_skus = []
if 'advanced_show_all_clusters' not in st.session_state:
    st.session_state.advanced_show_all_clusters = False
if 'advanced_models' not in st.session_state:
    st.session_state.advanced_models = ["auto_arima", "prophet", "ets", "theta", "lstm", "ensemble"]
if 'advanced_hyperparameter_tuning' not in st.session_state:
    st.session_state.advanced_hyperparameter_tuning = True
if 'advanced_apply_sense_check' not in st.session_state:
    st.session_state.advanced_apply_sense_check = True
if 'advanced_use_param_cache' not in st.session_state:
    st.session_state.advanced_use_param_cache = True
if 'advanced_forecasts' not in st.session_state:
    st.session_state.advanced_forecasts = {}
if 'advanced_clusters' not in st.session_state:
    st.session_state.advanced_clusters = None
if 'advanced_forecast_in_progress' not in st.session_state:
    st.session_state.advanced_forecast_in_progress = False
if 'advanced_forecast_progress' not in st.session_state:
    st.session_state.advanced_forecast_progress = 0
if 'advanced_current_sku' not in st.session_state:
    st.session_state.advanced_current_sku = ""
if 'advanced_current_model' not in st.session_state:
    st.session_state.advanced_current_model = ""
if 'parameter_tuning_in_progress' not in st.session_state:
    st.session_state.parameter_tuning_in_progress = False
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
# No longer need active_tab since we can't set tab index directly
# Secondary sales analysis session state variables
if 'secondary_sales_results' not in st.session_state:
    st.session_state.secondary_sales_results = {}
if 'run_secondary_analysis' not in st.session_state:
    st.session_state.run_secondary_analysis = False
if 'secondary_sales_algorithm' not in st.session_state:
    st.session_state.secondary_sales_algorithm = "robust_filter"
if 'forecast_data_type' not in st.session_state:
    st.session_state.forecast_data_type = "primary"  # or "secondary"

# Set page config
st.set_page_config(
    page_title="Advanced Demand Forecasting",
    page_icon="🚀",
    layout="wide"
)

# Check if data is loaded in session state
if 'sales_data' not in st.session_state or st.session_state.sales_data is None:
    st.warning("Please upload sales data on the main page first.")
    st.stop()

# Page title
st.title("🚀 Advanced AI-Powered Demand Forecasting")
st.markdown("""
This advanced module implements cutting-edge algorithms for highly accurate demand forecasts.
The system uses intelligent preprocessing, auto model selection, and human-like sense checking
to generate forecasts that outperform traditional methods.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Advanced Forecast Settings")

    # Forecast period selector
    forecast_periods = st.slider(
        "Forecast Periods (Months)",
        min_value=1,
        max_value=24,
        value=st.session_state.advanced_forecast_periods,
        step=1,
        help="Number of months to forecast into the future"
    )
    st.session_state.advanced_forecast_periods = forecast_periods

    # Divider
    st.divider()

    # Advanced Model Selection
    st.subheader("Model Selection")

    # Define all available models
    all_models = {
        "auto_arima": "Auto ARIMA/SARIMA (Statistical)",
        "prophet": "Prophet (Decomposition)",
        "ets": "ETS Models (Exponential Smoothing)",
        "theta": "Theta Method (Decomposition)",
        "lstm": "LSTM (Deep Learning)",
        "ensemble": "Ensemble (Best Model Combination)"
    }

    # Allow model selection with descriptions
    st.write("Select models to evaluate:")

    # Create multiselect for models with default values
    selected_models = []
    for model_key, model_name in all_models.items():
        selected = model_key in st.session_state.advanced_models
        if st.checkbox(model_name, value=selected, key=f"model_{model_key}"):
            selected_models.append(model_key)

    # Update session state with selected models
    st.session_state.advanced_models = selected_models

    # If no models are selected, show warning
    if not selected_models:
        st.warning("Please select at least one forecasting model.")

    # Divider
    st.divider()

    # Advanced Options Section
    st.subheader("Advanced Options")

    # Use Parameter Cache (moved up since hyperparameter tuning is now in dedicated tab)
    use_param_cache = st.toggle(
        "Use Parameter Cache",
        value=st.session_state.advanced_use_param_cache,
        help="Use previously optimized parameters from database for faster and more accurate forecasts"
    )
    st.session_state.advanced_use_param_cache = use_param_cache

    # Use Parameter Cache
    use_param_cache = st.toggle(
        "Use Parameter Cache",
        value=st.session_state.advanced_use_param_cache,
        help="Use previously optimized parameters from database for faster and more accurate forecasts"
    )
    st.session_state.advanced_use_param_cache = use_param_cache
    
    # Add a separate button for hyperparameter tuning
    st.divider()
    if not st.session_state.get('parameter_tuning_in_progress', False):
        if st.button("Run Hyperparameter Tuning", key="sidebar_run_hyperparameter_tuning"):
            st.session_state.parameter_tuning_in_progress = True
            st.session_state.active_tab = "Hyperparameter Tuning"
            st.session_state.tuning_log_messages = []  # Reset tuning log messages
            st.session_state.tuning_progress = 0  # Reset tuning progress
            st.rerun()
    else:
        st.info("Hyperparameter tuning in progress...")
        if st.button("View Tuning Status", key="goto_hyperparam_tuning"):
            st.session_state.active_tab = "Hyperparameter Tuning"
            st.rerun()

    # Apply Sense Check
    apply_sense_check = st.toggle(
        "Human-Like Sense Check",
        value=st.session_state.advanced_apply_sense_check,
        help="Apply business logic and pattern recognition to ensure realistic forecasts"
    )
    st.session_state.advanced_apply_sense_check = apply_sense_check

    # SKU Selection
    st.subheader("SKU Selection")

    # Get all SKUs from the data
    all_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())

    # SKU selection options
    selection_mode = st.radio(
        "SKU Selection Mode:",
        options=["All SKUs", "Selected SKUs", "By Cluster"],
        index=0,
        help="Choose which SKUs to forecast"
    )

    if selection_mode == "Selected SKUs":
        # Multi-select for specific SKUs
        selected_skus = st.multiselect(
            "Select SKUs to forecast:",
            options=all_skus,
            default=st.session_state.advanced_selected_skus[:5] if st.session_state.advanced_selected_skus else [],
            help="Choose specific SKUs to forecast"
        )
        st.session_state.advanced_selected_skus = selected_skus
    elif selection_mode == "By Cluster":
        # Show cluster selection if clusters are available
        if st.session_state.advanced_clusters is not None:
            available_clusters = st.session_state.advanced_clusters['cluster_name'].unique().tolist()
            selected_clusters = st.multiselect(
                "Select Clusters:",
                options=available_clusters,
                default=[available_clusters[0]] if available_clusters else [],
                help="Choose SKUs from specific clusters"
            )

            if selected_clusters:
                # Get SKUs from selected clusters
                cluster_skus = st.session_state.advanced_clusters[
                    st.session_state.advanced_clusters['cluster_name'].isin(selected_clusters)
                ]['sku'].tolist()
                st.session_state.advanced_selected_skus = cluster_skus

                # Show count of selected SKUs
                st.info(f"Selected {len(cluster_skus)} SKUs from {len(selected_clusters)} cluster(s)")
            else:
                st.session_state.advanced_selected_skus = []
        else:
            st.warning("Run forecast first to generate clusters and enable selection by cluster")
            st.session_state.advanced_selected_skus = []
    else:
        # All SKUs selected
        st.session_state.advanced_selected_skus = []  # Empty means all SKUs

    # Divider
    st.divider()

    # Run Forecast button in sidebar - always visible
    forecast_button_text = "Run Advanced Forecast"

    # Always show the button regardless of state
    if st.button(
        forecast_button_text, 
        key="run_advanced_forecast_button_sidebar",
        use_container_width=True
    ):
        # Set forecast in progress flag
        st.session_state.advanced_forecast_in_progress = True
        st.session_state.advanced_forecast_progress = 0
        st.session_state.run_advanced_forecast = True
        # Tab will remain the same
        st.rerun()  # Rerun to update the UI with forecast tab active

    # Show status message but don't hide the button
    if st.session_state.advanced_forecast_in_progress:
        st.info("Forecast generation in progress...")

    # Secondary Sales Analysis Section
    st.subheader("Secondary Sales Analysis")

    # Explain the primary vs secondary sales concept
    st.markdown("""
    In business models with distributors, primary sales (factory to distributor) 
    may not reflect actual consumer demand (secondary sales). This tool analyzes primary 
    sales patterns to estimate true secondary sales and identify sales noise.
    """)

    # Algorithm selection for secondary sales calculation
    st.write("Select algorithm for secondary sales estimation:")
    secondary_algorithm = st.radio(
        "Algorithm",
        options=["robust_filter", "decomposition", "arima_smoothing"],
        index=0,
        horizontal=True,
        help="Choose method for calculating secondary sales from primary sales data"
    )
    st.session_state.secondary_sales_algorithm = secondary_algorithm

    # Option to run analysis on selected SKU only or all SKUs
    run_for_all = st.checkbox("Run for all selected SKUs", value=False)

    # Data type selection for forecasting
    st.write("Select data type for forecasting:")
    forecast_data_type = st.radio(
        "Forecast based on:",
        options=["primary", "secondary"],
        index=0,
        horizontal=True,
        help="Choose whether to run forecasts on primary or calculated secondary sales data"
    )
    st.session_state.forecast_data_type = forecast_data_type

    # Button to run secondary sales analysis
    secondary_button_text = "Run Secondary Sales Analysis"
    if not run_for_all and st.session_state.advanced_selected_skus:
        secondary_button_text = f"Analyze Secondary Sales for Selected SKU"

def update_selected_skus():
    """Update the list of selected SKUs based on cluster selection"""
    if st.session_state.advanced_show_all_clusters:
        st.session_state.advanced_selected_skus = []  # All SKUs
    else:
        selected_clusters = [st.session_state.selected_cluster]
        # Get SKUs from selected clusters
        cluster_skus = st.session_state.advanced_clusters[
            st.session_state.advanced_clusters['cluster_name'].isin(selected_clusters)
        ]['sku'].tolist()
        st.session_state.advanced_selected_skus = cluster_skus

def plot_secondary_sales_analysis(sku, analysis_result):
    """
    Plot secondary sales analysis results with primary, secondary, and noise

    Parameters:
    -----------
    sku : str
        SKU identifier
    analysis_result : dict
        Analysis results from analyze_sku_sales_pattern

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with primary sales, secondary sales, and noise
    """
    # Extract data from analysis result
    if analysis_result['status'] != 'success' or 'data' not in analysis_result:
        return None

    data = analysis_result['data'].copy()  # Make a copy to avoid changing the original

    # Create subplot with 2 rows - one for primary/secondary sales, one for noise
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f'Primary vs Secondary Sales for SKU: {sku}',
            'Distribution Noise Component'
        ),
        row_heights=[0.7, 0.3]
    )

    # Add primary sales line (top subplot)
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['primary_sales'],
            mode='lines+markers',
            name='Primary Sales',
            line=dict(color='blue', width=2),
            legendgroup='primary'
        ),
        row=1, col=1
    )

    # Add secondary sales line (top subplot)
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['secondary_sales'],
            mode='lines+markers',
            name='Estimated Secondary Sales',
            line=dict(color='green', width=2),
            legendgroup='secondary'
        ),
        row=1, col=1
    )

    # Add filled area to highlight the difference
    # Calculate difference between primary and secondary sales
    difference = data['primary_sales'] - data['secondary_sales']
    positive_diff = difference.copy()
    negative_diff = difference.copy()
    positive_diff[positive_diff < 0] = 0
    negative_diff[negative_diff > 0] = 0

    # Add positive difference (overstocking) with fill
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['primary_sales'],
            mode='none',
            name='Overstocking',
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.3)',
            showlegend=True,
            legendgroup='diff_pos',
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['primary_sales'] - positive_diff,
            mode='none',
            showlegend=False,
            legendgroup='diff_pos',
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # Add noise as a bar chart (bottom subplot)
    fig.add_trace(
        go.Bar(
            x=data['date'],
            y=data['noise'],
            name='Distribution Noise',
            marker_color='rgba(255, 0, 0, 0.7)'
        ),
        row=2, col=1
    )

    # Add zero line for reference on noise chart
    fig.add_shape(
        type="line",
        x0=data['date'].min(),
        y0=0,
        x1=data['date'].max(),
        y1=0,
        line=dict(color="black", width=1, dash="dot"),
        row=2, col=1
    )

    # Customize layout
    fig.update_layout(
        title=f'Enhanced Secondary Sales Analysis for SKU: {sku}',
        xaxis_title='',
        xaxis2_title='Date',
        yaxis_title='Sales Quantity',
        yaxis2_title='Noise',
        legend=dict(
            x=0.01, 
            y=0.99, 
            bgcolor='rgba(255, 255, 255, 0.8)',
            traceorder='normal'
        ),
        hovermode='x unified',
        height=700
    )

    # Add informational annotations
    fig.add_annotation(
        text="Positive noise represents overstocking<br>Negative noise represents potential lost sales",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=12, color="gray")
    )

    # Calculate and display key metrics as annotations
    avg_primary = data['primary_sales'].mean()
    avg_secondary = data['secondary_sales'].mean()
    avg_noise = data['noise'].mean()
    noise_pct = (abs(data['noise']).sum() / data['primary_sales'].sum() * 100)

    metrics_text = (
        f"Avg Primary: {avg_primary:.2f}<br>"
        f"Avg Secondary: {avg_secondary:.2f}<br>"
        f"Avg Noise: {avg_noise:.2f}<br>"
        f"Noise %: {noise_pct:.2f}%"
    )

    fig.add_annotation(
        text=metrics_text,
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        showarrow=False,
        align="right",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="lightgray",
        borderwidth=1,
        font=dict(size=12)
    )

    return fig

def run_secondary_sales_analysis(selected_sku=None, run_for_all=False, algorithm="robust_filter"):
    """
    Run secondary sales analysis for selected SKU(s)

    Parameters:
    -----------
    selected_sku : str, optional
        SKU to analyze. If None and run_for_all is True, analyzes all selected SKUs.
    run_for_all : bool, optional
        Whether to run analysis for all selected SKUs
    algorithm : str, optional
        Algorithm to use for secondary sales estimation

    Returns:
    --------
    dict
        Results of the analysis
    """
    # Progress indicators
    progress_container = st.empty()
    status_container = st.empty()

    # Determine which SKUs to analyze
    skus_to_analyze = []
    if run_for_all:
        if st.session_state.advanced_selected_skus:
            skus_to_analyze = st.session_state.advanced_selected_skus
        else:
            skus_to_analyze = sorted(st.session_state.sales_data['sku'].unique().tolist())
        status_container.info(f"Analyzing secondary sales patterns for {len(skus_to_analyze)} SKUs...")
    elif selected_sku:
        skus_to_analyze = [selected_sku]
        status_container.info(f"Analyzing secondary sales pattern for SKU: {selected_sku}")
    else:
        status_container.warning("No SKU selected for analysis")
        return {}

    results = {}

    # Define progress callback
    def secondary_progress_callback(current_index, current_sku, total_skus):
        progress = min(float(current_index) / total_skus, 1.0)
        progress_container.progress(progress)
        status_container.info(f"Analyzing SKU {current_index+1}/{total_skus}: {current_sku}")

    try:
        # Run analysis
        results = bulk_analyze_sales(
            st.session_state.sales_data,
            selected_skus=skus_to_analyze,
            algorithm=algorithm,
            progress_callback=secondary_progress_callback
        )

        # Store in session state
        st.session_state.secondary_sales_results = results
        st.session_state.run_secondary_analysis = True

        # Clear progress indicators
        progress_container.empty()
        if len(skus_to_analyze) > 1:
            status_container.success(f"✅ Secondary sales analysis completed for {len(skus_to_analyze)} SKUs")
        else:
            status_container.success(f"✅ Secondary sales analysis completed for {skus_to_analyze[0]}")

        return results

    except Exception as e:
        progress_container.empty()
        status_container.error(f"❌ Error analyzing secondary sales: {str(e)}")
        return {}

def forecast_progress_callback(current_index, current_sku, total_skus, message=None, level="info"):
    """
    Enhanced callback function to update progress during forecasting and log detailed messages

    Parameters:
    -----------
    current_index : int
        Current SKU index being processed
    current_sku : str
        SKU identifier currently being processed
    total_skus : int
        Total number of SKUs to process
    message : str, optional
        Specific message to log about the current process
    level : str, optional
        Message level ('info', 'warning', 'error', 'success')
    """
    # Calculate progress percentage
    progress = min(float(current_index) / total_skus, 1.0)
    st.session_state.advanced_forecast_progress = progress
    st.session_state.advanced_current_sku = current_sku

    # Extract current model from message if present
    current_model = ""
    if message and "model" in message.lower():
        # Try to extract model name from messages like "Training and evaluating MODEL model"
        model_indicators = ["training", "evaluating", "testing", "selected"]
        for indicator in model_indicators:
            if indicator in message.lower():
                parts = message.split()
                for i, part in enumerate(parts):
                    if part.lower() in ["auto_arima", "prophet", "ets", "theta", "lstm", "tcn", "ensemble", "moving_average"]:
                        current_model = part.upper()
                        break

    # Store current model in session state
    if current_model:
        st.session_state.advanced_current_model = current_model

    # If we have log_messages in session state and a message was provided, add it to logs
    if 'log_messages' in st.session_state and message:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.log_messages.append({
            "timestamp": timestamp,
            "message": message,
            "level": level
        })

# Main content
# Create main tabs for different analyses
# Create tabs
tab_names = ["Sales Data Analysis", "Multi-Model Forecasting", "Secondary Sales Analysis", "Hyperparameter Tuning"]

# Use active_tab to determine which tab should be active
if 'active_tab' in st.session_state and st.session_state.active_tab is not None:
    # Find the index of the active tab
    try:
        active_tab_index = tab_names.index(st.session_state.active_tab)
        # Clear the active tab to avoid persisting after reload
        st.session_state.active_tab = None
    except ValueError:
        active_tab_index = 0
else:
    active_tab_index = 0

tab_sales, tab_forecast, tab_secondary, tab_hyperparameter = st.tabs(tab_names)

# Add any needed session state variables for hyperparameter tuning
if 'tuning_progress' not in st.session_state:
    st.session_state.tuning_progress = 0
if 'tuning_models' not in st.session_state:
    st.session_state.tuning_models = []
if 'tuning_results' not in st.session_state:
    st.session_state.tuning_results = {}
if 'tuning_log_messages' not in st.session_state:
    st.session_state.tuning_log_messages = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = tab_names[0]

# Tab 1: Sales Data Analysis
with tab_sales:
    # Allow SKU selection in main area
    if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
        # Get list of SKUs from sales data
        all_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())

        # Add a prominent SKU selector
        selected_sku = st.selectbox(
            "Select a SKU to view historical data:",
            options=all_skus,
            key="sales_analysis_sku_selector"
        )

        if selected_sku:
            st.subheader(f"Historical Data for {selected_sku}")

            # Filter data for selected SKU
            sku_data = st.session_state.sales_data[st.session_state.sales_data['sku'] == selected_sku]

            # Show data preview
            st.dataframe(sku_data, use_container_width=True)

            # Plot historical data
            fig = px.line(
                sku_data, 
                x='date', 
                y='quantity',
                title=f"Historical Sales for {selected_sku}",
                markers=True
            )

            # Improve the layout
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Quantity",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show basic statistics
            st.subheader("Basic Statistics")

            col1, col2 = st.columns(2)

            with col1:
                mean_val = sku_data['quantity'].mean()
                median_val = sku_data['quantity'].median()
                std_val = sku_data['quantity'].std()

                st.metric("Mean", f"{mean_val:.2f}")
                st.metric("Median", f"{median_val:.2f}")
                st.metric("Standard Deviation", f"{std_val:.2f}")

            with col2:
                # Calculate growth metrics if possible
                if len(sku_data) > 1:
                    first_val = sku_data.iloc[0]['quantity']
                    last_val = sku_data.iloc[-1]['quantity']
                    change = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0

                    # Maximum and minimum values
                    max_val = sku_data['quantity'].max()
                    min_val = sku_data['quantity'].min()

                    st.metric("Min", f"{min_val:.2f}")
                    st.metric("Max", f"{max_val:.2f}")
                    st.metric("First-to-Last Change", f"{change:.2f}%")
    else:
        st.warning("No sales data loaded. Please upload sales data first.")

# Tab 2: Secondary Sales Analysis
with tab_secondary:
    st.subheader("Secondary Sales Analysis")

    # Information about secondary sales
    st.markdown("""
    Secondary sales analysis helps estimate actual consumer demand from primary sales data.
    This tool analyzes sales patterns to separate real demand from distribution effects.
    """)

    # Create a button to run secondary sales analysis
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Run analysis to estimate secondary sales from primary sales data:")
    with col2:
        if st.button("Run Secondary Analysis", key="run_secondary_from_tab"):
            selected_sku = None
            if st.session_state.advanced_selected_sku:
                selected_sku = st.session_state.advanced_selected_sku

            # Run the analysis without triggering forecast generation
            run_secondary_sales_analysis(
                selected_sku=selected_sku,
                run_for_all=False,
                algorithm=st.session_state.secondary_sales_algorithm
            )

    # Display secondary sales results if available
    if st.session_state.run_secondary_analysis and st.session_state.secondary_sales_results:
        # Get list of analyzed SKUs
        analyzed_skus = sorted(list(st.session_state.secondary_sales_results.keys()))

        if analyzed_skus:
            # Select SKU for secondary sales analysis
            selected_secondary_sku = st.selectbox(
                "Select a SKU to view secondary sales analysis",
                options=analyzed_skus,
                index=0 if st.session_state.advanced_selected_sku not in analyzed_skus else analyzed_skus.index(st.session_state.advanced_selected_sku),
                key="secondary_analysis_sku_selector"
            )

            if selected_secondary_sku in st.session_state.secondary_sales_results:
                analysis_result = st.session_state.secondary_sales_results[selected_secondary_sku]

                # Show analysis overview
                st.subheader(f"Secondary Sales Analysis for {selected_secondary_sku}")

                # Status message
                if analysis_result['status'] == 'success':
                    st.success("✅ Analysis completed successfully")

                    # Show metrics
                    if 'metrics' in analysis_result:
                        metrics = analysis_result['metrics']
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Avg Primary Sales", f"{metrics['avg_primary']:.2f}")

                        with col2:
                            st.metric("Avg Secondary Sales", f"{metrics['avg_secondary']:.2f}")

                        with col3:
                            st.metric("Avg Noise", f"{metrics['avg_noise']:.2f}")

                        with col4:
                            st.metric("Noise %", f"{metrics['noise_percentage']:.2f}%")

                    # Show data table prominently
                    if 'data' in analysis_result:
                        st.subheader("Primary vs Secondary Sales Data")
                        st.dataframe(analysis_result['data'], use_container_width=True)

                    # Show enhanced visualization
                    st.subheader("Visual Analysis")
                    fig = plot_secondary_sales_analysis(selected_secondary_sku, analysis_result)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # Add explanation and insights
                    st.subheader("Insights")

                    if 'metrics' in analysis_result:
                        metrics = analysis_result['metrics']
                        noise_pct = metrics['noise_percentage']

                        if noise_pct > 20:
                            st.warning(f"⚠️ High distribution noise detected ({noise_pct:.2f}%). This SKU shows significant discrepancy between primary and secondary sales, suggesting potential supply chain inefficiencies.")
                        elif noise_pct > 10:
                            st.info(f"ℹ️ Moderate distribution noise detected ({noise_pct:.2f}%). Consider optimizing inventory levels for this SKU.")
                        else:
                            st.success(f"✅ Low distribution noise ({noise_pct:.2f}%). This SKU shows good alignment between primary and secondary sales patterns.")
                else:
                    st.error(f"❌ Analysis failed: {analysis_result.get('message', 'Unknown error')}")
    else:
        st.info("Run secondary sales analysis to view results. This will help differentiate between primary sales (to distributors) and estimated secondary sales (to end consumers).")

# Tab 3: Forecast Analysis
with tab_forecast:
    st.subheader("Multi-Model Demand Forecasting")
    
    # Model selection section
    st.write("#### 1. Select Forecasting Models")
    
    # Define available models with descriptions
    available_models = {
        "auto_arima": "Auto ARIMA (Statistical time series model with automatic parameter selection)",
        "prophet": "Prophet (Facebook's decomposable time series model, handles seasonality)",
        "ets": "ETS (Exponential smoothing state space model)",
        "theta": "Theta (Statistical forecasting method with decomposition)",
        "lstm": "LSTM (Deep learning model for sequence prediction)",
        "ensemble": "Ensemble (Combines multiple models for improved accuracy)"
    }
    
    # Multi-model selection
    model_col1, model_col2 = st.columns(2)
    
    selected_models = []
    with model_col1:
        for model_key in list(available_models.keys())[:3]:  # First half of models
            if st.checkbox(f"{model_key.upper()}", 
                        value=model_key in st.session_state.advanced_models,
                        help=available_models[model_key],
                        key=f"forecast_model_{model_key}"):
                selected_models.append(model_key)
    
    with model_col2:
        for model_key in list(available_models.keys())[3:]:  # Second half of models
            if st.checkbox(f"{model_key.upper()}", 
                        value=model_key in st.session_state.advanced_models,
                        help=available_models[model_key],
                        key=f"forecast_model_{model_key}"):
                selected_models.append(model_key)
    
    # Update session state with selected models
    st.session_state.advanced_models = selected_models
    
    # If no models are selected, show warning
    if not selected_models:
        st.warning("Please select at least one forecasting model.")
    
    # Forecast control section
    st.write("#### 2. Run Forecast")
    
    # Create a placeholder for the progress bar
    progress_placeholder = st.empty()
    
    # Run button for forecast
    if st.button(
        "Run Multi-Model Forecast", 
        key="run_multi_model_forecast_button",
        disabled=len(selected_models) == 0,
        use_container_width=True
    ):
        # Set forecast in progress flag
        st.session_state.advanced_forecast_in_progress = True
        st.session_state.advanced_forecast_progress = 0
        st.session_state.run_advanced_forecast = True
        st.rerun()  # Rerun to update UI
    
    # Show status message if forecast in progress
    if st.session_state.advanced_forecast_in_progress:
        st.info("Forecast generation in progress...")

    # Show progress bar when forecast is in progress
    if st.session_state.advanced_forecast_in_progress:
        with progress_placeholder.container():
            # Create a two-column layout for the progress display
            progress_cols = st.columns([3, 1])

            with progress_cols[0]:
                # Header for progress display with animation effect
                st.markdown('<h3 style="color:#0066cc;"><span class="highlight">🔄 Advanced Forecast Generation in Progress</span></h3>', unsafe_allow_html=True)

                # Progress bar with custom styling
                progress_bar = st.progress(st.session_state.advanced_forecast_progress)

                # Status text placeholder
                status_text = st.empty()
                status_text.info(f"Processing SKU: {st.session_state.advanced_current_sku}")

                # Add a progress details section
                progress_details = st.empty()
                progress_percentage = int(st.session_state.advanced_forecast_progress * 100)
                progress_details.markdown(f"""
                **Progress:** {progress_percentage}%  
                **Current SKU:** {st.session_state.advanced_current_sku}  
                **Current Model:** {st.session_state.advanced_current_model if st.session_state.advanced_current_model else "Initializing..."}  
                **Models being evaluated:** {', '.join(st.session_state.advanced_models)}
                """)

            with progress_cols[1]:
                # Add a spinning icon or other visual indicator
                st.markdown('<div class="loader"></div>', unsafe_allow_html=True)

            # Create a detailed log area
            log_area = st.expander("View Processing Log", expanded=True)
            with log_area:
                # Format log messages with appropriate styling
                log_html = '<div style="height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.8em; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'

                for log in st.session_state.log_messages[-100:]:  # Show last 100 messages
                    if log["level"] == "info":
                        color = "black"
                    elif log["level"] == "warning":
                        color = "orange"
                    elif log["level"] == "error":
                        color = "red"
                    elif log["level"] == "success":
                        color = "green"
                    else:
                        color = "blue"

                    log_html += f'<div style="margin-bottom: 3px;"><span style="color: gray;">[{log["timestamp"]}]</span> <span style="color: {color};">{log["message"]}</span></div>'

                log_html += '</div>'

                # Display the log
                st.markdown(log_html, unsafe_allow_html=True)

            # Define log content area placeholders
            log_header = st.empty()
            log_content = st.empty()

            # Function to add log messages
            def add_log_message(message, level="info"):
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.log_messages.append({"timestamp": timestamp, "message": message, "level": level})

                # Format log messages with appropriate styling
                log_html = '<div style="height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.8em; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'

                for log in st.session_state.log_messages[-100:]:  # Show last 100 messages to avoid performance issues
                    if log["level"] == "info":
                        color = "black"
                    elif log["level"] == "warning":
                        color = "orange"
                    elif log["level"] == "error":
                        color = "red"
                    elif log["level"] == "success":
                        color = "green"
                    else:
                        color = "blue"

                    log_html += f'<div style="margin-bottom: 3px;"><span style="color: gray;">[{log["timestamp"]}]</span> <span style="color: {color};">{log["message"]}</span></div>'

                log_html += '</div>'

                # Update the log display
                log_header.markdown("### Real-time Processing Log")
                log_content.markdown(log_html, unsafe_allow_html=True)

            # Add initial log message
            add_log_message(f"Starting advanced forecast generation for {len(st.session_state.advanced_selected_skus) if st.session_state.advanced_selected_skus else 'all'} SKUs", "info")
            add_log_message(f"Forecast periods: {st.session_state.advanced_forecast_periods}", "info")
            add_log_message(f"Models to evaluate: {', '.join(selected_models)}", "info")
            add_log_message(f"Hyperparameter tuning: {'Enabled' if st.session_state.advanced_hyperparameter_tuning else 'Disabled'}", "info")
            add_log_message(f"Human-like sense check: {'Enabled' if st.session_state.advanced_apply_sense_check else 'Disabled'}", "info")
            add_log_message("Beginning clustering and feature extraction...", "info")

            # Create a custom callback that also updates logs
            def enhanced_progress_callback(current_index, current_sku, total_skus, message=None, level="info"):
                # Update progress
                progress = min(float(current_index) / total_skus, 1.0)
                st.session_state.advanced_forecast_progress = progress
                st.session_state.advanced_current_sku = current_sku

                # Extract current model from message if present
                current_model = ""
                if message and "model" in message.lower():
                    # Try to extract model name from messages like "Training and evaluating MODEL model"
                    model_indicators = ["training", "evaluating", "testing", "selected"]
                    for indicator in model_indicators:
                        if indicator in message.lower():
                            parts = message.split()
                            for i, part in enumerate(parts):
                                if part.lower() in [m.lower() for m in selected_models]:
                                    current_model = part.upper()
                                    break

                # Add to log if there's a message
                if message:
                    add_log_message(f"[SKU: {current_sku}] {message}", level)
                else:
                    # Default message
                    add_log_message(f"Processing SKU: {current_sku} ({current_index+1}/{total_skus})", "info")

                # Update UI elements
                progress_bar.progress(progress)
                progress_percentage = int(progress * 100)
                progress_details.markdown(f"""
                **Progress:** {progress_percentage}%  
                **Current SKU:** {current_sku} ({current_index+1}/{total_skus})  
                **Current Model:** {st.session_state.advanced_current_model if st.session_state.advanced_current_model else "Initializing..."}  
                **Models being evaluated:** {', '.join(selected_models)}
                """)

            # Override print function to capture output to log
            original_print = print
            def custom_print(*args, **kwargs):
                message = " ".join(map(str, args))
                add_log_message(message)
                original_print(*args, **kwargs)  # Still print to console for debugging

            # Monkey patch print function
            import builtins
            builtins.print = custom_print

            try:
                # Start the forecasting process
                add_log_message("Starting advanced forecasting process...", "info")

                # Check if we need to use secondary sales data for forecasting
                forecast_data_type = st.session_state.forecast_data_type
                if forecast_data_type == "secondary" and st.session_state.secondary_sales_results:
                    # Create a modified version of sales_data with secondary sales
                    add_log_message("Using SECONDARY sales data for forecasting...", "info")

                    # Get the secondary sales data
                    secondary_data_list = []

                    for sku in st.session_state.secondary_sales_results:
                        if st.session_state.secondary_sales_results[sku]['status'] == 'success' and 'data' in st.session_state.secondary_sales_results[sku]:
                            data = st.session_state.secondary_sales_results[sku]['data']
                            for _, row in data.iterrows():
                                secondary_data_list.append({
                                    'date': row['date'],
                                    'sku': sku,
                                    'quantity': row['secondary_sales']  # Use secondary sales as quantity
                                })

                    if secondary_data_list:
                        # Create a new DataFrame with secondary sales
                        secondary_sales_df = pd.DataFrame(secondary_data_list)
                        add_log_message(f"Created secondary sales DataFrame with {len(secondary_sales_df)} records", "info")

                        # Use this for forecasting
                        sales_data_to_use = secondary_sales_df
                    else:
                        add_log_message("No secondary sales data available, using primary sales instead", "warning")
                        sales_data_to_use = st.session_state.sales_data
                else:
                    add_log_message("Using PRIMARY sales data for forecasting...", "info")
                    sales_data_to_use = st.session_state.sales_data

                # Extract features for clustering if not already done
                if st.session_state.advanced_clusters is None:
                    status_text.write("Extracting features for clustering...")
                    features_df = extract_features(sales_data_to_use)

                    # Update progress
                    st.session_state.advanced_forecast_progress = 0.1
                    progress_bar.progress(st.session_state.advanced_forecast_progress)

                    # Perform clustering
                    status_text.write("Clustering SKUs by pattern...")
                    cluster_info = cluster_skus(features_df)
                    st.session_state.advanced_clusters = cluster_info

                    # Update progress
                    st.session_state.advanced_forecast_progress = 0.2
                    progress_bar.progress(st.session_state.advanced_forecast_progress)
                else:
                    # Use existing clusters
                    cluster_info = st.session_state.advanced_clusters

                    # Update progress
                    st.session_state.advanced_forecast_progress = 0.2
                    progress_bar.progress(st.session_state.advanced_forecast_progress)

                # Generate forecasts
                status_text.write("Generating advanced forecasts...")

                # Get selected SKUs
                selected_skus = st.session_state.advanced_selected_skus

                # Make sure we have at least one model selected
                if not selected_models:
                    selected_models = ["auto_arima", "prophet", "ets"]

                forecasts = advanced_generate_forecasts(
                    sales_data=sales_data_to_use,
                    cluster_info=cluster_info,
                    forecast_periods=st.session_state.advanced_forecast_periods,
                    auto_select=True,
                    models_to_evaluate=selected_models,
                    selected_skus=selected_skus,
                    progress_callback=enhanced_progress_callback,
                    hyperparameter_tuning=st.session_state.advanced_hyperparameter_tuning,
                    apply_sense_check=st.session_state.advanced_apply_sense_check,
                    use_param_cache=st.session_state.advanced_use_param_cache,
                    schedule_tuning=False  # We'll use the dedicated button for parameter tuning
                )

                # Update session state with the generated forecasts
                st.session_state.advanced_forecasts = forecasts
                add_log_message(f"Successfully generated forecasts for {len(forecasts)} SKUs", "success")

                # Update progress to 100%
                progress_bar.progress(1.0)
                status_text.write("Forecast generation complete!")
                add_log_message("Forecast generation complete!", "success")

            except Exception as e:
                error_message = f"Error during forecast generation: {str(e)}"
                add_log_message(error_message, "error")
                st.error(error_message)

            finally:
                # Restore original print function
                builtins.print = original_print

            # Set final progress
            progress_bar.progress(1.0)
            status_text.write("Forecast generation complete!")
            progress_details.markdown("""
            **Progress:** 100%  
            **Status:** Complete  
            **Next steps:** Explore the forecast results below
            """)

    # Display forecast results if available
    if st.session_state.run_advanced_forecast and st.session_state.advanced_forecasts:
        st.subheader("Forecast Results")

        # Get list of SKUs with forecasts
        forecasted_skus = sorted(list(st.session_state.advanced_forecasts.keys()))

        if forecasted_skus:
            # Select SKU for forecast display
            selected_forecast_sku = st.selectbox(
                "Select a SKU to view forecast",
                options=forecasted_skus,
                index=0 if st.session_state.advanced_selected_sku not in forecasted_skus else forecasted_skus.index(st.session_state.advanced_selected_sku),
                key="forecast_sku_selector"
            )

            if selected_forecast_sku in st.session_state.advanced_forecasts:
                forecast_result = st.session_state.advanced_forecasts[selected_forecast_sku]

                # Show forecast overview
                st.subheader(f"Forecast for {selected_forecast_sku}")

                # Show metrics
                if 'metrics' in forecast_result:
                    metrics = forecast_result['metrics']
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("MAPE", f"{metrics['mape']:.2f}%")

                    with col2:
                        st.metric("RMSE", f"{metrics['rmse']:.2f}")

                    with col3:
                        st.metric("Selected Model", f"{forecast_result['selected_model']}")

                    with col4:
                        # Growth metric
                        if 'growth' in metrics:
                            st.metric("Growth", f"{metrics['growth']:.2f}%")
                        else:
                            st.metric("Growth", "N/A")

                # Show forecast plot
                st.subheader("Forecast Visualization")

                # Create the properly formatted data structure for visualization
                # Make a copy of the train set for historical data to avoid modifying the original
                historical_data = forecast_result.get('train_set', pd.DataFrame()).copy()
                
                # Create a DataFrame with dates converted to strings to avoid timestamp math issues
                last_date = historical_data['date'].max() if not historical_data.empty else None
                
                # Create forecast dataframe with dates converted to strings to avoid timestamp math
                forecast_data = pd.DataFrame({
                    # Convert index timestamps to datetime
                    'date': pd.to_datetime(forecast_result['forecast'].index),
                    'forecast': forecast_result['forecast'].values,
                    'lower_bound': forecast_result['lower_bound'].values,
                    'upper_bound': forecast_result['upper_bound'].values
                })
                
                visualization_data = {
                    'sku': forecast_result.get('sku', selected_forecast_sku),
                    'historical_data': historical_data,
                    'forecast_data': forecast_data
                }
                
                # Get the plot from visualization utility
                forecast_fig = plot_forecast(visualization_data, show_anomalies=True, confidence_interval=0.90)
                if forecast_fig:
                    st.plotly_chart(forecast_fig, use_container_width=True)

                # Show forecast data table
                st.subheader("Forecast Data")

                # Create a forecast dataframe for display
                forecast_df = pd.DataFrame({
                    'date': forecast_result['forecast'].index,
                    'forecast': forecast_result['forecast'].values,
                    'lower_bound': forecast_result['lower_bound'].values,
                    'upper_bound': forecast_result['upper_bound'].values
                })

                # Format date for display
                if 'ds' in forecast_df.columns:
                    forecast_df['date'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')

                    # Function to highlight data columns
                    def highlight_data_columns(df):
                        # Define styles for different column types
                        styles = []

                        # Style for actual values
                        if 'y' in df.columns:
                            styles.append({
                                'selector': 'td:nth-child(3)',  # y column is usually 3rd
                                'props': 'background-color: rgba(144, 238, 144, 0.2);'  # light green
                            })

                        # Style for predicted values
                        if 'yhat' in df.columns:
                            styles.append({
                                'selector': 'td:nth-child(4)',  # yhat column is usually 4th
                                'props': 'background-color: rgba(135, 206, 250, 0.2);'  # light blue
                            })

                        # Style for lower bound
                        if 'yhat_lower' in df.columns:
                            styles.append({
                                'selector': 'td:nth-child(5)',  # yhat_lower column
                                'props': 'color: #999; font-style: italic;'
                            })

                        # Style for upper bound
                        if 'yhat_upper' in df.columns:
                            styles.append({
                                'selector': 'td:nth-child(6)',  # yhat_upper column
                                'props': 'color: #999; font-style: italic;'
                            })

                        return styles

                    # Display the data with custom styling
                    st.dataframe(forecast_df, use_container_width=True)

                    # Option to download forecast data
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast Data",
                        data=csv,
                        file_name=f"forecast_{selected_forecast_sku}.csv",
                        mime="text/csv"
                    )

                # Show model comparison if available
                if 'model_comparison' in forecast_result:
                    st.subheader("Model Comparison")

                    # Extract model comparison data
                    model_comparison = forecast_result['model_comparison']

                    # Convert to DataFrame for display
                    if isinstance(model_comparison, dict):
                        model_data = []
                        for model_name, metrics in model_comparison.items():
                            model_metrics = {'model': model_name}
                            if isinstance(metrics, dict):
                                model_metrics.update(metrics)
                            model_data.append(model_metrics)

                        if model_data:
                            model_df = pd.DataFrame(model_data)

                            # Function to highlight best model
                            def highlight_best_model(row):
                                if row['model'] == forecast_result['selected_model']:
                                    return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)
                                return [''] * len(row)

                            # Apply styling
                            styled_df = model_df.style.apply(highlight_best_model, axis=1)

                            # Display styled table
                            st.dataframe(styled_df, use_container_width=True)

                            # Show visual comparison of models
                            comparison_fig = plot_model_comparison(model_comparison, forecast_result['selected_model'])
                            if comparison_fig:
                                st.plotly_chart(comparison_fig, use_container_width=True)
    else:
        # Show instructions for running a forecast
        st.info("Configure forecast settings in the sidebar and click 'Run Advanced Forecast' to generate demand forecasts.")

        # Show a placeholder visualization
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            ### Advanced AI Forecasting Features:

            - **Automatic Model Selection**: System evaluates multiple models and selects the best for each SKU
            - **Intelligent Preprocessing**: Outlier detection and data cleaning before forecasting
            - **Clustering**: Groups similar SKUs for better pattern recognition
            - **Hyperparameter Tuning**: Optimizes model parameters for maximum accuracy
            - **Human-like Sense Check**: Applies business logic to ensure realistic projections
            """)

        with col2:
            st.markdown("""
            ### Benefits:

            - Reduce forecast error by up to 25%
            - Optimize inventory with accurate demand predictions
            - Save time with automated model selection
            - Improve planning with reliable long-term forecasts
            - Enhance decision-making with detailed forecast explanations
            """)

    # Add hyperparameter tuning section
    if st.session_state.advanced_clusters is not None:
        # Add expander for advanced tuning
        with st.expander("Hyperparameter Tuning", expanded=False):
            st.markdown("""
            ### Model Parameter Tuning

            Fine-tune forecast model parameters to maximize accuracy for your specific data patterns.
            This process can significantly improve forecast accuracy but requires more computation time.
            """)

            if st.button("Run Hyperparameter Tuning", key="run_parameter_tuning"):
                st.session_state.parameter_tuning_in_progress = True

                # Progress display for tuning
                tuning_progress = st.progress(0)
                tuning_status = st.empty()

                # Create callback for tuning step progress
                def tuning_step_callback(current_step, total_steps, message, level="info"):
                    progress = min(float(current_step) / total_steps, 1.0)
                    tuning_progress.progress(progress)
                    
                    # Also add to the tuning log messages
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.tuning_log_messages.append({
                        "timestamp": timestamp,
                        "message": message,
                        "level": level
                    })

                    if level == "error":
                        tuning_status.error(message)
                    elif level == "warning":
                        tuning_status.warning(message)
                    else:
                        tuning_status.info(message)

                # Run tuning process
                try:
                    # Get selected SKUs for tuning
                    tuning_skus = st.session_state.advanced_selected_skus
                    if not tuning_skus:
                        # If no specific SKUs selected, choose representatives from each cluster
                        if st.session_state.advanced_clusters is not None:
                            cluster_groups = st.session_state.advanced_clusters.groupby('cluster_name')
                            tuning_skus = [group.iloc[0]['sku'] for _, group in cluster_groups]

                    # Start tuning
                    tuning_status.info(f"Starting hyperparameter tuning for {len(tuning_skus)} representative SKUs...")

                    # Call the optimizer
                    optimize_parameters_async(
                        st.session_state.sales_data,
                        tuning_skus,
                        st.session_state.advanced_models,
                        progress_callback=tuning_step_callback
                    )

                    # Complete
                    tuning_progress.progress(1.0)
                    tuning_status.success("✅ Hyperparameter tuning scheduled successfully! Optimized parameters will be used for future forecasts.")

                except Exception as e:
                    tuning_status.error(f"❌ Error during hyperparameter tuning: {str(e)}")

                finally:
                    st.session_state.parameter_tuning_in_progress = False

            # Show tuning status if available
            if st.button("Check Tuning Status", key="check_tuning_status"):
                status_info = get_optimization_status()

                if status_info:
                    st.json(status_info)
                else:
                    st.info("No active tuning jobs found.")

# Secondary sales analysis button in sidebar
if st.sidebar.button(
    secondary_button_text,
    key="run_secondary_analysis_button",
    use_container_width=True
):
    # Determine which SKU to analyze
    selected_sku = None
    if not run_for_all and st.session_state.advanced_selected_skus:
        selected_sku = st.session_state.advanced_selected_skus[0]

    # Run the analysis without triggering forecast generation
    run_secondary_sales_analysis(
        selected_sku=selected_sku,
        run_for_all=run_for_all,
        algorithm=st.session_state.secondary_sales_algorithm
    )
    st.rerun()  # Rerun to update the UI with secondary tab active

# Code to run the forecast if the button was clicked
if st.session_state.run_advanced_forecast:
    # Create container for the progress display
    forecast_container = st.empty()

    with forecast_container.container():
        try:
            # Start the forecasting process
            if 'log_messages' not in st.session_state:
                st.session_state.log_messages = []

            # Initialize log_messages as an empty list if needed
            if not hasattr(st.session_state, 'log_messages'):
                st.session_state.log_messages = []

            st.session_state.log_messages.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": "Starting advanced forecasting process...",
                "level": "info"
            })

            # Check if we need to use secondary sales data for forecasting
            forecast_data_type = st.session_state.forecast_data_type
            if forecast_data_type == "secondary" and st.session_state.secondary_sales_results:
                # Create a modified version of sales_data with secondary sales
                st.session_state.log_messages.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": "Using SECONDARY sales data for forecasting...",
                    "level": "info"
                })

                # Get the secondary sales data
                secondary_data_list = []

                for sku in st.session_state.secondary_sales_results:
                    if st.session_state.secondary_sales_results[sku]['status'] == 'success' and 'data' in st.session_state.secondary_sales_results[sku]:
                        data = st.session_state.secondary_sales_results[sku]['data']
                        for _, row in data.iterrows():
                            secondary_data_list.append({
                                'date': row['date'],
                                'sku': sku,
                                'quantity': row['secondary_sales']  # Use secondary sales as quantity
                            })

                if secondary_data_list:
                    # Create a new DataFrame with secondary sales
                    secondary_sales_df = pd.DataFrame(secondary_data_list)
                    st.session_state.log_messages.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": f"Created secondary sales DataFrame with {len(secondary_sales_df)} records",
                        "level": "info"
                    })

                    # Use this for forecasting
                    sales_data_to_use = secondary_sales_df
                else:
                    st.session_state.log_messages.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": "No secondary sales data available, using primary sales instead",
                        "level": "warning"
                    })
                    sales_data_to_use = st.session_state.sales_data
            else:
                st.session_state.log_messages.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": "Using PRIMARY sales data for forecasting...",
                    "level": "info"
                })
                sales_data_to_use = st.session_state.sales_data

            # Extract features for clustering if not already done
            if st.session_state.advanced_clusters is None:
                st.session_state.log_messages.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": "Extracting features for clustering...",
                    "level": "info"
                })
                features_df = extract_features(sales_data_to_use)

                # Update progress
                st.session_state.advanced_forecast_progress = 0.1

                # Perform clustering
                st.session_state.log_messages.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": "Clustering SKUs by pattern...",
                    "level": "info"
                })
                cluster_info = cluster_skus(features_df)
                st.session_state.advanced_clusters = cluster_info

                # Update progress
                st.session_state.advanced_forecast_progress = 0.2
            else:
                # Use existing clusters
                cluster_info = st.session_state.advanced_clusters

                # Update progress
                st.session_state.advanced_forecast_progress = 0.2

            # Generate forecasts
            st.session_state.log_messages.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": "Generating advanced forecasts...",
                "level": "info"
            })

            # Get selected SKUs
            selected_skus = st.session_state.advanced_selected_skus

            # Make sure we have at least one model selected
            if not st.session_state.advanced_models:
                st.session_state.advanced_models = ["auto_arima", "prophet", "ets"]

            forecasts = advanced_generate_forecasts(
                sales_data=sales_data_to_use,
                cluster_info=cluster_info,
                forecast_periods=st.session_state.advanced_forecast_periods,
                auto_select=True,
                models_to_evaluate=st.session_state.advanced_models,
                selected_skus=selected_skus,
                progress_callback=forecast_progress_callback,
                hyperparameter_tuning=st.session_state.advanced_hyperparameter_tuning,
                apply_sense_check=st.session_state.advanced_apply_sense_check,
                use_param_cache=st.session_state.advanced_use_param_cache,
                schedule_tuning=False  # We'll use the dedicated button for parameter tuning
            )

            # Update session state with the generated forecasts
            st.session_state.advanced_forecasts = forecasts
            st.session_state.log_messages.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": f"Successfully generated forecasts for {len(forecasts)} SKUs",
                "level": "success"
            })

            # Reset forecast flags
            st.session_state.advanced_forecast_in_progress = False
            st.session_state.forecast_progress = 1.0

            # Show success message
            st.success(f"Successfully generated forecasts for {len(forecasts)} SKUs!")

        except Exception as e:
            # Log error
            error_message = f"Error during forecast generation: {str(e)}"
            st.session_state.log_messages.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": error_message,
                "level": "error"
            })
            st.error(error_message)

            # Reset flags
            st.session_state.advanced_forecast_in_progress = False

        finally:
            # Set final progress
            st.session_state.advanced_forecast_progress = 1.0

            # Rerun to update the UI
            st.rerun()
            
# Tab 4: Hyperparameter Tuning
with tab_hyperparameter:
    st.subheader("Model Hyperparameter Tuning")
    
    # Introduction to hyperparameter tuning
    st.markdown("""
    Hyperparameter tuning finds the optimal configuration for each forecasting model to maximize accuracy.
    This process improves forecast quality by adapting models to your specific data patterns.
    
    ### Benefits of Hyperparameter Tuning:
    - **Improved Accuracy**: Find optimal parameters for your unique data patterns
    - **Model Customization**: Adapt models to specific SKU characteristics
    - **Reduced Error**: Minimize forecast error through optimized models
    - **Better Decision Making**: Make decisions based on more accurate predictions
    """)
    
    # Tuning configuration section
    st.write("#### 1. Tuning Configuration")
    
    tuning_col1, tuning_col2 = st.columns(2)
    
    with tuning_col1:
        # Model selection for tuning
        st.write("Select models to tune:")
        
        tuning_models = []
        model_options = {
            "auto_arima": "Auto ARIMA",
            "prophet": "Prophet",
            "ets": "ETS",
            "theta": "Theta", 
            "lstm": "LSTM"
        }
        
        for model_key, model_name in model_options.items():
            if st.checkbox(model_name, value=model_key in st.session_state.advanced_models, key=f"tune_{model_key}"):
                tuning_models.append(model_key)
        
        # If no models selected, show warning
        if not tuning_models:
            st.warning("Please select at least one model to tune.")
    
    with tuning_col2:
        # Tuning scope selection
        st.write("Tuning Scope:")
        
        # Select which SKUs to tune for
        tune_all_clusters = st.checkbox("Tune for all clusters", value=True)
        
        # Select specific number of iterations
        tuning_iterations = st.slider(
            "Number of iterations",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
            help="More iterations can produce better results but take longer to compute"
        )
        
        # Cache results checkbox
        cache_results = st.checkbox(
            "Cache tuning results", 
            value=True,
            help="Store optimized parameters for future use"
        )
    
    # Run tuning button  
    st.write("#### 2. Run Tuning Process")
    
    # Create button to start tuning
    if not st.session_state.parameter_tuning_in_progress:
        if st.button(
            "Run Hyperparameter Tuning",
            key="run_parameter_tuning",
            disabled=len(tuning_models) == 0,
            use_container_width=True
        ):
            st.session_state.parameter_tuning_in_progress = True
            st.session_state.tuning_progress = 0
            st.session_state.tuning_models = tuning_models
            # Tab will remain the same
            st.rerun()
    else:
        st.info("Hyperparameter tuning in progress...")
    
    # Show tuning progress when in progress
    if st.session_state.parameter_tuning_in_progress:
        # Create a two-column layout for the progress display
        tuning_progress_cols = st.columns([3, 1])
        
        with tuning_progress_cols[0]:
            # Header for progress display with animation effect
            st.markdown('<h3 style="color:#0066cc;"><span class="highlight">🔄 Hyperparameter Tuning in Progress</span></h3>', unsafe_allow_html=True)
            
            # Progress bar with custom styling
            tuning_progress = st.progress(st.session_state.tuning_progress)
            
            # Status text placeholder
            tuning_status = st.empty()
            
            # Add a progress details section
            tuning_progress_details = st.empty()
            tuning_progress_percentage = int(st.session_state.tuning_progress * 100)
            model_names = [model_options[model] for model in st.session_state.tuning_models]
            tuning_progress_details.markdown(f"""
            **Progress:** {tuning_progress_percentage}%  
            **Models being tuned:** {', '.join(model_names)}
            """)
        
        with tuning_progress_cols[1]:
            # Add a spinning icon or other visual indicator
            st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
        
        # Create a detailed log area
        tuning_log_area = st.expander("View Tuning Log", expanded=True)
        with tuning_log_area:
            # Format log messages with appropriate styling
            tuning_log_html = '<div style="height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.8em; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
            
            for log in st.session_state.tuning_log_messages[-100:]:  # Show last 100 messages
                if log["level"] == "info":
                    color = "black"
                elif log["level"] == "warning":
                    color = "orange"
                elif log["level"] == "error":
                    color = "red"
                elif log["level"] == "success":
                    color = "green"
                else:
                    color = "blue"
                
                tuning_log_html += f'<div style="margin-bottom: 3px;"><span style="color: gray;">[{log["timestamp"]}]</span> <span style="color: {color};">{log["message"]}</span></div>'
            
            tuning_log_html += '</div>'
            
            # Display the log
            st.markdown(tuning_log_html, unsafe_allow_html=True)
        
        # Create callback for tuning progress that uses separate logs from forecasting
        def tuning_progress_callback(sku, model_type, message, level="info"):
            # Update progress value from session state
            current_progress = st.session_state.get('tuning_progress', 0)
            tuning_progress.progress(current_progress)
            
            # Add message to the tuning log
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.tuning_log_messages.append({
                "timestamp": timestamp,
                "message": f"[{sku}] [{model_type}] {message}",
                "level": level
            })
            
            # Update status message
            if level == "error":
                tuning_status.error(f"[{sku}] [{model_type}] {message}")
            elif level == "warning":
                tuning_status.warning(f"[{sku}] [{model_type}] {message}")
            elif level == "success":
                tuning_status.success(f"[{sku}] [{model_type}] {message}")
            else:
                tuning_status.info(f"[{sku}] [{model_type}] {message}")
            
            # Return to enable chaining
            return current_progress
        
        # Run tuning process
        try:
            # Get selected SKUs for tuning
            tuning_skus = st.session_state.advanced_selected_skus
            if not tuning_skus or tune_all_clusters:
                # If no specific SKUs selected or tuning for all clusters,
                # choose representatives from each cluster
                if st.session_state.advanced_clusters is not None:
                    cluster_groups = st.session_state.advanced_clusters.groupby('cluster_name')
                    tuning_skus = [group.iloc[0]['sku'] for _, group in cluster_groups]
            
            # Start tuning
            tuning_status.info(f"Starting hyperparameter tuning for {len(tuning_skus)} representative SKUs...")
            
            # Prepare sales data
            if st.session_state.sales_data is not None:
                # For each selected SKU, run parameter optimization
                total_tuning_steps = len(tuning_skus) * len(st.session_state.tuning_models)
                current_step = 0
                
                # Prepare results storage
                if 'tuning_results' not in st.session_state:
                    st.session_state.tuning_results = {}
                
                # Process each SKU
                for idx, sku in enumerate(tuning_skus):
                    sku_data = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku].copy()
                    
                    if len(sku_data) < 12:  # Need at least 12 data points for meaningful tuning
                        tuning_status.warning(f"Skipping {sku}: Not enough data points for tuning")
                        continue
                    
                    # Process each model type
                    for model_type in st.session_state.tuning_models:
                        current_step += 1
                        st.session_state.tuning_progress = current_step / total_tuning_steps
                        
                        # Update status
                        tuning_status.info(f"Tuning parameters for {sku} using {model_type.upper()} model...")
                        
                        try:
                            # Here we would call the actual tuning function
                            # For now, we'll simulate the process
                            # This is where you would call your parameter optimizer
                            
                            # from utils.parameter_optimizer import optimize_parameters
                            # optimal_params = optimize_parameters(
                            #     sku_data, 
                            #     model_type, 
                            #     iterations=tuning_iterations,
                            #     callback=lambda msg, lvl='info': tuning_progress_callback(sku, model_type, msg, lvl)
                            # )
                            
                            # Simulate tuning process
                            import time
                            import random
                            
                            # Simulate parameter optimization taking time
                            for i in range(5):
                                time.sleep(0.2)  # Short delay for simulation
                                progress_msg = f"Iteration {i+1}/5: Testing parameter configuration"
                                tuning_progress_callback(sku, model_type, progress_msg)
                            
                            # Create dummy optimal parameters based on model type
                            if model_type == "auto_arima":
                                optimal_params = {"p": random.randint(1, 3), "d": 1, "q": random.randint(0, 2)}
                            elif model_type == "prophet":
                                optimal_params = {
                                    "changepoint_prior_scale": round(random.uniform(0.001, 0.5), 3),
                                    "seasonality_prior_scale": round(random.uniform(0.01, 10), 2),
                                    "seasonality_mode": random.choice(["additive", "multiplicative"])
                                }
                            elif model_type == "ets":
                                optimal_params = {
                                    "trend": random.choice([None, "add", "mul"]),
                                    "seasonal": random.choice([None, "add", "mul"]),
                                    "damped_trend": random.choice([True, False])
                                }
                            elif model_type == "theta":
                                optimal_params = {
                                    "theta": round(random.uniform(0, 2), 1)
                                }
                            elif model_type == "lstm":
                                optimal_params = {
                                    "units": random.choice([32, 64, 128]),
                                    "dropout": round(random.uniform(0.1, 0.5), 1),
                                    "epochs": random.randint(50, 200)
                                }
                            else:
                                optimal_params = {}
                            
                            # Store results
                            if sku not in st.session_state.tuning_results:
                                st.session_state.tuning_results[sku] = {}
                            
                            st.session_state.tuning_results[sku][model_type] = optimal_params
                            
                            # Report success
                            success_msg = f"Successfully tuned {model_type.upper()} parameters: {format_parameters(optimal_params, model_type)}"
                            tuning_progress_callback(sku, model_type, success_msg, "success")
                            
                        except Exception as e:
                            error_msg = f"Error tuning {model_type} for {sku}: {str(e)}"
                            tuning_progress_callback(sku, model_type, error_msg, "error")
                
                # Finalize progress
                st.session_state.tuning_progress = 1.0
                tuning_status.success(f"Hyperparameter tuning completed for {len(tuning_skus)} SKUs!")
                
                # Display tuning results
                st.subheader("Tuning Results")
                
                # Create tabs for each model type
                model_tabs = st.tabs([model_options[model] for model in st.session_state.tuning_models])
                
                # Populate each tab with the tuning results
                for i, model_type in enumerate(st.session_state.tuning_models):
                    with model_tabs[i]:
                        # Create a table of results
                        results_data = []
                        for sku in st.session_state.tuning_results:
                            if model_type in st.session_state.tuning_results[sku]:
                                params = st.session_state.tuning_results[sku][model_type]
                                
                                # Convert parameters to a more readable format
                                formatted_params = format_parameters(params, model_type)
                                
                                # Add to results data
                                results_data.append({"SKU": sku, "Parameters": formatted_params})
                        
                        if results_data:
                            # Convert to DataFrame for display
                            import pandas as pd
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True)
                        else:
                            st.info(f"No tuning results available for {model_options[model_type]}")
                
            else:
                tuning_status.error("No sales data loaded. Please upload sales data first.")
        
        except Exception as e:
            tuning_status.error(f"Error during hyperparameter tuning: {str(e)}")
        
        finally:
            # Set flag to indicate tuning is complete
            st.session_state.parameter_tuning_in_progress = False
            
            # Update progress to complete
            tuning_progress.progress(1.0)