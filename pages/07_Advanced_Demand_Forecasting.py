import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    
    # Hyperparameter Tuning
    hyperparameter_tuning = st.toggle(
        "Hyperparameter Tuning",
        value=st.session_state.advanced_hyperparameter_tuning,
        help="Automatically tune model parameters for best performance (slower but more accurate)"
    )
    st.session_state.advanced_hyperparameter_tuning = hyperparameter_tuning
    
    # Use Parameter Cache
    use_param_cache = st.toggle(
        "Use Parameter Cache",
        value=st.session_state.advanced_use_param_cache,
        help="Use previously optimized parameters from database for faster and more accurate forecasts"
    )
    st.session_state.advanced_use_param_cache = use_param_cache
    
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
    
    # Run Forecast button placeholder (will be populated in the main function)
    should_show_button = not st.session_state.advanced_forecast_in_progress
    forecast_button_text = "Run Advanced Forecast"
    
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
    
    data = analysis_result['data']
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add primary sales line
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['primary_sales'],
        mode='lines+markers',
        name='Primary Sales',
        line=dict(color='blue', width=2)
    ))
    
    # Add secondary sales line
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['secondary_sales'],
        mode='lines+markers',
        name='Estimated Secondary Sales',
        line=dict(color='green', width=2)
    ))
    
    # Add noise as a bar chart
    fig.add_trace(go.Bar(
        x=data['date'],
        y=data['noise'],
        name='Noise (Difference)',
        marker_color='rgba(255, 0, 0, 0.5)'
    ))
    
    # Customize layout
    fig.update_layout(
        title=f'Primary vs Secondary Sales Analysis for SKU: {sku}',
        xaxis_title='Date',
        yaxis_title='Sales Quantity',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
        hovermode='x unified',
        barmode='overlay'
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
if st.session_state.run_advanced_forecast and 'advanced_forecasts' in st.session_state and st.session_state.advanced_forecasts:
    # Show cluster analysis
    st.header("SKU Cluster Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Display cluster summary chart
        if st.session_state.advanced_clusters is not None:
            cluster_fig = plot_cluster_summary(st.session_state.advanced_clusters)
            st.plotly_chart(cluster_fig, use_container_width=True)

    with col2:
        # Show cluster details
        st.subheader("Cluster Characteristics")

        if 'advanced_clusters' in st.session_state and st.session_state.advanced_clusters is not None:
            cluster_groups = st.session_state.advanced_clusters.groupby('cluster_name').size().reset_index()
            cluster_groups.columns = ['Cluster', 'Count']

            # Calculate percentage
            total_skus = cluster_groups['Count'].sum()
            cluster_groups['Percentage'] = (cluster_groups['Count'] / total_skus * 100).round(1)
            cluster_groups['Percentage'] = cluster_groups['Percentage'].astype(str) + '%'

            st.dataframe(cluster_groups, use_container_width=True)

            # Show average metrics per cluster if available
            try:
                cluster_metrics = []
                
                for cluster in st.session_state.advanced_clusters['cluster_name'].unique():
                    # Get SKUs in this cluster
                    cluster_skus = st.session_state.advanced_clusters[
                        st.session_state.advanced_clusters['cluster_name'] == cluster
                    ]['sku'].tolist()
                    
                    # Calculate average metrics
                    mape_values = []
                    rmse_values = []
                    
                    for sku in cluster_skus:
                        if sku in st.session_state.advanced_forecasts:
                            forecast_data = st.session_state.advanced_forecasts[sku]
                            if 'model_evaluation' in forecast_data and 'metrics' in forecast_data['model_evaluation']:
                                best_model = forecast_data['model']
                                metrics = forecast_data['model_evaluation']['metrics'].get(best_model, {})
                                
                                if 'mape' in metrics and not np.isnan(metrics['mape']):
                                    mape_values.append(metrics['mape'])
                                
                                if 'rmse' in metrics and not np.isnan(metrics['rmse']):
                                    rmse_values.append(metrics['rmse'])
                    
                    # Calculate averages
                    avg_mape = np.mean(mape_values) if mape_values else np.nan
                    avg_rmse = np.mean(rmse_values) if rmse_values else np.nan
                    
                    cluster_metrics.append({
                        'Cluster': cluster,
                        'Avg MAPE': f"{avg_mape:.2f}%" if not np.isnan(avg_mape) else "N/A",
                        'Avg RMSE': f"{avg_rmse:.2f}" if not np.isnan(avg_rmse) else "N/A",
                        'SKUs': len(cluster_skus)
                    })
                
                if cluster_metrics:
                    st.subheader("Cluster Forecast Performance")
                    st.dataframe(pd.DataFrame(cluster_metrics), use_container_width=True)
            
            except Exception as e:
                st.error(f"Error calculating cluster metrics: {str(e)}")

    # SKU selection and details
    st.header("Forecast Details")

    # Get list of SKUs from forecast data
    sku_list = sorted(list(st.session_state.advanced_forecasts.keys()))

    # Select SKU section
    col1, col2 = st.columns([3, 1])

    with col1:
        # Find the default index for the dropdown
        default_index = 0
        if st.session_state.advanced_selected_sku in sku_list:
            default_index = sku_list.index(st.session_state.advanced_selected_sku)
        elif len(sku_list) > 0:
            default_index = 0

        selected_sku = st.selectbox(
            "Select a SKU to view forecast details",
            options=sku_list,
            index=default_index
        )
        st.session_state.advanced_selected_sku = selected_sku

    with col2:
        # Show basic info about the selected SKU
        if selected_sku and selected_sku in st.session_state.advanced_forecasts:
            forecast_data = st.session_state.advanced_forecasts[selected_sku]
            model_name = forecast_data['model'].upper()
            cluster_name = forecast_data['cluster_name']

            st.write(f"**Model:** {model_name}")
            st.write(f"**Cluster:** {cluster_name}")

            # Show a quick metric if available
            if 'model_evaluation' in forecast_data and 'metrics' in forecast_data['model_evaluation']:
                best_model = forecast_data['model']
                if best_model in forecast_data['model_evaluation']['metrics']:
                    metrics = forecast_data['model_evaluation']['metrics'][best_model]
                    if 'mape' in metrics and not np.isnan(metrics['mape']):
                        st.metric("Forecast Accuracy", f"{(100-metrics['mape']):.1f}%", help="Based on test data evaluation")

    # Show forecast details for selected SKU
    if selected_sku:
        forecast_data = st.session_state.advanced_forecasts[selected_sku]

        # Tab section for forecast views
        forecast_tabs = st.tabs(["Forecast Chart", "Model Comparison", "Forecast Metrics", "Pattern Analysis", "Sense Check", "Error Analysis", "Parameter Tuning"])

        with forecast_tabs[0]:
            # Forecast visualization section
            col1, col2 = st.columns([3, 1])

            with col1:
                # Get list of models to display
                # If we have multiple models selected, use them for visualization
                available_models = []
                selected_models_for_viz = []

                if 'model_evaluation' in forecast_data and 'all_models_forecasts' in forecast_data['model_evaluation']:
                    available_models = list(forecast_data['model_evaluation']['all_models_forecasts'].keys())

                    # Create checkboxes for display options
                    st.subheader("Display Options")

                    # Create columns for the checkboxes
                    col_options1, col_options2 = st.columns(2)

                    with col_options1:
                        # Option to show test predictions
                        show_test_predictions = st.checkbox(
                            "Show Test Predictions", 
                            value=False,
                            help="Display forecast line for test data period"
                        )

                        # Option to show all models or best model
                        show_all_models = st.checkbox(
                            "Show All Selected Models", 
                            value=False,
                            help="Show forecasts from all selected models"
                        )

                    with col_options2:
                        # Option to show confidence intervals
                        show_confidence = st.checkbox(
                            "Show Confidence Intervals", 
                            value=True,
                            help="Display confidence bands around forecast"
                        )

                        # Option to highlight anomalies
                        show_anomalies = st.checkbox(
                            "Highlight Anomalies", 
                            value=True,
                            help="Highlight detected anomalies in historical data"
                        )

                    # If showing all models, create multi-select for model selection
                    if show_all_models and available_models:
                        # Select which models to show
                        selected_models_for_viz = st.multiselect(
                            "Select models to display",
                            options=available_models,
                            default=[forecast_data['model']]  # Default to showing best model
                        )
                    else:
                        # Just show the best model
                        selected_models_for_viz = [forecast_data['model']]

                # Plot the forecast
                sales_data = st.session_state.sales_data
                
                # Use the selected models for visualization
                fig = plot_forecast(
                    sales_data=sales_data[sales_data['sku'] == selected_sku],
                    forecast_data=forecast_data,
                    sku=selected_sku,
                    selected_models=selected_models_for_viz
                )
                
                # If we're showing anomalies, highlight them on the chart
                if show_anomalies and 'train_set' in forecast_data:
                    try:
                        # Detect outliers in the training data
                        train_data = forecast_data['train_set']
                        outliers = detect_outliers(train_data['quantity'], method='zscore', threshold=3)
                        
                        if outliers.any():
                            # Add the outliers to the plot
                            outlier_points = train_data[outliers]
                            fig.add_trace(go.Scatter(
                                x=outlier_points['date'],
                                y=outlier_points['quantity'],
                                mode='markers',
                                marker=dict(color='red', size=10, symbol='circle-open'),
                                name='Anomalies'
                            ))
                    except Exception as e:
                        print(f"Error plotting anomalies: {str(e)}")
                
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Show forecast values
                st.subheader("Forecast Values")
                
                # Get forecast data
                forecast_series = forecast_data['forecast']
                
                # Create a DataFrame with the forecast values
                forecast_df = pd.DataFrame({
                    'Date': forecast_series.index,
                    'Forecast': forecast_series.values.round(2)
                })
                
                # Add confidence intervals if available
                if 'lower_bound' in forecast_data and 'upper_bound' in forecast_data:
                    forecast_df['Lower Bound'] = forecast_data['lower_bound'].values.round(2)
                    forecast_df['Upper Bound'] = forecast_data['upper_bound'].values.round(2)
                
                # Format the date column for display
                forecast_df['Date'] = forecast_df['Date'].dt.strftime('%b %Y')
                
                # Show the DataFrame
                st.dataframe(forecast_df, use_container_width=True)
                
                # Add download button for the forecast
                csv_buffer = io.BytesIO()
                forecast_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="Download Forecast",
                    data=csv_buffer,
                    file_name=f"{selected_sku}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        with forecast_tabs[1]:
            # Model comparison section
            st.subheader(f"Model Comparison for {selected_sku}")
            
            # Check if we have model evaluation data
            if 'model_evaluation' in forecast_data and 'metrics' in forecast_data['model_evaluation']:
                metrics = forecast_data['model_evaluation']['metrics']
                
                # Create a DataFrame for the metrics
                models = []
                mape_values = []
                rmse_values = []
                mae_values = []
                
                for model, model_metrics in metrics.items():
                    models.append(model.upper())
                    mape_values.append(model_metrics.get('mape', np.nan))
                    rmse_values.append(model_metrics.get('rmse', np.nan))
                    mae_values.append(model_metrics.get('mae', np.nan))
                
                metrics_df = pd.DataFrame({
                    'Model': models,
                    'MAPE (%)': mape_values,
                    'RMSE': rmse_values,
                    'MAE': mae_values
                })
                
                # Highlight the best model
                best_model = forecast_data['model'].upper()
                
                # Use a styled DataFrame
                def highlight_best_model(row):
                    styles = [''] * len(row)
                    if row['Model'] == best_model:
                        styles = ['background-color: rgba(75, 192, 192, 0.2)'] * len(row)
                    return styles
                
                # Sort by MAPE for display
                metrics_df = metrics_df.sort_values('MAPE (%)')
                
                # Display the styled DataFrame
                st.dataframe(metrics_df.style.apply(highlight_best_model, axis=1), use_container_width=True)
                
                # Also show a bar chart comparison
                if 'all_models_forecasts' in forecast_data['model_evaluation']:
                    # Use the plot_model_comparison function
                    comparison_fig = plot_model_comparison(selected_sku, forecast_data)
                    st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    # Also show forecast values from different models
                    all_models_forecasts = forecast_data['model_evaluation']['all_models_forecasts']
                    
                    # Get dates (use the first model's dates)
                    first_model = list(all_models_forecasts.keys())[0]
                    dates = all_models_forecasts[first_model].index
                    
                    # Create a DataFrame with all models' forecasts
                    forecasts_df = pd.DataFrame({'Date': dates})
                    
                    for model, forecast_series in all_models_forecasts.items():
                        forecasts_df[model.upper()] = forecast_series.values.round(2)
                    
                    # Format the date column for display
                    forecasts_df['Date'] = forecasts_df['Date'].dt.strftime('%b %Y')
                    
                    st.subheader("Forecasts by Model")
                    st.dataframe(forecasts_df, use_container_width=True)
            else:
                st.info("No model comparison data available for this SKU.")

        with forecast_tabs[2]:
            # Metrics details
            st.subheader(f"Forecast Metrics for {selected_sku}")
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Display basic forecast metrics
                if 'model_evaluation' in forecast_data and 'metrics' in forecast_data['model_evaluation']:
                    best_model = forecast_data['model']
                    if best_model in forecast_data['model_evaluation']['metrics']:
                        metrics = forecast_data['model_evaluation']['metrics'][best_model]
                        
                        # Create a metrics card
                        st.subheader("Accuracy Metrics")
                        metrics_df = pd.DataFrame({
                            'Metric': ['MAPE (%)', 'RMSE', 'MAE'],
                            'Value': [
                                f"{metrics.get('mape', np.nan):.2f}%", 
                                f"{metrics.get('rmse', np.nan):.2f}", 
                                f"{metrics.get('mae', np.nan):.2f}"
                            ]
                        })
                        
                        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                        
                        # Add interpretation
                        st.markdown(f"""
                        **Interpretation:**
                        - MAPE: Mean Absolute Percentage Error - measures forecast accuracy as percentage
                        - RMSE: Root Mean Square Error - emphasizes larger errors
                        - MAE: Mean Absolute Error - average absolute difference between forecast and actual
                        """)
            
            with col2:
                # Show model parameters if available
                st.subheader("Model Parameters")
                
                if 'model_evaluation' in forecast_data and 'forecasts' in forecast_data['model_evaluation']:
                    best_model = forecast_data['model']
                    all_forecasts = forecast_data['model_evaluation'].get('forecasts', {})
                    
                    if best_model in all_forecasts and 'params' in all_forecasts[best_model]:
                        params = all_forecasts[best_model]['params']
                        
                        # Format parameters for display
                        if isinstance(params, dict):
                            params_df = pd.DataFrame({
                                'Parameter': list(params.keys()),
                                'Value': [str(v) for v in params.values()]
                            })
                            
                            st.dataframe(params_df, use_container_width=True, hide_index=True)
                        else:
                            st.write(f"Parameters: {params}")
                    else:
                        st.info("No detailed parameters available for this model.")
                else:
                    st.info("No model parameters available.")

        with forecast_tabs[3]:
            # Pattern analysis section
            st.subheader(f"Pattern Analysis for {selected_sku}")
            
            if 'train_set' in forecast_data:
                train_data = forecast_data['train_set']
                
                try:
                    # Extract advanced features
                    features = extract_advanced_features(train_data['quantity'])
                    
                    # Create a DataFrame for display
                    features_df = pd.DataFrame({
                        'Feature': list(features.keys()),
                        'Value': list(features.values())
                    })
                    
                    # Format numeric values
                    features_df['Value'] = features_df['Value'].apply(
                        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else str(x)
                    )
                    
                    # Group features by category
                    basic_stats = features_df[features_df['Feature'].isin([
                        'mean', 'median', 'std', 'min', 'max', 'range', 'iqr'
                    ])]
                    
                    shape_features = features_df[features_df['Feature'].isin([
                        'skewness', 'kurtosis', 'cv'
                    ])]
                    
                    trend_features = features_df[features_df['Feature'].isin([
                        'trend_slope', 'trend_intercept'
                    ])]
                    
                    seasonality_features = features_df[features_df['Feature'].isin([
                        'autocorrelation_lag1', 'autocorrelation_lag12'
                    ])]
                    
                    stationarity_features = features_df[features_df['Feature'].isin([
                        'adf_pvalue', 'adf_statistic'
                    ])]
                    
                    intermittency_features = features_df[features_df['Feature'].isin([
                        'zero_count', 'zero_ratio'
                    ])]
                    
                    # Create columns for display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Basic Statistics")
                        st.dataframe(basic_stats, use_container_width=True, hide_index=True)
                        
                        st.subheader("Shape Features")
                        st.dataframe(shape_features, use_container_width=True, hide_index=True)
                        
                        st.subheader("Trend Analysis")
                        st.dataframe(trend_features, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.subheader("Seasonality Analysis")
                        st.dataframe(seasonality_features, use_container_width=True, hide_index=True)
                        
                        st.subheader("Stationarity Tests")
                        st.dataframe(stationarity_features, use_container_width=True, hide_index=True)
                        
                        st.subheader("Intermittency Analysis")
                        st.dataframe(intermittency_features, use_container_width=True, hide_index=True)
                    
                    # Change Points Analysis
                    st.subheader("Change Points Analysis")
                    
                    # Detect change points
                    change_points = detect_change_points(train_data['quantity'], method='window')
                    
                    if change_points:
                        # Create a figure
                        fig = go.Figure()
                        
                        # Add the main time series
                        fig.add_trace(go.Scatter(
                            x=train_data['date'],
                            y=train_data['quantity'],
                            mode='lines+markers',
                            name='Sales Data'
                        ))
                        
                        # Add vertical lines for change points
                        for cp in change_points:
                            if cp < len(train_data):
                                cp_date = train_data['date'].iloc[cp]
                                fig.add_shape(
                                    type="line",
                                    x0=cp_date,
                                    y0=0,
                                    x1=cp_date,
                                    y1=train_data['quantity'].max() * 1.1,
                                    line=dict(color="red", width=2, dash="dash")
                                )
                        
                        # Update layout
                        fig.update_layout(
                            title="Sales Data with Detected Change Points",
                            xaxis_title="Date",
                            yaxis_title="Quantity",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show change point dates
                        cp_dates = [train_data['date'].iloc[cp].strftime('%b %Y') for cp in change_points if cp < len(train_data)]
                        st.write(f"**Detected Change Points:** {', '.join(cp_dates)}")
                    else:
                        st.info("No significant change points detected in the time series.")
                    
                except Exception as e:
                    st.error(f"Error in pattern analysis: {str(e)}")
            else:
                st.info("No training data available for pattern analysis.")

        with forecast_tabs[4]:
            # Sense check section
            st.subheader(f"Forecast Sense Check for {selected_sku}")
            
            if 'sense_check' in forecast_data:
                sense_check = forecast_data['sense_check']
                
                # Display issues and adjustments
                issues = sense_check.get('issues_detected', [])
                adjustments = sense_check.get('adjustments_made', [])
                
                if issues:
                    st.warning("Issues Detected:")
                    for issue in issues:
                        st.markdown(f"- {issue}")
                else:
                    st.success("No issues detected in the forecast. The forecast appears to be reasonable.")
                
                if adjustments:
                    st.info("Adjustments Made:")
                    for adjustment in adjustments:
                        st.markdown(f"- {adjustment}")
                
                # Add explainable AI section
                st.subheader("Forecast Rationale")
                
                # Combine rationale from model selection and sense check
                rationale = []
                
                # Add model selection rationale
                best_model = forecast_data['model'].upper()
                rationale.append(f"- Selected {best_model} as the best model based on test performance")
                
                # Add data characteristics
                if 'train_set' in forecast_data:
                    train_data = forecast_data['train_set']
                    
                    # Simple trend analysis
                    x = np.arange(len(train_data))
                    trend_coef = np.polyfit(train_data['quantity'].values, x, 1)[0]
                    trend_direction = "upward" if trend_coef > 0.1 else "downward" if trend_coef < -0.1 else "flat"
                    rationale.append(f"- Data shows a {trend_direction} trend")
                    
                    # Seasonality check
                    if len(train_data) >= 12:
                        try:
                            acf_12 = pd.Series(train_data['quantity'].values).autocorr(lag=12)
                            if abs(acf_12) > 0.3:
                                rationale.append(f"- Detected significant annual seasonality (correlation: {acf_12:.2f})")
                        except:
                            pass
                    
                    # Volatility
                    cv = train_data['quantity'].std() / train_data['quantity'].mean() if train_data['quantity'].mean() > 0 else 0
                    volatility = "high" if cv > 0.5 else "moderate" if cv > 0.2 else "low"
                    rationale.append(f"- Data has {volatility} volatility (CV: {cv:.2f})")
                
                # Add sense check rationale
                if issues:
                    for issue in issues:
                        rationale.append(f"- Issue: {issue}")
                
                if adjustments:
                    for adjustment in adjustments:
                        rationale.append(f"- Action: {adjustment}")
                
                # Display rationale
                for r in rationale:
                    st.markdown(r)
                
                # Add additional context
                st.markdown("---")
                st.markdown("This sense check is designed to mimic a human analyst's review of the forecast, applying business logic and pattern recognition to ensure that the forecasts are realistic and consistent with historical patterns.")
            else:
                st.info("No sense check information available for this forecast.")
                
        with forecast_tabs[5]:
            # Error Analysis section
            st.subheader(f"Error Analysis for {selected_sku}")
            
            # Add tabs for different error analysis views
            error_analysis_tabs = st.tabs(["Summary Report", "Detailed Analysis", "Error Patterns"])
            
            with error_analysis_tabs[0]:
                # Summary report section
                if 'model_evaluation' in forecast_data and 'test_actuals' in forecast_data and 'test_predictions' in forecast_data:
                    # Get actuals and predictions
                    actuals = forecast_data['test_actuals']
                    predictions = forecast_data['test_predictions']
                    
                    if len(actuals) > 0 and len(predictions) > 0:
                        # Run error analysis
                        error_analysis = analyze_forecast_errors(actuals, predictions)
                        
                        # Generate and display the error report
                        report = generate_error_report(error_analysis, model_name=forecast_data['model'])
                        st.markdown(f"```\n{report}\n```")
                        
                        # Show recommendations based on error analysis
                        st.subheader("Recommendations")
                        
                        patterns = error_analysis.get('patterns', {})
                        if patterns.get('systematic_bias', False):
                            st.warning("⚠️ **Systematic Bias Detected**: The forecast consistently over/underestimates demand.")
                            st.markdown("""
                            **Recommendations:**
                            - Check for missing variables or trends in the data
                            - Consider adding external factors that might affect demand
                            - Try different model types that can better capture the patterns
                            """)
                        
                        if patterns.get('autocorrelation', False):
                            st.warning("⚠️ **Error Autocorrelation Detected**: Errors follow a pattern over time.")
                            st.markdown("""
                            **Recommendations:**
                            - Consider models that better capture temporal dependencies
                            - Add more time-related features or seasonality components
                            - Try a time series model with built-in autocorrelation handling
                            """)
                            st.markdown("""
                            **Recommendations:**
                            - Your model is missing important seasonal patterns
                            - Try models with stronger seasonal components
                            - Consider adding calendar features or external regressors
                            """)
                        
                        if patterns.get('extreme_errors', False):
                            st.warning("⚠️ **Large Error Spikes Detected**: Some periods have unusually large errors.")
                            st.markdown("""
                            **Recommendations:**
                            - Identify and handle outliers in the historical data
                            - Use robust preprocessing techniques
                            - Consider ensemble methods to reduce extreme errors
                            """)
                    else:
                        st.info("Insufficient test data for error analysis.")
                else:
                    st.info("No test data available for error analysis.")
            
            with error_analysis_tabs[1]:
                # Detailed analysis section
                if 'model_evaluation' in forecast_data and 'test_actuals' in forecast_data and 'test_predictions' in forecast_data:
                    # Get actuals and predictions
                    actuals = forecast_data['test_actuals']
                    predictions = forecast_data['test_predictions']
                    
                    if len(actuals) > 0 and len(predictions) > 0:
                        # Run error analysis
                        error_analysis = analyze_forecast_errors(actuals, predictions)
                        
                        # Display detailed metrics
                        st.subheader("Error Metrics")
                        
                        # Create columns for metrics display
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Basic error statistics
                            error_stats = error_analysis.get('error_stats', {})
                            stats_df = pd.DataFrame({
                                'Metric': [
                                    'Mean Error', 
                                    'Mean Absolute Error', 
                                    'Root Mean Square Error', 
                                    'Mean Absolute Percentage Error',
                                    'Median Absolute Error'
                                ],
                                'Value': [
                                    f"{error_stats.get('mean_error', 0):.2f}",
                                    f"{error_stats.get('mean_abs_error', 0):.2f}",
                                    f"{error_stats.get('rmse', 0):.2f}",
                                    f"{error_stats.get('mape', 0):.2f}%",
                                    f"{error_stats.get('median_abs_error', 0):.2f}"ian_abs_error', 0):.2f}"
                                ]
                            })
                            
                            st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            # Error distribution
                            error_dist = error_analysis.get('error_distribution', {})
                            dist_df = pd.DataFrame({
                                'Metric': [
                                    '25th Percentile', 
                                    'Median Error', 
                                    '75th Percentile', 
                                    'IQR',
                                    'Max Error'
                                ],
                                'Value': [
                                    f"{error_dist.get('q25', 0):.2f}",
                                    f"{error_dist.get('median', 0):.2f}",
                                    f"{error_dist.get('q75', 0):.2f}",
                                    f"{error_dist.get('iqr', 0):.2f}",
                                    f"{error_stats.get('max_error', 0):.2f}"
                                ]
                            })
                            
                            st.dataframe(dist_df, use_container_width=True, hide_index=True)
                        
                        # Display error distribution histogram
                        st.subheader("Error Distribution")er("Error Distribution")
                        if 'error_details' in error_analysis:
                            error_details = error_analysis['error_details']
                            
                            # Create histogram of errors
                            fig = px.histogram(
                                error_details, 
                                x='error',
                                nbins=20,
                                title="Distribution of Forecast Errors",
                                labels={'error': 'Error', 'count': 'Frequency'},
                                color_discrete_sequence=['rgba(75, 192, 192, 0.6)']
                            )
                            
                            # Add a vertical line at zero
                            fig.add_vline(x=0, line_dash="dash", line_color="red")
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create scatter plot of predicted vs actual
                            fig = px.scatter(
                                error_details,
                                x='actual',
                                y='predicted',
                                title="Actual vs Predicted Values",
                                labels={'actual': 'Actual', 'predicted': 'Predicted'},
                                color_discrete_sequence=['rgba(75, 192, 192, 0.6)']
                            )
                            
                            # Add a diagonal line (perfect prediction)
                            min_val = min(error_details['actual'].min(), error_details['predicted'].min())
                            max_val = max(error_details['actual'].max(), error_details['predicted'].max())
                            fig.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                line=dict(color='red', dash='dash'),
                                name='Perfect Prediction'
                            ))
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Insufficient test data for detailed error analysis.")
                else:
                    st.info("No test data available for error analysis.")
            
            with error_analysis_tabs[2]:
                # Error patterns section
                if 'model_evaluation' in forecast_data and 'test_actuals' in forecast_data and 'test_predictions' in forecast_data:
                    # Get actuals and predictions
                    actuals = forecast_data['test_actuals']
                    predictions = forecast_data['test_predictions']
                    dates = None
                    
                    if 'test_dates' in forecast_data:
                        dates = forecast_data['test_dates']dates' in forecast_data:
                        dates = forecast_data['test_dates']
                    
                    if len(actuals) > 0 and len(predictions) > 0:
                        # Run error analysis with dates if available
                        error_analysis = analyze_forecast_errors(actuals, predictions, dates)
                        
                        # Display error patterns over time
                        st.subheader("Error Patterns Over Time")
                        
                        if 'error_details' in error_analysis and dates is not None:
                            error_details = error_analysis['error_details']
                            
                            # Create a line plot of errors over time
                            fig = px.line(
                                error_details,
                                x='date',
                                y='error',
                                title="Forecast Errors Over Time",
                                labels={'date': 'Date', 'error': 'Error'},
                                color_discrete_sequence=['rgba(75, 192, 192, 0.8)']
                            )
                            
                            # Add a horizontal line at zero
                            fig.add_hline(y=0, line_dash="dash", line_color="red")
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create a plot of percentage errors
                            fig = px.line(
                                error_details,
                                x='date',
                                y='percent_error',
                                title="Percentage Errors Over Time",
                                labels={'date': 'Date', 'percent_error': 'Percentage Error (%)'},
                                color_discrete_sequence=['rgba(255, 99, 132, 0.8)']
                            )
                            
                            # Add a horizontal line at zero
                            fig.add_hline(y=0, line_dash="dash", line_color="red")
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Check for autocorrelation in errors
                            autocorr = error_analysis.get('autocorrelation', None)
                            if autocorr is not None and isinstance(autocorr, np.ndarray) and len(autocorr) > 1:
                                st.subheader("Error Autocorrelation")
                                
                                # Create a bar plot of autocorrelation
                                fig = px.bar(
                                    x=list(range(len(autocorr))),
                                    y=autocorr,
                                    title="Autocorrelation of Errors",
                                labels={'x': 'Lag', 'y': 'Autocorrelation'}
                                )ion of Forecast Errors",
                                    labels={'x': 'Lag', 'y': 'Autocorrelation'},
                                    color_discrete_sequence=['rgba(54, 162, 235, 0.8)']
                                )
                                
                                # Add confidence bounds (approximate 95% confidence)
                                n = len(error_details)
                                conf_level = 1.96 / np.sqrt(n)
                                fig.add_hline(y=conf_level, line_dash="dash", line_color="red")
                                fig.add_hline(y=-conf_level, line_dash="dash", line_color="red")
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Interpret autocorrelation
                                if autocorr[1] > conf_level:
                                    st.warning(f"Significant autocorrelation detected at lag 1 ({autocorr[1]:.2f}). This suggests the model is missing important patterns in the data.")
                                else:
                                    st.success("No significant autocorrelation detected in the forecast errors.")
                        else:
                            st.info("Insufficient data to analyze error patterns over time.")
                        
                        # Show bias analysis
                        bias_analysis = error_analysis.get('bias_analysis', {})
                        if bias_analysis:
                            st.subheader("Bias Analysis")
                            
                            bias = bias_analysis.get('bias', 0)
                            bias_pct = bias_analysis.get('bias_pct', 0)
                            pos_errors = bias_analysis.get('positive_errors', 0)
                            neg_errors = bias_analysis.get('negative_errors', 0)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Average Bias", f"{bias:.2f}")
                                st.metric("Bias Percentage", f"{bias_pct:.2f}%")
                            
                            with col2:
                                total_errors = pos_errors + neg_errors
                                pos_pct = (pos_errors / total_errors * 100) if total_errors > 0 else 0
                                neg_pct = (neg_errors / total_errors * 100) if total_errors > 0 else 0
                                
                                st.metric("Overestimations", f"{pos_errors} ({pos_pct:.1f}%)")
                                st.metric("Underestimations", f"{neg_errors} ({neg_pct:.1f}%)")
                            
                            # Interpret bias
                            if abs(bias_pct) > 10:
                                direction = "overestimating" if bias < 0 else "underestimating"
                                st.warning(f"Significant bias detected! Your model is consistently {direction} by {abs(bias_pct):.1f}% on average.")
                            else:
                                st.success(f"No significant bias detected. Average bias is {bias_pct:.1f}%.")
                    else:
                        st.info("Insufficient test data for error pattern analysis.")
                else:
                    st.info("No test data available for error pattern analysis.")
                
        with forecast_tabs[6]:  # Parameter Tuning tab
            # Parameter Tuning section
            st.subheader(f"Parameter Tuning for {selected_sku}")
            
            # Show current model and parameters
            best_model = forecast_data['model']
            
            # Description
            st.markdown("""
            This section allows you to manually tune parameters for the selected SKU and model.
            Parameter tuning can significantly improve forecast accuracy by finding the optimal
            configuration for each model-SKU combination.
            """)
            
            # Create columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Show current parameters
                st.subheader("Current Model Parameters")
                
                # Check if parameters are available
                params_available = False
                if 'model_evaluation' in forecast_data and 'forecasts' in forecast_data['model_evaluation']:
                    all_forecasts = forecast_data['model_evaluation'].get('forecasts', {})
                    
                    if best_model in all_forecasts and 'params' in all_forecasts[best_model]:
                        params = all_forecasts[best_model]['params']
                        params_available = True
                        
                        # Format parameters for display
                        if isinstance(params, dict):
                            params_df = pd.DataFrame({
                                'Parameter': list(params.keys()),
                                'Value': [str(v) for v in params.values()]
                            })
                            
                            st.dataframe(params_df, use_container_width=True, hide_index=True)
                        else:
                            st.write(f"Parameters: {params}")
                
                if not params_available:
                    st.info("No parameters available for this model.")
                
                # Model selection for tuning
                st.subheader("Tune Parameters")
                models_for_tuning = ["auto_arima", "prophet", "ets", "theta"]
                
                # Check if model is available for tuning
                if best_model not in models_for_tuning:
                    st.warning(f"The model '{best_model}' does not support parameter tuning. Only ARIMA, Prophet, ETS, and Theta models can be tuned.")
                else:
                    # Button to trigger tuning
                    tuning_col1, tuning_col2 = st.columns([2, 1])
                    
                    with tuning_col1:
                        # Checkbox for cross-validation
                        use_cv = st.checkbox("Use cross-validation", value=True, 
                                           help="Use cross-validation for more robust parameter optimization")
                        
                        # Select number of trials
                        n_trials = st.slider("Number of parameter combinations to try", 
                                           min_value=10, max_value=50, value=20, step=5,
                                           help="Higher values will take longer but may find better parameters")
                    
                    # Check if tuning is in progress
                    if st.session_state.parameter_tuning_in_progress:
                        # Check optimization status
                        optimization_status = get_optimization_status(selected_sku, best_model)
                        
                        if optimization_status:
                            # Show progress information
                            st.info(f"Parameter tuning in progress. Status: {optimization_status.get('status', 'Running')}")
                            st.progress(float(optimization_status.get('progress', 0)))
                            
                            # Add a refresh button
                            if st.button("Refresh Status"):
                                st.rerun()
                    else:
                        # Button to start tuning
                        if st.button("Optimize Parameters"):
                            # Execute parameter tuning
                            st.session_state.parameter_tuning_in_progress = True
                            
                            # Get data for the selected SKU
                            sku_data = None
                            if 'train_set' in forecast_data:
                                sku_data = forecast_data['train_set']
                            elif 'sales_data' in st.session_state:
                                sales_data = st.session_state.sales_data
                                sku_data = sales_data[sales_data['sku'] == selected_sku]
                            
                            if sku_data is not None:
                                try:
                                    # Show status message
                                    st.info(f"Starting parameter optimization for {selected_sku} with model {best_model}. This may take a few minutes.")
                                    
                                    # Define progress callback
                                    def tuning_progress_callback(current_step, total_steps, message, level="info"):
                                        # This function will be called during the optimization process
                                        if 'log_messages' in st.session_state and message:
                                            timestamp = datetime.now().strftime("%H:%M:%S")
                                            st.session_state.log_messages.append({
                                                "timestamp": timestamp,
                                                "message": message,
                                                "level": level
                                            })
                                    
                                    # Start optimization in background
                                    success = optimize_parameters_async(
                                        sku=selected_sku,
                                        model_type=best_model,
                                        data=sku_data,
                                        cross_validation=use_cv,
                                        n_trials=n_trials,
                                        progress_callback=tuning_progress_callback
                                    )
                                    
                                    if success:
                                        st.success("Parameter optimization started in the background. Refresh this page to see the results.")
                                    else:
                                        st.error("Failed to start parameter optimization.")
                                except Exception as e:
                                    st.error(f"Error starting parameter optimization: {str(e)}")
                                    st.session_state.parameter_tuning_in_progress = False
                            else:
                                st.error("Cannot find data for this SKU.")
                                st.session_state.parameter_tuning_in_progress = False
            
            with col2:
                # Show the benefits of parameter tuning
                st.subheader("Benefits of Parameter Tuning")
                
                st.markdown("""
                **Why tune parameters?**
                - Improve forecast accuracy
                - Reduce forecast error (MAPE)
                - Account for unique patterns in each SKU
                - Better handle seasonality and trends
                
                **When to tune parameters?**
                - When a SKU has high forecast error
                - After significant changes in demand patterns
                - For high-value or critical inventory items
                """)
                
                # Show links to documentation
                st.subheader("Model Parameters Explained")
                
                if best_model == "auto_arima":
                    st.markdown("""
                    **ARIMA Parameters:**
                    - **p**: Autoregressive order
                    - **d**: Differencing order
                    - **q**: Moving average order
                    - **P, D, Q**: Seasonal components
                    - **m**: Seasonal period
                    """)
                elif best_model == "prophet":
                    st.markdown("""
                    **Prophet Parameters:**
                    - **changepoint_prior_scale**: Flexibility of trend
                    - **seasonality_prior_scale**: Strength of seasonality
                    - **seasonality_mode**: Additive or multiplicative
                    - **growth**: Growth trend type
                    """)
                elif best_model == "ets":
                    st.markdown("""
                    **ETS Parameters:**
                    - **error**: Error type (additive/multiplicative)
                    - **trend**: Trend type and damping
                    - **seasonal**: Seasonality type
                    - **seasonal_periods**: Length of seasonality
                    """)
                elif best_model == "theta":
                    st.markdown("""
                    **Theta Parameters:**
                    - **theta**: Theta coefficient
                    - **seasonal_period**: Length of seasonality
                    - **use_boxcox**: Data transformation
                    """)
                
                # Recovery options
                st.subheader("Having Issues?")
                
                # Button to reset tuning state
                if st.button("Reset Tuning State"):
                    st.session_state.parameter_tuning_in_progress = False
                    st.success("Parameter tuning state has been reset.")
                    time.sleep(1)
                    st.rerun()

    # Add a large, clear section break to separate the forecast data table
    st.markdown("---")
    st.markdown("## Comprehensive Forecast Data Table")
    st.info("📊 This table shows historical and forecasted values with dates as columns. The table includes actual sales data and forecasts for each SKU/model combination.")

    # Prepare comprehensive data table
    if st.session_state.advanced_forecasts:
        # Create a dataframe to store all SKUs data with reoriented structure
        all_sku_data = []

        # Get historical dates (use the first forecast as reference for dates)
        first_sku = list(st.session_state.advanced_forecasts.keys())[0]
        first_forecast = st.session_state.advanced_forecasts[first_sku]

        # Use sales data for historical dates instead of relying on train_set
        if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
            # Identify unique dates in historical data
            historical_dates = pd.to_datetime(sorted(st.session_state.sales_data['date'].unique()))

            # Show all historical data points as requested by the user
            # Format dates for column names
            historical_cols = [date.strftime('%-d %b %Y') for date in historical_dates]

            # Get forecast dates from first SKU (for column headers)
            forecast_dates = first_forecast['forecast'].index
            forecast_date_cols = [date.strftime('%-d %b %Y') for date in forecast_dates]

            # Add SKU selector for the table
            all_skus = sorted(list(st.session_state.advanced_forecasts.keys()))

            # Add multi-select for table SKUs with clearer labeling
            st.subheader("Select Data for Table View")
            table_skus = st.multiselect(
                "Choose SKUs to Include",
                options=all_skus,
                default=all_skus[:min(5, len(all_skus))],  # Default to first 5 SKUs or less
                help="Select one or more SKUs to include in the table below"
            )

            # If no SKUs selected, default to showing all (up to a reasonable limit)
            if not table_skus:
                table_skus = all_skus[:min(5, len(all_skus))]
                st.info(f"Showing first {len(table_skus)} SKUs by default. Select specific SKUs above if needed.")

            # List all selected models in the sidebar for use in the table
            selected_models_for_table = st.session_state.advanced_models
            if not selected_models_for_table:
                # If no models were selected, include default models
                selected_models_for_table = ["auto_arima", "prophet", "ets", "theta"]
                st.info(f"No models were selected in the sidebar. Using default models: {', '.join([m.upper() for m in selected_models_for_table])}")

            # Process each selected SKU
            for sku in table_skus:
                forecast_data_for_sku = st.session_state.advanced_forecasts[sku]

                # Get actual sales data for this SKU
                sku_sales = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku].copy()
                sku_sales.set_index('date', inplace=True)

                # For each selected model (from sidebar), create a row in the table
                for model in selected_models_for_table:
                    model_lower = model.lower()
                    
                    # Mark if this is the best model
                    is_best_model = (model_lower == forecast_data_for_sku['model'])

                    # Create base row info
                    row = {
                        'sku_code': sku,
                        'sku_name': sku,  # Using SKU as name, replace with actual name if available
                        'model': model.upper(),
                        'best_model': '✓' if is_best_model else ''
                    }

                    # Add historical/actual values (no prefix, just the date)
                    for date, col_name in zip(historical_dates, historical_cols):
                        # Remove "Actual:" prefix but track these columns separately for styling
                        actual_col_name = col_name  # Just use the date as column name
                        if date in sku_sales.index:
                            row[actual_col_name] = int(sku_sales.loc[date, 'quantity']) if not pd.isna(sku_sales.loc[date, 'quantity']) else 0
                        else:
                            row[actual_col_name] = 0

                    # Find forecast data for this model
                    model_forecast_series = None
                    
                    # First check if this is the best model
                    if model_lower == forecast_data_for_sku['model']:
                        model_forecast_series = forecast_data_for_sku['forecast']
                    
                    # If not the best model or we couldn't find its forecast, check in all_models_forecasts
                    if model_forecast_series is None and 'model_evaluation' in forecast_data_for_sku:
                        if 'all_models_forecasts' in forecast_data_for_sku['model_evaluation']:
                            all_models = forecast_data_for_sku['model_evaluation']['all_models_forecasts']
                            if model_lower in all_models:
                                model_forecast_series = all_models[model_lower]

                    # Add forecast values for each date
                    for date, col_name in zip(forecast_dates, forecast_date_cols):
                        forecast_value = None
                        
                        # Check if we have forecast data for this model
                        if model_forecast_series is not None:
                            # Check if this date exists in the forecast
                            if date in model_forecast_series.index:
                                forecast_value = model_forecast_series[date]
                                
                                # Make sure the value is valid
                                if pd.notna(forecast_value) and not np.isnan(forecast_value):
                                    row[col_name] = int(round(forecast_value))
                                else:
                                    # Default to 0 for NaN values
                                    row[col_name] = 0
                            else:
                                # Date not found in forecast
                                row[col_name] = 0
                        else:
                            # No forecast data for this model
                            row[col_name] = 0
                    
                    # Add the row to our data collection
                    all_sku_data.append(row)

            # Create DataFrame from all data
            if all_sku_data:
                all_sku_df = pd.DataFrame(all_sku_data)

                # Identify column groups for styling
                all_cols = all_sku_df.columns.tolist()
                info_cols = ['sku_code', 'sku_name', 'model', 'best_model']

                # Since we removed prefixes, we need a different way to identify historical vs forecast columns
                # Use the fact that historical columns come from historical_cols and forecast columns from forecast_date_cols
                actual_cols = historical_cols
                forecast_cols = forecast_date_cols

                # Add column categorization for better visualization
                st.subheader("Table Column Explanation")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🔷 SKU Info Columns**")
                    for col in info_cols:
                        st.markdown(f"- {col}")
                    
                with col2:
                    st.markdown("**🔹 Historical Data**")
                    st.markdown(f"- {len(actual_cols)} date columns showing past sales")
                    
                with col3:
                    st.markdown("**🔶 Forecast Data**")  
                    st.markdown(f"- {len(forecast_cols)} future date columns")

                # Define a function for styling the dataframe
                def highlight_data_columns(df):
                    # Create a DataFrame of styles
                    styles = pd.DataFrame('', index=df.index, columns=df.columns)

                    # Apply background colors to different column types
                    for col in info_cols:
                        styles[col] = 'background-color: #F5F5F5; font-weight: 500'  # Light gray for info columns

                    for col in actual_cols:
                        styles[col] = 'background-color: #E3F2FD'  # Lighter blue for actual values

                    for col in forecast_cols:
                        styles[col] = 'background-color: #FFF8E1'  # Lighter yellow for forecast values

                    # Highlight best model rows
                    for i, val in enumerate(df['best_model']):
                        if val == '✓':
                            for col in df.columns:
                                styles.iloc[i, df.columns.get_loc(col)] += '; font-weight: bold'

                    # Add text alignment
                    for col in all_cols:
                        if col in info_cols:
                            styles[col] += '; text-align: left'
                        else:
                            styles[col] += '; text-align: right'
                            
                    # Highlight zeros in forecast columns with a different style to make them more visible
                    for i in range(len(df)):
                        for col in forecast_cols:
                            if col in df.columns and df.iloc[i][col] == 0:
                                styles.iloc[i, df.columns.get_loc(col)] += '; color: #999999; font-style: italic'

                    return styles

                # Add column group headers using expander
                with st.expander("Understanding the Table", expanded=True):
                    st.markdown("""
                    ### Table Legend
                    - **SKU Info**: Basic product information (gray background)
                    - **Historical Data**: Past sales values (blue background)
                    - **Forecast Data**: Predicted sales values (yellow background)
                    - **✓**: Indicates the best performing model for each SKU
                    - **Zero forecasts**: Values showing as 0 may indicate the model did not generate a forecast for that date
                    """)

                # Use styling to highlight data column types with frozen columns till model name
                st.dataframe(
                    all_sku_df.style.apply(highlight_data_columns, axis=None),
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

                # Add a note about any zero values
                zero_count = (all_sku_df[forecast_cols] == 0).sum().sum()
                if zero_count > 0:
                    st.info(f"📝 There are {zero_count} forecast values showing as 0 in the table. This could be because those models didn't generate forecasts for some dates, or the generated forecasts were exactly 0.")

                # Create Excel file with nice formatting for download
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    all_sku_df.to_excel(writer, sheet_name='Forecast Data', index=False)

                    # Get the xlsxwriter workbook and worksheet objects
                    workbook = writer.book
                    worksheet = writer.sheets['Forecast Data']

                    # Add formats
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#D7E4BC',
                        'border': 1
                    })

                    info_format = workbook.add_format({
                        'bg_color': '#F5F5F5',
                        'border': 1
                    })

                    actual_format = workbook.add_format({
                        'bg_color': '#E3F2FD',
                        'border': 1,
                        'num_format': '#,##0'
                    })

                    forecast_format = workbook.add_format({
                        'bg_color': '#FFF8E1',
                        'border': 1,
                        'num_format': '#,##0'
                    })

                    # Apply formats to the header
                    for col_num, value in enumerate(all_sku_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)

                    # Set column formats based on data type
                    for i, col in enumerate(all_sku_df.columns):
                        col_idx = all_sku_df.columns.get_loc(col)
                        if col in info_cols:
                            worksheet.set_column(col_idx, col_idx, 15, info_format)
                        elif col in actual_cols:
                            worksheet.set_column(col_idx, col_idx, 12, actual_format)
                        elif col in forecast_cols:
                            worksheet.set_column(col_idx, col_idx, 12, forecast_format)

                excel_buffer.seek(0)

                # Provide download buttons for the table in multiple formats
                col1, col2 = st.columns(2)
                with col1:
                    # Excel download
                    st.download_button(
                        label="📊 Download as Excel",
                        data=excel_buffer,
                        file_name=f"sku_forecast_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel",
                        help="Download a formatted Excel spreadsheet with the table data"
                    )

                with col2:
                    # CSV download
                    csv_buffer = io.BytesIO()
                    all_sku_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)

                    st.download_button(
                        label="📄 Download as CSV",
                        data=csv_buffer,
                        file_name=f"sku_forecast_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        help="Download a CSV file with the table data"
                    )
            else:
                st.warning("No data available for the selected SKUs.")
        else:
            st.warning("No sales data available to construct the comprehensive data table. Please upload sales data first.")
    else:
        st.warning("No forecast data available. Please run a forecast first.")

else:
    # When no forecast has been run yet, but data is available
    st.header("Sales Data Analysis")

    # Show instructions
    st.info("👈 Please configure and run the forecast analysis using the sidebar to get detailed forecasts.")

    # Allow SKU selection in main area
    if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
        # Get list of SKUs from sales data
        all_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())

        # Add a prominent SKU selector
        selected_sku = st.selectbox(
            "Select a SKU to view historical data:",
            options=all_skus
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
                min_val = sku_data['quantity'].min()
                max_val = sku_data['quantity'].max()
                cv = std_val / mean_val if mean_val > 0 else 0
                
                st.metric("Minimum", f"{min_val:.2f}")
                st.metric("Maximum", f"{max_val:.2f}")
                st.metric("Coefficient of Variation", f"{cv:.2f}")
    else:
        st.error("No sales data loaded. Please upload sales data first.")

# Add secondary sales analysis section if results exist
if st.session_state.run_secondary_analysis and st.session_state.secondary_sales_results:
    st.header("Secondary Sales Analysis")
    
    # Get list of analyzed SKUs
    analyzed_skus = sorted(list(st.session_state.secondary_sales_results.keys()))
    
    if analyzed_skus:
        # Select SKU for secondary sales analysis
        selected_secondary_sku = st.selectbox(
            "Select a SKU to view secondary sales analysis",
            options=analyzed_skus,
            index=0 if st.session_state.advanced_selected_sku not in analyzed_skus else analyzed_skus.index(st.session_state.advanced_selected_sku)
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
                
                # Plot the analysis
                if 'data' in analysis_result:
                    fig = plot_secondary_sales_analysis(selected_secondary_sku, analysis_result)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    with st.expander("View Data Table"):
                        st.dataframe(analysis_result['data'], use_container_width=True)
                        
                    # Download button for data
                    csv_buffer = io.BytesIO()
                    analysis_result['data'].to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Secondary Sales Data",
                        data=csv_buffer,
                        file_name=f"secondary_sales_{selected_secondary_sku}.csv",
                        mime="text/csv",
                    )
            else:
                st.error(f"❌ Analysis failed: {analysis_result['message']}")
    else:
        st.warning("No secondary sales analysis results available")

# Create a placeholder for the progress bar
progress_placeholder = st.empty()

# Create forecast button in sidebar
if should_show_button and st.sidebar.button(
    forecast_button_text, 
    key="run_advanced_forecast_button",
    use_container_width=True
):
    # Set forecast in progress flag
    st.session_state.advanced_forecast_in_progress = True
    st.session_state.advanced_forecast_progress = 0
    st.session_state.run_advanced_forecast = True
    
# Create secondary sales analysis button in sidebar
if st.sidebar.button(
    secondary_button_text,
    key="run_secondary_analysis_button",
    use_container_width=True
):
    # Determine which SKU to analyze
    selected_sku = None
    if not run_for_all and st.session_state.advanced_selected_sku:
        selected_sku = st.session_state.advanced_selected_sku
        
    # Run the analysis
    run_secondary_sales_analysis(
        selected_sku=selected_sku,
        run_for_all=run_for_all,
        algorithm=st.session_state.secondary_sales_algorithm
    )

    # Create an enhanced progress display
    with progress_placeholder.container():
        # Create a two-column layout for the progress display
        progress_cols = st.columns([3, 1])

        with progress_cols[0]:
            # Header for progress display with animation effect
            st.markdown('<h3 style="color:#0066cc;"><span class="highlight">🔄 Advanced Forecast Generation in Progress</span></h3>', unsafe_allow_html=True)

            # Progress bar with custom styling
            progress_bar = st.progress(0)

            # Status text placeholder
            status_text = st.empty()

            # Add a progress details section
            progress_details = st.empty()

        with progress_cols[1]:
            # Add a spinning icon or other visual indicator
            st.markdown('<div class="loader"></div>', unsafe_allow_html=True)

        try:
            # Start the forecasting process
            sales_data = st.session_state.sales_data
            
            # Extract features for clustering if not already done
            if st.session_state.advanced_clusters is None:
                status_text.write("Extracting features for clustering...")
                features_df = extract_features(sales_data)
                
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
            
            # Get selected models
            selected_models = st.session_state.advanced_models
            
            # Make sure we have at least one model selected
            if not selected_models:
                selected_models = ["auto_arima", "prophet", "ets"]
            
            # Create a detailed log area
            log_area = st.empty()
            log_container = log_area.container()
            log_header = log_container.empty()
            log_content = log_container.empty()
            
            # Initialize a list to store log messages
            if 'log_messages' not in st.session_state:
                st.session_state.log_messages = []
            else:
                st.session_state.log_messages = []  # Reset log messages for new run
            
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
            add_log_message(f"Starting advanced forecast generation for {len(selected_skus) if selected_skus else 'all'} SKUs", "info")
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
                        sales_data_to_use = sales_data
                else:
                    add_log_message("Using PRIMARY sales data for forecasting...", "info")
                    sales_data_to_use = sales_data
                
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
            **Next steps:** Explore the forecast results
            """)
            
            # Sleep briefly to show completion
            time.sleep(1)
            
            # Rerun the app to show the results
            st.rerun()
        
        except Exception as e:
            # Handle errors
            st.error(f"Error generating forecasts: {str(e)}")
            st.session_state.advanced_forecast_in_progress = False
        
        finally:
            # Reset progress tracking
            st.session_state.advanced_forecast_in_progress = False
            time.sleep(1)  # Keep the completed progress visible briefly

    # Clear the progress display after completion
    progress_placeholder.empty()