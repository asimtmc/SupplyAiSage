
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import time
from datetime import datetime, timedelta
from utils.data_processor import process_sales_data
from utils.forecast_engine import generate_forecasts, extract_features, cluster_skus
from utils.advanced_forecast import (
    advanced_generate_forecasts, 
    detect_outliers, 
    clean_time_series,
    extract_advanced_features
)
from utils.visualization import plot_forecast, plot_cluster_summary, plot_model_comparison

# Set page config
st.set_page_config(
    page_title="V2 Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

# Initialize cache for forecast results
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}

# Flag to track if models are loaded (lazy loading)
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Check if data is loaded in session state
if 'sales_data' not in st.session_state or st.session_state.sales_data is None:
    st.warning("Please upload sales data on the main page first.")
    st.stop()

# Page title
st.title("V2 AI-Powered Demand Forecasting")
st.markdown("""
This module uses advanced AI algorithms to generate accurate demand forecasts for your products.
The system automatically clusters SKUs by sales patterns, selects the best forecasting model for each,
and provides confidence intervals for risk-aware planning.
""")

# Initialize session state variables for this page
if 'forecast_periods' not in st.session_state:
    st.session_state.forecast_periods = 12  # default 12 months
if 'run_forecast' not in st.session_state:
    st.session_state.run_forecast = False
if 'selected_sku' not in st.session_state:
    st.session_state.selected_sku = None
if 'selected_skus' not in st.session_state:
    st.session_state.selected_skus = []
if 'show_all_clusters' not in st.session_state:
    st.session_state.show_all_clusters = False
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'sku_options' not in st.session_state:
    st.session_state.sku_options = []
if 'forecast_in_progress' not in st.session_state:
    st.session_state.forecast_in_progress = False
if 'forecast_progress' not in st.session_state:
    st.session_state.forecast_progress = 0
if 'forecast_current_sku' not in st.session_state:
    st.session_state.forecast_current_sku = ""
if 'apply_sense_check' not in st.session_state:
    st.session_state.apply_sense_check = True

# Create sidebar for settings
with st.sidebar:
    st.header("Forecast Settings")

    # Get available SKUs from sales data (always available)
    if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
        st.subheader("SKU Selection")

        # Function to update selected SKUs
        def update_selected_skus():
            new_selection = st.session_state.sku_multiselect
            st.session_state.selected_skus = new_selection
            if new_selection:
                st.session_state.selected_sku = new_selection[0]  # Set first selected SKU as primary

        # Get available SKUs from the sales data - ensure we have a stable list
        try:
            available_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())
        except Exception as e:
            st.error(f"Error getting SKUs from sales data: {str(e)}")
            available_skus = []

        # Safety check - ensure we have valid SKUs
        available_skus = [sku for sku in available_skus if sku is not None and str(sku).strip() != '']

        # Determine default selection - with robust error handling
        default_selection = []

        # If we already have selected SKUs, use them if they're still valid
        if st.session_state.selected_skus:
            default_selection = [sku for sku in st.session_state.selected_skus if sku in available_skus]

        # If no valid selection, default to first SKU if available
        if not default_selection and available_skus:
            default_selection = [available_skus[0]]

        # Info about available SKUs
        if available_skus:
            st.info(f"📊 {len(available_skus)} SKUs available for analysis")
        else:
            st.warning("No SKUs found in sales data. Please check your data upload.")

        # Create the multiselect widget with on_change callback
        sku_selection = st.multiselect(
            "Select SKUs to Analyze",
            options=available_skus,
            default=default_selection,
            on_change=update_selected_skus,
            key="sku_multiselect",
            help="Select one or more SKUs to analyze or forecast"
        )

        # Ensure selected_skus is always updated
        st.session_state.selected_skus = sku_selection

        # Update primary selected SKU if we have a selection
        if sku_selection:
            st.session_state.selected_sku = sku_selection[0]

        # Show current selection information
        if sku_selection:
            st.success(f"✅ Selected {len(sku_selection)} SKU(s) for analysis")
        else:
            st.warning("⚠️ Please select at least one SKU for analysis")

    # Forecast horizon slider
    forecast_periods = st.slider(
        "Forecast Periods (Months)",
        min_value=1,
        max_value=24,
        value=st.session_state.forecast_periods,
        step=1
    )
    st.session_state.forecast_periods = forecast_periods

    # Number of clusters
    num_clusters = st.slider(
        "Number of SKU Clusters",
        min_value=2,
        max_value=10,
        value=5,
        step=1
    )

    # Model evaluation and selection
    st.subheader("Model Selection")

    # Option to evaluate models on test data
    evaluate_models_flag = st.checkbox("Evaluate models on test data", value=True,
                                     help="Split data into training and test sets to evaluate model performance before forecasting")

    st.write("Select forecasting models to evaluate:")
    models_to_evaluate = []

    # Basic models
    if st.checkbox("Moving Average", value=True):
        models_to_evaluate.append("moving_average")

    # Advanced models from Advanced Forecasting 
    if st.checkbox("Auto ARIMA", value=True):
        models_to_evaluate.append("auto_arima")

    if st.checkbox("SARIMAX (Seasonal)", value=True):
        models_to_evaluate.append("sarima")

    if st.checkbox("Prophet", value=True):
        models_to_evaluate.append("prophet")

    if st.checkbox("ETS (Exponential Smoothing)", value=True):
        models_to_evaluate.append("ets")

    if st.checkbox("LSTM Neural Network", value=True):
        models_to_evaluate.append("lstm")

    if st.checkbox("Theta Method", value=True):
        models_to_evaluate.append("theta")

    if st.checkbox("Holt-Winters", value=True):
        models_to_evaluate.append("holtwinters")
        
    if st.checkbox("Decomposition", value=True, help="Uses time series decomposition to separate trend, seasonal, and residual components"):
        models_to_evaluate.append("decomposition")
        
    if st.checkbox("Ensemble Model", value=True, help="Combines multiple forecasting models for improved accuracy"):
        models_to_evaluate.append("ensemble")

    # Store selected models for visualization
    st.session_state.selected_models = models_to_evaluate

    # Add option to forecast all or selected SKUs
    st.subheader("Forecast Scope")

    # Determine which SKUs to forecast
    forecast_scope = st.radio(
        "Choose SKUs to analyze",
        ["Selected SKUs Only", "All SKUs"],
        index=1,  # Default to All SKUs
        horizontal=True
    )

    # Apply Sense Check
    apply_sense_check = st.toggle(
        "Human-Like Sense Check",
        value=st.session_state.apply_sense_check,
        key="sidebar_apply_sense_check",
        help="Apply business logic and pattern recognition to ensure realistic forecasts"
    )
    st.session_state.apply_sense_check = apply_sense_check

    # Get selected SKUs from session state
    selected_skus_to_forecast = []
    if 'selected_skus' in st.session_state and st.session_state.selected_skus:
        selected_skus_to_forecast = st.session_state.selected_skus

    # Progress tracking callback function
    def forecast_progress_callback(current_index, current_sku, total_skus, message=None, level="info"):
        # Update progress information in session state
        progress = min(float(current_index) / total_skus, 1.0)
        st.session_state.forecast_progress = progress
        st.session_state.forecast_current_sku = current_sku

        # Extract current model from message if available with improved detection
        if message:
            # Try multiple patterns to extract model information
            if "model" in message.lower():
                # Pattern 1: "...model X..."
                model_parts = message.lower().split("model")
                if len(model_parts) > 1:
                    model_text = model_parts[1].strip()
                    model_words = model_text.split()
                    if model_words:
                        model_name = model_words[0].strip(": ").upper()
                        st.session_state.current_model = model_name

            # Additional patterns for model detection
            model_keywords = ["auto_arima", "prophet", "sarima", "ets", "lstm", "ensemble", "moving_average", "theta", "holtwinters", "decomposition"]
            for keyword in model_keywords:
                if keyword in message.lower():
                    st.session_state.current_model = keyword.upper()
                    break

            # Check if we're in the evaluation phase
            if "evaluating" in message.lower() or "training" in message.lower():
                evaluation_phase = True
                if not hasattr(st.session_state, 'current_model') or not st.session_state.current_model:
                    st.session_state.current_model = "EVALUATING"

    # Run forecast button
    forecast_button_text = "Run Forecast Analysis"
    if forecast_scope == "Selected SKUs Only" and selected_skus_to_forecast:
        forecast_button_text = f"Run Forecast for {len(selected_skus_to_forecast)} Selected SKUs"
    elif forecast_scope == "Selected SKUs Only" and not selected_skus_to_forecast:
        st.warning("Please select at least one SKU to analyze.")

    # Only show the button if we have valid SKUs to forecast
    should_show_button = not (forecast_scope == "Selected SKUs Only" and not selected_skus_to_forecast)

    # Create a placeholder for the progress bar
    progress_placeholder = st.empty()

    # Create a run button with a unique key
    if should_show_button:
        # Generate a cache key based on selected parameters
        cache_key = f"{forecast_scope}_{len(selected_skus_to_forecast)}_{forecast_periods}_{num_clusters}_{'-'.join(models_to_evaluate)}"

        # Show different button text if cached results are available
        button_text = forecast_button_text

        run_forecast_clicked = st.button(
            button_text, 
            key="run_forecast_button",
            use_container_width=True
        )

        if run_forecast_clicked:
            # Set forecast in progress flag
            st.session_state.forecast_in_progress = True
            st.session_state.forecast_progress = 0
            st.session_state.run_forecast = True

            # Create an enhanced progress display
            with progress_placeholder.container():
                # Create a two-column layout for the progress display
                progress_cols = st.columns([3, 1])

            with progress_cols[0]:
                # Header for progress display with animation effect
                st.markdown('<h3 style="color:#0066cc;"><span class="highlight">🔄 Forecast Generation in Progress</span></h3>', unsafe_allow_html=True)

                # Progress bar with custom styling
                progress_bar = st.progress(0)

                # Status text placeholder
                status_text = st.empty()

                # Add a progress details section
                progress_details = st.empty()

            with progress_cols[1]:
                # Add an animated spinner for current processing step
                spinner_placeholder = st.empty()
                # Status indicator for phases
                phase_indicator = st.empty()

            try:
                # Create a log area for detailed process tracking
                log_area = st.expander("View Processing Log", expanded=True)
                with log_area:
                    # Create a log container
                    log_container = st.empty()
                    log_messages = []

                    # Function to add log messages
                    def add_log_message(message, level="info"):
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        log_messages.append({"timestamp": timestamp, "message": message, "level": level})

                        # Format log messages with appropriate styling
                        log_html = '<div style="height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.8em; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'

                        for log in log_messages[-100:]:  # Show last 100 messages
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
                        log_container.markdown(log_html, unsafe_allow_html=True)

                # Enhanced progress callback function that updates logs
                def enhanced_forecast_callback(current_index, current_sku, total_skus, message=None, level="info"):
                    # Update progress information in session state
                    progress = min(float(current_index) / total_skus, 1.0)
                    st.session_state.forecast_progress = progress
                    st.session_state.forecast_current_sku = current_sku

                    # Extract current model from message if available
                    if message and "model" in message.lower():
                        model_parts = message.split("model")
                        if len(model_parts) > 1:
                            model_name = model_parts[1].strip().split()[0].strip(":")
                            st.session_state.current_model = model_name

                    # Add to both local log and main process log
                    timestamp = datetime.now().strftime("%H:%M:%S")

                    # Create formatted log message
                    log_msg = f"[SKU: {current_sku}] {message}" if message else f"Processing SKU: {current_sku} ({current_index+1}/{total_skus})"

                    # Add to local log
                    add_log_message(log_msg, level)

                    # Add to main process log
                    if 'log_messages' not in st.session_state:
                        st.session_state.log_messages = []

                    st.session_state.log_messages.append({
                        "timestamp": timestamp,
                        "message": log_msg,
                        "level": level
                    })

                # Phase 1: Extract time series features for clustering
                add_log_message("Starting Phase 1: Time Series Feature Extraction", "info")
                with spinner_placeholder:
                    with st.spinner("Extracting features..."):
                        phase_indicator.markdown("**Phase 1/3**")
                        status_text.markdown("### Step 1: Time Series Feature Extraction")
                        progress_details.info("Analyzing sales patterns and extracting key time series features for every SKU...")
                        add_log_message("Extracting time series features from historical sales data...", "info")
                        features_df = extract_features(st.session_state.sales_data)
                        add_log_message(f"Feature extraction complete. Processed {len(features_df)} SKUs.", "success")
                        progress_bar.progress(10)
                        time.sleep(0.5)  # Add a small pause for visual effect

                # Phase 2: Cluster SKUs
                add_log_message("Starting Phase 2: SKU Clustering", "info")
                with spinner_placeholder:
                    with st.spinner("Clustering SKUs..."):
                        phase_indicator.markdown("**Phase 2/3**")
                        status_text.markdown("### Step 2: SKU Clustering")
                        progress_details.info("Grouping similar SKUs based on their sales patterns to optimize forecast model selection...")
                        add_log_message(f"Clustering SKUs into {num_clusters} groups based on sales patterns...", "info")
                        st.session_state.clusters = cluster_skus(features_df, n_clusters=num_clusters)
                        add_log_message("Clustering complete. SKUs have been grouped by similar patterns.", "success")
                        progress_bar.progress(20)
                        time.sleep(0.5)  # Add a small pause for visual effect

                # Phase 3: Generate forecasts
                add_log_message("Starting Phase 3: Forecast Generation", "info")
                phase_indicator.markdown("**Phase 3/3**")
                status_text.markdown("### Step 3: Forecast Generation")

                # Determine which SKUs to forecast
                skus_to_forecast = None
                if forecast_scope == "Selected SKUs Only" and selected_skus_to_forecast:
                    skus_to_forecast = selected_skus_to_forecast
                    progress_details.info(f"Generating forecasts for {len(skus_to_forecast)} selected SKUs using {len(models_to_evaluate)} different forecasting models...")
                    add_log_message(f"Preparing to generate forecasts for {len(skus_to_forecast)} selected SKUs", "info")
                else:
                    total_skus = len(features_df)
                    progress_details.info(f"Generating forecasts for all {total_skus} SKUs using {len(models_to_evaluate)} different forecasting models...")
                    add_log_message(f"Preparing to generate forecasts for all {total_skus} SKUs", "info")

                # Log models being used
                add_log_message(f"Models to evaluate: {', '.join(models_to_evaluate)}", "info")
                add_log_message(f"Forecast periods: {st.session_state.forecast_periods}", "info")
                add_log_message(f"Human-like sense check: {'Enabled' if st.session_state.apply_sense_check else 'Disabled'}", "info")
                add_log_message("Beginning forecast model training and evaluation...", "info")

                # Generate forecasts with model evaluation and progress tracking
                with spinner_placeholder:
                    with st.spinner("Building forecast models..."):
                        # Use advanced_generate_forecasts from advanced_forecast.py
                        st.session_state.forecasts = advanced_generate_forecasts(
                            sales_data=st.session_state.sales_data,
                            cluster_info=st.session_state.clusters,
                            forecast_periods=st.session_state.forecast_periods,
                            auto_select=True,
                            models_to_evaluate=models_to_evaluate,
                            selected_skus=skus_to_forecast,
                            progress_callback=enhanced_forecast_callback,
                            hyperparameter_tuning=False,
                            apply_sense_check=st.session_state.apply_sense_check,
                            use_param_cache=True
                        )

                # Update progress based on callback data with improved visuals
                # Create an animated progress update with model-wise tracking
                add_log_message("Finalizing forecast calculations...", "info")

                # Create a place to show detailed model progress
                model_progress_container = st.empty()
                model_details = {}

                # Progress tracking variables
                last_progress = 0
                last_log_time = time.time()

                # Main progress loop with timeout protection
                start_time = time.time()
                max_wait_time = 300  # Maximum 5 minutes wait to prevent infinite loop

                while st.session_state.forecast_progress < 1 and st.session_state.forecast_current_sku and (time.time() - start_time < max_wait_time):
                    current_progress = 20 + int(st.session_state.forecast_progress * 80)
                    if current_progress > last_progress:
                        progress_bar.progress(current_progress/100)
                        last_progress = current_progress

                    # Get current model and SKU info
                    current_sku = st.session_state.forecast_current_sku
                    current_model = getattr(st.session_state, 'current_model', 'Unknown')

                    # Update model tracking if we have new information
                    if current_sku and current_model != 'Unknown':
                        if current_sku not in model_details:
                            model_details[current_sku] = set()
                        model_details[current_sku].add(current_model)

                    # Show detailed model progress every second
                    if time.time() - last_log_time > 1:
                        # Create a formatted display of model progress
                        model_progress_text = ""
                        for sku, models in model_details.items():
                            model_list = ", ".join(models)
                            model_progress_text += f"**{sku}**: {model_list}\n\n"

                        # Update the model progress display
                        if model_progress_text:
                            model_progress_container.markdown(f"### Model Progress by SKU:\n{model_progress_text}")

                        last_log_time = time.time()

                        # Log to console for debugging
                        print(f"Progress: {current_progress}%, Current SKU: {current_sku}, Current Model: {current_model}")

                    with spinner_placeholder:
                        with st.spinner(f"Processing {current_sku}..."):
                            # Update progress display with more dynamic information
                            status_text.markdown(f"### Processing: **{current_sku}**")
                            progress_percentage = int(st.session_state.forecast_progress * 100)
                            progress_details.info(f"Completed: **{progress_percentage}%** | Current SKU: **{current_sku}** | Current Model: **{current_model}**")
                            phase_indicator.markdown(f"**Processing {progress_percentage}% complete**")
                            time.sleep(0.1)

                # Check if loop exited due to timeout
                if time.time() - start_time >= max_wait_time:
                    add_log_message("Forecast calculation taking longer than expected. Finalizing available results.", "warning")
                    # Force progress to complete
                    st.session_state.forecast_progress = 1

                # Complete the progress bar with success animation
                progress_bar.progress(100)
                spinner_placeholder.success("✅ Complete")
                phase_indicator.markdown("**Finished!**")
                status_text.markdown("### ✨ Forecast Generation Completed Successfully!")
                progress_details.success("All forecasts have been generated and are ready to explore!")

                # Add final log messages
                add_log_message("Forecast generation complete!", "success")

                # If forecasts were generated, set default selected SKU
                if st.session_state.forecasts:
                    sku_list = sorted(list(st.session_state.forecasts.keys()))
                    st.session_state.sku_options = sku_list
                    if sku_list and not st.session_state.selected_sku in sku_list:
                        st.session_state.selected_sku = sku_list[0]

                    st.session_state.models_loaded = True

                num_skus = len(st.session_state.forecasts)
                if num_skus > 0:
                    add_log_message(f"Successfully generated forecasts for {num_skus} SKUs!", "success")
                    st.success(f"Successfully generated forecasts for {num_skus} SKUs!")
                else:
                    add_log_message("No forecasts were generated. Please check your data and selected SKUs.", "error")
                    st.error("No forecasts were generated. Please check your data and selected SKUs.")

            except Exception as e:
                error_message = f"Error during forecast generation: {str(e)}"
                add_log_message(error_message, "error")
                st.error(error_message)

            finally:
                # Reset progress tracking
                st.session_state.forecast_in_progress = False
                add_log_message("Forecast process finished.", "info")
                time.sleep(1)  # Keep the completed progress visible briefly

            # Clear the progress display after completion
            progress_placeholder.empty()

# Main content
if st.session_state.run_forecast and 'forecasts' in st.session_state and st.session_state.forecasts:
    # Show cluster analysis
    st.header("SKU Cluster Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Display cluster summary chart
        cluster_fig = plot_cluster_summary(st.session_state.clusters)
        st.plotly_chart(cluster_fig, use_container_width=True)

    with col2:
        # Show cluster details
        st.subheader("Cluster Characteristics")

        if 'clusters' in st.session_state and st.session_state.clusters is not None:
            cluster_groups = st.session_state.clusters.groupby('cluster_name').size().reset_index()
            cluster_groups.columns = ['Cluster', 'Count']

            # Calculate percentage
            total_skus = cluster_groups['Count'].sum()
            cluster_groups['Percentage'] = (cluster_groups['Count'] / total_skus * 100).round(1)
            cluster_groups['Percentage'] = cluster_groups['Percentage'].astype(str) + '%'

            st.dataframe(cluster_groups, use_container_width=True)

            # Option to show all SKUs and their clusters
            show_all = st.checkbox("Show All SKUs and Their Clusters", value=st.session_state.show_all_clusters)
            st.session_state.show_all_clusters = show_all

            if show_all:
                sku_clusters = st.session_state.clusters[['sku', 'cluster_name']].sort_values('cluster_name')
                st.dataframe(sku_clusters, use_container_width=True)

    # Show SKU-wise Model Selection Table
    st.header("SKU-wise Model Selection")

    # Create a dataframe with SKU and model information
    model_selection_data = []

    for sku, forecast_data in st.session_state.forecasts.items():
        # Get the model name according to advanced_forecast format
        selected_model = forecast_data.get('selected_model', forecast_data.get('model', 'Unknown'))

        model_info = {
            'SKU': sku,
            'Selected Model': selected_model.upper(),
            'Cluster': forecast_data.get('cluster_name', 'Unknown')
        }

        # Add model evaluation metrics if available
        if 'metrics' in forecast_data:
            metrics = forecast_data['metrics']
            if 'rmse' in metrics:
                model_info['RMSE'] = round(metrics['rmse'], 2)
            if 'mape' in metrics:
                model_info['MAPE (%)'] = round(metrics['mape'], 2) if not np.isnan(metrics['mape']) else None

        model_selection_data.append(model_info)

    # Create and display the dataframe
    model_selection_df = pd.DataFrame(model_selection_data)

    # Set default filter values
    selected_model_filter = 'All'
    selected_cluster_filter = 'All'

    # Add table filters
    col1, col2 = st.columns(2)

    with col1:
        # Filter by model type
        if 'Selected Model' in model_selection_df.columns:
            unique_models = ['All'] + sorted(model_selection_df['Selected Model'].unique().tolist())
            selected_model_filter = st.selectbox("Filter by Model Type", options=unique_models)

    with col2:
        # Filter by cluster
        if 'Cluster' in model_selection_df.columns:
            unique_clusters = ['All'] + sorted(model_selection_df['Cluster'].unique().tolist())
            selected_cluster_filter = st.selectbox("Filter by Cluster", options=unique_clusters)

    # Apply filters
    filtered_df = model_selection_df.copy()

    if selected_model_filter != 'All':
        filtered_df = filtered_df[filtered_df['Selected Model'] == selected_model_filter]

    if selected_cluster_filter != 'All':
        filtered_df = filtered_df[filtered_df['Cluster'] == selected_cluster_filter]

    # Display the filtered table with pagination to improve performance
    total_rows = len(filtered_df)
    rows_per_page = 20  # Limit rows shown per page

    # Add pagination controls
    if total_rows > rows_per_page:
        page_col1, page_col2 = st.columns([1, 3])
        with page_col1:
            current_page = st.selectbox(
                "Page",
                options=list(range(1, (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0) + 1)),
                key="forecast_table_page"
            )
        with page_col2:
            st.info(f"Showing page {current_page} of {filtered_df.shape[0]} total rows")

        # Calculate start and end indices for current page
        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)

        # Display only the current page of data
        st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True)
    else:
        # If we have fewer rows than a page, just display all
        st.dataframe(filtered_df, use_container_width=True)

    # Add a main process log console at the bottom of the page
    st.header("Forecast Process Log")

    # Create a container for the main process log
    process_log_container = st.container()

    with process_log_container:
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []

        # Display the log messages
        if st.session_state.log_messages:
            log_html = '<div style="height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.8em; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'

            for log in st.session_state.log_messages[-100:]:  # Show last 100 messages
                timestamp = log.get("timestamp", "")
                message = log.get("message", "")
                level = log.get("level", "info")

                if level == "info":
                    color = "black"
                elif level == "warning":
                    color = "orange"
                elif level == "error":
                    color = "red"
                elif level == "success":
                    color = "green"
                else:
                    color = "blue"

                log_html += f'<div style="margin-bottom: 3px;"><span style="color: gray;">[{timestamp}]</span> <span style="color: {color};">{message}</span></div>'

            log_html += '</div>'

            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.info("No process logs available yet. Run a forecast to see detailed logs.")

    # Forecast explorer
    st.header("Forecast Explorer")

    # Create a SKU selection area
    # Allow user to select a SKU to view detailed forecast
    sku_list = list(st.session_state.forecasts.keys())

    # Safely get an index for the selected SKU
    if st.session_state.selected_sku is None or st.session_state.selected_sku not in sku_list:
        default_index = 0
    else:
        try:
            default_index = sku_list.index(st.session_state.selected_sku)
        except (ValueError, IndexError):
            default_index = 0

    selected_sku = st.selectbox(
        "Select a SKU to view forecast details",
        options=sku_list,
        index=default_index,
        key="forecast_sku_selector"
    )
    st.session_state.selected_sku = selected_sku

    # Show forecast details for selected SKU
    if selected_sku:
        forecast_data = st.session_state.forecasts[selected_sku]

        # Tab section for forecast views
        forecast_tabs = st.tabs(["Forecast Chart", "Model Comparison", "Forecast Metrics"])

        with forecast_tabs[0]:
            # Get available models for this SKU
            available_models = []
            selected_models_for_viz = []

            # Check for model_comparison in the forecast data
            if 'model_comparison' in forecast_data:
                available_models = list(forecast_data['model_comparison'].keys())

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
                # Custom model selection (only show if not showing all models)
                if not show_all_models and available_models:
                    # Get model options capitalized
                    model_options = [model.upper() for model in available_models]

                    # Ensure best model is included in the options
                    default_model = forecast_data.get('selected_model', '').upper()
                    if default_model not in model_options and default_model.lower() in available_models:
                        model_options.append(default_model)

                    # Get the selected models from sidebar
                    selected_sidebar_models = []
                    for m in st.session_state.selected_models:
                        model_upper = m.upper()
                        if model_upper in model_options or m.lower() in available_models:
                            selected_sidebar_models.append(model_upper if model_upper in model_options else m.upper())

                    # If none of the sidebar selected models are available, default to the best model
                    if not selected_sidebar_models:
                        selected_sidebar_models = [default_model] if default_model in model_options else []

                    # Create multiselect for custom model selection
                    custom_models = st.multiselect(
                        "Select Models to Display",
                        options=model_options,
                        default=selected_sidebar_models,
                        help="Select one or more models to display on chart"
                    )
                    # Convert back to lowercase for consistency
                    custom_models_lower = [model.lower() for model in custom_models]
                else:
                    custom_models_lower = []

            # Determine which models to display based on selections
            if show_all_models and available_models:
                # Use ALL models that were selected in the sidebar AND are available in the forecast data
                selected_models_for_viz = []
                for model in st.session_state.selected_models:
                    model_lower = model.lower()
                    # Check if this model exists in the forecast data
                    if model_lower in available_models:
                        selected_models_for_viz.append(model_lower)
            elif custom_models_lower:
                # Use custom selection from multiselect
                selected_models_for_viz = custom_models_lower
            else:
                # Default to the primary model if nothing is explicitly selected
                selected_models_for_viz = [forecast_data.get('selected_model', '')]

            # Ensure we have at least one model in the list
            if not selected_models_for_viz and available_models:
                selected_models_for_viz = [available_models[0]]

            # Prepare visualization data in the format expected by plot_forecast
            visualization_data = {
                'sku': selected_sku,
                'historical_data': forecast_data.get('train_set', pd.DataFrame()),
                'forecast_data': pd.DataFrame({
                    'date': pd.to_datetime(forecast_data['forecast'].index),
                    'forecast': forecast_data['forecast'].values,
                    'lower_bound': forecast_data.get('lower_bound', forecast_data['forecast']).values,
                    'upper_bound': forecast_data.get('upper_bound', forecast_data['forecast']).values
                })
            }

            # Get selected_models_for_viz if specified
            forecast_data['selected_models_for_viz'] = selected_models_for_viz

            # Display forecast chart with selected models
            if 'show_anomalies' not in locals():  # Defining it if not already defined
                show_anomalies = True

            # Get confidence level from session state or default to 80%
            confidence_interval = 0.8  # Default value

            # Use plot_forecast from visualization.py with prepared data
            # The function requires both sales_data and forecast_data parameters
            forecast_fig = plot_forecast(
                sales_data=st.session_state.sales_data,
                forecast_data=forecast_data,
                sku=selected_sku,
                selected_models=selected_models_for_viz,
                show_anomalies=show_anomalies,
                confidence_interval=confidence_interval
            )

            st.plotly_chart(forecast_fig, use_container_width=True)

            # Add a note about model selection
            if selected_models_for_viz:
                st.info(f"Displaying forecasts for models: {', '.join([m.upper() for m in selected_models_for_viz])}")

            # Show training/test split information if available
            if 'train_set' in forecast_data and 'test_set' in forecast_data:
                train_count = len(forecast_data['train_set'])
                test_count = len(forecast_data['test_set'])
                total_points = train_count + test_count
                train_pct = int((train_count / total_points) * 100)
                test_pct = int((test_count / total_points) * 100)

                st.caption(f"Data split: {train_count} training points ({train_pct}%) and {test_count} test points ({test_pct}%)")

            # Add Forecast Details as expandable section (collapsed by default)
            with st.expander("Forecast Details", expanded=False):
                if selected_sku and selected_sku in st.session_state.forecasts:
                    # Basic metrics at the top
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**SKU:** {selected_sku}")

                    with col2:
                        st.markdown(f"**Cluster:** {forecast_data.get('cluster_name', 'Unknown')}")

                    with col3:
                        model_name = forecast_data.get('selected_model', forecast_data.get('model', 'Unknown')).upper()
                        st.markdown(f"**Model Used:** {model_name}")

                    # Show accuracy metric if available
                    if 'metrics' in forecast_data and 'mape' in forecast_data['metrics']:
                        mape = forecast_data['metrics']['mape']
                        if not np.isnan(mape):
                            st.metric("Forecast Accuracy", f"{(100-mape):.1f}%", help="Based on test data evaluation")

                    # Forecast confidence
                    confidence_color = "green" if model_name.lower() != 'moving_average' else "orange"
                    confidence_text = "High" if model_name.lower() != 'moving_average' else "Medium"
                    st.markdown(f"**Forecast Confidence:** <span style='color:{confidence_color}'>{confidence_text}</span>", unsafe_allow_html=True)

                    # Create basic forecast table
                    forecast_table = pd.DataFrame({
                        'Date': forecast_data['forecast'].index,
                        'Forecast': forecast_data['forecast'].values.round(0).astype(int)
                    })

                    # Add model comparisons if available
                    if 'model_comparison' in forecast_data:
                        model_forecasts = forecast_data['model_comparison']

                        # Show models explicitly selected by the user
                        if show_all_models:
                            # Use ALL models from sidebar that are available in the forecasts
                            models_to_display = []
                            for model in st.session_state.selected_models:
                                model_lower = model.lower()
                                if model_lower in model_forecasts:
                                    models_to_display.append(model_lower)
                        elif custom_models_lower:
                            # Use custom selection from multiselect
                            models_to_display = [m for m in custom_models_lower if m in model_forecasts]
                        else:
                            # If no models selected, use the best model
                            models_to_display = [forecast_data.get('selected_model', '')]

                        # Add each selected model's forecast as a column
                        for model in models_to_display:
                            if model in model_forecasts:
                                model_forecast = model_forecasts[model]

                                # Add each selected model's forecast as a column
                                forecast_values = []
                                for date in forecast_table['Date']:
                                    try:
                                        date_obj = pd.to_datetime(date)
                                        # Check if the date is in the model forecast
                                        if isinstance(model_forecast, pd.Series) and date_obj in model_forecast.index:
                                            forecast_val = model_forecast[date_obj]
                                        elif isinstance(model_forecast, pd.DataFrame) and date_obj in model_forecast.index:
                                            forecast_val = model_forecast.loc[date_obj]
                                        else:
                                            # If not found, use NaN
                                            forecast_val = np.nan

                                        # Handle NaN values before conversion to int
                                        if pd.isna(forecast_val) or np.isnan(forecast_val):
                                            forecast_values.append(0)
                                        else:
                                            # Double-check to ensure we're not trying to convert NaN
                                            try:
                                                forecast_values.append(int(round(forecast_val)))
                                            except (ValueError, TypeError):
                                                # If conversion fails for any reason, use 0
                                                forecast_values.append(0)
                                    except Exception as e:
                                        # Catch any unexpected errors
                                        print(f"Error processing forecast date {date} for model {model}: {str(e)}")
                                        forecast_values.append(0)

                                # Ensure there are no NaN values in the table
                                forecast_table[f'{model.upper()} Forecast'] = forecast_values
                                forecast_table.fillna(0, inplace=True)

                    # Format the date column to be more readable
                    forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')

                    # Display the enhanced table with styling
                    st.subheader("Forecast Data Table")
                    st.dataframe(
                        forecast_table.style.highlight_max(subset=['Forecast'], color='#d6eaf8')
                                         .highlight_min(subset=['Forecast'], color='#fadbd8'),
                        use_container_width=True,
                        height=min(35 * (len(forecast_table) + 1), 400)  # Dynamically size table height with scrolling
                    )

        with forecast_tabs[1]:
            # Model comparison visualization
            if 'model_comparison' in forecast_data:
                # Visual comparison of models
                st.subheader("Model Performance Comparison")

                # Use plot_model_comparison for advanced model comparison
                model_comparison_fig = plot_model_comparison(
                    forecast_data['model_comparison'], 
                    forecast_data.get('selected_model', ''),
                    selected_models_for_viz
                )

                if model_comparison_fig:
                    st.plotly_chart(model_comparison_fig, use_container_width=True)
                else:
                    st.info("Could not generate model comparison visualization with the available data.")

                # Add explanation of the evaluation process
                st.info("The system evaluates each selected forecasting model on historical test data. " +
                        "The model with the lowest error metrics is automatically selected as the best model.")

                # Show training and test data details if available
                if 'train_set' in forecast_data and 'test_set' in forecast_data:
                    st.subheader("Training and Test Data")

                    col1, col2 = st.columns(2)

                    with col1:
                        train_start = forecast_data['train_set'].index.min().strftime('%Y-%m-%d')
                        train_end = forecast_data['train_set'].index.max().strftime('%Y-%m-%d')
                        train_count = len(forecast_data['train_set'])

                        st.metric("Training Period", f"{train_start} to {train_end}")
                        st.metric("Training Data Points", train_count)

                    with col2:
                        test_start = forecast_data['test_set'].index.min().strftime('%Y-%m-%d')
                        test_end = forecast_data['test_set'].index.max().strftime('%Y-%m-%d')
                        test_count = len(forecast_data['test_set'])

                        st.metric("Test Period", f"{test_start} to {test_end}")
                        st.metric("Test Data Points", test_count)

        with forecast_tabs[2]:
            # Detailed metrics and accuracy information
            if 'metrics' in forecast_data:
                st.subheader("Model Evaluation Results")

                # Create table of model evaluation metrics
                metrics_data = []

                # Handle different structures of metrics data
                if 'model_comparison' in forecast_data:
                    for model_name, model_forecast in forecast_data['model_comparison'].items():
                        # Extract or calculate metrics for each model
                        if isinstance(model_forecast, dict) and 'metrics' in model_forecast:
                            model_metrics = model_forecast['metrics']
                        else:
                            # If metrics not directly available, use the main metrics
                            model_metrics = forecast_data['metrics']

                        metrics_data.append({
                            'Model': model_name.upper(),
                            'RMSE': round(model_metrics.get('rmse', 0), 2),
                            'MAPE (%)': round(model_metrics.get('mape', 0), 2) if not np.isnan(model_metrics.get('mape', 0)) else "N/A",
                            'MAE': round(model_metrics.get('mae', 0), 2),
                            'Best Model': '✓' if model_name == forecast_data.get('selected_model', '') else ''
                        })
                else:
                    # Just use the primary metrics
                    metrics = forecast_data['metrics']
                    metrics_data.append({
                        'Model': forecast_data.get('selected_model', 'Unknown').upper(),
                        'RMSE': round(metrics.get('rmse', 0), 2),
                        'MAPE (%)': round(metrics.get('mape', 0), 2) if not np.isnan(metrics.get('mape', 0)) else "N/A",
                        'MAE': round(metrics.get('mae', 0), 2),
                        'Best Model': '✓'
                    })

                # Create DataFrame and display
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df.sort_values('RMSE'), use_container_width=True, height=350)

                # Add explanation about metrics
                st.markdown("""
                **Metrics Explanation:**
                * **RMSE (Root Mean Square Error)**: Measures the square root of the average squared difference between predicted and actual values. Lower values are better.
                * **MAPE (Mean Absolute Percentage Error)**: Measures the average percentage difference between predicted and actual values. Lower values are better.
                * **MAE (Mean Absolute Error)**: Measures the average absolute difference between predicted and actual values. Lower values are better.
                """)

                # Add explanation about model selection
                st.info("The system selected the model with the lowest RMSE as the best model for forecasting this SKU.")

    # Forecast export
    st.header("Export Forecasts")

    # Prepare forecast data for export
    if st.button("Prepare Forecast Export", key="prepare_forecast_export"):
        with st.spinner("Preparing forecast data..."):
            # Create a DataFrame with all forecasts
            export_data = []

            for sku, forecast_data in st.session_state.forecasts.items():
                for date, value in forecast_data['forecast'].items():
                    export_data.append({
                        'sku': sku,
                        'date': date,
                        'forecast': round(value),
                        'model': forecast_data.get('selected_model', 'Unknown'),
                        'cluster': forecast_data.get('cluster_name', 'Unknown')
                    })

            export_df = pd.DataFrame(export_data)

            # Display export preview
            st.subheader("Export Preview")
            st.dataframe(export_df.head(10), use_container_width=True)

            # Convert to Excel for download
            excel_buffer = io.BytesIO()
            export_df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
            excel_buffer.seek(0)

            # Create download button
            st.download_button(
                label="Download Forecast as Excel",
                data=excel_buffer,
                file_name=f"forecasts_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )

    # Add a large, clear section break to separate the forecast data table
    st.markdown("---")
    st.markdown("## Comprehensive Forecast Data Table")
    st.info("📊 This table shows historical and forecasted values with dates as columns. The table includes actual sales data and forecasts for each SKU/model combination.")

    # Prepare comprehensive data table
    if st.session_state.forecasts:
        # Create a dataframe to store all SKUs data with reoriented structure
        all_sku_data = []

        # Get historical dates (use the first forecast as reference for dates)
        first_sku = list(st.session_state.forecasts.keys())[0]
        first_forecast = st.session_state.forecasts[first_sku]

        # Use sales data for historical dates
        if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
            # Identify unique dates in historical data
            historical_dates = pd.to_datetime(sorted(st.session_state.sales_data['date'].unique()))

            # Format dates for column names
            historical_cols = [date.strftime('%-d %b %Y') for date in historical_dates]

            # Get forecast dates from first SKU (for column headers)
            forecast_dates = first_forecast['forecast'].index
            forecast_date_cols = [date.strftime('%-d %b %Y') for date in forecast_dates]

            # Add SKU selector for the table
            all_skus = sorted(list(st.session_state.forecasts.keys()))

            # Add multi-select for table SKUs with clearer labeling
            st.subheader("Select Data for Table View")
            table_skus = st.multiselect(
                "Choose SKUs to Include",
                options=all_skus,
                default=all_skus[:min(5, len(all_skus))],  # Default to first 5 SKUs or less
                key="table_skus",
                help="Select one or more SKUs to include in the table below"
            )

            # If no SKUs selected, default to showing all (up to a reasonable limit)
            if not table_skus:
                table_skus = all_skus[:min(5, len(all_skus))]
                st.info(f"Showing first {len(table_skus)} SKUs by default. Select specific SKUs above if needed.")

            # Process each selected SKU
            for sku in table_skus:
                forecast_data_for_sku = st.session_state.forecasts[sku]

                # Get all models for this SKU based on what was selected in the sidebar
                models_to_include = []

                # Get available models for this SKU
                available_models = []
                if 'model_comparison' in forecast_data_for_sku:
                    # Add available models to the list
                    available_models = list(forecast_data_for_sku['model_comparison'].keys())

                # Only include models that were selected in the sidebar
                for model in st.session_state.selected_models:
                    model_lower = model.lower()
                    # Include the model if it was selected and is available
                    if model_lower in available_models:
                        models_to_include.append(model_lower)

                # Get actual sales data for this SKU
                sku_sales = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku].copy()
                sku_sales.set_index('date', inplace=True)

                # For each model, create a row in the table
                for model in models_to_include:
                    # Mark if this is the best model
                    is_best_model = (model == forecast_data_for_sku.get('selected_model', ''))

                    # Create base row info
                    row = {
                        'sku_code': sku,
                        'sku_name': sku,  # Using SKU as name, replace with actual name if available
                        'model': model.upper(),
                        'best_model': '✓' if is_best_model else ''
                    }

                    # Get model forecast data
                    if model.lower() == forecast_data_for_sku.get('selected_model', ''):
                        # Use the primary model forecast
                        model_forecast = forecast_data_for_sku['forecast']
                    elif 'model_comparison' in forecast_data_for_sku and model.lower() in forecast_data_for_sku['model_comparison']:
                        # Get the specific model forecast data
                        model_forecast = forecast_data_for_sku['model_comparison'][model.lower()]
                    else:
                        # If the model isn't available, use an empty Series
                        model_forecast = pd.Series()

                    # Add historical/actual values (no prefix, just the date)
                    for date, col_name in zip(historical_dates, historical_cols):
                        # Remove "Actual:" prefix but track these columns separately for styling
                        actual_col_name = col_name  # Just use the date as column name
                        if date in sku_sales.index:
                            row[actual_col_name] = int(sku_sales.loc[date, 'quantity']) if not pd.isna(sku_sales.loc[date, 'quantity']) else 0
                        else:
                            row[actual_col_name] = 0

                    # Add forecast values (no prefix, just the date) - ensuring dates match
                    for date, col_name in zip(forecast_dates, forecast_date_cols):
                        forecast_col_name = col_name  # Just use the date as column name

                        # Check if this forecast exists in model_comparison
                        model_forecast_series = None

                        if 'model_comparison' in forecast_data_for_sku and model.lower() in forecast_data_for_sku['model_comparison']:
                            model_forecast_series = forecast_data_for_sku['model_comparison'][model.lower()]
                        elif model.lower() == forecast_data_for_sku.get('selected_model', ''):
                            model_forecast_series = forecast_data_for_sku['forecast']

                        # Now check if date exists in the model's forecast
                        try:
                            if model_forecast_series is not None:
                                if isinstance(model_forecast_series, pd.Series) and date in model_forecast_series.index:
                                    forecast_value = model_forecast_series[date]
                                elif isinstance(model_forecast_series, pd.DataFrame) and date in model_forecast_series.index:
                                    forecast_value = model_forecast_series.loc[date]
                                else:
                                    forecast_value = np.nan

                                try:
                                    # Check if value is NaN before conversion
                                    if not pd.isna(forecast_value) and not np.isnan(forecast_value):
                                        row[forecast_col_name] = int(round(forecast_value))
                                    else:
                                        # Handle NaN values
                                        row[forecast_col_name] = 0
                                except Exception as e:
                                    # Log the error with details
                                    print(f"Error converting forecast value for {model} at {date}: {str(e)}")
                                    row[forecast_col_name] = 0
                            else:
                                # If we can't find the forecast, set to 0
                                row[forecast_col_name] = 0
                        except Exception as e:
                            # Catch any unexpected errors and use a safe fallback
                            print(f"Error processing forecast for {model} at {date}: {str(e)}")
                            row[forecast_col_name] = 0

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

                    return styles

                # Add column group headers using expander
                forecast_explanation = """
                - **SKU Info**: Basic product information
                - **Actual Values**: Historical sales shown with blue background
                - **Forecast Values**: Predicted sales shown with yellow background
                - **✓**: Indicates the best performing model for each SKU
                """
                with st.expander("Understanding the Table", expanded=False):
                    st.markdown(forecast_explanation)

                # Add pagination for better performance with large datasets
                table_rows_per_page = 15  # Limit rows per page
                total_table_rows = len(all_sku_df)

                if total_table_rows > table_rows_per_page:
                    # Create pagination controls
                    table_page_col1, table_page_col2 = st.columns([1, 3])

                    with table_page_col1:
                        table_page = st.selectbox(
                            "Page",
                            options=list(range(1, (total_table_rows // table_rows_per_page) + (1 if total_table_rows % table_rows_per_page > 0 else 0) + 1)),
                            key="comprehensive_table_page"
                        )

                    with table_page_col2:
                        st.info(f"Showing page {table_page} of {(total_table_rows // table_rows_per_page) + (1 if total_table_rows % table_rows_per_page > 0 else 0)} (total rows: {total_table_rows})")

                    # Calculate start and end indices for the current page
                    start_idx = (table_page - 1) * table_rows_per_page
                    end_idx = min(start_idx + table_rows_per_page, total_table_rows)

                    # Get current page data
                    page_df = all_sku_df.iloc[start_idx:end_idx].copy()
                else:
                    # If fewer rows than page size, show all
                    page_df = all_sku_df

                # Use styling to highlight data column types with frozen columns till model name
                st.dataframe(
                    page_df.style.apply(highlight_data_columns, axis=None),
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
        col1, col2 = st.columns([3, 1])

        with col1:
            # Select SKU to display
            # Safely calculate the default index
            if st.session_state.selected_sku is None or st.session_state.selected_sku not in all_skus:
                default_index = 0
            else:
                try:
                    default_index = all_skus.index(st.session_state.selected_sku)
                except (ValueError, IndexError):
                    default_index = 0

            selected_sku_preview = st.selectbox(
                "Select a SKU to view historical sales data",
                options=all_skus,
                index=default_index,
                key="selected_sku_preview"
            )

            # Update session state
            st.session_state.selected_sku = selected_sku_preview

        with col2:
            # Show basic summary for the selected SKU
            sku_data = st.session_state.sales_data[st.session_state.sales_data['sku'] == selected_sku_preview]

            if not sku_data.empty:
                data_points = len(sku_data)
                avg_sales = round(sku_data['quantity'].mean(), 2)
                st.metric("Data Points", data_points)
                st.metric("Avg. Monthly Sales", avg_sales)

        # Show historical data chart for selected SKU
        if st.session_state.selected_sku:
            st.subheader(f"Historical Sales for {st.session_state.selected_sku}")

            # Filter data for the selected SKU
            sku_data = st.session_state.sales_data[st.session_state.sales_data['sku'] == st.session_state.selected_sku]

            # Create a simple plotly chart for historical data
            if not sku_data.empty:
                import plotly.express as px

                # Ensure data is sorted by date
                sku_data = sku_data.sort_values('date')

                # Create the chart
                fig = px.line(
                    sku_data, 
                    x='date', 
                    y='quantity',
                    title=f'Historical Sales for {st.session_state.selected_sku}',
                    labels={'quantity': 'Units Sold', 'date': 'Date'}
                )

                # Update layout
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Units Sold',
                    template='plotly_white'
                )

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

                # Show a small table with yearly or quarterly totals
                st.subheader("Sales Summary")

                # Add year and quarter columns
                sku_data['year'] = sku_data['date'].dt.year
                sku_data['quarter'] = sku_data['date'].dt.quarter

                # Create summary tables
                yearly_summary = sku_data.groupby('year')['quantity'].sum().reset_index()
                yearly_summary.columns = ['Year', 'Total Sales']

                quarterly_summary = sku_data.groupby(['year', 'quarter'])['quantity'].sum().reset_index()
                quarterly_summary.columns = ['Year', 'Quarter', 'Total Sales']
                quarterly_summary['Period'] = quarterly_summary['Year'].astype(str) + '-Q' + quarterly_summary['Quarter'].astype(str)
                quarterly_summary = quarterly_summary[['Period', 'Total Sales']]

                # Show summary tables in columns
                col1, col2 = st.columns(2)

                with col1:
                    st.write("Yearly Summary")
                    st.dataframe(yearly_summary, use_container_width=True)

                with col2:
                    st.write("Quarterly Summary")
                    st.dataframe(quarterly_summary, use_container_width=True)

    # Show a preview of the overall sales data
    st.subheader("Sales Data Preview")
    st.dataframe(st.session_state.sales_data.head(10), use_container_width=True)

    # Show summary statistics
    st.subheader("Sales Data Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total SKUs", len(st.session_state.sales_data['sku'].unique()))

    with col2:
        st.metric("Date Range", f"{st.session_state.sales_data['date'].min().strftime('%b %Y')} - {st.session_state.sales_data['date'].max().strftime('%b %Y')}")

    with col3:
        st.metric("Total Records", len(st.session_state.sales_data))
