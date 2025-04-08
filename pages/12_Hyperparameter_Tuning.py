import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import json
from datetime import datetime
from utils.data_processor import process_sales_data
from utils.parameter_optimizer import optimize_parameters_async, get_optimization_status, get_model_parameters
from utils.database import get_model_parameters, save_model_parameters

# Set page config
st.set_page_config(
    page_title="Hyperparameter Tuning",
    page_icon="🔧",
    layout="wide"
)

# Initialize session state variables
if 'sales_data' not in st.session_state or st.session_state.sales_data is None:
    st.warning("Please upload sales data on the main page first.")
    st.stop()

if 'tuning_in_progress' not in st.session_state:
    st.session_state.tuning_in_progress = False

if 'tuning_skus' not in st.session_state:
    st.session_state.tuning_skus = []

if 'tuning_models' not in st.session_state:
    st.session_state.tuning_models = []

if 'tuning_logs' not in st.session_state:
    st.session_state.tuning_logs = []

if 'tuning_results' not in st.session_state:
    st.session_state.tuning_results = {}

if 'tuning_progress' not in st.session_state:
    st.session_state.tuning_progress = 0

# Page title
st.title("🔧 Hyperparameter Tuning")
st.markdown("""
This module allows you to optimize model hyperparameters for specific SKUs to maximize forecast accuracy.
Tuned parameters are stored in the database for use in future forecasts.
""")

# Main layout with two columns
tuning_col1, tuning_col2 = st.columns([2, 1])

with tuning_col1:
    # SKU selection
    st.subheader("1. Select SKUs for Tuning")

    # Get all SKUs from the data
    all_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())

    # Select all or specific SKUs
    sku_selection_mode = st.radio(
        "SKU Selection Mode",
        ["Select Specific SKUs", "Tune All SKUs"],
        horizontal=True
    )

    if sku_selection_mode == "Select Specific SKUs":
        selected_skus = st.multiselect(
            "Select SKUs to tune",
            options=all_skus,
            default=all_skus[:min(3, len(all_skus))],
            help="Choose which SKUs to optimize parameters for"
        )
    else:
        st.info(f"All {len(all_skus)} SKUs will be tuned. This may take some time.")
        selected_skus = all_skus

    # Model selection
    st.subheader("2. Select Models for Tuning")

    tunable_models = {
        "auto_arima": "Auto ARIMA",
        "prophet": "Prophet",
        "ets": "ETS (Exponential Smoothing)",
        "theta": "Theta Method",
        "lstm": "LSTM Neural Network"
    }

    model_selection_mode = st.radio(
        "Model Selection Mode",
        ["Select Specific Models", "Tune All Models"],
        horizontal=True
    )

    if model_selection_mode == "Select Specific Models":
        selected_models = []

        # Create two columns for model selection checkboxes
        model_cols = st.columns(2)

        for i, (model_key, model_name) in enumerate(tunable_models.items()):
            col_idx = i % 2
            with model_cols[col_idx]:
                if st.checkbox(model_name, value=(model_key in ["auto_arima", "prophet"]), key=f"tune_model_{model_key}"):
                    selected_models.append(model_key)
    else:
        selected_models = list(tunable_models.keys())
        st.info(f"All {len(selected_models)} models will be tuned.")

    # Tuning options
    st.subheader("3. Tuning Options")

    tuning_options_cols = st.columns(2)

    with tuning_options_cols[0]:
        cross_validation = st.checkbox("Use cross-validation", value=True, 
                                      help="Use time series cross-validation for more robust parameter estimation")

    with tuning_options_cols[1]:
        n_trials = st.slider("Number of parameter combinations to try", 
                            min_value=10, max_value=100, value=30, step=10,
                            help="More trials may find better parameters but take longer")

    # Start tuning button
    start_button_disabled = len(selected_skus) == 0 or len(selected_models) == 0 or st.session_state.tuning_in_progress

    if st.button("Start Hyperparameter Tuning", disabled=start_button_disabled, use_container_width=True, type="primary"):
        # Store selected SKUs and models in session state
        st.session_state.tuning_skus = selected_skus
        st.session_state.tuning_models = selected_models
        st.session_state.tuning_in_progress = True
        st.session_state.tuning_logs = []
        st.session_state.tuning_progress = 0
        st.session_state.tuning_results = {}
        st.rerun()

with tuning_col2:
    # Tuning progress section
    st.subheader("Tuning Status")

    if st.session_state.tuning_in_progress:
        if st.button("Stop Tuning", use_container_width=True):
            st.session_state.tuning_in_progress = False
            st.info("Tuning process has been stopped.")
            st.rerun()

        # Show progress bar
        progress_placeholder = st.empty()

        # Show current SKU/model being processed
        status_placeholder = st.empty()

        # Calculate expected completion time
        total_tasks = len(st.session_state.tuning_skus) * len(st.session_state.tuning_models)
        status_placeholder.info(f"Processing {total_tasks} tuning tasks...")

# Progress monitor and tuning execution
if st.session_state.tuning_in_progress:
    # Create a container for logs and progress
    tuning_monitor = st.container()

    with tuning_monitor:
        st.write("### 4. Tuning Progress")

        # Progress bar
        progress_bar = st.progress(0)

        # Status text
        status_text = st.empty()

        # Create tabs for detailed model logs
        model_log_tabs = st.tabs(["All Logs", "Auto ARIMA", "Prophet", "ETS", "Theta", "LSTM"])

        with model_log_tabs[0]:
            all_logs_placeholder = st.empty()

        with model_log_tabs[1]:
            arima_logs_placeholder = st.empty()

        with model_log_tabs[2]:
            prophet_logs_placeholder = st.empty()

        with model_log_tabs[3]:
            ets_logs_placeholder = st.empty()

        with model_log_tabs[4]:
            theta_logs_placeholder = st.empty()

        with model_log_tabs[5]:
            lstm_logs_placeholder = st.empty()


        # Function to update tuning logs
        def tuning_progress_callback(sku, model, message, level="info", details=None):
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = {
                "timestamp": timestamp,
                "sku": sku,
                "model": model,
                "message": message,
                "level": level,
                "details": details  # Store detailed logs
            }

            # Add to logs
            if 'tuning_logs' not in st.session_state:
                st.session_state.tuning_logs = []

            st.session_state.tuning_logs.append(log_entry)

            # Update main log display
            all_log_text = ""
            for entry in st.session_state.tuning_logs[-20:]:  # Show last 20 log entries
                prefix = f"[{entry['timestamp']}] [{entry['sku']}] [{entry['model']}]"
                if entry['level'] == "error":
                    all_log_text += f"🔴 {prefix} {entry['message']}\n"
                elif entry['level'] == "warning":
                    all_log_text += f"🟠 {prefix} {entry['message']}\n"
                elif entry['level'] == "success":
                    all_log_text += f"🟢 {prefix} {entry['message']}\n"
                else:
                    all_log_text += f"🔵 {prefix} {entry['message']}\n"

            all_logs_placeholder.code(all_log_text)

            # Update model-specific logs
            model_placeholders = {
                "auto_arima": arima_logs_placeholder,
                "prophet": prophet_logs_placeholder,
                "ets": ets_logs_placeholder,
                "theta": theta_logs_placeholder,
                "lstm": lstm_logs_placeholder
            }

            # Update the corresponding model tab
            if model in model_placeholders:
                # Filter logs for this model
                model_logs = [log for log in st.session_state.tuning_logs 
                             if log['model'] == model or 
                                (log['model'] == 'selection' and model in log['message'])]

                model_log_text = ""
                for entry in model_logs[-50:]:  # Show last 50 model-specific entries
                    prefix = f"[{entry['timestamp']}] [{entry['sku']}]"

                    if entry['level'] == "error":
                        model_log_text += f"🔴 {prefix} {entry['message']}\n"
                    elif entry['level'] == "warning":
                        model_log_text += f"🟠 {prefix} {entry['message']}\n"
                    elif entry['level'] == "success":
                        model_log_text += f"🟢 {prefix} {entry['message']}\n"
                    else:
                        model_log_text += f"🔵 {prefix} {entry['message']}\n"

                    # Add detailed logs if available
                    if entry['details']:
                        model_log_text += f"    └─ Details: {entry['details']}\n"

                model_placeholders[model].code(model_log_text)

        # Run the tuning process
        try:
            # Helper function to format parameters nicely
            def format_parameters(params, model_type):
                if model_type == "auto_arima":
                    return f"p={params.get('p', 'auto')}, d={params.get('d', 'auto')}, q={params.get('q', 'auto')}"
                elif model_type == "prophet":
                    return f"changepoint_prior_scale={params.get('changepoint_prior_scale', 0.05):.3f}, seasonality_prior_scale={params.get('seasonality_prior_scale', 10.0):.3f}"
                elif model_type == "ets":
                    return f"trend={params.get('trend', 'add')}, seasonal={params.get('seasonal', None)}, damped={params.get('damped_trend', False)}"
                elif model_type == "theta":
                    return f"Theta parameters optimized"
                elif model_type == "lstm":
                    return f"units={params.get('units', 50)}, layers={params.get('n_layers', 2)}"
                else:
                    return str(params)

            # Process each SKU and model
            total_count = len(st.session_state.tuning_skus) * len(st.session_state.tuning_models)
            current_count = 0

            for sku in st.session_state.tuning_skus:
                # Filter data for this SKU
                sku_data_filtered = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku].copy()

                if len(sku_data_filtered) < 8:
                    tuning_progress_callback(
                        sku, "all", 
                        f"Insufficient data for tuning (needs at least 8 data points, found {len(sku_data_filtered)})", 
                        "warning"
                    )
                    continue

                for model_type in st.session_state.tuning_models:
                    try:
                        # Update status
                        current_count += 1
                        progress_percentage = current_count / total_count

                        status_text.info(f"Processing SKU: **{sku}**, Model: **{model_type.upper()}** ({current_count}/{total_count})")
                        progress_bar.progress(progress_percentage)

                        # Log start of tuning
                        tuning_progress_callback(sku, model_type, f"Starting parameter optimization", "info")

                        # Run parameter optimization (this would be async in a real implementation)
                        # For demo, simulate the optimization process with a placeholder function
                        tuning_progress_callback(sku, model_type, f"Testing different parameter combinations...", "info")

                        # Setup logging for detailed model information
                        import logging
                        import io

                        # Create a string IO object to capture log output
                        log_capture = io.StringIO()

                        # Configure a handler to capture detailed logs
                        handler = logging.StreamHandler(log_capture)
                        handler.setLevel(logging.DEBUG)

                        # Create formatter
                        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        handler.setFormatter(formatter)

                        # Add handlers to loggers
                        model_logger = logging.getLogger(model_type)
                        model_logger.setLevel(logging.DEBUG)
                        model_logger.addHandler(handler)

                        # For statsmodels (used by ARIMA and ETS)
                        if model_type in ['auto_arima', 'ets']:
                            sm_logger = logging.getLogger('statsmodels')
                            sm_logger.setLevel(logging.INFO)
                            sm_logger.addHandler(handler)

                        # For prophet
                        if model_type == 'prophet':
                            prophet_logger = logging.getLogger('prophet')
                            prophet_logger.setLevel(logging.INFO)
                            prophet_logger.addHandler(handler)

                        # Enhanced callback to include detailed logs
                        def enhanced_callback(s, m, msg, level="info"):
                            # Get the current captured logs
                            log_details = log_capture.getvalue()

                            # Reset the capture buffer if it's getting too large
                            if len(log_details) > 5000:
                                log_capture.truncate(0)
                                log_capture.seek(0)

                            # Call the original callback with the captured logs
                            tuning_progress_callback(s, m, msg, level, details=log_details)

                        # Simulate optimization by calling the actual optimize_parameters_async function
                        # This would be replaced with real parameter optimization
                        result = optimize_parameters_async(
                            sku=sku,
                            model_type=model_type,
                            data=sku_data_filtered,
                            cross_validation=cross_validation,
                            n_trials=n_trials,
                            progress_callback=enhanced_callback,
                            priority=True
                        )

                        # Clean up loggers
                        model_logger.removeHandler(handler)
                        if model_type in ['auto_arima', 'ets']:
                            sm_logger.removeHandler(handler)
                        if model_type == 'prophet':
                            prophet_logger.removeHandler(handler)

                        # Sleep to simulate processing time
                        time.sleep(0.5)

                        # Get the optimized parameters from the database
                        optimal_params = get_model_parameters(sku, model_type)

                        # Store results in session state
                        if sku not in st.session_state.tuning_results:
                            st.session_state.tuning_results[sku] = {}

                        # Check if we have valid parameters and store them
                        if optimal_params and 'parameters' in optimal_params:
                            st.session_state.tuning_results[sku][model_type] = optimal_params['parameters']
                        else:
                            # If no parameters were found, store an empty dict
                            st.session_state.tuning_results[sku][model_type] = {}

                        # Report success with formatted parameters
                        if optimal_params and 'parameters' in optimal_params:
                            success_msg = f"Successfully tuned parameters: {format_parameters(optimal_params['parameters'], model_type)}"
                            tuning_progress_callback(sku, model_type, success_msg, "success")
                        else:
                            tuning_progress_callback(sku, model_type, "Optimization completed but no parameters returned", "warning")

                    except Exception as e:
                        error_msg = f"Error tuning {model_type} for {sku}: {str(e)}"
                        tuning_progress_callback(sku, model_type, error_msg, "error")

            # Finalize progress
            progress_bar.progress(1.0)
            status_text.success(f"Hyperparameter tuning completed for {len(st.session_state.tuning_skus)} SKUs!")

            # Set tuning_in_progress to False when done
            st.session_state.tuning_in_progress = False

        except Exception as e:
            st.error(f"An error occurred during tuning: {str(e)}")
            st.session_state.tuning_in_progress = False

# Display tuning results visualization
if not st.session_state.tuning_in_progress and (st.session_state.tuning_results or 'tuning_results' in st.session_state):
    st.write("### 5. Tuning Results Visualization")

    # Get all tuned parameters from database for all SKUs that were tuned
    tuning_results = {}

    # Try to get results from session state first
    if st.session_state.tuning_results:
        tuning_results = st.session_state.tuning_results
    else:
        # If not in session state, fetch from database
        for sku in st.session_state.tuning_skus:
            tuning_results[sku] = {}
            for model_type in st.session_state.tuning_models:
                params = get_model_parameters(sku, model_type)
                if params and 'parameters' in params:
                    tuning_results[sku][model_type] = params['parameters']

    if tuning_results:
        # Create tabs for each model
        model_names = {
            "auto_arima": "Auto ARIMA", 
            "prophet": "Prophet",
            "ets": "ETS",
            "theta": "Theta",
            "lstm": "LSTM"
        }

        # Get unique models that have results
        unique_models = set()
        for sku_results in tuning_results.values():
            unique_models.update(sku_results.keys())

        if unique_models:
            # Create tabs for each model
            model_tabs = st.tabs([model_names.get(model, model.upper()) for model in unique_models])

            # Create visualizations for each model
            for i, model in enumerate(unique_models):
                with model_tabs[i]:
                    st.subheader(f"{model_names.get(model, model.upper())} Parameters")

                    # Extract parameters for this model
                    model_params = {}
                    for sku, sku_results in tuning_results.items():
                        if model in sku_results:
                            model_params[sku] = sku_results[model]

                    if model_params:
                        # Create parameter visualization based on model type
                        if model == "auto_arima":
                            # Create a table for ARIMA parameters (p, d, q)
                            arima_data = []
                            for sku, params in model_params.items():
                                if params:
                                    arima_data.append({
                                        "SKU": sku,
                                        "p": params.get("p", "N/A"),
                                        "d": params.get("d", "N/A"),
                                        "q": params.get("q", "N/A")
                                    })

                            if arima_data:
                                arima_df = pd.DataFrame(arima_data)
                                st.dataframe(arima_df, use_container_width=True)

                                # Create a bar chart for p, d, q values
                                fig = go.Figure()

                                for param in ["p", "d", "q"]:
                                    fig.add_trace(go.Bar(
                                        x=arima_df["SKU"],
                                        y=arima_df[param],
                                        name=param.upper()
                                    ))

                                fig.update_layout(
                                    title="ARIMA Parameters by SKU",
                                    xaxis_title="SKU",
                                    yaxis_title="Parameter Value",
                                    barmode="group"
                                )

                                st.plotly_chart(fig, use_container_width=True)

                        elif model == "prophet":
                            # Create visualization for Prophet parameters
                            prophet_data = []
                            for sku, params in model_params.items():
                                if params:
                                    prophet_data.append({
                                        "SKU": sku,
                                        "changepoint_prior_scale": params.get("changepoint_prior_scale", 0.05),
                                        "seasonality_prior_scale": params.get("seasonality_prior_scale", 10.0),
                                        "seasonality_mode": params.get("seasonality_mode", "additive")
                                    })

                            if prophet_data:
                                prophet_df = pd.DataFrame(prophet_data)
                                st.dataframe(prophet_df, use_container_width=True)

                                # Create scatter plot for changepoint vs seasonality
                                fig = px.scatter(
                                    prophet_df, 
                                    x="changepoint_prior_scale", 
                                    y="seasonality_prior_scale",
                                    color="seasonality_mode",
                                    hover_name="SKU",
                                    log_x=True,
                                    log_y=True,
                                    title="Prophet Hyperparameters"
                                )

                                fig.update_layout(
                                    xaxis_title="Changepoint Prior Scale",
                                    yaxis_title="Seasonality Prior Scale"
                                )

                                st.plotly_chart(fig, use_container_width=True)

                        elif model == "ets":
                            # Create visualization for ETS parameters
                            ets_data = []
                            for sku, params in model_params.items():
                                if params:
                                    ets_data.append({
                                        "SKU": sku,
                                        "trend": params.get("trend", "None"),
                                        "seasonal": params.get("seasonal", "None"),
                                        "seasonal_periods": params.get("seasonal_periods", 0),
                                        "damped_trend": params.get("damped_trend", False)
                                    })

                            if ets_data:
                                ets_df = pd.DataFrame(ets_data)
                                ets_df['trend'] = ets_df['trend'].fillna("None")
                                ets_df['seasonal'] = ets_df['seasonal'].fillna("None")

                                st.dataframe(ets_df, use_container_width=True)

                                # Create grouped bar chart for trend and seasonal type
                                trend_counts = ets_df['trend'].value_counts().reset_index()
                                trend_counts.columns = ['trend', 'count']

                                seasonal_counts = ets_df['seasonal'].value_counts().reset_index()
                                seasonal_counts.columns = ['seasonal', 'count']

                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=trend_counts['trend'],
                                    y=trend_counts['count'],
                                    name='Trend Type'
                                ))

                                fig.add_trace(go.Bar(
                                    x=seasonal_counts['seasonal'],
                                    y=seasonal_counts['count'],
                                    name='Seasonal Type'
                                ))

                                fig.update_layout(
                                    title="ETS Model Components",
                                    xaxis_title="Component Type",
                                    yaxis_title="Count",
                                    barmode='group'
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # Show damped trend distribution
                                damped_counts = ets_df['damped_trend'].value_counts().reset_index()
                                damped_counts.columns = ['Damped', 'Count']

                                fig = px.pie(
                                    damped_counts,
                                    values='Count',
                                    names='Damped',
                                    title="Damped Trend Usage"
                                )

                                st.plotly_chart(fig, use_container_width=True)

                        else:
                            # Generic parameter visualization for other models
                            st.write(f"Parameters for {model_names.get(model, model.upper())}")

                            # Create a generic table view
                            param_data = []
                            for sku, params in model_params.items():
                                if params:
                                    row = {"SKU": sku}
                                    # Add parameters as columns
                                    for param_name, param_value in params.items():
                                        row[param_name] = param_value
                                    param_data.append(row)

                            if param_data:
                                # Create DataFrame
                                param_df = pd.DataFrame(param_data)
                                st.dataframe(param_df, use_container_width=True)
                    else:
                        st.info(f"No tuning results available for {model_names.get(model, model.upper())}")
        else:
            st.info("No tuning results available yet. Run hyperparameter tuning first.")
    else:
        st.info("No tuning results available yet. Run hyperparameter tuning first.")

# Add information about integrating with forecasting engine
st.write("### 6. Integration with Demand Forecasting")
st.markdown("""
The tuned parameters from this page are automatically saved to the database and will be used by the 
forecasting engine. When you run forecasts on any of the Demand Forecasting pages, the system will check 
for optimized parameters and use them if available, resulting in more accurate forecasts.

Benefits of hyperparameter tuning:
- **Improved Accuracy**: Custom parameters for each SKU's unique patterns
- **Better Model Selection**: Helps identify which models work best for different SKUs
- **Seasonal Detection**: Optimizes seasonal components for accurate trend capture
- **Reduced Error**: Minimizes forecast errors through scientific optimization
""")

# Show database status
with st.expander("Parameter Database Status", expanded=False):
    # Get all optimized parameters from database (summary only)
    st.write("#### Optimized Parameters Available in Database")

    try:
        # Create a summary of tuned parameters in the database
        tuned_params_summary = []

        for sku in all_skus:
            for model in ["auto_arima", "prophet", "ets", "theta", "lstm"]:
                params = get_model_parameters(sku, model)
                if params and 'last_updated' in params:
                    tuned_params_summary.append({
                        "SKU": sku,
                        "Model": model_names.get(model, model.upper()),
                        "Parameters": "Available",
                        "Last Updated": params['last_updated'].strftime("%Y-%m-%d"),
                        "Score": f"{params['best_score']:.4f}" if 'best_score' in params and params['best_score'] is not None else "N/A"
                    })

        if tuned_params_summary:
            summary_df = pd.DataFrame(tuned_params_summary)
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No optimized parameters found in the database.")
    except Exception as e:
        st.error(f"Error fetching parameter database status: {str(e)}")