import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import time
from datetime import datetime, timedelta
from utils.data_processor import process_sales_data
from utils.forecast_engine import extract_features, cluster_skus, generate_forecasts, evaluate_models, train_lstm_model, forecast_with_lstm, select_best_model
from utils.visualization import plot_forecast, plot_cluster_summary, plot_model_comparison
# Import parameter optimizer functions for using tuned parameters
from utils.parameter_optimizer import get_model_parameters
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

# Implement Croston Method for intermittent demand forecasting
def croston_forecast(time_series, forecast_periods, alpha=0.1):
    """
    Croston's method for intermittent demand forecasting.
    
    Parameters:
    -----------
    time_series : pandas.Series
        Time series of demand values (can contain zeros).
    forecast_periods : int
        Number of periods to forecast.
    alpha : float, optional (default=0.1)
        Smoothing parameter for both demand size and interval.
        
    Returns:
    --------
    pandas.Series
        Forecasted values for the specified number of periods.
    """
    # Convert to numpy array for easier operations
    y = time_series.values
    
    # Initialize variables
    y_i = y[0]  # Demand size
    q = 1       # Interval between demands
    p = 1       # Position in interval
    
    # Calculate forecast for the historical period
    forecasts = []
    
    for i in range(len(y)):
        # If demand occurs
        if y[i] > 0:
            # Update demand size estimate
            y_i = alpha * y[i] + (1 - alpha) * y_i
            # Update interval estimate
            q = alpha * p + (1 - alpha) * q
            # Reset position
            p = 1
        else:
            # Increment position
            p += 1
        
        # The forecast is demand size / interval
        forecasts.append(y_i / q)
    
    # Create future forecasts (same value for all future periods)
    future_forecasts = [y_i / q] * forecast_periods
    
    # Create a pandas Series with the original index plus future dates
    # Get the last date in the original time series
    last_date = time_series.index[-1]
    
    # Create future dates
    freq = pd.infer_freq(time_series.index)
    if freq is None:
        # If frequency can't be inferred, use 'MS' (month start) as default
        freq = 'MS'
    
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                 periods=forecast_periods, 
                                 freq=freq)
    
    # Combine historical and future forecasts
    all_forecasts = pd.Series(forecasts + future_forecasts, 
                              index=list(time_series.index) + list(future_dates))
    
    # Return only the future part
    return all_forecasts.iloc[-forecast_periods:]

# Function to check if a time series has intermittent demand
def is_intermittent_demand(time_series, threshold=0.4):
    """
    Check if a time series has intermittent demand (high percentage of zeros).
    
    Parameters:
    -----------
    time_series : pandas.Series
        Time series of demand values.
    threshold : float, optional (default=0.4)
        Threshold for percentage of zeros to consider demand intermittent.
        
    Returns:
    --------
    bool
        True if the demand is intermittent, False otherwise.
    """
    # Calculate percentage of zero values
    zero_percentage = (time_series == 0).mean()
    
    # Return True if percentage of zeros exceeds threshold
    return zero_percentage > threshold

# Initialize variables that might be used in multiple places
all_sku_data = []
show_all_models = False
custom_models_lower = []
first_forecast = None

# Page configuration is already set in app.py
# Do not call st.set_page_config() here as it will cause errors

# Initialize cache for forecast results
if 'v2_forecast_cache' not in st.session_state:
    st.session_state.v2_forecast_cache = {}

# Flag to track if models are loaded (lazy loading)
if 'v2_models_loaded' not in st.session_state:
    st.session_state.v2_models_loaded = False

# Check if data is loaded in session state
if 'sales_data' not in st.session_state or st.session_state.sales_data is None:
    st.warning("Please upload sales data on the main page first.")
    st.stop()

# Page title
st.title("V2 Demand Forecasting with Croston Method")
st.markdown("""
This module specializes in accurate demand forecasting for products with intermittent demand patterns.
The system automatically identifies SKUs with frequent zero values (>40%) and applies the Croston method,
which is optimized for intermittent demand. For regular demand patterns, standard forecasting models are used.
""")

# Add information about Croston method in an expander
with st.expander("About Croston Method for Intermittent Demand", expanded=True):
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### What is Intermittent Demand?
        
        Intermittent demand is characterized by:
        - Frequent periods with zero demand
        - Irregular demand occurrences
        - High variability in demand sizes
        
        This pattern is common in:
        - Spare parts
        - Slow-moving inventory
        - Seasonal or highly specialized products
        
        **In this application:** SKUs with >40% zero values in their history are automatically classified as intermittent.
        """)
    
    with col2:
        st.markdown("""
        ### How Croston's Method Works
        
        Croston's method specifically handles intermittent demand by:
        
        1. Separating demand into two parts:
           - Demand size (when demand occurs)
           - Time interval between demands
        
        2. Forecasting each part separately using exponential smoothing
        
        3. Final forecast = Expected demand size ÷ Expected demand interval
        """)

# Initialize session state variables for this page
if 'v2_forecast_periods' not in st.session_state:
    st.session_state.v2_forecast_periods = 12  # default 12 months
if 'v2_run_forecast' not in st.session_state:
    st.session_state.v2_run_forecast = False
if 'v2_selected_sku' not in st.session_state:
    st.session_state.v2_selected_sku = None
if 'v2_selected_skus' not in st.session_state:
    st.session_state.v2_selected_skus = []
if 'v2_show_all_clusters' not in st.session_state:
    st.session_state.v2_show_all_clusters = False
if 'v2_selected_models' not in st.session_state:
    st.session_state.v2_selected_models = []
if 'v2_sku_options' not in st.session_state:
    st.session_state.v2_sku_options = []
if 'v2_forecast_in_progress' not in st.session_state:
    st.session_state.v2_forecast_in_progress = False
if 'v2_forecast_progress' not in st.session_state:
    st.session_state.v2_forecast_progress = 0
if 'v2_forecast_current_sku' not in st.session_state:
    st.session_state.v2_forecast_current_sku = ""

# Create sidebar for settings
with st.sidebar:
    st.header("Forecast Settings")

    # Get available SKUs from sales data (always available)
    if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
        st.subheader("SKU Selection")

        # Function to update selected SKUs
        def update_selected_skus():
            new_selection = st.session_state.v2_sku_multiselect
            st.session_state.v2_selected_skus = new_selection
            if new_selection:
                st.session_state.v2_selected_sku = new_selection[0]  # Set first selected SKU as primary

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
        if st.session_state.v2_selected_skus:
            default_selection = [sku for sku in st.session_state.v2_selected_skus if sku in available_skus]

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
            key="v2_sku_multiselect",
            help="Select one or more SKUs to analyze or forecast"
        )

        # Ensure selected_skus is always updated
        st.session_state.v2_selected_skus = sku_selection

        # Update primary selected SKU if we have a selection
        if sku_selection:
            st.session_state.v2_selected_sku = sku_selection[0]

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
        value=st.session_state.v2_forecast_periods,
        step=1
    )
    st.session_state.v2_forecast_periods = forecast_periods

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

    # Original models from Demand Forecasting page
    if st.checkbox("ARIMA", value=True):
        models_to_evaluate.append("arima")

    if st.checkbox("SARIMAX (Seasonal)", value=True):
        models_to_evaluate.append("sarima")

    if st.checkbox("Prophet", value=True):
        models_to_evaluate.append("prophet")

    if st.checkbox("LSTM Neural Network", value=True):
        models_to_evaluate.append("lstm")

    if st.checkbox("Holt-Winters", value=True):
        models_to_evaluate.append("holtwinters")

    if st.checkbox("Decomposition", value=True, help="Uses time series decomposition to separate trend, seasonal, and residual components"):
        models_to_evaluate.append("decomposition")

    if st.checkbox("Ensemble Model", value=True, help="Combines multiple forecasting models for improved accuracy"):
        models_to_evaluate.append("ensemble")

    # Additional advanced models
    if st.checkbox("Auto ARIMA", value=True):
        models_to_evaluate.append("auto_arima")

    if st.checkbox("ETS (Exponential Smoothing)", value=True):
        models_to_evaluate.append("ets")

    if st.checkbox("Theta Method", value=True):
        models_to_evaluate.append("theta")

    if st.checkbox("Moving Average", value=True):
        models_to_evaluate.append("moving_average")
        
    # Croston method (specific to this page)
    if st.checkbox("Croston Method", value=True, help="Specialized method for intermittent demand patterns (products with frequent zero values)"):
        models_to_evaluate.append("croston")

    # Store selected models for visualization
    st.session_state.v2_selected_models = models_to_evaluate
    
    # Add option to use tuned parameters from hyperparameter tuning
    st.subheader("Parameter Options")
    
    # Initialize session state for using tuned parameters if not exists
    if 'use_tuned_parameters' not in st.session_state:
        st.session_state.use_tuned_parameters = False
    
    # Create columns for options and status
    param_col1, param_col2 = st.columns([3, 1])
    
    with param_col1:
        # Checkbox for using tuned parameters
        prev_param_value = st.session_state.use_tuned_parameters
        st.session_state.use_tuned_parameters = st.checkbox(
            "Use tuned parameters from Hyperparameter Tuning",
            value=st.session_state.use_tuned_parameters,
            help="Apply optimized model parameters from the Hyperparameter Tuning page. Will use default parameters if tuned parameters are not available for a specific SKU-model combination."
        )
        
        # If parameter setting changed, clear any cached forecasts to force refresh
        if prev_param_value != st.session_state.use_tuned_parameters:
            if 'v2_forecast_cache' in st.session_state:
                st.session_state.v2_forecast_cache = {}
                st.toast("Cleared forecast cache - please run forecast again with new parameter settings", icon="ℹ️")
    
    with param_col2:
        # Show parameter status
        if st.session_state.use_tuned_parameters:
            st.success("Tuned parameters: ON")
        else:
            st.info("Using default parameters")
    
    # Add option to forecast all or selected SKUs
    st.subheader("Forecast Scope")

    # Determine which SKUs to forecast
    forecast_scope = st.radio(
        "Choose SKUs to analyze",
        ["Selected SKUs Only", "All SKUs"],
        index=1,  # Default to All SKUs
        horizontal=True
    )

    # Get selected SKUs from session state
    selected_skus_to_forecast = []
    if 'v2_selected_skus' in st.session_state and st.session_state.v2_selected_skus:
        selected_skus_to_forecast = st.session_state.v2_selected_skus

    # Progress tracking callback function
    def forecast_progress_callback(current_index, current_sku, total_skus):
        # Update progress information in session state
        st.session_state.v2_forecast_progress = int((current_index / total_skus) * 100)
        st.session_state.v2_forecast_current_sku = current_sku

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

        # Check if we have cached results for these parameters
        cached_results_available = cache_key in st.session_state.v2_forecast_cache

        # Show different button text if cached results are available
        button_text = f"{forecast_button_text} (Cached)" if cached_results_available else forecast_button_text

        run_forecast_clicked = st.button(
            button_text, 
            key="v2_run_forecast_button",
            use_container_width=True
        )

        if run_forecast_clicked:
            # If we have cached results, use them
            if cached_results_available:
                st.toast("Using cached forecast results", icon="✅")
                st.session_state.v2_forecasts = st.session_state.v2_forecast_cache[cache_key]['forecasts']
                st.session_state.v2_clusters = st.session_state.v2_forecast_cache[cache_key]['clusters']
                st.session_state.v2_run_forecast = True
                st.session_state.v2_models_loaded = True

                # Show success message
                st.success(f"Loaded cached forecasts for {len(st.session_state.v2_forecasts)} SKUs!")

            else:
                # Set forecast in progress flag
                st.session_state.v2_forecast_in_progress = True
                st.session_state.v2_forecast_progress = 0
                st.session_state.v2_run_forecast = True

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
                    # Phase 1: Extract time series features for clustering
                    with spinner_placeholder:
                        with st.spinner("Extracting features..."):
                            phase_indicator.markdown("**Phase 1/3**")
                            status_text.markdown("### Step 1: Time Series Feature Extraction")
                            progress_details.info("Analyzing sales patterns and extracting key time series features for every SKU...")
                            features_df = extract_features(st.session_state.sales_data)
                            progress_bar.progress(10)
                            time.sleep(0.5)  # Add a small pause for visual effect

                    # Phase 2: Cluster SKUs
                    with spinner_placeholder:
                        with st.spinner("Clustering SKUs..."):
                            phase_indicator.markdown("**Phase 2/3**")
                            status_text.markdown("### Step 2: SKU Clustering")
                            progress_details.info("Grouping similar SKUs based on their sales patterns to optimize forecast model selection...")
                            st.session_state.v2_clusters = cluster_skus(features_df, n_clusters=num_clusters)
                            progress_bar.progress(20)
                            time.sleep(0.5)  # Add a small pause for visual effect

                    # Phase 3: Generate forecasts
                    phase_indicator.markdown("**Phase 3/3**")
                    status_text.markdown("### Step 3: Forecast Generation")

                    # Determine which SKUs to forecast
                    skus_to_forecast = None
                    if forecast_scope == "Selected SKUs Only" and selected_skus_to_forecast:
                        skus_to_forecast = selected_skus_to_forecast
                        progress_details.info(f"Generating forecasts for {len(skus_to_forecast)} selected SKUs using {len(models_to_evaluate)} different forecasting models...")
                    else:
                        total_skus = len(features_df)
                        progress_details.info(f"Generating forecasts for all {total_skus} SKUs using {len(models_to_evaluate)} different forecasting models...")

                    # Generate forecasts with model evaluation and progress tracking
                    with spinner_placeholder:
                        with st.spinner("Building forecast models..."):
                            # For simplicity, use the standard generate_forecasts function which already works
                            st.session_state.v2_forecasts = generate_forecasts(
                                st.session_state.sales_data,
                                st.session_state.v2_clusters,
                                forecast_periods=st.session_state.v2_forecast_periods,
                                evaluate_models_flag=evaluate_models_flag,
                                models_to_evaluate=models_to_evaluate,
                                selected_skus=skus_to_forecast,
                                progress_callback=forecast_progress_callback,
                                use_tuned_parameters=st.session_state.use_tuned_parameters
                            )

                    # Update progress based on callback data with improved visuals
                    # Create an animated progress update
                    last_progress = 0
                    while st.session_state.v2_forecast_progress < 100 and st.session_state.v2_forecast_current_sku:
                        current_progress = 20 + int(st.session_state.v2_forecast_progress * 0.8)
                        if current_progress > last_progress:
                            progress_bar.progress(current_progress)
                            last_progress = current_progress

                        with spinner_placeholder:
                            with st.spinner(f"Processing {st.session_state.v2_forecast_current_sku}..."):
                                # Update progress display with more dynamic information
                                status_text.markdown(f"### Processing: **{st.session_state.v2_forecast_current_sku}**")
                                progress_percentage = st.session_state.v2_forecast_progress
                                progress_details.info(f"Completed: **{progress_percentage}%** | Current SKU: **{st.session_state.v2_forecast_current_sku}**")
                                phase_indicator.markdown(f"**Processing {progress_percentage}% complete**")
                                time.sleep(0.1)

                    # Complete the progress bar with success animation
                    progress_bar.progress(100)
                    spinner_placeholder.success("✅ Complete")
                    phase_indicator.markdown("**Finished!**")
                    status_text.markdown("### ✨ Forecast Generation Completed Successfully!")
                    progress_details.success("All forecasts have been generated and are ready to explore!")

                    # If forecasts were generated, set default selected SKU
                    if st.session_state.v2_forecasts:
                        sku_list = sorted(list(st.session_state.v2_forecasts.keys()))
                        st.session_state.v2_sku_options = sku_list
                        if sku_list and not st.session_state.v2_selected_sku in sku_list:
                            st.session_state.v2_selected_sku = sku_list[0]

                        # Cache the forecast results for future use
                        cache_key = f"{forecast_scope}_{len(selected_skus_to_forecast)}_{forecast_periods}_{num_clusters}_{'-'.join(models_to_evaluate)}"
                        st.session_state.v2_forecast_cache[cache_key] = {
                            'forecasts': st.session_state.v2_forecasts,
                            'clusters': st.session_state.v2_clusters,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        st.session_state.v2_models_loaded = True

                    num_skus = len(st.session_state.v2_forecasts)
                    if num_skus > 0:
                        st.success(f"Successfully generated forecasts for {num_skus} SKUs!")
                    else:
                        st.error("No forecasts were generated. Please check your data and selected SKUs.")

                except Exception as e:
                    st.error(f"Error during forecast generation: {str(e)}")

                finally:
                    # Reset progress tracking
                    st.session_state.v2_forecast_in_progress = False
                    time.sleep(1)  # Keep the completed progress visible briefly

            # Clear the progress display after completion
            progress_placeholder.empty()

# Main content
if st.session_state.v2_run_forecast and 'v2_forecasts' in st.session_state and st.session_state.v2_forecasts:
    # Show cluster analysis
    st.header("SKU Cluster Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Display cluster summary chart
        cluster_fig = plot_cluster_summary(st.session_state.v2_clusters)
        st.plotly_chart(cluster_fig, use_container_width=True)

    with col2:
        # Show cluster details
        st.subheader("Cluster Characteristics")

        if 'v2_clusters' in st.session_state and st.session_state.v2_clusters is not None:
            cluster_groups = st.session_state.v2_clusters.groupby('cluster_name').size().reset_index()
            cluster_groups.columns = ['Cluster', 'Count']

            # Calculate percentage
            total_skus = cluster_groups['Count'].sum()
            cluster_groups['Percentage'] = (cluster_groups['Count'] / total_skus * 100).round(1)
            cluster_groups['Percentage'] = cluster_groups['Percentage'].astype(str) + '%'

            st.dataframe(cluster_groups, use_container_width=True)

            # Option to show all SKUs and their clusters
            show_all = st.checkbox("Show All SKUs and Their Clusters", value=st.session_state.v2_show_all_clusters)
            st.session_state.v2_show_all_clusters = show_all

            if show_all:
                sku_clusters = st.session_state.v2_clusters[['sku', 'cluster_name']].sort_values('cluster_name')
                st.dataframe(sku_clusters, use_container_width=True)

    # Show SKU-wise Model Selection Table
    st.header("SKU-wise Model Selection")

    # Create a dataframe with SKU and model information
    model_selection_data = []

    for sku, forecast_data in st.session_state.v2_forecasts.items():
        model_info = {
            'SKU': sku,
            'Selected Model': forecast_data['model_evaluation']['best_model'].upper() if 'model_evaluation' in forecast_data and 'best_model' in forecast_data['model_evaluation'] else forecast_data['model'].upper(),
            'Cluster': forecast_data['cluster_name']
        }

        # Add model evaluation metrics if available
        if 'model_evaluation' in forecast_data and forecast_data['model_evaluation']['metrics']:
            best_model = forecast_data['model_evaluation']['best_model']
            if best_model in forecast_data['model_evaluation']['metrics']:
                metrics = forecast_data['model_evaluation']['metrics'][best_model]
                model_info['RMSE'] = round(metrics['rmse'], 2)
                model_info['MAPE (%)'] = round(metrics['mape'], 2) if not np.isnan(metrics['mape']) else None

            # Add reason for selection
            model_info['Selection Reason'] = "Best performance on test data" if best_model == model_info['Selected Model'].lower() else "Fallback choice"

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

    # Display the filtered table
    st.dataframe(filtered_df, use_container_width=True)

    # Forecast explorer
    st.header("Forecast Explorer")

    # Create a SKU selection area
    # Allow user to select a SKU to view detailed forecast
    sku_list = list(st.session_state.v2_forecasts.keys())

    # Safely get an index for the selected SKU
    if st.session_state.v2_selected_sku is None or st.session_state.v2_selected_sku not in sku_list:
        default_index = 0
    else:
        try:
            default_index = sku_list.index(st.session_state.v2_selected_sku)
        except (ValueError, IndexError):
            default_index = 0

    selected_sku = st.selectbox(
        "Select a SKU to view forecast details",
        options=sku_list,
        index=default_index,
        key="v2_forecast_sku_selector"
    )
    st.session_state.v2_selected_sku = selected_sku

    # Show forecast details for selected SKU
    if selected_sku:
        forecast_data = st.session_state.v2_forecasts[selected_sku]

        # Tab section for forecast views
        # Check if this SKU has intermittent demand (display before the tabs)
        if forecast_data is not None:
            # Get the history data
            sku_history = forecast_data['history']
            
            # Calculate percentage of zero values
            zero_percentage = (sku_history == 0).mean() * 100
            is_intermittent = zero_percentage > 40
            
            # Create a container with info about the demand pattern
            demand_pattern_col1, demand_pattern_col2 = st.columns([3, 1])
            
            with demand_pattern_col1:
                if is_intermittent:
                    st.info(f"**Intermittent Demand Pattern Detected**: This SKU has {zero_percentage:.1f}% zero values in its history. The Croston method is recommended for forecasting.")
                else:
                    st.success(f"**Regular Demand Pattern**: This SKU has only {zero_percentage:.1f}% zero values in its history. Standard forecasting methods are appropriate.")
            
            with demand_pattern_col2:
                # Display a badge for the demand type
                if is_intermittent:
                    st.markdown("""
                    <div style="background-color:#E8F5E9; padding:8px; border-radius:5px; text-align:center;">
                        <span style="font-weight:bold; color:#2E7D32;">INTERMITTENT DEMAND</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color:#E3F2FD; padding:8px; border-radius:5px; text-align:center;">
                        <span style="font-weight:bold; color:#1565C0;">REGULAR DEMAND</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        forecast_tabs = st.tabs(["Forecast Chart", "Model Comparison", "Forecast Metrics"])

        with forecast_tabs[0]:
            # Forecast visualization section - full width
            # Get list of models to display
            # If we have multiple models selected, use them for visualization
            available_models = []
            selected_models_for_viz = []

            if 'model_evaluation' in forecast_data and 'all_models_forecasts' in forecast_data['model_evaluation']:
                # Get all available models from the forecast data
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
                    # Custom model selection (only show if not showing all models)
                    if not show_all_models:
                        # Get model options capitalized
                        model_options = [model.upper() for model in available_models]

                        # Ensure best model is included in the options
                        default_model = forecast_data['model'].upper()
                        if default_model not in model_options and default_model.lower() in available_models:
                            model_options.append(default_model)

                        # Get the selected models from sidebar
                        selected_sidebar_models = []
                        for m in st.session_state.v2_selected_models:
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
                if show_all_models:
                    # Use ALL models that were selected in the sidebar AND are available in the forecast data
                    selected_models_for_viz = []
                    for model in st.session_state.v2_selected_models:
                        model_lower = model.lower()
                        # Check if this model exists in the forecast data
                        if model_lower in available_models:
                            selected_models_for_viz.append(model_lower)
                elif custom_models_lower:
                    # Use custom selection from multiselect
                    selected_models_for_viz = custom_models_lower
                else:
                    # Default to the primary model if nothing is explicitly selected
                    selected_models_for_viz = [forecast_data['model']]

                # Ensure we have at least one model in the list
                if not selected_models_for_viz and available_models:
                    selected_models_for_viz = [available_models[0]]

                # Set test prediction flag based on checkbox
                forecast_data['show_test_predictions'] = show_test_predictions

            # Display forecast chart with selected models (FULL WIDTH)
            forecast_fig = plot_forecast(st.session_state.sales_data, forecast_data, selected_sku, selected_models_for_viz)
            st.plotly_chart(forecast_fig, use_container_width=True)

            # Add a note about model selection
            if selected_models_for_viz:
                st.info(f"Displaying forecasts for models: {', '.join([m.upper() for m in selected_models_for_viz])}")

            # Debug information to help troubleshoot
            with st.expander("Debug Information", expanded=False):
                st.write("Selected Models:", selected_models_for_viz)
                st.write("Available Models:", available_models if 'available_models' in locals() else "Not available")
                st.write("Model Evaluation:", "Available" if 'model_evaluation' in forecast_data else "Not available")
                if 'model_evaluation' in forecast_data and 'all_models_forecasts' in forecast_data['model_evaluation']:
                    st.write("Models with forecasts:", list(forecast_data['model_evaluation']['all_models_forecasts'].keys()))

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
                if selected_sku and selected_sku in st.session_state.v2_forecasts:
                    # Basic metrics at the top
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**SKU:** {selected_sku}")

                    with col2:
                        st.markdown(f"**Cluster:** {forecast_data['cluster_name']}")

                    with col3:
                        st.markdown(f"**Model Used:** {forecast_data['model'].upper()}")

                    # Show accuracy metric if available
                    if 'model_evaluation' in forecast_data and 'metrics' in forecast_data['model_evaluation']:
                        best_model = forecast_data['model_evaluation']['best_model']
                        if best_model in forecast_data['model_evaluation']['metrics']:
                            metrics = forecast_data['model_evaluation']['metrics'][best_model]
                            if 'mape' in metrics and not np.isnan(metrics['mape']):
                                st.metric("Forecast Accuracy", f"{(100-metrics['mape']):.1f}%", help="Based on test data evaluation")

                    # Forecast confidence
                    confidence_color = "green" if forecast_data['model'] != 'moving_average' else "orange"
                    confidence_text = "High" if forecast_data['model'] != 'moving_average' else "Medium"
                    st.markdown(f"**Forecast Confidence:** <span style='color:{confidence_color}'>{confidence_text}</span>", unsafe_allow_html=True)

                    # Create basic forecast table - without confidence intervals as requested
                    forecast_table = pd.DataFrame({
                        'Date': forecast_data['forecast'].index,
                        'Forecast': forecast_data['forecast'].values.round(0).astype(int)
                    })

                    # If we have model evaluation data for multiple models, show them side by side in the table
                    if 'model_evaluation' in forecast_data and 'all_models_forecasts' in forecast_data['model_evaluation']:
                        model_forecasts = forecast_data['model_evaluation']['all_models_forecasts']

                        # Show models explicitly selected by the user
                        if show_all_models:
                            # Use ALL models from sidebar that are available in the forecasts
                            models_to_display = []
                            for model in st.session_state.v2_selected_models:
                                model_lower = model.lower()
                                if model_lower in model_forecasts:
                                    models_to_display.append(model_lower)
                        elif custom_models_lower:
                            # Use custom selection from multiselect
                            models_to_display = [m for m in custom_models_lower 
                                              if m in model_forecasts]
                        else:
                            # If no models selected, use the best model (from model_evaluation)
                            if 'model_evaluation' in forecast_data and 'best_model' in forecast_data['model_evaluation']:
                                models_to_display = [forecast_data['model_evaluation']['best_model']]
                            else:
                                models_to_display = [forecast_data['model']]

                        # Add each selected model's forecast as a column - ensuring each value is properly obtained
                        for model in models_to_display:
                            if model in model_forecasts:
                                model_forecast = model_forecasts[model]

                                # Add each selected model's forecast as a column
                                forecast_values = []
                                for date in forecast_table['Date']:
                                    try:
                                        date_obj = pd.to_datetime(date)
                                        # First check if the exact date is in the model forecast index
                                        if date_obj in model_forecast.index:
                                            forecast_val = model_forecast[date_obj]
                                        # If not, try to get closest date if it's a forecast model with different index
                                        else:
                                            # Print debugging information
                                            closest_dates = model_forecast.index[model_forecast.index <= date_obj]
                                            if len(closest_dates) > 0:
                                                closest_date = closest_dates[-1]  # Get the most recent date before this one
                                                forecast_val = model_forecast[closest_date]
                                            else:
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

                                # Add the values to the forecast table with proper NaN handling
                                forecast_table[f'{model.upper()} Forecast'] = forecast_values

                                # Ensure there are no NaN values in the table
                                forecast_table.fillna(0, inplace=True)

                    # Format the date column to be more readable
                    forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')

                    # Display the enhanced table with styling
                    st.subheader("Forecast Data Table")
                    st.dataframe(
                        forecast_table.style.highlight_max(subset=['Forecast'], color='#d6eaf8')
                                         .highlight_min(subset=['Forecast'], color='#fadbd8')
                                         .format({'Range (±)': '{} units'}),
                        use_container_width=True,
                        height=min(35 * (len(forecast_table) + 1), 400)  # Dynamically size table height with scrolling
                    )
        with forecast_tabs[1]:
            # Model comparison visualization
            if 'model_evaluation' in forecast_data and forecast_data['model_evaluation']['metrics']:
                # Visual comparison of models
                st.subheader("Model Performance Comparison")

                # Use plot_model_comparison with correct parameters
                model_comparison_fig = plot_model_comparison(selected_sku, forecast_data)
                st.plotly_chart(model_comparison_fig, use_container_width=True)

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
            if 'model_evaluation' in forecast_data and forecast_data['model_evaluation']['metrics']:
                st.subheader("Model Evaluation Results")

                # Create table of model evaluation metrics
                metrics_data = []
                for model_name, metrics in forecast_data['model_evaluation']['metrics'].items():
                    metrics_data.append({
                        'Model': model_name.upper(),
                        'RMSE': round(metrics['rmse'], 2),
                        'MAPE (%)': round(metrics['mape'], 2) if not np.isnan(metrics['mape']) else "N/A",
                        'MAE': round(metrics['mae'], 2),
                        'Best Model': '✓' if model_name == forecast_data['model_evaluation']['best_model'] else ''
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
    if st.button("Prepare Forecast Export", key="v2_prepare_forecast_export"):
        with st.spinner("Preparing forecast data..."):
            # Create a DataFrame with all forecasts
            export_data = []

            for sku, forecast_data in st.session_state.v2_forecasts.items():
                for date, value in forecast_data['forecast'].items():
                    export_data.append({
                        'sku': sku,
                        'date': date,
                        'forecast': round(value),
                        'model': forecast_data['model'],
                        'cluster': forecast_data['cluster_name']
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
                file_name=f"v2_forecasts_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )

    # Add a large, clear section break to separate the forecast data table
    st.markdown("---")
    st.markdown("## Comprehensive Forecast Data Table")
    st.info("📊 This table shows historical and forecasted values with dates as columns. The table includes actual sales data and forecasts for each SKU/model combination.")

    # Prepare comprehensive data table
    if st.session_state.v2_forecasts:
        # Create a dataframe to store all SKUs data with reoriented structure
        all_sku_data = []

        # Get historical dates (use the first forecast as reference for dates)
        first_sku = list(st.session_state.v2_forecasts.keys())[0]
        first_forecast = st.session_state.v2_forecasts[first_sku]  # Define first_forecast here

        # Use sales data for historical dates instead of relying on train_set
        if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
            # Identify unique dates in historical data
            historical_dates = pd.to_datetime(sorted(st.session_state.sales_data['date'].unique()))

            # Show all historical data points as requested by the user
            # Commented out the limit to show all historical dates
            # if len(historical_dates) > 6:
            #     historical_dates = historical_dates[-6:]

            # Format dates for column names
            historical_cols = [date.strftime('%-d %b %Y') for date in historical_dates]

            # Get forecast dates from first SKU (for column headers)
            forecast_dates = first_forecast['forecast'].index
            forecast_date_cols = [date.strftime('%-d %b %Y') for date in forecast_dates]

            # Add SKU selector for the table
            all_skus = sorted(list(st.session_state.v2_forecasts.keys()))

            # Add multi-select for table SKUs with clearer labeling
            st.subheader("Select Data for Table View")
            table_skus = st.multiselect(
                "Choose SKUs to Include",
                options=all_skus,
                default=all_skus[:min(5, len(all_skus))],  # Default to first 5 SKUs or less
                key="v2_table_skus",
                help="Select one or more SKUs to include in the table below"
            )

            # If no SKUs selected, default to showing all (up to a reasonable limit)
            if not table_skus:
                table_skus = all_skus[:min(5, len(all_skus))]
                st.info(f"Showing first {len(table_skus)} SKUs by default. Select specific SKUs above if needed.")

            # Process each selected SKU
            for sku in table_skus:
                forecast_data_for_sku = st.session_state.v2_forecasts[sku]

                # Get all models for this SKU based on what was selected in the sidebar
                # Only include models that are in selected_models from the sidebar
                models_to_include = []

                # Get available models for this SKU
                available_models = []
                if 'model_evaluation' in forecast_data_for_sku and 'all_models_forecasts' in forecast_data_for_sku['model_evaluation']:
                    # Add available models to the list
                    available_models = list(forecast_data_for_sku['model_evaluation']['all_models_forecasts'].keys())

                # Only include models that were selected in the sidebar
                for model in st.session_state.v2_selected_models:
                    model_lower = model.lower()
                    # Include the model if it was selected and is available
                    if model_lower in available_models:
                        models_to_include.append(model_lower)

                # Get actual sales data for this SKU
                sku_sales = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku].copy()
                sku_sales.set_index('date', inplace=True)

                # For each model, create a row in the table
                for model in models_to_include:
                    # Find the model with lowest MAPE for this SKU
                    best_mape_model = None
                    lowest_mape = float('inf')
                    
                    # Check all models for this SKU and find the one with lowest MAPE
                    if 'model_evaluation' in forecast_data_for_sku and 'metrics' in forecast_data_for_sku['model_evaluation']:
                        for model_name, metrics in forecast_data_for_sku['model_evaluation']['metrics'].items():
                            if 'mape' in metrics and not np.isnan(metrics['mape']):
                                if metrics['mape'] < lowest_mape:
                                    lowest_mape = metrics['mape']
                                    best_mape_model = model_name
                    
                    # Mark if this is the best model (lowest MAPE)
                    is_best_model = (best_mape_model is not None and model.lower() == best_mape_model)

                    # Create base row info
                    row = {
                        'sku_code': sku,
                        'sku_name': sku,  # Using SKU as name, replace with actual name if available
                        'model': model.upper(),
                        'best_model': '✓' if is_best_model else ''
                    }
                    
                    # Add MAPE and MAE metrics if available
                    if ('model_evaluation' in forecast_data_for_sku and 
                        'metrics' in forecast_data_for_sku['model_evaluation'] and 
                        model.lower() in forecast_data_for_sku['model_evaluation']['metrics']):
                        metrics = forecast_data_for_sku['model_evaluation']['metrics'][model.lower()]
                        
                        # Add MAPE with proper handling of NaN values
                        if 'mape' in metrics:
                            row['MAPE (%)'] = f"{metrics['mape']:.2f}%" if not np.isnan(metrics['mape']) else "N/A"
                        
                        # Add MAE 
                        if 'mae' in metrics:
                            row['MAE'] = f"{metrics['mae']:.2f}" if not np.isnan(metrics['mae']) else "N/A"

                    # Get model forecast data
                    if model.lower() == forecast_data_for_sku['model']:
                        # Use the primary model forecast
                        model_forecast = forecast_data_for_sku['forecast']
                    elif ('model_evaluation' in forecast_data_for_sku and 
                          'all_models_forecasts' in forecast_data_for_sku['model_evaluation'] and 
                          model.lower() in forecast_data_for_sku['model_evaluation']['all_models_forecasts']):
                        # Get the specific model forecast data
                        model_forecast = forecast_data_for_sku['model_evaluation']['all_models_forecasts'][model.lower()]
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

                        # Check if this forecast exists in all_models_forecasts
                        model_forecast_series = None

                        if ('model_evaluation' in forecast_data_for_sku and 
                            'all_models_forecasts' in forecast_data_for_sku['model_evaluation'] and 
                            model.lower() in forecast_data_for_sku['model_evaluation']['all_models_forecasts']):
                            model_forecast_series = forecast_data_for_sku['model_evaluation']['all_models_forecasts'][model.lower()]

                        # Now check if date exists in the model's forecast
                        try:
                            if model_forecast_series is not None and date in model_forecast_series.index:
                                forecast_value = model_forecast_series[date]
                                try:
                                    # Check if value is NaN before conversion
                                    if not pd.isna(forecast_value) and not np.isnan(forecast_value):
                                        row[forecast_col_name] = int(round(forecast_value))
                                    else:
                                        # Handle NaN values
                                        print(f"Warning: NaN value detected for {model} at {date}")
                                        row[forecast_col_name] = 0
                                except Exception as e:
                                    # Log the error with details
                                    print(f"Error converting forecast value for {model} at {date}: {str(e)}")
                                    row[forecast_col_name] = 0
                            else:
                                # If we can't find the forecast, set to 0
                                print(f"No forecast found for {model} at {date}")
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
                
                # Add metrics columns for styling
                metrics_cols = ['MAPE (%)', 'MAE']
                
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
                        
                    for col in metrics_cols:
                        if col in df.columns:
                            styles[col] = 'background-color: #E8F5E9; font-weight: 500'  # Light green for metrics columns

                    for col in actual_cols:
                        styles[col] = 'background-color: #E3F2FD'  # Lighter blue for actual values

                    for col in forecast_cols:
                        styles[col] = 'background-color: #FFF8E1'  # Lighter yellow for forecast values
                        
                    # Check for intermittent demand SKUs
                    for i, sku_name in enumerate(df['sku']):
                        # Get SKU from the dataframe
                        sku = sku_name
                        
                        # Check if this SKU has intermittent demand
                        is_intermittent = False
                        if sku in st.session_state.v2_forecasts:
                            history = st.session_state.v2_forecasts[sku]['history']
                            # Calculate percentage of zero values
                            zero_percentage = (history == 0).mean() * 100
                            is_intermittent = zero_percentage > 40
                            
                            # If intermittent and Croston is available in models
                            if is_intermittent:
                                # Mark the SKU row with green left border to indicate intermittent
                                styles.iloc[i, df.columns.get_loc('sku')] += '; border-left: 4px solid #4CAF50'
                                
                                # Check if any column contains "CROSTON" (case insensitive)
                                for col in df.columns:
                                    if 'CROSTON' in col.upper():
                                        # Highlight Croston method with a special background
                                        styles.iloc[i, df.columns.get_loc(col)] = 'background-color: #E8F5E9; font-weight: bold'
                                
                                # Add intermittent demand indicator to notes column if it exists
                                if 'notes' in df.columns:
                                    curr_note = df.iloc[i, df.columns.get_loc('notes')]
                                    if pd.isna(curr_note) or curr_note == '':
                                        df.iloc[i, df.columns.get_loc('notes')] = "Intermittent demand detected"
                                    else:
                                        df.iloc[i, df.columns.get_loc('notes')] += "; Intermittent demand detected"

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
                - **Green Border**: Indicates an SKU with intermittent demand pattern (>40% zero values)
                - **Metrics**: Performance indicators (MAPE & MAE) shown with green background
                - **Actual Values**: Historical sales shown with blue background
                - **Forecast Values**: Predicted sales shown with yellow background
                - **✓**: Indicates the model with the lowest MAPE (%) for each SKU
                - **MAPE (%)**: Mean Absolute Percentage Error - lower is better
                - **MAE**: Mean Absolute Error - lower is better
                - **Croston Method**: Highlighted in green for intermittent demand SKUs
                """
                with st.expander("Understanding the Table", expanded=False):
                    st.markdown(forecast_explanation)

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
                            help="Check mark indicates model with lowest MAPE (%)"
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
                    
                    metrics_format = workbook.add_format({
                        'bg_color': '#E8F5E9',
                        'border': 1,
                        'num_format': '0.00'
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
                        elif col in metrics_cols:
                            worksheet.set_column(col_idx, col_idx, 12, metrics_format)
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
                        file_name=f"v2_sku_forecast_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
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
                        file_name=f"v2_sku_forecast_data_{datetime.now().strftime('%Y%m%d')}.csv",
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
            if st.session_state.v2_selected_sku is None or st.session_state.v2_selected_sku not in all_skus:
                default_index = 0
            else:
                try:
                    default_index = all_skus.index(st.session_state.v2_selected_sku)
                except (ValueError, IndexError):
                    default_index = 0

            selected_sku_preview = st.selectbox(
                "Select a SKU to view historical sales data",
                options=all_skus,
                index=default_index,
                key="v2_selected_sku_preview"
            )

            # Update session state
            st.session_state.v2_selected_sku = selected_sku_preview

        with col2:
            # Show basic summary for the selected SKU
            sku_data = st.session_state.sales_data[st.session_state.sales_data['sku'] == selected_sku_preview]

            if not sku_data.empty:
                data_points = len(sku_data)
                avg_sales = round(sku_data['quantity'].mean(), 2)
                st.metric("Data Points", data_points)
                st.metric("Avg. Monthly Sales", avg_sales)

        # Show historical data chart for selected SKU
        if st.session_state.v2_selected_sku:
            st.subheader(f"Historical Sales for {st.session_state.v2_selected_sku}")

            # Filter data for the selected SKU
            sku_data = st.session_state.sales_data[st.session_state.sales_data['sku'] == st.session_state.v2_selected_sku]

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
                    title=f'Historical Sales for {st.session_state.v2_selected_sku}',
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