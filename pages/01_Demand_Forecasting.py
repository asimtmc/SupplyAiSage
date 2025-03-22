import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import time
from datetime import datetime
from utils.data_processor import process_sales_data
from utils.forecast_engine import extract_features, cluster_skus, generate_forecasts
from utils.visualization import plot_forecast, plot_cluster_summary, plot_model_comparison

# Set page config
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

# Check if data is loaded in session state
if 'sales_data' not in st.session_state or st.session_state.sales_data is None:
    st.warning("Please upload sales data on the main page first.")
    st.stop()

# Page title
st.title("AI-Powered Demand Forecasting")
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

    # Get selected SKUs from session state
    selected_skus_to_forecast = []
    if 'selected_skus' in st.session_state and st.session_state.selected_skus:
        selected_skus_to_forecast = st.session_state.selected_skus

    # Progress tracking callback function
    def forecast_progress_callback(current_index, current_sku, total_skus):
        # Update progress information in session state
        st.session_state.forecast_progress = int((current_index / total_skus) * 100)
        st.session_state.forecast_current_sku = current_sku

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
        run_forecast_clicked = st.button(
            forecast_button_text, 
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
                            st.session_state.clusters = cluster_skus(features_df, n_clusters=num_clusters)
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
                            st.session_state.forecasts = generate_forecasts(
                                st.session_state.sales_data,
                                st.session_state.clusters,
                                forecast_periods=st.session_state.forecast_periods,
                                evaluate_models_flag=evaluate_models_flag,
                                models_to_evaluate=models_to_evaluate,
                                selected_skus=skus_to_forecast,
                                progress_callback=forecast_progress_callback
                            )
                    
                    # Update progress based on callback data with improved visuals
                    # Create an animated progress update
                    last_progress = 0
                    while st.session_state.forecast_progress < 100 and st.session_state.forecast_current_sku:
                        current_progress = 20 + int(st.session_state.forecast_progress * 0.8)
                        if current_progress > last_progress:
                            progress_bar.progress(current_progress)
                            last_progress = current_progress
                            
                        with spinner_placeholder:
                            with st.spinner(f"Processing {st.session_state.forecast_current_sku}..."):
                                # Update progress display with more dynamic information
                                status_text.markdown(f"### Processing: **{st.session_state.forecast_current_sku}**")
                                progress_percentage = st.session_state.forecast_progress
                                progress_details.info(f"Completed: **{progress_percentage}%** | Current SKU: **{st.session_state.forecast_current_sku}**")
                                phase_indicator.markdown(f"**Processing {progress_percentage}% complete**")
                                time.sleep(0.1)
                    
                    # Complete the progress bar with success animation
                    progress_bar.progress(100)
                    spinner_placeholder.success("✅ Complete")
                    phase_indicator.markdown("**Finished!**")
                    status_text.markdown("### ✨ Forecast Generation Completed Successfully!")
                    progress_details.success("All forecasts have been generated and are ready to explore!")
                    
                    # If forecasts were generated, set default selected SKU
                    if st.session_state.forecasts:
                        sku_list = sorted(list(st.session_state.forecasts.keys()))
                        st.session_state.sku_options = sku_list
                        if sku_list and not st.session_state.selected_sku in sku_list:
                            st.session_state.selected_sku = sku_list[0]
                    
                    num_skus = len(st.session_state.forecasts)
                    if num_skus > 0:
                        st.success(f"Successfully generated forecasts for {num_skus} SKUs!")
                    else:
                        st.error("No forecasts were generated. Please check your data and selected SKUs.")
                    
                except Exception as e:
                    st.error(f"Error during forecast generation: {str(e)}")
                
                finally:
                    # Reset progress tracking
                    st.session_state.forecast_in_progress = False
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
        model_info = {
            'SKU': sku,
            'Selected Model': forecast_data['model'].upper(),
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
            model_info['Selection Reason'] = "Best performance on test data" if best_model == forecast_data['model'] else "Fallback choice"

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

    # Create a more prominent SKU selection area with columns
    col1, col2 = st.columns([2, 1])

    with col1:
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
            index=default_index
        )
        st.session_state.selected_sku = selected_sku

    with col2:
        # Show basic info about the selected SKU
        if selected_sku and selected_sku in st.session_state.forecasts:
            forecast_data = st.session_state.forecasts[selected_sku]
            model_name = forecast_data['model'].upper()
            cluster_name = forecast_data['cluster_name']

            st.write(f"**Model:** {model_name}")
            st.write(f"**Cluster:** {cluster_name}")

            # Show a quick metric if available
            if 'model_evaluation' in forecast_data and 'metrics' in forecast_data['model_evaluation']:
                best_model = forecast_data['model_evaluation']['best_model']
                if best_model in forecast_data['model_evaluation']['metrics']:
                    metrics = forecast_data['model_evaluation']['metrics'][best_model]
                    if 'mape' in metrics and not np.isnan(metrics['mape']):
                        st.metric("Forecast Accuracy", f"{(100-metrics['mape']):.1f}%", help="Based on test data evaluation")

    # Show forecast details for selected SKU
    if selected_sku:
        forecast_data = st.session_state.forecasts[selected_sku]

        # Tab section for forecast views
        forecast_tabs = st.tabs(["Forecast Chart", "Model Comparison", "Forecast Metrics"])

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
                        # Custom model selection (only show if not showing all models)
                        if not show_all_models and len(available_models) > 1:
                            # Get model options capitalized
                            model_options = [model.upper() for model in available_models]
                            # Default to the best model and ensure it's in the options list
                            default_model = forecast_data['model'].upper()
                            if default_model not in model_options:
                                model_options.append(default_model)
                                
                            # Create multiselect for custom model selection
                            custom_models = st.multiselect(
                                "Select Models to Display",
                                options=model_options,
                                default=[default_model] if default_model in model_options else [],
                                help="Select one or more models to display on chart"
                            )
                            # Convert back to lowercase
                            custom_models_lower = [model.lower() for model in custom_models]
                        else:
                            custom_models_lower = []

                    # Determine which models to display based on selections
                    if show_all_models:
                        # Use all selected models from sidebar
                        selected_models_for_viz = [m for m in st.session_state.selected_models if m in available_models]
                    elif custom_models_lower:
                        # Use custom selection
                        selected_models_for_viz = custom_models_lower
                    else:
                        # Default to best model only
                        selected_models_for_viz = [forecast_data['model']]

                    # Set test prediction flag based on checkbox
                    if show_test_predictions:
                        forecast_data['show_test_predictions'] = True

                # Display forecast chart with selected models
                forecast_fig = plot_forecast(st.session_state.sales_data, forecast_data, selected_sku, selected_models_for_viz)
                st.plotly_chart(forecast_fig, use_container_width=True)

                # Show training/test split information if available
                if 'train_set' in forecast_data and 'test_set' in forecast_data:
                    train_count = len(forecast_data['train_set'])
                    test_count = len(forecast_data['test_set'])
                    total_points = train_count + test_count
                    train_pct = int((train_count / total_points) * 100)
                    test_pct = int((test_count / total_points) * 100)

                    st.caption(f"Data split: {train_count} training points ({train_pct}%) and {test_count} test points ({test_pct}%)")
                

            with col2:
                # Show forecast details (same as before)
                st.subheader("Forecast Details")

                st.markdown(f"**SKU:** {selected_sku}")
                st.markdown(f"**Cluster:** {forecast_data['cluster_name']}")
                st.markdown(f"**Model Used:** {forecast_data['model'].upper()}")

                # Forecast confidence
                confidence_color = "green" if forecast_data['model'] != 'moving_average' else "orange"
                confidence_text = "High" if forecast_data['model'] != 'moving_average' else "Medium"
                st.markdown(f"**Forecast Confidence:** <span style='color:{confidence_color}'>{confidence_text}</span>", unsafe_allow_html=True)

                # Enhanced forecast table with additional details
                st.subheader("Forecast Data Table")
                
                # Create basic forecast table - without confidence intervals as requested
                forecast_table = pd.DataFrame({
                    'Date': forecast_data['forecast'].index,
                    'Forecast': forecast_data['forecast'].values.round(0).astype(int)
                })
                
                # If we have model evaluation data for multiple models, show them side by side in the table
                if (show_all_models or len(custom_models_lower) > 0) and 'model_evaluation' in forecast_data and 'all_models_forecasts' in forecast_data['model_evaluation']:
                    model_forecasts = forecast_data['model_evaluation']['all_models_forecasts']
                    
                    # Determine which models to include
                    models_to_display = custom_models_lower if len(custom_models_lower) > 0 else available_models
                    
                    # Add each selected model's forecast as a column
                    for model in models_to_display:
                        if model in model_forecasts and model != forecast_data['model']:  # Skip primary model as it's already in 'Forecast'
                            model_forecast = model_forecasts[model]
                            if len(model_forecast) == len(forecast_table):
                                forecast_table[f'{model.upper()} Forecast'] = model_forecast.values.round(0).astype(int)
                
                # Format the date column to be more readable
                forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')
                
                # Display the enhanced table with styling
                st.dataframe(
                    forecast_table.style.highlight_max(subset=['Forecast'], color='#d6eaf8')
                                      .highlight_min(subset=['Forecast'], color='#fadbd8')
                                      .format({'Range (±)': '{} units'}),
                    use_container_width=True,
                    height=min(35 * (len(forecast_table) + 1), 400)  # Dynamically size table height with scrolling
                )


        # Add the forecast data table outside of the tabs to take full width
        st.header("Forecast Data Table")
        st.info("This table shows historical and forecasted values with dates as columns. The table includes actual sales data and forecasts for each SKU/model combination.")
        
        # Prepare comprehensive data table
        if st.session_state.forecasts:
            # Create a dataframe to store all SKUs data with reoriented structure
            all_sku_data = []
            
            # Get historical dates (use the first forecast as reference for dates)
            first_sku = list(st.session_state.forecasts.keys())[0]
            first_forecast = st.session_state.forecasts[first_sku]
            
            # Make sure we have train data to extract historical dates
            if 'train_set' in first_forecast:
                # Identify unique dates in historical data
                historical_dates = pd.to_datetime(sorted(st.session_state.sales_data['date'].unique()))
                
                # Limit to a reasonable number of historical columns (e.g., last 6 months)
                if len(historical_dates) > 6:
                    historical_dates = historical_dates[-6:]
                
                # Format dates for column names
                historical_cols = [date.strftime('%-d %b %Y') for date in historical_dates]
                
                # Get forecast dates from first SKU (for column headers)
                forecast_dates = first_forecast['forecast'].index
                forecast_date_cols = [date.strftime('%-d %b %Y') for date in forecast_dates]
                
                # Add SKU selector for the table
                all_skus = sorted(list(st.session_state.forecasts.keys()))
                
                # Add multi-select for table SKUs
                table_skus = st.multiselect(
                    "Select SKUs to include in the table",
                    options=all_skus,
                    default=[selected_sku] if selected_sku in all_skus else [],
                    help="Select specific SKUs to include in the table below"
                )
                
                # If no SKUs selected, use the currently selected one
                if not table_skus:
                    table_skus = [selected_sku] if selected_sku in all_skus else []
                
                # Process each selected SKU
                for sku in table_skus:
                    forecast_data_for_sku = st.session_state.forecasts[sku]
                    
                    # Get all models for this SKU
                    models_to_include = [forecast_data_for_sku['model']]  # Start with best model
                    
                    if 'model_evaluation' in forecast_data_for_sku and 'all_models_forecasts' in forecast_data_for_sku['model_evaluation']:
                        # Add other models if available
                        for model in forecast_data_for_sku['model_evaluation']['all_models_forecasts']:
                            if model != forecast_data_for_sku['model']:  # Don't duplicate best model
                                models_to_include.append(model)
                    
                    # Get actual sales data for this SKU
                    sku_sales = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku].copy()
                    sku_sales.set_index('date', inplace=True)
                    
                    # For each model, create a row in the table
                    for model in models_to_include:
                        # Mark if this is the best model
                        is_best_model = (model == forecast_data_for_sku['model'])
                        
                        # Create base row info
                        row = {
                            'sku_code': sku,
                            'sku_name': sku,  # Using SKU as name, replace with actual name if available
                            'model': model.upper(),
                            'best_model': '✓' if is_best_model else ''
                        }
                        
                        # Get model forecast data
                        if model == forecast_data_for_sku['model']:
                            model_forecast = forecast_data_for_sku['forecast']
                        elif 'model_evaluation' in forecast_data_for_sku and 'all_models_forecasts' in forecast_data_for_sku['model_evaluation']:
                            model_forecast = forecast_data_for_sku['model_evaluation']['all_models_forecasts'].get(model, pd.Series())
                        else:
                            model_forecast = pd.Series()
                            
                        # Add historical/actual values (prefixed with 'Actual:')
                        for date, col_name in zip(historical_dates, historical_cols):
                            actual_col_name = f"Actual: {col_name}"
                            if date in sku_sales.index:
                                row[actual_col_name] = int(sku_sales.loc[date, 'quantity']) if not pd.isna(sku_sales.loc[date, 'quantity']) else 0
                            else:
                                row[actual_col_name] = 0
                        
                        # Add forecast values (prefixed with 'Forecast:') - ensuring dates match
                        for date, col_name in zip(forecast_dates, forecast_date_cols):
                            forecast_col_name = f"Forecast: {col_name}"
                            if date in model_forecast.index:
                                row[forecast_col_name] = int(model_forecast[date])
                            else:
                                row[forecast_col_name] = 0
                        
                        all_sku_data.append(row)
                
                # Create DataFrame from all data
                if all_sku_data:
                    all_sku_df = pd.DataFrame(all_sku_data)
                    
                    # Identify column groups for styling
                    all_cols = all_sku_df.columns.tolist()
                    info_cols = ['sku_code', 'sku_name', 'model', 'best_model']
                    actual_cols = [col for col in all_cols if col.startswith('Actual:')]
                    forecast_cols = [col for col in all_cols if col.startswith('Forecast:')]
                    
                    # Define a function for styling the dataframe
                    def highlight_data_columns(df):
                        # Create a DataFrame of styles
                        styles = pd.DataFrame('', index=df.index, columns=df.columns)
                        
                        # Apply background colors to different column types
                        for col in actual_cols:
                            styles[col] = 'background-color: #E8F4F9'  # Light blue for actual values
                        
                        for col in forecast_cols:
                            styles[col] = 'background-color: #FFF9C4'  # Light yellow for forecast values
                        
                        # Highlight best model rows
                        for i, val in enumerate(df['best_model']):
                            if val == '✓':
                                for col in df.columns:
                                    styles.iloc[i, df.columns.get_loc(col)] += '; font-weight: bold'
                        
                        return styles
                    
                    # Use styling to highlight data column types
                    st.dataframe(
                        all_sku_df.style.apply(highlight_data_columns, axis=None),
                        use_container_width=True,
                        height=500
                    )
                    
                    # Provide a download button for the table
                    csv_buffer = io.BytesIO()
                    all_sku_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Table as CSV",
                        data=csv_buffer,
                        file_name=f"sku_forecast_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data available for the selected SKUs.")
            else:
                st.warning("No historical training data available to construct the comprehensive data table.")
        else:
            st.warning("No forecast data available. Please run a forecast first.")
        with forecast_tabs[1]:
            # Model comparison visualization
            if 'model_evaluation' in forecast_data and forecast_data['model_evaluation']['metrics']:
                # Visual comparison of models
                st.subheader("Model Performance Comparison")

                # Use plot_model_comparison
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
                
        # The All SKUs Data Table tab was removed as the information is now integrated in the Forecast Chart tab

    # Forecast export
    st.header("Export Forecasts")

    # Prepare forecast data for export
    if st.button("Prepare Forecast Export"):
        with st.spinner("Preparing forecast data..."):
            # Create a DataFrame with all forecasts
            export_data = []

            for sku, forecast_data in st.session_state.forecasts.items():
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
                file_name=f"forecasts_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )

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
                index=default_index
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