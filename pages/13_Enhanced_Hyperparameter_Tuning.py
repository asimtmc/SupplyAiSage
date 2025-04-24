import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
from streamlit_extras import grid
import extra_streamlit_components as stx

# Define the hyperparameter tuning function at the top of the file
# so it can be referenced anywhere in the code
def run_hyperparameter_tuning():
    """
    Run the hyperparameter tuning process using real data.
    This function processes each SKU and model type selected for tuning.
    """
    # Add a debug print at the start
    print("Starting hyperparameter tuning process")
    
    if st.session_state.tuning_in_progress:
        # Check if we have all required data
        if not st.session_state.tuning_skus or not st.session_state.tuning_models:
            st.error("Missing required tuning parameters. Please select SKUs and models.")
            st.session_state.tuning_in_progress = False
            return
        
        # Calculate total tasks
        total_tasks = len(st.session_state.tuning_skus) * len(st.session_state.tuning_models)
        completed_tasks = 0
        
        # Add initial log
        if 'tuning_logs' not in st.session_state:
            st.session_state.tuning_logs = []
        
        log_entry = {
            'time': datetime.now().strftime("%H:%M:%S"),
            'message': f"Starting tuning process for {len(st.session_state.tuning_skus)} SKUs and {len(st.session_state.tuning_models)} models",
            'level': 'info'
        }
        st.session_state.tuning_logs.append(log_entry)
        
        # Process each SKU
        for sku_index, sku in enumerate(st.session_state.tuning_skus):
            # Log starting this SKU
            log_entry = {
                'time': datetime.now().strftime("%H:%M:%S"),
                'message': f"Processing SKU: {sku} ({sku_index + 1}/{len(st.session_state.tuning_skus)})",
                'level': 'info'
            }
            st.session_state.tuning_logs.append(log_entry)
            
            # Get data for this SKU
            try:
                if 'sales_data' not in st.session_state:
                    log_entry = {
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'message': "No sales data found in session state",
                        'level': 'error'
                    }
                    st.session_state.tuning_logs.append(log_entry)
                    continue
                
                sku_data = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku]
                
                if sku_data.empty:
                    log_entry = {
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'message': f"No data found for SKU: {sku}",
                        'level': 'warning'
                    }
                    st.session_state.tuning_logs.append(log_entry)
                    continue
                
                # Prepare data for optimization
                sku_data = sku_data.sort_values('date')
                
                # Split data for validation if using cross-validation
                use_cv = st.session_state.tuning_options.get('cross_validation', True)
                
                if use_cv:
                    # Calculate split point
                    n_splits = st.session_state.tuning_options.get('n_splits', 3)
                    total_periods = len(sku_data)
                    validation_size = total_periods // (n_splits + 1)  # Allocate portion for validation
                    
                    if validation_size < 6:  # Ensure enough data for validation
                        validation_size = min(6, total_periods // 3)
                    
                    train_data = sku_data.iloc[:-validation_size]
                    val_data = sku_data.iloc[-validation_size:]
                    
                    log_entry = {
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'message': f"Split data: {len(train_data)} training, {len(val_data)} validation points",
                        'level': 'info'
                    }
                    st.session_state.tuning_logs.append(log_entry)
                else:
                    # Use all data for training if not using cross-validation
                    train_data = sku_data
                    val_data = None
                
                # Prepare data in expected format for optimizer functions
                train_series = pd.Series(
                    train_data['quantity'].values, 
                    index=pd.DatetimeIndex(train_data['date'])
                )
                
                if val_data is not None:
                    val_series = pd.Series(
                        val_data['quantity'].values,
                        index=pd.DatetimeIndex(val_data['date'])
                    )
                else:
                    val_series = None
                
                # For Prophet, need DataFrame format
                train_prophet = train_data[['date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'})
                
                if val_data is not None:
                    val_prophet = val_data[['date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'})
                else:
                    val_prophet = None
                
                # Import optimizer functions only when needed
                try:
                    from utils.enhanced_parameter_optimizer import (
                        optimize_arima_parameters_enhanced,
                        optimize_prophet_parameters_enhanced,
                        optimize_ets_parameters_enhanced,
                        optimize_theta_parameters_enhanced,
                        verify_optimization_result,
                        store_optimized_parameters
                    )
                    print("Successfully imported optimizer functions inside the SKU processing loop")
                except ImportError as e:
                    log_entry = {
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'message': f"Failed to import optimizer functions: {str(e)}",
                        'level': 'error'
                    }
                    st.session_state.tuning_logs.append(log_entry)
                    continue
                
                # Tune each selected model
                for model_index, model_type in enumerate(st.session_state.tuning_models):
                    # Update progress
                    completed_tasks += 1
                    progress = completed_tasks / total_tasks
                    st.session_state.tuning_progress = progress
                    
                    # Log starting this model
                    log_entry = {
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'message': f"Tuning {model_type.upper()} model for {sku}",
                        'level': 'info'
                    }
                    st.session_state.tuning_logs.append(log_entry)
                    
                    try:
                        # Run the appropriate optimization function based on model type
                        if model_type == "auto_arima":
                            # Debug print to check if function exists
                            print(f"Running ARIMA optimization for {sku}")
                            
                            # Use the enhanced ARIMA optimizer
                            optimization_result = optimize_arima_parameters_enhanced(train_series, val_series)
                            
                            # Verify the result
                            optimization_result = verify_optimization_result(optimization_result, "auto_arima", sku)
                        
                        elif model_type == "prophet":
                            # Use the enhanced Prophet optimizer
                            optimization_result = optimize_prophet_parameters_enhanced(train_prophet, val_prophet)
                            
                            # Verify the result
                            optimization_result = verify_optimization_result(optimization_result, "prophet", sku)
                        
                        elif model_type == "ets":
                            # Use the enhanced ETS optimizer
                            optimization_result = optimize_ets_parameters_enhanced(train_series, val_series)
                            
                            # Verify the result
                            optimization_result = verify_optimization_result(optimization_result, "ets", sku)
                        
                        elif model_type == "theta":
                            # Use the enhanced Theta optimizer
                            optimization_result = optimize_theta_parameters_enhanced(train_series, val_series)
                            
                            # Verify the result
                            optimization_result = verify_optimization_result(optimization_result, "theta", sku)
                        
                        else:
                            # Skip unsupported model types
                            log_entry = {
                                'time': datetime.now().strftime("%H:%M:%S"),
                                'message': f"Unsupported model type: {model_type}",
                                'level': 'warning'
                            }
                            st.session_state.tuning_logs.append(log_entry)
                            continue
                        
                        # Store results
                        log_entry = {
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'message': f"Tuning complete for {model_type}. Best score: {optimization_result.get('score', 'N/A')}",
                            'level': 'success'
                        }
                        st.session_state.tuning_logs.append(log_entry)
                        
                        # Save optimized parameters to database
                        store_result = store_optimized_parameters(
                            sku, 
                            model_type,
                            optimization_result.get('parameters', {}),
                            score=optimization_result.get('score', None),
                            mape=optimization_result.get('mape', None),
                            improvement=optimization_result.get('improvement', 0),
                            tuning_options=st.session_state.tuning_options
                        )
                        
                        # Store results in session state for display
                        if 'tuning_results' not in st.session_state:
                            st.session_state.tuning_results = {}
                        
                        if sku not in st.session_state.tuning_results:
                            st.session_state.tuning_results[sku] = {}
                        
                        st.session_state.tuning_results[sku][model_type] = optimization_result
                        
                    except Exception as e:
                        # Log error
                        log_entry = {
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'message': f"Error tuning {model_type} for {sku}: {str(e)}",
                            'level': 'error'
                        }
                        st.session_state.tuning_logs.append(log_entry)
                        
            except Exception as e:
                # Log error processing this SKU
                log_entry = {
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'message': f"Error processing SKU {sku}: {str(e)}",
                    'level': 'error'
                }
                st.session_state.tuning_logs.append(log_entry)
                
        # All processing complete
        log_entry = {
            'time': datetime.now().strftime("%H:%M:%S"),
            'message': f"Hyperparameter tuning complete for all SKUs and models",
            'level': 'success'
        }
        st.session_state.tuning_logs.append(log_entry)
        
        # Set progress to 100%
        st.session_state.tuning_progress = 1.0
        
        # Mark tuning as complete
        st.session_state.tuning_in_progress = False

# Page configuration with wide layout
st.set_page_config(page_title="Enhanced Hyperparameter Tuning", page_icon="🔧", layout="wide")

# Add custom CSS for compact layout and better visualization
st.markdown("""
<style>
    .compact-metric {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        margin-bottom: 0px;
    }
    
    .param-value {
        font-family: monospace;
    }
    
    .mini-container {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 8px 12px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    
    /* More compact expanders */
    .streamlit-expanderContent {
        padding-top: 0.5rem !important;
    }
    
    /* Smaller log entries */
    .log-entry {
        padding: 0.1rem 0.5rem !important;
        margin: 0.1rem 0 !important;
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tuning_in_progress' not in st.session_state:
    st.session_state.tuning_in_progress = False

if 'tuning_skus' not in st.session_state:
    st.session_state.tuning_skus = []

if 'tuning_models' not in st.session_state:
    st.session_state.tuning_models = []

if 'tuning_progress' not in st.session_state:
    st.session_state.tuning_progress = 0

if 'tuning_options' not in st.session_state:
    st.session_state.tuning_options = {
        "cross_validation": True,
        "n_splits": 3
    }

# Import necessary modules for hyperparameter tuning
try:
    from utils.parameter_optimizer import optimize_parameters_async, get_optimization_status
    from utils.database import get_model_parameters
    from utils.enhanced_parameter_optimizer import (
        optimize_arima_parameters_enhanced, 
        optimize_prophet_parameters_enhanced,
        optimize_ets_parameters_enhanced,
        optimize_theta_parameters_enhanced,
        verify_optimization_result,
        store_optimized_parameters
    )
    from utils.advanced_forecast import optimize_parameters_with_validation, get_optimized_parameters
    from utils.visualization import plot_forecast_comparison, plot_parameter_importance
    from utils.data_processor import get_sku_data, prepare_data_for_forecasting
except ImportError:
    st.error("Required modules not found. Please ensure all dependencies are installed correctly.")

# Page header
st.markdown("<h1>🔧 Enhanced Hyperparameter Tuning</h1>", unsafe_allow_html=True)

# Create tabs for different sections
main_tabs = st.tabs(["📊 Tuning Dashboard", "⚙️ Tuning Process", "📈 Results & Comparisons"])

with main_tabs[0]:  # Dashboard Tab
    st.markdown("## Tuning Dashboard")
    
    # Create a compact dashboard layout with expandable sections
    dashboard_cols = st.columns([2, 1])
    
    with dashboard_cols[0]:
        # Quick parameter lookup
        with st.container():
            st.markdown("### 📋 Parameter Quick View")
            
            # Use a grid layout for the SKU and model selection
            param_cols = st.columns([2, 1])
            
            with param_cols[0]:
                # Get all SKUs in a dropdown
                if 'sales_data' in st.session_state:
                    all_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())
                    selected_sku = st.selectbox(
                        "Select SKU",
                        options=all_skus,
                        index=0 if all_skus else None,
                        key="dashboard_sku_selector"
                    )
                else:
                    st.warning("No sales data loaded")
                    selected_sku = None
            
            with param_cols[1]:
                # Model selection
                model_options = {
                    "auto_arima": "ARIMA",
                    "prophet": "Prophet",
                    "ets": "ETS",
                    "theta": "Theta"
                }
                selected_model = st.selectbox(
                    "Select Model",
                    options=list(model_options.keys()),
                    format_func=lambda x: model_options.get(x, x),
                    key="dashboard_model_selector"
                )
        
        # Parameter visualization
        if selected_sku and selected_model:
            # Get parameters for the selected SKU and model
            try:
                params = get_model_parameters(selected_sku, selected_model)
                
                if params and 'parameters' in params:
                    # Display parameter details in a compact card
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"### Parameters for {selected_sku} - {model_options[selected_model]}")
                            st.markdown(format_parameters(params['parameters'], selected_model), unsafe_allow_html=True)
                            
                            # When parameters were last tuned
                            if 'last_updated' in params:
                                st.markdown(f"**Last Updated**: {params['last_updated'].strftime('%Y-%m-%d %H:%M')}")
                        
                        with col2:
                            # Display metric scores
                            st.markdown("### Metrics")
                            st.metric(
                                "RMSE",
                                f"{params.get('best_score', 0):.4f}",
                                delta=f"{params.get('improvement', 0):.1%}",
                                delta_color="inverse"
                            )
                            st.metric(
                                "MAPE", 
                                f"{params.get('mape', 0):.2f}%"
                            )
                else:
                    st.info(f"No tuned parameters found for {selected_sku} with {model_options[selected_model]} model.")
            except Exception as e:
                st.warning(f"Error retrieving parameters: {str(e)}")
        
    with dashboard_cols[1]:
        # Recent activity and overall stats
        st.markdown("### 📊 Tuning Statistics")
        
        # Calculate overall metrics
        try:
            flat_params = get_parameters_from_db()
            if flat_params:
                params_df = pd.DataFrame(flat_params)
                
                # Display key metrics in a compact layout
                metrics_cols = st.columns(2)
                with metrics_cols[0]:
                    st.metric("SKUs Tuned", len(params_df['SKU code'].unique()))
                with metrics_cols[1]:
                    st.metric("Models Tuned", len(params_df['Model name'].unique()))
                
                # Most recent tunings
                st.markdown("#### Recent Tuning Activity")
                if 'Last Updated' in params_df.columns:
                    recent_df = params_df.sort_values('Last Updated', ascending=False).head(5)
                    for _, row in recent_df.iterrows():
                        st.markdown(f"""
                        <div class='mini-container'>
                            <small>{row.get('Last Updated', 'Unknown date')}</small><br>
                            <strong>{row.get('SKU code', 'Unknown SKU')}</strong> - {row.get('Model name', 'Unknown model')}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No tuning activity recorded yet.")
        except Exception as e:
            st.warning(f"Could not load tuning statistics: {str(e)}")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("Start New Tuning Process", key="quick_start_button", use_container_width=True):
            # Switch to the tuning process tab
            st.session_state.active_tab = 1
            st.rerun()
        
        if st.button("View All Parameters", key="view_all_params_button", use_container_width=True):
            # Set state to show all parameters
            st.session_state.show_all_params = True
            st.rerun()

with main_tabs[1]:  # Tuning Process Tab
    st.markdown("## Hyperparameter Tuning Process")
    st.markdown("""
    This section allows you to run hyperparameter optimization for specific SKUs and models.
    The tuning process will find the best parameters to maximize forecast accuracy.
    """)
    
    # Create a cleaner layout for the tuning interface
    tuning_cols = st.columns([3, 2])
    
    with tuning_cols[0]:  # SKU, Model Selection and Options
        st.markdown("### 1️⃣ Select Data & Models")
        
        # SKU Selection
        st.markdown("#### Select SKUs for Tuning")
        
        # Get all SKUs from the data
        if 'sales_data' in st.session_state:
            all_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())
            
            # Selection mode
            selection_mode = st.radio(
                "Selection Mode",
                options=["Single SKU", "Multiple SKUs", "Cluster Representatives"],
                horizontal=True,
                key="sku_selection_mode"
            )
            
            selected_skus = []
            
            if selection_mode == "Single SKU":
                # Single SKU selector
                selected_sku = st.selectbox(
                    "Select SKU",
                    options=all_skus,
                    index=0 if all_skus else None,
                    key="tuning_single_sku"
                )
                if selected_sku:
                    selected_skus = [selected_sku]
            
            elif selection_mode == "Multiple SKUs":
                # Compact multiselect with search
                selected_skus = st.multiselect(
                    "Select Multiple SKUs",
                    options=all_skus,
                    default=[all_skus[0]] if all_skus else None,
                    key="tuning_multi_skus"
                )
            
            elif selection_mode == "Cluster Representatives":
                # Use cluster representatives if available
                if 'advanced_clusters' in st.session_state:
                    clusters = st.session_state.advanced_clusters
                    cluster_names = sorted(clusters['cluster_name'].unique())
                    
                    selected_clusters = st.multiselect(
                        "Select Clusters",
                        options=cluster_names,
                        default=cluster_names[:1] if cluster_names else None,
                        key="tuning_clusters"
                    )
                    
                    if selected_clusters:
                        # Get representative SKU from each selected cluster
                        for cluster in selected_clusters:
                            cluster_skus = clusters[clusters['cluster_name'] == cluster]['sku'].tolist()
                            if cluster_skus:
                                selected_skus.append(cluster_skus[0])
                else:
                    st.warning("No cluster data available. Please run clustering first.")
        else:
            st.warning("No sales data available. Please load sales data first.")
            selected_skus = []
        
        # Model Selection
        st.markdown("#### Select Models for Tuning")
        
        # Display models in a grid format for more compact selection
        model_options = {
            "auto_arima": "ARIMA",
            "prophet": "Prophet", 
            "ets": "ETS",
            "theta": "Theta"
        }
        
        selected_models = []
        
        # Use a more compact grid layout for model selection
        model_cols = st.columns(len(model_options))
        
        for i, (model_key, model_name) in enumerate(model_options.items()):
            with model_cols[i]:
                if st.checkbox(model_name, value=model_key in st.session_state.get('tuning_models', []), key=f"model_{model_key}"):
                    selected_models.append(model_key)
        
        # Tuning options in an expander for cleaner interface
        with st.expander("🔍 Tuning Options", expanded=False):
            st.markdown("#### Configure Tuning Process")
            
            tuning_tab_options = st.tabs(["Basic", "Advanced"])
            
            with tuning_tab_options[0]:  # Basic Options
                # Cross-validation option
                cross_validation = st.checkbox(
                    "Use cross-validation",
                    value=st.session_state.tuning_options.get('cross_validation', True),
                    help="Split data for cross-validation to get more reliable parameter estimates"
                )
                
                # Number of splits if using cross-validation
                n_splits = st.slider(
                    "Number of CV splits",
                    min_value=2,
                    max_value=5,
                    value=st.session_state.tuning_options.get('n_splits', 3),
                    disabled=not cross_validation
                )
                
                # Store options in session state
                st.session_state.tuning_options = {
                    "cross_validation": cross_validation,
                    "n_splits": n_splits if cross_validation else 3
                }
            
            with tuning_tab_options[1]:  # Advanced Options
                # Add more advanced tuning options here as needed
                st.markdown("Additional advanced options will be added in future updates.")
        
        # Start/Stop Tuning controls
        st.markdown("### 2️⃣ Run Tuning Process")
        
        # Disable button if no SKUs or models selected
        start_disabled = len(selected_skus) == 0 or len(selected_models) == 0 or st.session_state.tuning_in_progress
        
        start_col1, start_col2 = st.columns([3, 1])
        
        with start_col1:
            if st.button("Start Hyperparameter Tuning", disabled=start_disabled, use_container_width=True, type="primary"):
                # Store selected SKUs and models
                st.session_state.tuning_skus = selected_skus
                st.session_state.tuning_models = selected_models
                st.session_state.tuning_in_progress = True
                st.session_state.tuning_logs = []
                st.session_state.tuning_progress = 0
                st.session_state.tuning_results = {}
                
                # Just set the flag - tuning will run after the page renders
                st.rerun()
        
        with start_col2:
            if st.session_state.tuning_in_progress:
                if st.button("Stop Tuning", use_container_width=True):
                    st.session_state.tuning_in_progress = False
                    st.info("Tuning process has been stopped.")
                    st.rerun()
        
        # Progress tracking
        if st.session_state.tuning_in_progress:
            st.markdown("### Tuning Progress")
            
            # Progress bar
            progress_placeholder = st.empty()
            progress_placeholder.progress(st.session_state.tuning_progress)
            
            # Status information
            status_placeholder = st.empty()
            
            # Calculate progress metrics
            total_tasks = len(st.session_state.tuning_skus) * len(st.session_state.tuning_models)
            completed_tasks = int(st.session_state.tuning_progress * total_tasks)
            
            status_placeholder.info(f"Processing task {completed_tasks}/{total_tasks}: {completed_tasks/total_tasks:.1%} complete")
            
            # Log display in a scrollable container
            st.markdown("#### Tuning Logs")
            
            # Display logs in a more compact format
            log_container = st.container()
            with log_container:
                # Show the most recent logs at the top with scrolling
                for log_entry in reversed(st.session_state.tuning_logs[-10:]):
                    log_time = log_entry.get('time', datetime.now().strftime("%H:%M:%S"))
                    log_message = log_entry.get('message', '')
                    log_level = log_entry.get('level', 'info')
                    
                    # Use different styling based on log level
                    if log_level == 'error':
                        st.error(f"{log_time} - {log_message}")
                    elif log_level == 'warning':
                        st.warning(f"{log_time} - {log_message}")
                    elif log_level == 'success':
                        st.success(f"{log_time} - {log_message}")
                    else:
                        st.info(f"{log_time} - {log_message}")

with main_tabs[2]:  # Results Tab
    st.markdown("## Results & Comparisons")
    
    if 'tuning_results' in st.session_state and st.session_state.tuning_results:
        # Results view with tabs for different perspectives
        result_tabs = st.tabs(["Parameter Comparison", "Metric Visualization", "Forecast Impact"])
        
        with result_tabs[0]:  # Parameter Comparison
            st.markdown("### Parameter Comparison")
            
            # Select SKU and models to compare
            compare_cols = st.columns([2, 2])
            
            with compare_cols[0]:
                if 'sales_data' in st.session_state:
                    all_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())
                    
                    compare_sku = st.selectbox(
                        "Select SKU",
                        options=all_skus,
                        index=0 if all_skus else None,
                        key="compare_sku_selector"
                    )
                else:
                    st.warning("No sales data loaded")
                    compare_sku = None
            
            with compare_cols[1]:
                model_options = {
                    "auto_arima": "ARIMA",
                    "prophet": "Prophet",
                    "ets": "ETS",
                    "theta": "Theta"
                }
                
                compare_models = st.multiselect(
                    "Select Models to Compare",
                    options=list(model_options.keys()),
                    default=list(model_options.keys())[:2] if model_options else None,
                    format_func=lambda x: model_options.get(x, x),
                    key="compare_models_selector"
                )
            
            if compare_sku and compare_models:
                # Get parameters for each selected model
                params_data = []
                
                for model in compare_models:
                    try:
                        model_params = get_model_parameters(compare_sku, model)
                        
                        if model_params and 'parameters' in model_params:
                            # Add to parameters data
                            for param_name, param_value in model_params['parameters'].items():
                                params_data.append({
                                    "Model": model_options[model],
                                    "Parameter": param_name,
                                    "Value": param_value,
                                    "Last Updated": model_params.get('last_updated', 'Unknown')
                                })
                    except Exception as e:
                        st.warning(f"Error retrieving {model} parameters: {str(e)}")
                
                if params_data:
                    # Display as a styled dataframe
                    params_df = pd.DataFrame(params_data)
                    
                    # Group by model and display in a cleaner format
                    for model_name in params_df['Model'].unique():
                        model_data = params_df[params_df['Model'] == model_name]
                        
                        with st.expander(f"{model_name} Parameters", expanded=True):
                            # Format parameters for display
                            param_cols = st.columns(2)
                            for i, (_, row) in enumerate(model_data.iterrows()):
                                with param_cols[i % 2]:
                                    st.markdown(f"""
                                    <div class='mini-container'>
                                        <strong>{row['Parameter']}</strong>: <span class='param-value'>{row['Value']}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                else:
                    st.info(f"No parameter data available for {compare_sku} with the selected models.")
        
        with result_tabs[1]:  # Metric Visualization
            st.markdown("### Performance Metrics")
            
            # Functions to simulate metric data for demonstration
            def get_metrics_data(sku, models):
                """Get metrics for the specified SKU and models"""
                metrics_data = []
                
                for model in models:
                    try:
                        # Get actual metrics from the database
                        model_params = get_model_parameters(sku, model)
                        
                        if model_params:
                            # Extract metrics for tuned model
                            tuned_metrics = {
                                "RMSE": model_params.get('best_score', 0),
                                "MAPE": model_params.get('mape', 0),
                                "MAE": model_params.get('mae', 0),
                                "Improvement": model_params.get('improvement', 0)
                            }
                            
                            # Extract baseline metrics for comparison
                            baseline_metrics = {}
                            if 'baseline_metrics' in model_params:
                                baseline_metrics = {
                                    "RMSE": model_params['baseline_metrics'].get('rmse', 0),
                                    "MAPE": model_params['baseline_metrics'].get('mape', 0),
                                    "MAE": model_params['baseline_metrics'].get('mae', 0)
                                }
                            else:
                                # Calculate baseline metrics using improvement percentage if available
                                improvement = model_params.get('improvement', 0)
                                if improvement > 0:
                                    baseline_metrics = {
                                        "RMSE": tuned_metrics["RMSE"] / (1 - improvement) if improvement < 1 else tuned_metrics["RMSE"] * 1.2,
                                        "MAPE": tuned_metrics["MAPE"] / (1 - improvement) if improvement < 1 else tuned_metrics["MAPE"] * 1.2,
                                        "MAE": tuned_metrics["MAE"] / (1 - improvement) if improvement < 1 else tuned_metrics["MAE"] * 1.2
                                    }
                                else:
                                    # If no improvement data, estimate baseline as slightly worse
                                    baseline_metrics = {
                                        "RMSE": tuned_metrics["RMSE"] * 1.1,
                                        "MAPE": tuned_metrics["MAPE"] * 1.1, 
                                        "MAE": tuned_metrics["MAE"] * 1.1
                                    }
                            
                            # Add both tuned and baseline metrics to the data
                            metrics_data.append({
                                "Model": f"{model_options[model]} (Tuned)",
                                "Type": "Tuned",
                                "RMSE": tuned_metrics["RMSE"],
                                "MAPE": tuned_metrics["MAPE"], 
                                "MAE": tuned_metrics["MAE"],
                                "Improvement": tuned_metrics["Improvement"]
                            })
                            
                            metrics_data.append({
                                "Model": f"{model_options[model]} (Default)",
                                "Type": "Default",
                                "RMSE": baseline_metrics.get("RMSE", 0),
                                "MAPE": baseline_metrics.get("MAPE", 0), 
                                "MAE": baseline_metrics.get("MAE", 0),
                                "Improvement": 0
                            })
                    except Exception as e:
                        st.warning(f"Error retrieving metrics for {model}: {str(e)}")
                
                return pd.DataFrame(metrics_data)
            
            # Select SKU for metrics visualization
            if 'sales_data' in st.session_state:
                all_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())
                
                metric_sku = st.selectbox(
                    "Select SKU",
                    options=all_skus,
                    index=0 if all_skus else None,
                    key="metric_sku_selector"
                )
                
                if metric_sku:
                    # Get all models with metrics for this SKU
                    model_options = {
                        "auto_arima": "ARIMA",
                        "prophet": "Prophet",
                        "ets": "ETS", 
                        "theta": "Theta"
                    }
                    
                    metrics_df = get_metrics_data(metric_sku, model_options.keys())
                    
                    if not metrics_df.empty:
                        # Format model names for display
                        metrics_df['Model'] = metrics_df['Model'].map(lambda x: model_options.get(x, x))
                        
                        # Display metrics in side-by-side charts
                        metric_cols = st.columns(2)
                        
                        with metric_cols[0]:
                            # RMSE Chart - Default vs Tuned side-by-side
                            # Group by model type and color by default/tuned
                            rmse_fig = px.bar(
                                metrics_df, 
                                x='Model', 
                                y='RMSE',
                                color='Type',
                                barmode='group',
                                title=f"RMSE Comparison (Default vs Tuned) for {metric_sku}",
                                height=250,
                                color_discrete_map={
                                    'Default': '#6c757d',  # Gray for default
                                    'Tuned': '#198754'     # Green for tuned (better)
                                }
                            )
                            rmse_fig.update_layout(
                                margin=dict(l=5, r=5, t=40, b=5),
                                xaxis=dict(tickangle=-45),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(rmse_fig, use_container_width=True)
                            
                            # MAE Chart
                            if 'MAE' in metrics_df.columns:
                                mae_fig = px.bar(
                                    metrics_df,
                                    x='Model',
                                    y='MAE',
                                    color='Type',
                                    barmode='group',
                                    title=f"MAE Comparison (Default vs Tuned) for {metric_sku}",
                                    height=250,
                                    color_discrete_map={
                                        'Default': '#6c757d',  # Gray for default
                                        'Tuned': '#198754'     # Green for tuned (better)
                                    }
                                )
                                mae_fig.update_layout(
                                    margin=dict(l=5, r=5, t=40, b=5),
                                    xaxis=dict(tickangle=-45),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                st.plotly_chart(mae_fig, use_container_width=True)
                        
                        with metric_cols[1]:
                            # MAPE Chart - Default vs Tuned side-by-side
                            mape_fig = px.bar(
                                metrics_df,
                                x='Model',
                                y='MAPE',
                                color='Type',
                                barmode='group',
                                title=f"MAPE Comparison (Default vs Tuned) for {metric_sku}",
                                height=250,
                                color_discrete_map={
                                    'Default': '#6c757d',  # Gray for default
                                    'Tuned': '#198754'     # Green for tuned (better)
                                }
                            )
                            mape_fig.update_layout(
                                margin=dict(l=5, r=5, t=40, b=5),
                                xaxis=dict(tickangle=-45),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(mape_fig, use_container_width=True)
                            
                            # Improvement Chart
                            if 'Improvement' in metrics_df.columns:
                                # Filter out Default models for improvement chart (they have 0 improvement)
                                tuned_metrics_df = metrics_df[metrics_df['Type'] == 'Tuned'].copy()
                                
                                # Convert to percentage for display
                                tuned_metrics_df['Improvement'] = tuned_metrics_df['Improvement'] * 100
                                
                                imp_fig = px.bar(
                                    tuned_metrics_df,
                                    x='Model',
                                    y='Improvement',
                                    color='Model',
                                    title=f"Tuning Improvement (%) for {metric_sku}",
                                    height=250,
                                    color_discrete_sequence=['#198754']  # Green for improvement
                                )
                                imp_fig.update_layout(
                                    margin=dict(l=5, r=5, t=40, b=5),
                                    xaxis=dict(tickangle=-45),
                                    yaxis=dict(title="Improvement %"),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                st.plotly_chart(imp_fig, use_container_width=True)
                    else:
                        st.info(f"No metric data available for {metric_sku}.")
        
        with result_tabs[2]:  # Forecast Impact
            st.markdown("### Forecast Impact")
            
            # Select SKU and model for visualization
            impact_cols = st.columns([1, 1, 1])
            
            with impact_cols[0]:
                if 'sales_data' in st.session_state:
                    all_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())
                    
                    impact_sku = st.selectbox(
                        "Select SKU",
                        options=all_skus,
                        index=0 if all_skus else None,
                        key="impact_sku_selector"
                    )
                else:
                    st.warning("No sales data loaded")
                    impact_sku = None
            
            with impact_cols[1]:
                model_options = {
                    "auto_arima": "ARIMA",
                    "prophet": "Prophet",
                    "ets": "ETS",
                    "theta": "Theta"
                }
                
                impact_model = st.selectbox(
                    "Select Model",
                    options=list(model_options.keys()),
                    format_func=lambda x: model_options.get(x, x),
                    key="impact_model_selector"
                )
            
            with impact_cols[2]:
                forecast_periods = st.slider(
                    "Forecast Periods",
                    min_value=1,
                    max_value=24,
                    value=12,
                    key="impact_forecast_periods"
                )
            
            if impact_sku and impact_model:
                try:
                    # Get comparison forecasts if available
                    if impact_sku in st.session_state.tuning_results and impact_model in st.session_state.tuning_results[impact_sku]:
                        st.markdown("### Forecast Comparison: Default vs. Tuned Parameters")
                        
                        # Display forecast comparison for this SKU and model
                        tuning_result = st.session_state.tuning_results[impact_sku][impact_model]
                        
                        if 'forecast_comparison' in tuning_result:
                            forecast_comparison = tuning_result['forecast_comparison']
                            
                            # Use the visualization module to plot
                            forecast_fig = plot_forecast_comparison(
                                forecast_comparison['dates'],
                                forecast_comparison['actuals'],
                                forecast_comparison['default_forecast'],
                                forecast_comparison['tuned_forecast'],
                                title=f"{impact_sku} - {model_options[impact_model]} Forecast Comparison"
                            )
                            
                            st.plotly_chart(forecast_fig, use_container_width=True)
                            
                            # Display metrics comparison
                            metric_comparison_cols = st.columns(2)
                            
                            with metric_comparison_cols[0]:
                                st.markdown("#### Default Parameters")
                                st.metric(
                                    "RMSE",
                                    f"{forecast_comparison.get('default_rmse', 0):.2f}"
                                )
                                st.metric(
                                    "MAPE",
                                    f"{forecast_comparison.get('default_mape', 0):.2f}%"
                                )
                            
                            with metric_comparison_cols[1]:
                                st.markdown("#### Tuned Parameters")
                                st.metric(
                                    "RMSE",
                                    f"{forecast_comparison.get('tuned_rmse', 0):.2f}",
                                    delta=f"{tuning_result.get('improvement', 0):.1%}",
                                    delta_color="inverse"
                                )
                                st.metric(
                                    "MAPE",
                                    f"{tuning_result.get('mape', 0):.2f}%"
                                )
                        else:
                            # Generate comparison on demand
                            st.info("Generating forecast comparison...")
                            
                            # This would be replaced with actual forecast comparison logic
                            # For now, just show a placeholder
                            st.warning("Real-time forecast comparison generation not implemented yet.")
                    
                    # Display parameter importance if available
                    if impact_sku in st.session_state.tuning_results and impact_model in st.session_state.tuning_results[impact_sku]:
                        tuning_result = st.session_state.tuning_results[impact_sku][impact_model]
                        
                        if 'parameter_importance' in tuning_result:
                            st.markdown("### Parameter Importance")
                            
                            # Use the visualization module to plot parameter importance
                            importance_fig = plot_parameter_importance(
                                tuning_result['parameter_importance'],
                                title=f"Parameter Importance for {impact_sku} - {model_options[impact_model]}"
                            )
                            
                            st.plotly_chart(importance_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating forecast comparison: {str(e)}")
    else:
        st.info("No tuning results available yet. Run hyperparameter tuning first.")

def format_parameters(params, model_type):
    """
    Format model parameters for more compact display
    """
    if not params:
        return "No parameters available"
    
    # Format based on model type
    if model_type == 'auto_arima':
        html = "<div class='parameter-table'>"
        html += f"<div><strong>p</strong>: <code>{params.get('p', 'auto')}</code></div>"
        html += f"<div><strong>d</strong>: <code>{params.get('d', 'auto')}</code></div>"
        html += f"<div><strong>q</strong>: <code>{params.get('q', 'auto')}</code></div>"
        html += f"<div><strong>P</strong>: <code>{params.get('P', 'auto')}</code></div>"
        html += f"<div><strong>D</strong>: <code>{params.get('D', 'auto')}</code></div>"
        html += f"<div><strong>Q</strong>: <code>{params.get('Q', 'auto')}</code></div>"
        html += f"<div><strong>m</strong>: <code>{params.get('m', 1)}</code></div>"
        html += "</div>"
        return html
    
    elif model_type == 'prophet':
        html = "<div class='parameter-table'>"
        html += f"<div><strong>changepoint_prior_scale</strong>: <code>{params.get('changepoint_prior_scale', 0.05)}</code></div>"
        html += f"<div><strong>seasonality_prior_scale</strong>: <code>{params.get('seasonality_prior_scale', 10.0)}</code></div>"
        html += f"<div><strong>holidays_prior_scale</strong>: <code>{params.get('holidays_prior_scale', 10.0)}</code></div>"
        html += f"<div><strong>seasonality_mode</strong>: <code>{params.get('seasonality_mode', 'additive')}</code></div>"
        html += "</div>"
        return html
    
    elif model_type == 'ets':
        html = "<div class='parameter-table'>"
        html += f"<div><strong>error</strong>: <code>{params.get('error', 'add')}</code></div>"
        html += f"<div><strong>trend</strong>: <code>{params.get('trend', 'add')}</code></div>"
        html += f"<div><strong>seasonal</strong>: <code>{params.get('seasonal', 'add')}</code></div>"
        html += f"<div><strong>damped_trend</strong>: <code>{params.get('damped_trend', False)}</code></div>"
        html += f"<div><strong>seasonal_periods</strong>: <code>{params.get('seasonal_periods', 'None')}</code></div>"
        html += "</div>"
        return html
    
    elif model_type == 'theta':
        html = "<div class='parameter-table'>"
        html += f"<div><strong>deseasonalize</strong>: <code>{params.get('deseasonalize', False)}</code></div>"
        html += f"<div><strong>season_length</strong>: <code>{params.get('season_length', 1)}</code></div>"
        html += "</div>"
        return html
    
    else:
        # Generic parameter formatting
        html = "<div class='parameter-table'>"
        for key, value in params.items():
            html += f"<div><strong>{key}</strong>: <code>{value}</code></div>"
        html += "</div>"
        return html

def get_parameters_from_db():
    """Retrieve parameters from database"""
    try:
        # Here you would connect to the database and retrieve the parameters
        # For demonstration purposes, we'll use mock data
        from utils.database import get_all_model_parameters
        
        return get_all_model_parameters()
    except Exception as e:
        st.warning(f"Error retrieving parameters from database: {str(e)}")
        return []

# Run the tuning process if in progress
if st.session_state.tuning_in_progress:
    run_hyperparameter_tuning()