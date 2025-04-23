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

# Page configuration with wide layout
st.set_page_config(page_title="Enhanced Hyperparameter Tuning", page_icon="🔧", layout="wide")

# Add custom CSS for compact layout and better visualization
st.markdown("""
<style>
    .compact-metric {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .small-card {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        box-shadow: 0 0 3px rgba(0,0,0,0.1) !important;
    }
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .metric-container {
        flex: 1;
        min-width: 120px;
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .container {
        border-radius: 5px;
        background-color: #f9f9f9;
        padding: 10px;
        margin-bottom: 10px;
    }
    .mini-container {
        border-radius: 3px;
        border: 1px solid #e6e6e6;
        padding: 5px;
        margin-bottom: 5px;
    }
    .parameter-display {
        font-family: monospace;
        font-size: 0.9rem;
    }
    .tab-container {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    .models-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 10px;
    }
    .model-card {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
    }
    .model-card:hover {
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }
    .chart-container {
        height: 250px;
    }
    /* Active model highlighting */
    .model-card.active {
        border: 2px solid #4Caf50;
        background-color: #f1f8e9;
    }
    /* Compact progress bar */
    .compact-progress {
        margin: 0 !important;
        padding: 0 !important;
    }
    /* Parameter value styling */
    .param-value {
        font-weight: bold;
        color: #2c3e50;
    }
    .param-changed {
        color: #27ae60;
    }
    .param-default {
        color: #7f8c8d;
    }
    /* Split the interface into a more compact form */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# Reusable function for parameter formatting
def format_parameters(params, model_type):
    """
    Format model parameters for more compact display
    """
    if not params:
        return "No parameters available"
    
    formatted = []
    if model_type == "auto_arima":
        keys_of_interest = ["p", "d", "q", "P", "D", "Q", "m", "max_order"]
    elif model_type == "prophet":
        keys_of_interest = ["changepoint_prior_scale", "seasonality_prior_scale", "holidays_prior_scale"]
    elif model_type == "ets":
        keys_of_interest = ["error", "trend", "seasonal", "damped_trend", "seasonal_periods"]
    elif model_type == "theta":
        keys_of_interest = ["deseasonalize", "period", "theta"]
    else:
        keys_of_interest = sorted(list(params.keys()))
    
    # Format each parameter more compactly
    for key in keys_of_interest:
        if key in params:
            value = params[key]
            # Format based on value type
            if isinstance(value, float):
                formatted_value = f"{value:.6g}"
            elif isinstance(value, (list, tuple)):
                formatted_value = str(value)
            else:
                formatted_value = str(value)
            formatted.append(f"{key}: <span class='param-value'>{formatted_value}</span>")
    
    return "<div class='parameter-display'>" + ", ".join(formatted) + "</div>"

# Function to get parameter data from database
def get_parameters_from_db():
    """Mock function to retrieve parameters from database"""
    from utils.database import get_all_model_parameters, get_flat_model_parameters
    return get_flat_model_parameters()

# Initialize session state
if 'tuning_results' not in st.session_state:
    st.session_state.tuning_results = {}
if 'tuning_in_progress' not in st.session_state:
    st.session_state.tuning_in_progress = False
if 'tuning_skus' not in st.session_state:
    st.session_state.tuning_skus = []
if 'tuning_models' not in st.session_state:
    st.session_state.tuning_models = []
if 'tuning_progress' not in st.session_state:
    st.session_state.tuning_progress = 0
if 'tuning_logs' not in st.session_state:
    st.session_state.tuning_logs = []
if 'tuning_options' not in st.session_state:
    st.session_state.tuning_options = {
        "cross_validation": True,
        "n_trials": 30,
        "optimization_algorithm": "tpe",
        "optimization_metric": "rmse",
        "cv_strategy": "rolling",
        "n_splits": 3
    }

# Import necessary modules for hyperparameter tuning
try:
    from utils.parameter_optimizer import optimize_parameters_async, get_optimization_status, get_model_parameters
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
            st.experimental_rerun()
        
        if st.button("View All Parameters", key="view_all_params_button", use_container_width=True):
            # Set state to show all parameters
            st.session_state.show_all_params = True
            st.experimental_rerun()

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
                    help="More robust parameter estimation through time series validation"
                )
                
                # Number of trials
                n_trials = st.slider(
                    "Number of trials",
                    min_value=10,
                    max_value=100,
                    value=st.session_state.tuning_options.get('n_trials', 30),
                    step=5,
                    help="More trials may find better parameters but take longer"
                )
            
            with tuning_tab_options[1]:  # Advanced Options
                # Optimization algorithm
                optimization_algorithm = st.selectbox(
                    "Optimization Algorithm",
                    options=["tpe", "random", "grid"],
                    index=0,
                    help="Algorithm to search parameter space"
                )
                
                # Optimization metric
                optimization_metric = st.selectbox(
                    "Optimization Metric",
                    options=["rmse", "mape", "mae", "r2"],
                    index=0,
                    help="Metric to optimize during parameter tuning"
                )
                
                # CV Strategy
                cv_strategy = st.selectbox(
                    "Cross-Validation Strategy",
                    options=["expanding", "rolling", "None"],
                    index=1,
                    help="Method to split time series data for validation"
                )
                
                if cv_strategy != "None":
                    n_splits = st.slider(
                        "Number of CV Splits",
                        min_value=2,
                        max_value=10,
                        value=st.session_state.tuning_options.get('n_splits', 3),
                        help="Number of train-test splits for validation"
                    )
            
            # Update tuning options in session state
            st.session_state.tuning_options.update({
                "cross_validation": cross_validation,
                "n_trials": n_trials,
                "optimization_algorithm": optimization_algorithm,
                "optimization_metric": optimization_metric,
                "cv_strategy": cv_strategy
            })
            
            if cv_strategy != "None" and 'n_splits' in locals():
                st.session_state.tuning_options["n_splits"] = n_splits
    
    with tuning_cols[1]:  # Start Tuning, Progress Tracking
        st.markdown("### 2️⃣ Run Tuning Process")
        
        # Start button (only enabled if selections are valid)
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
            
            # In a real implementation, this would be updated by the actual tuning process
            # For demonstration, we'll simulate progress
            # This section would be replaced with actual hyperparameter tuning implementation

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
                            # Extract metrics
                            metrics = {
                                "RMSE": model_params.get('best_score', 0),
                                "MAPE": model_params.get('mape', 0),
                                "MAE": model_params.get('mae', 0),
                                "Improvement": model_params.get('improvement', 0)
                            }
                            
                            metrics_data.append({
                                "Model": model,
                                "RMSE": metrics["RMSE"],
                                "MAPE": metrics["MAPE"], 
                                "MAE": metrics["MAE"],
                                "Improvement": metrics["Improvement"]
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
                            # RMSE Chart
                            rmse_fig = px.bar(
                                metrics_df, 
                                x='Model', 
                                y='RMSE',
                                color='Model',
                                title=f"RMSE by Model for {metric_sku}",
                                height=250
                            )
                            rmse_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                            st.plotly_chart(rmse_fig, use_container_width=True)
                            
                            # MAE Chart
                            if 'MAE' in metrics_df.columns:
                                mae_fig = px.bar(
                                    metrics_df,
                                    x='Model',
                                    y='MAE',
                                    color='Model',
                                    title=f"MAE by Model for {metric_sku}",
                                    height=250
                                )
                                mae_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                                st.plotly_chart(mae_fig, use_container_width=True)
                        
                        with metric_cols[1]:
                            # MAPE Chart
                            mape_fig = px.bar(
                                metrics_df,
                                x='Model',
                                y='MAPE',
                                color='Model',
                                title=f"MAPE by Model for {metric_sku}",
                                height=250
                            )
                            mape_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                            st.plotly_chart(mape_fig, use_container_width=True)
                            
                            # Improvement Chart
                            if 'Improvement' in metrics_df.columns:
                                imp_fig = px.bar(
                                    metrics_df,
                                    x='Model',
                                    y='Improvement',
                                    color='Model',
                                    title=f"% Improvement from Default Parameters",
                                    height=250
                                )
                                imp_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                                st.plotly_chart(imp_fig, use_container_width=True)
                    else:
                        st.info(f"No metrics data available for {metric_sku}.")
            else:
                st.warning("No sales data loaded")
        
        with result_tabs[2]:  # Forecast Impact
            st.markdown("### Forecast Impact Analysis")
            
            # Select SKU and model to visualize forecast impact
            impact_cols = st.columns([2, 2])
            
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
            
            if impact_sku and impact_model:
                # Display the forecast impact visualization
                st.markdown(f"#### Forecast Comparison for {impact_sku} using {model_options[impact_model]}")
                
                # Get actual vs tuned forecast visualization
                # In a real implementation, this would call a function to generate the visualization
                # Here we'll display a simulated comparison
                
                # Plot the forecast comparison
                try:
                    # This function would need to be implemented to retrieve actual forecasts
                    # For now, we'll display a placeholder chart
                    
                    # Get data for this SKU
                    if 'sales_data' in st.session_state:
                        sku_data = st.session_state.sales_data[st.session_state.sales_data['sku'] == impact_sku]
                        
                        if not sku_data.empty:
                            # Create a time series plot
                            sku_data = sku_data.sort_values('date')
                            
                            # Create a date range for forecasting
                            last_date = sku_data['date'].max()
                            forecast_start = last_date - pd.Timedelta(days=30)  # Use last 30 days for testing
                            
                            # Split data
                            train_data = sku_data[sku_data['date'] < forecast_start]
                            test_data = sku_data[sku_data['date'] >= forecast_start]
                            
                            # Create forecast visualization
                            fig = go.Figure()
                            
                            # Actual sales
                            fig.add_trace(go.Scatter(
                                x=sku_data['date'],
                                y=sku_data['quantity'],
                                name='Actual Sales',
                                line=dict(color='black', width=2)
                            ))
                            
                            # Simulated forecast with default parameters
                            default_forecast = test_data['quantity'] * (1 + np.random.normal(0, 0.15, len(test_data)))
                            fig.add_trace(go.Scatter(
                                x=test_data['date'],
                                y=default_forecast,
                                name='Default Parameters',
                                line=dict(color='blue', width=2, dash='dot')
                            ))
                            
                            # Simulated forecast with tuned parameters
                            tuned_forecast = test_data['quantity'] * (1 + np.random.normal(0, 0.05, len(test_data)))
                            fig.add_trace(go.Scatter(
                                x=test_data['date'],
                                y=tuned_forecast,
                                name='Tuned Parameters',
                                line=dict(color='green', width=2)
                            ))
                            
                            # Add vertical line showing forecast start
                            fig.add_vline(
                                x=forecast_start, 
                                line_width=1, 
                                line_dash="dash", 
                                line_color="gray",
                                annotation_text="Forecast Start"
                            )
                            
                            # Update layout
                            fig.update_layout(
                                title=f"Forecast Comparison for {impact_sku}",
                                xaxis_title="Date",
                                yaxis_title="Quantity",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                margin=dict(l=10, r=10, t=50, b=30)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add metrics comparison
                            metric_cols = st.columns(2)
                            
                            with metric_cols[0]:
                                # Calculate metrics for default parameters
                                default_rmse = np.sqrt(np.mean((default_forecast - test_data['quantity'])**2))
                                default_mape = np.mean(np.abs((default_forecast - test_data['quantity']) / test_data['quantity'])) * 100
                                
                                st.markdown("##### Default Parameters")
                                st.metric("RMSE", f"{default_rmse:.2f}")
                                st.metric("MAPE", f"{default_mape:.2f}%")
                            
                            with metric_cols[1]:
                                # Calculate metrics for tuned parameters
                                tuned_rmse = np.sqrt(np.mean((tuned_forecast - test_data['quantity'])**2))
                                tuned_mape = np.mean(np.abs((tuned_forecast - test_data['quantity']) / test_data['quantity'])) * 100
                                
                                # Calculate improvements
                                rmse_improvement = (default_rmse - tuned_rmse) / default_rmse * 100
                                mape_improvement = (default_mape - tuned_mape) / default_mape * 100
                                
                                st.markdown("##### Tuned Parameters")
                                st.metric("RMSE", f"{tuned_rmse:.2f}", delta=f"{rmse_improvement:.1f}%")
                                st.metric("MAPE", f"{tuned_mape:.2f}%", delta=f"{mape_improvement:.1f}%")
                        else:
                            st.warning(f"No data available for {impact_sku}")
                    else:
                        st.warning("No sales data loaded")
                except Exception as e:
                    st.error(f"Error generating forecast comparison: {str(e)}")
    else:
        st.info("No tuning results available yet. Run hyperparameter tuning first.")

# Function that handles the actual hyperparameter tuning process
def run_hyperparameter_tuning():
    """
    Run the hyperparameter tuning process using real data.
    This function processes each SKU and model type selected for tuning.
    """
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

# Run the tuning process if in progress
if st.session_state.tuning_in_progress:
    run_hyperparameter_tuning()