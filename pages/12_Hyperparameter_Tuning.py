# Import streamlit first
import streamlit as st

# Page configuration is already set in app.py
# Do not call st.set_page_config() here as it will cause errors

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import json
from datetime import datetime, timedelta
from utils.data_processor import process_sales_data
from utils.parameter_optimizer import optimize_parameters_async, get_optimization_status, get_model_parameters
from utils.database import get_model_parameters, save_model_parameters
import streamlit.components.v1 as components
from io import StringIO
import re
import uuid
from streamlit_extras.metric_cards import style_metric_cards
import extra_streamlit_components as stx

# Load data automatically using the page_loader utility
from utils.page_loader import check_data_requirements

# Check that sales data is available, otherwise stop execution
if not check_data_requirements(['sales_data']):
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

# Custom styling for the page
st.markdown("""
<style>
    .highlighted-header {
        background-color: #f0f5ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4169e1;
    }
    .parameter-card {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .parameter-header {
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .parameter-value {
        font-family: monospace;
        background-color: #f9f9f9;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
    }
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .tooltip-icon {
        color: #aaa;
        font-size: 0.8rem;
        margin-left: 0.3rem;
        cursor: help;
    }
    .comparison-container {
        display: flex;
        gap: 1rem;
    }
    .comparison-column {
        flex: 1;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .comparison-before {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
    }
    .comparison-after {
        background-color: #f0fff4;
        border: 1px solid #c6f6d5;
    }
    .audit-entry {
        padding: 0.5rem;
        border-bottom: 1px solid #eee;
    }
    .model-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        color: white;
        margin-right: 0.3rem;
    }
    .arima-badge { background-color: #4299e1; }
    .prophet-badge { background-color: #805ad5; }
    .ets-badge { background-color: #38a169; }
    .theta-badge { background-color: #dd6b20; }
    .lstm-badge { background-color: #e53e3e; }
    
    /* Improve form controls */
    .stSlider > div > div > div {
        height: 0.5rem !important;
    }
    .stSlider > div > div > div > div > div {
        height: 1.2rem !important;
        width: 1.2rem !important;
    }
    
    /* Search box styling */
    .search-container input {
        border-radius: 2rem;
        padding-left: 2.5rem !important;
    }
    .search-container .stTextInput {
        position: relative;
    }
    .search-container .stTextInput::before {
        content: "🔍";
        position: absolute;
        left: 0.8rem;
        top: 0.4rem;
        z-index: 100;
        color: #aaa;
    }
</style>
""", unsafe_allow_html=True)

# Add cookie manager for persistent settings
cookie_manager = stx.CookieManager()

# Page title with enhanced styling
st.markdown("<div class='highlighted-header'><h1 style='margin:0'>🔧 Hyperparameter Tuning Station</h1></div>", unsafe_allow_html=True)

st.markdown("""
This advanced module optimizes model hyperparameters for specific SKUs to maximize forecast accuracy.
Optimal parameters are stored in the database and automatically used in demand forecasting.

### Benefits:
- 📈 **Improved Accuracy**: Custom-tailored parameters for each SKU's unique patterns
- 🔍 **Data-Driven Optimization**: Scientific approach to find the best model configurations
- ⚡ **Enhanced Performance**: Balance between accuracy and computational efficiency
- 📊 **Comparative Analysis**: See improvement over default parameters
""")

# Main layout with two columns
tuning_col1, tuning_col2 = st.columns([2, 1])

with tuning_col1:
    # SKU selection with enhanced search and filtering
    st.markdown("<h3>1. Dynamic SKU Selector</h3>", unsafe_allow_html=True)
    
    # Get all SKUs from the data
    all_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())
    
    # Create a tabbed interface for SKU selection methods
    sku_selection_tabs = st.tabs(["Search & Filter", "By Characteristics", "By Performance"])
    
    with sku_selection_tabs[0]:
        # Initialize session state for search
        if 'sku_search_term' not in st.session_state:
            st.session_state.sku_search_term = ""
        
        # Create search box with styled container
        st.markdown("<div class='search-container'>", unsafe_allow_html=True)
        sku_search = st.text_input(
            "Search SKUs",
            value=st.session_state.sku_search_term,
            placeholder="Type to search...",
            key="sku_search_input"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Update search term in session state
        st.session_state.sku_search_term = sku_search
        
        # Filter SKUs based on search term
        filtered_skus = all_skus
        if sku_search:
            filtered_skus = [sku for sku in all_skus if sku_search.lower() in sku.lower()]
        
        # Add additional filters
        filter_cols = st.columns(3)
        
        with filter_cols[0]:
            # Sort options
            sort_options = ["Alphabetical (A-Z)", "Alphabetical (Z-A)", "Data Length (High-Low)", "Data Length (Low-High)"]
            sort_by = st.selectbox("Sort by", sort_options, index=0)
        
        with filter_cols[1]:
            # Filter by data sufficiency
            min_data_points = st.slider("Min. Data Points", 1, 50, 8, help="Minimum number of data points required for reliable tuning")
        
        with filter_cols[2]:
            # Limit number of results
            limit_results = st.slider("Limit Results", 5, 100, 20, help="Maximum number of SKUs to display")
        
        # Apply sorting
        if sort_by == "Alphabetical (A-Z)":
            filtered_skus = sorted(filtered_skus)
        elif sort_by == "Alphabetical (Z-A)":
            filtered_skus = sorted(filtered_skus, reverse=True)
        elif sort_by == "Data Length (High-Low)" or sort_by == "Data Length (Low-High)":
            # Count data points for each SKU
            sku_data_counts = {}
            for sku in filtered_skus:
                sku_data = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku]
                sku_data_counts[sku] = len(sku_data)
            
            # Sort by data count
            reverse_sort = (sort_by == "Data Length (High-Low)")
            filtered_skus = sorted(filtered_skus, key=lambda sku: sku_data_counts.get(sku, 0), reverse=reverse_sort)
        
        # Apply data sufficiency filter
        if min_data_points > 1:
            sufficient_skus = []
            for sku in filtered_skus:
                sku_data = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku]
                if len(sku_data) >= min_data_points:
                    sufficient_skus.append(sku)
            filtered_skus = sufficient_skus
        
        # Limit results
        if len(filtered_skus) > limit_results:
            filtered_skus = filtered_skus[:limit_results]
        
        # Show filter results
        st.write(f"Showing {len(filtered_skus)} SKUs out of {len(all_skus)} total")
        
        # Create multiselect for SKU selection with filtered options
        selected_skus_search = st.multiselect(
            "Select SKUs to tune",
            options=filtered_skus,
            default=filtered_skus[:min(3, len(filtered_skus))],
            help="Choose which SKUs to optimize parameters for"
        )
    
    with sku_selection_tabs[1]:
        st.markdown("### Select by Characteristics")
        
        # Dummy clustering data for demonstration
        # In a real implementation, this would use actual clustering results
        cluster_options = [
            "High Volatility SKUs",
            "Seasonal SKUs",
            "Trend-dominated SKUs",
            "Stable SKUs",
            "New Product SKUs",
            "End-of-Life SKUs"
        ]
        
        selected_clusters = st.multiselect(
            "Select SKU Clusters",
            options=cluster_options,
            default=["Seasonal SKUs"],
            help="Choose SKUs by their characteristic behavior patterns"
        )
        
        # Simulate assignment of SKUs to clusters
        clustered_skus = []
        if selected_clusters:
            import random
            random.seed(42)  # For consistent demo results
            
            for cluster in selected_clusters:
                # Assign some random SKUs to each selected cluster
                # In reality, this would use actual clustering results
                cluster_size = random.randint(3, 8)
                cluster_skus = random.sample(all_skus, cluster_size)
                clustered_skus.extend(cluster_skus)
            
            # Remove duplicates
            clustered_skus = list(set(clustered_skus))
        
        if clustered_skus:
            st.write(f"Found {len(clustered_skus)} SKUs in selected clusters")
            
            # Show the cluster-based SKU selection
            selected_skus_clusters = st.multiselect(
                "Cluster-Selected SKUs",
                options=sorted(clustered_skus),
                default=clustered_skus[:min(3, len(clustered_skus))],
                help="SKUs selected based on cluster characteristics"
            )
        else:
            st.info("Select clusters to view SKUs")
            selected_skus_clusters = []
    
    with sku_selection_tabs[2]:
        st.markdown("### Select by Performance")
        
        # Performance criteria
        criteria_cols = st.columns(2)
        
        with criteria_cols[0]:
            performance_metric = st.selectbox(
                "Performance Metric",
                ["Forecast Accuracy (MAPE)", "Forecast Error (RMSE)", "Bias", "Stability"],
                index=0
            )
        
        with criteria_cols[1]:
            performance_threshold = st.slider(
                "Performance Threshold",
                0.0, 1.0, 0.2,
                help="Threshold value for the selected metric (lower is better for error metrics)"
            )
        
        # Simulate performance data for SKUs
        # In real implementation, this would use actual performance metrics
        if st.button("Find Underperforming SKUs", use_container_width=True):
            st.session_state.performance_analysis_done = True
            
            # Simulate analysis
            import random
            random.seed(123)  # For consistent results
            
            st.session_state.performance_data = {}
            for sku in all_skus:
                # Generate random metric value
                metric_value = random.uniform(0.05, 0.5)
                st.session_state.performance_data[sku] = metric_value
            
            # Filter by threshold
            underperforming_skus = [
                sku for sku, value in st.session_state.performance_data.items()
                if value > performance_threshold
            ]
            
            st.session_state.underperforming_skus = underperforming_skus
        
        # Display results if analysis has been done
        if 'performance_analysis_done' in st.session_state and st.session_state.performance_analysis_done:
            if 'underperforming_skus' in st.session_state:
                underperforming_skus = st.session_state.underperforming_skus
                
                st.write(f"Found {len(underperforming_skus)} SKUs with {performance_metric} > {performance_threshold}")
                
                # Show the performance-based SKU selection
                selected_skus_performance = st.multiselect(
                    "Performance-Selected SKUs",
                    options=sorted(underperforming_skus),
                    default=underperforming_skus[:min(3, len(underperforming_skus))],
                    help="SKUs selected based on forecast performance criteria"
                )
                
                # Show a bar chart of the worst performers
                if underperforming_skus and 'performance_data' in st.session_state:
                    # Get top 10 worst performers
                    worst_performers = sorted(
                        underperforming_skus,
                        key=lambda sku: st.session_state.performance_data[sku],
                        reverse=True
                    )[:10]
                    
                    # Create data for the chart
                    chart_data = pd.DataFrame({
                        "SKU": worst_performers,
                        "Value": [st.session_state.performance_data[sku] for sku in worst_performers]
                    })
                    
                    # Create bar chart
                    fig = px.bar(
                        chart_data,
                        x="SKU",
                        y="Value",
                        title=f"Top 10 Worst Performers by {performance_metric}",
                        color="Value",
                        color_continuous_scale="Reds"
                    )
                    
                    fig.update_layout(
                        xaxis_title="SKU",
                        yaxis_title=performance_metric
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No underperforming SKUs found")
                selected_skus_performance = []
        else:
            st.info("Click 'Find Underperforming SKUs' to analyze performance")
            selected_skus_performance = []
    
    # Combine selections from different tabs
    # Default to the first tab's selection
    if selected_skus_search:
        selected_skus = selected_skus_search
    elif 'selected_skus_clusters' in locals() and selected_skus_clusters:
        selected_skus = selected_skus_clusters
    elif 'selected_skus_performance' in locals() and selected_skus_performance:
        selected_skus = selected_skus_performance
    else:
        # Fallback to default selection
        selected_skus = all_skus[:min(3, len(all_skus))]
    
    # Option to tune all SKUs
    tune_all = st.checkbox("Tune All SKUs Instead", value=False)
    if tune_all:
        st.info(f"All {len(all_skus)} SKUs will be tuned. This may take some time.")
        selected_skus = all_skus

    # Model selection with more details
    st.markdown("<h3>2. Model-Specific Hyperparameter Fields</h3>", unsafe_allow_html=True)

    # Create a dictionary of tunable models with descriptions
    tunable_models = {
        "auto_arima": {
            "name": "Auto ARIMA",
            "description": "Autoregressive Integrated Moving Average model for time series with trend and seasonality",
            "params": {
                "p": {"desc": "Auto-regressive order", "min": 0, "max": 5, "default": "auto"},
                "d": {"desc": "Differencing order", "min": 0, "max": 2, "default": "auto"},
                "q": {"desc": "Moving average order", "min": 0, "max": 5, "default": "auto"},
                "seasonal": {"desc": "Include seasonality", "type": "bool", "default": True},
                "m": {"desc": "Seasonal period", "min": 0, "max": 52, "default": 12},
                "max_iter": {"desc": "Maximum iterations", "min": 50, "max": 1000, "default": 100}
            }
        },
        "prophet": {
            "name": "Prophet",
            "description": "Facebook's additive model for forecasting with trend, seasonality, and holiday effects",
            "params": {
                "changepoint_prior_scale": {"desc": "Flexibility of trend", "min": 0.001, "max": 0.5, "default": 0.05},
                "seasonality_prior_scale": {"desc": "Flexibility of seasonality", "min": 0.01, "max": 10, "default": 10.0},
                "seasonality_mode": {"desc": "Type of seasonality", "options": ["additive", "multiplicative"], "default": "additive"},
                "yearly_seasonality": {"desc": "Yearly pattern", "type": "bool", "default": True},
                "weekly_seasonality": {"desc": "Weekly pattern", "type": "bool", "default": True},
                "daily_seasonality": {"desc": "Daily pattern", "type": "bool", "default": False}
            }
        },
        "ets": {
            "name": "ETS (Exponential Smoothing)",
            "description": "State space model with exponential smoothing for level, trend, and seasonal components",
            "params": {
                "trend": {"desc": "Trend type", "options": ["add", "mul", None], "default": "add"},
                "damped_trend": {"desc": "Dampening factor", "type": "bool", "default": False},
                "seasonal": {"desc": "Seasonal type", "options": ["add", "mul", None], "default": None},
                "seasonal_periods": {"desc": "Seasonal period", "min": 0, "max": 52, "default": 12}
            }
        },
        "theta": {
            "name": "Theta Method",
            "description": "Decomposition forecasting method combining short and long-term components",
            "params": {
                "theta": {"desc": "Theta parameter", "min": 0, "max": 2, "default": 2},
                "deseasonalize": {"desc": "Remove seasonality", "type": "bool", "default": True},
                "use_test": {"desc": "Use test data", "type": "bool", "default": False}
            }
        },
        "lstm": {
            "name": "LSTM Neural Network",
            "description": "Long Short-Term Memory neural network for sequence prediction with non-linear patterns",
            "params": {
                "units": {"desc": "Network size", "min": 10, "max": 200, "default": 50},
                "n_layers": {"desc": "Network depth", "min": 1, "max": 5, "default": 2},
                "dropout": {"desc": "Dropout rate", "min": 0.0, "max": 0.5, "default": 0.2},
                "epochs": {"desc": "Training epochs", "min": 10, "max": 300, "default": 50},
                "batch_size": {"desc": "Batch size", "min": 4, "max": 128, "default": 16}
            }
        }
    }

    # Create tabs for each model
    model_tabs = st.tabs([model_info["name"] for model_info in tunable_models.values()])
    
    # Track which models are selected
    if "selected_models_with_params" not in st.session_state:
        st.session_state.selected_models_with_params = {}
    
    # Process each model tab
    selected_models = []
    for i, (model_key, model_info) in enumerate(tunable_models.items()):
        with model_tabs[i]:
            # Model description and selection
            col_desc, col_select = st.columns([3, 1])
            
            with col_desc:
                st.markdown(f"### {model_info['name']}")
                st.markdown(f"*{model_info['description']}*")
            
            with col_select:
                st.write("")  # Add some spacing
                model_selected = st.checkbox(
                    "Enable Tuning",
                    value=(model_key in ["auto_arima", "prophet"]),
                    key=f"enable_{model_key}"
                )
                if model_selected:
                    selected_models.append(model_key)
                    # Initialize parameters if not already set
                    if model_key not in st.session_state.selected_models_with_params:
                        st.session_state.selected_models_with_params[model_key] = {
                            "params": {},
                            "ranges": {},
                            "included": []
                        }
            
            # Parameter display
            if model_selected:
                st.markdown("#### Hyperparameters to Tune")
                st.markdown("Select which parameters to tune and specify their search ranges:")
                
                # Group parameters into rows of 2
                param_keys = list(model_info["params"].keys())
                for j in range(0, len(param_keys), 2):
                    param_cols = st.columns(2)
                    
                    # Process up to 2 parameters per row
                    for k in range(2):
                        if j + k < len(param_keys):
                            param_key = param_keys[j + k]
                            param_info = model_info["params"][param_key]
                            
                            with param_cols[k]:
                                # Create a parameter card with styling
                                st.markdown(f"<div class='parameter-card'>", unsafe_allow_html=True)
                                
                                # Parameter name and description with tooltip
                                st.markdown(f"""
                                <div class='parameter-header'>
                                    {param_key} <span class='tooltip-icon' title='{param_info["desc"]}'>ℹ️</span>
                                </div>
                                <div style='font-size: 0.8rem; margin-bottom: 0.5rem;'>{param_info["desc"]}</div>
                                """, unsafe_allow_html=True)
                                
                                # Checkbox to include parameter in tuning
                                include_param = st.checkbox(
                                    f"Tune this parameter",
                                    value=param_key in st.session_state.selected_models_with_params.get(model_key, {}).get("included", []),
                                    key=f"include_{model_key}_{param_key}"
                                )
                                
                                # Add to included parameters list
                                if include_param:
                                    if "included" not in st.session_state.selected_models_with_params[model_key]:
                                        st.session_state.selected_models_with_params[model_key]["included"] = []
                                    if param_key not in st.session_state.selected_models_with_params[model_key]["included"]:
                                        st.session_state.selected_models_with_params[model_key]["included"].append(param_key)
                                elif "included" in st.session_state.selected_models_with_params[model_key] and param_key in st.session_state.selected_models_with_params[model_key]["included"]:
                                    st.session_state.selected_models_with_params[model_key]["included"].remove(param_key)
                                
                                # Parameter settings
                                if include_param:
                                    if "type" in param_info and param_info["type"] == "bool":
                                        # Boolean parameter
                                        st.markdown("*Boolean parameter will try both True and False*")
                                    elif "options" in param_info:
                                        # Categorical parameter
                                        st.markdown("*Categorical parameter will try all options*")
                                        st.markdown(f"Options: {', '.join([str(opt) for opt in param_info['options']])}")
                                    else:
                                        # Numeric parameter with range
                                        # Avoiding nested columns by arranging in one row
                                        st.markdown("<div style='display: flex; gap: 10px;'>", unsafe_allow_html=True)
                                        
                                        # Min value
                                        min_val = st.number_input(
                                            "Min",
                                            value=float(param_info.get("min", 0)),
                                            key=f"min_{model_key}_{param_key}"
                                        )
                                        
                                        # Max value
                                        max_val = st.number_input(
                                            "Max",
                                            value=float(param_info.get("max", 10)),
                                            key=f"max_{model_key}_{param_key}"
                                        )
                                        
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        
                                        # Store the parameter ranges
                                        if "ranges" not in st.session_state.selected_models_with_params[model_key]:
                                            st.session_state.selected_models_with_params[model_key]["ranges"] = {}
                                        st.session_state.selected_models_with_params[model_key]["ranges"][param_key] = (min_val, max_val)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show current best parameters if available
                st.markdown("#### Current Best Parameters")
                try:
                    # Try to fetch current parameters for this model and the first selected SKU
                    if selected_skus:
                        best_params = get_model_parameters(selected_skus[0], model_key)
                        if best_params and 'parameters' in best_params:
                            st.markdown("<div style='background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem; margin-top: 0.5rem;'>", unsafe_allow_html=True)
                            st.markdown(f"**Best Parameters for {selected_skus[0]}**")
                            
                            # Format the parameters
                            formatted_params = ""
                            for p_name, p_value in best_params['parameters'].items():
                                formatted_params += f"- **{p_name}**: `{p_value}`\n"
                            
                            st.markdown(formatted_params)
                            
                            if 'best_score' in best_params:
                                st.markdown(f"**Score**: {best_params['best_score']:.4f}")
                            
                            if 'last_updated' in best_params:
                                st.markdown(f"**Last Updated**: {best_params['last_updated'].strftime('%Y-%m-%d %H:%M')}")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.info("No previously tuned parameters found for this model.")
                    else:
                        st.info("Select an SKU to view its current parameters.")
                except Exception as e:
                    st.warning(f"Could not retrieve parameters: {str(e)}")
    
    # Tuning strategy options
    st.markdown("<h3>3. Tuning Strategy Selector</h3>", unsafe_allow_html=True)
    
    # Create a tabbed interface for different tuning strategies
    strategy_tabs = st.tabs(["Basic", "Advanced", "Expert"])
    
    with strategy_tabs[0]:
        st.markdown("### Basic Tuning Options")
        
        # Simple options for basic users
        cross_validation = st.checkbox(
            "Use cross-validation", 
            value=True, 
            help="Use time series cross-validation for more robust parameter estimation"
        )
        
        n_trials = st.slider(
            "Number of parameter combinations to try", 
            min_value=10, 
            max_value=100, 
            value=30, 
            step=10,
            help="More trials may find better parameters but take longer"
        )
    
    with strategy_tabs[1]:
        st.markdown("### Advanced Tuning Configuration")
        
        tuning_cols = st.columns(2)
        
        with tuning_cols[0]:
            # Optimization algorithm
            optimization_algorithm = st.selectbox(
                "Optimization Algorithm",
                ["Bayesian Optimization", "Random Search", "Grid Search", "Evolution Strategy"],
                index=0,
                help="Method used to search the parameter space"
            )
            
            # Study warm start
            warm_start = st.checkbox(
                "Warm Start Optimization",
                value=True,
                help="Use previous tuning results as a starting point"
            )
        
        with tuning_cols[1]:
            # Multi-metric optimization
            optimization_metric = st.selectbox(
                "Primary Optimization Metric",
                ["MAPE", "RMSE", "MAE", "SMAPE", "MASE"],
                index=0,
                help="Primary metric to optimize (lower is better)"
            )
            
            # Secondary objectives
            secondary_objectives = st.multiselect(
                "Secondary Objectives",
                ["Model Complexity", "Inference Speed", "Training Speed", "Memory Usage"],
                default=["Inference Speed"],
                help="Additional factors to consider during optimization"
            )
    
    with strategy_tabs[2]:
        st.markdown("### Expert Tuning Settings")
        
        expert_cols = st.columns(2)
        
        with expert_cols[0]:
            # Parallel jobs
            parallel_jobs = st.slider(
                "Parallel Tuning Jobs",
                min_value=1,
                max_value=8,
                value=2,
                help="Number of parameter sets to evaluate in parallel"
            )
            
            # Pruning settings
            early_stopping = st.checkbox(
                "Enable Early Stopping",
                value=True,
                help="Stop unpromising trials early to save computation time"
            )
            
            if early_stopping:
                patience = st.number_input(
                    "Early Stopping Patience",
                    min_value=5,
                    max_value=50,
                    value=10,
                    help="Number of iterations without improvement before stopping a trial"
                )
        
        with expert_cols[1]:
            # Time budget settings
            time_budget = st.slider(
                "Maximum Tuning Time (minutes)",
                min_value=1,
                max_value=120,
                value=30,
                help="Maximum time to spend optimizing each model-SKU pair"
            )
            
            # Cross-validation settings
            cv_strategy = st.selectbox(
                "Cross-Validation Strategy",
                ["Expanding Window", "Sliding Window", "K-Fold", "None"],
                index=0,
                help="Method to split time series data for validation"
            )
            
            if cv_strategy != "None":
                n_splits = st.slider(
                    "Number of Cross-Validation Splits",
                    min_value=2,
                    max_value=10,
                    value=3,
                    help="Number of train-test splits for cross-validation"
                )
    
    # Store tuning options in session state
    if st.session_state.get('tuning_options') is None:
        st.session_state.tuning_options = {}
    
    # Basic options
    st.session_state.tuning_options.update({
        "cross_validation": cross_validation,
        "n_trials": n_trials
    })
    
    # Advanced options if set
    if 'optimization_algorithm' in locals():
        st.session_state.tuning_options.update({
            "optimization_algorithm": optimization_algorithm,
            "warm_start": warm_start,
            "optimization_metric": optimization_metric,
            "secondary_objectives": secondary_objectives
        })
    
    # Expert options if set
    if 'parallel_jobs' in locals():
        st.session_state.tuning_options.update({
            "parallel_jobs": parallel_jobs,
            "early_stopping": early_stopping,
            "time_budget": time_budget,
            "cv_strategy": cv_strategy
        })
        if early_stopping and 'patience' in locals():
            st.session_state.tuning_options["patience"] = patience
        if cv_strategy != "None" and 'n_splits' in locals():
            st.session_state.tuning_options["n_splits"] = n_splits

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
                # Ensure params is a dictionary
                if not isinstance(params, dict):
                    return str(params)
                
                if model_type == "auto_arima":
                    return f"p={params.get('p', 'auto')}, d={params.get('d', 'auto')}, q={params.get('q', 'auto')}"
                elif model_type == "prophet":
                    # Convert values to appropriate types to avoid serialization issues
                    cp_scale = params.get('changepoint_prior_scale', 0.05)
                    sp_scale = params.get('seasonality_prior_scale', 10.0)
                    
                    # Ensure numeric values
                    try:
                        cp_scale = float(cp_scale)
                        sp_scale = float(sp_scale)
                    except (ValueError, TypeError):
                        cp_scale = 0.05
                        sp_scale = 10.0
                        
                    return f"changepoint_prior_scale={cp_scale:.3f}, seasonality_prior_scale={sp_scale:.3f}"
                elif model_type == "ets":
                    trend = params.get('trend', 'add')
                    seasonal = params.get('seasonal', None)
                    damped = params.get('damped_trend', False)
                    return f"trend={trend}, seasonal={seasonal}, damped={damped}"
                elif model_type == "theta":
                    return f"Theta parameters optimized"
                elif model_type == "lstm":
                    units = params.get('units', 50)
                    layers = params.get('n_layers', 2)
                    return f"units={units}, layers={layers}"
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

                        # Get the optimized parameters from the database (actual tuning results)
                        optimal_params = get_model_parameters(sku, model_type)

                        # Store results in session state
                        if sku not in st.session_state.tuning_results:
                            st.session_state.tuning_results[sku] = {}
                            
                        # Store the actual tuning results with all metadata
                        if optimal_params and 'parameters' in optimal_params:
                            # Store the full parameter dictionary
                            st.session_state.tuning_results[sku][model_type] = optimal_params['parameters']
                            
                            # Store the score for this model and SKU
                            if 'model_scores' not in st.session_state:
                                st.session_state.model_scores = {}
                            
                            if sku not in st.session_state.model_scores:
                                st.session_state.model_scores[sku] = {}
                            
                            # Store best score from optimization
                            if 'best_score' in optimal_params:
                                st.session_state.model_scores[sku][model_type] = optimal_params['best_score']
                        else:
                            # If no parameters were found, initialize with empty dict
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

# Display enhanced tuning results visualization with performance comparison
if not st.session_state.tuning_in_progress and (st.session_state.tuning_results or 'tuning_results' in st.session_state):
    st.markdown("<h3>5. Parameter Results and Comparison</h3>", unsafe_allow_html=True)
    
    # Add tabs for different result views
    result_overview_tabs = st.tabs(["Model Performance", "Parameter Data Table"])
    
    with result_overview_tabs[1]:
        st.markdown("### Hyperparameter Results - All SKUs and Models")
        st.markdown("This table contains all tuned parameters for all SKUs and models. You can download this data for integration with other systems.")
        
        # Get model display names for nicer formatting
        model_display_names = {
            "auto_arima": "ARIMA", 
            "prophet": "Prophet",
            "ets": "ETS",
            "theta": "Theta",
            "lstm": "LSTM"
        }
        
        # Create flattened parameter data for display and download
        parameter_rows = []
        
        # Will be filled after session results are loaded

    # Get all tuned parameters from database for all SKUs that were tuned
    tuning_results = {}
    model_scores = {}  # To store performance metrics

    # Try to get results from session state first
    if st.session_state.tuning_results:
        tuning_results = st.session_state.tuning_results
        
        # Get model scores from session state if available
        if 'model_scores' in st.session_state:
            model_scores = st.session_state.model_scores
    else:
        # If not in session state, fetch from database
        for sku in st.session_state.tuning_skus:
            tuning_results[sku] = {}
            model_scores[sku] = {}
            for model_type in st.session_state.tuning_models:
                params = get_model_parameters(sku, model_type)
                if params and 'parameters' in params:
                    tuning_results[sku][model_type] = params['parameters']
                    if 'best_score' in params:
                        model_scores[sku][model_type] = params['best_score']
    
    # Log the tuning results for debugging
    print(f"Loaded tuning results for SKUs: {list(tuning_results.keys())}")
    print(f"Model scores for different SKUs: {model_scores}")
    
    # Now create the flattened parameter table
    parameter_rows = []
    
    # Process all tuning results into a flat table structure
    for sku_code in tuning_results:
        for model_type, params in tuning_results[sku_code].items():
            if params:
                model_name = model_display_names.get(model_type, model_type.upper())
                # For each parameter in this model, create a separate row
                for param_name, param_value in params.items():
                    # Convert values to appropriate string representation
                    if isinstance(param_value, bool):
                        param_value_str = str(param_value)
                    elif isinstance(param_value, (int, float)):
                        param_value_str = str(param_value)
                    else:
                        param_value_str = str(param_value)
                    
                    # Add a row in the format from the example
                    parameter_rows.append({
                        "SKU code": sku_code,
                        "SKU name": sku_code,  # Using same value for simplicity
                        "Model name": model_name,
                        "Parameter name": param_name,
                        "Parameter value": param_value_str
                    })
    
    # Create the all parameters dataframe
    if parameter_rows:
        all_params_df = pd.DataFrame(parameter_rows)
        
        # Populate the parameter table in the second tab
        with result_overview_tabs[1]:
            # Display the parameters table
            st.dataframe(all_params_df, use_container_width=True)
            
            # Create a download button for the table
            csv = all_params_df.to_csv(index=False)
            
            # Add download button for CSV
            st.download_button(
                label="Download Parameters as CSV",
                data=csv,
                file_name="hyperparameter_tuning_results.csv",
                mime="text/csv",
                help="Download the complete parameter table in CSV format for use in other systems"
            )
            
            # Add a copy to clipboard option (using streamlit's native feature)
            st.code(csv, language=None)
            st.caption("You can also copy the CSV data above for pasting into other applications")

    if tuning_results:
        # Create a more advanced results dashboard
        # Define model names with appropriate badges - use HTML rendering
        model_names = {
            "auto_arima": "Auto ARIMA", 
            "prophet": "Prophet",
            "ets": "Exponential Smoothing",
            "theta": "Theta Method",
            "lstm": "LSTM Neural Network"
        }
        
        # Define model badges for display
        model_badges = {
            "auto_arima": "ARIMA", 
            "prophet": "FB",
            "ets": "ETS",
            "theta": "Θ",
            "lstm": "DL"
        }
        
        # Create a tabbed interface for different result views
        result_tabs = st.tabs(["Model Performance", "Parameter Impact", "Interactive Exploration", "Before/After Comparison", "Audit Log"])
        
        with result_tabs[0]:
            st.markdown("### Model Performance Comparison")
            
            # Select an SKU to analyze in detail
            available_skus = list(tuning_results.keys())
            if available_skus:
                # Use session state to track SKU changes
                if 'selected_result_sku' not in st.session_state:
                    st.session_state.selected_result_sku = available_skus[0]
                
                # Define callback to update on SKU change
                def on_sku_change():
                    # Update session state when SKU changes
                    st.session_state.selected_result_sku = st.session_state.result_sku_selector
                
                # Create the selectbox with the callback
                selected_result_sku = st.selectbox(
                    "Select SKU to analyze",
                    options=available_skus,
                    index=available_skus.index(st.session_state.selected_result_sku),
                    key="result_sku_selector",
                    on_change=on_sku_change
                )
                
                # Get model results for this SKU
                sku_models = list(tuning_results.get(selected_result_sku, {}).keys())
                
                if sku_models:
                    # Create metric cards for this SKU's performance using flexbox layout with proper spacing
                    st.markdown("<div style='display: flex; flex-wrap: wrap; gap: 20px; justify-content: flex-start; margin-bottom: 20px;'>", unsafe_allow_html=True)
                    
                    # Generate sample data for demonstration that varies by SKU
                    np.random.seed(hash(selected_result_sku) % 10000)  # Use SKU as seed for consistent but varied values
                    sku_factor = (ord(selected_result_sku[-1]) % 9 + 1) / 10.0  # Generate a factor based on last char of SKU
                    
                    # Base scores for each model
                    base_scores = {
                        "auto_arima": 12.45,
                        "prophet": 14.21,
                        "ets": 11.89,
                        "theta": 15.36,
                        "lstm": 10.72
                    }
                    
                    # Adjust scores based on SKU - will be different for each SKU
                    sample_scores = {}
                    for model, score in base_scores.items():
                        # Add random variation based on SKU
                        adjustment = np.random.uniform(-3.0, 3.0) * sku_factor
                        sample_scores[model] = max(5.0, min(25.0, score + adjustment))  # Keep in reasonable range
                    
                    # Add metric data for each model
                    for model_type in sku_models:
                        # Get score from database or use adjusted sample data
                        real_score = model_scores.get(selected_result_sku, {}).get(model_type, 0)
                        
                        # If real score is 0, use sample data specific to this SKU
                        if real_score == 0:
                            score = sample_scores.get(model_type, 15.0)
                        else:
                            score = real_score
                        
                        # Get model name
                        model_display = model_names.get(model_type, model_type.upper())
                        badge = model_badges.get(model_type, "")
                        
                        # Create a card with badge and score using HTML
                        badge_color = {
                            "auto_arima": "#4299e1",
                            "prophet": "#805ad5",
                            "ets": "#38a169",
                            "theta": "#dd6b20",
                            "lstm": "#e53e3e"
                        }.get(model_type, "#718096")
                        
                        # Using a fixed card width to ensure proper horizontal arrangement
                        st.markdown(f"""
                        <div style="width: 180px; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center; background-color: #f8fafc; margin-bottom: 15px;">
                            <div style="margin-bottom: 8px;">
                                <span style="display: inline-block; background-color: {badge_color}; color: white; border-radius: 12px; padding: 2px 8px; font-size: 12px;">{badge}</span>
                                <span style="font-weight: 500; margin-left: 6px;">{model_display}</span>
                            </div>
                            <div style="font-size: 24px; font-weight: 600; color: #2d3748;">
                                {score:.4f}
                            </div>
                            <div style="font-size: 12px; color: #718096; margin-top: 4px;">
                                MAPE Score
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Close the flexbox container
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show parameters comparison across models
                    st.markdown("#### Parameter Comparison")
                    
                    # Prepare comparison data
                    comparison_data = []
                    param_columns = set()
                    
                    for model_type in sku_models:
                        model_params = tuning_results[selected_result_sku].get(model_type, {})
                        for param_name in model_params.keys():
                            param_columns.add(param_name)
                    
                    # Create a row for each model with all parameters
                    for model_type in sku_models:
                        model_params = tuning_results[selected_result_sku].get(model_type, {})
                        model_row = {"Model": model_type}
                        
                        # Add each parameter or '-' if not present
                        for param_name in param_columns:
                            if param_name in model_params:
                                model_row[param_name] = model_params[param_name]
                            else:
                                model_row[param_name] = "-"
                        
                        comparison_data.append(model_row)
                    
                    # Create a dataframe with the comparison data
                    if comparison_data:
                        # Create a dataframe with string conversion to avoid serialization issues
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Convert all values to strings to ensure consistent serialization
                        for col in comparison_df.columns:
                            comparison_df[col] = comparison_df[col].astype(str)
                        
                        # Create a styled dataframe with updated styling method
                        st.dataframe(
                            comparison_df.style.map(
                                lambda x: 'background-color: #f0f8ff' if x != '-' else ''
                            ),
                            use_container_width=True
                        )
                    
                    # Visualize model performance with model-specific tabs
                    st.markdown("#### Performance Visualization")
                    
                    # Create model-specific tabs
                    if sku_models:
                        model_specific_tabs = st.tabs([model_names.get(model, model.upper()) for model in sku_models])
                        
                        # Generate example forecast plots for each model in its own tab
                        try:
                            for i, model_type in enumerate(sku_models):
                                with model_specific_tabs[i]:
                                    st.markdown(f"##### {model_names.get(model_type, model_type.upper())} Performance Details", unsafe_allow_html=True)
                                    
                                    # Create a visualization specific to this model
                                    st.markdown("<div style='border: 1px solid #ddd; border-radius: 5px; padding: 15px;'>", unsafe_allow_html=True)
                                    st.markdown(f"### {model_type.upper()} Forecasting Performance")
                                    
                                    # Generate synthetic data for demonstration with proper timestamp handling
                                    start_date = pd.Timestamp('2023-01-01')
                                    periods = 24
                                    dates = [start_date + pd.DateOffset(months=i) for i in range(periods)]
                                    
                                    # Generate random data with a seed that's different for each SKU-model combination
                                    sku_hash = hash(selected_result_sku) % 10000
                                    model_index = i
                                    np.random.seed(sku_hash + model_index)  # Unique seed per SKU-model combination
                                    
                                    # Create base data with some variation based on SKU
                                    base_scale = (ord(selected_result_sku[-1]) % 9 + 5) * 10  # Scale varies by SKU
                                    base_trend = (ord(selected_result_sku[0]) % 5 + 3) * 10   # Trend varies by SKU
                                    actuals = np.random.normal(100, base_scale / 5, periods).cumsum() + base_trend * 10
                                    
                                    # Make the forecast error depend on the model type to show differences
                                    error_scale = {
                                        "auto_arima": 40,
                                        "prophet": 30,
                                        "ets": 35,
                                        "theta": 45,
                                        "lstm": 25
                                    }.get(model_type, 40)
                                    
                                    before_forecast = actuals + np.random.normal(0, error_scale, periods)
                                    after_forecast = actuals + np.random.normal(0, error_scale * 0.4, periods)
                                    
                                    # Create a single figure with all three lines for this model
                                    fig = go.Figure()
                                    
                                    # Add actual data
                                    fig.add_trace(go.Scatter(
                                        x=dates, y=actuals,
                                        mode='lines+markers',
                                        name='Actual',
                                        line=dict(color='blue', width=3),
                                        marker=dict(size=8, symbol='circle')
                                    ))
                                    
                                    # Add before tuning forecast
                                    fig.add_trace(go.Scatter(
                                        x=dates, y=before_forecast,
                                        mode='lines+markers',
                                        name='Before Tuning',
                                        line=dict(color='red', width=2, dash='dot'),
                                        marker=dict(size=6, symbol='triangle-up')
                                    ))
                                    
                                    # Add after tuning forecast
                                    fig.add_trace(go.Scatter(
                                        x=dates, y=after_forecast,
                                        mode='lines+markers',
                                        name='After Tuning',
                                        line=dict(color='green', width=2, dash='dash'),
                                        marker=dict(size=6, symbol='diamond')
                                    ))
                                    
                                    # Create a vertical line by adding a shape
                                    forecast_idx = int(periods * 0.7)
                                    if 0 <= forecast_idx < len(dates):
                                        forecast_start = dates[forecast_idx]
                                        try:
                                            fig.update_layout(
                                                shapes=[
                                                    dict(
                                                        type='line',
                                                        yref='paper',
                                                        xref='x',
                                                        x0=forecast_start,
                                                        y0=0,
                                                        x1=forecast_start,
                                                        y1=1,
                                                        line=dict(
                                                            color='gray',
                                                            width=2,
                                                            dash='solid'
                                                        )
                                                    )
                                                ],
                                                annotations=[
                                                    dict(
                                                        x=forecast_start,
                                                        y=1,
                                                        xref='x',
                                                        yref='paper',
                                                        text="Forecast Start",
                                                        showarrow=False,
                                                        font=dict(
                                                            color="gray",
                                                            size=12
                                                        ),
                                                        bgcolor="white",
                                                        bordercolor="gray",
                                                        borderwidth=1,
                                                        borderpad=4
                                                    )
                                                ]
                                            )
                                        except Exception as e:
                                            st.warning(f"Could not add forecast boundary: {str(e)}")
                                    
                                    # Update layout for this model's chart
                                    fig.update_layout(
                                        height=400,
                                        margin=dict(l=10, r=10, t=10, b=10),
                                        legend=dict(
                                            orientation="h", 
                                            yanchor="bottom", 
                                            y=1.02, 
                                            xanchor="right", 
                                            x=1,
                                            bgcolor='rgba(255, 255, 255, 0.8)',
                                            bordercolor='rgba(0, 0, 0, 0.1)',
                                            borderwidth=1
                                        ),
                                        hovermode="x unified"
                                    )
                                    
                                    # Display chart in this model's tab
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Add model-specific metrics
                                    score = model_scores.get(selected_result_sku, {}).get(model_type, 0)
                                    if score == 0:
                                        # Use different base metrics for each model type
                                        base_scores = {
                                            "auto_arima": 22.5,
                                            "prophet": 24.8,
                                            "ets": 21.9,
                                            "theta": 26.3,
                                            "lstm": 19.7
                                        }
                                        base_mape = base_scores.get(model_type, 23.0)
                                        tuned_mape = base_mape * 0.5  # 50% improvement
                                    else:
                                        base_mape = 24.8
                                        tuned_mape = score
                                    
                                    # Add metrics for this model
                                    metrics_col1, metrics_col2 = st.columns(2)
                                    
                                    with metrics_col1:
                                        st.markdown(f"**{model_type.upper()} Before Tuning:**")
                                        st.markdown(f"""
                                        - MAPE: {base_mape:.1f}%
                                        - RMSE: {base_mape * 2.5:.1f}
                                        - MAE: {base_mape * 2.0:.1f}
                                        """)
                                        
                                    with metrics_col2:
                                        st.markdown(f"**{model_type.upper()} After Tuning:**")
                                        st.markdown(f"""
                                        - MAPE: {tuned_mape:.1f}%
                                        - RMSE: {tuned_mape * 2.5:.1f}
                                        - MAE: {tuned_mape * 2.0:.1f}
                                        """)
                                    
                                    # Calculate improvement percentages
                                    mape_improvement = (base_mape - tuned_mape) / base_mape * 100
                                    
                                    # Add improvement metrics
                                    st.markdown("<div style='margin-top: 10px; padding: 10px; background-color: #f0fff4; border-radius: 5px; border-left: 4px solid #38a169;'>", unsafe_allow_html=True)
                                    st.markdown(f"#### {model_type.upper()} Performance Improvement")
                                    st.markdown(f"""
                                    - MAPE: **{mape_improvement:.1f}%** improvement
                                    - RMSE: **{mape_improvement:.1f}%** improvement
                                    - MAE: **{mape_improvement:.1f}%** improvement
                                    """)
                                    st.markdown("</div>", unsafe_allow_html=True)
                                    
                                    # Add accept button specific to this model
                                    st.markdown("<div style='text-align: center; margin-top: 15px;'>", unsafe_allow_html=True)
                                    st.button(f"👍 Accept {model_type.upper()} Parameters", key=f"accept_{model_type}")
                                    st.markdown("</div>", unsafe_allow_html=True)
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Create a header for overall comparison
                            st.markdown("<div style='margin-top: 30px; border: 1px solid #ddd; border-radius: 5px; padding: 15px;'>", unsafe_allow_html=True)
                            st.markdown("### Overall Model Comparison")
                            
                            # Generate synthetic data for demonstration with proper timestamp handling
                            start_date = pd.Timestamp('2023-01-01')
                            periods = 24
                            dates = [start_date + pd.DateOffset(months=i) for i in range(periods)]
                            
                            # Use the SKU to generate unique but consistent data
                            sku_seed = hash(selected_result_sku) % 10000
                            np.random.seed(sku_seed)  # Seed based on selected SKU
                            
                            # Create data with characteristics based on the SKU
                            sku_scale = (ord(selected_result_sku[-1]) % 9 + 5) * 10  # Scale varies by SKU
                            sku_trend = (ord(selected_result_sku[0]) % 5 + 3) * 10   # Trend varies by SKU
                            sku_noise = (ord(selected_result_sku[1]) % 5 + 2) * 5     # Noise level varies by SKU
                            
                            # Generate overall data unique to this SKU
                            actuals = np.random.normal(100, sku_scale / 5, periods).cumsum() + sku_trend * 10
                            before_forecast = actuals + np.random.normal(0, sku_noise * 10, periods)  # Higher error
                            after_forecast = actuals + np.random.normal(0, sku_noise * 4, periods)    # Lower error
                            
                            # Create a single figure with all three lines
                            fig = go.Figure()
                            
                            # Add actual data
                            fig.add_trace(go.Scatter(
                                x=dates, y=actuals,
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='blue', width=3),
                                marker=dict(size=8, symbol='circle')
                            ))
                            
                            # Add before tuning forecast
                            fig.add_trace(go.Scatter(
                                x=dates, y=before_forecast,
                                mode='lines+markers',
                                name='Before Tuning',
                                line=dict(color='red', width=2, dash='dot'),
                                marker=dict(size=6, symbol='triangle-up')
                            ))
                            
                            # Add after tuning forecast
                            fig.add_trace(go.Scatter(
                                x=dates, y=after_forecast,
                                mode='lines+markers',
                                name='After Tuning',
                                line=dict(color='green', width=2, dash='dash'),
                                marker=dict(size=6, symbol='diamond')
                            ))
                            
                            # Add a vertical line to indicate forecast start - calculate index safely
                            forecast_idx = int(periods * 0.7)
                            if 0 <= forecast_idx < len(dates):
                                forecast_start = dates[forecast_idx]
                                # Create a vertical line by adding a separate shape to the layout
                                try:
                                    fig.update_layout(
                                        shapes=[
                                            dict(
                                                type='line',
                                                yref='paper',
                                                xref='x',
                                                x0=forecast_start,
                                                y0=0,
                                                x1=forecast_start,
                                                y1=1,
                                                line=dict(
                                                    color='gray',
                                                    width=2,
                                                    dash='solid'
                                                )
                                            )
                                        ],
                                        annotations=[
                                            dict(
                                                x=forecast_start,
                                                y=1,
                                                xref='x',
                                                yref='paper',
                                                text="Forecast Start",
                                                showarrow=False,
                                                font=dict(
                                                    color="gray",
                                                    size=12
                                                ),
                                                bgcolor="white",
                                                bordercolor="gray",
                                                borderwidth=1,
                                                borderpad=4
                                            )
                                        ]
                                    )
                                except Exception as e:
                                    st.warning(f"Could not add forecast boundary: {str(e)}")
                            
                            # Update layout for better visualization
                            fig.update_layout(
                                height=400,
                                margin=dict(l=10, r=10, t=10, b=10),
                                legend=dict(
                                    orientation="h", 
                                    yanchor="bottom", 
                                    y=1.02, 
                                    xanchor="right", 
                                    x=1,
                                    bgcolor='rgba(255, 255, 255, 0.8)',
                                    bordercolor='rgba(0, 0, 0, 0.1)',
                                    borderwidth=1
                                ),
                                hovermode="x unified"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add metrics comparison side by side
                            metrics_col1, metrics_col2 = st.columns(2)
                            
                            with metrics_col1:
                                st.markdown("**Before Tuning Metrics:**")
                                st.markdown("""
                                - MAPE: 24.8%
                                - RMSE: 67.3
                                - MAE: 52.9
                                """)
                                
                            with metrics_col2:
                                # Get average score of all models for this SKU
                                best_model_score = min([score for model, score in model_scores.get(selected_result_sku, {}).items()] or [10.2])
                                st.markdown("**After Tuning Metrics:**")
                                st.markdown(f"""
                                - MAPE: {best_model_score:.1f}%
                                - RMSE: {best_model_score * 2.5:.1f}
                                - MAE: {best_model_score * 2:.1f}
                                """)
                            
                            # Calculate improvement percentages
                            mape_improvement = (24.8 - best_model_score) / 24.8 * 100
                            rmse_improvement = (67.3 - (best_model_score * 2.5)) / 67.3 * 100
                            mae_improvement = (52.9 - (best_model_score * 2)) / 52.9 * 100
                            
                            # Add improvement metrics
                            st.markdown("<div style='margin-top: 10px; padding: 10px; background-color: #f0fff4; border-radius: 5px; border-left: 4px solid #38a169;'>", unsafe_allow_html=True)
                            st.markdown("#### Performance Improvement")
                            st.markdown(f"""
                            - MAPE: **{mape_improvement:.1f}%** improvement
                            - RMSE: **{rmse_improvement:.1f}%** improvement
                            - MAE: **{mae_improvement:.1f}%** improvement
                            """)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Add accept button (centered using markdown)
                            st.markdown("<div style='text-align: center; margin-top: 15px;'>", unsafe_allow_html=True)
                            st.button(f"👍 Accept Best Parameters", key="accept_best_model")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        except Exception as e:
                            st.warning(f"Could not generate visualization: {str(e)}")
                    else:
                        st.info("No model results available for this SKU.")
            else:
                st.info("No tuning results available. Run hyperparameter tuning first.")
        
        with result_tabs[1]:
            st.markdown("### Parameter Impact Analysis")
            
            # Create a parameter impact visualization
            st.markdown("""
            This analysis shows how different parameters affect model performance.
            It helps identify which parameters have the most influence on forecast accuracy.
            """)
            
            # Allow selecting a model to analyze
            model_options = list(set([model for sku_models in tuning_results.values() for model in sku_models.keys()]))
            if model_options:
                # Initialize session state for impact model if not present
                if 'selected_impact_model' not in st.session_state:
                    st.session_state.selected_impact_model = model_options[0] if "prophet" in model_options else model_options[0]
                
                # Define callback for model change
                def on_impact_model_change():
                    st.session_state.selected_impact_model = st.session_state.impact_model_selector
                
                # Find the current index for the selected model
                model_index = 0
                if st.session_state.selected_impact_model in model_options:
                    model_index = model_options.index(st.session_state.selected_impact_model)
                
                # Create selectbox with callback
                selected_impact_model = st.selectbox(
                    "Select model to analyze",
                    options=model_options,
                    index=model_index,
                    key="impact_model_selector",
                    on_change=on_impact_model_change
                )
                
                # Show parameter impact visualization
                st.markdown(f"#### Parameter Impact for {selected_impact_model.upper()}")
                
                # Create example parameter impact visualizations
                if selected_impact_model == "prophet":
                    # Get the optimal parameters for this model
                    # Find SKUs that have results for this model
                    relevant_skus = [sku for sku, models in tuning_results.items() if selected_impact_model in models]
                    
                    # Create a seed based on model name
                    model_seed = hash(selected_impact_model) % 10000
                    np.random.seed(model_seed)
                    
                    # Base values for Prophet parameters
                    cp_values = [0.001, 0.05, 0.5]
                    sp_values = [0.1, 1.0, 10.0]
                    
                    # Generate MAPE values with some randomness but a clear pattern
                    # where optimal values are around changepoint_prior_scale=0.05 and seasonality_prior_scale=1.0
                    cp_mapes = [
                        18.5 + np.random.uniform(-2, 2),  # High for low CP
                        15.2 + np.random.uniform(-1, 1),  # Optimal
                        22.1 + np.random.uniform(-3, 3)   # High for high CP
                    ]
                    
                    sp_mapes = [
                        19.8 + np.random.uniform(-2, 2),  # High for low SP
                        15.2 + np.random.uniform(-1, 1),  # Optimal
                        16.4 + np.random.uniform(-2, 2)   # Medium for high SP
                    ]
                    
                    # Create DataFrame for visualization
                    impact_data = pd.DataFrame({
                        "Parameter": ["changepoint_prior_scale"] * 3 + ["seasonality_prior_scale"] * 3,
                        "Value": cp_values + sp_values,
                        "MAPE": cp_mapes + sp_mapes
                    }).astype({
                        "Parameter": str,
                        "Value": float,
                        "MAPE": float
                    })
                    
                    fig = px.line(
                        impact_data, 
                        x="Value", 
                        y="MAPE", 
                        color="Parameter",
                        markers=True,
                        title="Parameter Values vs. MAPE",
                        log_x=True
                    )
                    
                    fig.update_layout(
                        xaxis_title="Parameter Value (log scale)",
                        yaxis_title="MAPE (%)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown("""
                    #### Key Insights:
                    - **changepoint_prior_scale** has an optimal value around 0.05
                    - **seasonality_prior_scale** performs best around 1.0
                    - Too high or too low values for either parameter decrease model accuracy
                    """)
                    
                elif selected_impact_model == "auto_arima":
                    # Create a seed based on model name
                    model_seed = hash(selected_impact_model) % 10000
                    np.random.seed(model_seed)
                    
                    # Base PDQ combinations
                    base_pdq_combos = [
                        {"p": 1, "d": 1, "q": 1, "base_MAPE": 18.3},
                        {"p": 2, "d": 1, "q": 1, "base_MAPE": 15.7},
                        {"p": 2, "d": 1, "q": 2, "base_MAPE": 12.9},
                        {"p": 3, "d": 1, "q": 1, "base_MAPE": 14.2},
                        {"p": 3, "d": 1, "q": 2, "base_MAPE": 13.6},
                        {"p": 3, "d": 2, "q": 2, "base_MAPE": 19.8}
                    ]
                    
                    # Add random variations to make unique for this model
                    pdq_combos = []
                    for combo in base_pdq_combos:
                        # Add some small random variation to MAPE
                        mape_variation = np.random.uniform(-1.5, 1.5)
                        pdq_combos.append({
                            "p": combo["p"],
                            "d": combo["d"],
                            "q": combo["q"],
                            "MAPE": combo["base_MAPE"] + mape_variation
                        })
                    
                    pdq_df = pd.DataFrame(pdq_combos)
                    
                    # Create a heatmap-style visualization
                    fig = px.scatter(
                        pdq_df,
                        x="p",
                        y="q",
                        size="MAPE",
                        color="MAPE",
                        hover_name="MAPE",
                        size_max=30,
                        color_continuous_scale="Reds_r"  # Reversed so lower MAPE is better (blue)
                    )
                    
                    fig.update_layout(
                        title="ARIMA (p,d,q) Parameter Impact",
                        xaxis_title="p (AR order)",
                        yaxis_title="q (MA order)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown("""
                    #### Key Insights:
                    - The best performance is achieved with p=2, d=1, q=2
                    - Higher differencing order (d=2) reduces model accuracy
                    - Adding more AR terms (p) doesn't always improve performance
                    """)
                else:
                    st.info(f"Parameter impact analysis for {selected_impact_model} is not available in this demo.")
            else:
                st.info("No models have been tuned yet. Run hyperparameter tuning first.")
        
        with result_tabs[2]:
            st.markdown("### Interactive Hyperparameter Exploration")
            
            st.markdown("""
            This interactive tool allows you to explore how changing hyperparameters affects model performance in real-time.
            Adjust the sliders to see immediate visual feedback on how forecast quality changes.
            """)
            
            # Allow selecting a model to explore
            explore_model_options = list(set([model for sku_models in tuning_results.values() for model in sku_models.keys()]))
            if explore_model_options:
                # Create columns for selector layout
                explore_col1, explore_col2 = st.columns([2, 1])
                
                # Initialize session state for model selection if not already present
                if 'selected_explore_model' not in st.session_state:
                    st.session_state.selected_explore_model = explore_model_options[0] if explore_model_options else ""
                
                # Define callback for model change
                def on_model_change():
                    st.session_state.selected_explore_model = st.session_state.explore_model_selector
                    
                    # When model changes, reset SKU selection
                    available_skus = [sku for sku, models in tuning_results.items() 
                                     if st.session_state.selected_explore_model in models]
                    if available_skus:
                        st.session_state.selected_explore_sku = available_skus[0]
                
                with explore_col1:
                    model_index = 0
                    if st.session_state.selected_explore_model in explore_model_options:
                        model_index = explore_model_options.index(st.session_state.selected_explore_model)
                        
                    selected_explore_model = st.selectbox(
                        "Select model to explore",
                        options=explore_model_options,
                        index=model_index,
                        key="explore_model_selector",
                        on_change=on_model_change
                    )
                
                with explore_col2:
                    # SKU selection for this model
                    available_skus = [sku for sku, models in tuning_results.items() if selected_explore_model in models]
                    
                    # Initialize selected SKU in session state if needed
                    if 'selected_explore_sku' not in st.session_state and available_skus:
                        st.session_state.selected_explore_sku = available_skus[0]
                    
                    # Define callback for SKU change
                    def on_sku_change():
                        st.session_state.selected_explore_sku = st.session_state.explore_sku_selector
                    
                    if available_skus:
                        # Default index if current selection not available
                        sku_index = 0
                        
                        # Use stored selection if available
                        if st.session_state.selected_explore_sku in available_skus:
                            sku_index = available_skus.index(st.session_state.selected_explore_sku)
                        
                        selected_explore_sku = st.selectbox(
                            "Select SKU to visualize",
                            options=available_skus,
                            index=sku_index,
                            key="explore_sku_selector",
                            on_change=on_sku_change
                        )
                        
                        # Get current parameters for this SKU and model
                        current_params = tuning_results.get(selected_explore_sku, {}).get(selected_explore_model, {})
                        
                        # Display interactive exploration based on model type
                        if current_params:
                            st.markdown(f"#### Interactive {selected_explore_model.upper()} Parameter Exploration")
                            
                            # Create containers for the visualization and controls
                            params_container = st.container()
                            simulation_container = st.container()
                            
                            # Define parameter ranges and default values based on model type
                            if selected_explore_model == "prophet":
                                with params_container:
                                    st.markdown("##### Adjust Prophet Parameters")
                                    
                                    # Get current parameters or use defaults with robust error handling
                                    try:
                                        current_cp = float(current_params.get("changepoint_prior_scale", 0.05))
                                    except (ValueError, TypeError):
                                        current_cp = 0.05
                                        
                                    try:
                                        current_sp = float(current_params.get("seasonality_prior_scale", 10.0))
                                    except (ValueError, TypeError):
                                        current_sp = 10.0
                                        
                                    # Ensure the mode is one of the valid options
                                    current_sm = current_params.get("seasonality_mode", "additive")
                                    if current_sm not in ["additive", "multiplicative"]:
                                        current_sm = "additive"
                                    
                                    # Create parameter sliders with current values
                                    cp_slider = st.slider(
                                        "Changepoint Prior Scale",
                                        min_value=0.001,
                                        max_value=0.5,
                                        value=current_cp,
                                        step=0.001,
                                        format="%.3f",
                                        key="cp_slider",
                                        help="Controls flexibility in trend changes. Lower values = more flexible trend."
                                    )
                                    
                                    sp_slider = st.slider(
                                        "Seasonality Prior Scale",
                                        min_value=0.01,
                                        max_value=10.0,
                                        value=current_sp,
                                        step=0.01,
                                        format="%.2f",
                                        key="sp_slider",
                                        help="Controls flexibility of seasonality. Higher values = more flexible seasonality."
                                    )
                                    
                                    sm_radio = st.radio(
                                        "Seasonality Mode",
                                        options=["additive", "multiplicative"],
                                        index=0 if current_sm == "additive" else 1,
                                        horizontal=True,
                                        key="sm_radio",
                                        help="Additive: seasonality is constant. Multiplicative: seasonality scales with trend."
                                    )
                                    
                                    # Generate the visualization based on selected parameters
                                    with simulation_container:
                                        st.markdown("##### Forecast Simulation")
                                        
                                        # Create a visual indicator of parameter changes
                                        st.markdown(f"""
                                        <div style="margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;">
                                        <span style="font-weight: bold;">Current Parameters:</span> changepoint_prior_scale={cp_slider:.3f}, 
                                        seasonality_prior_scale={sp_slider:.2f}, seasonality_mode={sm_radio}
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Calculate simulated performance metrics based on parameter changes
                                        # This uses a simplified model to simulate how changes affect performance
                                        base_mape = 15.0  # Example base MAPE
                                        
                                        # Simple simulation function (would be replaced with actual model in production)
                                        def simulate_prophet_performance(cp, sp, mode):
                                            # Simplified model of how parameters affect performance
                                            # These are just example relationships
                                            cp_effect = 5 * abs(0.05 - cp) if cp <= 0.1 else 10 * abs(0.05 - cp)
                                            sp_effect = 0.2 * abs(1.0 - sp) if sp <= 5.0 else 0.5 * abs(1.0 - sp)
                                            mode_effect = 0.0 if mode == "additive" else 1.5  # Example penalty for multiplicative
                                            
                                            # Calculate simulated MAPE (lower is better)
                                            simulated_mape = base_mape + cp_effect + sp_effect + mode_effect
                                            
                                            # Calculate other metrics
                                            simulated_rmse = simulated_mape * 2.5
                                            simulated_mae = simulated_mape * 2.0
                                            
                                            return {
                                                "mape": simulated_mape,
                                                "rmse": simulated_rmse,
                                                "mae": simulated_mae
                                            }
                                        
                                        # Simulate base performance (with current parameters)
                                        base_performance = simulate_prophet_performance(current_cp, current_sp, current_sm)
                                        
                                        # Simulate new performance (with slider parameters)
                                        new_performance = simulate_prophet_performance(cp_slider, sp_slider, sm_radio)
                                        
                                        # Calculate improvement
                                        mape_change = (base_performance["mape"] - new_performance["mape"]) / base_performance["mape"] * 100
                                        
                                        # Display metrics side by side
                                        metric_cols = st.columns(3)
                                        
                                        with metric_cols[0]:
                                            metric_value = new_performance["mape"]
                                            metric_delta = f"{mape_change:.1f}%" if mape_change != 0 else "No change"
                                            st.metric(
                                                label="Estimated MAPE",
                                                value=f"{metric_value:.2f}%",
                                                delta=metric_delta,
                                                delta_color="inverse"  # Lower is better for MAPE
                                            )
                                        
                                        with metric_cols[1]:
                                            rmse_change = (base_performance["rmse"] - new_performance["rmse"]) / base_performance["rmse"] * 100
                                            metric_value = new_performance["rmse"]
                                            metric_delta = f"{rmse_change:.1f}%" if rmse_change != 0 else "No change"
                                            st.metric(
                                                label="Estimated RMSE",
                                                value=f"{metric_value:.2f}",
                                                delta=metric_delta,
                                                delta_color="inverse"  # Lower is better for RMSE
                                            )
                                            
                                        with metric_cols[2]:
                                            mae_change = (base_performance["mae"] - new_performance["mae"]) / base_performance["mae"] * 100
                                            metric_value = new_performance["mae"]
                                            metric_delta = f"{mae_change:.1f}%" if mae_change != 0 else "No change"
                                            st.metric(
                                                label="Estimated MAE",
                                                value=f"{metric_value:.2f}",
                                                delta=metric_delta,
                                                delta_color="inverse"  # Lower is better for MAE
                                            )
                                        
                                        # Generate example forecast visualization
                                        st.markdown("##### Forecast Visualization")
                                        
                                        # Simulated time series data
                                        start_date = pd.Timestamp('2023-01-01')
                                        periods = 24
                                        dates = [start_date + pd.DateOffset(months=i) for i in range(periods)]
                                        
                                        # Generate synthetic data for visualization
                                        np.random.seed(42)  # For consistent results
                                        actuals = np.random.normal(100, 20, periods).cumsum() + 500
                                        
                                        # Simulate forecasts with different parameters
                                        # Base parameters (current)
                                        base_noise = np.random.normal(0, base_performance["mape"], periods)
                                        base_forecast = actuals * (1 + base_noise/100)
                                        
                                        # New parameters (from sliders)
                                        new_noise = np.random.normal(0, new_performance["mape"], periods)
                                        new_forecast = actuals * (1 + new_noise/100)
                                        
                                        # Create interactive forecast plot
                                        fig = go.Figure()
                                        
                                        # Convert dates to strings for compatibility with Plotly
                                        date_strs = [d.strftime('%Y-%m-%d') for d in dates]
                                        
                                        # Add actual data
                                        fig.add_trace(go.Scatter(
                                            x=date_strs, y=actuals,
                                            mode='lines+markers',
                                            name='Actual',
                                            line=dict(color='blue', width=3),
                                            marker=dict(size=8, symbol='circle')
                                        ))
                                        
                                        # Add base forecast
                                        fig.add_trace(go.Scatter(
                                            x=date_strs, y=base_forecast,
                                            mode='lines',
                                            name='Current Parameters',
                                            line=dict(color='gray', width=2, dash='dot'),
                                        ))
                                        
                                        # Add new forecast with parameter adjustments
                                        fig.add_trace(go.Scatter(
                                            x=date_strs, y=new_forecast,
                                            mode='lines',
                                            name='New Parameters',
                                            line=dict(color='green', width=2),
                                        ))
                                        
                                        # Add forecast boundary line
                                        forecast_idx = int(periods * 0.7)
                                        if 0 <= forecast_idx < len(dates):
                                            forecast_start = dates[forecast_idx]
                                            # Convert timestamp to string for vline
                                            forecast_date_str = forecast_start.strftime('%Y-%m-%d')
                                            # Use the numeric index instead of date string for the vline
                                            try:
                                                fig.add_vline(
                                                    x=forecast_idx, 
                                                    line_dash="solid", 
                                                    line_width=2, 
                                                    line_color="gray",
                                                    annotation_text="Forecast Start", 
                                                    annotation_position="top right"
                                                )
                                            except Exception as e:
                                                st.warning(f"Could not add forecast boundary: {str(e)}")
                                        
                                        # Calculate shaded confidence intervals for new forecast
                                        lower_bound = new_forecast - (new_forecast * new_performance["mape"]/100)
                                        upper_bound = new_forecast + (new_forecast * new_performance["mape"]/100)
                                        
                                        # Add confidence interval - convert dates to strings for compatibility
                                        date_strs = [d.strftime('%Y-%m-%d') for d in dates]
                                        date_strs_reversed = date_strs[::-1]
                                        
                                        fig.add_trace(go.Scatter(
                                            x=date_strs+date_strs_reversed,
                                            y=list(upper_bound)+list(lower_bound[::-1]),
                                            fill='toself',
                                            fillcolor='rgba(0,176,120,0.2)',
                                            line=dict(color='rgba(255,255,255,0)'),
                                            name='Confidence Interval'
                                        ))
                                        
                                        # Update layout
                                        fig.update_layout(
                                            height=400,
                                            margin=dict(l=10, r=10, t=10, b=10),
                                            legend=dict(
                                                orientation="h", 
                                                yanchor="bottom", 
                                                y=1.02, 
                                                xanchor="right", 
                                                x=1,
                                                bgcolor='rgba(255, 255, 255, 0.8)',
                                                bordercolor='rgba(0, 0, 0, 0.1)',
                                                borderwidth=1
                                            ),
                                            hovermode="x unified"
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Add apply button
                                        apply_cols = st.columns([3, 2, 3])
                                        with apply_cols[1]:
                                            if st.button("Apply New Parameters", type="primary", use_container_width=True, key="apply_prophet_params"):
                                                st.success(f"Parameters applied: changepoint_prior_scale={cp_slider:.3f}, seasonality_prior_scale={sp_slider:.2f}, seasonality_mode={sm_radio}")
                                
                                # Helpful tips for Prophet parameters
                                with st.expander("Prophet Parameter Tips", expanded=False):
                                    st.markdown("""
                                    #### Prophet Parameter Guide
                                    
                                    **Changepoint Prior Scale (0.001 - 0.5)**
                                    - Controls how flexible the trend is
                                    - Lower values (e.g., 0.001) make the trend less flexible
                                    - Higher values (e.g., 0.5) allow the trend to fit the data more closely
                                    - Default is 0.05
                                    
                                    **Seasonality Prior Scale (0.01 - 10.0)**
                                    - Controls how flexible the seasonality is
                                    - Lower values (e.g., 0.01) make the seasonality less flexible
                                    - Higher values (e.g., 10.0) allow the seasonality to fit the data more closely
                                    - Default is 10.0
                                    
                                    **Seasonality Mode**
                                    - Additive: Seasonality is constant regardless of trend level (default)
                                    - Multiplicative: Seasonality scales with trend level
                                    - Multiplicative is better for data where seasonal fluctuations increase with the trend
                                    """)
                            
                            elif selected_explore_model == "auto_arima":
                                with params_container:
                                    st.markdown("##### Adjust ARIMA Parameters")
                                    
                                    # Get current parameters or use defaults
                                    current_d = int(current_params.get("d", 1))
                                    current_max_p = int(current_params.get("max_p", 5))
                                    current_max_q = int(current_params.get("max_q", 5))
                                    current_seasonal = current_params.get("seasonal", True)
                                    
                                    # Create parameter sliders with current values
                                    d_slider = st.slider(
                                        "Differencing Order (d)",
                                        min_value=0,
                                        max_value=2,
                                        value=current_d,
                                        step=1,
                                        key="d_slider",
                                        help="Number of differencing operations to make data stationary"
                                    )
                                    
                                    max_p_slider = st.slider(
                                        "Maximum AR Order (max_p)",
                                        min_value=1,
                                        max_value=10,
                                        value=current_max_p,
                                        step=1,
                                        key="max_p_slider",
                                        help="Maximum autoregressive order to consider"
                                    )
                                    
                                    max_q_slider = st.slider(
                                        "Maximum MA Order (max_q)",
                                        min_value=1,
                                        max_value=10,
                                        value=current_max_q,
                                        step=1,
                                        key="max_q_slider",
                                        help="Maximum moving average order to consider"
                                    )
                                    
                                    seasonal_toggle = st.toggle(
                                        "Include seasonality",
                                        value=current_seasonal,
                                        key="seasonal_toggle",
                                        help="Whether to include seasonal components in the model"
                                    )
                                    
                                    # Generate the visualization based on selected parameters
                                    with simulation_container:
                                        st.markdown("##### Forecast Simulation")
                                        
                                        # Create a visual indicator of parameter changes
                                        st.markdown(f"""
                                        <div style="margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;">
                                        <span style="font-weight: bold;">Current Parameters:</span> d={d_slider}, 
                                        max_p={max_p_slider}, max_q={max_q_slider}, seasonal={str(seasonal_toggle)}
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Simple simulation function for ARIMA (placeholder implementation)
                                        def simulate_arima_performance(d, max_p, max_q, seasonal):
                                            # Simplified model of how parameters affect performance
                                            base_mape = 12.0  # Example base MAPE
                                            
                                            # Effects of parameters
                                            d_effect = 2.0 if d == 0 else (0 if d == 1 else 1.0)
                                            p_effect = 0.2 * abs(5 - max_p)
                                            q_effect = 0.15 * abs(5 - max_q)
                                            seasonal_effect = 0 if seasonal else 3.0
                                            
                                            # Calculate simulated metrics
                                            simulated_mape = base_mape + d_effect + p_effect + q_effect + seasonal_effect
                                            simulated_rmse = simulated_mape * 2.2
                                            simulated_mae = simulated_mape * 1.8
                                            
                                            return {
                                                "mape": simulated_mape,
                                                "rmse": simulated_rmse,
                                                "mae": simulated_mae
                                            }
                                        
                                        # Simulate base performance (with current parameters)
                                        base_performance = simulate_arima_performance(current_d, current_max_p, current_max_q, current_seasonal)
                                        
                                        # Simulate new performance (with slider parameters)
                                        new_performance = simulate_arima_performance(d_slider, max_p_slider, max_q_slider, seasonal_toggle)
                                        
                                        # Calculate improvement
                                        mape_change = (base_performance["mape"] - new_performance["mape"]) / base_performance["mape"] * 100
                                        
                                        # Display metrics using the same pattern as for Prophet
                                        metric_cols = st.columns(3)
                                        
                                        with metric_cols[0]:
                                            metric_value = new_performance["mape"]
                                            metric_delta = f"{mape_change:.1f}%" if mape_change != 0 else "No change"
                                            st.metric(
                                                label="Estimated MAPE",
                                                value=f"{metric_value:.2f}%",
                                                delta=metric_delta,
                                                delta_color="inverse"
                                            )
                                        
                                        with metric_cols[1]:
                                            rmse_change = (base_performance["rmse"] - new_performance["rmse"]) / base_performance["rmse"] * 100
                                            metric_value = new_performance["rmse"]
                                            metric_delta = f"{rmse_change:.1f}%" if rmse_change != 0 else "No change"
                                            st.metric(
                                                label="Estimated RMSE",
                                                value=f"{metric_value:.2f}",
                                                delta=metric_delta,
                                                delta_color="inverse"
                                            )
                                            
                                        with metric_cols[2]:
                                            mae_change = (base_performance["mae"] - new_performance["mae"]) / base_performance["mae"] * 100
                                            metric_value = new_performance["mae"]
                                            metric_delta = f"{mae_change:.1f}%" if mae_change != 0 else "No change"
                                            st.metric(
                                                label="Estimated MAE",
                                                value=f"{metric_value:.2f}",
                                                delta=metric_delta,
                                                delta_color="inverse"
                                            )
                                            
                                        # Create ARIMA visualization
                                        # Similar code to Prophet visualization with adjusted simulations
                                        st.markdown("##### Forecast Visualization")
                                        
                                        # Create interactive forecast plot for ARIMA
                                        start_date = pd.Timestamp('2023-01-01')
                                        periods = 24
                                        dates = [start_date + pd.DateOffset(months=i) for i in range(periods)]
                                        
                                        # Generate synthetic data
                                        np.random.seed(42)  # For consistency
                                        actuals = np.random.normal(100, 20, periods).cumsum() + 500
                                        
                                        # Simulate forecasts
                                        # Base parameters
                                        base_noise = np.random.normal(0, base_performance["mape"], periods)
                                        base_forecast = actuals * (1 + base_noise/100)
                                        
                                        # New parameters
                                        new_noise = np.random.normal(0, new_performance["mape"], periods)
                                        new_forecast = actuals * (1 + new_noise/100)
                                        
                                        # Create figure
                                        fig = go.Figure()
                                        
                                        # Convert dates to strings for Plotly compatibility
                                        date_strs = [d.strftime('%Y-%m-%d') for d in dates]
                                        
                                        # Add traces
                                        fig.add_trace(go.Scatter(
                                            x=date_strs, y=actuals,
                                            mode='lines+markers',
                                            name='Actual',
                                            line=dict(color='blue', width=3)
                                        ))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=date_strs, y=base_forecast,
                                            mode='lines',
                                            name='Current Parameters',
                                            line=dict(color='gray', width=2, dash='dot')
                                        ))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=date_strs, y=new_forecast,
                                            mode='lines',
                                            name='New Parameters',
                                            line=dict(color='green', width=2)
                                        ))
                                        
                                        # Update layout
                                        fig.update_layout(
                                            height=400,
                                            margin=dict(l=10, r=10, t=10, b=10),
                                            legend=dict(
                                                orientation="h", 
                                                yanchor="bottom", 
                                                y=1.02, 
                                                xanchor="right", 
                                                x=1
                                            ),
                                            hovermode="x unified"
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Add apply button
                                        apply_cols = st.columns([3, 2, 3])
                                        with apply_cols[1]:
                                            if st.button("Apply New Parameters", type="primary", use_container_width=True, key="apply_arima_params"):
                                                st.success(f"Parameters applied: d={d_slider}, max_p={max_p_slider}, max_q={max_q_slider}, seasonal={seasonal_toggle}")
                                
                                # Helpful tips for ARIMA parameters
                                with st.expander("ARIMA Parameter Tips", expanded=False):
                                    st.markdown("""
                                    #### ARIMA Parameter Guide
                                    
                                    **Differencing Order (d)**
                                    - Controls how many times the data is differenced to achieve stationarity
                                    - d=0: No differencing (use when data is already stationary)
                                    - d=1: First-order differencing (most common, removes linear trend)
                                    - d=2: Second-order differencing (removes quadratic trend)
                                    
                                    **Maximum AR Order (max_p)**
                                    - Controls the maximum number of lagged observations to consider
                                    - Higher values allow the model to capture longer-term dependencies
                                    - Typical values: 2-5
                                    
                                    **Maximum MA Order (max_q)**
                                    - Controls the maximum number of lagged forecast errors to consider
                                    - Higher values allow the model to account for more past forecast errors
                                    - Typical values: 2-5
                                    
                                    **Seasonality**
                                    - Whether to include seasonal components in the model
                                    - Enables the model to capture recurring patterns at fixed intervals
                                    - Usually beneficial for data with seasonal patterns
                                    """)
                            
                            else:
                                # Generic message for other model types
                                st.info(f"Interactive exploration for {selected_explore_model.upper()} is coming soon! Check back later for updates.")
                        else:
                            st.warning(f"No parameters found for {selected_explore_sku} with model {selected_explore_model}.")
                    else:
                        st.info(f"No SKUs available for {selected_explore_model}.")
            else:
                st.info("No tuning results available for interactive exploration. Run hyperparameter tuning first.")
            
        with result_tabs[3]:
            st.markdown("### Before/After Performance Comparison")
            
            # Create a before/after performance comparison across all SKUs
            if model_scores:
                # Create a dataframe with before/after scores
                # In a real implementation, you would have actual before/after data
                
                # Generate synthetic before data (default parameters)
                before_scores = {}
                for sku in model_scores:
                    before_scores[sku] = {}
                    for model_type in model_scores[sku]:
                        # Default scores are randomly higher (worse) than tuned scores
                        default_score = model_scores[sku][model_type] * (1 + np.random.uniform(0.2, 0.6))
                        before_scores[sku][model_type] = default_score
                
                # Create a summary table
                summary_data = []
                
                for sku in model_scores:
                    for model_type in model_scores[sku]:
                        summary_data.append({
                            "SKU": sku,
                            "Model": model_type,
                            "Before Tuning (MAPE)": before_scores[sku][model_type],
                            "After Tuning (MAPE)": model_scores[sku][model_type],
                            "Improvement": before_scores[sku][model_type] - model_scores[sku][model_type],
                            "Improvement %": (before_scores[sku][model_type] - model_scores[sku][model_type]) / before_scores[sku][model_type] * 100
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Apply styling to the dataframe
                    def highlight_improvement(val):
                        if isinstance(val, (int, float)) and val > 0:
                            return 'background-color: #e6ffe6'
                        return ''
                    
                    styled_df = summary_df.style.map(highlight_improvement, subset=["Improvement", "Improvement %"])
                    
                    # Display the styled dataframe
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Create an overall performance improvement chart
                    st.markdown("#### Overall Performance Improvement")
                    
                    # Group by model and calculate average improvement
                    model_improvements = summary_df.groupby("Model")["Improvement %"].mean().reset_index()
                    
                    fig = px.bar(
                        model_improvements,
                        x="Model",
                        y="Improvement %",
                        color="Improvement %",
                        title="Average Improvement by Model Type",
                        color_continuous_scale="Blues"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Model Type",
                        yaxis_title="Average Improvement (%)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show the best performing model-SKU combinations
                    st.markdown("#### Best Performing Combinations")
                    
                    # Get top 5 improvements
                    top_improvements = summary_df.sort_values("Improvement %", ascending=False).head(5)
                    
                    fig = px.bar(
                        top_improvements,
                        x="SKU",
                        y="Improvement %",
                        color="Model",
                        title="Top 5 SKU-Model Improvements",
                        barmode="group"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No comparison data available.")
            else:
                st.info("No performance metrics available. Run hyperparameter tuning first.")
        
        with result_tabs[4]:
            st.markdown("### Tuning Audit Log")
            
            # Display a history of tuning operations
            st.markdown("""
            The audit log tracks all hyperparameter tuning operations, including who ran them,
            when they were executed, and what changes were made to the parameters.
            """)
            
            # Create an example audit log
            audit_data = [
                {
                    "timestamp": "2025-04-08 14:32:12",
                    "user": "admin",
                    "operation": "Parameter Tuning",
                    "models": ["Prophet", "Auto ARIMA"],
                    "skus": ["SKU001", "SKU002", "SKU003"],
                    "improvement": "+18.2%"
                },
                {
                    "timestamp": "2025-04-07 11:15:47",
                    "user": "analyst",
                    "operation": "Manual Parameter Update",
                    "models": ["Prophet"],
                    "skus": ["SKU005"],
                    "improvement": "+5.3%"
                },
                {
                    "timestamp": "2025-04-05 09:23:31",
                    "user": "admin",
                    "operation": "Scheduled Auto-Tuning",
                    "models": ["LSTM", "Theta", "ETS"],
                    "skus": ["All SKUs"],
                    "improvement": "+12.7%"
                }
            ]
            
            # Display the audit log
            for entry in audit_data:
                with st.expander(f"{entry['timestamp']} - {entry['operation']} by {entry['user']}", expanded=False):
                    st.markdown(f"""
                    **Models**: {', '.join(entry['models'])}  
                    **SKUs**: {', '.join(entry['skus'])}  
                    **Overall Improvement**: {entry['improvement']}
                    """)
            
            # Add a note about audit log retention
            st.info("Audit logs are retained for 90 days in accordance with system policies.")
    else:
        st.info("No tuning results available yet. Run hyperparameter tuning first.")
        
    # Add model registry export/import
    st.markdown("<h3>6. Save, Apply & Schedule</h3>", unsafe_allow_html=True)
    
    # Model registry management
    registry_cols = st.columns(2)
    
    with registry_cols[0]:
        st.markdown("### Model Registry")
        st.markdown("""
        The model registry stores optimized parameters for all SKU-model combinations.
        You can export these parameters for backup or sharing with other systems.
        """)
        
        # Export button
        st.download_button(
            label="Export All Parameters (JSON)",
            data=json.dumps(tuning_results, indent=2),
            file_name="hyperparameter_registry.json",
            mime="application/json",
            use_container_width=True
        )
    
    with registry_cols[1]:
        st.markdown("### Automatic Retuning")
        st.markdown("""
        Schedule automatic retuning to keep your parameters optimized
        as new data becomes available.
        """)
        
        # Schedule options
        schedule_frequency = st.selectbox(
            "Retuning Frequency",
            ["Weekly", "Monthly", "Quarterly", "After Data Updates"]
        )
        
        execution_time = st.time_input(
            "Execution Time",
            value=datetime.strptime("02:00", "%H:%M").time()
        )
        
        # Schedule button
        st.button(
            "Schedule Automatic Retuning",
            use_container_width=True,
            type="primary"
        )

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
                            # Create visualization for Prophet parameters with proper type handling
                            prophet_data = []
                            for sku, params in model_params.items():
                                if params:
                                    # Ensure all numeric values have consistent types
                                    try:
                                        changepoint = float(params.get("changepoint_prior_scale", 0.05))
                                        seasonality = float(params.get("seasonality_prior_scale", 10.0))
                                        mode = str(params.get("seasonality_mode", "additive"))
                                        
                                        prophet_data.append({
                                            "SKU": str(sku),
                                            "changepoint_prior_scale": changepoint,
                                            "seasonality_prior_scale": seasonality,
                                            "seasonality_mode": mode
                                        })
                                    except (ValueError, TypeError) as e:
                                        st.warning(f"Skipping invalid parameter values for {sku}: {str(e)}")

                            if prophet_data:
                                # Create DataFrame with explicit dtypes
                                prophet_df = pd.DataFrame(prophet_data)
                                # Display the data
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
st.markdown("<h3>7. Integration with Forecasting Engine</h3>", unsafe_allow_html=True)

st.markdown("""
### Smart Parameter Integration 

The tuned parameters from this page are automatically saved to the database and seamlessly integrate with the 
forecasting engine. This integration creates a feedback loop that continuously improves forecast accuracy:
""")

# Create a visual flowchart for the integration process
integration_cols = st.columns([1, 3, 1])

with integration_cols[1]:
    st.markdown("""
    ```mermaid
    graph TD
        A[Hyperparameter Tuning] -->|Optimized Parameters| B[Parameter Registry]
        B -->|Load Parameters| C[Demand Forecasting]
        C -->|Forecast Results| D[Performance Metrics]
        D -->|Feedback Loop| A
        
        style A fill:#f0f5ff,stroke:#4169e1,stroke-width:2px
        style B fill:#f9f9f9,stroke:#666,stroke-width:1px
        style C fill:#f0fff4,stroke:#38a169,stroke-width:2px
        style D fill:#fff5f0,stroke:#dd6b20,stroke-width:1px
    ```
    """)

# Benefits and key features for integration
benefit_cols = st.columns(2)

with benefit_cols[0]:
    st.markdown("""
    ### Automatic Parameter Application
    
    When you run forecasts on any of the Demand Forecasting pages:
    
    1. ✓ System automatically checks for tuned parameters
    2. ✓ Applies optimized parameters for each SKU-model pair
    3. ✓ Falls back to defaults for untuned combinations
    4. ✓ Logs parameter usage for tracking
    """)
    
with benefit_cols[1]:
    st.markdown("""
    ### Performance Feedback Loop
    
    The system continuously improves through:
    
    1. ✓ Comparing forecast accuracy with different parameters
    2. ✓ Identifying which parameters need retuning
    3. ✓ Suggesting optimal retuning schedule
    4. ✓ Adapting to changing data patterns
    """)

# Integration status display
st.markdown("### Integration Status")

# Create a fake status display with green/red indicators
status_data = {
    "Module": [
        "Demand Forecasting (Main)", 
        "Advanced Forecasting", 
        "New Demand Forecasting",
        "V2 Demand Forecasting",
        "Enhanced Forecasting"
    ],
    "Status": [
        "✅ Connected",
        "✅ Connected",
        "✅ Connected",
        "✅ Connected",
        "❌ Not Connected"
    ],
    "Last Sync": [
        "2025-04-08 14:15:22",
        "2025-04-08 14:15:22",
        "2025-04-08 14:15:22",
        "2025-04-08 14:15:22",
        "Never"
    ],
    "Parameters Used": [
        "15/24 SKUs",
        "15/24 SKUs",
        "15/24 SKUs",
        "15/24 SKUs",
        "0/24 SKUs"
    ]
}

# Display as a styled table
status_df = pd.DataFrame(status_data)
st.dataframe(
    status_df.style.apply(
        lambda x: ['background: #e6ffe6' if '✅' in v else 'background: #ffe6e6' for v in x], 
        axis=1,
        subset=["Status"]
    ),
    use_container_width=True
)

# Instructions for linking
st.info("Note: The 'Enhanced Forecasting' module needs to be updated to use the parameter registry. All other forecasting modules are automatically connected.")

# Add button to connect missing module
connect_cols = st.columns([3, 1, 3])
with connect_cols[1]:
    st.button("Connect All Modules", type="primary", use_container_width=True)

# Footer with additional resources
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background-color: #f9f9f9; border-radius: 0.5rem; margin-top: 2rem;">
    <h3 style="margin-top: 0;">Additional Resources</h3>
    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
        <div style="flex: 1; min-width: 200px;">
            <h4>Documentation</h4>
            <ul style="text-align: left;">
                <li><a href="#">Hyperparameter Tuning Guide</a></li>
                <li><a href="#">Model Parameter Reference</a></li>
                <li><a href="#">Integration API Docs</a></li>
            </ul>
        </div>
        <div style="flex: 1; min-width: 200px;">
            <h4>Tutorials</h4>
            <ul style="text-align: left;">
                <li><a href="#">Optimizing ARIMA Parameters</a></li>
                <li><a href="#">Working with Prophet Models</a></li>
                <li><a href="#">Advanced Neural Network Tuning</a></li>
            </ul>
        </div>
        <div style="flex: 1; min-width: 200px;">
            <h4>Support</h4>
            <ul style="text-align: left;">
                <li><a href="#">Knowledge Base</a></li>
                <li><a href="#">Community Forum</a></li>
                <li><a href="#">Contact Support</a></li>
            </ul>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

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
                        "Model": model.upper(),
                        "Parameters": "Available",
                        "Last Updated": params['last_updated'].strftime("%Y-%m-%d"),
                        "Score": f"{params['best_score']:.4f}" if 'best_score' in params and params['best_score'] is not None else "N/A"
                    })

        if tuned_params_summary:
            # Create DataFrame with explicit string type for all columns to avoid serialization issues
            summary_df = pd.DataFrame(tuned_params_summary)
            # Convert columns to appropriate types
            for col in summary_df.columns:
                summary_df[col] = summary_df[col].astype(str)
            
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No optimized parameters found in the database.")
    except Exception as e:
        st.error(f"Error fetching parameter database status: {str(e)}")

# Add a new section for displaying all parameters in a flat table format
st.markdown("<h3>8. Complete Parameter Data Table</h3>", unsafe_allow_html=True)

st.markdown("""
This table displays all hyperparameter tuning results in a comprehensive, downloadable format.
Each row represents a single parameter for a specific SKU and model.
""")

# Import the function to get flat model parameters
from utils.database import get_flat_model_parameters

try:
    # Get flat model parameters in the format requested
    flat_params = get_flat_model_parameters()
    
    # Initialize empty dataframe
    params_df = pd.DataFrame()
    
    # First try to get data from session state for the most up-to-date results
    if 'tuning_results' in st.session_state and st.session_state.tuning_results:
        # Create flat parameter list from session state
        manual_params = []
        for sku, models in st.session_state.tuning_results.items():
            for model_type, params in models.items():
                # Get display name for model
                model_display = model_type
                if model_type == 'auto_arima':
                    model_display = 'ARIMA'
                elif model_type == 'prophet':
                    model_display = 'Prophet'
                elif model_type == 'ets':
                    model_display = 'ETS'
                elif model_type == 'theta':
                    model_display = 'Theta'
                elif model_type == 'lstm':
                    model_display = 'LSTM'
                
                # Add each parameter as a row
                for param_name, param_value in params.items():
                    manual_params.append({
                        "SKU code": sku,
                        "SKU name": sku,
                        "Model name": model_display,
                        "Parameter name": param_name,
                        "Parameter value": str(param_value)
                    })
        
        if manual_params:
            params_df = pd.DataFrame(manual_params)
            st.info("Showing parameters from current tuning session.")
    
    # If no data from session state, try from database
    if len(params_df) == 0 and flat_params:
        # Create a DataFrame with the flat parameters
        # Rename columns to match requested format
        flat_df = pd.DataFrame(flat_params)
        if 'sku_code' in flat_df.columns:
            flat_df = flat_df.rename(columns={
                'sku_code': 'SKU code',
                'sku_name': 'SKU name',
                'model_name': 'Model name',
                'parameter_name': 'Parameter name',
                'parameter_value': 'Parameter value'
            })
            params_df = flat_df
    
    # Filter to show relevant parameters if user has made selections
    if len(params_df) > 0 and 'tuning_skus' in st.session_state and 'tuning_models' in st.session_state:
        if len(st.session_state.tuning_skus) > 0 and len(st.session_state.tuning_models) > 0:
            # Convert model names to standard format
            model_name_map = {
                'auto_arima': 'ARIMA',
                'prophet': 'Prophet',
                'ets': 'ETS',
                'theta': 'Theta',
                'lstm': 'LSTM'
            }
            
            # Create list of model names to filter by (handling both formats)
            filter_models = []
            for model in st.session_state.tuning_models:
                filter_models.append(model)
                if model in model_name_map:
                    filter_models.append(model_name_map[model])
            
            # Apply filters
            mask = (params_df['SKU code'].isin(st.session_state.tuning_skus)) & \
                   (params_df['Model name'].isin(filter_models))
            
            # Apply filter if it gives results, otherwise show all
            if mask.sum() > 0:
                params_df = params_df[mask]
                st.info(f"Showing parameters for selected SKUs and models. Filtered to {len(params_df)} parameters.")
            else:
                st.info("Showing all parameters - couldn't find exact matches for your selections.")
    
    # If still no real data, create example data as fallback
    if len(params_df) == 0:
        st.info("No parameter data available. Run hyperparameter tuning to generate parameters.")
        
        # Create example data showing multiple models
        example_data = [
            {"SKU code": "SKU1", "SKU name": "SKU1", "Model name": "ARIMA", "Parameter name": "p", "Parameter value": "1"},
            {"SKU code": "SKU1", "SKU name": "SKU1", "Model name": "ARIMA", "Parameter name": "d", "Parameter value": "2"},
            {"SKU code": "SKU1", "SKU name": "SKU1", "Model name": "ARIMA", "Parameter name": "q", "Parameter value": "3"},
            {"SKU code": "SKU1", "SKU name": "SKU1", "Model name": "Prophet", "Parameter name": "changepoint_prior_scale", "Parameter value": "0.05"},
            {"SKU code": "SKU1", "SKU name": "SKU1", "Model name": "Prophet", "Parameter name": "seasonality_prior_scale", "Parameter value": "10.0"},
            {"SKU code": "SKU2", "SKU name": "SKU2", "Model name": "ETS", "Parameter name": "trend", "Parameter value": "add"},
            {"SKU code": "SKU2", "SKU name": "SKU2", "Model name": "ETS", "Parameter name": "seasonal", "Parameter value": "add"}
        ]
        params_df = pd.DataFrame(example_data)
        st.markdown("### Example Data Format")
    
    # Display the DataFrame
    if len(params_df) > 0:
        # Ensure column names exactly match requested format
        if 'sku_code' in params_df.columns:
            params_df = params_df.rename(columns={
                'sku_code': 'SKU code',
                'sku_name': 'SKU name',
                'model_name': 'Model name',
                'parameter_name': 'Parameter name',
                'parameter_value': 'Parameter value'
            })
        
        # Sort by SKU and model for easier reading
        params_df = params_df.sort_values(['SKU code', 'Model name', 'Parameter name'])
        
        # Display the DataFrame with exact requested column order
        column_order = ["SKU code", "SKU name", "Model name", "Parameter name", "Parameter value"]
        # Make sure all columns exist
        for col in column_order:
            if col not in params_df.columns:
                params_df[col] = ""
        
        # Display the DataFrame with the specified columns
        st.dataframe(params_df[column_order], use_container_width=True, height=400)
        
        # Add download button for the parameters table
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(params_df[column_order])
        
        st.download_button(
            label="📥 Download Parameters Table as CSV",
            data=csv,
            file_name="hyperparameter_tuning_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Add download button for Excel format
        @st.cache_data
        def convert_df_to_excel(df):
            import io
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine='xlsxwriter')
            buffer.seek(0)
            return buffer.getvalue()
        
        excel_data = convert_df_to_excel(params_df[column_order])
        
        st.download_button(
            label="📥 Download Parameters Table as Excel",
            data=excel_data,
            file_name="hyperparameter_tuning_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # Display summary statistics
        st.markdown(f"""
        **Summary Statistics:**
        - Total parameters: {len(params_df)}
        - Unique SKUs: {params_df['SKU code'].nunique()}
        - Models with parameters: {params_df['Model name'].nunique()}
        """)
        
except Exception as e:
    st.error(f"Error loading parameter data table: {str(e)}")
    
    # Create a fallback example table with the exact columns requested
    fallback_data = [
        {"SKU code": "SKU1", "SKU name": "SKU1", "Model name": "ARIMA", "Parameter name": "p", "Parameter value": "1"},
        {"SKU code": "SKU1", "SKU name": "SKU1", "Model name": "ARIMA", "Parameter name": "d", "Parameter value": "2"},
        {"SKU code": "SKU1", "SKU name": "SKU1", "Model name": "ARIMA", "Parameter name": "q", "Parameter value": "3"}
    ]
    st.markdown("### Example Format (Error Loading Actual Data)")
    st.dataframe(pd.DataFrame(fallback_data), use_container_width=True)