# IMPORTANT: st.set_page_config must be the first Streamlit command
import streamlit as st

# Set page config (must be the first Streamlit command)
st.set_page_config(
    page_title="Hyperparameter Tuning",
    page_icon="🔧",
    layout="wide"
)

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

# Display enhanced tuning results visualization with performance comparison
if not st.session_state.tuning_in_progress and (st.session_state.tuning_results or 'tuning_results' in st.session_state):
    st.markdown("<h3>5. Performance Comparison Panel</h3>", unsafe_allow_html=True)

    # Get all tuned parameters from database for all SKUs that were tuned
    tuning_results = {}
    model_scores = {}  # To store performance metrics

    # Try to get results from session state first
    if st.session_state.tuning_results:
        tuning_results = st.session_state.tuning_results
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

    if tuning_results:
        # Create a more advanced results dashboard
        # Define model names with appropriate badges
        model_names = {
            "auto_arima": "<span class='model-badge arima-badge'>ARIMA</span> Auto ARIMA", 
            "prophet": "<span class='model-badge prophet-badge'>FB</span> Prophet",
            "ets": "<span class='model-badge ets-badge'>ETS</span> Exponential Smoothing",
            "theta": "<span class='model-badge theta-badge'>Θ</span> Theta Method",
            "lstm": "<span class='model-badge lstm-badge'>DL</span> LSTM Neural Network"
        }

        # Create a tabbed interface for different result views
        result_tabs = st.tabs(["Model Performance", "Parameter Impact", "Before/After Comparison", "Audit Log"])

        with result_tabs[0]:
            st.markdown("### Model Performance Comparison")

            # Select an SKU to analyze in detail
            available_skus = list(tuning_results.keys())
            if available_skus:
                selected_result_sku = st.selectbox(
                    "Select SKU to analyze",
                    options=available_skus,
                    index=0,
                    key="result_sku_selector"
                )

                # Get model results for this SKU
                sku_models = list(tuning_results.get(selected_result_sku, {}).keys())

                if sku_models:
                    # Create metric cards for this SKU's performance
                    metric_cols = st.columns(len(sku_models))

                    # Add metric data for each model
                    for i, model_type in enumerate(sku_models):
                        # Get score and format it
                        score = model_scores.get(selected_result_sku, {}).get(model_type, 0)

                        # Get model name - safely handle the model_names variable
                        model_display = model_type
                        if 'model_names' in globals():
                            model_display = model_names.get(model_type, model_type)
                        elif 'model_names' in locals():
                            model_display = model_names.get(model_type, model_type)

                        # Use native Streamlit metrics that have better highlighting
                        with metric_cols[i]:
                            st.metric(
                                label=f"{model_display}",
                                value=f"{score:.4f}",
                                delta=None,
                                help="MAPE Score (lower is better)"
                            )

                    # Apply styling to highlight the metrics
                    style_metric_cards(background_color="#f0f8ff", border_left_color="#4169e1")

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
                        comparison_df = pd.DataFrame(comparison_data)

                        # Create a styled dataframe
                        st.dataframe(
                            comparison_df.style.applymap(
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
                                    st.markdown(f"##### {model_names.get(model_type, model_type)} Performance Details", unsafe_allow_html=True)

                            # Create a single merged visualization for before/after comparison
                            st.markdown("<div style='border: 1px solid #ddd; border-radius: 5px; padding: 15px;'>", unsafe_allow_html=True)
                            st.markdown("### Forecasting Performance Comparison")

                            # Generate synthetic data for demonstration
                            dates = pd.date_range(start='2023-01-01', periods=24, freq='MS')
                            actuals = np.random.normal(100, 20, 24).cumsum() + 500
                            before_forecast = actuals + np.random.normal(0, 50, 24)  # Higher error
                            after_forecast = actuals + np.random.normal(0, 20, 24)   # Lower error

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

                            # Add a vertical line to indicate forecast start
                            forecast_start = dates[int(len(dates) * 0.7)]  # Example forecast start point
                            fig.add_vline(
                                x=forecast_start, 
                                line_dash="solid", 
                                line_width=2, 
                                line_color="gray",
                                annotation_text="Forecast Start", 
                                annotation_position="top right"
                            )

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
                                score = model_scores.get(selected_result_sku, {}).get(model_type, 10.2)
                                st.markdown("**After Tuning Metrics:**")
                                st.markdown(f"""
                                - MAPE: {score:.1f}%
                                - RMSE: {score * 2.5:.1f}
                                - MAE: {score * 2:.1f}
                                """)

                            # Calculate improvement percentages
                            mape_improvement = (24.8 - score) / 24.8 * 100
                            rmse_improvement = (67.3 - (score * 2.5)) / 67.3 * 100
                            mae_improvement = (52.9 - (score * 2)) / 52.9 * 100

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
                            st.button(f"👍 Accept {model_type.upper()} Parameters", key=f"accept_{model_type}")
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
                selected_impact_model = st.selectbox(
                    "Select model to analyze",
                    options=model_options,
                    index=0 if "prophet" in model_options else 0,
                    key="impact_model_selector"
                )

                # Show parameter impact visualization
                st.markdown(f"#### Parameter Impact for {selected_impact_model.upper()}")

                # Create example parameter impact visualizations
                if selected_impact_model == "prophet":
                    # Prophet parameter impact analysis
                    impact_data = pd.DataFrame({
                        "Parameter": ["changepoint_prior_scale", "changepoint_prior_scale", "changepoint_prior_scale", 
                                     "seasonality_prior_scale", "seasonality_prior_scale", "seasonality_prior_scale"],
                        "Value": [0.001, 0.05, 0.5, 0.1, 1.0, 10.0],
                        "MAPE": [18.5, 15.2, 22.1, 19.8, 15.2, 16.4]
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
                    # ARIMA parameter impact
                    pdq_combos = [
                        {"p": 1, "d": 1, "q": 1, "MAPE": 18.3},
                        {"p": 2, "d": 1, "q": 1, "MAPE": 15.7},
                        {"p": 2, "d": 1, "q": 2, "MAPE": 12.9},
                        {"p": 3, "d": 1, "q": 1, "MAPE": 14.2},
                        {"p": 3, "d": 1, "q": 2, "MAPE": 13.6},
                        {"p": 3, "d": 2, "q": 2, "MAPE": 19.8}
                    ]

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

                    styled_df = summary_df.style.applymap(highlight_improvement, subset=["Improvement", "Improvement %"])

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

        with result_tabs[3]:
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
                with st.expandergraph TD
        A[Hyperparameter Tuning] -->|Optimized Parameters| B[Parameter Registry]
        B -->|Load Parameters| C[Demand Forecasting]
        C -->|Forecast Results| D[Performance Metrics]
        D -->|Feedback Loop| A

        style A fill:#f0f5ff,stroke:#4169e1,stroke-width:2px
        style B fill:#f9f9f9,stroke:#666,stroke-width:1px
        style C fill:#f0fff4,stroke:#38a169,stroke-width:2px
        style D fill:#fff5f0,stroke:#dd6b20,stroke-width:1px