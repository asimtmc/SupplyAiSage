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

def forecast_progress_callback(current_index, current_sku, total_skus):
    """Callback function to update progress during forecasting"""
    # Calculate progress percentage
    progress = min(float(current_index) / total_skus, 1.0)
    st.session_state.advanced_forecast_progress = progress
    st.session_state.advanced_current_sku = current_sku

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
        forecast_tabs = st.tabs(["Forecast Chart", "Model Comparison", "Forecast Metrics", "Pattern Analysis", "Sense Check"])

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

    # Show comprehensive data table
    st.header("Comprehensive Forecast Data Table")
    
    # Prepare comprehensive data table
    if st.session_state.advanced_forecasts:
        # Create a dataframe to store all SKUs data with reoriented structure
        all_sku_data = []
        
        # Get historical dates (use the first forecast as reference for dates)
        first_sku = list(st.session_state.advanced_forecasts.keys())[0]
        first_forecast = st.session_state.advanced_forecasts[first_sku]
        
        # Make sure we have train data to extract historical dates
        if 'train_set' in first_forecast:
            # Identify unique dates in historical data
            historical_dates = pd.to_datetime(sorted(st.session_state.sales_data['date'].unique()))
            
            # Limit to a reasonable number of historical columns (e.g., last 6 months)
            if len(historical_dates) > 6:
                historical_dates = historical_dates[-6:]
            
            # Format dates for column names
            historical_cols = [date.strftime('%-d %b %Y') for date in historical_dates]
            
            # Get future dates from the first forecast
            future_dates = first_forecast['forecast'].index
            future_cols = [date.strftime('%-d %b %Y') for date in future_dates]
            
            # Column header formatting
            def highlight_data_columns(df):
                # Apply formatting to distinguish historical and forecast columns
                styles = []
                for col in df.columns:
                    if col in ['SKU', 'Model', 'Cluster']:
                        styles.append('')
                    elif col in historical_cols:
                        styles.append('background-color: rgba(144, 238, 144, 0.2)')  # Light green for historical
                    else:
                        styles.append('background-color: rgba(135, 206, 250, 0.2)')  # Light blue for forecast
                return styles
            
            # Process each SKU
            for sku, forecast in st.session_state.advanced_forecasts.items():
                # Create a row for this SKU
                sku_row = {
                    'SKU': sku,
                    'Model': forecast['model'].upper(),
                    'Cluster': forecast['cluster_name']
                }
                
                # Add historical values
                sku_historical = st.session_state.sales_data[st.session_state.sales_data['sku'] == sku]
                for date in historical_dates:
                    date_str = date.strftime('%-d %b %Y')
                    matching_rows = sku_historical[sku_historical['date'] == date]
                    if not matching_rows.empty:
                        sku_row[date_str] = matching_rows['quantity'].iloc[0]
                    else:
                        sku_row[date_str] = None
                
                # Add forecast values
                for date in future_dates:
                    date_str = date.strftime('%-d %b %Y')
                    sku_row[date_str] = forecast['forecast'][date]
                
                all_sku_data.append(sku_row)
            
            # Create DataFrame and style it
            if all_sku_data:
                all_sku_df = pd.DataFrame(all_sku_data)
                
                # Apply styling to the header
                styled_df = all_sku_df.style.apply(highlight_data_columns, axis=1)
                
                # Create an expander for the table
                with st.expander("Forecast Data Table (Click to Expand)", expanded=False):
                    # Create two columns for the table and download button
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.dataframe(styled_df, use_container_width=True)
                    
                    with col2:
                        # CSV download
                        csv_buffer = io.BytesIO()
                        all_sku_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        st.download_button(
                            label="📄 Download as CSV",
                            data=csv_buffer,
                            file_name=f"advanced_forecast_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            help="Download a CSV file with the table data"
                        )
            else:
                st.warning("No data available for the selected SKUs.")
        else:
            st.warning("No training data available to construct the comprehensive data table. Please upload sales data first.")
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

# Create a placeholder for the progress bar
progress_placeholder = st.empty()

# Create a run button with a unique key
if should_show_button and st.sidebar.button(
    forecast_button_text, 
    key="run_advanced_forecast_button",
    use_container_width=True
):
    # Set forecast in progress flag
    st.session_state.advanced_forecast_in_progress = True
    st.session_state.advanced_forecast_progress = 0
    st.session_state.run_advanced_forecast = True

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
                
                # Add to log if there's a message
                if message:
                    add_log_message(f"[{current_sku}] {message}", level)
                else:
                    # Default message
                    add_log_message(f"Processing SKU: {current_sku} ({current_index+1}/{total_skus})", "info")
                
                # Update UI elements
                progress_bar.progress(progress)
                progress_percentage = int(progress * 100)
                progress_details.markdown(f"""
                **Progress:** {progress_percentage}%  
                **Current SKU:** {current_sku} ({current_index+1}/{total_skus})  
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
                forecasts = advanced_generate_forecasts(
                    sales_data=sales_data,
                    cluster_info=cluster_info,
                    forecast_periods=st.session_state.advanced_forecast_periods,
                    auto_select=True,
                    models_to_evaluate=selected_models,
                    selected_skus=selected_skus,
                    progress_callback=enhanced_progress_callback,
                    hyperparameter_tuning=st.session_state.advanced_hyperparameter_tuning,
                    apply_sense_check=st.session_state.advanced_apply_sense_check
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