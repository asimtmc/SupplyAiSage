import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime
from utils.data_processor import process_sales_data
from utils.forecast_engine import extract_features, cluster_skus, generate_forecasts
from utils.visualization import plot_forecast, plot_cluster_summary

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
if 'show_all_clusters' not in st.session_state:
    st.session_state.show_all_clusters = False

# Create sidebar for settings
with st.sidebar:
    st.header("Forecast Settings")
    
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
    
    # Run forecast button
    if st.button("Run Forecast Analysis"):
        st.session_state.run_forecast = True
        with st.spinner("Running forecast analysis..."):
            # Extract time series features for clustering
            features_df = extract_features(st.session_state.sales_data)
            
            # Cluster SKUs
            st.session_state.clusters = cluster_skus(features_df, n_clusters=num_clusters)
            
            # Generate forecasts with model evaluation
            st.session_state.forecasts = generate_forecasts(
                st.session_state.sales_data,
                st.session_state.clusters,
                forecast_periods=st.session_state.forecast_periods,
                evaluate_models_flag=evaluate_models_flag,
                models_to_evaluate=models_to_evaluate
            )
            
            st.success(f"Successfully generated forecasts for {len(st.session_state.forecasts)} SKUs!")

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
    
    # Allow user to select a SKU to view detailed forecast
    sku_list = list(st.session_state.forecasts.keys())
    selected_sku = st.selectbox(
        "Select a SKU to view forecast details",
        options=sku_list,
        index=0 if st.session_state.selected_sku is None else sku_list.index(st.session_state.selected_sku)
    )
    st.session_state.selected_sku = selected_sku
    
    # Show forecast details for selected SKU
    if selected_sku:
        forecast_data = st.session_state.forecasts[selected_sku]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display forecast chart
            forecast_fig = plot_forecast(st.session_state.sales_data, forecast_data, selected_sku)
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            # If model evaluation results are available, show them
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
                st.dataframe(metrics_df.sort_values('RMSE'), use_container_width=True)
                
                # Add explanation
                st.info("The system evaluated all selected models on historical test data and automatically selected the best performing model for this SKU. Lower RMSE and MAPE values indicate better forecast accuracy.")
        
        with col2:
            # Show forecast details
            st.subheader("Forecast Details")
            
            st.markdown(f"**SKU:** {selected_sku}")
            st.markdown(f"**Cluster:** {forecast_data['cluster_name']}")
            st.markdown(f"**Model Used:** {forecast_data['model'].upper()}")
            
            # Forecast confidence
            confidence_color = "green" if forecast_data['model'] != 'moving_average' else "orange"
            confidence_text = "High" if forecast_data['model'] != 'moving_average' else "Medium"
            st.markdown(f"**Forecast Confidence:** <span style='color:{confidence_color}'>{confidence_text}</span>", unsafe_allow_html=True)
            
            # Forecast table
            forecast_table = pd.DataFrame({
                'Date': forecast_data['forecast'].index,
                'Forecast': forecast_data['forecast'].values.round(0),
                'Lower Bound': forecast_data['lower_bound'].values.round(0),
                'Upper Bound': forecast_data['upper_bound'].values.round(0)
            })
            
            st.dataframe(forecast_table, use_container_width=True)
    
    # Forecast export
    st.header("Export Forecasts")
    
    # Prepare forecast data for export
    if st.button("Prepare Forecast Export"):
        with st.spinner("Preparing forecast data..."):
            # Create a DataFrame with all forecasts
            export_data = []
            
            for sku, forecast_data in st.session_state.forecasts.items():
                for date, value in forecast_data['forecast'].items():
                    lower = forecast_data['lower_bound'].get(date, 0)
                    upper = forecast_data['upper_bound'].get(date, 0)
                    
                    export_data.append({
                        'sku': sku,
                        'date': date,
                        'forecast': round(value),
                        'lower_bound': round(lower),
                        'upper_bound': round(upper),
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
    # Show instructions when no forecast has been run
    st.info("👈 Please configure and run the forecast analysis using the sidebar.")
    
    # Show a preview of the sales data
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
