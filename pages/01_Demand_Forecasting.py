import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
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
if 'show_all_clusters' not in st.session_state:
    st.session_state.show_all_clusters = False
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'sku_options' not in st.session_state:
    st.session_state.sku_options = []

# Create sidebar for settings
with st.sidebar:
    st.header("Forecast Settings")
    
    # Get available SKUs from sales data (always available)
    if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
        st.subheader("SKU Selection")
        
        # Get available SKUs from the sales data
        available_skus = sorted(st.session_state.sales_data['sku'].unique().tolist())
        
        # If we have forecasts, use those SKUs instead
        if st.session_state.run_forecast and 'forecasts' in st.session_state and st.session_state.forecasts:
            sku_list = sorted(list(st.session_state.forecasts.keys()))
        else:
            sku_list = available_skus
            
        st.session_state.sku_options = sku_list
        
        # Select SKUs (multi-select if multiple SKUs should be shown)
        selected_skus = st.multiselect(
            "Select SKUs to Analyze",
            options=sku_list,
            default=st.session_state.selected_sku if st.session_state.selected_sku in sku_list else sku_list[0] if sku_list else None,
            help="Select one or more SKUs to analyze or forecast"
        )
        
        # Store all selected SKUs in session state
        st.session_state.selected_skus = selected_skus
        
        # Update selected primary SKU in session state if selection changed
        if selected_skus:
            st.session_state.selected_sku = selected_skus[0]  # Keep first as primary
    
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
    
    # Extract selected SKUs from the multiselect
    selected_skus_to_forecast = []
    if 'selected_skus' in locals() and selected_skus:
        selected_skus_to_forecast = selected_skus
    
    # Run forecast button
    forecast_button_text = "Run Forecast Analysis"
    if forecast_scope == "Selected SKUs Only" and selected_skus_to_forecast:
        forecast_button_text = f"Run Forecast for {len(selected_skus_to_forecast)} Selected SKUs"
    elif forecast_scope == "Selected SKUs Only" and not selected_skus_to_forecast:
        st.warning("Please select at least one SKU to analyze.")
        
    # Only show the button if we have valid SKUs to forecast
    should_show_button = not (forecast_scope == "Selected SKUs Only" and not selected_skus_to_forecast)
    
    if should_show_button and st.button(forecast_button_text):
        st.session_state.run_forecast = True
        with st.spinner("Running forecast analysis..."):
            # Extract time series features for clustering
            features_df = extract_features(st.session_state.sales_data)
            
            # Cluster SKUs
            st.session_state.clusters = cluster_skus(features_df, n_clusters=num_clusters)
            
            # Determine which SKUs to forecast
            skus_to_forecast = None
            if forecast_scope == "Selected SKUs Only" and selected_skus_to_forecast:
                skus_to_forecast = selected_skus_to_forecast
                
            # Generate forecasts with model evaluation
            st.session_state.forecasts = generate_forecasts(
                st.session_state.sales_data,
                st.session_state.clusters,
                forecast_periods=st.session_state.forecast_periods,
                evaluate_models_flag=evaluate_models_flag,
                models_to_evaluate=models_to_evaluate,
                selected_skus=skus_to_forecast
            )
            
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
        selected_sku = st.selectbox(
            "Select a SKU to view forecast details",
            options=sku_list,
            index=0 if st.session_state.selected_sku is None else sku_list.index(st.session_state.selected_sku)
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
                    
                    # Create multi-select for choosing models to display
                    display_options = ["Best Model Only", "All Selected Models", "Custom Selection"]
                    display_choice = st.radio(
                        "Display Options", 
                        display_options,
                        horizontal=True
                    )
                    
                    if display_choice == "Best Model Only":
                        selected_models_for_viz = [forecast_data['model']]
                    elif display_choice == "All Selected Models":
                        # Use the models selected in the sidebar
                        selected_models_for_viz = [m for m in st.session_state.selected_models if m in available_models]
                    else:  # Custom Selection
                        model_options = [model.upper() for model in available_models]
                        selected_model_names = st.multiselect(
                            "Select models to display on chart",
                            options=model_options,
                            default=[forecast_data['model'].upper()],
                            help="Select one or more forecasting models to visualize"
                        )
                        # Convert back to lowercase for internal processing
                        selected_models_for_viz = [model.lower() for model in selected_model_names]
                
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
                
                # Forecast table
                forecast_table = pd.DataFrame({
                    'Date': forecast_data['forecast'].index,
                    'Forecast': forecast_data['forecast'].values.round(0),
                    'Lower Bound': forecast_data['lower_bound'].values.round(0),
                    'Upper Bound': forecast_data['upper_bound'].values.round(0)
                })
                
                st.dataframe(forecast_table, use_container_width=True)
    
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
            selected_sku_preview = st.selectbox(
                "Select a SKU to view historical sales data",
                options=all_skus,
                index=0 if st.session_state.selected_sku is None else 
                       all_skus.index(st.session_state.selected_sku) if st.session_state.selected_sku in all_skus else 0
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
