#!/usr/bin/env python3
import sys

filename = 'pages/01_Demand_Forecasting.py'
line_to_add_after = 772
text_to_add = '''
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
'''

with open(filename, 'r') as file:
    lines = file.readlines()

lines.insert(line_to_add_after, text_to_add)

with open(filename, 'w') as file:
    file.writelines(lines)

print(f"Line inserted after line {line_to_add_after} in {filename}")
