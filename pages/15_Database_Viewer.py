import streamlit as st
import pandas as pd
import json
import sqlite3
import io
from utils.database import (
    get_all_files, 
    get_forecast_history, 
    get_forecast_details,
    get_all_model_parameters,
    get_secondary_sales
)

# Page title
st.title("Database Viewer")
st.markdown("""
This module provides direct access to all data stored in the system's database,
allowing you to view and export any table for analysis or reporting.
""")

# Create a dropdown to select the table to view
table_options = {
    "Uploaded Files": "uploaded_files",
    "Forecasts": "forecast_results",
    "Model Parameters": "model_parameter_cache",
    "Secondary Sales": "secondary_sales"
}

selected_option = st.selectbox(
    "Select data to view", 
    options=list(table_options.keys()),
    index=0
)

# Container for table data
data_container = st.container()

# Function to convert data to Excel
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    return output.getvalue()

# Display the selected table data
with data_container:
    # Create a horizontal line for visual separation
    st.markdown("---")
    
    # Different handling based on the selected table
    if selected_option == "Uploaded Files":
        st.subheader("Uploaded Files")
        
        # Get all files from the database
        files = get_all_files()
        
        if files:
            # Convert to DataFrame for display
            df = pd.DataFrame(files)
            
            # Reorder and rename columns for better display
            df = df[['filename', 'file_type', 'created_at', 'id']]
            df.columns = ['Filename', 'Type', 'Upload Date', 'ID']
            
            # Display the table
            st.dataframe(df)
            
            # Add download button
            st.download_button(
                label="Download as Excel",
                data=to_excel(df),
                file_name="uploaded_files.xlsx",
                mime="application/vnd.ms-excel"
            )
        else:
            st.info("No files have been uploaded to the database yet.")
    
    elif selected_option == "Forecasts":
        st.subheader("Forecast Results")
        
        # Get all forecasts from the database
        forecasts = get_forecast_history()
        
        if forecasts:
            # Convert to DataFrame for display
            df = pd.DataFrame(forecasts)
            
            # Format metric columns to show fewer decimal places
            if 'mape' in df.columns:
                df['mape'] = df['mape'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            if 'rmse' in df.columns:
                df['rmse'] = df['rmse'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            if 'mae' in df.columns:
                df['mae'] = df['mae'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                
            # Rename columns for better display
            df.columns = ['ID', 'SKU', 'Model', 'Date', 'MAPE (%)', 'RMSE', 'MAE']
            
            # Display the table
            st.dataframe(df)
            
            # Add download button
            st.download_button(
                label="Download as Excel",
                data=to_excel(df),
                file_name="forecast_results.xlsx",
                mime="application/vnd.ms-excel"
            )
            
            # Option to view detailed forecast
            st.markdown("### View Detailed Forecast")
            forecast_id = st.selectbox(
                "Select a forecast to view details",
                options=df['ID'].tolist(),
                format_func=lambda x: f"{df[df['ID']==x]['SKU'].values[0]} - {df[df['ID']==x]['Model'].values[0]} - {df[df['ID']==x]['Date'].values[0]}"
            )
            
            if forecast_id:
                # Get detailed forecast data
                details = get_forecast_details(forecast_id)
                
                if details:
                    st.write("#### Forecast Details")
                    
                    # Display basic info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**SKU:** {details['sku']}")
                        st.write(f"**Model:** {details['model_type']}")
                        st.write(f"**Date:** {details['forecast_date']}")
                    
                    with col2:
                        st.write(f"**MAPE:** {details['mape']:.2f}%" if details['mape'] is not None else "**MAPE:** N/A")
                        st.write(f"**RMSE:** {details['rmse']:.2f}" if details['rmse'] is not None else "**RMSE:** N/A")
                        st.write(f"**MAE:** {details['mae']:.2f}" if details['mae'] is not None else "**MAE:** N/A")
                    
                    # Parse and display forecast data
                    st.write("#### Forecast Values")
                    try:
                        forecast_data = json.loads(details['forecast_data'])
                        forecast_df = pd.DataFrame(forecast_data)
                        st.dataframe(forecast_df)
                        
                        # Add download button for detailed forecast
                        st.download_button(
                            label="Download Detailed Forecast",
                            data=to_excel(forecast_df),
                            file_name=f"forecast_details_{details['sku']}_{details['model_type']}.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                    except Exception as e:
                        st.error(f"Error parsing forecast data: {str(e)}")
                    
                    # Parse and display model parameters
                    if details['model_params']:
                        st.write("#### Model Parameters")
                        try:
                            model_params = json.loads(details['model_params'])
                            # Convert dict to DataFrame for better display
                            params_df = pd.DataFrame([model_params])
                            st.dataframe(params_df)
                        except:
                            st.write(details['model_params'])
                else:
                    st.error("Could not retrieve forecast details.")
        else:
            st.info("No forecasts have been saved to the database yet.")
            
    elif selected_option == "Model Parameters":
        st.subheader("Model Parameters")
        
        # Direct database query as a fallback for model parameters
        try:
            # Use sqlite3 directly to query model parameters
            conn = sqlite3.connect("data/supply_chain.db")
            df_params = pd.read_sql_query("SELECT id, sku, model_type, parameters, last_updated, tuning_iterations, best_score FROM model_parameter_cache", conn)
            conn.close()
            
            if not df_params.empty:
                # Format columns for better display
                df_params = df_params.rename(columns={
                    'sku': 'SKU',
                    'model_type': 'Model Type',
                    'parameters': 'Parameters',
                    'last_updated': 'Last Updated',
                    'tuning_iterations': 'Iterations',
                    'best_score': 'Best Score'
                })
                
                # Display the table
                st.dataframe(df_params)
                
                # Add download button
                st.download_button(
                    label="Download as Excel",
                    data=to_excel(df_params),
                    file_name="model_parameters.xlsx",
                    mime="application/vnd.ms-excel"
                )
            else:
                st.info("No model parameters found in the database.")
                
        except Exception as e:
            st.error(f"Error retrieving model parameters: {str(e)}")
            st.info("No optimized model parameters have been saved to the database yet.")
            
    elif selected_option == "Secondary Sales":
        st.subheader("Secondary Sales Data")
        
        # Direct database query for secondary sales
        try:
            # Use sqlite3 directly to query secondary sales
            conn = sqlite3.connect("data/supply_chain.db")
            df_sales = pd.read_sql_query("SELECT sku, date, primary_sales, estimated_secondary_sales, noise, algorithm_used FROM secondary_sales ORDER BY sku, date", conn)
            conn.close()
            
            if not df_sales.empty:
                # Format columns for better display
                df_sales = df_sales.rename(columns={
                    'sku': 'SKU',
                    'date': 'Date',
                    'primary_sales': 'Primary Sales',
                    'estimated_secondary_sales': 'Secondary Sales',
                    'noise': 'Difference',
                    'algorithm_used': 'Algorithm'
                })
                
                # Display the table
                st.dataframe(df_sales)
                
                # Add download button
                st.download_button(
                    label="Download as Excel",
                    data=to_excel(df_sales),
                    file_name="secondary_sales.xlsx",
                    mime="application/vnd.ms-excel"
                )
            else:
                st.info("No secondary sales data found in the database.")
                
        except Exception as e:
            st.error(f"Error retrieving secondary sales data: {str(e)}")
            st.info("No secondary sales data has been generated yet.")
    
# No direct SQL query section as requested