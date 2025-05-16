import streamlit as st
import pandas as pd
import sqlite3
import io
import os

def load_sample_data():
    """Load sample data from the samples directory"""
    try:
        # Check for sample data file
        sample_path = 'data/samples/sample_data.xlsx'
        if os.path.exists(sample_path):
            data = pd.read_excel(sample_path)
            
            # Process data if needed - check date column format, etc.
            if 'date' in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data['date']):
                    data['date'] = pd.to_datetime(data['date'])
            
            return data
        else:
            # Fallback to CSV if Excel file doesn't exist
            csv_path = 'data/samples/sales_sample.csv'
            if os.path.exists(csv_path):
                data = pd.read_csv(csv_path)
                
                # Process data if needed - check date column format, etc.
                if 'date' in data.columns:
                    if not pd.api.types.is_datetime64_any_dtype(data['date']):
                        data['date'] = pd.to_datetime(data['date'])
                
                return data
        
        return None
    except Exception as e:
        print(f"Error loading sample data: {str(e)}")
        return None

def load_data_if_needed():
    """Loads sales data from the database if it's not already in session state,
    or loads sample data if no user data exists.
    
    Returns:
        bool: True if data is loaded, False otherwise
    """
    # Check if data is already loaded
    if 'sales_data' in st.session_state and not st.session_state.sales_data.empty:
        return True
    
    # Try loading from the database first
    try:
        conn = sqlite3.connect('data/supply_chain.db')
        cursor = conn.cursor()
        
        # Get latest sales file
        cursor.execute(
            "SELECT file_data FROM uploaded_files WHERE file_type = 'sales_data' ORDER BY upload_date DESC LIMIT 1"
        )
        result = cursor.fetchone()
        
        if result:
            file_data = result[0]
            # Load into dataframe
            data = pd.read_excel(io.BytesIO(file_data))
            
            # Process data if needed - check date column format, etc.
            if 'date' in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data['date']):
                    data['date'] = pd.to_datetime(data['date'])
            
            # Store in session state
            st.session_state.sales_data = data
            conn.close()
            return True
        else:
            # No data in database, try loading sample data
            sample_data = load_sample_data()
            if sample_data is not None:
                st.session_state.sales_data = sample_data
                conn.close()
                return True
            
            conn.close()
            return False
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        
        # Try loading sample data
        sample_data = load_sample_data()
        if sample_data is not None:
            st.session_state.sales_data = sample_data
            return True
        
        return False