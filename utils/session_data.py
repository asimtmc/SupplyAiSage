import streamlit as st
import pandas as pd
import sqlite3
import io

def load_data_if_needed():
    """Loads sales data from the database if it's not already in session state
    
    Returns:
        bool: True if data is loaded, False otherwise
    """
    # Check if data is already loaded
    if 'sales_data' in st.session_state and not st.session_state.sales_data.empty:
        return True
    
    # Load from the database
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
            conn.close()
            return False
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False