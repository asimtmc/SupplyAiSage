import streamlit as st
import pandas as pd
import os
import sqlite3
from datetime import datetime
import io

# Import custom modules
from utils.session_data import load_data_if_needed

# Set page configuration
st.set_page_config(
    page_title="Intermittent Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Make sure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('data/samples'):
    os.makedirs('data/samples')

# Initialize SQLite database if not exists
def init_db():
    """Initialize the database with required tables."""
    db_path = 'data/supply_chain.db'
    
    # Create new database if it doesn't exist
    conn = sqlite3.connect(db_path)
    
    # Create tables if they don't exist
    conn.execute('''
    CREATE TABLE IF NOT EXISTS uploaded_files (
        id TEXT PRIMARY KEY,
        filename TEXT,
        file_data BLOB,
        upload_date TIMESTAMP,
        file_type TEXT
    )
    ''')
    
    conn.execute('''
    CREATE TABLE IF NOT EXISTS forecast_results (
        id TEXT PRIMARY KEY,
        sku TEXT,
        model TEXT,
        forecast_date TIMESTAMP,
        forecast_data TEXT,
        metadata TEXT
    )
    ''')
    
    conn.execute('''
    CREATE TABLE IF NOT EXISTS model_parameter_cache (
        id TEXT PRIMARY KEY,
        sku TEXT,
        model_type TEXT,
        parameters TEXT,
        last_updated TIMESTAMP,
        metrics TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    
    return f"sqlite:///{db_path}"

# Initialize database
db_url = init_db()
print(f"Using SQLite database: {db_url}")

# Main page content
st.title("Intermittent Demand Forecasting")
st.markdown("""
### Specialized Tool for Intermittent Demand Patterns

This application focuses on accurate demand forecasting for products with intermittent 
demand patterns using the Croston method and other specialized algorithms.

#### Key Features:
- **Automatic Seasonal Detection**: Intelligently identifies seasonal patterns
- **Specialized Algorithms**: Optimized for sporadic demand patterns
- **Hyperparameter Tuning**: Fine-tunes models for better accuracy
- **Interactive Visualizations**: Explore forecasts with dynamic charts
""")

# Automatically load sample data if no other data exists
data_loaded = load_data_if_needed()

# File uploader
uploaded_file = st.file_uploader("Upload your sales data (Excel file)", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)
        
        # Check required columns
        required_columns = ['date', 'sku', 'quantity']
        if not all(col in df.columns for col in required_columns):
            st.error("The uploaded file must contain 'date', 'sku', and 'quantity' columns.")
        else:
            # Display preview
            st.success(f"Successfully uploaded {uploaded_file.name}")
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Save to database
            conn = sqlite3.connect('data/supply_chain.db')
            file_id = uploaded_file.name.replace(' ', '_')
            
            # Check if file already exists
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM uploaded_files WHERE id = ?", (file_id,))
            existing_file = cursor.fetchone()
            
            if existing_file:
                # Update existing file
                cursor.execute(
                    "UPDATE uploaded_files SET file_data = ?, upload_date = ?, file_type = ? WHERE id = ?",
                    (uploaded_file.getvalue(), datetime.now(), 'sales_data', file_id)
                )
                st.info(f"Updated existing file: {uploaded_file.name}")
            else:
                # Insert new file
                cursor.execute(
                    "INSERT INTO uploaded_files (id, filename, file_data, upload_date, file_type) VALUES (?, ?, ?, ?, ?)",
                    (file_id, uploaded_file.name, uploaded_file.getvalue(), datetime.now(), 'sales_data')
                )
                st.info(f"Saved new file to database: {uploaded_file.name}")
            
            conn.commit()
            conn.close()
            
            # Set session state for access in other pages
            st.session_state.sales_data = df
            
            # Add navigation instruction
            st.success("Data uploaded successfully! Navigate to the Intermittent Demand Forecasting page to generate forecasts.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
elif data_loaded:
    # Display preview of loaded data
    st.write("Using loaded data:")
    st.dataframe(st.session_state.sales_data.head())
    st.info(f"Total rows: {len(st.session_state.sales_data)}")
    st.success("Sample data loaded automatically. Navigate to the Intermittent Demand Forecasting page to generate forecasts.")
else:
    st.warning("No data loaded. Please upload a file.")

# Add footer
st.markdown("---")
st.markdown("© 2025 Intermittent Demand Forecasting | Optimized for Heroku Deployment")