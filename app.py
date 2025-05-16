import streamlit as st
import pandas as pd
import os
import sqlite3
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Demand Forecasting Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Make sure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Initialize SQLite database if not exists
def init_db():
    """Initialize the database with required tables."""
    db_path = 'data/supply_chain.db'
    conn = sqlite3.connect(db_path)
    
    # Create tables if they don't exist
    conn.execute('''
    CREATE TABLE IF NOT EXISTS uploaded_files (
        id TEXT PRIMARY KEY,
        filename TEXT,
        file_data BLOB,
        file_size INTEGER,
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
st.title("Demand Forecasting Platform")
st.markdown("""
### Welcome to the Demand Forecasting Platform

This application specializes in accurate demand forecasting for products with intermittent demand patterns.
The system automatically identifies SKUs with frequent zero values and applies the Croston method, 
which is optimized for intermittent demand.

#### Features:
- **Automatic Detection**: Identifies products with intermittent demand patterns
- **Specialized Algorithms**: Uses Croston method for intermittent demand
- **Hyperparameter Tuning**: Optimizes model parameters for better accuracy
- **Interactive Visualizations**: Explores forecasts with dynamic charts
""")

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
                    "UPDATE uploaded_files SET file_data = ?, file_size = ?, upload_date = ?, file_type = ? WHERE id = ?",
                    (uploaded_file.getvalue(), uploaded_file.size, datetime.now(), 'sales_data', file_id)
                )
                st.info(f"Updated existing file: {uploaded_file.name}")
            else:
                # Insert new file
                cursor.execute(
                    "INSERT INTO uploaded_files (id, filename, file_data, file_size, upload_date, file_type) VALUES (?, ?, ?, ?, ?, ?)",
                    (file_id, uploaded_file.name, uploaded_file.getvalue(), uploaded_file.size, datetime.now(), 'sales_data')
                )
                st.info(f"Saved new file to database: {uploaded_file.name}")
            
            conn.commit()
            conn.close()
            
            # Set session state for access in other pages
            st.session_state.sales_data = df
            
            # Add navigation instruction
            st.success("Data uploaded successfully! Navigate to the Demand Forecasting page to generate forecasts.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    # Check if we have data in the database
    conn = sqlite3.connect('data/supply_chain.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM uploaded_files WHERE file_type = 'sales_data'")
    file_count = cursor.fetchone()[0]
    
    if file_count > 0:
        st.info(f"Total files in database: {file_count}")
        cursor.execute("SELECT id, filename, file_size FROM uploaded_files WHERE file_type = 'sales_data'")
        files = cursor.fetchall()
        
        for file_id, filename, file_size in files:
            st.write(f"Found file: {filename} (ID: {file_id}, Size: {file_size} bytes)")
            
            # Attempt to load the file
            cursor.execute("SELECT file_data FROM uploaded_files WHERE id = ?", (file_id,))
            file_data = cursor.fetchone()[0]
            
            try:
                # Load to pandas dataframe
                import io
                data = pd.read_excel(io.BytesIO(file_data))
                st.session_state.sales_data = data
                st.success(f"Successfully parsed file {filename} - found {len(data)} rows")
            except Exception as e:
                st.warning(f"Could not parse file {filename}: {str(e)}")
    else:
        st.warning("No files found in the database. Please upload a file.")
    
    conn.close()

# Add footer
st.markdown("---")
st.markdown("© 2025 Demand Forecasting Platform | Optimized for Heroku Deployment")