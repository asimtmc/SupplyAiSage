import streamlit as st
import pandas as pd
import io
import numpy as np
from utils.data_processor import process_sales_data, process_bom_data, process_supplier_data
import os

# Set page configuration
st.set_page_config(
    page_title="AI Supply Chain Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'bom_data' not in st.session_state:
    st.session_state.bom_data = None
if 'supplier_data' not in st.session_state:
    st.session_state.supplier_data = None
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = None
if 'clusters' not in st.session_state:
    st.session_state.clusters = None

# Main page header
st.title("AI-Powered Supply Chain Platform")
st.markdown("""
This platform helps you optimize your supply chain with AI-powered forecasting, 
production planning, and material procurement. Simply upload your Excel files to get started.
""")

# Data upload section
st.header("Data Upload")
st.markdown("Upload your Excel files containing sales history, bill of materials, and supplier information.")

# Create three columns for the different file uploads
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Sales History")
    sales_file = st.file_uploader("Upload Sales Data (Excel)", type=["xlsx", "xls"], key="sales_upload")
    
    if sales_file is not None:
        try:
            # Process the sales data
            st.session_state.sales_data = process_sales_data(sales_file)
            st.success(f"Successfully loaded sales data with {len(st.session_state.sales_data)} records!")
            
            # Display sample of the data
            st.write("Preview:")
            st.dataframe(st.session_state.sales_data.head())
        except Exception as e:
            st.error(f"Error processing sales data: {str(e)}")

with col2:
    st.subheader("Bill of Materials")
    bom_file = st.file_uploader("Upload BOM Data (Excel)", type=["xlsx", "xls"], key="bom_upload")
    
    if bom_file is not None:
        try:
            # Process the BOM data
            st.session_state.bom_data = process_bom_data(bom_file)
            st.success(f"Successfully loaded BOM data with {len(st.session_state.bom_data)} records!")
            
            # Display sample of the data
            st.write("Preview:")
            st.dataframe(st.session_state.bom_data.head())
        except Exception as e:
            st.error(f"Error processing BOM data: {str(e)}")

with col3:
    st.subheader("Supplier Information")
    supplier_file = st.file_uploader("Upload Supplier Data (Excel)", type=["xlsx", "xls"], key="supplier_upload")
    
    if supplier_file is not None:
        try:
            # Process the supplier data
            st.session_state.supplier_data = process_supplier_data(supplier_file)
            st.success(f"Successfully loaded supplier data with {len(st.session_state.supplier_data)} records!")
            
            # Display sample of the data
            st.write("Preview:")
            st.dataframe(st.session_state.supplier_data.head())
        except Exception as e:
            st.error(f"Error processing supplier data: {str(e)}")

# Dashboard overview section (only shown when data is loaded)
if st.session_state.sales_data is not None:
    st.header("Quick Insights")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Total sales
        total_sales = st.session_state.sales_data['quantity'].sum() if 'quantity' in st.session_state.sales_data.columns else 0
        st.metric(label="Total Units Sold", value=f"{total_sales:,}")
    
    with col2:
        # Number of SKUs
        skus = len(st.session_state.sales_data['sku'].unique()) if 'sku' in st.session_state.sales_data.columns else 0
        st.metric(label="Number of SKUs", value=skus)
    
    with col3:
        # Time range
        if 'date' in st.session_state.sales_data.columns:
            date_range = f"{st.session_state.sales_data['date'].min().strftime('%b %Y')} - {st.session_state.sales_data['date'].max().strftime('%b %Y')}"
        else:
            date_range = "N/A"
        st.metric(label="Date Range", value=date_range)
    
    with col4:
        # Data completeness
        if st.session_state.sales_data is not None and st.session_state.bom_data is not None and st.session_state.supplier_data is not None:
            completeness = "Complete"
            delta_color = "normal"
        else:
            missing = []
            if st.session_state.sales_data is None:
                missing.append("Sales")
            if st.session_state.bom_data is None:
                missing.append("BOM")
            if st.session_state.supplier_data is None:
                missing.append("Supplier")
            completeness = f"Missing: {', '.join(missing)}" if missing else "Complete"
            delta_color = "off" if missing else "normal"
        
        st.metric(label="Data Completeness", value=completeness)

    # Navigation guidance
    st.markdown("""
    ## Next Steps
    
    Now that you've uploaded your data, explore the different modules of the platform:
    
    1. **Demand Forecasting**: View AI-generated forecasts for your products
    2. **Production Planning**: Get optimized production schedules
    3. **Material Procurement**: Plan your material needs
    4. **KPI Dashboard**: Monitor key performance indicators
    5. **What-If Scenarios**: Simulate different supply chain scenarios
    
    Use the sidebar navigation to access these modules.
    """)

else:
    # Show instructions when no data is loaded
    st.info("👆 Please upload your Excel files to get started with the AI analysis!")
    
    st.markdown("""
    ### Sample File Formats
    
    **Sales History**:
    - Required columns: date, sku, quantity
    - Optional columns: region, customer, revenue
    
    **Bill of Materials (BOM)**:
    - Required columns: sku, material_id, quantity_required
    - Optional columns: material_name, unit_cost
    
    **Supplier Information**:
    - Required columns: material_id, supplier_id, lead_time_days
    - Optional columns: moq (minimum order quantity), price_per_unit
    """)

# Footer
st.markdown("---")
st.markdown("© 2023 AI Supply Chain Platform | All Rights Reserved")
