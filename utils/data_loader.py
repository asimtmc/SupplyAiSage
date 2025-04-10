"""
Data loading utility for the Supply Chain Platform.
This module handles loading data from the database into session state.
"""
import streamlit as st
from utils.database import get_file_by_type

def load_data_from_database():
    """
    Function to load data from database and update status.
    This is a standalone version of the function that can be imported by pages.
    
    Returns:
        bool: True if data was loaded successfully, False otherwise
    """
    # Initialize the status dictionary if it doesn't exist
    if 'db_load_status' not in st.session_state:
        st.session_state.db_load_status = {}
    
    # Clear previous status
    st.session_state.db_load_status = {}
    
    try:
        # Load sales data
        sales_data_file = get_file_by_type('sales_data')
        if sales_data_file:
            st.session_state.sales_data = sales_data_file[1]
            st.session_state.db_load_status['sales_data'] = {
                'status': 'success',
                'message': f"✅ Successfully loaded sales data: {sales_data_file[0]}"
            }
        else:
            st.session_state.db_load_status['sales_data'] = {
                'status': 'warning',
                'message': "⚠️ No sales data found in database"
            }
            st.session_state.sales_data = None
            
        # Load BOM data
        bom_data_file = get_file_by_type('bom_data')
        if bom_data_file:
            st.session_state.bom_data = bom_data_file[1]
            st.session_state.db_load_status['bom_data'] = {
                'status': 'success',
                'message': f"✅ Successfully loaded BOM data: {bom_data_file[0]}"
            }
        else:
            st.session_state.db_load_status['bom_data'] = {
                'status': 'warning',
                'message': "⚠️ No BOM data found in database"
            }
            st.session_state.bom_data = None
            
        # Load supplier data
        supplier_data_file = get_file_by_type('supplier_data')
        if supplier_data_file:
            st.session_state.supplier_data = supplier_data_file[1]
            st.session_state.db_load_status['supplier_data'] = {
                'status': 'success',
                'message': f"✅ Successfully loaded supplier data: {supplier_data_file[0]}"
            }
        else:
            st.session_state.db_load_status['supplier_data'] = {
                'status': 'warning',
                'message': "⚠️ No supplier data found in database"
            }
            st.session_state.supplier_data = None
        
        return True
    
    except Exception as e:
        st.session_state.db_load_status['error'] = {
            'status': 'error',
            'message': f"❌ Error loading data: {str(e)}"
        }
        return False