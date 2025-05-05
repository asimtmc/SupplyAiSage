import streamlit as st
import pandas as pd
from utils.database import get_file_by_type, get_all_files

def load_data_if_needed():
    """
    Automatically loads the required data files if they haven't been loaded yet.
    Call this at the beginning of each page to ensure data is available.
    """
    # Check if sales data is already loaded
    if 'sales_data' not in st.session_state or st.session_state.sales_data is None:
        try:
            # Load sales data from database
            sales_data_file = get_file_by_type('sales_data')
            if sales_data_file:
                st.session_state.sales_data = sales_data_file[1]
                print(f"Auto-loaded sales data: {sales_data_file[0]}")
            
            # Load BOM data from database
            bom_data_file = get_file_by_type('bom_data')
            if bom_data_file:
                st.session_state.bom_data = bom_data_file[1]
                print(f"Auto-loaded BOM data: {bom_data_file[0]}")
            
            # Load supplier data from database
            supplier_data_file = get_file_by_type('supplier_data')
            if supplier_data_file:
                st.session_state.supplier_data = supplier_data_file[1]
                print(f"Auto-loaded supplier data: {supplier_data_file[0]}")
                
        except Exception as e:
            print(f"Error auto-loading data: {str(e)}")
    
    # Return True if we have sales data, False otherwise
    return 'sales_data' in st.session_state and st.session_state.sales_data is not None