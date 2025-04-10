import streamlit as st
from utils.data_loader import load_data_from_database

def load_data_automatically():
    """
    Automatically load data from the database if needed.
    This function allows pages to directly load data without requiring navigation through the main app.
    
    Returns:
        bool: True if data was loaded successfully, False otherwise
    """
    # Check if data is already loaded
    if 'sales_data' in st.session_state and st.session_state.sales_data is not None:
        return True
    
    try:
        # Use the standalone data loader function
        success = load_data_from_database()
        return success
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False

def check_data_requirements(requirements=None):
    """
    Check that required data is available in the session state.
    If not, attempt to load it. If loading fails, show an error message.
    
    Args:
        requirements (list): List of strings for required data keys
                            (e.g., ['sales_data', 'bom_data'])
                            
    Returns:
        bool: True if all requirements are met, False otherwise
    """
    if requirements is None:
        requirements = ['sales_data']
    
    # First attempt to load data if needed
    load_data_automatically()
    
    # Check if all requirements are met
    missing_data = []
    for req in requirements:
        if req not in st.session_state or st.session_state[req] is None:
            missing_data.append(req)
    
    if missing_data:
        requirements_text = ", ".join([req.replace('_data', '') for req in missing_data])
        st.warning(f"Missing required data: {requirements_text}. Please upload this data on the main page.")
        return False
    
    return True