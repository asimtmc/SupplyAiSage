"""
Entry point for the application, forwards to the main forecasting page
"""

import streamlit as st
import os
import pandas as pd
from utils.session_data import load_sample_data

# Page configuration
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-load sample data if not already loaded
if 'sales_data' not in st.session_state or st.session_state.sales_data.empty:
    try:
        sample_data = load_sample_data()
        if sample_data is not None:
            st.session_state.sales_data = sample_data
    except Exception as e:
        print(f"Error auto-loading sample data: {e}")

# Main app
st.title("📊 Supply Chain Demand Forecasting")
st.subheader("Specialized for Intermittent Demand")

st.markdown("""
### Welcome to the Demand Forecasting Tool

This application helps you predict future demand for products with intermittent or irregular patterns.
It uses advanced methods like Croston's method which is specifically designed for items with "lumpy" demand.

**To get started:**
1. Go to the V2 Demand Forecasting page in the sidebar
2. Sample data is automatically loaded 
3. Upload your own data or use the sample data
4. Select a product (SKU) and generate forecasts

The application is optimized for inventory planning with irregular demand patterns.
""")

# Display the button to go to the forecasting page
if st.button("Go to Forecasting Page", use_container_width=True):
    # Navigate to the forecasting page
    st.switch_page("pages/14_V2_Demand_Forecasting_Croston.py")