"""
Entry point for the application, forwards to the main forecasting page
"""

import streamlit as st
import os

# Create empty main page that redirects to the forecasting page
st.set_page_config(page_title="Redirecting...", page_icon="📊")

# Redirect to the forecasting page
st.markdown(
    f"""
    <meta http-equiv="refresh" content="0; url=/14_V2_Demand_Forecasting_Croston">
    <h1>Redirecting to Demand Forecasting...</h1>
    If you are not redirected automatically, click <a href="/14_V2_Demand_Forecasting_Croston">here</a>.
    """, 
    unsafe_allow_html=True
)