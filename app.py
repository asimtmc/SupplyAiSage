import streamlit as st
import pandas as pd
import io
import numpy as np
from utils.data_processor import process_sales_data, process_bom_data, process_supplier_data
from utils.database import save_uploaded_file, get_file_by_type
import os
from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo
from streamlit_extras.badges import badge
from streamlit_card import card
import time
import extra_streamlit_components as stx
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI Supply Chain Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern high-tech look
st.markdown("""
<style>
    /* Primary Colors */
    :root {
        --primary-color: #4CAF50;
        --secondary-color: #2E7D32;
        --accent-color: #81C784;
        --background-color: #f8f9fa;
        --text-color: #212121;
        --light-text: #757575;
    }
    
    /* Main container styling */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 600 !important;
    }
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1.5rem !important;
    }
    h2 {
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 0.5rem;
        margin-top: 2rem !important;
    }
    
    /* Card styling */
    .stCard {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: var(--secondary-color);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* File uploader styling */
    .stFileUploader {
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed var(--accent-color);
        margin-bottom: 1rem;
    }
    
    /* Metric container styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: var(--primary-color) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: white;
        padding: 1.5rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    
    /* Modern alert boxes */
    .st-ae {
        border-radius: 10px;
        padding: 1.2rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        margin-bottom: 1rem !important;
    }
    .st-bx {
        border-radius: 8px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Success message styling */
    .element-container div[data-testid="stText"] .success {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 16px;
        border-radius: 4px;
        border-left: 5px solid #2e7d32;
        margin: 16px 0;
    }
    
    /* Feature box */
    .feature-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    /* Animated gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #4CAF50, #2196F3, #9C27B0);
        background-size: 200% auto;
        color: transparent;
        -webkit-background-clip: text;
        background-clip: text;
        animation: gradient 8s linear infinite;
        font-weight: bold;
        font-size: 3rem;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

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
if 'init_db_load' not in st.session_state:
    st.session_state.init_db_load = False

# Try to load data from database on initial load
if not st.session_state.init_db_load:
    # Try to get the most recent files of each type
    try:
        sales_data_file = get_file_by_type('sales_data')
        if sales_data_file:
            st.session_state.sales_data = sales_data_file[1]
            
        bom_data_file = get_file_by_type('bom_data')
        if bom_data_file:
            st.session_state.bom_data = bom_data_file[1]
            
        supplier_data_file = get_file_by_type('supplier_data')
        if supplier_data_file:
            st.session_state.supplier_data = supplier_data_file[1]
            
        # Mark as loaded
        st.session_state.init_db_load = True
    except Exception as e:
        # Silently fail - we'll let users upload files
        pass

# Main page header with animated gradient
st.markdown('<h1 class="gradient-text">AI-Powered Supply Chain Platform</h1>', unsafe_allow_html=True)

# Add badge for version
badge(type="github", name="SupplyChainAI/Platform", url="https://github.com")
st.markdown("  ")  # Add spacing

# Modern introduction with highlighted sections
st.markdown("""
<div class="feature-box">
    <h2 style="margin-top: 0;">🚀 Intelligent Supply Chain Management</h2>
    <p>Optimize your entire supply chain with AI-powered forecasting, smart production planning, and strategic material procurement. Our platform leverages advanced machine learning algorithms to provide actionable insights for your business.</p>
</div>
""", unsafe_allow_html=True)

# Create a cookie manager for persistent user settings
cookie_manager = stx.CookieManager()
user_id = cookie_manager.get("user_id")
if not user_id:
    # Generate a random user ID if none exists
    user_id = str(int(time.time()))
    cookie_manager.set("user_id", user_id, expires_at=datetime(2030, 1, 1))

# Platform features display
st.markdown('<h2>Key Features</h2>', unsafe_allow_html=True)

# Create feature cards using columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-icon">🔮</div>
        <h3>AI-Powered Forecasting</h3>
        <p>Our intelligent algorithms automatically select the best forecasting model for each product based on its unique sales pattern.</p>
        <ul>
            <li>LSTM Neural Networks</li>
            <li>Prophet Time Series</li>
            <li>SARIMA Forecasting</li>
            <li>Holt-Winters Models</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-icon">🏭</div>
        <h3>Smart Production Planning</h3>
        <p>Maximize efficiency and minimize costs with AI-optimized production schedules.</p>
        <ul>
            <li>Dynamic Capacity Planning</li>
            <li>Resource Optimization</li>
            <li>Cost Minimization</li>
            <li>Daily Production Scheduling</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-icon">📦</div>
        <h3>Strategic Procurement</h3>
        <p>Get precise recommendations for when, what, and how much to order from suppliers.</p>
        <ul>
            <li>Material Requirements Planning</li>
            <li>Supplier Lead Time Analysis</li>
            <li>MOQ Optimization</li>
            <li>Delivery Schedule Coordination</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Data upload section with tabs
colored_header(
    label="Data Management",
    description="Upload and manage your supply chain data",
    color_name="green-70",
)

# Create tabs for upload and database options
tab1, tab2 = st.tabs(["📤 Upload Files", "💾 Database Files"])

with tab1:
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
                
                # Save to database
                file_id = save_uploaded_file(sales_file, 'sales_data', 'Sales history data')
                
                # Show success message with special styling
                st.markdown(f"""
                <div style="background-color: #e8f5e9; color: #2e7d32; padding: 16px; border-radius: 4px; border-left: 5px solid #2e7d32; margin: 16px 0;">
                    ✅ Successfully loaded sales data with {len(st.session_state.sales_data)} records and saved to database!
                </div>
                """, unsafe_allow_html=True)
                
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
                
                # Save to database
                file_id = save_uploaded_file(bom_file, 'bom_data', 'Bill of materials data')
                
                # Show success message with special styling
                st.markdown(f"""
                <div style="background-color: #e8f5e9; color: #2e7d32; padding: 16px; border-radius: 4px; border-left: 5px solid #2e7d32; margin: 16px 0;">
                    ✅ Successfully loaded BOM data with {len(st.session_state.bom_data)} records and saved to database!
                </div>
                """, unsafe_allow_html=True)
                
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
                
                # Save to database
                file_id = save_uploaded_file(supplier_file, 'supplier_data', 'Supplier information data')
                
                # Show success message with special styling
                st.markdown(f"""
                <div style="background-color: #e8f5e9; color: #2e7d32; padding: 16px; border-radius: 4px; border-left: 5px solid #2e7d32; margin: 16px 0;">
                    ✅ Successfully loaded supplier data with {len(st.session_state.supplier_data)} records and saved to database!
                </div>
                """, unsafe_allow_html=True)
                
                # Display sample of the data
                st.write("Preview:")
                st.dataframe(st.session_state.supplier_data.head())
            except Exception as e:
                st.error(f"Error processing supplier data: {str(e)}")

with tab2:
    st.markdown("Access previously uploaded files from the database.")
    
    # Add link to file management page
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <a href="/File_Management" target="_self" style="text-decoration: none;">
            <div style="background-color: #4CAF50; color: white; padding: 1rem 2rem; border-radius: 8px; display: inline-block; font-weight: bold; transition: all 0.3s;">
                <span style="font-size: 1.2rem;">📂 Go to File Management</span>
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Show message about database
    st.info("The File Management page allows you to view and download all previously uploaded files.")
    
    # Show tips for data management
    st.markdown("""
    ### Data Management Tips
    
    - All uploaded files are automatically saved to a secure database
    - Files can be accessed from any device using the same account
    - Data is retained between sessions, eliminating the need for re-uploads
    - You can download any file in its original Excel format
    """)

# Dashboard overview section (only shown when data is loaded)
if st.session_state.sales_data is not None:
    colored_header(
        label="Quick Insights",
        description="Key metrics and data overview",
        color_name="green-70",
    )
    
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

    # Navigation guidance with modern cards
    st.subheader("Next Steps")
    
    # Create cards for each module
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem; color: #4CAF50;">📈</div>
            <h3>Demand Forecasting</h3>
            <p>View AI-generated forecasts for your products with multiple forecasting models.</p>
            <a href="/01_Demand_Forecasting" target="_self" style="text-decoration: none;">
                <div style="background-color: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 4px; display: inline-block; margin-top: 0.5rem;">
                    Go to Forecasting
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem; color: #4CAF50;">🏭</div>
            <h3>Production Planning</h3>
            <p>Get optimized production schedules based on forecasted demand and capacity.</p>
            <a href="/02_Production_Planning" target="_self" style="text-decoration: none;">
                <div style="background-color: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 4px; display: inline-block; margin-top: 0.5rem;">
                    Go to Planning
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem; color: #4CAF50;">📊</div>
            <h3>KPI Dashboard</h3>
            <p>Monitor key performance indicators and get actionable insights.</p>
            <a href="/04_KPI_Dashboard" target="_self" style="text-decoration: none;">
                <div style="background-color: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 4px; display: inline-block; margin-top: 0.5rem;">
                    Go to Dashboard
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)

else:
    # Show instructions when no data is loaded
    st.warning("👆 Please upload your Excel files to get started with the AI analysis!")
    
    # Sample file formats in a modern card
    st.markdown("""
    <div class="feature-box">
        <h3>Sample File Formats</h3>
        
        <div style="margin-top: 1rem;">
            <h4 style="color: #4CAF50;">Sales History</h4>
            <ul>
                <li><strong>Required columns:</strong> date, sku, quantity</li>
                <li><strong>Optional columns:</strong> region, customer, revenue</li>
            </ul>
        </div>
        
        <div style="margin-top: 1rem;">
            <h4 style="color: #4CAF50;">Bill of Materials (BOM)</h4>
            <ul>
                <li><strong>Required columns:</strong> sku, material_id, quantity_required</li>
                <li><strong>Optional columns:</strong> material_name, unit_cost</li>
            </ul>
        </div>
        
        <div style="margin-top: 1rem;">
            <h4 style="color: #4CAF50;">Supplier Information</h4>
            <ul>
                <li><strong>Required columns:</strong> material_id, supplier_id, lead_time_days</li>
                <li><strong>Optional columns:</strong> moq (minimum order quantity), price_per_unit</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2023 AI Supply Chain Platform | All Rights Reserved")
