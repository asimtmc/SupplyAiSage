import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta
from utils.planning import calculate_material_requirements
from utils.visualization import plot_material_requirements

# Set page config
st.set_page_config(
    page_title="Material Procurement",
    page_icon="🧮",
    layout="wide"
)

# Check if data is loaded in session state
if 'forecasts' not in st.session_state or not st.session_state.forecasts:
    st.warning("Please run forecast analysis on the Demand Forecasting page first.")
    st.stop()

if 'bom_data' not in st.session_state or st.session_state.bom_data is None:
    st.warning("Please upload Bill of Materials (BOM) data on the main page first.")
    st.stop()

if 'supplier_data' not in st.session_state or st.session_state.supplier_data is None:
    st.warning("Please upload Supplier data on the main page first.")
    st.stop()

# Page title
st.title("Material Procurement Planning")
st.markdown("""
This module converts your production forecasts into material requirements plans.
Determine what materials to order, when to order them, and in what quantities.
""")

# Initialize session state variables for this page
if 'material_requirements' not in st.session_state:
    st.session_state.material_requirements = None
if 'run_mrp' not in st.session_state:
    st.session_state.run_mrp = False

# Create sidebar for settings
with st.sidebar:
    st.header("Procurement Settings")
    
    # Safety stock settings
    safety_stock_percent = st.slider(
        "Safety Stock Buffer (%)",
        min_value=0,
        max_value=50,
        value=15,
        step=5,
        help="Additional material to order as buffer"
    )
    
    # Order consolidation
    consolidation_days = st.slider(
        "Order Consolidation (Days)",
        min_value=0,
        max_value=30,
        value=7,
        step=1,
        help="Consolidate orders within this many days"
    )
    
    # Run MRP button
    if st.button("Calculate Material Requirements"):
        st.session_state.run_mrp = True
        with st.spinner("Calculating material requirements..."):
            # Apply safety stock to forecasts
            adjusted_forecasts = {}
            for sku, forecast in st.session_state.forecasts.items():
                adjusted_forecast = forecast.copy()
                adjusted_forecast['forecast'] = forecast['forecast'] * (1 + safety_stock_percent / 100)
                adjusted_forecasts[sku] = adjusted_forecast
            
            # Calculate material requirements
            mrp_result = calculate_material_requirements(
                adjusted_forecasts,
                st.session_state.bom_data,
                st.session_state.supplier_data
            )
            
            # Apply order consolidation if needed
            if consolidation_days > 0 and len(mrp_result) > 0:
                # Convert order_date to datetime if it's not already
                if not isinstance(mrp_result['order_date'].iloc[0], pd.Timestamp):
                    mrp_result['order_date'] = pd.to_datetime(mrp_result['order_date'])
                
                # Add a consolidation group column
                mrp_result['order_group'] = mrp_result['order_date'].dt.to_period('W')
                
                # Group by material_id and order_group
                consolidated = mrp_result.groupby(['material_id', 'order_group']).agg({
                    'quantity_required': 'sum',
                    'lead_time_days': 'max',
                    'moq': 'first',
                    'order_date': 'min'  # Use earliest date in group
                }).reset_index()
                
                # Recalculate order quantity
                consolidated['order_quantity'] = consolidated.apply(
                    lambda x: max(x['quantity_required'], x['moq']), axis=1
                )
                
                # Drop the consolidation group column
                consolidated = consolidated.drop('order_group', axis=1)
                
                # Store in session state
                st.session_state.material_requirements = consolidated
            else:
                # Store original result
                st.session_state.material_requirements = mrp_result
            
            st.success("Material requirements calculated successfully!")

# Main content
if st.session_state.run_mrp and st.session_state.material_requirements is not None:
    # Show material requirements summary
    st.header("Material Requirements Summary")
    
    # Create chart
    materials_fig = plot_material_requirements(st.session_state.material_requirements)
    st.plotly_chart(materials_fig, use_container_width=True)
    
    # Material requirements details
    st.header("Procurement Schedule")
    
    # Group by material
    material_totals = st.session_state.material_requirements.groupby('material_id').agg({
        'order_quantity': 'sum',
        'lead_time_days': 'mean'
    }).reset_index().sort_values('order_quantity', ascending=False)
    
    # Show table
    st.dataframe(material_totals, use_container_width=True)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["By Date", "By Material"])
    
    with tab1:
        # Sort by order date
        date_view = st.session_state.material_requirements.sort_values('order_date')
        
        # Format the date column
        date_view['order_date'] = date_view['order_date'].dt.strftime('%Y-%m-%d')
        
        # Show table
        st.dataframe(date_view, use_container_width=True)
        
        # Create a calendar heatmap of orders
        if len(date_view) > 0:
            # Group by date
            date_orders = date_view.groupby('order_date')['order_quantity'].sum().reset_index()
            date_orders['order_date'] = pd.to_datetime(date_orders['order_date'])
            
            # Create the figure
            fig = px.bar(
                date_orders,
                x='order_date',
                y='order_quantity',
                title='Order Quantities by Date',
                labels={'order_date': 'Order Date', 'order_quantity': 'Order Quantity'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Allow selecting a specific material
        materials = sorted(st.session_state.material_requirements['material_id'].unique())
        selected_material = st.selectbox("Select Material", options=materials)
        
        # Filter data
        material_view = st.session_state.material_requirements[
            st.session_state.material_requirements['material_id'] == selected_material
        ].sort_values('order_date')
        
        # Format the date column
        material_view['order_date'] = material_view['order_date'].dt.strftime('%Y-%m-%d')
        
        # Show table
        st.dataframe(material_view, use_container_width=True)
        
        # Material details from supplier data
        if 'supplier_data' in st.session_state and st.session_state.supplier_data is not None:
            supplier_info = st.session_state.supplier_data[
                st.session_state.supplier_data['material_id'] == selected_material
            ]
            
            if len(supplier_info) > 0:
                st.subheader("Supplier Information")
                st.dataframe(supplier_info, use_container_width=True)
    
    # Export option
    st.header("Export Material Requirements")
    
    if st.button("Prepare Material Requirements Export"):
        # Prepare export data
        export_data = st.session_state.material_requirements.copy()
        if 'order_date' in export_data.columns and isinstance(export_data['order_date'].iloc[0], pd.Timestamp):
            export_data['order_date'] = export_data['order_date'].dt.strftime('%Y-%m-%d')
        
        # Create a purchase order format
        po_data = export_data.copy()
        po_data['po_number'] = "PO-" + po_data.index.astype(str).str.zfill(4)
        
        # Add cost information if available
        if 'supplier_data' in st.session_state and 'price_per_unit' in st.session_state.supplier_data.columns:
            # Merge with supplier data to get price
            supplier_prices = st.session_state.supplier_data[['material_id', 'price_per_unit']]
            po_data = pd.merge(po_data, supplier_prices, on='material_id', how='left')
            
            # Calculate total cost
            po_data['total_cost'] = po_data['order_quantity'] * po_data['price_per_unit'].fillna(0)
        
        # Display preview
        st.subheader("Export Preview")
        st.dataframe(po_data.head(10), use_container_width=True)
        
        # Convert to Excel for download
        excel_buffer = io.BytesIO()
        po_data.to_excel(excel_buffer, index=False, engine='xlsxwriter')
        excel_buffer.seek(0)
        
        # Create download button
        st.download_button(
            label="Download Purchase Orders as Excel",
            data=excel_buffer,
            file_name=f"purchase_orders_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.ms-excel"
        )

else:
    # Show instructions when no material requirements have been calculated
    st.info("👈 Please configure and calculate material requirements using the sidebar.")
    
    # Show some useful information about the available data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bill of Materials (BOM) Preview")
        st.dataframe(st.session_state.bom_data.head(10), use_container_width=True)
        
        # BOM stats
        num_skus = len(st.session_state.bom_data['sku'].unique())
        num_materials = len(st.session_state.bom_data['material_id'].unique())
        
        st.metric("SKUs in BOM", num_skus)
        st.metric("Unique Materials", num_materials)
    
    with col2:
        st.subheader("Supplier Data Preview")
        st.dataframe(st.session_state.supplier_data.head(10), use_container_width=True)
        
        # Supplier stats
        num_suppliers = len(st.session_state.supplier_data['supplier_id'].unique()) if 'supplier_id' in st.session_state.supplier_data.columns else 0
        avg_lead_time = st.session_state.supplier_data['lead_time_days'].mean().round(1)
        
        st.metric("Number of Suppliers", num_suppliers)
        st.metric("Average Lead Time (days)", avg_lead_time)
    
    st.markdown("""
    ### Material Requirements Planning Process
    
    1. The system uses the demand forecasts you've already generated
    2. Combines this with your Bill of Materials (BOM) to determine material needs
    3. Accounts for supplier lead times and minimum order quantities
    4. Applies your safety stock preferences to ensure buffer inventory
    5. Consolidates orders within your specified timeframe to minimize order costs
    
    Click the "Calculate Material Requirements" button in the sidebar to start.
    """)
