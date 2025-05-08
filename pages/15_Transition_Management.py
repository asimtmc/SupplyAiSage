import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta
from streamlit_extras.colored_header import colored_header
from streamlit_extras.grid import grid
from utils.database import save_uploaded_file, get_file_by_type
from utils.data_loader import load_data_from_database
import plotly.figure_factory as ff

# Initialize session state variables
if 'transition_data' not in st.session_state:
    st.session_state.transition_data = None
if 'fg_master_data' not in st.session_state:
    st.session_state.fg_master_data = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'soh_data' not in st.session_state:
    st.session_state.soh_data = None
if 'open_orders_data' not in st.session_state:
    st.session_state.open_orders_data = None
if 'bom_data' not in st.session_state:
    st.session_state.bom_data = None
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "SKU Transition Dashboard"
if 'selected_sku' not in st.session_state:
    st.session_state.selected_sku = None
if 'filter_category' not in st.session_state:
    st.session_state.filter_category = None
if 'filter_component_type' not in st.session_state:
    st.session_state.filter_component_type = None
if 'date_range' not in st.session_state:
    st.session_state.date_range = None

# Auto-load data if not already loaded
load_data_from_database()

# Page title with styled header
colored_header(
    label="Transition Management Tool",
    description="Manage formulation and artwork transitions across your product portfolio",
    color_name="blue-70"
)

# Custom CSS for styling
st.markdown("""
<style>
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa421;
        font-weight: bold;
    }
    .risk-low {
        color: #26c281;
        font-weight: bold;
    }
    .status-planning {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 2px 8px;
    }
    .status-progress {
        background-color: #ffe0b2;
        border-radius: 4px;
        padding: 2px 8px;
    }
    .status-live {
        background-color: #c8e6c9;
        border-radius: 4px;
        padding: 2px 8px;
    }
    .alert-banner {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .info-card {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for global filters
with st.sidebar:
    st.header("Data Management")
    
    # File upload section
    st.subheader("Upload Data")
    
    # Create an expander for the upload with template information
    with st.expander("Upload Transition Management Excel File"):
        st.markdown("""
        ### Single Excel File with 6 Sheets
        Upload a single Excel file containing all 6 required sheets following the template structure.
        """)
        
        # Generate and display download template button
        from utils.sample_generator import generate_sample_transition_excel
        
        template_file = generate_sample_transition_excel()
        st.download_button(
            "📥 Download Template", 
            data=template_file, 
            file_name="Transition_Management_Template.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download a sample Excel template with the required structure"
        )
        
        # File uploader
        transition_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"], key="transition_upload")
        
        # Process uploaded file
        if transition_file is not None:
            try:
                # Save file to database first
                file_id = save_uploaded_file(transition_file, 'transition_data', 'Transition management data')
                st.success("File uploaded to database successfully!")
                
                # Process the file and load into session state
                # Create a progress bar for loading
                progress_bar = st.progress(0)
                st.info("Processing file, please wait...")
                
                # Reset file position before reading
                transition_file.seek(0)
                
                # Read all sheets from the Excel file
                excel_file = pd.ExcelFile(transition_file)
                
                # Check if all required sheets exist
                required_sheets = ["FG Master", "BOM", "FG Forecast", "SOH", "Open Orders", "Transition Timeline"]
                missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_file.sheet_names]
                
                if missing_sheets:
                    st.error(f"Error: Missing sheets in uploaded file: {', '.join(missing_sheets)}")
                else:
                    # Load each sheet with progress updates
                    progress_bar.progress(10)
                    
                    # Use the newly created function in data_loader to process the excel file
                    # This keeps the code DRY and ensures consistent processing
                    from utils.data_loader import load_transition_data_from_database
                    
                    # This will reload all data from database including the just-uploaded file
                    load_success = load_transition_data_from_database()
                    
                    progress_bar.progress(100)
                    
                    if load_success:
                        st.success("✅ All data loaded successfully! Please check the dashboard tabs.")
                        
                        # Provide a way to refresh the page
                        if st.button("Refresh Dashboard"):
                            st.rerun()
                    else:
                        # Show database load status errors if any
                        for key, status in st.session_state.db_load_status.items():
                            if status['status'] == 'error':
                                st.error(status['message'])
                            elif status['status'] == 'warning':
                                st.warning(status['message'])
                
            except Exception as e:
                st.error(f"Error processing Excel file: {str(e)}")
    
    # Add database status expander to show what's loaded
    with st.expander("Database Status"):
        if 'db_load_status' in st.session_state:
            for key, status in st.session_state.db_load_status.items():
                if status['status'] == 'success':
                    st.success(status['message'])
                elif status['status'] == 'warning':
                    st.warning(status['message'])
                elif status['status'] == 'error':
                    st.error(status['message'])
        else:
            st.info("Database status not available")
    
    # Global filters
    st.header("Filters")
    
    # Category filter
    if st.session_state.fg_master_data is not None and 'category' in st.session_state.fg_master_data.columns:
        categories = ['All'] + sorted(st.session_state.fg_master_data['category'].unique().tolist())
        st.session_state.filter_category = st.selectbox("Product Category", categories)
    
    # Component type filter
    if st.session_state.bom_data is not None and 'component_type' in st.session_state.bom_data.columns:
        component_types = ['All'] + sorted(st.session_state.bom_data['component_type'].unique().tolist())
        st.session_state.filter_component_type = st.selectbox("Component Type", component_types)
    
    # Transition Status filter
    statuses = ['All', 'Planning', 'In Progress', 'Go-Live']
    st.session_state.filter_status = st.selectbox("Transition Status", statuses)
    
    # Date range
    if st.session_state.transition_data is not None:
        # Determine date columns from transition data
        date_cols = [col for col in st.session_state.transition_data.columns if 'date' in col.lower()]
        if date_cols:
            min_date = datetime.now().date()
            max_date = min_date + timedelta(days=365)
            for col in date_cols:
                if pd.api.types.is_datetime64_any_dtype(st.session_state.transition_data[col]):
                    try:
                        col_min = st.session_state.transition_data[col].min().date()
                        col_max = st.session_state.transition_data[col].max().date()
                        min_date = min(min_date, col_min)
                        max_date = max(max_date, col_max)
                    except (AttributeError, ValueError) as e:
                        pass
            
            st.session_state.date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date - timedelta(days=30),
                max_value=max_date + timedelta(days=30)
            )

# Main content area with tabs for the core modules
tab_names = [
    "SKU Transition Dashboard", 
    "Impacted BOM Mapping", 
    "Material Depletion Simulator",
    "Smart Ordering Assistant", 
    "Write-Off Risk Analyzer", 
    "Transition Calendar View"
]

selected_tab = st.radio("Navigation", tab_names, horizontal=True)
st.session_state.selected_tab = selected_tab

# Check if data is loaded
if (st.session_state.transition_data is None or 
    st.session_state.fg_master_data is None or 
    st.session_state.forecast_data is None or 
    st.session_state.soh_data is None or 
    st.session_state.open_orders_data is None):
    
    st.warning("⚠️ Please upload all required data files in the sidebar to use the Transition Management Tool")
    
    # Show sample data structure expected
    with st.expander("Expected Excel File Structure"):
        st.markdown("""
        ### Required Excel File Structure
        
        The Excel file should contain the following **six sheets**, each with specific column headers:
        
        #### 1. FG Master
        | FG Code | Description      | Category   |
        |---------|------------------|------------|
        | FG1001  | Shampoo 200ml    | Hair Care  |
        
        #### 2. BOM (Bill of Materials)
        | FG Code | Component Code | Component Type | Qty per FG |
        |---------|----------------|----------------|------------|
        | FG1001  | RM101          | RM             | 0.5        |
        
        #### 3. FG Forecast
        | FG Code | Month   | Forecast Qty |
        |---------|---------|---------------|
        | FG1001  | 2024-06 | 10000         |
        
        #### 4. SOH (Stock on Hand)
        | Component Code | Component Type | Stock on Hand |
        |----------------|----------------|----------------|
        | RM101          | RM             | 5000           |
        
        #### 5. Open Orders
        | Component Code | Component Type | Open Order Qty | Expected Arrival |
        |----------------|----------------|----------------|------------------|
        | RM101          | RM             | 2000           | 2024-06-10       |
        
        #### 6. Transition Timeline
        | FG Code | Old RM/PM    | New RM/PM    | Start Date | Go-Live Date |
        |---------|--------------|--------------|------------|---------------|
        | FG1001  | RM101/PM201  | RM103/PM203  | 2024-06-01 | 2024-07-01    |
        
        ### Notes:
        - Column names must be **exactly as shown** (case-sensitive)
        - Dates should be in `YYYY-MM-DD` format
        - Upload a single Excel file with all six sheets
        """)
else:
    # 1. SKU Transition Dashboard
    if st.session_state.selected_tab == "SKU Transition Dashboard":
        st.header("SKU Transition Dashboard")
        
        # Create Gantt chart with mock data for now (will be replaced with real data)
        df = st.session_state.transition_data.copy()
        
        # Apply filters
        if st.session_state.filter_category != 'All' and st.session_state.fg_master_data is not None:
            filtered_skus = st.session_state.fg_master_data[
                st.session_state.fg_master_data['category'] == st.session_state.filter_category
            ]['sku_code'].unique().tolist()
            df = df[df['sku_code'].isin(filtered_skus)]
        
        if st.session_state.filter_status != 'All':
            df = df[df['status'] == st.session_state.filter_status]
        
        # Create Gantt chart data
        gantt_data = []
        for _, row in df.iterrows():
            start_date = row['planned_start_date']
            end_date = row['planned_go_live_date']
            sku = row['sku_code']
            status = row['status']
            
            # Map status to color
            color_map = {
                'Planning': 'rgb(169, 169, 169)',  # Grey
                'In Progress': 'rgb(255, 165, 0)',  # Orange
                'Go-Live': 'rgb(60, 179, 113)'  # Green
            }
            color = color_map.get(status, 'rgb(169, 169, 169)')
            
            gantt_data.append(dict(
                Task=sku, 
                Start=start_date, 
                Finish=end_date, 
                Status=status,
                Color=color
            ))
        
        if gantt_data:
            gantt_df = pd.DataFrame(gantt_data)
            fig = ff.create_gantt(
                gantt_df, 
                colors=gantt_df['Color'].tolist(),
                index_col='Status',
                show_colorbar=True,
                group_tasks=True,
                showgrid_x=True,
                showgrid_y=True,
                height=600
            )
            fig.update_layout(
                title="SKU Transition Schedule",
                xaxis_title="Timeline",
                yaxis_title="SKU",
                legend_title="Status",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # SKU Selection for detailed view
            st.subheader("SKU Details")
            selected_sku = st.selectbox("Select SKU for detailed view", 
                                        options=sorted(df['sku_code'].unique().tolist()))
            st.session_state.selected_sku = selected_sku
            
            # Display selected SKU details
            if selected_sku:
                sku_data = df[df['sku_code'] == selected_sku].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Transition Type", sku_data['transition_type'])
                with col2:
                    st.metric("Status", sku_data['status'])
                with col3:
                    st.metric("Priority", sku_data['priority'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Old Version", sku_data['old_version'])
                with col2:
                    st.metric("New Version", sku_data['new_version'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Planning Start", sku_data['planned_start_date'].strftime('%Y-%m-%d'))
                with col2:
                    st.metric("Go-Live Target", sku_data['planned_go_live_date'].strftime('%Y-%m-%d'))
                with col3:
                    if pd.notna(sku_data.get('actual_go_live_date')):
                        st.metric("Actual Go-Live", sku_data['actual_go_live_date'].strftime('%Y-%m-%d'))
                    else:
                        st.metric("Actual Go-Live", "Not yet")
        else:
            st.info("No transition data available based on the current filters.")
    
    # 2. Impacted BOM Mapping
    elif st.session_state.selected_tab == "Impacted BOM Mapping":
        st.header("Impacted BOM Mapping")
        
        # Load BOM data (assuming it's in session_state.bom_data)
        if st.session_state.bom_data is not None:
            # Get unique SKUs from transition data
            transition_skus = st.session_state.transition_data['sku_code'].unique().tolist()
            
            # Filter BOM data for transition SKUs
            # Use 'sku' column instead of 'sku_code' to match the actual column name in the BOM data
            impacted_bom = st.session_state.bom_data[st.session_state.bom_data['sku'].isin(transition_skus)]
            
            # Get unique materials from filtered BOM
            impacted_materials = impacted_bom['material_id'].unique().tolist()
            
            # Show summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("SKUs in Transition", len(transition_skus))
            with col2:
                st.metric("Impacted Materials", len(impacted_materials))
            with col3:
                st.metric("Total BOM Connections", len(impacted_bom))
            
            # Visualize BOM connections
            st.subheader("Material to SKU Mapping")
            
            # Create a network diagram or heatmap
            # For simplicity, showing a table first
            st.dataframe(impacted_bom[['sku', 'material_id', 'quantity_required']])
            
            # Create heatmap of material usage across SKUs
            pivot_data = impacted_bom.pivot_table(
                index='material_id',
                columns='sku',
                values='quantity_required',
                aggfunc='sum',
                fill_value=0
            )
            
            # Limit to top 20 materials and SKUs for visibility
            top_materials = pivot_data.sum(axis=1).nlargest(20).index
            top_skus = pivot_data.sum(axis=0).nlargest(20).index
            pivot_subset = pivot_data.loc[top_materials, top_skus]
            
            # Create heatmap
            fig = px.imshow(
                pivot_subset, 
                labels=dict(x="SKU Code", y="Material ID", color="Quantity Required"),
                title="Material Usage Across SKUs in Transition",
                color_continuous_scale='YlOrRd',
                aspect="auto"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Material selection for detailed view
            st.subheader("Material Details")
            selected_material = st.selectbox(
                "Select Material for detailed view", 
                options=sorted(impacted_materials)
            )
            
            if selected_material:
                # Show which SKUs use this material
                material_usage = impacted_bom[impacted_bom['material_id'] == selected_material]
                
                st.write(f"SKUs using {selected_material}:")
                st.dataframe(material_usage[['sku_code', 'quantity_required']])
                
                # Create bar chart of material usage across SKUs
                fig = px.bar(
                    material_usage, 
                    x='sku_code', 
                    y='quantity_required',
                    title=f"Usage of {selected_material} Across SKUs",
                    labels={'sku_code': 'SKU Code', 'quantity_required': 'Quantity Required'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("BOM data is required for this view. Please upload BOM data on the main page.")
    
    # 3. Material Depletion Simulator
    elif st.session_state.selected_tab == "Material Depletion Simulator":
        st.header("Material Depletion Simulator")
        
        # Display inventory vs forecast for materials
        if (st.session_state.bom_data is not None and 
            st.session_state.soh_data is not None and 
            st.session_state.forecast_data is not None):
            
            # Get materials used in transition SKUs
            transition_skus = st.session_state.transition_data['sku_code'].unique().tolist()
            impacted_bom = st.session_state.bom_data[st.session_state.bom_data['sku'].isin(transition_skus)]
            impacted_materials = impacted_bom['material_id'].unique().tolist()
            
            # Material selection
            selected_material = st.selectbox(
                "Select Material to simulate depletion", 
                options=sorted(impacted_materials)
            )
            
            if selected_material:
                # Get current inventory
                material_soh = st.session_state.soh_data[
                    st.session_state.soh_data['material_id'] == selected_material
                ]
                
                if len(material_soh) > 0:
                    current_stock = material_soh['qty_on_hand'].iloc[0]
                    
                    # Get SKUs using this material
                    skus_using_material = impacted_bom[
                        impacted_bom['material_id'] == selected_material
                    ]['sku_code'].unique().tolist()
                    
                    # Get forecast for these SKUs
                    sku_forecasts = st.session_state.forecast_data[
                        st.session_state.forecast_data['sku_code'].isin(skus_using_material)
                    ]
                    
                    # Calculate material consumption over time
                    forecast_dates = sorted(sku_forecasts['date'].unique())
                    consumption_over_time = []
                    remaining_stock = current_stock
                    
                    for date in forecast_dates:
                        # Get forecast for this date
                        date_forecast = sku_forecasts[sku_forecasts['date'] == date]
                        
                        # Calculate material consumption for this date
                        material_usage = 0
                        for _, forecast_row in date_forecast.iterrows():
                            sku = forecast_row['sku_code']
                            forecast_qty = forecast_row['forecast_qty']
                            
                            # Get BOM entry for this SKU and material
                            bom_entry = impacted_bom[
                                (impacted_bom['sku'] == sku) & 
                                (impacted_bom['material_id'] == selected_material)
                            ]
                            
                            if len(bom_entry) > 0:
                                qty_required = bom_entry['quantity_required'].iloc[0]
                                material_usage += forecast_qty * qty_required
                        
                        # Update remaining stock
                        remaining_stock -= material_usage
                        
                        consumption_over_time.append({
                            'date': date,
                            'consumption': material_usage,
                            'remaining_stock': max(0, remaining_stock)
                        })
                    
                    # Create DataFrame from consumption data
                    consumption_df = pd.DataFrame(consumption_over_time)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Stock", f"{current_stock:,.0f}")
                    with col2:
                        if len(consumption_df) > 0:
                            depletion_date = "Not depleted"
                            depleted_rows = consumption_df[consumption_df['remaining_stock'] == 0]
                            if len(depleted_rows) > 0:
                                depletion_date = depleted_rows.iloc[0]['date'].strftime('%Y-%m-%d')
                            st.metric("Estimated Depletion Date", depletion_date)
                    with col3:
                        if len(consumption_df) > 0:
                            days_until_depleted = "N/A"
                            depleted_rows = consumption_df[consumption_df['remaining_stock'] == 0]
                            if len(depleted_rows) > 0:
                                depletion_date = depleted_rows.iloc[0]['date']
                                days_until_depleted = (depletion_date - datetime.now().date()).days
                                if days_until_depleted < 0:
                                    days_until_depleted = "Already depleted"
                                else:
                                    days_until_depleted = f"{days_until_depleted} days"
                            st.metric("Time Until Depletion", days_until_depleted)
                    
                    # Create line chart of material depletion
                    fig = go.Figure()
                    
                    # Add remaining stock line
                    fig.add_trace(go.Scatter(
                        x=consumption_df['date'],
                        y=consumption_df['remaining_stock'],
                        mode='lines+markers',
                        name='Remaining Stock',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # Add consumption bars
                    fig.add_trace(go.Bar(
                        x=consumption_df['date'],
                        y=consumption_df['consumption'],
                        name='Consumption',
                        marker_color='indianred'
                    ))
                    
                    # Add reference line for current stock
                    fig.add_shape(
                        type="line",
                        x0=consumption_df['date'].min(),
                        y0=current_stock,
                        x1=consumption_df['date'].max(),
                        y1=current_stock,
                        line=dict(
                            color="green",
                            width=2,
                            dash="dash",
                        )
                    )
                    
                    # Add annotation for current stock
                    fig.add_annotation(
                        x=consumption_df['date'].min(),
                        y=current_stock,
                        text="Current Stock Level",
                        showarrow=True,
                        arrowhead=1,
                        ax=-50,
                        ay=-30
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Material Depletion Forecast for {selected_material}",
                        xaxis_title="Date",
                        yaxis_title="Quantity",
                        barmode='overlay',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk assessment
                    st.subheader("Risk Assessment")
                    
                    # Determine if there's a depletion risk
                    depletion_risk = False
                    days_to_depletion = None
                    
                    depleted_rows = consumption_df[consumption_df['remaining_stock'] == 0]
                    if len(depleted_rows) > 0:
                        depletion_date = depleted_rows.iloc[0]['date']
                        days_to_depletion = (depletion_date - datetime.now().date()).days
                        if days_to_depletion < 30:  # Less than 30 days is a risk
                            depletion_risk = True
                    
                    # Display risk alert if necessary
                    if depletion_risk:
                        st.markdown(
                            f"""
                            <div class="alert-banner">
                                <h4>⚠️ Material Depletion Risk</h4>
                                <p>Material {selected_material} is projected to be depleted in {days_to_depletion} days.</p>
                                <p>Consider adjusting production schedules or placing additional orders.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="info-card">
                                <h4>✅ No Immediate Depletion Risk</h4>
                                <p>Material {selected_material} has sufficient stock for the forecasted demand.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.warning(f"No stock on hand data found for material {selected_material}")
        else:
            st.warning("BOM, SOH, and Forecast data are required for this simulation. Please upload all necessary data files.")
    
    # 4. Smart Ordering Assistant
    elif st.session_state.selected_tab == "Smart Ordering Assistant":
        st.header("Smart Ordering Assistant")
        
        # Guide for optimal ordering during transitions
        if (st.session_state.bom_data is not None and 
            st.session_state.soh_data is not None and
            st.session_state.forecast_data is not None and
            st.session_state.open_orders_data is not None):
            
            # Get materials used in transition SKUs
            transition_skus = st.session_state.transition_data['sku_code'].unique().tolist()
            impacted_bom = st.session_state.bom_data[st.session_state.bom_data['sku'].isin(transition_skus)]
            impacted_materials = impacted_bom['material_id'].unique().tolist()
            
            # Calculate ordering recommendations
            recommendations = []
            for material in impacted_materials:
                # Get current stock
                material_soh = st.session_state.soh_data[
                    st.session_state.soh_data['material_id'] == material
                ]
                
                if len(material_soh) == 0:
                    continue
                    
                current_stock = material_soh['qty_on_hand'].iloc[0]
                
                # Get SKUs using this material
                skus_using_material = impacted_bom[
                    impacted_bom['material_id'] == material
                ]['sku_code'].unique().tolist()
                
                # Get forecast for these SKUs
                sku_forecasts = st.session_state.forecast_data[
                    st.session_state.forecast_data['sku_code'].isin(skus_using_material)
                ]
                
                # Calculate material consumption over time
                forecast_dates = sorted(sku_forecasts['date'].unique())
                remaining_stock = current_stock
                depletion_date = None
                
                # Check for open orders
                open_orders = st.session_state.open_orders_data[
                    st.session_state.open_orders_data['material_id'] == material
                ]
                
                total_on_order = open_orders['order_qty'].sum() if len(open_orders) > 0 else 0
                
                # Calculate days of supply
                daily_usage = 0
                if len(sku_forecasts) > 0:
                    # Calculate average daily usage
                    total_usage = 0
                    for _, forecast_row in sku_forecasts.iterrows():
                        sku = forecast_row['sku_code']
                        forecast_qty = forecast_row['forecast_qty']
                        
                        # Get BOM entry for this SKU and material
                        bom_entry = impacted_bom[
                            (impacted_bom['sku'] == sku) & 
                            (impacted_bom['material_id'] == material)
                        ]
                        
                        if len(bom_entry) > 0:
                            qty_required = bom_entry['quantity_required'].iloc[0]
                            total_usage += forecast_qty * qty_required
                    
                    daily_usage = total_usage / len(forecast_dates) if len(forecast_dates) > 0 else 0
                
                days_of_supply = int(current_stock / daily_usage) if daily_usage > 0 else float('inf')
                
                # Determine action based on days of supply
                action = "No action needed"
                if days_of_supply < 30:
                    action = "Order immediately"
                elif days_of_supply < 60:
                    action = "Schedule order soon"
                
                # Check transition impact
                transition_impact = "Low"
                material_usage_count = len(skus_using_material)
                if material_usage_count > 5:
                    transition_impact = "High"
                elif material_usage_count > 2:
                    transition_impact = "Medium"
                
                # Add recommendation
                recommendations.append({
                    'material_id': material,
                    'current_stock': current_stock,
                    'on_order': total_on_order,
                    'daily_usage': daily_usage,
                    'days_of_supply': days_of_supply,
                    'transition_impact': transition_impact,
                    'action': action
                })
            
            # Convert to DataFrame
            if recommendations:
                recommendations_df = pd.DataFrame(recommendations)
                
                # Sort by days of supply (ascending)
                recommendations_df = recommendations_df.sort_values('days_of_supply')
                
                # Display urgent actions
                st.subheader("Urgent Actions")
                urgent_df = recommendations_df[recommendations_df['days_of_supply'] < 30]
                
                if len(urgent_df) > 0:
                    st.dataframe(urgent_df)
                    
                    # Create bar chart of days of supply for urgent materials
                    fig = px.bar(
                        urgent_df,
                        x='material_id',
                        y='days_of_supply',
                        color='transition_impact',
                        color_discrete_map={
                            'High': 'red',
                            'Medium': 'orange',
                            'Low': 'green'
                        },
                        title="Days of Supply for Urgent Materials",
                        labels={
                            'material_id': 'Material ID',
                            'days_of_supply': 'Days of Supply',
                            'transition_impact': 'Transition Impact'
                        }
                    )
                    fig.update_layout(xaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No urgent actions required")
                
                # Display all recommendations
                st.subheader("All Recommendations")
                st.dataframe(recommendations_df)
                
                # Material selection for detailed recommendation
                st.subheader("Material-Specific Recommendation")
                selected_material = st.selectbox(
                    "Select Material for detailed recommendation",
                    options=sorted(recommendations_df['material_id'].unique().tolist())
                )
                
                if selected_material:
                    material_rec = recommendations_df[recommendations_df['material_id'] == selected_material].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Stock", f"{material_rec['current_stock']:,.0f}")
                    with col2:
                        st.metric("On Order", f"{material_rec['on_order']:,.0f}")
                    with col3:
                        st.metric("Days of Supply", f"{material_rec['days_of_supply']:,.0f}")
                    
                    # Create gauge chart for days of supply
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=material_rec['days_of_supply'],
                        title={'text': "Days of Supply"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "red"},
                                {'range': [30, 60], 'color': "orange"},
                                {'range': [60, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 30
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendation box
                    if material_rec['days_of_supply'] < 30:
                        st.markdown(
                            f"""
                            <div class="alert-banner">
                                <h4>🚨 Urgent Action Required</h4>
                                <p>Material {selected_material} has only {material_rec['days_of_supply']:.0f} days of supply remaining.</p>
                                <p><strong>Recommendation:</strong> Place order immediately for at least {material_rec['daily_usage'] * 60:.0f} units (60 days of supply).</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    elif material_rec['days_of_supply'] < 60:
                        st.markdown(
                            f"""
                            <div class="info-card" style="background-color: #fff3e0; border-left-color: #ff9800;">
                                <h4>⚠️ Action Recommended</h4>
                                <p>Material {selected_material} has {material_rec['days_of_supply']:.0f} days of supply remaining.</p>
                                <p><strong>Recommendation:</strong> Schedule order within 2 weeks for {material_rec['daily_usage'] * 60:.0f} units (60 days of supply).</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="info-card" style="background-color: #e8f5e9; border-left-color: #4caf50;">
                                <h4>✅ No Immediate Action Required</h4>
                                <p>Material {selected_material} has sufficient supply ({material_rec['days_of_supply']:.0f} days).</p>
                                <p><strong>Recommendation:</strong> Monitor usage and review in 30 days.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            st.warning("BOM, SOH, Forecast, and Open Orders data are required for ordering recommendations. Please upload all necessary data files.")
    
    # 5. Write-Off Risk Analyzer
    elif st.session_state.selected_tab == "Write-Off Risk Analyzer":
        st.header("Write-Off Risk Analyzer")
        
        if (st.session_state.bom_data is not None and 
            st.session_state.soh_data is not None):
            
            # Get materials that will be obsolete after transitions
            transition_data = st.session_state.transition_data.copy()
            obsolete_materials = []
            
            for _, transition in transition_data.iterrows():
                sku = transition['sku_code']
                old_version = transition['old_version']
                new_version = transition['new_version']
                transition_type = transition['transition_type']
                status = transition['status']
                
                # Skip completed transitions
                if status == 'Go-Live':
                    continue
                
                # Get BOM for old version
                # This is a simplification, we'd need actual version-specific BOM data
                old_bom = st.session_state.bom_data[st.session_state.bom_data['sku'] == sku]
                
                # Simulate obsolete materials based on transition type
                if transition_type == 'Formulation':
                    # For formulation changes, assume some raw materials will be obsolete
                    # In a real system, we'd have version-specific BOM data
                    for _, bom_item in old_bom.iterrows():
                        material = bom_item['material_id']
                        qty_required = bom_item['quantity_required']
                        
                        # Get current stock
                        material_soh = st.session_state.soh_data[
                            st.session_state.soh_data['material_id'] == material
                        ]
                        
                        if len(material_soh) > 0:
                            current_stock = material_soh['qty_on_hand'].iloc[0]
                            
                            # Check if this material will be obsolete
                            # For demonstration, assume 50% chance a material becomes obsolete in formulation changes
                            is_obsolete = hash(material + sku) % 2 == 0  # Deterministic "random" for demo
                            
                            if is_obsolete:
                                obsolete_materials.append({
                                    'sku_code': sku,
                                    'material_id': material,
                                    'current_stock': current_stock,
                                    'transition_type': transition_type,
                                    'planned_go_live': transition['planned_go_live_date'],
                                    'obsolescence_reason': 'Formula change',
                                    'write_off_value': current_stock * 1.5  # Dummy value multiplier
                                })
                elif transition_type == 'Artwork':
                    # For artwork changes, packaging materials may be obsolete
                    for _, bom_item in old_bom.iterrows():
                        material = bom_item['material_id']
                        qty_required = bom_item['quantity_required']
                        
                        # Check if this is a packaging material (simplified logic)
                        is_packaging = material.startswith('PKG') or 'LABEL' in material
                        
                        if is_packaging:
                            # Get current stock
                            material_soh = st.session_state.soh_data[
                                st.session_state.soh_data['material_id'] == material
                            ]
                            
                            if len(material_soh) > 0:
                                current_stock = material_soh['qty_on_hand'].iloc[0]
                                
                                obsolete_materials.append({
                                    'sku_code': sku,
                                    'material_id': material,
                                    'current_stock': current_stock,
                                    'transition_type': transition_type,
                                    'planned_go_live': transition['planned_go_live_date'],
                                    'obsolescence_reason': 'Artwork change',
                                    'write_off_value': current_stock * 2.0  # Dummy value multiplier
                                })
            
            # Convert to DataFrame
            if obsolete_materials:
                obsolete_df = pd.DataFrame(obsolete_materials)
                
                # Sort by write-off value (descending)
                obsolete_df = obsolete_df.sort_values('write_off_value', ascending=False)
                
                # Display total write-off value
                total_write_off = obsolete_df['write_off_value'].sum()
                
                st.metric("Total Potential Write-Off Value", f"${total_write_off:,.2f}")
                
                # Create pie chart of write-off by transition type
                write_off_by_type = obsolete_df.groupby('transition_type')['write_off_value'].sum().reset_index()
                
                fig = px.pie(
                    write_off_by_type,
                    values='write_off_value',
                    names='transition_type',
                    title="Write-Off Value by Transition Type",
                    hole=0.4
                )
                
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Create bar chart of top materials by write-off value
                    top_materials = obsolete_df.groupby('material_id')['write_off_value'].sum().reset_index()
                    top_materials = top_materials.sort_values('write_off_value', ascending=False).head(10)
                    
                    fig = px.bar(
                        top_materials,
                        x='material_id',
                        y='write_off_value',
                        title="Top 10 Materials by Write-Off Value",
                        labels={
                            'material_id': 'Material ID',
                            'write_off_value': 'Write-Off Value ($)'
                        }
                    )
                    fig.update_layout(xaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed write-off risks
                st.subheader("Detailed Write-Off Risks")
                st.dataframe(obsolete_df)
                
                # Risk mitigation suggestions
                st.subheader("Risk Mitigation Suggestions")
                
                # Generate suggestions based on data
                high_value_materials = obsolete_df[obsolete_df['write_off_value'] > 1000]['material_id'].unique().tolist()
                
                if high_value_materials:
                    st.markdown(
                        f"""
                        <div class="alert-banner">
                            <h4>⚠️ High-Value Write-Off Risks Identified</h4>
                            <p>The following materials have significant write-off potential:</p>
                            <ul>
                                {"".join([f"<li><strong>{m}</strong>: ${obsolete_df[obsolete_df['material_id'] == m]['write_off_value'].iloc[0]:,.2f}</li>" for m in high_value_materials[:5]])}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("""
                    ### Suggested Mitigation Actions:
                    
                    1. **Align transition dates** to maximize material usage
                    2. **Explore alternative uses** for high-value materials
                    3. **Negotiate returns** with suppliers where possible
                    4. **Adjust production plans** to deplete stock before transition
                    5. **Evaluate packaging sharing** across product lines
                    """)
            else:
                st.info("No write-off risks identified based on the current transition plan.")
        else:
            st.warning("BOM and SOH data are required for write-off risk analysis. Please upload all necessary data files.")
    
    # 6. Transition Calendar View
    elif st.session_state.selected_tab == "Transition Calendar View":
        st.header("Transition Calendar View")
        
        # Create calendar view of all transitions
        if st.session_state.transition_data is not None:
            df = st.session_state.transition_data.copy()
            
            # Apply filters
            if st.session_state.filter_category != 'All' and st.session_state.fg_master_data is not None:
                filtered_skus = st.session_state.fg_master_data[
                    st.session_state.fg_master_data['Category'] == st.session_state.filter_category
                ]['sku_code'].unique().tolist()
                df = df[df['sku_code'].isin(filtered_skus)]
            
            if st.session_state.filter_status != 'All':
                df = df[df['status'] == st.session_state.filter_status]
            
            # Create timeline data
            timeline_data = []
            for _, row in df.iterrows():
                sku = row['sku_code']
                start_date = row['planned_start_date']
                end_date = row['planned_go_live_date']
                status = row['status']
                transition_type = row['transition_type']
                
                # Map status to color
                color_map = {
                    'Planning': 'grey',
                    'In Progress': 'orange',
                    'Go-Live': 'green'
                }
                color = color_map.get(status, 'grey')
                
                # Create timeline item
                timeline_data.append({
                    'Task': sku,
                    'Resource': transition_type,
                    'Start': start_date,
                    'Finish': end_date,
                    'Status': status,
                    'Color': color
                })
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                
                # Group by month
                timeline_df['Month'] = timeline_df['Start'].dt.strftime('%Y-%m')
                
                # Count transitions by month and type
                transitions_by_month = timeline_df.groupby(['Month', 'Resource']).size().reset_index(name='Count')
                
                # Create stacked bar chart of transitions by month
                fig = px.bar(
                    transitions_by_month,
                    x='Month',
                    y='Count',
                    color='Resource',
                    title="Transitions by Month and Type",
                    labels={
                        'Month': 'Month',
                        'Count': 'Number of Transitions',
                        'Resource': 'Transition Type'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create timeline chart using plotly
                fig = px.timeline(
                    timeline_df, 
                    x_start='Start', 
                    x_end='Finish', 
                    y='Task',
                    color='Status',
                    color_discrete_map={
                        'Planning': 'grey',
                        'In Progress': 'orange',
                        'Go-Live': 'green'
                    },
                    hover_data=['Resource']
                )
                
                fig.update_layout(
                    title="Transition Timeline",
                    xaxis_title="Date",
                    yaxis_title="SKU",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create heatmap of transition density
                timeline_df['StartWeek'] = timeline_df['Start'].dt.strftime('%Y-%U')
                timeline_df['Resource'] = timeline_df['Resource']
                
                # Count transitions by week and type
                transitions_by_week = timeline_df.groupby(['StartWeek', 'Resource']).size().reset_index(name='Count')
                
                # Pivot for heatmap
                heatmap_data = transitions_by_week.pivot(
                    index='Resource',
                    columns='StartWeek',
                    values='Count'
                ).fillna(0)
                
                # Create heatmap
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Week", y="Transition Type", color="Count"),
                    title="Transition Density Heatmap",
                    color_continuous_scale='YlOrRd',
                    aspect="auto"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show overlapping transitions
                st.subheader("Overlapping Transitions")
                
                # Find overlapping date ranges
                overlaps = []
                for i, row1 in timeline_df.iterrows():
                    for j, row2 in timeline_df.iterrows():
                        if i < j:  # Avoid comparing the same pair twice
                            # Check if date ranges overlap
                            if (row1['Start'] <= row2['Finish'] and row1['Finish'] >= row2['Start']):
                                overlap_days = min(row1['Finish'], row2['Finish']) - max(row1['Start'], row2['Start'])
                                overlap_days = overlap_days.days
                                
                                if overlap_days > 0:
                                    overlaps.append({
                                        'SKU1': row1['Task'],
                                        'SKU2': row2['Task'],
                                        'Start': max(row1['Start'], row2['Start']),
                                        'End': min(row1['Finish'], row2['Finish']),
                                        'Overlap_Days': overlap_days,
                                        'Type1': row1['Resource'],
                                        'Type2': row2['Resource']
                                    })
                
                if overlaps:
                    overlaps_df = pd.DataFrame(overlaps)
                    
                    # Sort by overlap days
                    overlaps_df = overlaps_df.sort_values('Overlap_Days', ascending=False)
                    
                    # Display total and average overlaps
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Overlapping Transitions", len(overlaps_df))
                    with col2:
                        st.metric("Max Overlap Days", overlaps_df['Overlap_Days'].max())
                    with col3:
                        st.metric("Avg Overlap Days", round(overlaps_df['Overlap_Days'].mean(), 1))
                    
                    # Display overlapping transitions
                    st.dataframe(overlaps_df)
                    
                    # Create network graph of overlapping SKUs
                    if len(overlaps_df) > 0:
                        # Extract unique SKUs
                        all_skus = set(overlaps_df['SKU1'].tolist() + overlaps_df['SKU2'].tolist())
                        
                        # Create edges
                        edges = []
                        for _, row in overlaps_df.iterrows():
                            edges.append((row['SKU1'], row['SKU2'], row['Overlap_Days']))
                        
                        # Create nodes
                        nodes = list(all_skus)
                        
                        # Placeholder for a network graph - in a real implementation, 
                        # we would use a library like networkx with plotly for visualization
                        st.info("Network visualization of overlapping transitions would be shown here")
                        
                else:
                    st.info("No overlapping transitions found.")
            else:
                st.info("No transition data available based on the current filters.")
        else:
            st.warning("Transition data is required for the calendar view. Please upload transition data.")