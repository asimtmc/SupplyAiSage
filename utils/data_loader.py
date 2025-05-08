"""
Data loading utility for the Supply Chain Platform.
This module handles loading data from the database into session state.
"""
import streamlit as st
import pandas as pd
from utils.database import get_file_by_type
from datetime import datetime

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
        
        # Also load transition management data if available
        load_transition_data_from_database()
        
        return True
    
    except Exception as e:
        st.session_state.db_load_status['error'] = {
            'status': 'error',
            'message': f"❌ Error loading data: {str(e)}"
        }
        return False

def load_transition_data_from_database():
    """
    Function to load transition management data from the database.
    Sets session state variables for transition management.
    """
    try:
        # Load transition data
        transition_file = get_file_by_type('transition_data')
        if transition_file:
            # Process the Excel file with all sheets for transition management
            try:
                excel_file = pd.ExcelFile(pd.read_excel(transition_file[1]))
                
                # Check if all required sheets exist
                required_sheets = ["FG Master", "BOM", "FG Forecast", "SOH", "Open Orders", "Transition Timeline"]
                missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_file.sheet_names]
                
                if not missing_sheets:
                    # 1. Load FG Master
                    fg_master_df = pd.read_excel(excel_file, sheet_name="FG Master")
                    fg_master_df = fg_master_df.rename(columns={
                        "FG Code": "sku_code",
                        "Description": "description",
                        "Category": "category"
                    })
                    st.session_state.fg_master_data = fg_master_df
                    
                    # 2. Load BOM (if not already loaded)
                    if st.session_state.bom_data is None:
                        bom_df = pd.read_excel(excel_file, sheet_name="BOM")
                        bom_df = bom_df.rename(columns={
                            "FG Code": "sku_code",
                            "Component Code": "material_id",
                            "Component Type": "component_type",
                            "Qty per FG": "quantity_required"
                        })
                        st.session_state.bom_data = bom_df
                    
                    # 3. Load FG Forecast
                    forecast_df = pd.read_excel(excel_file, sheet_name="FG Forecast")
                    forecast_df = forecast_df.rename(columns={
                        "FG Code": "sku_code",
                        "Month": "date",
                        "Forecast Qty": "forecast_qty"
                    })
                    if not pd.api.types.is_datetime64_any_dtype(forecast_df["date"]):
                        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
                    st.session_state.forecast_data = forecast_df
                    
                    # 4. Load SOH
                    soh_df = pd.read_excel(excel_file, sheet_name="SOH")
                    soh_df = soh_df.rename(columns={
                        "Component Code": "material_id",
                        "Component Type": "component_type",
                        "Stock on Hand": "qty_on_hand"
                    })
                    st.session_state.soh_data = soh_df
                    
                    # 5. Load Open Orders
                    open_orders_df = pd.read_excel(excel_file, sheet_name="Open Orders")
                    open_orders_df = open_orders_df.rename(columns={
                        "Component Code": "material_id",
                        "Component Type": "component_type",
                        "Open Order Qty": "order_qty",
                        "Expected Arrival": "expected_date"
                    })
                    if not pd.api.types.is_datetime64_any_dtype(open_orders_df["expected_date"]):
                        open_orders_df["expected_date"] = pd.to_datetime(open_orders_df["expected_date"])
                    st.session_state.open_orders_data = open_orders_df
                    
                    # 6. Load Transition Timeline
                    transition_df = pd.read_excel(excel_file, sheet_name="Transition Timeline")
                    
                    # Convert dates to datetime
                    if not pd.api.types.is_datetime64_any_dtype(transition_df["Start Date"]):
                        transition_df["Start Date"] = pd.to_datetime(transition_df["Start Date"])
                    if not pd.api.types.is_datetime64_any_dtype(transition_df["Go-Live Date"]):
                        transition_df["Go-Live Date"] = pd.to_datetime(transition_df["Go-Live Date"])
                    
                    # Add status field based on dates
                    today = datetime.now().date()
                    transition_df["status"] = "Planning"
                    
                    for i, row in transition_df.iterrows():
                        start_date = row["Start Date"].date() if hasattr(row["Start Date"], "date") else row["Start Date"]
                        end_date = row["Go-Live Date"].date() if hasattr(row["Go-Live Date"], "date") else row["Go-Live Date"]
                        
                        if today >= end_date:
                            transition_df.at[i, "status"] = "Go-Live"
                        elif today >= start_date:
                            transition_df.at[i, "status"] = "In Progress"
                    
                    # Rename columns for consistency
                    transition_df = transition_df.rename(columns={
                        "FG Code": "sku_code",
                        "Old RM/PM": "old_version",
                        "New RM/PM": "new_version",
                        "Start Date": "planned_start_date",
                        "Go-Live Date": "planned_go_live_date"
                    })
                    
                    # Add transition type based on RM/PM naming
                    transition_df["transition_type"] = transition_df["old_version"].apply(
                        lambda x: "Formulation" if "RM" in str(x) else "Artwork"
                    )
                    
                    # Add priority (default medium)
                    transition_df["priority"] = "Medium"
                    
                    st.session_state.transition_data = transition_df
                    
                    # Update status
                    st.session_state.db_load_status['transition_data'] = {
                        'status': 'success',
                        'message': f"✅ Successfully loaded transition management data: {transition_file[0]}"
                    }
                    
                    return True
                else:
                    st.session_state.db_load_status['transition_data'] = {
                        'status': 'warning',
                        'message': f"⚠️ Missing sheets in transition data: {', '.join(missing_sheets)}"
                    }
                    return False
            except Exception as e:
                st.session_state.db_load_status['transition_data'] = {
                    'status': 'error',
                    'message': f"❌ Error processing transition data: {str(e)}"
                }
                return False
        else:
            # No transition data found
            if 'transition_data' not in st.session_state.db_load_status:
                st.session_state.db_load_status['transition_data'] = {
                    'status': 'warning',
                    'message': "⚠️ No transition data found in database"
                }
            return False
            
    except Exception as e:
        st.session_state.db_load_status['transition_data'] = {
            'status': 'error',
            'message': f"❌ Error loading transition data: {str(e)}"
        }
        return False