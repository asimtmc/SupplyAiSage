"""
Sample data generator for Transition Management Excel template.
"""
import pandas as pd
import io
from datetime import datetime, timedelta

def generate_sample_transition_excel():
    """
    Generates a sample Excel file with the required structure for transition management.
    Returns a BytesIO object containing the Excel file.
    """
    # Create a BytesIO object to hold the Excel file
    output = io.BytesIO()
    
    # Create an Excel writer
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # 1. FG Master
    fg_master_data = {
        'FG Code': ['FG1001', 'FG1002', 'FG1003', 'FG1004', 'FG1005'],
        'Description': ['Shampoo 200ml', 'Conditioner 200ml', 'Body Wash 250ml', 'Face Cream 50g', 'Hand Soap 300ml'],
        'Category': ['Hair Care', 'Hair Care', 'Body Care', 'Face Care', 'Hand Care']
    }
    fg_master_df = pd.DataFrame(fg_master_data)
    fg_master_df.to_excel(writer, sheet_name='FG Master', index=False)
    
    # 2. BOM (Bill of Materials)
    bom_data = []
    for fg_code in fg_master_data['FG Code']:
        # Add RM components (raw materials)
        for i in range(1, 4):  # 3 raw materials per product
            bom_data.append({
                'FG Code': fg_code,
                'Component Code': f'RM{100 + i}',
                'Component Type': 'RM',
                'Qty per FG': round(0.1 * i, 2)
            })
        
        # Add PM components (packaging materials)
        for i in range(1, 3):  # 2 packaging materials per product
            bom_data.append({
                'FG Code': fg_code,
                'Component Code': f'PM{200 + i}',
                'Component Type': 'PM',
                'Qty per FG': i
            })
    
    bom_df = pd.DataFrame(bom_data)
    bom_df.to_excel(writer, sheet_name='BOM', index=False)
    
    # 3. FG Forecast
    today = datetime.now()
    forecast_data = []
    
    for fg_code in fg_master_data['FG Code']:
        for month_offset in range(6):  # 6 months forecast
            month_date = today.replace(day=1) + timedelta(days=30 * month_offset)
            forecast_qty = 5000 + (month_offset * 500)  # Increasing forecast
            
            forecast_data.append({
                'FG Code': fg_code,
                'Month': month_date.strftime('%Y-%m'),
                'Forecast Qty': forecast_qty
            })
    
    forecast_df = pd.DataFrame(forecast_data)
    forecast_df.to_excel(writer, sheet_name='FG Forecast', index=False)
    
    # 4. SOH (Stock on Hand)
    soh_data = []
    
    # RM stock
    for i in range(1, 6):
        soh_data.append({
            'Component Code': f'RM{100 + i}',
            'Component Type': 'RM',
            'Stock on Hand': 10000 + (i * 1000)
        })
    
    # PM stock
    for i in range(1, 5):
        soh_data.append({
            'Component Code': f'PM{200 + i}',
            'Component Type': 'PM',
            'Stock on Hand': 20000 + (i * 2000)
        })
    
    soh_df = pd.DataFrame(soh_data)
    soh_df.to_excel(writer, sheet_name='SOH', index=False)
    
    # 5. Open Orders
    open_orders_data = []
    
    # Some open orders for RMs
    for i in range(1, 4):
        days_to_arrival = i * 7  # 7, 14, 21 days
        open_orders_data.append({
            'Component Code': f'RM{100 + i}',
            'Component Type': 'RM',
            'Open Order Qty': 5000,
            'Expected Arrival': (today + timedelta(days=days_to_arrival)).strftime('%Y-%m-%d')
        })
    
    # Some open orders for PMs
    for i in range(1, 3):
        days_to_arrival = i * 10  # 10, 20 days
        open_orders_data.append({
            'Component Code': f'PM{200 + i}',
            'Component Type': 'PM',
            'Open Order Qty': 10000,
            'Expected Arrival': (today + timedelta(days=days_to_arrival)).strftime('%Y-%m-%d')
        })
    
    open_orders_df = pd.DataFrame(open_orders_data)
    open_orders_df.to_excel(writer, sheet_name='Open Orders', index=False)
    
    # 6. Transition Timeline
    transition_data = [
        {
            'FG Code': 'FG1001',
            'Old RM/PM': 'RM101/PM201',
            'New RM/PM': 'RM103/PM203',
            'Start Date': (today - timedelta(days=30)).strftime('%Y-%m-%d'),
            'Go-Live Date': (today + timedelta(days=30)).strftime('%Y-%m-%d')
        },
        {
            'FG Code': 'FG1002',
            'Old RM/PM': 'RM102',
            'New RM/PM': 'RM104',
            'Start Date': (today - timedelta(days=15)).strftime('%Y-%m-%d'),
            'Go-Live Date': (today + timedelta(days=45)).strftime('%Y-%m-%d')
        },
        {
            'FG Code': 'FG1003',
            'Old RM/PM': 'PM201',
            'New RM/PM': 'PM204',
            'Start Date': today.strftime('%Y-%m-%d'),
            'Go-Live Date': (today + timedelta(days=60)).strftime('%Y-%m-%d')
        },
        {
            'FG Code': 'FG1004',
            'Old RM/PM': 'RM103/PM202',
            'New RM/PM': 'RM105/PM204',
            'Start Date': (today + timedelta(days=15)).strftime('%Y-%m-%d'),
            'Go-Live Date': (today + timedelta(days=75)).strftime('%Y-%m-%d')
        },
        {
            'FG Code': 'FG1005',
            'Old RM/PM': 'PM201',
            'New RM/PM': 'PM203',
            'Start Date': (today + timedelta(days=30)).strftime('%Y-%m-%d'),
            'Go-Live Date': (today + timedelta(days=90)).strftime('%Y-%m-%d')
        }
    ]
    
    transition_df = pd.DataFrame(transition_data)
    transition_df.to_excel(writer, sheet_name='Transition Timeline', index=False)
    
    # Save the Excel file
    writer.close()
    
    # Reset the pointer to the beginning of the BytesIO object
    output.seek(0)
    
    return output