import pandas as pd
import io
import streamlit as st

# Apply caching to data processing functions
@st.cache_data(ttl=3600, show_spinner=False)
def process_sales_data(file):
    """
    Process and clean sales history data from Excel file

    Parameters:
    -----------
    file : file object
        The uploaded Excel file containing sales data

    Returns:
    --------
    pandas.DataFrame
        Processed and cleaned sales data
    """
    # Read Excel file
    df = pd.read_excel(file)

    # Standardize column names (lowercase and remove spaces)
    df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]

    # Check required columns
    required_cols = ['date', 'sku', 'quantity']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        # If the column names are different, try to find similar columns
        if 'date' in missing_cols and any(col for col in df.columns if 'date' in col or 'time' in col):
            date_col = next(col for col in df.columns if 'date' in col or 'time' in col)
            df.rename(columns={date_col: 'date'}, inplace=True)
            missing_cols.remove('date')

        if 'sku' in missing_cols and any(col for col in df.columns if 'product' in col or 'item' in col or 'sku' in col or 'part' in col):
            sku_col = next(col for col in df.columns if 'product' in col or 'item' in col or 'sku' in col or 'part' in col)
            df.rename(columns={sku_col: 'sku'}, inplace=True)
            missing_cols.remove('sku')

        if 'quantity' in missing_cols and any(col for col in df.columns if 'qty' in col or 'quant' in col or 'amount' in col or 'units' in col):
            qty_col = next(col for col in df.columns if 'qty' in col or 'quant' in col or 'amount' in col or 'units' in col)
            df.rename(columns={qty_col: 'quantity'}, inplace=True)
            missing_cols.remove('quantity')

    # If still missing required columns, raise exception
    if missing_cols:
        raise ValueError(f"Sales data is missing required columns: {', '.join(missing_cols)}. Please ensure your file has columns for date, sku, and quantity.")

    # Convert date column to datetime
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        raise ValueError(f"Could not convert date column to datetime format: {str(e)}")

    # Ensure quantity is numeric
    try:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    except Exception as e:
        raise ValueError(f"Could not convert quantity column to numeric format: {str(e)}")

    # Handle missing values
    if df['quantity'].isna().any():
        # Fill missing quantities with 0 or the median
        df['quantity'] = df['quantity'].fillna(0)

    # Remove duplicates
    df = df.drop_duplicates()

    # Sort by date and SKU
    df = df.sort_values(by=['date', 'sku'])

    # Add additional features for time analysis
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week

    return df

@st.cache_data(ttl=3600, show_spinner=False)
def process_bom_data(file):
    """
    Process and clean bill of materials (BOM) data from Excel file

    Parameters:
    -----------
    file : file object
        The uploaded Excel file containing BOM data

    Returns:
    --------
    pandas.DataFrame
        Processed and cleaned BOM data
    """
    # Read Excel file
    df = pd.read_excel(file)

    # Standardize column names (lowercase and remove spaces)
    df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]

    # Check required columns
    required_cols = ['sku', 'material_id', 'quantity_required']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        # If column names are different, try to find similar columns
        mapping = {
            'sku': ['product', 'item', 'finished_good', 'fg', 'product_id', 'part_number'],
            'material_id': ['raw_material', 'component', 'ingredient', 'part', 'material', 'rm_id', 'component_id'],
            'quantity_required': ['qty', 'amount', 'usage', 'consumption', 'required', 'qty_per']
        }

        for missing_col in missing_cols.copy():
            for potential_match in mapping.get(missing_col, []):
                matches = [col for col in df.columns if potential_match in col]
                if matches:
                    df.rename(columns={matches[0]: missing_col}, inplace=True)
                    missing_cols.remove(missing_col)
                    break

    # If still missing required columns, raise exception
    if missing_cols:
        raise ValueError(f"BOM data is missing required columns: {', '.join(missing_cols)}. Please ensure your file has columns for sku, material_id, and quantity_required.")

    # Ensure quantity is numeric
    try:
        df['quantity_required'] = pd.to_numeric(df['quantity_required'], errors='coerce')
    except Exception as e:
        raise ValueError(f"Could not convert quantity_required column to numeric format: {str(e)}")

    # Handle missing values
    if df['quantity_required'].isna().any():
        # Fill missing quantities with the median or raise an error
        if df['quantity_required'].notna().sum() > 0:
            median_qty = df['quantity_required'].median()
            df['quantity_required'] = df['quantity_required'].fillna(median_qty)
        else:
            raise ValueError("All values in quantity_required column are missing.")

    # Remove duplicates
    df = df.drop_duplicates()

    # Add material_name if missing but 'name' or 'description' exists
    if 'material_name' not in df.columns:
        name_cols = [col for col in df.columns if 'name' in col or 'description' in col]
        if name_cols:
            df['material_name'] = df[name_cols[0]]

    return df

@st.cache_data(ttl=3600, show_spinner=False)
def process_supplier_data(file):
    """
    Process and clean supplier information data from Excel file

    Parameters:
    -----------
    file : file object
        The uploaded Excel file containing supplier data

    Returns:
    --------
    pandas.DataFrame
        Processed and cleaned supplier data
    """
    # Read Excel file
    df = pd.read_excel(file)

    # Standardize column names (lowercase and remove spaces)
    df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]

    # Check required columns
    required_cols = ['material_id', 'supplier_id', 'lead_time_days']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        # If column names are different, try to find similar columns
        mapping = {
            'material_id': ['raw_material', 'component', 'part', 'material', 'rm_id', 'part_number', 'item_id'],
            'supplier_id': ['vendor', 'vendor_id', 'supplier_code', 'supplier_name'],
            'lead_time_days': ['lead_time', 'lt', 'days', 'delivery_time', 'procurement_time']
        }

        for missing_col in missing_cols.copy():
            for potential_match in mapping.get(missing_col, []):
                matches = [col for col in df.columns if potential_match in col]
                if matches:
                    df.rename(columns={matches[0]: missing_col}, inplace=True)
                    missing_cols.remove(missing_col)
                    break

    # If still missing required columns, raise exception
    if missing_cols:
        raise ValueError(f"Supplier data is missing required columns: {', '.join(missing_cols)}. Please ensure your file has columns for material_id, supplier_id, and lead_time_days.")

    # Ensure lead_time_days is numeric
    try:
        df['lead_time_days'] = pd.to_numeric(df['lead_time_days'], errors='coerce')
    except Exception as e:
        raise ValueError(f"Could not convert lead_time_days column to numeric format: {str(e)}")

    # Handle missing values in lead_time_days
    if df['lead_time_days'].isna().any():
        # Fill missing lead times with the median
        median_lt = df['lead_time_days'].median()
        df['lead_time_days'] = df['lead_time_days'].fillna(median_lt)

    # Add MOQ if missing
    if 'moq' not in df.columns:
        df['moq'] = 1  # Default MOQ of 1 unit
    else:
        # Ensure MOQ is numeric
        df['moq'] = pd.to_numeric(df['moq'], errors='coerce').fillna(1)

    # Add price_per_unit if missing
    if 'price_per_unit' not in df.columns and 'price' in df.columns:
        df['price_per_unit'] = df['price']
    elif 'price_per_unit' not in df.columns:
        df['price_per_unit'] = 0.0  # Default price

    # Remove duplicates
    df = df.drop_duplicates()

    return df