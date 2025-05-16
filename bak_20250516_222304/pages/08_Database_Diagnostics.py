
import streamlit as st
import pandas as pd
from utils.db_diagnostics import check_database, get_table_data
import sqlite3
import os
import io

st.set_page_config(
    page_title="Database Diagnostics",
    page_icon="🔍",
    layout="wide"
)

st.title("Database Diagnostics")
st.markdown("This page helps diagnose database issues by showing the structure and content of the database.")

# Run the database check
db_info = check_database()

# Display basic database information
st.header("Database Information")
col1, col2 = st.columns(2)

with col1:
    st.metric("Database Exists", "Yes" if db_info["db_exists"] else "No")
    
with col2:
    st.metric("Database Size", f"{db_info['db_size'] / 1024:.2f} KB")

# Display tables
if db_info["db_exists"]:
    st.header("Database Tables")
    
    for table in db_info["tables"]:
        with st.expander(f"{table} ({db_info['record_counts'][table]} records)"):
            st.write(f"**Record count:** {db_info['record_counts'][table]}")
            
            # Display table structure if it's the uploaded_files table
            if table == "uploaded_files" and "uploaded_files_structure" in db_info:
                st.subheader("Table Structure")
                structure_df = pd.DataFrame(db_info["uploaded_files_structure"], 
                                            columns=["cid", "name", "type", "notnull", "default_value", "pk"])
                st.dataframe(structure_df)
            
            # Get and display sample data from the table
            st.subheader("Sample Data")
            table_data = get_table_data(table)
            if table_data is not None and not table_data.empty:
                st.dataframe(table_data)
            else:
                st.info("No data found in this table.")

    # Add utility to insert test data
    st.header("Database Utilities")
    
    with st.expander("Insert Test Data"):
        st.write("Insert test data to verify database functionality.")
        
        file_type = st.selectbox(
            "Select file type", 
            ["sales_data", "bom_data", "supplier_data"]
        )
        
        if st.button("Insert Test Data"):
            try:
                # Create a simple test DataFrame
                if file_type == "sales_data":
                    df = pd.DataFrame({
                        "date": pd.date_range(start="2023-01-01", periods=12, freq="MS"),
                        "sku": ["TEST-SKU"] * 12,
                        "quantity": [100, 120, 90, 110, 130, 140, 120, 110, 100, 90, 110, 120]
                    })
                elif file_type == "bom_data":
                    df = pd.DataFrame({
                        "sku": ["TEST-SKU"] * 3,
                        "material_id": ["MAT001", "MAT002", "MAT003"],
                        "quantity_required": [2, 1, 5]
                    })
                else: # supplier_data
                    df = pd.DataFrame({
                        "material_id": ["MAT001", "MAT002", "MAT003"],
                        "supplier_id": ["SUP001", "SUP002", "SUP003"],
                        "lead_time_days": [5, 10, 7]
                    })
                
                # Save to Excel
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)
                
                # Connect to database
                conn = sqlite3.connect("data/supply_chain.db")
                cursor = conn.cursor()
                
                # Check if file type already exists
                cursor.execute("SELECT id FROM uploaded_files WHERE file_type = ?", (file_type,))
                existing = cursor.fetchone()
                
                if existing:
                    # Delete existing file
                    cursor.execute("DELETE FROM uploaded_files WHERE file_type = ?", (file_type,))
                
                # Insert new test file
                import uuid
                import datetime
                
                file_id = str(uuid.uuid4())
                filename = f"test_{file_type}.xlsx"
                file_data = excel_buffer.getvalue()
                now = datetime.datetime.now()
                
                cursor.execute(
                    "INSERT INTO uploaded_files (id, filename, file_type, description, created_at, updated_at, file_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (file_id, filename, file_type, "Test data", now, now, file_data)
                )
                
                conn.commit()
                conn.close()
                
                st.success(f"Test data for {file_type} inserted successfully!")
            except Exception as e:
                st.error(f"Error inserting test data: {str(e)}")
else:
    st.error("Database file does not exist.")
