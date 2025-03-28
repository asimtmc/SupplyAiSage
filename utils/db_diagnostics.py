
import sqlite3
import os
import pandas as pd

def check_database():
    """Check the database structure and content."""
    results = {}
    
    # Check if database file exists
    db_path = "data/supply_chain.db"
    results["db_exists"] = os.path.exists(db_path)
    results["db_size"] = os.path.getsize(db_path) if results["db_exists"] else 0
    
    if not results["db_exists"]:
        return results
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Get list of tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    results["tables"] = [table[0] for table in tables]
    
    # Get record counts for each table
    record_counts = {}
    for table in results["tables"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        record_counts[table] = count
    
    results["record_counts"] = record_counts
    
    # Check uploaded_files table structure and content
    if 'uploaded_files' in results["tables"]:
        cursor.execute("PRAGMA table_info(uploaded_files)")
        results["uploaded_files_structure"] = cursor.fetchall()
        
        # Get sample records from uploaded_files
        cursor.execute("SELECT id, filename, file_type, created_at FROM uploaded_files LIMIT 5")
        results["uploaded_files_sample"] = cursor.fetchall()
    
    conn.close()
    return results

def get_table_data(table_name, limit=10):
    """Get data from a specific table."""
    db_path = "data/supply_chain.db"
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error getting data from {table_name}: {str(e)}")
        return None
