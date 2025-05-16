"""
Database initialization script to create the required tables
and structure for the application.
"""

import os
import sqlite3

def initialize_database():
    """Initialize the database and create all tables."""
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    db_path = 'data/supply_chain.db'
    
    # Create a new SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS uploaded_files (
        id TEXT PRIMARY KEY,
        filename TEXT,
        file_data BLOB,
        upload_date TIMESTAMP,
        file_type TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS forecast_results (
        id TEXT PRIMARY KEY,
        sku TEXT,
        model TEXT,
        forecast_date TIMESTAMP,
        forecast_data TEXT,
        metadata TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_parameter_cache (
        id TEXT PRIMARY KEY,
        sku TEXT,
        model_type TEXT,
        parameters TEXT,
        last_updated TIMESTAMP,
        metrics TEXT
    )
    ''')
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    
    print(f"Database initialized successfully at {db_path}")

if __name__ == "__main__":
    initialize_database()