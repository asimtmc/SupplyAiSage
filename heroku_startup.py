"""
Heroku startup script - this runs before the application starts
to ensure the database is properly initialized and optimized for deployment
"""

import os
import sqlite3
import shutil
import glob

def initialize_db():
    """Initialize the database if it doesn't exist"""
    print("Checking database...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Connect to the database (creates it if it doesn't exist)
    db_path = 'data/supply_chain.db'
    conn = sqlite3.connect(db_path)
    
    # Create tables
    conn.execute('''
    CREATE TABLE IF NOT EXISTS uploaded_files (
        id TEXT PRIMARY KEY,
        filename TEXT,
        file_data BLOB,
        upload_date TIMESTAMP,
        file_type TEXT
    )
    ''')
    
    conn.execute('''
    CREATE TABLE IF NOT EXISTS forecast_results (
        id TEXT PRIMARY KEY,
        sku TEXT,
        model TEXT,
        forecast_date TIMESTAMP,
        forecast_data TEXT,
        metadata TEXT
    )
    ''')
    
    conn.execute('''
    CREATE TABLE IF NOT EXISTS model_parameter_cache (
        id TEXT PRIMARY KEY,
        sku TEXT,
        model_type TEXT,
        parameters TEXT,
        last_updated TIMESTAMP,
        metrics TEXT
    )
    ''')
    
    # Optimize the database
    conn.execute("VACUUM")
    conn.commit()
    conn.close()
    
    print(f"Database initialized at {db_path}")

def cleanup_for_deployment():
    """Remove unnecessary files to reduce slug size"""
    # Remove cache files
    cache_patterns = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".DS_Store"
    ]
    
    for pattern in cache_patterns:
        for path in glob.glob(f"**/{pattern}", recursive=True):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    
    print("Cleaned up unnecessary files")

def main():
    """Initialize the database when running on Heroku"""
    print("Running Heroku startup script...")
    
    # Initialize the database
    initialize_db()
    
    # Clean up unnecessary files
    cleanup_for_deployment()
    
    print("Heroku startup complete.")

if __name__ == "__main__":
    main()