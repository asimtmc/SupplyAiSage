
import sqlite3
import os
import sys

def main():
    """Check if the database file exists and has the expected tables"""
    db_path = "data/supply_chain.db"
    
    # Check if database file exists
    if not os.path.exists(db_path):
        print(f"ERROR: Database file does not exist at {os.path.abspath(db_path)}")
        return 1
    
    print(f"Database file exists at {os.path.abspath(db_path)}")
    print(f"File size: {os.path.getsize(db_path) / 1024:.2f} KB")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("ERROR: No tables found in the database")
            return 1
        
        print(f"Found {len(tables)} tables in the database:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"  - {table[0]}: {count} records")
            
            # If it's the uploaded_files table, check for any records
            if table[0] == 'uploaded_files':
                cursor.execute("SELECT id, filename, file_type, LENGTH(file_data) FROM uploaded_files")
                files = cursor.fetchall()
                if files:
                    print("\nUploaded files:")
                    for file in files:
                        print(f"  - {file[1]} (Type: {file[2]}, Size: {file[3]} bytes)")
                else:
                    print("\nNo files have been uploaded yet.")
        
        conn.close()
        return 0
    
    except Exception as e:
        print(f"ERROR: Failed to check database: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
