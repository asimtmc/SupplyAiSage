"""
Heroku startup script - this runs before the application starts
to ensure the database is properly initialized
"""
import sys
import os
from utils.db_config import init_db

def main():
    """Initialize the database when running on Heroku"""
    print("Initializing database for Heroku deployment...")
    
    # Check if running on Heroku
    if os.environ.get('PORT'):
        print("Running in Heroku environment")
    else:
        print("Running in local environment")
    
    # Initialize the database
    init_db()
    print("Database initialization complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())