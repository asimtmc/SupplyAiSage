"""
Heroku startup script - this runs before the application starts
to ensure the database is properly initialized and optimized for deployment
"""
import sys
import os
import shutil
import subprocess
from utils.db_config import init_db

def cleanup_for_deployment():
    """Remove unnecessary files to reduce slug size"""
    print("Performing final deployment optimization...")
    
    # Directories to remove if they exist
    dirs_to_remove = [
        '.git',
        '.github',
        'tests',
        'docs',
        '__pycache__',
        '.pytest_cache',
        '.ipynb_checkpoints',
    ]
    
    for directory in dirs_to_remove:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Removed {directory}")
            except Exception as e:
                print(f"Error removing {directory}: {e}")
    
    # Clean up TensorFlow extras if installed in the virtual environment
    tf_dirs = [
        'lib/python3.10/site-packages/tensorflow/include',
        'lib/python3.10/site-packages/tensorflow/examples',
        'lib/python3.10/site-packages/tensorflow/python/debug',
    ]
    
    for tf_dir in tf_dirs:
        if os.path.exists(tf_dir):
            try:
                shutil.rmtree(tf_dir)
                print(f"Removed {tf_dir}")
            except Exception as e:
                print(f"Error removing {tf_dir}: {e}")
    
    print("Deployment optimization complete")

def main():
    """Initialize the database when running on Heroku"""
    print("Initializing database for Heroku deployment...")
    
    # Check if running on Heroku
    if os.environ.get('PORT'):
        print("Running in Heroku environment")
        
        # Additional Heroku optimizations
        cleanup_for_deployment()
        
        # Set Heroku specific environment variables
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
        
        # Run database initialization
        init_db()
        print("Database initialization complete")
    else:
        print("Running in local environment")
        init_db()
        print("Database initialization complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())