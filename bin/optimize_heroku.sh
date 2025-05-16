#!/bin/bash

# This script optimizes the application for Heroku deployment
# by removing unnecessary files and dependencies

echo "Optimizing application for Heroku deployment..."

# Clean up __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove any leftover .pyc files
find . -name "*.pyc" -delete

# Remove test files
find . -name "test_*.py" -delete

# Remove development only files
rm -f *.log
rm -f *.bak

# Remove unnecessary notebooks
find . -name "*.ipynb" -delete

# Optimize the database
if [ -f "data/supply_chain.db" ]; then
    echo "Optimizing database..."
    sqlite3 data/supply_chain.db "VACUUM;"
fi

echo "Optimization complete. Application is ready for deployment."