#!/bin/bash

# Script to clean up the application for Heroku deployment
# This removes unused pages and modules to reduce slug size

echo "Starting cleanup for Heroku deployment..."

# Create backup directory
BACKUP_DIR="bak_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Keep only the active pages
echo "Backing up pages..."
mkdir -p "$BACKUP_DIR/pages"

# Copy all pages to backup first
cp -r pages/* "$BACKUP_DIR/pages/"

# Keep only essential pages
KEEP_PAGES=(
    "14_V2_Demand_Forecasting_Croston.py"
    "config"
)

# Remove pages that are not in the keep list
for file in pages/*.py; do
    filename=$(basename "$file")
    if [[ ! " ${KEEP_PAGES[@]} " =~ " ${filename} " ]]; then
        echo "Removing unused page: $file"
        rm "$file"
    fi
done

# Clean up utility modules
echo "Backing up utils..."
mkdir -p "$BACKUP_DIR/utils"
cp -r utils/* "$BACKUP_DIR/utils/"

# Keep only essential utility modules
KEEP_UTILS=(
    "__init__.py"
    "croston.py"
    "database.py"
    "data_processor.py"
    "parameter_optimizer.py"
    "seasonal_detector.py"
    "session_data.py"
    "visualization.py"
)

# Remove utilities that are not in the keep list
for file in utils/*.py; do
    filename=$(basename "$file")
    if [[ ! " ${KEEP_UTILS[@]} " =~ " ${filename} " ]]; then
        echo "Removing unused utility: $file"
        rm "$file"
    fi
done

# Clean up cache files and directories
echo "Cleaning cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete
find . -name "*.log" -delete

echo "Cleanup complete."
echo "Backup saved to: $BACKUP_DIR"
echo ""
echo "Remaining files:"
find . -name "*.py" | sort