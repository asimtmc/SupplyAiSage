#!/bin/bash
set -e

echo "Cleaning up for Heroku deployment optimization..."

# Remove any cached files and artifacts
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.pyd" -delete

# Clean up models_cache directory if it exists
if [ -d "models_cache" ]; then
  rm -rf models_cache/*
  echo "Cleaned models cache"
fi

# Remove specific directories that may contain large files
for dir in .ipynb_checkpoints .pytest_cache .coverage; do
  find . -name "$dir" -type d -exec rm -rf {} +
done

echo "Cleaning complete. Slug size should be reduced."