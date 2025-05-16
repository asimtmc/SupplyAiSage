#!/bin/bash
set -e

echo "Aggressive cleaning for Heroku deployment optimization..."

# Replace pyproject.toml with slim version
if [ -f "pyproject.toml.slim" ]; then
  cp pyproject.toml.slim pyproject.toml
  echo "Using slim dependencies"
fi

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

# Clean up attached_assets (keep only essential files)
if [ -d "attached_assets" ]; then
  mkdir -p /tmp/essential_assets
  # Copy only essential files if needed
  # cp attached_assets/essential_file.ext /tmp/essential_assets/
  rm -rf attached_assets/*
  # Restore essential files if needed
  # cp /tmp/essential_assets/* attached_assets/
  echo "Cleaned attached assets"
fi

# Remove test files and directories
find . -name "*_test.py" -delete
find . -name "test_*.py" -delete
find . -name "test_*.sh" -delete
find . -path "*/test/*" -delete
find . -path "*/tests/*" -delete

# Remove specific directories that may contain large files
for dir in .ipynb_checkpoints .pytest_cache .coverage docs tests test .git .github .vscode .idea; do
  find . -name "$dir" -type d -exec rm -rf {} +
done

# Remove all example and demo files
find . -name "example*" -delete
find . -name "demo*" -delete
find . -name "sample*" -delete

# Remove unused file types
find . -name "*.md" -delete
find . -name "*.rst" -delete
find . -name "*.txt" -not -name "requirements*.txt" -not -name "heroku_prod_requirements.txt" -not -name "apt.txt" -delete
find . -name "*.jpg" -delete
find . -name "*.jpeg" -delete
find . -name "*.png" -delete
find . -name "*.heic" -delete
find . -name "*.gif" -delete
find . -name "*.mp4" -delete
find . -name "*.mov" -delete
find . -name "*.pdf" -delete
find . -name "*.ipynb" -delete

# Remove specific packages not needed
rm -rf node_modules/
find . -path "*/tensorflow/examples" -type d -exec rm -rf {} +
find . -path "*/tensorflow/docs" -type d -exec rm -rf {} +
find . -path "*/torch/examples" -type d -exec rm -rf {} +
find . -path "*/torch/docs" -type d -exec rm -rf {} +

# Remove specific large files that might exist in site-packages
find . -path "*/site-packages/tensorflow/include" -type d -exec rm -rf {} +
find . -path "*/site-packages/tensorflow/examples" -type d -exec rm -rf {} +
find . -path "*/site-packages/tensorflow/lite" -type d -exec rm -rf {} +
find . -path "*/site-packages/tensorflow/contrib" -type d -exec rm -rf {} +
find . -path "*/site-packages/tensorflow/docs" -type d -exec rm -rf {} +

# Make sure bin scripts are executable
chmod +x bin/*.sh 2>/dev/null || true

echo "Aggressive cleaning complete. Slug size should be drastically reduced."