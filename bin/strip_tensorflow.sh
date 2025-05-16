#!/bin/bash
set -e

echo "Stripping TensorFlow to minimum required components..."

# Find the TensorFlow installation directory
TF_DIR=$(pip show tensorflow-cpu | grep Location | awk '{print $2}')/tensorflow

if [ -d "$TF_DIR" ]; then
  echo "Found TensorFlow at $TF_DIR"
  
  # Directories to keep
  KEEP_DIRS=(
    "python/keras"
    "python/saved_model"
    "python/feature_column"
    "python/layers"
    "python/ops"
    "__pycache__"
  )

  # Create a temporary directory for the keepers
  mkdir -p /tmp/tf_keepers
  
  # Copy the essential directories
  for dir in "${KEEP_DIRS[@]}"; do
    if [ -d "$TF_DIR/$dir" ]; then
      mkdir -p "/tmp/tf_keepers/$(dirname $dir)"
      cp -r "$TF_DIR/$dir" "/tmp/tf_keepers/$(dirname $dir)/"
      echo "Preserved $dir"
    fi
  done
  
  # Keep essential Python files
  find "$TF_DIR/python" -name "*.py" -size -50k | while read file; do
    rel_path=${file#$TF_DIR/}
    mkdir -p "/tmp/tf_keepers/$(dirname $rel_path)"
    cp "$file" "/tmp/tf_keepers/$(dirname $rel_path)/"
  done
  
  # Keep essential data files
  find "$TF_DIR" -name "*.pb" -size -1M | while read file; do
    rel_path=${file#$TF_DIR/}
    mkdir -p "/tmp/tf_keepers/$(dirname $rel_path)"
    cp "$file" "/tmp/tf_keepers/$(dirname $rel_path)/"
  done
  
  # Clear TensorFlow directory
  rm -rf "$TF_DIR"/*
  
  # Restore the essential files
  cp -r /tmp/tf_keepers/* "$TF_DIR/"
  
  # Clean up
  rm -rf /tmp/tf_keepers
  
  echo "TensorFlow has been stripped to minimum required components."
else
  echo "TensorFlow directory not found!"
fi

echo "Stripping TensorFlow complete."