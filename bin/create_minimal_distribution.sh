#!/bin/bash
set -e

echo "Creating minimal distribution for Heroku deployment..."

# Create a clean directory for the distribution
DIST_DIR="dist"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Essential application files
cp -r app.py "$DIST_DIR/"
cp -r utils "$DIST_DIR/"
cp -r pages "$DIST_DIR/"
cp -r data "$DIST_DIR/"

# Essential Heroku files
cp Procfile "$DIST_DIR/"
cp runtime.txt "$DIST_DIR/"
cp requirements-minimal.txt "$DIST_DIR/"
cp heroku_startup.py "$DIST_DIR/"
cp apt.txt "$DIST_DIR/"
cp -r bin "$DIST_DIR/"

# Create Heroku app and deploy
cd "$DIST_DIR"
echo "Minimal distribution created in $DIST_DIR directory"
echo "To deploy to Heroku, run:"
echo "  cd $DIST_DIR"
echo "  heroku create"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'Initial deployment'"
echo "  git push heroku master"

echo "Minimal distribution creation complete."