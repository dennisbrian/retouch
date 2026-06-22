#!/bin/bash
# Standalone PyInstaller build script for Pro Max Retouch Studio

# Exit on first error
set -e

echo "=== 1. Installing Desktop Dependencies ==="
python3 -m pip install -r requirements-gui.txt

echo "=== 2. Compiling Standalone Desktop Application ==="
# On macOS, --windowed creates a double-clickable .app bundle
# We copy package files and the models/ directory into the application bundle
python3 -m PyInstaller --windowed \
            --name "Pro Max Retouch Studio" \
            --add-data "retouch:retouch" \
            --add-data "models:models" \
            desktop.py

echo "=== 3. Build Completed Successfully! ==="
echo "Your double-clickable app is available at: dist/Pro Max Retouch Studio.app"
echo "You can double click and run it directly without using the terminal."
