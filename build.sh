#!/usr/bin/env bash
# Exit immediately on error
set -e

echo "=== [1/3] Updating pip and essential build tools ==="
python -m pip install --upgrade pip setuptools wheel

echo "=== [2/3] Installing Linux system dependencies for OpenCV and MediaPipe ==="
# On Render native Python environment or Ubuntu container
if command -v apt-get &> /dev/null; then
    apt-get update -y && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev || true
fi

echo "=== [3/3] Installing Python dependencies from requirements.txt ==="
# Prefer opencv-python-headless if needed in headless Linux cloud
pip install opencv-python-headless>=4.9
pip install -r requirements.txt

echo "=== Build finished successfully ==="
