#!/bin/bash

# Startup script for LBF Human Interaction Web Application

echo "=============================================="
echo "LBF Human Interaction Web Application"
echo "=============================================="
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "AHT" ]]; then
    echo "⚠️  Warning: AHT conda environment not activated"
    echo "Run: conda activate /scratch/cluster/jyliu/conda_envs/AHT"
    echo ""
    read -p "Would you like to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Flask not installed"
    echo "Installing Flask dependencies..."
    pip install -r requirements.txt
fi

# Set JAX to use CPU (to avoid GPU conflicts)
export JAX_PLATFORM_NAME=cpu

# Set PYTHONPATH
export PYTHONPATH=/scratch/cluster/jyliu/Documents/jax-aht:$PYTHONPATH

# Create collected_data directory if it doesn't exist
mkdir -p collected_data

echo ""
echo "✅ Environment ready!"
echo ""
echo "Starting Flask server..."
echo "Access the application at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="
echo ""

# Start the Flask application
python app.py
