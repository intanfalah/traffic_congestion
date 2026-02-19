#!/bin/bash

# Setup script for Traffic Congestion Analysis System
# This script creates a virtual environment and installs all dependencies

echo "=============================================="
echo "🚦 Traffic Congestion Analysis - Setup"
echo "=============================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Check if Python version is 3.8 or higher
required_version="3.8"
if python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "   ✅ Python version is compatible (3.8+)"
else
    echo "   ❌ Python 3.8 or higher is required"
    exit 1
fi

echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment (.venv)..."
    python -m venv .venv
    if [ $? -eq 0 ]; then
        echo "   ✅ Virtual environment created"
    else
        echo "   ❌ Failed to create virtual environment"
        exit 1
    fi
else
    echo "🔧 Virtual environment already exists (.venv)"
fi

echo ""

# Activate virtual environment
echo "🚀 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source .venv/Scripts/activate
else
    # macOS/Linux
    source .venv/bin/activate
fi

if [ $? -eq 0 ]; then
    echo "   ✅ Virtual environment activated"
else
    echo "   ❌ Failed to activate virtual environment"
    exit 1
fi

echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q
if [ $? -eq 0 ]; then
    echo "   ✅ Pip upgraded"
else
    echo "   ⚠️  Failed to upgrade pip (continuing anyway)"
fi

echo ""

# Install requirements
echo "📦 Installing dependencies..."
echo "   This may take a few minutes..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "   ✅ All dependencies installed successfully"
else
    echo ""
    echo "   ❌ Failed to install some dependencies"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the virtual environment, run:"
echo ""
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "    .venv\\Scripts\\activate"
else
    echo "    source .venv/bin/activate"
fi
echo ""
echo "Then run the application:"
echo ""
echo "    # Start web streaming server"
echo "    python web_stream.py"
echo ""
echo "    # In another terminal, run detection"
echo "    python predict_stream.py"
echo ""
