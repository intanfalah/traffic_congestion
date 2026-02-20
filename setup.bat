@echo off
chcp 65001 >nul

REM Setup script for Traffic Congestion Analysis System (Windows)
REM This script creates a virtual environment and installs all dependencies

echo ==============================================
echo 🚦 Traffic Congestion Analysis - Setup
echo ==============================================
echo.

REM Check Python version
echo 📋 Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo    ❌ Python not found. Please install Python 3.8 or higher.
    exit /b 1
)

for /f "tokens=2" %%a in ('python --version') do set python_version=%%a
echo    Found Python %python_version%

REM Check if Python version is 3.8 or higher
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo    ❌ Python 3.8 or higher is required
    exit /b 1
)
echo    ✅ Python version is compatible (3.8+)
echo.

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 🔧 Creating virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo    ❌ Failed to create virtual environment
        exit /b 1
    )
    echo    ✅ Virtual environment created
) else (
    echo 🔧 Virtual environment already exists (.venv)
)
echo.

REM Activate virtual environment
echo 🚀 Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo    ❌ Failed to activate virtual environment
    exit /b 1
)
echo    ✅ Virtual environment activated
echo.

REM Upgrade pip
echo ⬆️  Upgrading pip...
pip install --upgrade pip -q
if errorlevel 1 (
    echo    ⚠️  Failed to upgrade pip (continuing anyway)
) else (
    echo    ✅ Pip upgraded
)
echo.

REM Install requirements
echo 📦 Installing dependencies...
echo    This may take a few minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo    ❌ Failed to install some dependencies
    exit /b 1
)
echo.
echo    ✅ All dependencies installed successfully

echo.
echo ==============================================
echo ✅ Setup Complete!
echo ==============================================
echo.
echo To activate the virtual environment, run:
echo.
echo    .venv\Scripts\activate
echo.
echo Then run the application:
echo.
echo    # Start web streaming server
echo    python web_stream.py
echo.
echo    # In another terminal, run detection
echo    python predict_stream.py
echo.

pause
