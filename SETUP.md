# Setup Guide - Traffic Congestion Analysis System

This guide will help you set up the project with a proper Python virtual environment.

## Requirements

- **Python**: 3.8 or higher (tested on 3.11)
- **OS**: Windows, macOS, or Linux
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (NVIDIA with CUDA for faster processing)

---

## Quick Setup (Automated)

### macOS / Linux

```bash
# Run the setup script
./setup.sh
```

### Windows

```cmd
# Run the setup script
setup.bat
```

The script will:
1. Check Python version
2. Create a virtual environment (`.venv`)
3. Activate the virtual environment
4. Install all dependencies from `requirements.txt`

---

## Manual Setup

If you prefer to set up manually, follow these steps:

### Step 1: Check Python Version

```bash
python --version
```

Ensure you have Python 3.8 or higher.

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Verifying Installation

After setup, verify everything is working:

```bash
# Activate virtual environment (if not already activated)
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Check installed packages
pip list

# Test imports
python -c "import torch; import cv2; import ultralytics; print('✅ All imports successful!')"
```

---

## Running the Application

### 1. Activate Virtual Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```cmd
.venv\Scripts\activate
```

You'll know it's activated when you see `(.venv)` in your terminal prompt.

### 2. Start Web Streaming Server

```bash
python web_stream.py
```

Open browser at: http://127.0.0.1:8080

### 3. Run Vehicle Detection (in new terminal)

```bash
# Activate virtual environment in the new terminal
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Run detection
python predict_stream.py
```

---

## Deactivating Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

---

## Installing Additional Dependencies

If you need to install new packages, make sure the virtual environment is activated:

```bash
source .venv/bin/activate  # Activate first
pip install <package-name>
```

Then update `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

## Troubleshooting

### "Permission denied" when running setup.sh

```bash
chmod +x setup.sh
./setup.sh
```

### "python" command not found (Windows)

Use `py` instead:
```cmd
py --version
py -m venv .venv
```

### "pip is not recognized"

Try:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### CUDA/GPU Issues

If you have an NVIDIA GPU and want CUDA support:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Port already in use (8080)

Use a different port:
```bash
PORT=9000 python web_stream.py
```

---

## Project Structure After Setup

```
traffic_congestion/
├── .venv/                     # Virtual environment (auto-created)
├── templates/                 # HTML templates
├── deep_sort_pytorch/         # DeepSORT tracking
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script (macOS/Linux)
├── setup.bat                  # Setup script (Windows)
├── SETUP.md                   # This file
├── web_stream.py              # Web streaming server
├── predict_stream.py          # Stream detection
├── predict.py                 # Main detection
├── train.py                   # Training
├── val.py                     # Validation
├── process_results.py         # Traffic metrics
├── stream_config.yaml         # Stream config
└── ...
```

---

## Environment Variables

Optional environment variables you can set:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Web server port | `8080` |
| `FIREBASE_CRED_PATH` | Path to Firebase credentials | `firebase-key.json` |
| `FIREBASE_DB_URL` | Firebase database URL | See `predict_stream.py` |

Example:
```bash
export PORT=9000
export FIREBASE_CRED_PATH=/path/to/firebase-key.json
python web_stream.py
```

---

## Updating Dependencies

To update all packages to their latest versions:

```bash
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

---

## Uninstalling

To remove the virtual environment and start fresh:

```bash
deactivate
rm -rf .venv  # macOS/Linux
# OR
rmdir /s /q .venv  # Windows
```

Then run setup again.

---

## Need Help?

If you encounter issues:

1. Check Python version: `python --version`
2. Verify virtual environment is activated: `which python`
3. Check installed packages: `pip list`
4. Review error messages carefully

For more information, see:
- [STREAMING.md](STREAMING.md) - Streaming feature docs
- [AGENTS.md](AGENTS.md) - Project documentation
