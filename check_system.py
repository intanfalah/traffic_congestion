#!/usr/bin/env python3
"""
System Check Script
Verify all dependencies are installed correctly
"""

import sys
import subprocess

def check_python_package(package, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
        return True, "✅ Installed"
    except ImportError:
        return False, f"❌ Not installed - run: pip install {package}"

def check_system_command(cmd, name):
    """Check if a system command is available"""
    try:
        result = subprocess.run([cmd, "--version"], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0][:50]
            return True, f"✅ {version}"
        return False, f"⚠️  Command found but returned error"
    except FileNotFoundError:
        return False, f"❌ Not found - install {name}"
    except Exception as e:
        return False, f"❌ Error: {e}"

def main():
    print("=" * 70)
    print("🔍 SYSTEM CHECK - Smart Traffic Management")
    print("=" * 70)
    
    all_ok = True
    
    # Check Python version
    print("\n📌 Python Version:")
    py_version = sys.version.split()[0]
    py_major, py_minor = map(int, py_version.split('.')[:2])
    if py_major >= 3 and py_minor >= 8:
        print(f"   ✅ Python {py_version}")
    else:
        print(f"   ❌ Python {py_version} - Requires 3.8+")
        all_ok = False
    
    # Check Python packages
    print("\n📦 Python Packages:")
    packages = [
        ("flask", "flask"),
        ("flask-socketio", "flask_socketio"),
        ("ultralytics", "ultralytics"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("requests", "requests"),
        ("beautifulsoup4", "bs4"),
    ]
    
    for pkg, imp in packages:
        ok, msg = check_python_package(pkg, imp)
        print(f"   {pkg:20} {msg}")
        if not ok:
            all_ok = False
    
    # Check system commands
    print("\n🔧 System Commands:")
    commands = [
        ("ffmpeg", "FFmpeg"),
    ]
    
    for cmd, name in commands:
        ok, msg = check_system_command(cmd, name)
        print(f"   {cmd:20} {msg}")
        if not ok:
            all_ok = False
    
    # Check YOLO model
    print("\n🤖 YOLO Model:")
    try:
        from ultralytics import YOLO
        from pathlib import Path
        model_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "yolov8n.pt"
        if model_path.exists():
            print(f"   ✅ yolov8n.pt found")
        else:
            print(f"   ⚠️  yolov8n.pt not cached - will download on first run")
    except Exception as e:
        print(f"   ❌ Error checking YOLO: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Ready to run!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Terminal 1: python app.py")
        print("  2. Terminal 2: python add_real_cctvs.py")
        print("  3. Browser: http://127.0.0.1:5005")
    else:
        print("⚠️  SOME CHECKS FAILED - Install missing dependencies")
        print("=" * 70)
        print("\nFix commands:")
        print("  pip install flask flask-socketio ultralytics opencv-python")
        print("  brew install ffmpeg  # (or apt-get install ffmpeg on Linux)")
    print("=" * 70)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
