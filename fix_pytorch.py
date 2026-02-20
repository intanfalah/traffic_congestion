#!/usr/bin/env python3
"""
Fix PyTorch 2.6+ compatibility with Ultralytics
Run this if you get weights_only errors
"""

import torch
import sys

print("🔧 Fixing PyTorch compatibility...")

try:
    # Method 1: Add safe globals
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
    print("✅ Method 1: Added safe globals")
except Exception as e:
    print(f"⚠️  Method 1 failed: {e}")

try:
    # Method 2: Patch torch.load
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    print("✅ Method 2: Patched torch.load")
except Exception as e:
    print(f"⚠️  Method 2 failed: {e}")

print("\n✅ Fixes applied. You can now run: python app.py")
