#!/usr/bin/env python3
"""
App Launcher with PyTorch 2.6+ Compatibility Fix
"""

import sys
import warnings

# Suppress the weights_only warning and patch torch.load
import torch
original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    """Wrapper for torch.load that sets weights_only=False"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Replace torch.load
torch.load = safe_torch_load

print("🔧 Applied PyTorch 2.6+ compatibility patch")
print("   torch.load wrapped with weights_only=False")
print()

# Now import and run the app
from app import app, socketio, traffic_system, init_demo_data
import os

if __name__ == '__main__':
    # Initialize
    print("=" * 60)
    print("🚦 Smart Traffic Management System")
    print("=" * 60)
    
    # Initialize model
    traffic_system.init_model()
    
    # Initialize database
    traffic_system.db.init_database()
    
    # Load existing CCTVs from database
    traffic_system.load_cctvs_from_db()
    
    # Load demo data only if explicitly enabled
    if os.environ.get('USE_DEMO_DATA', 'false').lower() == 'true':
        print("\n[Setup] Loading demo CCTV data...")
        init_demo_data()
    else:
        print("\n[Setup] Demo data disabled.")
        print("       Add CCTVs using: python add_real_cctvs.py")
        print("       Or run: python add_cctv_interactive.py")
    
    print("\n[Server] Starting on http://127.0.0.1:5000")
    print("=" * 60)
    
    # Run server
    port = int(os.environ.get('PORT', 5005))
    print(f'\n[Server] Starting on http://127.0.0.1:{port}')
    print("=" * 60)
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
