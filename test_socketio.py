#!/usr/bin/env python3
"""
Test SocketIO functionality
"""

import socketio
import time
import sys

sio = socketio.Client()

@sio.event
def connect():
    print("✅ Connected to server")

@sio.event
def disconnect():
    print("❌ Disconnected from server")

@sio.on('init_data')
def on_init_data(data):
    print(f"✅ Received init_data with {len(data.get('cctvs', []))} CCTVs")
    sio.disconnect()
    sys.exit(0)

@sio.on('connect_error')
def on_connect_error(data):
    print(f"❌ Connection error: {data}")
    sys.exit(1)

print("Testing SocketIO connection to http://127.0.0.1:5005...")

try:
    sio.connect('http://127.0.0.1:5005', wait_timeout=10)
    sio.wait()
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
