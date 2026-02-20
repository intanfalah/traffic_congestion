#!/usr/bin/env python3
"""
Single CCTV Test - Flask App
Test with just one CCTV to verify the full pipeline
"""

import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000|max_delay;5000000'

import threading
import time
from datetime import datetime
from collections import deque

from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Single CCTV config
CCTV_CONFIG = {
    'id': 'cctv_001',
    'name': 'Indraprasta Imam Bonjol',
    'stream_url': 'https://livepantau.semarangkota.go.id/3cc2431b-3ee5-4c91-8330-251c021cd510/video1_stream.m3u8',
    'latitude': -6.9785713,
    'longitude': 110.411635
}

class SingleCCTVSystem:
    def __init__(self):
        self.model = None
        self.cap = None
        self.running = False
        self.processed_frame = None
        self.frame_lock = threading.Lock()
        
        # Stats
        self.frame_count = 0
        self.vehicle_count = 0
        self.fps = 0
        self.traffic_data = {
            'vehicle_count': 0,
            'congestion_level': 'UNKNOWN',
            'los': '-',
            'last_updated': None
        }
        
    def start(self):
        """Start detection"""
        if self.running:
            return
        
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
    
    def _run(self):
        """Main detection loop"""
        self.running = True
        print("[SingleCCTV] Starting...")
        
        # Load model
        print("[SingleCCTV] Loading YOLO...")
        self.model = YOLO('yolov8n.pt')
        print("[SingleCCTV] Model loaded")
        
        # Open stream
        print(f"[SingleCCTV] Opening stream...")
        self.cap = cv2.VideoCapture(CCTV_CONFIG['stream_url'], cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            print("[SingleCCTV] Failed to open stream")
            self.running = False
            return
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[SingleCCTV] Stream: {width}x{height}")
        
        # Process loop
        frame_num = 0
        start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_num += 1
            
            # Process every 3rd frame
            if frame_num % 3 == 0:
                try:
                    results = self.model(frame, conf=0.3)
                    det = results[0].boxes
                    
                    count = 0
                    for box in det:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        count += 1
                    
                    self.vehicle_count += count
                    
                except Exception as e:
                    print(f"[SingleCCTV] Processing error: {e}")
            
            # Draw info
            elapsed = time.time() - start_time
            if elapsed > 0:
                self.fps = frame_num / elapsed
            
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Vehicles: {self.vehicle_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Store frame
            with self.frame_lock:
                self.processed_frame = frame.copy()
            
            self.frame_count += 1
            
            # Update traffic data every 10 seconds
            if elapsed >= 10:
                self._update_traffic_data()
                frame_num = 0
                start_time = time.time()
        
        self.cap.release()
        print("[SingleCCTV] Stopped")
    
    def _update_traffic_data(self):
        """Update traffic metrics"""
        vcount = self.vehicle_count
        
        if vcount < 5:
            level, los = 'FREE_FLOW', 'A'
        elif vcount < 15:
            level, los = 'MODERATE', 'C'
        elif vcount < 30:
            level, los = 'CONGESTED', 'D'
        else:
            level, los = 'SEVERE', 'F'
        
        self.traffic_data = {
            'vehicle_count': vcount,
            'congestion_level': level,
            'los': los,
            'fps': round(self.fps, 1),
            'last_updated': datetime.now().isoformat()
        }
        
        print(f"[SingleCCTV] Traffic update: {vcount} vehicles, {level}")
        
        try:
            socketio.emit('traffic_update', {
                'cctv_id': CCTV_CONFIG['id'],
                'data': self.traffic_data
            })
        except:
            pass
        
        self.vehicle_count = 0
    
    def get_frame(self):
        """Get latest frame"""
        with self.frame_lock:
            if self.processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.processed_frame)
                if ret:
                    return buffer.tobytes()
        return None
    
    def stop(self):
        self.running = False

# Initialize
system = SingleCCTVSystem()

# Routes
@app.route('/')
def index():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Single CCTV Test</title>
        <style>
            body {{ font-family: Arial; background: #1a1a2e; color: white; padding: 20px; }}
            h1 {{ color: #e94560; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .stream {{ background: black; border: 2px solid #e94560; border-radius: 10px; overflow: hidden; }}
            .stream img {{ width: 100%; display: block; }}
            .stats {{ background: rgba(255,255,255,0.1); padding: 15px; margin-top: 20px; border-radius: 8px; }}
            .stat {{ display: inline-block; margin-right: 30px; }}
            .stat-value {{ font-size: 24px; color: #e94560; font-weight: bold; }}
            .stat-label {{ color: #8892b0; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Single CCTV Test</h1>
            <h2>{CCTV_CONFIG['name']}</h2>
            <p>Location: {CCTV_CONFIG['latitude']}, {CCTV_CONFIG['longitude']}</p>
            
            <div class="stream">
                <img src="/stream" />
            </div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-value" id="fps">--</div>
                    <div class="stat-label">FPS</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="vehicles">--</div>
                    <div class="stat-label">Vehicles</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="congestion">--</div>
                    <div class="stat-label">Congestion</div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
        <script>
            const socket = io();
            socket.on('traffic_update', (data) => {{
                document.getElementById('fps').textContent = data.data.fps || '--';
                document.getElementById('vehicles').textContent = data.data.vehicle_count || '--';
                document.getElementById('congestion').textContent = data.data.congestion_level || '--';
            }});
        </script>
    </body>
    </html>
    """

@app.route('/api/status')
def get_status():
    return jsonify({
        'cctv': CCTV_CONFIG,
        'traffic': system.traffic_data,
        'running': system.running,
        'fps': system.fps,
        'frames': system.frame_count
    })

def generate_stream():
    while True:
        frame = system.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/stream')
def video_stream():
    return Response(generate_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('init', {'cctv': CCTV_CONFIG})

if __name__ == '__main__':
    print("=" * 60)
    print("🚦 Single CCTV Test")
    print("=" * 60)
    print(f"CCTV: {CCTV_CONFIG['name']}")
    print(f"URL: {CCTV_CONFIG['stream_url'][:50]}...")
    print("\nStarting detection...")
    
    system.start()
    
    print("\n[Server] http://127.0.0.1:5005")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=5005, debug=False)
