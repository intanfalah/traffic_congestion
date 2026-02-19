#!/usr/bin/env python3
"""
Smart Traffic Management System - EFFICIENT VERSION
Single YOLO model shared across threads with proper synchronization
"""

import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000|max_delay;5000000'

import json
import threading
import time
import queue
from datetime import datetime
from collections import deque, defaultdict

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO

# Database
from database.db_manager import DatabaseManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'traffic-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

class SharedModel:
    """Thread-safe shared YOLO model"""
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        self._init_model()
    
    def _init_model(self):
        """Initialize model once"""
        print("[SharedModel] Loading YOLOv8n...")
        self.model = YOLO('yolov8n.pt')
        print("[SharedModel] Model ready")
    
    def predict(self, frame):
        """Thread-safe prediction"""
        with self.lock:
            return self.model(frame, conf=0.3)

class TrafficSystem:
    def __init__(self):
        self.cctvs = {}
        self.detectors = {}
        self.traffic_data = defaultdict(lambda: {
            'vehicle_count': 0,
            'vehicle_types': defaultdict(int),
            'congestion_level': 'UNKNOWN',
            'last_updated': None,
            'history': deque(maxlen=100)
        })
        self.shared_model = SharedModel()
        self.db = DatabaseManager()
    
    def load_cctvs_from_db(self):
        """Load existing CCTVs from database into memory"""
        db_cctvs = self.db.get_cctvs()
        loaded = 0
        for row in db_cctvs:
            cctv_id = row['id']
            if cctv_id not in self.cctvs:
                self.cctvs[cctv_id] = {
                    'id': cctv_id,
                    'name': row['name'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'stream_url': row['stream_url'],
                    'road_segment_id': row.get('road_segment_id'),
                    'active': False,
                    'status': row.get('status', 'inactive')
                }
                loaded += 1
        print(f"[System] Loaded {loaded} CCTVs from database")
        
    def add_cctv(self, cctv_id, name, lat, lng, stream_url):
        self.cctvs[cctv_id] = {
            'id': cctv_id,
            'name': name,
            'latitude': lat,
            'longitude': lng,
            'stream_url': stream_url,
            'active': False,
            'status': 'inactive'
        }
        return self.cctvs[cctv_id]
    
    def start_cctv(self, cctv_id):
        try:
            if cctv_id not in self.cctvs:
                return False, "CCTV not found"
            
            cctv = self.cctvs[cctv_id]
            print(f"[System] Starting: {cctv['name']}")
            
            detector = EfficientDetector(cctv_id, cctv['stream_url'], self)
            detector.start()
            self.detectors[cctv_id] = detector
            cctv['active'] = True
            cctv['status'] = 'starting'
            
            return True, "Starting..."
        except Exception as e:
            print(f"[System] Error: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def stop_cctv(self, cctv_id):
        if cctv_id in self.detectors:
            self.detectors[cctv_id].stop()
            del self.detectors[cctv_id]
        if cctv_id in self.cctvs:
            self.cctvs[cctv_id]['active'] = False
            self.cctvs[cctv_id]['status'] = 'inactive'
        return True
    
    def update_traffic_data(self, cctv_id, data):
        self.traffic_data[cctv_id].update(data)
        self.traffic_data[cctv_id]['last_updated'] = datetime.now().isoformat()
        try:
            socketio.emit('traffic_update', {'cctv_id': cctv_id, 'data': data})
        except:
            pass

class EfficientDetector(threading.Thread):
    """Efficient detector using shared model"""
    
    def __init__(self, cctv_id, stream_url, system):
        super().__init__(daemon=True)
        self.cctv_id = cctv_id
        self.stream_url = stream_url
        self.system = system
        self.running = False
        self.processed_frame = None
        self.frame_lock = threading.Lock()
        
        # Stats
        self.frame_count = 0
        self.process_every = 5  # Process every 5th frame
        self.vehicle_count = 0
        self.last_update = time.time()
        
    def run(self):
        self.running = True
        print(f"[Detector {self.cctv_id}] Starting...")
        
        cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"[Detector {self.cctv_id}] Failed to open stream")
            return
        
        print(f"[Detector {self.cctv_id}] Stream opened")
        self.system.cctvs[self.cctv_id]['status'] = 'active'
        
        frame_num = 0
        start_time = time.time()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame_num += 1
            
            # Only process every Nth frame
            if frame_num % self.process_every == 0:
                processed = self.process_frame(frame)
                with self.frame_lock:
                    self.processed_frame = processed
                self.frame_count += 1
            else:
                # Just draw info on original frame
                self.draw_info(frame)
                with self.frame_lock:
                    self.processed_frame = frame
            
            # Update stats every 5 seconds
            if time.time() - start_time >= 5:
                fps = self.frame_count / 5
                print(f"[Detector {self.cctv_id}] FPS: {fps:.1f}")
                self.calculate_metrics()
                self.frame_count = 0
                start_time = time.time()
        
        cap.release()
    
    def process_frame(self, frame):
        """Process with shared model"""
        try:
            results = self.system.shared_model.predict(frame)
            det = results[0].boxes
            
            count = 0
            for box in det:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                count += 1
            
            self.vehicle_count += count
            
        except Exception as e:
            print(f"[Detector {self.cctv_id}] Processing error: {e}")
        
        self.draw_info(frame)
        return frame
    
    def draw_info(self, frame):
        """Draw overlay info"""
        cv2.putText(frame, f"Vehicles: {self.vehicle_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"CCTV: {self.cctv_id}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def calculate_metrics(self):
        """Calculate and send traffic data"""
        if self.vehicle_count < 5:
            level, los = 'FREE_FLOW', 'A'
        elif self.vehicle_count < 15:
            level, los = 'MODERATE', 'C'
        elif self.vehicle_count < 30:
            level, los = 'CONGESTED', 'D'
        else:
            level, los = 'SEVERE', 'F'
        
        data = {
            'vehicle_count': self.vehicle_count,
            'congestion_level': level,
            'los': los,
            'timestamp': datetime.now().isoformat()
        }
        self.system.update_traffic_data(self.cctv_id, data)
        self.vehicle_count = 0
    
    def get_frame(self):
        with self.frame_lock:
            if self.processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.processed_frame)
                if ret:
                    return buffer.tobytes()
        return None
    
    def stop(self):
        self.running = False

# Flask routes
traffic_system = TrafficSystem()

# Thread-safe lock for accessing traffic_system
import threading
ts_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('map_dashboard.html')

@app.route('/traffic-status')
def traffic_status():
    return render_template('traffic_status.html')

@app.route('/debug-socket')
def debug_socket():
    return render_template('debug_socket.html')

@app.route('/test')
def test_page():
    """Simple test page that shows data directly"""
    cctvs = list(traffic_system.cctvs.values())
    traffic = dict(traffic_system.traffic_data)
    
    html = '''
    <!DOCTYPE html>
    <html>
    <head><title>Test</title>
    <style>
        body { font-family: Arial; padding: 20px; background: #1a1a2e; color: white; }
        .cctv { background: rgba(255,255,255,0.1); padding: 15px; margin: 10px 0; border-radius: 8px; }
        h1 { color: #e94560; }
        .active { color: #4CAF50; }
        .link { color: #2196F3; }
    </style>
    </head>
    <body>
        <h1>🚦 CCTVs ({{ count }})</h1>
        {% for cctv in cctvs %}
        <div class="cctv">
            <h3>{{ cctv.name }} <span class="active">({{ cctv.status }})</span></h3>
            <p>ID: {{ cctv.id }}</p>
            <p>Location: {{ cctv.latitude }}, {{ cctv.longitude }}</p>
            <p>Stream: <a class="link" href="/stream/{{ cctv.id }}">View Stream</a></p>
            {% if cctv.id in traffic %}
            <p>Vehicles: {{ traffic[cctv.id].vehicle_count }}</p>
            <p>Congestion: {{ traffic[cctv.id].congestion_level }}</p>
            {% endif %}
        </div>
        {% endfor %}
    </body>
    </html>
    '''
    from flask import render_template_string
    return render_template_string(html, cctvs=cctvs, traffic=traffic, count=len(cctvs))

@app.route('/api/cctvs')
def get_cctvs():
    return jsonify({'cctvs': list(traffic_system.cctvs.values())})

@app.route('/api/cctvs', methods=['POST'])
def add_cctv():
    data = request.json
    with ts_lock:
        cctv = traffic_system.add_cctv(
            data['id'], data['name'], data['latitude'], 
            data['longitude'], data['stream_url']
        )
    return jsonify({'success': True, 'cctv': cctv})

@app.route('/api/cctvs/<cctv_id>/start', methods=['POST'])
def start_cctv(cctv_id):
    success, msg = traffic_system.start_cctv(cctv_id)
    return jsonify({'success': success, 'message': msg})

@app.route('/api/cctvs/<cctv_id>/stop', methods=['POST'])
def stop_cctv(cctv_id):
    traffic_system.stop_cctv(cctv_id)
    return jsonify({'success': True})

@app.route('/api/traffic/status')
def get_traffic_status():
    """Get traffic status for all CCTVs"""
    # Convert defaultdict to regular dict for JSON serialization
    result = {}
    for cctv_id, data in traffic_system.traffic_data.items():
        result[cctv_id] = {
            'vehicle_count': data.get('vehicle_count', 0),
            'congestion_level': data.get('congestion_level', 'UNKNOWN'),
            'los': data.get('los', '-'),
            'last_updated': data.get('last_updated'),
            'fps': data.get('fps', 0)
        }
    return jsonify(result)

@app.route('/api/traffic/roads')
def get_road_segments():
    """Get road segments with traffic data"""
    roads = []
    with ts_lock:
        for cctv_id, cctv in traffic_system.cctvs.items():
            traffic = traffic_system.traffic_data.get(cctv_id, {})
            roads.append({
                'id': cctv.get('road_segment_id', cctv_id),
                'name': cctv['name'],
                'cctv_id': cctv_id,
                'congestion_level': traffic.get('congestion_level', 'UNKNOWN'),
                'los': traffic.get('los', '-'),
                'vehicle_count': traffic.get('vehicle_count', 0),
                'vehicles_per_minute': traffic.get('vehicles_per_minute', 0),
                'status': 'active' if cctv.get('active') else 'inactive'
            })
    return jsonify({'roads': roads})

@app.route('/api/cctvs/<cctv_id>/status')
def get_cctv_status(cctv_id):
    """Get individual CCTV status"""
    cctv = traffic_system.cctvs.get(cctv_id)
    if not cctv:
        return jsonify({'error': 'CCTV not found'}), 404
    
    traffic = traffic_system.traffic_data.get(cctv_id, {})
    return jsonify({
        'cctv': cctv,
        'traffic': {
            'vehicle_count': traffic.get('vehicle_count', 0),
            'congestion_level': traffic.get('congestion_level', 'UNKNOWN'),
            'los': traffic.get('los', '-'),
            'last_updated': traffic.get('last_updated')
        }
    })

def generate_stream(cctv_id):
    while True:
        if cctv_id in traffic_system.detectors:
            frame = traffic_system.detectors[cctv_id].get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/stream/<cctv_id>')
def video_stream(cctv_id):
    return Response(generate_stream(cctv_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    try:
        with ts_lock:
            cctvs_data = list(traffic_system.cctvs.values())
            traffic_data = dict(traffic_system.traffic_data)
        socketio.emit('init_data', {
            'cctvs': cctvs_data,
            'traffic': traffic_data
        })
    except Exception as e:
        print(f'[SocketIO] ERROR: {e}')
        import traceback
        traceback.print_exc()

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    print("=" * 60)
    print("🚦 Smart Traffic System - EFFICIENT")
    print("=" * 60)
    
    # Initialize database and load CCTVs
    traffic_system.db.init_database()
    traffic_system.load_cctvs_from_db()
    
    print("\n[Setup] Single shared YOLO model")
    print("       Processing every 5th frame for efficiency")
    print("\nRun: python add_real_cctvs.py")
    
    port = int(os.environ.get('PORT', 5005))
    print(f"\n[Server] http://127.0.0.1:{port}")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
