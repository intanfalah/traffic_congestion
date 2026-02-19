#!/usr/bin/env python3
"""
Smart Traffic Management System - OPTIMIZED VERSION
Low-latency real-time processing with minimal buffering
"""

import os
# Set FFmpeg options BEFORE importing cv2
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000|max_delay;5000000'

import json
import threading
import time
import queue
from datetime import datetime, timedelta
from collections import deque, defaultdict
from pathlib import Path

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO

# DeepSORT
import torch

# Fix for PyTorch 2.6+ weights_only issue with Ultralytics
try:
    import torch.serialization
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass  # Older PyTorch versions don't need this

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# Database
from database.db_manager import DatabaseManager
from database.models import CCTV, RoadSegment, TrafficData

app = Flask(__name__)
app.config['SECRET_KEY'] = 'traffic-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
class TrafficSystem:
    def __init__(self):
        self.cctvs = {}  # CCTV configurations
        self.streams = {}  # Active stream handlers
        self.detectors = {}  # Detection workers
        self.traffic_data = defaultdict(lambda: {
            'vehicle_count': 0,
            'vehicle_types': defaultdict(int),
            'speed_estimate': 0,
            'congestion_level': 'UNKNOWN',
            'last_updated': None,
            'history': deque(maxlen=100)  # Keep last 100 data points
        })
        self.db = DatabaseManager()
        self.model = None
        self.running = False
    
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
        
    def init_model(self):
        """Initialize YOLO model"""
        model_path = os.environ.get('YOLO_MODEL', 'yolov8n.pt')
        print(f"[System] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print("[System] Model loaded successfully")
        
    def add_cctv(self, cctv_id, name, lat, lng, stream_url, road_segment_id=None):
        """Add a new CCTV to the system"""
        self.cctvs[cctv_id] = {
            'id': cctv_id,
            'name': name,
            'latitude': lat,
            'longitude': lng,
            'stream_url': stream_url,
            'road_segment_id': road_segment_id,
            'active': False,
            'status': 'inactive'
        }
        # Save to database
        self.db.add_cctv(cctv_id, name, lat, lng, stream_url, road_segment_id)
        return self.cctvs[cctv_id]
    
    def start_cctv(self, cctv_id):
        """Start detection on a CCTV"""
        try:
            if cctv_id not in self.cctvs:
                return False, "CCTV not found"
            
            if cctv_id in self.detectors and self.detectors[cctv_id].running:
                return True, "Already running"
            
            cctv = self.cctvs[cctv_id]
            print(f"[System] Starting CCTV: {cctv['name']}")
            
            detector = OptimizedDetector(cctv_id, cctv['stream_url'], self)
            detector.start()
            self.detectors[cctv_id] = detector
            cctv['active'] = True
            cctv['status'] = 'starting'  # Will be 'active' once stream opens
            
            return True, "Starting..."
        except Exception as e:
            print(f"[System] Error starting CCTV {cctv_id}: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def stop_cctv(self, cctv_id):
        """Stop detection on a CCTV"""
        if cctv_id in self.detectors:
            self.detectors[cctv_id].stop()
            del self.detectors[cctv_id]
        
        if cctv_id in self.cctvs:
            self.cctvs[cctv_id]['active'] = False
            self.cctvs[cctv_id]['status'] = 'inactive'
        
        return True
    
    def get_traffic_status(self, cctv_id=None):
        """Get traffic status for all or specific CCTV"""
        if cctv_id:
            return self.traffic_data.get(cctv_id)
        return dict(self.traffic_data)
    
    def update_traffic_data(self, cctv_id, data):
        """Update traffic data and emit to clients"""
        self.traffic_data[cctv_id].update(data)
        self.traffic_data[cctv_id]['last_updated'] = datetime.now().isoformat()
        self.traffic_data[cctv_id]['history'].append({
            'timestamp': datetime.now().isoformat(),
            **data
        })
        
        # Save to database
        self.db.add_traffic_data(cctv_id, data)
        
        # Emit to connected clients (with error handling)
        try:
            socketio.emit('traffic_update', {
                'cctv_id': cctv_id,
                'data': data
            })
        except Exception as e:
            print(f'[Socket] Emit error: {e}')


class OptimizedDetector(threading.Thread):
    """Optimized detector with separate capture and processing threads"""
    
    def __init__(self, cctv_id, stream_url, system):
        super().__init__(daemon=True)
        self.cctv_id = cctv_id
        self.stream_url = stream_url
        self.system = system
        self.running = False
        
        # Use a queue with maxsize=1 to always get latest frame
        # This prevents buffer bloat and ensures low latency
        self.frame_queue = queue.Queue(maxsize=1)
        self.processed_frame = None
        self.latest_frame_lock = threading.Lock()
        
        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_process_time = time.time()
        self.processing_fps = 0
        
        # Tracking state (simplified - no DeepSORT for speed)
        self.vehicle_count = {'total': 0, 'current': 0}
        self.vehicle_types = defaultdict(int)
        
    def capture_frames(self):
        """Capture thread - continuously read frames and keep only latest"""
        print(f"[Capture {self.cctv_id}] Starting...")
        
        cap = None
        retry_delay = 5
        
        while self.running:
            # 1. Connect if needed
            if cap is None or not cap.isOpened():
                try:
                    if cap: cap.release()
                    cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
                    
                    if not cap.isOpened():
                        print(f"[Capture {self.cctv_id}] Failed to open stream, retrying in {retry_delay}s...")
                        self.system.cctvs[self.cctv_id]['status'] = 'error'
                        time.sleep(retry_delay)
                        continue
                        
                    print(f"[Capture {self.cctv_id}] Stream opened")
                    self.system.cctvs[self.cctv_id]['status'] = 'active'
                except Exception as e:
                    print(f"[Capture {self.cctv_id}] Connection error: {e}")
                    self.system.cctvs[self.cctv_id]['status'] = 'error'
                    time.sleep(retry_delay)
                    continue
            
            # 2. Read frame
            ret, frame = cap.read()
            if not ret:
                print(f"[Capture {self.cctv_id}] Stream error/EOF, reconnecting...")
                cap.release()
                cap = None
                self.system.cctvs[self.cctv_id]['status'] = 'error'
                time.sleep(1)
                continue
            
            # Try to put frame in queue
            # If queue is full (previous frame not processed), drop it
            try:
                self.frame_queue.put_nowait(frame)
                self.frame_count += 1
            except queue.Full:
                # Drop frame - processing can't keep up
                self.dropped_frames += 1
                pass
        
        cap.release()
        print(f"[Capture {self.cctv_id}] Stopped. Frames: {self.frame_count}, Dropped: {self.dropped_frames}")
    
    def run(self):
        """Main processing thread"""
        self.running = True
        print(f"[Detector {self.cctv_id}] Starting optimized processing...")
        
        # Load YOLO model
        try:
            model = YOLO('yolov8n.pt')
            print(f"[Detector {self.cctv_id}] Model loaded")
        except Exception as e:
            print(f"[Detector {self.cctv_id}] Failed to load model: {e}")
            return
        
        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        capture_thread.start()
        
        self.system.cctvs[self.cctv_id]['status'] = 'active'
        
        frames_processed = 0
        start_time = time.time()
        
        while self.running:
            try:
                # Get latest frame (with timeout to allow checking self.running)
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Process frame
            processed = self.process_frame(frame, model)
            
            # Store processed frame for streaming
            with self.latest_frame_lock:
                self.processed_frame = processed
            
            frames_processed += 1
            
            # Calculate processing FPS
            elapsed = time.time() - start_time
            if elapsed >= 5.0:  # Every 5 seconds
                self.processing_fps = frames_processed / elapsed
                print(f"[Detector {self.cctv_id}] Processing FPS: {self.processing_fps:.1f}, "
                      f"Dropped: {self.dropped_frames}")
                frames_processed = 0
                start_time = time.time()
            
            # Calculate traffic metrics every 10 seconds
            current_time = time.time()
            if current_time - self.last_process_time >= 10:
                self.calculate_traffic_metrics()
                self.last_process_time = current_time
        
        print(f"[Detector {self.cctv_id}] Stopped")
    
    def process_frame(self, frame, model):
        """Process a single frame - optimized for speed"""
        # Run YOLO detection
        results = model(frame, conf=0.3)
        det = results[0].boxes
        
        height, width = frame.shape[:2]
        line_y = height // 2
        
        # Count vehicles
        current_count = 0
        for box in det:
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            obj_name = results[0].names.get(cls, 'unknown')
            
            # Draw box
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, obj_name, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            current_count += 1
            self.vehicle_types[obj_name] += 1
        
        self.vehicle_count['current'] = current_count
        self.vehicle_count['total'] += current_count
        
        # Draw counting line
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 2)
        
        # Draw info
        cv2.putText(frame, f"Count: {current_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.processing_fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def calculate_traffic_metrics(self):
        """Calculate traffic congestion metrics"""
        # Simple metric: current vehicle count
        current = self.vehicle_count['current']
        
        # Estimate congestion level
        if current < 5:
            congestion = 'FREE_FLOW'
            level = 'A'
        elif current < 15:
            congestion = 'MODERATE'
            level = 'C'
        elif current < 30:
            congestion = 'CONGESTED'
            level = 'D'
        else:
            congestion = 'SEVERE'
            level = 'F'
        
        data = {
            'vehicle_count': self.vehicle_count['total'],
            'current_vehicles': current,
            'vehicle_types': dict(self.vehicle_types),
            'congestion_level': congestion,
            'los': level,
            'processing_fps': round(self.processing_fps, 1),
            'dropped_frames': self.dropped_frames,
            'timestamp': datetime.now().isoformat()
        }
        
        self.system.update_traffic_data(self.cctv_id, data)
        
        # Reset counters
        self.vehicle_count['total'] = 0
        self.vehicle_types.clear()
    
    def get_frame(self):
        """Get latest processed frame"""
        with self.latest_frame_lock:
            if self.processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.processed_frame)
                if ret:
                    return buffer.tobytes()
        return None
    
    def stop(self):
        """Stop the detector"""
        self.running = False


# Initialize system
traffic_system = TrafficSystem()


@app.route('/')
def index():
    """Main page - Map with CCTV locations"""
    return render_template('map_dashboard.html')


@app.route('/traffic-status')
def traffic_status():
    """Traffic status page with road network"""
    return render_template('traffic_status.html')


@app.route('/api/cctvs')
def get_cctvs():
    """Get all CCTV locations"""
    return jsonify({
        'cctvs': list(traffic_system.cctvs.values())
    })


@app.route('/api/cctvs', methods=['POST'])
def add_cctv():
    """Add a new CCTV"""
    data = request.json
    cctv = traffic_system.add_cctv(
        cctv_id=data['id'],
        name=data['name'],
        lat=data['latitude'],
        lng=data['longitude'],
        stream_url=data['stream_url'],
        road_segment_id=data.get('road_segment_id')
    )
    return jsonify({'success': True, 'cctv': cctv})


@app.route('/api/cctvs/<cctv_id>/start', methods=['POST'])
def start_cctv(cctv_id):
    """Start detection on a CCTV"""
    success, message = traffic_system.start_cctv(cctv_id)
    return jsonify({'success': success, 'message': message})


@app.route('/api/cctvs/<cctv_id>/stop', methods=['POST'])
def stop_cctv(cctv_id):
    """Stop detection on a CCTV"""
    traffic_system.stop_cctv(cctv_id)
    return jsonify({'success': True})


@app.route('/api/cctvs/<cctv_id>/status')
def get_cctv_status(cctv_id):
    """Get CCTV status and traffic data"""
    cctv = traffic_system.cctvs.get(cctv_id)
    traffic = traffic_system.get_traffic_status(cctv_id)
    return jsonify({
        'cctv': cctv,
        'traffic': traffic
    })


@app.route('/api/traffic/status')
def get_all_traffic_status():
    """Get traffic status for all CCTVs"""
    return jsonify(traffic_system.get_traffic_status())


@app.route('/api/traffic/roads')
def get_road_segments():
    """Get road segments with traffic data"""
    roads = traffic_system.db.get_road_segments_with_traffic()
    return jsonify({'roads': roads})


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    try:
        print('Client connected')
        emit('init_data', {
            'cctvs': list(traffic_system.cctvs.values()),
            'traffic': traffic_system.get_traffic_status()
        })
    except Exception as e:
        print(f'[Socket] Connect error: {e}')


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')


def generate_stream(cctv_id):
    """Generate MJPEG stream for a CCTV"""
    # Load no signal image
    no_signal_path = os.path.join('static', 'images', 'no-signal.jpg')
    no_signal_frame = None
    if os.path.exists(no_signal_path):
        no_signal_frame = open(no_signal_path, 'rb').read()
        
    while True:
        frame_data = None
        if cctv_id in traffic_system.detectors:
            frame_data = traffic_system.detectors[cctv_id].get_frame()
            
        if frame_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        elif no_signal_frame:
            # Yield no signal frame if detector not ready or no frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + no_signal_frame + b'\r\n')
                   
        time.sleep(0.1 if not frame_data else 0.033)  # Sleep longer if no frame


@app.route('/stream/<cctv_id>')
def video_stream(cctv_id):
    """Video stream endpoint"""
    return Response(
        generate_stream(cctv_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    # Initialize
    print("=" * 60)
    print("🚦 Smart Traffic Management System - OPTIMIZED")
    print("=" * 60)
    
    # Initialize model
    traffic_system.init_model()
    
    # Initialize database
    traffic_system.db.init_database()
    
    # Load existing CCTVs from database
    traffic_system.load_cctvs_from_db()
    
    # Match CCTVs to OSM road segments
    if len(traffic_system.cctvs) > 0:
        print("\n[Setup] Matching CCTVs to OSM road segments...")
        try:
            from osm_road_matcher import match_cctvs_to_roads
            match_cctvs_to_roads(db=traffic_system.db)
            # Reload CCTVs to pick up updated road_segment_ids
            traffic_system.cctvs.clear()
            traffic_system.load_cctvs_from_db()
        except Exception as e:
            print(f"[Setup] Road matching failed (non-fatal): {e}")
    
    print("\n[Setup] Optimized for low-latency processing")
    print("       - Separate capture and processing threads")
    print("       - Frame dropping if processing can't keep up")
    print("       - Always processes latest frame")
    
    print("\n[Setup] Add CCTVs using: python add_real_cctvs.py")
    
    port = int(os.environ.get('PORT', 5005))
    print(f"\n[Server] Starting on http://127.0.0.1:{port}")
    print("=" * 60)
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
