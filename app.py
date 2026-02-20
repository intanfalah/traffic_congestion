#!/usr/bin/env python3
"""
Smart Traffic Management System
Main Flask application with map-based CCTV dashboard
"""

import os

# Set FFmpeg options BEFORE importing cv2
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000|max_delay;5000000'

import json
import threading
import time
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
            
            detector = DetectionWorker(cctv_id, cctv['stream_url'], self)
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


class DetectionWorker(threading.Thread):
    """Background detection worker for a single CCTV"""
    
    def __init__(self, cctv_id, stream_url, system):
        super().__init__(daemon=True)
        self.cctv_id = cctv_id
        self.stream_url = stream_url
        self.system = system
        self.running = False
        self.cap = None
        self.frame_buffer = deque(maxlen=30)  # 1 second at 30fps
        self.processed_frame = None
        self.model = None  # Will be loaded in run()
        
        # Initialize DeepSORT (optional - falls back to simple tracking if checkpoint missing)
        self.deepsort = None
        self.use_deepsort = False
        try:
            cfg_deep = get_config()
            cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
            
            # Check if checkpoint exists
            import os
            if os.path.exists(cfg_deep.DEEPSORT.REID_CKPT):
                self.deepsort = DeepSort(
                    cfg_deep.DEEPSORT.REID_CKPT,
                    max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                    min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg_deep.DEEPSORT.MAX_AGE,
                    n_init=cfg_deep.DEEPSORT.N_INIT,
                    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                    use_cuda=torch.cuda.is_available()
                )
                self.use_deepsort = True
                print(f"[Detector {self.cctv_id}] DeepSORT initialized")
            else:
                print(f"[Detector {self.cctv_id}] DeepSORT checkpoint not found, using simple detection")
        except Exception as e:
            print(f"[Detector {self.cctv_id}] DeepSORT init failed: {e}, using simple detection")
        
        # Tracking state
        self.counted_vehicles = {}
        self.vehicle_count = {'in': 0, 'out': 0}
        self.vehicle_types = defaultdict(int)
        self.data_deque = {}
        self.frame_count = 0
        self.last_process_time = time.time()
        
    def run(self):
        """Main detection loop"""
        self.running = True
        print(f"[Detector {self.cctv_id}] Starting...")
        print(f"[Detector {self.cctv_id}] Stream URL: {self.stream_url[:60]}...")
        
        # Load YOLO model (each thread needs its own)
        print(f"[Detector {self.cctv_id}] Loading YOLO model...")
        try:
            self.model = YOLO('yolov8n.pt')
            print(f"[Detector {self.cctv_id}] Model loaded")
        except Exception as e:
            print(f"[Detector {self.cctv_id}] Failed to load model: {e}")
            self.system.cctvs[self.cctv_id]['status'] = 'error'
            return
        
        # Open stream with FFmpeg backend for HLS support
        print(f"[Detector {self.cctv_id}] Opening with FFmpeg...")
        self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            print(f"[Detector {self.cctv_id}] Failed to open stream, retrying with HTTP...")
            # Try with GStreamer as fallback
            gst_pipeline = f'souphttpsrc location={self.stream_url} ! hlsdemux ! decodebin ! videoconvert ! appsink'
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                print(f"[Detector {self.cctv_id}] Failed to open stream completely")
                self.system.cctvs[self.cctv_id]['status'] = 'error'
                return
        
        print(f"[Detector {self.cctv_id}] Stream opened successfully")
        self.system.cctvs[self.cctv_id]['status'] = 'active'
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"[Detector {self.cctv_id}] Stream error, reconnecting...")
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.stream_url)
                continue
            
            self.frame_count += 1
            
            # Process every 3rd frame for performance (10fps processing)
            if self.frame_count % 3 == 0:
                self.process_frame(frame)
            
            # Store in buffer for streaming
            self.frame_buffer.append(frame)
            
            # Calculate traffic metrics every 10 seconds
            current_time = time.time()
            if current_time - self.last_process_time >= 10:
                self.calculate_traffic_metrics()
                self.last_process_time = current_time
        
        self.cap.release()
        print(f"[Detector {self.cctv_id}] Stopped")
    
    def process_frame(self, frame):
        """Process a single frame for detection"""
        # Run YOLO detection
        results = self.model(frame, conf=0.3)
        det = results[0].boxes
        
        if len(det) > 0:
            xywh_bboxs = []
            confs = []
            oids = []
            
            for box in det:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                x_c = (xyxy[0] + xyxy[2]) / 2
                y_c = (xyxy[1] + xyxy[3]) / 2
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                
                xywh_bboxs.append([x_c, y_c, w, h])
                confs.append([conf])
                oids.append(cls)
            
            if len(xywh_bboxs) > 0 and self.use_deepsort and self.deepsort:
                # Use DeepSORT for tracking
                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                
                outputs = self.deepsort.update(xywhs, confss, oids, frame)
                
                if len(outputs) > 0:
                    self.track_vehicles(frame, outputs, results[0].names)
            else:
                # Simple detection without tracking - just count vehicles
                self.simple_detection(frame, xywh_bboxs, confs, oids, results[0].names)
        
        # Draw overlay
        self.draw_overlay(frame)
        self.processed_frame = frame
    
    def track_vehicles(self, frame, outputs, names):
        """Track and count vehicles with DeepSORT"""
        height, width = frame.shape[:2]
        line_y = height // 2
        
        for output in outputs:
            x1, y1, x2, y2, track_id, cls_id = output
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            cls_id = int(cls_id)
            
            center_y = (y1 + y2) // 2
            obj_name = names.get(cls_id, 'unknown')
            
            # Initialize tracking for new vehicle
            if track_id not in self.counted_vehicles:
                self.counted_vehicles[track_id] = {
                    'counted': False,
                    'class': obj_name,
                    'positions': deque(maxlen=10)
                }
            
            self.counted_vehicles[track_id]['positions'].append(center_y)
            
            # Count when crossing the line
            positions = self.counted_vehicles[track_id]['positions']
            if len(positions) >= 2 and not self.counted_vehicles[track_id]['counted']:
                if (positions[0] < line_y and positions[-1] >= line_y) or \
                   (positions[0] > line_y and positions[-1] <= line_y):
                    self.counted_vehicles[track_id]['counted'] = True
                    self.vehicle_count['in'] += 1
                    self.vehicle_types[obj_name] += 1
    
    def simple_detection(self, frame, xywh_bboxs, confs, oids, names):
        """Simple vehicle detection without DeepSORT tracking"""
        height, width = frame.shape[:2]
        line_y = height // 2
        
        # Draw bounding boxes
        for i, (xywh, conf, oid) in enumerate(zip(xywh_bboxs, confs, oids)):
            x_c, y_c, w, h = xywh
            x1, y1 = int(x_c - w/2), int(y_c - h/2)
            x2, y2 = int(x_c + w/2), int(y_c + h/2)
            
            obj_name = names.get(oid, 'unknown')
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{obj_name} {conf[0]:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Simple counting: just count all detected vehicles
            # (without tracking, we can't do proper line crossing)
            self.vehicle_types[obj_name] += 1
        
        # Update count (simplified - just total detections)
        self.vehicle_count['in'] = sum(self.vehicle_types.values())
    
    def draw_overlay(self, frame):
        """Draw detection overlay on frame"""
        height, width = frame.shape[:2]
        line_y = height // 2
        
        # Draw counting line
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 2)
        
        # Draw counts
        cv2.putText(frame, f"Count: {self.vehicle_count['in']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def calculate_traffic_metrics(self):
        """Calculate traffic congestion metrics"""
        vehicles_per_minute = self.vehicle_count['in'] * 6  # Scale 10s to 1min
        
        # Determine congestion level based on vehicle count
        if vehicles_per_minute < 10:
            congestion = 'FREE_FLOW'
            level = 'A'
        elif vehicles_per_minute < 30:
            congestion = 'MODERATE'
            level = 'C'
        elif vehicles_per_minute < 60:
            congestion = 'CONGESTED'
            level = 'D'
        else:
            congestion = 'SEVERE'
            level = 'F'
        
        data = {
            'vehicle_count': self.vehicle_count['in'],
            'vehicles_per_minute': vehicles_per_minute,
            'vehicle_types': dict(self.vehicle_types),
            'congestion_level': congestion,
            'los': level,
            'timestamp': datetime.now().isoformat()
        }
        
        self.system.update_traffic_data(self.cctv_id, data)
        
        # Reset counters
        self.vehicle_count = {'in': 0, 'out': 0}
        self.vehicle_types.clear()
        self.counted_vehicles.clear()
    
    def get_frame(self):
        """Get latest processed frame"""
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
        # Don't re-raise to prevent disconnect


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')


def generate_stream(cctv_id):
    """Generate MJPEG stream for a CCTV"""
    while True:
        if cctv_id in traffic_system.detectors:
            frame = traffic_system.detectors[cctv_id].get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30fps


@app.route('/stream/<cctv_id>')
def video_stream(cctv_id):
    """Video stream endpoint"""
    return Response(
        generate_stream(cctv_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def init_demo_data():
    """Initialize with demo CCTV data (Semarang area)"""
    demo_cctvs = [
        {
            'id': 'cctv_1',
            'name': 'Jl. Pahlawan - Simpang Lima',
            'lat': -6.9902,
            'lng': 110.4229,
            'url': '0',  # Default webcam for demo
            'road_id': 'road_1'
        },
        {
            'id': 'cctv_2',
            'name': 'Jl. MT Haryono - Pandanaran',
            'lat': -6.9965,
            'lng': 110.4310,
            'url': '0',
            'road_id': 'road_2'
        },
        {
            'id': 'cctv_3',
            'name': 'Jl. Ahmad Yani - Kaligawe',
            'lat': -6.9725,
            'lng': 110.4450,
            'url': '0',
            'road_id': 'road_3'
        },
        {
            'id': 'cctv_4',
            'name': 'Jl. Gajah Mada - Kudu',
            'lat': -6.9830,
            'lng': 110.4100,
            'url': '0',
            'road_id': 'road_4'
        },
        {
            'id': 'cctv_5',
            'name': 'Jl. Pemuda - Balai Kota',
            'lat': -6.9800,
            'lng': 110.4080,
            'url': '0',
            'road_id': 'road_5'
        }
    ]
    
    for cctv in demo_cctvs:
        traffic_system.add_cctv(
            cctv_id=cctv['id'],
            name=cctv['name'],
            lat=cctv['lat'],
            lng=cctv['lng'],
            stream_url=cctv['url'],
            road_segment_id=cctv['road_id']
        )
        # Start detection automatically
        traffic_system.start_cctv(cctv['id'])


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
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
