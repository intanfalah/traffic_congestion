#!/usr/bin/env python3
"""
Smart Traffic Management System
Main Flask application with map-based CCTV dashboard
"""

import os

# Set FFmpeg options BEFORE importing cv2.
# These feeds are HLS over HTTP (not RTSP), so use HTTP timeouts + auto-reconnect.
# rw_timeout/timeout (microseconds) make a stalled read() return False instead of
# blocking the detector thread forever; reconnect* lets FFmpeg re-fetch dropped
# HLS segments. (The old RTSP-only `stimeout` did nothing here, so a stalled
# stream hung the worker with status stuck on 'active' and no frames.)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rw_timeout;15000000|timeout;15000000|'
    'reconnect;1|reconnect_streamed;1|reconnect_delay_max;5|'
    'buffer_size;1024000|max_delay;5000000'
)

import json
import math
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

# Lane splitting / road offsetting
import lane_utils

app = Flask(__name__)
app.config['SECRET_KEY'] = 'traffic-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# COCO class ids we treat as vehicles: car, motorcycle, bus, truck.
# Filtering to these avoids YOLO false positives (train/boat/person) polluting
# the boxes, the line-crossing count, and the density estimate.
VEHICLE_CLASSES = {2, 3, 5, 7}

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
    
    @staticmethod
    def _json_safe(entry):
        """Return a JSON-serialisable copy of a traffic-data entry.

        The stored entry holds a `history` deque (not serialisable by jsonify);
        convert it to a list and copy the rest shallowly.
        """
        if entry is None:
            return None
        safe = dict(entry)
        hist = safe.get('history')
        if hist is not None:
            safe['history'] = list(hist)
        return safe

    def get_traffic_status(self, cctv_id=None):
        """Get JSON-safe traffic status for all or a specific CCTV."""
        if cctv_id:
            return self._json_safe(self.traffic_data.get(cctv_id))
        return {cid: self._json_safe(data) for cid, data in self.traffic_data.items()}
    
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

        # Lane split line (normalised coords) for this camera + live per-lane density
        self.split_line, self.lane_flip = lane_utils.get_split_line(cctv_id)
        self.lane_density = {'A': 0, 'B': 0}

        # Congestion signals: vehicles PRESENT per processed frame (density),
        # and recent centroid history per DeepSORT track (speed / stopped).
        self.density_samples = deque(maxlen=200)
        self.lane_samples = {'A': deque(maxlen=200), 'B': deque(maxlen=200)}
        self.track_history = {}
        self.frame_h = None
        
    def _interruptible_sleep(self, seconds):
        """Sleep in short slices so a stop() request is honored promptly."""
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(0.2)

    def _open_stream(self):
        """Try to open the video stream. Returns True and sets self.cap on success."""
        # FFmpeg backend first (best HLS support)
        cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            # GStreamer HLS pipeline as fallback
            cap.release()
            gst_pipeline = f'souphttpsrc location={self.stream_url} ! hlsdemux ! decodebin ! videoconvert ! appsink'
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self.cap = cap
            return True
        cap.release()
        return False

    def run(self):
        """Main detection loop with automatic reconnection.

        Keeps trying to (re)open the stream with exponential backoff so a CCTV
        that is offline at startup — or that drops mid-stream — recovers on its
        own once the upstream feed returns, instead of the worker exiting.
        """
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

        # Reconnection backoff (seconds)
        base_delay = 5
        max_delay = 30
        delay = base_delay
        attempt = 0

        # Outer loop: keep (re)connecting until stopped
        while self.running:
            attempt += 1
            print(f"[Detector {self.cctv_id}] Opening stream (attempt {attempt})...")
            self.system.cctvs[self.cctv_id]['status'] = 'connecting'

            if not self._open_stream():
                self.system.cctvs[self.cctv_id]['status'] = 'error'
                print(f"[Detector {self.cctv_id}] Open failed; retrying in {delay}s")
                self._interruptible_sleep(delay)
                delay = min(delay * 2, max_delay)
                continue

            # Connected — reset backoff for the next disconnect
            print(f"[Detector {self.cctv_id}] Stream opened successfully")
            self.system.cctvs[self.cctv_id]['status'] = 'active'
            attempt = 0
            delay = base_delay

            # Read frames until the stream drops or we're stopped
            self._process_stream()

            # Stream dropped or stop requested — release and loop back to reconnect
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            if self.running:
                self.system.cctvs[self.cctv_id]['status'] = 'reconnecting'
                self._interruptible_sleep(delay)

        # Final cleanup
        if self.cap is not None:
            self.cap.release()
        print(f"[Detector {self.cctv_id}] Stopped")

    def _process_stream(self):
        """Read and process frames until the stream drops or stop() is called."""
        consecutive_failures = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                # Tolerate brief hiccups; only reconnect on sustained failure
                if consecutive_failures >= 30:
                    print(f"[Detector {self.cctv_id}] Stream dropped, reconnecting...")
                    return
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            self.frame_count += 1

            # Process every 3rd frame for performance (10fps processing).
            # Guard so a single bad frame can't kill the detector thread.
            if self.frame_count % 3 == 0:
                try:
                    self.process_frame(frame)
                except Exception as e:
                    print(f"[Detector {self.cctv_id}] process_frame error: {e}")

            # Store in buffer for streaming
            self.frame_buffer.append(frame)

            # Calculate traffic metrics every 10 seconds
            current_time = time.time()
            if current_time - self.last_process_time >= 10:
                self.calculate_traffic_metrics()
                self.last_process_time = current_time
    
    def process_frame(self, frame):
        """Process a single frame for detection"""
        # Run YOLO detection
        results = self.model(frame, conf=0.3)
        det = results[0].boxes
        self.frame_h = frame.shape[0]
        vehicles_this_frame = 0
        lane_counts = {'A': 0, 'B': 0}

        if len(det) > 0:
            xywh_bboxs = []
            confs = []
            oids = []
            
            for box in det:
                xyxy = box.xyxy[0].cpu().numpy()
                # Use Python floats: 0-d numpy scalars break torch.Tensor(confs)
                # with "len() of unsized object" once DeepSORT is enabled.
                conf = float(box.conf[0].item())
                cls = int(box.cls[0].item())

                # Only track/count vehicles (skip person/train/boat/etc.)
                if cls not in VEHICLE_CLASSES:
                    continue

                vehicles_this_frame += 1

                # Draw a box on every detected vehicle each frame, so vehicles
                # are highlighted immediately. DeepSORT below only does counting.
                obj_name = results[0].names.get(cls, 'vehicle')
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                x2, y2 = int(xyxy[2]), int(xyxy[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{obj_name} {conf:.2f}", (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Tally which lane this vehicle is in (by its ground point)
                lane_counts[lane_utils.lane_of(
                    (x1 + x2) / 2, y2, frame.shape[1], self.frame_h,
                    self.split_line, self.lane_flip)] += 1

                x_c = float((xyxy[0] + xyxy[2]) / 2)
                y_c = float((xyxy[1] + xyxy[3]) / 2)
                w = float(xyxy[2] - xyxy[0])
                h = float(xyxy[3] - xyxy[1])

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
        
        # Record how many vehicles are PRESENT in this frame (density signal)
        self.density_samples.append(vehicles_this_frame)
        self.lane_samples['A'].append(lane_counts['A'])
        self.lane_samples['B'].append(lane_counts['B'])

        # Draw overlay
        self.draw_overlay(frame)
        self.processed_frame = frame
    
    def track_vehicles(self, frame, outputs, names):
        """Count vehicles crossing the line using DeepSORT track identities.

        Detection boxes are drawn in process_frame; here we only track each id's
        vertical motion and increment the count when it crosses the line. The
        vehicle's box is briefly redrawn blue on the frame it is counted.
        """
        height, width = frame.shape[:2]
        line_y = height // 2

        for output in outputs:
            x1, y1, x2, y2, track_id, cls_id = output
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            cls_id = int(cls_id)

            center_y = (y1 + y2) // 2
            obj_name = names.get(cls_id, 'unknown')

            # Record centroid history for speed estimation (stopped detection)
            hist = self.track_history.setdefault(track_id, deque(maxlen=15))
            hist.append((time.time(), (x1 + x2) // 2, center_y))

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
                    # Flash the crossing vehicle blue as count confirmation
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 3)
    
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
    
    def _stopped_fraction(self):
        """Fraction of recently-seen tracks that are (nearly) stationary.

        A track counts as stopped when its centroid moved slower than ~1% of
        the frame height per second over its recent history. Returns None when
        there are no usable tracks (e.g. DeepSORT unavailable).
        """
        now = time.time()
        stopped_speed = 0.01 * (self.frame_h or 720)  # px/sec
        moving = stopped = 0
        for hist in self.track_history.values():
            if len(hist) < 2:
                continue
            t0, x0, y0 = hist[0]
            t1, x1, y1 = hist[-1]
            if now - t1 > 2 or t1 - t0 < 0.5:
                continue  # stale track, or too short a window to measure
            speed = math.hypot(x1 - x0, y1 - y0) / (t1 - t0)
            if speed < stopped_speed:
                stopped += 1
            else:
                moving += 1
        total = moving + stopped
        return (stopped / total) if total else None

    def calculate_traffic_metrics(self):
        """Publish traffic metrics for the last 10s window.

        Congestion is classified from DENSITY (avg vehicles present) plus the
        stopped fraction of tracked vehicles — NOT from line-crossing flow,
        which reads near zero both on an empty road and in a gridlock.
        vehicles_per_minute is still reported, but only as throughput.
        """
        vehicles_per_minute = self.vehicle_count['in'] * 6  # Scale 10s to 1min

        samples = list(self.density_samples)
        density = (sum(samples) / len(samples)) if samples else 0
        stopped_frac = self._stopped_fraction()
        congestion, level = lane_utils.classify_congestion(density, stopped_frac)

        # Per-lane average density over the window
        lanes = {}
        for name, lane_dq in self.lane_samples.items():
            vals = list(lane_dq)
            avg = (sum(vals) / len(vals)) if vals else 0
            lc, llos = lane_utils.lane_congestion(round(avg))
            lanes[name] = {'density': round(avg, 1),
                           'congestion_level': lc, 'los': llos}

        data = {
            'vehicle_count': round(density),      # vehicles present (density)
            'density': round(density, 1),
            'stopped_fraction': round(stopped_frac, 2) if stopped_frac is not None else None,
            'vehicles_per_minute': vehicles_per_minute,   # throughput only
            'vehicle_types': dict(self.vehicle_types),
            'congestion_level': congestion,
            'los': level,
            'lanes': lanes,
            'source': 'live',
            'timestamp': datetime.now().isoformat()
        }

        self.system.update_traffic_data(self.cctv_id, data)

        # Reset window counters
        self.vehicle_count = {'in': 0, 'out': 0}
        self.vehicle_types.clear()
        self.counted_vehicles.clear()
        self.density_samples.clear()
        self.lane_samples['A'].clear()
        self.lane_samples['B'].clear()
        # Drop stale track histories so the dict doesn't grow unbounded
        now = time.time()
        self.track_history = {
            tid: h for tid, h in self.track_history.items()
            if h and now - h[-1][0] <= 10
        }
    
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


class TrafficEstimator(threading.Thread):
    """Background density-based congestion estimator.

    Samples every CCTV on a rotating schedule WITHOUT streaming video: it opens
    a feed, grabs a frame, counts the vehicles present (density), maps that to a
    congestion level, updates the shared traffic data, and closes the feed. Only
    one camera is open at a time, so all cameras get live congestion colours on
    the map without the many-concurrent-streams starvation problem.

    Cameras a viewer is actively watching (their on-demand DetectionWorker is
    running) are skipped, since that worker already reports richer flow metrics.
    """

    # COCO vehicle classes: car, motorcycle, bus, truck
    VEHICLE_CLASSES = {2, 3, 5, 7}

    def __init__(self, system, interval=30):
        super().__init__(daemon=True)
        self.system = system
        self.interval = interval  # target seconds between refreshes of a camera
        self.running = False
        self.model = None

    def run(self):
        self.running = True
        print(f"[Estimator] Loading YOLO model (density mode, every {self.interval}s)...")
        self.model = YOLO('yolov8n.pt')
        print("[Estimator] Started")
        while self.running:
            cctv_ids = list(self.system.cctvs.keys())
            if not cctv_ids:
                time.sleep(2)
                continue
            # Pace so a full round of all cameras takes about `interval` seconds
            per_cam = max(self.interval / len(cctv_ids), 2)
            for cctv_id in cctv_ids:
                if not self.running:
                    break
                self._tick(cctv_id)
                time.sleep(per_cam)

    def _tick(self, cctv_id):
        """Sample one camera unless a viewer's worker already owns its feed."""
        det = self.system.detectors.get(cctv_id)
        if det and getattr(det, 'running', False):
            return
        try:
            self._sample(cctv_id)
        except Exception as e:
            print(f"[Estimator] {cctv_id} sample error: {e}")

    def _sample(self, cctv_id):
        """Open the feed briefly, count vehicles in one frame, update congestion."""
        cctv = self.system.cctvs.get(cctv_id)
        if not cctv:
            return
        cap = cv2.VideoCapture(cctv['stream_url'], cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            if not cap.isOpened():
                return  # unreachable right now; leave last-known congestion
            # Read a few frames and keep the most recent (skip stale buffer)
            frame = None
            for _ in range(5):
                ret, f = cap.read()
                if ret and f is not None:
                    frame = f
            if frame is None:
                return
        finally:
            cap.release()

        h, w = frame.shape[:2]
        split_line, flip = lane_utils.get_split_line(cctv_id)
        count, vehicle_types, lane_counts = self._count_by_lane(
            frame, w, h, split_line, flip)

        congestion, los = self._density_to_congestion(count)
        lanes = {}
        for name, c in lane_counts.items():
            lc, llos = lane_utils.lane_congestion(c)
            lanes[name] = {'density': c, 'congestion_level': lc, 'los': llos}
        self.system.update_traffic_data(cctv_id, {
            'vehicle_count': count,
            'density': count,
            'vehicles_per_minute': 0,          # density mode, not a flow metric
            'vehicle_types': dict(vehicle_types),
            'congestion_level': congestion,
            'los': los,
            'lanes': lanes,
            'source': 'density',
            'timestamp': datetime.now().isoformat(),
        })

    def _count_by_lane(self, frame, w, h, split_line, flip):
        """Detect vehicles in a frame and tally total + per-lane counts."""
        results = self.model(frame, conf=0.3, verbose=False)
        boxes = results[0].boxes
        names = results[0].names
        count = 0
        vehicle_types = defaultdict(int)
        lane_counts = {'A': 0, 'B': 0}
        for box in boxes:
            cls = int(box.cls[0].item())
            if cls not in self.VEHICLE_CLASSES:
                continue
            count += 1
            vehicle_types[names.get(cls, 'vehicle')] += 1
            # Assign to a lane by the vehicle's ground point (bottom-centre)
            xyxy = box.xyxy[0].cpu().numpy()
            gx = float((xyxy[0] + xyxy[2]) / 2)
            gy = float(xyxy[3])
            lane_counts[lane_utils.lane_of(gx, gy, w, h, split_line, flip)] += 1
        return count, vehicle_types, lane_counts

    @staticmethod
    def _density_to_congestion(count):
        """Single-frame density -> congestion, via the shared classifier.

        No speed signal here (one frame, no tracks), so this is density-only.
        """
        return lane_utils.classify_congestion(count)

    def stop(self):
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


@app.route('/api/cctvs/<cctv_id>/lane_line', methods=['GET', 'POST'])
def cctv_lane_line(cctv_id):
    """Get or set a camera's lane split line (normalised 0..1 image coords)."""
    if request.method == 'POST':
        data = request.json or {}
        line = data.get('split_line')
        # Expect [[x1,y1],[x2,y2]] with each value in 0..1
        if (not isinstance(line, list) or len(line) != 2
                or not all(isinstance(p, list) and len(p) == 2 for p in line)):
            return jsonify({'success': False, 'error': 'split_line must be [[x1,y1],[x2,y2]]'}), 400
        entry = lane_utils.set_split_line(cctv_id, line, bool(data.get('flip', False)))
        return jsonify({'success': True, 'lane': entry})
    split_line, flip = lane_utils.get_split_line(cctv_id)
    return jsonify({
        'split_line': split_line,
        'flip': flip,
        'calibrated': lane_utils.is_calibrated(cctv_id),
    })


@app.route('/api/traffic/status')
def get_all_traffic_status():
    """Get traffic status for all CCTVs"""
    return jsonify(traffic_system.get_traffic_status())


@app.route('/api/traffic/roads')
def get_road_segments():
    """Get road segments with traffic data, including per-lane offset geometry."""
    roads = traffic_system.db.get_road_segments_with_traffic()
    for road in roads:
        _attach_lane_geometry(road)
    return jsonify({'roads': roads})


def _attach_lane_geometry(road):
    """If the road's camera reports per-lane density, add two offset polylines
    (one per lane) so the map can draw each carriageway coloured separately."""
    coords = road.get('coordinates') or []
    cctv_id = road.get('cctv_id')
    traffic = traffic_system.traffic_data.get(cctv_id, {}) if cctv_id else {}
    lanes = traffic.get('lanes')
    if len(coords) < 2 or not lanes:
        return
    offsets = {'A': lane_utils.LANE_OFFSET_M, 'B': -lane_utils.LANE_OFFSET_M}
    road['lanes'] = [
        {
            'name': name,
            'congestion_level': info.get('congestion_level', 'UNKNOWN'),
            'density': info.get('density', 0),
            'los': info.get('los', '-'),
            'coordinates': lane_utils.offset_polyline(coords, offsets[name]),
        }
        for name, info in sorted(lanes.items())
    ]


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


@app.route('/snapshot/<cctv_id>')
def video_snapshot(cctv_id):
    """Return a single JPEG frame for polling-based display"""
    if cctv_id in traffic_system.detectors:
        frame = traffic_system.detectors[cctv_id].get_frame()
        if frame:
            return Response(frame, mimetype='image/jpeg',
                            headers={'Cache-Control': 'no-store'})
    # Return a 1x1 black pixel when no frame is available yet
    import base64
    black = base64.b64decode(
        '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U'
        'HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN'
        'DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy'
        'MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA'
        'AAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA'
        '/9oADAMBAAIRAxEAPwCwABmX/9k='
    )
    return Response(black, mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-store'})


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

    # On-demand detection: detectors are NOT started here. The dashboard starts a
    # camera's detector when its popup is opened and stops it when closed
    # (POST /api/cctvs/<id>/start|stop). This keeps only the viewed camera(s)
    # running so every stream reliably delivers video, instead of running all
    # feeds at once and starving each other.
    # Set AUTO_START_ALL=true to restore the old "start every camera" behaviour.
    if os.environ.get('AUTO_START_ALL', 'false').lower() == 'true':
        for cctv_id in list(traffic_system.cctvs.keys()):
            ok, msg = traffic_system.start_cctv(cctv_id)
            print(f"[Setup] start {cctv_id}: {msg}")
    else:
        print(f"[Setup] On-demand mode: {len(traffic_system.cctvs)} CCTVs loaded, "
              "detectors start when a camera is opened.")

    # Background density estimator: keeps live congestion on the map for ALL
    # cameras without streaming video (samples each one briefly, ~every 30s).
    # Set DENSITY_ESTIMATE=false to disable.
    if os.environ.get('DENSITY_ESTIMATE', 'true').lower() == 'true':
        interval = int(os.environ.get('DENSITY_INTERVAL', '30'))
        traffic_system.estimator = TrafficEstimator(traffic_system, interval=interval)
        traffic_system.estimator.start()
        print(f"[Setup] Background density estimator running (every {interval}s).")

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
