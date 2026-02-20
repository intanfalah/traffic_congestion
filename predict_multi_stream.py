#!/usr/bin/env python3
"""
Multi-Stream Vehicle Detection
Processes multiple video streams simultaneously with independent tracking.

Usage:
    # Detect on multiple streams
    python predict_multi_stream.py \
        --streams "http://127.0.0.1:8080/video_feed/camera1" \
        --streams "http://127.0.0.1:8080/video_feed/camera2" \
        --streams "http://127.0.0.1:8080/video_feed/camera3"
    
    # Or use config file
    python predict_multi_stream.py --config stream_config.yaml
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import json
import threading
import queue
import sys
import os
from collections import deque, defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from process_results import process_vehicle_data

try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False


@dataclass
class StreamConfig:
    """Configuration for a single stream"""
    stream_id: str
    url: str
    name: str
    line_y: int = 500
    line_x_start: int = 100
    line_x_end: int = 1050
    active: bool = True


@dataclass
class StreamState:
    """Runtime state for a single stream"""
    stream_id: str
    vehicle_in: Dict = field(default_factory=dict)
    vehicle_out: Dict = field(default_factory=dict)
    counted_vehicles: Dict = field(default_factory=dict)
    data_deque: Dict = field(default_factory=dict)
    data_queue: queue.Queue = field(default_factory=queue.Queue)
    deepsort = None
    frame_count: int = 0
    fps: float = 0.0
    
    def get_counts(self):
        return {"in": self.vehicle_in.copy(), "out": self.vehicle_out.copy()}


# Global state
stream_states: Dict[str, StreamState] = {}
states_lock = threading.Lock()
firebase_enabled = False
ref = None
log_ref = None

# Color palette
palette = (2 * 11 - 1, 2 * 15 - 1, 2 ** 20 - 1)


def init_firebase():
    """Initialize Firebase"""
    global firebase_enabled, ref, log_ref
    
    if not FIREBASE_AVAILABLE:
        return
    
    try:
        cred_path = os.environ.get('FIREBASE_CRED_PATH', 'firebase-key.json')
        database_url = os.environ.get('FIREBASE_DB_URL',
            'https://traffic-vision-d32aa-default-rtdb.asia-southeast1.firebasedatabase.app')
        
        if os.path.exists(cred_path):
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {'databaseURL': database_url})
            firebase_enabled = True
            ref = db.reference("traffic_snapshot")
            log_ref = db.reference('traffic_history')
            print("✅ Firebase initialized")
    except Exception as e:
        print(f"⚠️  Firebase: {e}")


def init_tracker():
    """Initialize DeepSORT tracker"""
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    
    return DeepSort(
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


def xyxy_to_xywh(*xyxy):
    """Convert xyxy to xywh format"""
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    return x_c, y_c, bbox_w, bbox_h


def compute_color_for_labels(label):
    """Get color for class label"""
    if label == 0:
        return (85, 45, 255)      # person
    elif label == 2:
        return (222, 82, 175)     # car
    elif label == 3:
        return (0, 204, 255)      # motorcycle
    elif label == 5:
        return (0, 149, 255)      # bus
    else:
        return tuple([int((p * (label ** 2 - label + 1)) % 255) for p in palette])


def draw_border(img, pt1, pt2, color, thickness, r, d):
    """Draw fancy border"""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)
    return img


def UI_box(x, img, color=None, label=None, line_thickness=None):
    """Draw bounding box with label"""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3),
                         (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                   [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A, B, C, D):
    """Check if line segments intersect"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    """Counter-clockwise check"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def get_direction(point1, point2):
    """Get direction string"""
    direction = ""
    if point1[1] > point2[1]:
        direction += "South"
    elif point1[1] < point2[1]:
        direction += "North"
    if point1[0] > point2[0]:
        direction += "East"
    elif point1[0] < point2[0]:
        direction += "West"
    return direction


def process_frame(frame, state: StreamState, model, config: StreamConfig):
    """Process a single frame for a stream"""
    line = [(config.line_x_start, config.line_y), (config.line_x_end, config.line_y)]
    
    # Draw counting line
    cv2.line(frame, line[0], line[1], (46, 162, 112), 3)
    
    height, width, _ = frame.shape
    
    # Run detection
    results = model(frame, conf=0.3)
    det = results[0].boxes
    
    if len(det) > 0:
        xywh_bboxs = []
        confs = []
        oids = []
        
        for box in det:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*[torch.tensor(xyxy)])
            xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
            confs.append([conf])
            oids.append(cls)
        
        if len(xywh_bboxs) > 0 and state.deepsort:
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)
            
            outputs = state.deepsort.update(xywhs, confss, oids, frame)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]
                
                # Draw boxes and count vehicles
                for i, box in enumerate(bbox_xyxy):
                    x1, y1, x2, y2 = [int(b) for b in box]
                    center = (int((x2 + x1) / 2), int((y2 + y2) / 2))
                    id = int(identities[i])
                    obj_name = model.names[int(object_id[i])]
                    
                    if id not in state.counted_vehicles:
                        state.counted_vehicles[id] = {'North': False, 'South': False}
                    
                    if id not in state.data_deque:
                        state.data_deque[id] = deque(maxlen=64)
                    
                    color = compute_color_for_labels(int(object_id[i]))
                    label = f"{id}:{obj_name}"
                    
                    state.data_deque[id].appendleft(center)
                    
                    # Count vehicles crossing line
                    if len(state.data_deque[id]) >= 2:
                        direction = get_direction(state.data_deque[id][0], state.data_deque[id][1])
                        if intersect(state.data_deque[id][0], state.data_deque[id][1], line[0], line[1]):
                            cv2.line(frame, line[0], line[1], (255, 255, 255), 3)
                            if "South" in direction and not state.counted_vehicles[id]['South']:
                                state.vehicle_out[obj_name] = state.vehicle_out.get(obj_name, 0) + 1
                                state.counted_vehicles[id]['South'] = True
                                state.data_queue.put(state.get_counts())
                            elif "North" in direction and not state.counted_vehicles[id]['North']:
                                state.vehicle_in[obj_name] = state.vehicle_in.get(obj_name, 0) + 1
                                state.counted_vehicles[id]['North'] = True
                                state.data_queue.put(state.get_counts())
                    
                    UI_box(box, frame, label=label, color=color, line_thickness=2)
                    
                    # Draw trail
                    for j in range(1, len(state.data_deque[id])):
                        if state.data_deque[id][j - 1] is None or state.data_deque[id][j] is None:
                            continue
                        thickness = int(np.sqrt(64 / float(j + j)) * 1.5)
                        cv2.line(frame, state.data_deque[id][j - 1], state.data_deque[id][j], color, thickness)
    
    # Display counts
    y_offset = 65
    for idx, (key, value) in enumerate(state.vehicle_in.items()):
        cnt_str = f"{key}:{value}"
        cv2.line(frame, (width - 200, 25), (width, 25), [85, 45, 255], 40)
        cv2.putText(frame, 'Vehicles Entering', (width - 1000, 35), 0, 1, [225, 255, 255], thickness=2)
        cv2.line(frame, (width - 50, y_offset + (idx * 40)), (width, y_offset + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(frame, cnt_str, (width - 550, y_offset + 10 + (idx * 40)), 0, 1, [255, 255, 255], thickness=2)
    
    for idx, (key, value) in enumerate(state.vehicle_out.items()):
        cnt_str = f"{key}:{value}"
        cv2.line(frame, (20, 25), (500, 25), [85, 45, 255], 40)
        cv2.putText(frame, 'Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2)
        cv2.line(frame, (20, y_offset + (idx * 40)), (127, y_offset + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(frame, cnt_str, (11, y_offset + 10 + (idx * 40)), 0, 1, [255, 255, 255], thickness=2)
    
    return frame


def detection_worker(config: StreamConfig, model_path: str, display: bool):
    """Worker thread for processing a single stream"""
    global stream_states
    
    print(f"[{config.stream_id}] Starting detection on: {config.url}")
    
    # Initialize state
    state = StreamState(stream_id=config.stream_id)
    state.deepsort = init_tracker()
    
    with states_lock:
        stream_states[config.stream_id] = state
    
    # Load model (each thread has its own model instance for thread safety)
    model = YOLO(model_path)
    
    # Open stream
    cap = cv2.VideoCapture(config.url)
    if not cap.isOpened():
        print(f"[{config.stream_id}] ❌ Failed to open: {config.url}")
        return
    
    print(f"[{config.stream_id}] ✅ Stream opened")
    
    last_time = time.time()
    
    while config.active:
        ret, frame = cap.read()
        if not ret:
            print(f"[{config.stream_id}] ⚠️  Stream error, retrying...")
            time.sleep(1)
            cap = cv2.VideoCapture(config.url)
            continue
        
        # Process frame
        processed = process_frame(frame, state, model, config)
        state.frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        if current_time - last_time >= 1.0:
            state.fps = state.frame_count / (current_time - last_time)
            state.frame_count = 0
            last_time = current_time
        
        # Display if enabled
        if display:
            cv2.imshow(f'Stream: {config.name}', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyWindow(f'Stream: {config.name}')
    print(f"[{config.stream_id}] 🛑 Stopped")


def process_and_send_data_all(interval: int = 600):
    """Process and send data from all streams"""
    
    def process_all():
        try:
            with states_lock:
                all_data = {}
                for stream_id, state in stream_states.items():
                    if not state.data_queue.empty():
                        # Aggregate data from this stream
                        stream_data = {"in": {}, "out": {}}
                        while not state.data_queue.empty():
                            counts = state.data_queue.get()
                            for direction in ["in", "out"]:
                                for key, value in counts[direction].items():
                                    stream_data[direction][key] = stream_data[direction].get(key, 0) + value
                        
                        # Process this stream's data
                        if stream_data["in"] or stream_data["out"]:
                            results = process_vehicle_data(stream_data, capacity=1325)
                            all_data[stream_id] = {
                                "entering": results['in'].get('vehicle_class_counts'),
                                "leaving": results['out'].get('vehicle_class_counts'),
                                "LOS": {"in": results['in'].get('LOS'), "out": results['out'].get('LOS')},
                                "DS": {"in": results['in'].get('DS'), "out": results['out'].get('DS')}
                            }
                        
                        # Reset counters
                        state.vehicle_in = {}
                        state.vehicle_out = {}
                        state.counted_vehicles = {}
                
                # Send to Firebase if enabled
                if firebase_enabled and ref and all_data:
                    firebase_data = {
                        "streams": all_data,
                        "timestamp": datetime.now().isoformat()
                    }
                    ref.child('multi_stream_data').set(firebase_data)
                    print(f"✅ Multi-stream data sent at: {datetime.now()}")
                    
        except Exception as e:
            print(f"❌ Error processing multi-stream data: {e}")
        finally:
            threading.Timer(interval, process_all).start()
    
    process_all()


def parse_opt():
    parser = argparse.ArgumentParser(description='Multi-Stream Vehicle Detection')
    parser.add_argument('--streams', type=str, nargs='+', required=True,
                       help='List of stream URLs')
    parser.add_argument('--names', type=str, nargs='+',
                       help='Names for each stream (optional)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLOv8 model path')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--line-y', type=int, default=500,
                       help='Counting line Y position')
    parser.add_argument('--display', action='store_true',
                       help='Show video windows')
    parser.add_argument('--no-firebase', action='store_true',
                       help='Disable Firebase')
    return parser.parse_args()


def main():
    opt = parse_opt()
    
    print("=" * 60)
    print("🚦 Multi-Stream Vehicle Detection")
    print("=" * 60)
    
    if not opt.no_firebase:
        init_firebase()
    
    # Create stream configs
    configs = []
    for i, url in enumerate(opt.streams):
        name = opt.names[i] if opt.names and i < len(opt.names) else f"Camera {i+1}"
        stream_id = f"stream_{i+1}"
        configs.append(StreamConfig(
            stream_id=stream_id,
            url=url,
            name=name,
            line_y=opt.line_y
        ))
    
    print(f"\n📡 Processing {len(configs)} streams:")
    for cfg in configs:
        print(f"   • {cfg.name}: {cfg.url}")
    print()
    
    # Start data processing thread
    threading.Thread(target=process_and_send_data_all, daemon=True).start()
    
    # Start detection workers
    threads = []
    for cfg in configs:
        t = threading.Thread(
            target=detection_worker,
            args=(cfg, opt.model, opt.display),
            daemon=True
        )
        t.start()
        threads.append(t)
        time.sleep(0.5)  # Stagger starts
    
    print("-" * 60)
    print("Press Ctrl+C to stop all streams")
    print("-" * 60)
    
    try:
        while True:
            time.sleep(1)
            
            # Print status
            with states_lock:
                for stream_id, state in stream_states.items():
                    in_total = sum(state.vehicle_in.values())
                    out_total = sum(state.vehicle_out.values())
                    print(f"\r[{stream_id}] FPS:{state.fps:5.1f} IN:{in_total:3d} OUT:{out_total:3d}", end="")
                    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all streams...")
        for cfg in configs:
            cfg.active = False
        for t in threads:
            t.join(timeout=2)
    
    print("\n✅ All streams stopped")


if __name__ == "__main__":
    main()
