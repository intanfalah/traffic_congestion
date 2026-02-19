#!/usr/bin/env python3
"""
Vehicle Detection using Web Stream Source (Modern Ultralytics API)
This script runs vehicle detection on a video stream from the web streaming server.

Usage:
    # First, start the web streaming server:
    python web_stream.py
    
    # Then, run detection on the stream:
    python predict_stream.py --stream-url http://127.0.0.1:8080/video_feed
    
    # Or use the default stream URL:
    python predict_stream.py
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
from collections import deque
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from process_results import process_vehicle_data

# Try to import Firebase
try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("⚠️  firebase-admin not installed. Firebase integration disabled.")

# Configuration
STREAM_URL = "http://127.0.0.1:8080/video_feed"  # Default web stream URL

# Global variables
palette = (2 * 11 - 1, 2 * 15 - 1, 2 ** 20 - 1)
data_deque = {}
deepsort = None
vehicle_out = {}
vehicle_in = {}
line = [(100, 500), (1050, 500)]
start_time = time.time()
data_queue = queue.Queue()
counted_vehicles = {}

# Firebase
firebase_enabled = False
ref = None
log_ref = None


def init_firebase():
    """Initialize Firebase connection"""
    global firebase_enabled, ref, log_ref
    
    if not FIREBASE_AVAILABLE:
        print("⚠️  Firebase not available (firebase-admin not installed)")
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
        else:
            print(f"⚠️  Firebase credentials not found: {cred_path}")
            
    except Exception as e:
        print(f"⚠️  Firebase init failed: {e}")


def init_tracker():
    """Initialize DeepSORT tracker"""
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg_deep.DEEPSORT.MAX_AGE,
        n_init=cfg_deep.DEEPSORT.N_INIT,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=torch.cuda.is_available() if 'torch' in dir() else False
    )
    print("✅ DeepSORT tracker initialized")


# Import torch for cuda check
import torch


def xyxy_to_xywh(*xyxy):
    """Calculate the relative bounding box from absolute pixel values."""
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """Simple function that adds fixed color depending on the class"""
    if label == 0:  # person
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
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
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def get_direction(point1, point2):
    direction_str = ""
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    return direction_str


def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    global counted_vehicles
    cv2.line(img, line[0], line[1], (46, 162, 112), 3)

    height, width, _ = img.shape
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))
        id = int(identities[i]) if identities is not None else 0
        obj_name = names[object_id[i]]

        if id not in counted_vehicles:
            counted_vehicles[id] = {'North': False, 'South': False}

        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)

        color = compute_color_for_labels(object_id[i])
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                if "South" in direction and not counted_vehicles[id]['South']:
                    vehicle_out[obj_name] = vehicle_out.get(obj_name, 0) + 1
                    counted_vehicles[id]['South'] = True
                    data_queue.put({"in": vehicle_in.copy(), "out": vehicle_out.copy()})
                elif "North" in direction and not counted_vehicles[id]['North']:
                    vehicle_in[obj_name] = vehicle_in.get(obj_name, 0) + 1
                    counted_vehicles[id]['North'] = True
                    data_queue.put({"in": vehicle_in.copy(), "out": vehicle_out.copy()})

        UI_box(box, img, label=label, color=color, line_thickness=2)

        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

    # Display counts
    for idx, (key, value) in enumerate(vehicle_in.items()):
        cnt_str = str(key) + ":" + str(value)
        cv2.line(img, (width - 200, 25), (width, 25), [85, 45, 255], 40)
        cv2.putText(img, f'Number of Vehicles Entering', (width - 1000, 35), 0, 1, 
                   [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (width - 50, 65 + (idx * 40)), (width, 65 + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(img, cnt_str, (width - 550, 75 + (idx * 40)), 0, 1, 
                   [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    for idx, (key, value) in enumerate(vehicle_out.items()):
        cnt_str1 = str(key) + ":" + str(value)
        cv2.line(img, (20, 25), (500, 25), [85, 45, 255], 40)
        cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, 
                   [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (20, 65 + (idx * 40)), (127, 65 + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(img, cnt_str1, (11, 75 + (idx * 40)), 0, 1, 
                   [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    return img


def process_and_send_data():
    """Process vehicle data and send to Firebase every 10 minutes"""
    global vehicle_in, vehicle_out, counted_vehicles
    
    try:
        vehicle_in = {}
        vehicle_out = {}
        counted_vehicles = {}

        if not data_queue.empty():
            all_vehicle_data = {"in": {}, "out": {}}
            queue_size = data_queue.qsize()
            
            for _ in range(queue_size):
                vehicle_data = data_queue.get()
                for direction in ["in", "out"]:
                    for key, value in vehicle_data[direction].items():
                        all_vehicle_data[direction][key] = all_vehicle_data[direction].get(key, 0) + value

            if all_vehicle_data["in"] or all_vehicle_data["out"]:
                results = process_vehicle_data(all_vehicle_data, capacity=1325)
                firebase_data = {
                    "entering": results['in'].get('vehicle_class_counts'),
                    "leaving": results['out'].get('vehicle_class_counts'),
                    "LOS": {"in": results['in'].get('LOS'), "out": results['out'].get('LOS')},
                    "DS": {"in": results['in'].get('DS'), "out": results['out'].get('DS')},
                    "location": "Jl. Setiabudi",
                    "timestamp": datetime.now().isoformat()
                }
                
                if firebase_enabled and ref:
                    ref.child('current_data').set(firebase_data)
                    log_ref.push(firebase_data)
                    print(f"✅ Data sent to Firebase at: {datetime.now()}")
                else:
                    with open("vehicle_data_stream.json", "w") as f:
                        json.dump(firebase_data, f, indent=2)
                    print(f"💾 Data saved locally at: {datetime.now()}")
            else:
                print(f"ℹ️  No vehicle data to process at: {datetime.now()}")
        else:
            print(f"ℹ️  Queue empty at: {datetime.now()}")
            
    except Exception as e:
        print(f"❌ Error processing data: {e}")
    finally:
        threading.Timer(600, process_and_send_data).start()


def run_detection(stream_url, model_path, conf_threshold, line_y, save_video):
    """Main detection loop using modern YOLO API"""
    global line, vehicle_in, vehicle_out, counted_vehicles
    
    # Update line position
    line = [(100, line_y), (1050, line_y)]
    
    # Initialize tracker
    init_tracker()
    
    # Load YOLO model
    print(f"🤖 Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Try downloading the model first:")
        print("   yolo download yolov8n")
        return
    
    # Open video stream
    print(f"📡 Opening stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"❌ Failed to open stream: {stream_url}")
        print("💡 Make sure the web streaming server is running:")
        print("   python web_stream.py")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    
    print(f"✅ Stream opened: {width}x{height} @ {fps}fps")
    
    # Video writer if saving
    writer = None
    if save_video:
        output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Saving video to: {output_path}")
    
    print(f"🎯 Confidence threshold: {conf_threshold}")
    print("-" * 60)
    print("Press 'q' to quit, 'p' to pause")
    print("-" * 60)
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Stream ended or error")
                time.sleep(1)
                continue
            
            # Run YOLO detection
            results = model(frame, conf=conf_threshold)
            
            # Process detections
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
                
                if len(xywh_bboxs) > 0:
                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)
                    
                    outputs = deepsort.update(xywhs, confss, oids, frame)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -2]
                        object_id = outputs[:, -1]
                        draw_boxes(frame, bbox_xyxy, model.names, object_id, identities)
            
            # Save frame if recording
            if writer:
                writer.write(frame)
            
            # Display
            cv2.imshow('Traffic Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'▶️' if not paused else '⏸️'} {'Resumed' if not paused else 'Paused'}")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Detection stopped")
    print(f"📊 Vehicles entering: {vehicle_in}")
    print(f"📊 Vehicles leaving: {vehicle_out}")


def parse_opt():
    parser = argparse.ArgumentParser(description='Vehicle Detection on Web Stream')
    parser.add_argument('--stream-url', type=str, default=STREAM_URL,
                       help='URL of the video stream (default: http://127.0.0.1:8080/video_feed)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLOv8 model path (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--line-y', type=int, default=500,
                       help='Y-coordinate of counting line (default: 500)')
    parser.add_argument('--no-firebase', action='store_true',
                       help='Disable Firebase integration')
    parser.add_argument('--save', action='store_true',
                       help='Save output video')
    return parser.parse_args()


def main():
    opt = parse_opt()
    
    print("=" * 60)
    print("🚦 Traffic Congestion Analysis - Stream Detection")
    print("=" * 60)
    
    # Initialize Firebase (optional)
    if not opt.no_firebase:
        init_firebase()
    
    # Start data processing thread
    threading.Thread(target=process_and_send_data, daemon=True).start()
    
    # Run detection
    try:
        run_detection(
            stream_url=opt.stream_url,
            model_path=opt.model,
            conf_threshold=opt.conf,
            line_y=opt.line_y,
            save_video=opt.save
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
