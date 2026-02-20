#!/usr/bin/env python3
"""
Test Single CCTV
Run with just one CCTV to verify everything works
"""

import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000|max_delay;5000000'

import cv2
import time
import threading
from ultralytics import YOLO
import torch

# Test with first CCTV
TEST_CCTV = {
    "id": "test_cctv",
    "name": "Indraprasta Imam Bonjol",
    "stream_url": "https://livepantau.semarangkota.go.id/3cc2431b-3ee5-4c91-8330-251c021cd510/video1_stream.m3u8"
}

class SimpleDetector:
    """Simple single-threaded detector"""
    
    def __init__(self):
        self.model = None
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.vehicle_count = 0
        
    def run(self):
        """Main loop"""
        print("=" * 70)
        print(f"🎥 Testing: {TEST_CCTV['name']}")
        print(f"   URL: {TEST_CCTV['stream_url'][:60]}...")
        print("=" * 70)
        
        # Load model
        print("\n[1/4] Loading YOLO model...")
        try:
            self.model = YOLO('yolov8n.pt')
            print("   ✅ Model loaded")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Open stream
        print("\n[2/4] Opening stream...")
        self.cap = cv2.VideoCapture(TEST_CCTV['stream_url'], cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            print("   ❌ Failed to open stream")
            return False
        
        # Get stream info
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"   ✅ Stream opened: {width}x{height} @ {fps}fps")
        
        # Process frames
        print("\n[3/4] Processing frames (press 'q' to quit)...")
        print("   Waiting for first frame...")
        
        self.running = True
        start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("   ⚠️  Frame read failed, retrying...")
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            
            # Process every 5th frame
            if self.frame_count % 5 == 0:
                frame = self.process_frame(frame)
            
            # Show frame
            cv2.imshow('CCTV Test', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("   User quit")
                break
            
            # Print stats every 5 seconds
            elapsed = time.time() - start_time
            if elapsed >= 5 and self.frame_count % 30 == 0:
                fps = self.frame_count / elapsed
                print(f"   FPS: {fps:.1f}, Vehicles: {self.vehicle_count}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        elapsed = time.time() - start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print("\n[4/4] Summary")
        print("=" * 70)
        print(f"   Total frames: {self.frame_count}")
        print(f"   Runtime: {elapsed:.1f} seconds")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Vehicles detected: {self.vehicle_count}")
        
        if avg_fps > 5:
            print("   ✅ Performance: GOOD")
        elif avg_fps > 2:
            print("   ⚠️  Performance: ACCEPTABLE")
        else:
            print("   ❌ Performance: POOR")
        
        return True
    
    def process_frame(self, frame):
        """Process frame with YOLO"""
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
            
            # Draw info
            cv2.putText(frame, f"Vehicles: {count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Total: {self.vehicle_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"   Processing error: {e}")
        
        return frame

def main():
    detector = SimpleDetector()
    success = detector.run()
    
    if success:
        print("\n✅ Single CCTV test passed!")
        print("   You can now run with all 3 CCTVs")
    else:
        print("\n❌ Test failed")
        print("   Check: 1) FFmpeg installed, 2) Network connection, 3) Stream URL")

if __name__ == "__main__":
    main()
