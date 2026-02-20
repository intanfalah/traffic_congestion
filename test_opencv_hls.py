#!/usr/bin/env python3
"""
Test OpenCV HLS Stream Reading
Verify FFmpeg backend is working
"""

import os
# Must set before importing cv2
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000|max_delay;5000000'

import cv2
import sys

# Test stream
TEST_URL = "https://livepantau.semarangkota.go.id/3cc2431b-3ee5-4c91-8330-251c021cd510/video1_stream.m3u8"

def test_stream():
    print("=" * 70)
    print("🎥 Testing OpenCV HLS Stream Reading")
    print("=" * 70)
    
    print(f"\nFFmpeg options set: {os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'NOT SET')}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"\nTesting stream: {TEST_URL[:60]}...")
    
    # Open stream
    print("\n1. Opening stream with FFmpeg backend...")
    cap = cv2.VideoCapture(TEST_URL, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("   ❌ Failed to open with FFmpeg")
        print("\n2. Trying default backend...")
        cap = cv2.VideoCapture(TEST_URL)
        
        if not cap.isOpened():
            print("   ❌ Failed to open with default backend")
            print("\n⚠️  POSSIBLE CAUSES:")
            print("   - FFmpeg not installed: brew install ffmpeg")
            print("   - OpenCV not built with FFmpeg support")
            print("   - Network/firewall blocking stream")
            print("   - Stream URL invalid")
            return False
    
    print("   ✅ Stream opened successfully")
    
    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n3. Stream properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    
    # Read frames
    print("\n4. Reading frames...")
    frames_read = 0
    max_frames = 10
    
    while frames_read < max_frames:
        ret, frame = cap.read()
        if ret:
            frames_read += 1
            if frames_read == 1:
                print(f"   ✅ First frame read: {frame.shape}")
        else:
            print(f"   ❌ Failed to read frame {frames_read + 1}")
            break
    
    cap.release()
    
    print(f"\n5. Summary:")
    print(f"   Frames read: {frames_read}/{max_frames}")
    
    if frames_read > 0:
        print("\n   ✅ OpenCV can read HLS streams!")
        print("   You can now run: python app.py")
        return True
    else:
        print("\n   ❌ Could not read any frames")
        return False

if __name__ == "__main__":
    success = test_stream()
    sys.exit(0 if success else 1)
