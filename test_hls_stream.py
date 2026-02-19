#!/usr/bin/env python3
"""
Test HLS Stream Reader
Verify we can read the Semarang CCTV streams
"""

import cv2
import sys

# Test stream URLs
TEST_STREAMS = [
    ("Indraprasta Imam Bonjol", "https://livepantau.semarangkota.go.id/3cc2431b-3ee5-4c91-8330-251c021cd510/video1_stream.m3u8"),
    ("Kaligarang", "https://livepantau.semarangkota.go.id/e9203185-ee2e-4eb0-83a4-46b80c3bcc1a/video1_stream.m3u8"),
    ("Kalibanteng 2", "https://livepantau.semarangkota.go.id/b216444c-25db-4be2-bb30-fcb044f7c83f/video1_stream.m3u8"),
]

def test_stream(name, url, timeout=10):
    """Test reading from a single stream"""
    print(f"\n📹 Testing: {name}")
    print(f"   URL: {url[:70]}...")
    
    try:
        # Set FFmpeg options for HLS streams
        import os
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000'
        
        # Open stream
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"   ❌ Failed to open stream")
            return False
        
        print(f"   ✅ Stream opened")
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print(f"   ❌ Failed to read frame")
            cap.release()
            return False
        
        height, width = frame.shape[:2]
        print(f"   ✅ Frame read successfully: {width}x{height}")
        
        # Try reading a few more frames
        for i in range(4):
            ret, _ = cap.read()
            if not ret:
                print(f"   ⚠️  Frame {i+2} failed")
                break
        else:
            print(f"   ✅ Multiple frames read successfully")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("=" * 70)
    print("🎥 HLS STREAM TEST - Semarang CCTVs")
    print("=" * 70)
    
    # Check FFmpeg
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"\n✅ FFmpeg found: {version[:50]}")
        else:
            print(f"\n⚠️  FFmpeg found but returned error")
    except FileNotFoundError:
        print("\n❌ FFmpeg NOT FOUND - Required for HLS streams")
        print("   Install with: brew install ffmpeg")
        return 1
    
    # Test each stream
    results = []
    for name, url in TEST_STREAMS:
        ok = test_stream(name, url)
        results.append((name, ok))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    success = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"   {status} - {name}")
    
    print(f"\n   Total: {success}/{total} streams working")
    
    if success == 0:
        print("\n⚠️  No streams working - check:")
        print("   1. Internet connection")
        print("   2. FFmpeg installation")
        print("   3. Stream URLs are accessible")
        return 1
    elif success < total:
        print("\n⚠️  Some streams not working - may be temporary")
        return 0
    else:
        print("\n✅ All streams working! Ready to run.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
