import cv2
import time
import os

# Try setting environment variable to force ffmpeg backend options (sometimes helps with HLS/HTTPS)
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp" # Not for HLS but maybe helpful?
# Or just try standard first.

url = "https://livepantau.semarangkota.go.id/3cc2431b-3ee5-4c91-8330-251c021cd510/video1_stream.m3u8"

print(f"Attempting to open stream: {url}")
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ Failed to open stream!")
    # Try with explicit backend if available
    try:
        print("Retrying with CAP_FFMPEG...")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    except:
        pass

if cap.isOpened():
    print("✅ Stream opened successfully!")
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame captured! Size: {frame.shape}")
    else:
        print("❌ Stream opened but failed to read frame.")
    cap.release()
else:
    print("❌ Failed to open stream even with backend hint.")
    
# Check build info
print("\nOpenCV Build Information:")
print(cv2.getBuildInformation().split('\n')[0:20]) # First 20 lines usually show backends
