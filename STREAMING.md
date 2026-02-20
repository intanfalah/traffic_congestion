# Web Streaming for Traffic Congestion Analysis

This feature provides a web-based interface for viewing and managing video streams used as input for vehicle detection.

## Features

- 🎥 **Live Video Streaming** - View camera feeds in your browser
- 📊 **Real-time Status** - Monitor FPS, connection status, and stream health
- 🎛️ **Multiple Sources** - Support for webcams, IP cameras (RTSP), and HTTP streams
- 🔌 **Easy Integration** - Use the stream URL directly in the detection system
- 📱 **Responsive UI** - Works on desktop and mobile devices

## Quick Start

### 1. Start the Web Streaming Server

```bash
python web_stream.py
```

The server will start on `http://127.0.0.1:5005`

### 2. Open the Web Interface

Navigate to `http://127.0.0.1:8080` in your browser.

### 3. Configure Video Source

- Select a source from the dropdown (default: webcam)
- Or add a new source (RTSP URL, HTTP stream, or device ID)
- Click "Start Stream"

### 4. Run Vehicle Detection

In a new terminal:

```bash
# Using default stream URL
python predict_stream.py

# Or specify custom stream URL
python predict_stream.py --stream-url http://127.0.0.1:8080/video_feed
```

## Usage Examples

### Using Webcam

```bash
# Default webcam (device 0)
python predict_stream.py --stream-url http://127.0.0.1:5005/video_feed
```

### Using IP Camera (RTSP)

1. Add the RTSP URL in the web interface:
   - Name: `IP Camera`
   - URL: `rtsp://username:password@192.168.1.100:554/stream`

2. Run detection:
```bash
python predict_stream.py
```

### Using Video File

The web stream server can also stream video files:

1. Add source:
   - Name: `Test Video`
   - URL: `/path/to/traffic.mp4`

2. Run detection:
```bash
python predict_stream.py
```

## API Endpoints

### Stream Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/video_feed` | GET | MJPEG video stream (use this for detection) |
| `/api/stream/start` | POST | Start stream from source |
| `/api/stream/stop` | POST | Stop current stream |
| `/api/stream/status` | GET | Get stream status |

### Source Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sources` | GET | List all sources |
| `/api/sources` | POST | Add new source |
| `/api/sources/<id>` | DELETE | Delete a source |

### Example API Usage

```bash
# Start stream from source
curl -X POST http://127.0.0.1:5005/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{"source": 0, "name": "Webcam"}'

# Check status
curl http://127.0.0.1:5005/api/stream/status

# Add IP camera
curl -X POST http://127.0.0.1:5005/api/sources \
  -H "Content-Type: application/json" \
  -d '{"id": "ipcam1", "name": "Main Road", "url": "rtsp://192.168.1.100/stream"}'
```

## Configuration

Edit `stream_config.yaml` to customize default settings:

```yaml
stream_server:
  host: "0.0.0.0"
  port: 8080

sources:
  camera1:
    url: 0
    name: "Default Webcam"
    active: true

detection:
  model: "yolov8n.pt"
  confidence: 0.3
  
line:
  y_position: 500
```

## Command Line Options (predict_stream.py)

```bash
python predict_stream.py [OPTIONS]

Options:
  --stream-url TEXT    Stream URL (default: http://127.0.0.1:8080/video_feed)
  --model TEXT         YOLOv8 model path (default: yolov8n.pt)
  --conf FLOAT         Confidence threshold (default: 0.3)
  --imgsz INT          Inference size (default: 640)
  --line-y INT         Counting line Y position (default: 500)
  --no-firebase        Disable Firebase integration
  --save               Save output video
  -h, --help           Show help message
```

## Architecture

```
┌─────────────────┐      HTTP      ┌─────────────┐
│   Web Browser   │◄──────────────►│ Web Server  │
│   (View Stream) │   MJPEG Stream │ (Flask)     │
└─────────────────┘                └──────┬──────┘
                                          │
                     ┌────────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  /video_feed    │
            │  MJPEG Stream   │
            └────────┬────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   predict_stream.py │
          │  (YOLOv8 + DeepSORT)│
          │   Vehicle Detection │
          └──────────┬──────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   ┌─────────────┐      ┌──────────────┐
   │   Firebase  │      │ Local Files  │
   │   Database  │      │ (JSON/Video) │
   └─────────────┘      └──────────────┘
```

## Troubleshooting

### Stream not loading

1. Check if the stream URL is correct
2. Verify the source is accessible
3. Check browser console for errors

### Low FPS

1. Reduce inference size: `--imgsz 416`
2. Use lighter model: `--model yolov8n.pt`
3. Check network bandwidth for IP cameras

### Detection not starting

1. Ensure web stream server is running
2. Check stream is active: `curl http://127.0.0.1:5005/api/stream/status`
3. Verify stream URL in detection command

### Firebase connection failed

1. Place `firebase-key.json` in project root
2. Or set environment variable: `export FIREBASE_CRED_PATH=/path/to/key.json`
3. Use `--no-firebase` to run without Firebase

## Browser Compatibility

- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+

## Notes

- The web stream uses MJPEG format which is widely supported
- For production, consider adding authentication to the web interface
- RTSP streams may have higher latency than HTTP streams
- GPU acceleration is recommended for real-time processing
