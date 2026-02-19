# Multi-Stream Vehicle Detection

This feature allows you to process **multiple video feeds simultaneously** from different cameras or sources.

## Features

- 🎥 **Multiple Streams** - Process unlimited cameras simultaneously
- 📊 **Independent Tracking** - Each stream has its own DeepSORT tracker
- 🔄 **Parallel Processing** - Each stream runs in its own thread
- 📈 **Aggregated Data** - Combined traffic metrics from all streams
- 💻 **Web Dashboard** - View all streams in a single web interface

---

## Quick Start

### Option 1: Multi-Stream Web Server (View Only)

1. **Start the multi-stream server:**
```bash
python web_stream_multi.py
```

2. **Open browser:**
```
http://127.0.0.1:8080
```

3. **Add streams via web interface:**
   - Stream ID: `camera1`
   - Name: `Main Road`
   - Source: `0` (webcam) or `rtsp://...`

### Option 2: Multi-Stream Detection (Process Multiple Feeds)

Run detection on multiple streams simultaneously:

```bash
# Example: 3 cameras
python predict_multi_stream.py \
    --streams "http://127.0.0.1:8080/video_feed/camera1" \
    --streams "http://127.0.0.1:8080/video_feed/camera2" \
    --streams "rtsp://192.168.1.100:554/stream" \
    --names "Main Entrance" "Highway North" "Downtown" \
    --display
```

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Multi-Stream Server             │
                    │        (web_stream_multi.py)            │
                    └─────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
   ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
   │  Camera 1    │        │  Camera 2    │        │  Camera N    │
   │  /video_feed │        │  /video_feed │        │  /video_feed │
   └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────────────┐
                    │     Multi-Stream Detection              │
                    │     (predict_multi_stream.py)           │
                    │                                         │
                    │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
                    │  │Thread 1 │ │Thread 2 │ │Thread N │   │
                    │  │DeepSORT │ │DeepSORT │ │DeepSORT │   │
                    │  └────┬────┘ └────┬────┘ └────┬────┘   │
                    │       └───────────┼───────────┘         │
                    │                   ▼                     │
                    │         ┌─────────────────┐             │
                    │         │  Combined Data  │             │
                    │         │    Firebase     │             │
                    │         └─────────────────┘             │
                    └─────────────────────────────────────────┘
```

---

## Usage Examples

### Example 1: Multiple Webcams

```bash
python predict_multi_stream.py \
    --streams "0" \
    --streams "1" \
    --streams "2" \
    --names "Camera Front" "Camera Back" "Camera Side" \
    --display
```

### Example 2: Mixed Sources

```bash
python predict_multi_stream.py \
    --streams "0" \
    --streams "rtsp://admin:pass@192.168.1.100:554/live" \
    --streams "http://192.168.1.101:8080/video" \
    --streams "/path/to/video.mp4" \
    --names "Webcam" "IP Cam 1" "IP Cam 2" "Recording" \
    --model yolov8n.pt
```

### Example 3: Using Web Stream Server

First, start the web server and add cameras via web UI, then:

```bash
python predict_multi_stream.py \
    --streams "http://127.0.0.1:8080/video_feed/camera1" \
    --streams "http://127.0.0.1:8080/video_feed/camera2" \
    --streams "http://127.0.0.1:8080/video_feed/camera3" \
    --display
```

---

## API Endpoints (Multi-Stream Server)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Multi-stream dashboard |
| `/video_feed/<stream_id>` | GET | MJPEG feed for specific camera |
| `/api/streams` | GET | List all streams |
| `/api/streams` | POST | Add new stream |
| `/api/streams/<id>` | DELETE | Remove stream |
| `/api/streams/<id>/stop` | POST | Stop stream |
| `/api/streams/<id>/start` | POST | Restart stream |

### API Examples

```bash
# Add a stream
curl -X POST http://127.0.0.1:8080/api/streams \
  -H "Content-Type: application/json" \
  -d '{
    "id": "camera1",
    "name": "Main Road",
    "source": "rtsp://192.168.1.100/stream"
  }'

# List all streams
curl http://127.0.0.1:8080/api/streams

# Delete a stream
curl -X DELETE http://127.0.0.1:8080/api/streams/camera1
```

---

## Configuration File

Use `multi_stream_config.yaml` to define streams:

```yaml
streams:
  - id: camera1
    name: "Main Entrance"
    url: 0
    line_y: 500
    
  - id: camera2
    name: "Highway North"
    url: "rtsp://192.168.1.100:554/stream"
    line_y: 400
    
  - id: camera3
    name: "Downtown"
    url: "http://192.168.1.101:8080/video"
    line_y: 600

detection:
  model: "yolov8n.pt"
  confidence: 0.3
```

---

## Performance Considerations

### CPU Usage
- Each stream uses ~1 CPU core for detection
- 4 streams = ~4 cores recommended
- GPU acceleration highly recommended for 3+ streams

### Memory Usage
- Each stream: ~500MB - 1GB RAM
- 4 streams: ~2-4GB RAM recommended

### Network Bandwidth
- Each RTSP stream: ~2-4 Mbps
- 4 streams: ~8-16 Mbps network capacity needed

### Optimization Tips

1. **Use GPU**: Add CUDA support for real-time processing
2. **Lower resolution**: Use `--imgsz 416` for faster processing
3. **Reduce FPS**: Process every Nth frame for slow-moving traffic
4. **Skip display**: Remove `--display` for headless operation

---

## Command Line Options

```bash
python predict_multi_stream.py [OPTIONS]

Options:
  --streams STR ...     List of stream URLs (required)
  --names STR ...       Names for each stream
  --model STR           YOLO model path (default: yolov8n.pt)
  --conf FLOAT          Confidence threshold (default: 0.3)
  --line-y INT          Counting line Y position (default: 500)
  --display             Show video windows
  --no-firebase         Disable Firebase
```

---

## Firebase Data Structure

Multi-stream data is saved with this structure:

```json
{
  "multi_stream_data": {
    "streams": {
      "stream_1": {
        "entering": {"MC": 10, "LV": 5, "HV": 2},
        "leaving": {"MC": 8, "LV": 3, "HV": 1},
        "LOS": {"in": "B", "out": "A"},
        "DS": {"in": 0.45, "out": 0.32}
      },
      "stream_2": { ... },
      "stream_3": { ... }
    },
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

---

## Troubleshooting

### "Failed to open stream"
- Check stream URL is correct
- Verify network connectivity
- For RTSP: Check credentials and codec support

### High CPU usage
- Use GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Reduce number of simultaneous streams
- Use lighter model: `yolov8n.pt` instead of `yolov8l.pt`

### Lag/Delay in streams
- Reduce buffer size in camera settings
- Use wired network instead of WiFi
- Lower stream resolution

### Window display issues
- macOS: Use `export DISPLAY=:0` if using X11
- Linux: May need `sudo apt install libgl1-mesa-glx`

---

## Files Overview

| File | Purpose |
|------|---------|
| `web_stream_multi.py` | Multi-stream web server |
| `predict_multi_stream.py` | Multi-stream detection |
| `templates/multi_stream.html` | Web dashboard UI |
| `multi_stream_config.yaml` | Stream configuration |
| `MULTI_STREAM.md` | This documentation |

---

## Next Steps

1. **Test with single stream first**
2. **Add streams one by one**
3. **Monitor resource usage**
4. **Optimize for your hardware**

For questions, see main documentation in [STREAMING.md](STREAMING.md) and [SETUP.md](SETUP.md).
