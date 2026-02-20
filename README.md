# 🚦 Smart Traffic Management System

A comprehensive traffic monitoring and analysis system similar to Semarang's CCTV traffic portal. Features real-time vehicle detection, congestion analysis, and interactive map visualization.

![System Architecture](docs/architecture.png)

## 🌟 Key Features

### 1. 🗺️ Interactive CCTV Map (Main Page)
- **Map-based dashboard** showing all CCTV locations
- **Color-coded markers** indicating real-time traffic status
- **Click on any CCTV** to open popup with live stream
- **Live vehicle detection** shown directly in popup
- **Real-time statistics**: vehicle count, congestion level, LOS

### 2. 📹 Multi-Stream Video Processing
- **Process multiple CCTVs simultaneously**
- **Background detection** using YOLOv8 + DeepSORT
- **Real-time counting** of vehicles crossing virtual lines
- **Vehicle classification**: Cars, Motorcycles, Buses, Trucks
- **Independent tracking** for each CCTV stream

### 3. 📊 Traffic Analysis Engine
- **Vehicles per minute (VPM)** calculation
- **Congestion level estimation**:
  - 🟢 **Free Flow**: < 10 v/min (LOS A-B)
  - 🟡 **Moderate**: 10-30 v/min (LOS C)
  - 🟠 **Congested**: 30-60 v/min (LOS D-E)
  - 🔴 **Severe**: > 60 v/min (LOS F)
- **Level of Service (LOS)** calculation
- **Historical data** storage and trend analysis

### 4. 🛣️ Road Network Visualization
- **Separate traffic status page**
- **Road segments** with color-coded congestion
- **Spatial database** integration (SQLite/PostGIS)
- **Network-wide statistics** and summaries
- **Congestion distribution** charts

### 5. 🔄 Real-time Updates
- **WebSocket** for live data updates
- **No page refresh** required
- **Synchronized** map and list views
- **Automatic data persistence** to database

---

## 📁 Project Structure

```
traffic_congestion/
├── app.py                      # Main Flask application
├── database/
│   ├── __init__.py
│   ├── models.py              # Data models
│   └── db_manager.py          # Database operations
├── templates/
│   ├── map_dashboard.html     # CCTV map page
│   └── traffic_status.html    # Road network page
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── deep_sort_pytorch/         # DeepSORT tracking
├── requirements.txt           # Dependencies
├── setup.sh                   # Setup script
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam or video files for testing
- (Optional) IP Camera streams (RTSP/HTTP)

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download YOLO Model

```bash
# Download YOLOv8n (fast, recommended for testing)
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. Run the System

```bash
python app.py
```

The system will start on `http://127.0.0.1:5005`

### 4. Access the Dashboard

1. **Main Page** (CCTV Map): http://127.0.0.1:5005
2. **Traffic Status** (Road Network): http://127.0.0.1:5005/traffic-status

---

## 📖 Usage Guide

### Adding CCTV Cameras

The system comes with 5 demo CCTVs in Semarang area. To add more:

**Via API:**
```bash
curl -X POST http://127.0.0.1:5005/api/cctvs \
  -H "Content-Type: application/json" \
  -d '{
    "id": "cctv_new",
    "name": "New Location",
    "latitude": -6.9900,
    "longitude": 110.4200,
    "stream_url": "0"
  }'
```

**Stream URL formats:**
- Webcam: `0` or `1`
- Video file: `/path/to/video.mp4`
- RTSP: `rtsp://user:pass@ip:554/stream`
- HTTP: `http://ip:8080/video`
- MJPEG stream: `http://ip:8080/video_feed`

### Understanding the Dashboard

#### CCTV Map Page
| Element | Description |
|---------|-------------|
| 🟢 Green marker | Free flow traffic |
| 🟡 Yellow marker | Moderate traffic |
| 🟠 Orange marker | Congested traffic |
| 🔴 Red marker | Severe congestion |
| Click marker | Open live stream popup |
| Sidebar | List of all CCTVs with stats |

#### Traffic Status Page
| Metric | Description |
|--------|-------------|
| Vehicles/Min | Average vehicles per minute |
| LOS | Level of Service (A=best, F=worst) |
| Congestion | Free/Moderate/Congested/Severe |
| System LOS | Overall network health |

---

## ⚙️ Configuration

### Environment Variables

```bash
# YOLO Model (default: yolov8n.pt)
export YOLO_MODEL=yolov8l.pt

# Database path (default: database/traffic.db)
export DB_PATH=database/traffic.db

# Detection interval (seconds)
export DETECTION_INTERVAL=10

# Congestion thresholds (vehicles/min)
export THRESHOLD_FREE=10
export THRESHOLD_MODERATE=30
export THRESHOLD_CONGESTED=60
```

### Traffic Analysis Parameters

Edit in `app.py`:

```python
# Line position for counting (Y coordinate)
line_y = height // 2

# Processing frame skip (every Nth frame)
process_every_n_frames = 3

# Data aggregation interval (seconds)
aggregation_interval = 10
```

---

## 🔧 API Endpoints

### CCTV Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cctvs` | GET | List all CCTVs |
| `/api/cctvs` | POST | Add new CCTV |
| `/api/cctvs/<id>/start` | POST | Start detection |
| `/api/cctvs/<id>/stop` | POST | Stop detection |
| `/api/cctvs/<id>/status` | GET | Get CCTV status |

### Traffic Data
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/traffic/status` | GET | Get all traffic data |
| `/api/traffic/roads` | GET | Get road segments with traffic |

### Streaming
| Endpoint | Description |
|----------|-------------|
| `/stream/<cctv_id>` | MJPEG stream for CCTV |

### WebSocket Events
| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client → Server | Client connected |
| `init_data` | Server → Client | Initial data load |
| `traffic_update` | Server → Client | Real-time traffic update |

---

## 🗄️ Database Schema

### Tables

**cctvs**: CCTV camera information
```sql
- id (TEXT PRIMARY KEY)
- name (TEXT)
- latitude (REAL)
- longitude (REAL)
- stream_url (TEXT)
- road_segment_id (TEXT)
- status (TEXT)
- created_at (TIMESTAMP)
```

**road_segments**: Road network geometry
```sql
- id (TEXT PRIMARY KEY)
- name (TEXT)
- road_type (TEXT)
- coordinates (JSON)
- speed_limit (INTEGER)
- capacity (INTEGER)
```

**traffic_data**: Historical traffic measurements
```sql
- id (INTEGER PRIMARY KEY)
- cctv_id (TEXT)
- road_segment_id (TEXT)
- timestamp (TIMESTAMP)
- vehicle_count (INTEGER)
- vehicles_per_minute (REAL)
- cars, motorcycles, buses, trucks (INTEGER)
- congestion_level (TEXT)
- los (TEXT)
```

---

## 🎓 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  CCTV Map    │  │ Traffic      │  │   Stream     │      │
│  │  (Leaflet)   │  │ Status       │  │   Popups     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼────────────────┼────────────────┼────────────────┘
          │                │                │
          └────────────────┴────────────────┘
                           │
                    WebSocket (SocketIO)
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                    Flask Backend                             │
│  ┌──────────────┐  ┌──────┴───────┐  ┌──────────────┐      │
│  │   CCTV API   │  │   Traffic    │  │   Stream     │      │
│  │   Routes     │  │   Analysis   │  │   Handler    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼────────────────┼────────────────┼────────────────┘
          │                │                │
          └────────────────┴────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│               Detection Workers (Threading)                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ CCTV 1  │ │ CCTV 2  │ │ CCTV 3  │ │  ...    │           │
│  │ Worker  │ │ Worker  │ │ Worker  │ │ Worker  │           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
└───────┼───────────┼───────────┼───────────┼─────────────────┘
        │           │           │           │
        └───────────┴───────────┴───────────┘
                    │
        ┌───────────┴───────────┐
        │     YOLOv8 + DeepSORT  │
        │   Vehicle Detection    │
        └────────────────────────┘
```

---

## ⚡ Performance Optimization

### For Multiple Streams (5+ CCTVs)

1. **Use GPU Acceleration**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

2. **Reduce Processing Frequency**
```python
# Process every 5th frame instead of every 3rd
process_every_n_frames = 5
```

3. **Lower Resolution**
```python
# Resize frames before detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

4. **Use Lighter Model**
```python
# Use nano model instead of large
model = YOLO('yolov8n.pt')  # Instead of yolov8l.pt
```

### Hardware Recommendations

| Streams | CPU | RAM | GPU | Network |
|---------|-----|-----|-----|---------|
| 1-3 | 4 cores | 4GB | Optional | 10 Mbps |
| 4-8 | 8 cores | 8GB | GTX 1060+ | 50 Mbps |
| 9-16 | 16 cores | 16GB | RTX 3060+ | 100 Mbps |

---

## 🔌 Integrating with Semarang CCTV Portal

To integrate with the actual Semarang CCTV system:

1. **Scrape CCTV URLs** from https://pantausemar.semarangkota.go.id/
2. **Add to system via API**:
```python
import requests

# Add each CCTV
for cctv in semarang_cctvs:
    requests.post('http://localhost:5000/api/cctvs', json={
        'id': cctv['id'],
        'name': cctv['name'],
        'latitude': cctv['lat'],
        'longitude': cctv['lng'],
        'stream_url': cctv['stream_url']
    })
```

---

## 🐛 Troubleshooting

### "Failed to open stream"
- Check stream URL is accessible
- For RTSP: Verify credentials and codec
- Try opening with VLC first

### High CPU Usage
- Enable GPU acceleration
- Reduce number of streams
- Increase `process_every_n_frames`

### "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Map not loading
- Check internet connection (Leaflet CDN)
- Verify browser console for errors

---

## 📸 Screenshots

### CCTV Map Dashboard
![Map Dashboard](docs/screenshot-map.png)

### Traffic Status Page
![Traffic Status](docs/screenshot-traffic.png)

### Live Stream Popup
![Stream Popup](docs/screenshot-popup.png)

---

## 📜 License

MIT License - See LICENSE file

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- DeepSORT by nwojke
- Leaflet.js for mapping
- Flask and SocketIO for real-time communication
