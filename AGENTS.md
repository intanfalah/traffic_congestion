# Traffic Congestion Analysis System

## Project Overview

This is a **Traffic Congestion Analysis System** that uses computer vision and deep learning to detect, track, and analyze vehicle traffic in real-time. The system combines YOLOv8 object detection with DeepSORT tracking to count vehicles, classify them by type, and calculate traffic congestion metrics (Level of Service/LOS and Degree of Saturation/DS).

### Core Functionality

- **Vehicle Detection**: Uses YOLOv8 (Ultralytics) to detect vehicles in video streams
- **Object Tracking**: Implements DeepSORT algorithm to maintain consistent vehicle IDs across frames
- **Direction Counting**: Tracks vehicles crossing a virtual line to count "entering" (North) and "leaving" (South) traffic
- **Vehicle Classification**: Groups vehicles into 3 categories:
  - **MC** (Motorcycle): motorcycle, person
  - **LV** (Light Vehicle): car, pickup, van
  - **HV** (Heavy Vehicle): bus, truck, trailer
- **Traffic Metrics**: Calculates traffic flow (Q), degree of saturation (DS), and Level of Service (LOS A-F)
- **Real-time Dashboard**: Sends data to Firebase Realtime Database for monitoring

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv8 (Ultralytics) |
| Object Tracking | DeepSORT with PyTorch |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |
| Configuration | Hydra, YAML |
| Database | Firebase Realtime Database |
| Language | Python 3 |

---

## Project Structure

```
traffic_congestion/
├── predict.py                 # Main inference script - video processing & tracking
├── predict_stream.py          # Inference on web stream source
├── web_stream.py              # Flask web server for video streaming
├── stream_config.yaml         # Stream configuration
├── STREAMING.md               # Streaming feature documentation
├── templates/                 # HTML templates for web interface
│   └── index.html             # Main streaming page
├── train.py                   # YOLOv8 model training script
├── val.py                     # Model validation script
├── process_results.py         # Traffic metrics calculation (LOS, DS, Q)
├── firebase_config.py         # Firebase initialization (legacy)
├── vehicle_data.json          # Temporary vehicle count data storage
├── predict.log                # Application logs
├── yolov8n.pt                 # YOLOv8 nano pretrained weights
├── yolov8l.pt                 # YOLOv8 large pretrained weights
├── traffic.mp4                # Sample video for testing
├── __init__.py                # Package marker
├── python_to_database.py      # Empty placeholder file
└── deep_sort_pytorch/         # DeepSORT tracking implementation
    ├── configs/
    │   └── deep_sort.yaml     # DeepSORT configuration
    ├── deep_sort/
    │   ├── deep_sort.py       # Main DeepSORT tracker class
    │   ├── deep/              # Feature extraction (Re-ID model)
    │   │   ├── checkpoint/    # Model weights storage (ckpt.t7)
    │   │   ├── feature_extractor.py
    │   │   ├── model.py       # CNN architecture (ResNet-like)
    │   │   ├── train.py       # Re-ID model training
    │   │   └── test.py        # Re-ID model evaluation
    │   └── sort/              # SORT algorithm components
    │       ├── kalman_filter.py
    │       ├── linear_assignment.py
    │       ├── nn_matching.py
    │       ├── iou_matching.py
    │       ├── tracker.py
    │       ├── track.py
    │       └── detection.py
    ├── utils/
    │   ├── parser.py          # YAML config parser
    │   ├── draw.py            # Visualization utilities
    │   └── ...
    └── README.md
```

---

## Configuration

### DeepSORT Configuration (`deep_sort_pytorch/configs/deep_sort.yaml`)

```yaml
DEEPSORT:
  REID_CKPT: "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
  MAX_DIST: 0.2              # Maximum cosine distance for matching
  MIN_CONFIDENCE: 0.3        # Minimum detection confidence
  NMS_MAX_OVERLAP: 0.5       # NMS threshold
  MAX_IOU_DISTANCE: 0.7      # Maximum IOU distance
  MAX_AGE: 70                # Maximum frames to keep lost tracks
  N_INIT: 3                  # Frames to confirm a track
  NN_BUDGET: 100             # Feature budget per target
```

### Line Configuration (in `predict.py`)

The virtual counting line is defined in `predict.py`:
```python
line = [(100, 500), (1050, 500)]  # Coordinates for counting line
```

### Firebase Configuration

Firebase is configured in `predict.py` with hardcoded paths:
- Certificate path: `E:\ta\percobaan paling berhasil\firebase-key.json`
- Database URL: `https://traffic-vision-d32aa-default-rtdb.asia-southeast1.firebasedatabase.app`

**Note**: Update these paths for your environment before running.

---

## Build and Run Commands

### Prerequisites

**Python**: 3.8 or higher required

#### Option 1: Automated Setup (Recommended)

**macOS/Linux:**
```bash
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

This will create a virtual environment (`.venv`) and install all dependencies.

#### Option 2: Manual Setup

Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

See [SETUP.md](SETUP.md) for detailed setup instructions.

### Download Required Model Files

1. **YOLOv8 weights**: Already included (`yolov8n.pt`, `yolov8l.pt`)
2. **DeepSORT checkpoint**: Download `ckpt.t7` from the [original repository](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6) and place in:
   ```
   deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
   ```

### Running the Application

#### Main Inference (Real-time Traffic Analysis)

**Option 1: Direct Video Source**
```bash
python predict.py model=yolov8n.pt source=traffic.mp4
```

With custom video:
```bash
python predict.py model=yolov8n.pt source=/path/to/video.mp4
```

**Option 2: Web Stream Source (New)**

1. Start the streaming server:
```bash
python web_stream.py
```

2. Open browser at `http://127.0.0.1:5005` and start a stream

3. Run detection on the web stream:
```bash
python predict_stream.py
```

With custom options:
```bash
python predict_stream.py --stream-url http://127.0.0.1:5005/video_feed --model yolov8n.pt --conf 0.3
```

See [STREAMING.md](STREAMING.md) for detailed streaming documentation.

#### Training YOLOv8
```bash
python train.py model=yolov8n.yaml data=coco128.yaml epochs=100 imgsz=640
```

#### Validation
```bash
python val.py model=yolov8n.pt data=coco128.yaml
```

---

## Code Style Guidelines

### Language
- Comments and documentation in **Indonesian** and **English** mixed
- Variable names: snake_case
- Class names: PascalCase
- Constants: UPPER_CASE

### Key Naming Conventions
- Vehicle counting: `vehicle_in` (North direction), `vehicle_out` (South direction)
- Track ID: `identities` or `track_id`
- Object ID: `oid` (YOLO class ID)
- Counting line: `line` - tuple of two points

### Important Data Structures

**Vehicle Count Data** (`vehicle_data.json`):
```json
{
  "in": {"car": 5, "motorcycle": 3},
  "out": {"car": 2, "truck": 1}
}
```

**Firebase Data Structure**:
```json
{
  "traffic_snapshot": {
    "current_data": {
      "entering": {"MC": 10, "LV": 5, "HV": 2},
      "leaving": {"MC": 8, "LV": 3, "HV": 1},
      "LOS": {"in": "B", "out": "A"},
      "DS": {"in": 0.45, "out": 0.32},
      "location": "Jl. Setiabudi",
      "timestamp": "2024-01-01T12:00:00"
    }
  },
  "traffic_history": { ... }
}
```

---

## Testing Instructions

### Manual Testing

1. **Test with sample video**:
   ```bash
   python predict.py model=yolov8n.pt source=traffic.mp4
   ```

2. **Verify Firebase connection**: Check if data appears in Firebase Realtime Database

3. **Check logs**: Monitor `predict.log` for tracker initialization messages

### Unit Testing

The project currently lacks formal unit tests. Key components to verify:
- `process_results.py`: Test vehicle classification logic
- DeepSORT tracker: Verify tracking consistency

---

## Architecture Details

### Data Flow

1. **Video Input** → OpenCV capture
2. **Frame Processing** → YOLOv8 detection (class, bbox, confidence)
3. **Feature Extraction** → DeepSORT CNN (appearance features)
4. **Tracking** → Kalman filter + Hungarian algorithm (track assignment)
5. **Counting** → Line intersection check with direction
6. **Aggregation** → Vehicle classification (MC/LV/HV)
7. **Metrics** → Q, DS, LOS calculation (every 10 minutes)
8. **Storage** → Firebase Realtime Database update

### Key Algorithms

**Traffic Flow (Q) Calculation**:
```python
Q = MC_count * 0.25 + LV_count * 1.0 + HV_count * 1.2
```

**Degree of Saturation (DS)**:
```python
DS = (Q * 12) / C  # C = road capacity (default: 1325)
```

**Level of Service (LOS)**:
| DS Range | LOS |
|----------|-----|
| ≤ 0.6    | A   |
| ≤ 0.7    | B   |
| ≤ 0.8    | C   |
| ≤ 0.9    | D   |
| ≤ 1.0    | E   |
| > 1.0    | F   |

---

## Security Considerations

### Hardcoded Credentials
**WARNING**: Firebase credentials are hardcoded in:
- `predict.py` (lines 341-345)
- `firebase_config.py`

**Recommendation**: Move credentials to environment variables:
```python
import os
cred_path = os.environ.get('FIREBASE_CRED_PATH')
database_url = os.environ.get('FIREBASE_DB_URL')
```

### File Paths
Multiple absolute Windows paths are hardcoded:
- `E:/tugas_akhir/...` (predict.py line 333)
- `E:/firebase/...` (firebase_config.py line 5)

These need to be updated for the deployment environment.

---

## Known Issues and Limitations

1. **Duplicate initialization**: `counted_vehicles` initialized 3 times in `predict.py`
2. **Platform dependency**: Hardcoded Windows paths
3. **Empty placeholder**: `python_to_database.py` is empty
4. **Typo**: Line 120 in `process_results.py` has `if __name__ == "_main_":` (missing underscores)
5. **Missing checkpoint**: `ckpt.t7` must be downloaded separately

---

## Deployment Notes

### Environment Setup

1. Install Python 3.8+
2. Install CUDA (optional, for GPU acceleration)
3. Install dependencies
4. Download DeepSORT checkpoint
5. Configure Firebase credentials
6. Adjust line coordinates for your camera view

### Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with CUDA support for real-time processing
- **Camera**: IP camera or video file input

---

## References

- DeepSORT Paper: [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)
- Original DeepSORT: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)
- DeepSORT PyTorch: [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
