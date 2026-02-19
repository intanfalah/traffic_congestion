# System Architecture - Real-time Processing

## Overview

The system uses **real-time processing** (NOT save-then-process). Here's how it works:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   HLS Stream    │────▶│  Capture Thread │────▶│  Latest Frame   │
│  (Semarang CCTV)│     │  (OpenCV)       │     │  Queue (size=1) │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                              ┌────────────────────────┐
                              │ Drop old frames if     │
                              │ processing is slow     │
                              └────────────────────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────┐
                                              │ Processing      │
                                              │ Thread (YOLO)   │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Web Stream      │
                                              │ (MJPEG)         │
                                              └─────────────────┘
```

## Key Features

### 1. **Separate Capture & Processing Threads**
- **Capture Thread**: Continuously reads from HLS stream
- **Processing Thread**: Runs YOLO detection
- **Benefit**: No blocking - capture never waits for processing

### 2. **Latest Frame Queue (size=1)**
```python
self.frame_queue = queue.Queue(maxsize=1)
```
- Only keeps the **most recent frame**
- If processing is slow, old frames are **dropped**
- **Benefit**: Always processing fresh data, minimal latency

### 3. **Frame Dropping Strategy**
```python
try:
    self.frame_queue.put_nowait(frame)  # Try to add
    self.frame_count += 1
except queue.Full:
    self.dropped_frames += 1  # Drop if full
```
- If YOLO can't keep up, we drop frames rather than fall behind
- **Benefit**: Real-time performance, never accumulates delay

### 4. **Processing Statistics**
- **Processing FPS**: How fast YOLO can process
- **Dropped Frames**: How many we skipped to maintain real-time
- **Display**: Shows live in the video overlay

## Latency Breakdown

| Stage | Typical Latency |
|-------|----------------|
| HLS Buffer | 2-5 seconds (network) |
| OpenCV Capture | 100-300ms |
| YOLO Processing | 50-200ms (per frame) |
| Web Streaming | 100-200ms |
| **Total** | **~3-6 seconds** |

## Comparison: Old vs New

### Old Approach (Buffer-based)
```python
# Problem: Buffer accumulates frames
self.frame_buffer = deque(maxlen=30)  # 1 second buffer

# If processing is slow:
# - Buffer fills up
# - Lag increases over time
# - 10+ second delay possible
```

### New Approach (Queue-based)
```python
# Solution: Drop old frames
self.frame_queue = queue.Queue(maxsize=1)  # Only latest

# If processing is slow:
# - Old frames dropped
# - Always processing fresh frame
# - Constant 3-6 second latency
```

## Trade-offs

| Aspect | Buffer Approach | Queue Approach |
|--------|----------------|----------------|
| Latency | Increases over time | Constant |
| Frame Rate | Smooth (but delayed) | May skip frames |
| CPU Usage | Can spike | Regulated |
| Real-time | ❌ No | ✅ Yes |

## When to Use Each

### Use Buffer Approach (Original app.py):
- Recording video for later analysis
- When smooth playback is more important than real-time
- Offline processing

### Use Queue Approach (Optimized app_optimized.py):
- Live monitoring
- Real-time alerts
- Traffic management systems
- When latest data matters most

## Performance Tuning

### Reduce Latency Further:

1. **Lower YOLO resolution**:
```python
results = model(frame, conf=0.3, imgsz=416)  # Instead of 640
```

2. **Process every Nth frame**:
```python
if frame_count % 5 == 0:  # Instead of every frame
    process_frame(frame)
```

3. **Use lighter model**:
```python
model = YOLO('yolov8n.pt')  # Nano (fastest)
# Instead of yolov8l.pt (slowest)
```

4. **Reduce HLS segment duration** (if you control the stream):
```
# HLS with 1-second segments instead of 5-second
```

## Monitoring

Check these metrics in the video overlay:
- **FPS**: Processing speed
- **Dropped**: How many frames we skipped
- **Current**: Vehicles currently in frame

If dropped frames are high:
- Reduce processing resolution
- Use lighter model
- Or add GPU acceleration
