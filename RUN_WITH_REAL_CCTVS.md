# 🚀 Run with Real Semarang CCTVs

This guide helps you run the system with the 3 real CCTVs you provided.

---

## 📋 Your CCTVs

| ID | Name | Location | Stream URL |
|----|------|----------|------------|
| 1 | Indraprasta Imam Bonjol | -6.9785713, 110.411635 | [HLS Stream](https://livepantau.semarangkota.go.id/3cc2431b-3ee5-4c91-8330-251c021cd510/video1_stream.m3u8) |
| 2 | Kaligarang | -6.9957663, 110.4023126 | [HLS Stream](https://livepantau.semarangkota.go.id/e9203185-ee2e-4eb0-83a4-46b80c3bcc1a/video1_stream.m3u8) |
| 3 | Kalibanteng 2 | -6.9845739, 110.3835144 | [HLS Stream](https://livepantau.semarangkota.go.id/b216444c-25db-4be2-bb30-fcb044f7c83f/video1_stream.m3u8) |

---

## 🏃 Quick Start

### Step 1: Start the Main Application

```bash
# Terminal 1 - Start the Flask server
source .venv/bin/activate
python app.py
```

You should see:
```
🚦 Smart Traffic Management System
============================================================
[System] Loading YOLO model: yolov8n.pt
[System] Model loaded successfully
[Database] Initialized at database/traffic.db

[Setup] Demo data disabled.
       Add CCTVs using: python add_real_cctvs.py
       Or run: python add_cctv_interactive.py

[Server] Starting on http://127.0.0.1:5005
```

### Step 2: Add the Real CCTVs

In a **NEW terminal** (keep the first one running):

```bash
# Terminal 2 - Add the CCTVs
source .venv/bin/activate
python add_real_cctvs.py
```

You should see:
```
============================================================
🚦 ADDING REAL SEMARANG CCTVs
============================================================

Press Enter to continue...

📹 Adding: Indraprasta Imam Bonjol
  Testing: https://livepantau.semarangkota.go.id/...
  ✅ CCTV added successfully
  🎥 Starting detection...
  ✅ Detection started!

... (repeats for all 3 CCTVs)

✅ Successfully added: 3/3 CCTVs

🌐 View the map: http://127.0.0.1:5005
📊 Traffic status: http://127.0.0.1:5005/traffic-status
```

### Step 3: View the Dashboard

Open in your browser:

**🗺️ CCTV Map**: http://127.0.0.1:5005
- See 3 markers on the map in Semarang
- Colors indicate real-time traffic status
- Click any marker to see live stream with vehicle detection

**📊 Traffic Status**: http://127.0.0.1:5005/traffic-status
- Road network view with congestion levels
- Statistics for all 3 locations
- Historical data charts

---

## 🔧 How It Works

### For Each CCTV:

1. **HLS Stream (.m3u8)** → OpenCV captures frames
2. **YOLOv8** detects vehicles in each frame
3. **DeepSORT** tracks vehicles across frames
4. **Line crossing detection** counts vehicles
5. **Traffic analysis** calculates vehicles/minute
6. **Congestion level** determined:
   - 🟢 Free Flow: < 10 v/min
   - 🟡 Moderate: 10-30 v/min
   - 🟠 Congested: 30-60 v/min
   - 🔴 Severe: > 60 v/min
7. **WebSocket** updates map in real-time
8. **Database** stores historical data

---

## 🎮 Controls

### While Viewing a Stream Popup:
- **Detection runs automatically** in background
- **Vehicle count** updates every 10 seconds
- **Close popup** to stop viewing (detection continues)

### System Status:
- Green dot = System online
- Numbers update automatically
- No page refresh needed

---

## 📊 Expected Performance

With 3 HLS streams on typical hardware:

| Hardware | FPS per Stream | Total CPU |
|----------|----------------|-----------|
| CPU only (4 cores) | ~5-10 | 60-80% |
| CPU only (8 cores) | ~10-15 | 50-70% |
| With GPU (GTX 1060) | ~15-25 | 30-40% |
| With GPU (RTX 3060) | ~25-30 | 20-30% |

**Tips for better performance:**
- Use `yolov8n.pt` (smallest model)
- Detection processes every 3rd frame by default
- Close browser tabs you're not viewing

---

## 🔍 Troubleshooting

### "Failed to open stream"

**Cause**: OpenCV can't decode HLS stream

**Fix**: Install FFmpeg with HLS support
```bash
# Mac
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
# Add to PATH
```

### "Stream opens but no video"

**Cause**: Network latency or authentication

**Fix**:
1. Test URL directly in VLC:
   ```bash
   vlc "https://livepantau.semarangkota.go.id/.../video1_stream.m3u8"
   ```
2. Check if URL requires authentication
3. Check internet connection

### High CPU usage

**Fix**:
```python
# Edit app.py line ~192
# Change from:
if self.frame_count % 3 == 0:  # Process every 3rd frame
# To:
if self.frame_count % 5 == 0:  # Process every 5th frame (slower)
```

### "No module named cv2"

```bash
pip install opencv-python opencv-contrib-python
```

---

## 🗄️ Data Storage

All data is saved to `database/traffic.db`:

```sql
-- View recent traffic data
SELECT cctv_id, vehicle_count, vehicles_per_minute, congestion_level, timestamp
FROM traffic_data
ORDER BY timestamp DESC
LIMIT 20;
```

Access via SQLite:
```bash
sqlite3 database/traffic.db
```

---

## 🌐 API Access

### Get All CCTVs
```bash
curl http://127.0.0.1:5005/api/cctvs
```

### Get Traffic Data
```bash
curl http://127.0.0.1:5005/api/traffic/status
```

### Get Specific CCTV Status
```bash
curl http://127.0.0.1:5005/api/cctvs/cctv_001/status
```

### Stop a CCTV
```bash
curl -X POST http://127.0.0.1:5005/api/cctvs/cctv_001/stop
```

### Restart a CCTV
```bash
curl -X POST http://127.0.0.1:5005/api/cctvs/cctv_001/start
```

---

## 🎯 Next Steps

1. ✅ **Verify streams are working** - Check browser for video
2. ✅ **Watch detection** - Vehicles should get boxes and IDs
3. ✅ **Check counting** - Numbers should increase as vehicles pass
4. ✅ **View traffic status** - Go to /traffic-status page
5. ✅ **Let it run** - Collect data for 10+ minutes
6. ✅ **Analyze** - Check historical trends in database

---

## 💾 Backup Your CCTVs

The CCTVs are saved to `semarang_real_cctvs.json`:

```bash
# Save current config
cp semarang_real_cctvs.json my_cctvs_backup.json

# Later, restore with:
python add_cctv_interactive.py
# → Select option 2: Add CCTVs from JSON file
```

---

## 🆘 Need Help?

If streams don't work:
1. Test URL in VLC first
2. Check terminal for error messages
3. Verify FFmpeg is installed
4. Check network connectivity

If detection is slow:
1. Check CPU usage (top/htop)
2. Reduce number of concurrent streams
3. Use GPU if available

---

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Map shows 3 markers in correct Semarang locations
- ✅ Clicking marker shows live video
- ✅ Green boxes appear around vehicles
- ✅ Numbers update every 10 seconds
- ✅ Marker colors change based on traffic
- ✅ No errors in terminal

---

**Ready? Let's go! 🚦**

```bash
# Terminal 1
python app.py

# Terminal 2
python add_real_cctvs.py

# Browser
open http://127.0.0.1:5005
```
