# 🚀 Quick Start Guide

## Smart Traffic Management System

### What You Get

This system provides a complete traffic monitoring solution similar to Semarang's CCTV portal:

1. **🗺️ Interactive Map** - Shows all CCTV locations with color-coded congestion status
2. **📹 Live Streams** - Click any CCTV to see live video with vehicle detection overlay
3. **🤖 Background Processing** - Automatically counts vehicles on all streams
4. **📊 Traffic Analysis** - Calculates congestion levels and LOS (Level of Service)
5. **🛣️ Road Network View** - Separate page showing traffic status across road network
6. **💾 Data Storage** - SQLite database with spatial support for historical analysis

---

## 🎯 Quick Steps (In Order)

### Step 1: Install Dependencies (One-time)

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate      # Windows

# Install all packages
pip install -r requirements.txt
```

### Step 2: Download YOLO Model (One-time)

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Step 3: Run the Main Application

```bash
python app.py
```

You should see output like:
```
🚦 Smart Traffic Management System
==============================================
[Setup] Loading demo CCTV data...
[Server] Starting on http://127.0.0.1:5005
```

### Step 4: Open Browser and Test

Open these URLs in your browser:
- **Main Dashboard (Map)**: http://127.0.0.1:5005
- **Traffic Status**: http://127.0.0.1:5005/traffic-status

**You should see:**
- A map with 5 demo CCTV markers in Semarang
- Click any marker to see live video with detection

---

## 🔌 Adding Real CCTV Streams from Pantau Semarang

Since the website uses heavy JavaScript protection, you'll need to manually extract stream URLs:

### Method: Browser Network Inspector

#### 1. Keep the Flask App Running
Don't close the terminal where `python app.py` is running!

#### 2. Open Pantau Semarang Website
Go to: https://pantausemar.semarangkota.go.id/

#### 3. Open Developer Tools
- Press **F12** (Windows/Linux)
- Or **Cmd+Option+I** (Mac)

#### 4. Inspect Network Traffic
1. Click on the **Network** tab
2. Click **Fetch/XHR** or **Media** filter button
3. Clear the log (click 🚫 icon)

#### 5. Capture Stream URL
1. On the map, **click a CCTV marker**
2. Watch the Network tab for new requests
3. Look for URLs containing:
   - `.m3u8` (HLS streams)
   - `.mp4` (video files)
   - `stream` in the URL
4. Click on the request
5. Go to **Headers** tab
6. Copy the **Request URL**

#### 6. Add to Your System

**Option A: Use Interactive Tool**
```bash
# In a NEW terminal (keep app.py running)
source .venv/bin/activate
python add_cctv_interactive.py
```
Follow the prompts to paste your captured URL.

**Option B: Use Browser Helper**
```bash
# Open the browser helper tool
open utils/browser_capture.html  # Mac
# OR
start utils/browser_capture.html  # Windows
```
Paste your captured data and it will generate commands.

**Option C: Direct API Call**
```bash
curl -X POST http://127.0.0.1:5005/api/cctvs \
  -H "Content-Type: application/json" \
  -d '{
    "id": "simpang_lima_real",
    "name": "Simpang Lima (Real)",
    "latitude": -6.9902,
    "longitude": 110.4229,
    "stream_url": "PASTE_YOUR_CAPTURED_URL_HERE"
  }'
```

---

## 📖 Using the Interactive Tools

### Tool 1: Add CCTV Interactive

```bash
python add_cctv_interactive.py
```

Features:
- Step-by-step CCTV configuration
- URL validation
- Automatic coordinate suggestions
- Test connection before adding

### Tool 2: Browser Capture Helper

Open `utils/browser_capture.html` in your browser:
```bash
open utils/browser_capture.html  # Mac
start utils/browser_capture.html  # Windows
```

Paste captured network data and it will:
- Extract stream URLs automatically
- Generate ready-to-run API commands
- Validate URL formats

### Tool 3: Website Inspector

```bash
python utils/inspect_semarang.py
```

This attempts to automatically find API endpoints (may not work due to JS protection).

---

## 🎨 Understanding the Dashboard

### CCTV Map Page
| Element | Description |
|---------|-------------|
| 🟢 Green marker | Free flow traffic (< 10 vehicles/min) |
| 🟡 Yellow marker | Moderate traffic (10-30 vehicles/min) |
| 🟠 Orange marker | Congested traffic (30-60 vehicles/min) |
| 🔴 Red marker | Severe congestion (> 60 vehicles/min) |
| Click marker | Opens live stream popup with detection overlay |
| Sidebar | List of all CCTVs with live statistics |

### Traffic Status Page
| Metric | Description |
|--------|-------------|
| Vehicles/Min | Average vehicles per minute |
| LOS | Level of Service (A=best, F=worst) |
| Congestion | Free/Moderate/Congested/Severe |
| System LOS | Overall network health |

---

## 🗄️ System Architecture

```
User clicks CCTV on map
        ↓
Popup opens with MJPEG stream
        ↓
Background worker detects vehicles (YOLOv8)
        ↓
DeepSORT tracks vehicles across frames
        ↓
Counter increments when crossing line
        ↓
Calculates congestion (vehicles/min)
        ↓
Updates map color via WebSocket
        ↓
Saves to SQLite database
        ↓
Updates traffic status page
```

---

## 🛠️ Troubleshooting

### "Module not found"
```bash
pip install flask-socketio beautifulsoup4 requests
```

### "Failed to open stream"
- Check camera permissions
- Try different device index: `0`, `1`, `2`
- For IP cameras: Verify credentials and network

### Map shows no markers
- Check browser console (F12) for JavaScript errors
- Verify WebSocket connection in Network tab
- Refresh the page

### High CPU usage
- Reduce number of CCTVs
- Use lighter YOLO model: `yolov8n.pt`
- Detection processes every 3rd frame by default

### Cannot extract URLs from Pantau Semarang
- The site uses JavaScript protection
- Use browser Network tab method (guaranteed to work)
- Look for `.m3u8` URLs specifically
- Some CCTVs may require authentication

---

## 🎯 Example Workflow

```bash
# Terminal 1: Start the main system
source .venv/bin/activate
python app.py

# Terminal 2: Add CCTVs interactively
source .venv/bin/activate
python add_cctv_interactive.py

# Follow prompts:
# - ID: simpang_lima_01
# - Name: Simpang Lima - Jl. Pahlawan
# - Stream URL: [paste from browser]
# - Lat/Lng: -6.9902, 110.4229

# Browser: View dashboard
open http://127.0.0.1:5005
```

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application |
| `add_cctv_interactive.py` | Interactive CCTV addition tool |
| `utils/browser_capture.html` | Browser helper for URL extraction |
| `utils/inspect_semarang.py` | Website inspector |
| `database/traffic.db` | SQLite database |
| `templates/map_dashboard.html` | CCTV map page |
| `templates/traffic_status.html` | Road network page |

---

## 💡 Pro Tips

1. **Start with webcam** - Test detection works before adding real streams
2. **One CCTV at a time** - Add and verify each stream individually
3. **Use video files** - Download traffic videos from YouTube if streams don't work
4. **Check browser console** - F12 → Console shows JavaScript errors
5. **Network tab is your friend** - Most reliable way to find stream URLs

---

## 🆘 Need Help?

If you can't extract URLs:
1. Some CCTVs may be behind authentication
2. Try different districts on the map
3. Use video files as alternative: `python add_cctv_interactive.py` → enter file path
4. Check `EXTRACT_CCTVS.md` for more detailed extraction methods

---

## ✅ Next Steps After Setup

1. ✅ Test with demo CCTVs (webcam)
2. ✅ Extract one real stream URL from Pantau Semarang
3. ✅ Add real CCTV using interactive tool
4. ✅ Verify detection works on real stream
5. ✅ Add more CCTVs
6. ✅ Check traffic status page
7. ✅ Analyze historical data in database

Happy monitoring! 🚦
