# How to Extract CCTV URLs from Pantau Semarang

Since the website loads CCTV data dynamically, here's the **manual extraction method**:

## Method 1: Browser Developer Tools (Recommended)

### Step 1: Open the Website
1. Go to: https://pantausemar.semarangkota.go.id/
2. Click on a district (e.g., "Semarang Tengah")
3. You should see CCTV markers on the map

### Step 2: Open Developer Tools
1. Press **F12** (or right-click → Inspect)
2. Go to **Network** tab
3. Click the **XHR** or **Fetch/XHR** filter

### Step 3: Click a CCTV Marker
1. Click on any CCTV icon on the map
2. Look for new network requests in the Developer Tools
3. You should see requests like:
   - `getCctvStream`
   - `stream.m3u8`
   - `.mp4` files

### Step 4: Extract Stream URL
1. Click on the network request
2. Look at **Headers** or **Response**
3. Find the stream URL, which might look like:
   ```
   https://stream.semarangkota.go.id/cctv/simpanglima/playlist.m3u8
   https://cctv.semarangkota.go.id/live/simpanglima.m3u8
   rtsp://cctv.semarangkota.go.id:554/stream
   ```

### Step 5: Add to System
Once you have a stream URL, add it:

```bash
curl -X POST http://127.0.0.1:5005/api/cctvs \
  -H "Content-Type: application/json" \
  -d '{
    "id": "simpang_lima",
    "name": "Simpang Lima",
    "latitude": -6.9902,
    "longitude": 110.4229,
    "stream_url": "https://stream.semarangkota.go.id/cctv/simpanglima/playlist.m3u8"
  }'
```

---

## Method 2: Page Source Inspection

### Step 1: View Page Source
1. Go to the CCTV page
2. Press **Ctrl+U** (View Page Source)
3. Or right-click → "View Page Source"

### Step 2: Search for URLs
1. Press **Ctrl+F** to search
2. Search for patterns like:
   - `.m3u8`
   - `stream`
   - `cctv`
   - `rtsp`
   - `.mp4`

### Step 3: Find CCTV Data
Look for JavaScript arrays or JSON objects containing CCTV information:

```javascript
// Example of what you might find
var cctvs = [
  {
    "name": "Simpang Lima",
    "lat": -6.9902,
    "lng": 110.4229,
    "stream": "https://..."
  },
  ...
];
```

---

## Method 3: Using Browser Console

### Step 1: Open Console
1. Press **F12**
2. Go to **Console** tab

### Step 2: Run JavaScript
Try running these commands to find CCTV data:

```javascript
// Look for variables containing CCTV data
Object.keys(window).filter(k => k.toLowerCase().includes('cctv'))

// Search in all variables
for (let key in window) {
  if (typeof window[key] === 'object' && window[key] !== null) {
    let str = JSON.stringify(window[key]);
    if (str.includes('stream') && str.includes('lat')) {
      console.log(key, window[key]);
    }
  }
}
```

---

## Method 4: Network Analysis (Advanced)

### Step 1: Monitor API Calls
1. Open Developer Tools → Network
2. Clear the log (click 🚫 icon)
3. Reload the page (Ctrl+R)
4. Click on a district

### Step 2: Find API Endpoints
Look for API calls that return CCTV data:
```
/api/cctvs?district=semarang_tengah
/getCctvs?category=...
/cctv/list?...
```

### Step 3: Analyze Response
Click on the API call and check the **Preview** or **Response** tab

---

## Example: Adding Extracted CCTVs

Once you have URLs, create a file `real_cctvs.py`:

```python
import requests

CCTVS = [
    {
        "id": "simpang_lima",
        "name": "Simpang Lima",
        "latitude": -6.9902,
        "longitude": 110.4229,
        "stream_url": "PASTE_URL_HERE"
    },
    # Add more...
]

for cctv in CCTVS:
    requests.post('http://127.0.0.1:5005/api/cctvs', json=cctv)
    print(f"Added {cctv['name']}")
```

---

## Quick Test

To test if a stream URL works:

```bash
# For HLS streams (.m3u8)
ffplay -i "https://stream.example.com/cctv.m3u8"

# For RTSP streams
ffplay -i "rtsp://user:pass@ip:554/stream"

# Or open in VLC Media Player
```

---

## Common Stream URL Patterns

| Type | Pattern | Example |
|------|---------|---------|
| HLS | `.m3u8` | `https://cctv.city.go.id/stream.m3u8` |
| RTSP | `rtsp://` | `rtsp://admin:pass@192.168.1.100:554/live` |
| HTTP | `.mp4` | `http://camera.local:8080/video.mp4` |
| MJPEG | `.mjpg` | `http://camera.local:8080/video.mjpg` |

---

## Troubleshooting

### "No stream found"
- Some CCTVs may require authentication
- Stream might be geo-blocked
- Try different districts

### "Stream won't play"
- Install ffmpeg: `brew install ffmpeg` (Mac) or `apt install ffmpeg` (Linux)
- Some streams use proprietary formats

### "Access denied"
- The website may have anti-scraping measures
- Use the browser extension method instead

---

## Alternative: Use Video Files for Testing

If you can't extract real streams, use video files:

```bash
# Download sample traffic video
wget https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4 \
  -O traffic_sample.mp4

# Add as CCTV
curl -X POST http://127.0.0.1:5005/api/cctvs \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_video",
    "name": "Test Video",
    "latitude": -6.9900,
    "longitude": 110.4200,
    "stream_url": "/path/to/traffic_sample.mp4"
  }'
```

---

## Need Help?

If you're having trouble extracting URLs:
1. Check the browser console for errors
2. Some CCTVs may be behind authentication
3. The stream format might not be directly accessible
4. Consider using **screen capture** as alternative

Once you have even ONE working stream URL, the system will work! 🎉
