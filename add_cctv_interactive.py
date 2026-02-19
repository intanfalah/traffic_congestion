#!/usr/bin/env python3
"""
Interactive CCTV Addition Tool
Add CCTVs manually with an interactive CLI
"""

import requests
import json
from urllib.parse import urlparse

API_URL = "http://127.0.0.1:5005"


def test_stream_url(url):
    """Test if a stream URL is accessible"""
    print(f"  Testing stream: {url[:60]}...")
    
    try:
        # For HTTP/HTTPS streams
        if url.startswith(('http://', 'https://')):
            response = requests.head(url, timeout=5, allow_redirects=True)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                print(f"  ✅ Stream accessible (Content-Type: {content_type})")
                return True
            else:
                print(f"  ⚠️  HTTP {response.status_code} - may still work")
                return True
        
        # For RTSP streams, we can't easily test
        elif url.startswith('rtsp://'):
            print(f"  ℹ️  RTSP stream - cannot test via HTTP, will try anyway")
            return True
        
        # For local files/devices
        else:
            print(f"  ℹ️  Local stream - assuming valid")
            return True
            
    except Exception as e:
        print(f"  ⚠️  Test failed: {e}")
        print(f"     Will still try to add (may be network issue)")
        return True


def add_cctv_interactive():
    """Interactively add a CCTV"""
    print("\n" + "=" * 60)
    print("🎥 ADD NEW CCTV")
    print("=" * 60)
    
    # Get CCTV ID
    print("\n1. Enter CCTV ID (lowercase, no spaces, e.g., 'simpang_lima'):")
    cctv_id = input("   ID: ").strip().lower().replace(' ', '_')
    if not cctv_id:
        cctv_id = f"cctv_{int(time.time())}"
    
    # Get name
    print("\n2. Enter CCTV Name (e.g., 'Simpang Lima - Jl. Pahlawan'):")
    name = input("   Name: ").strip()
    if not name:
        name = cctv_id.replace('_', ' ').title()
    
    # Get stream URL
    print("\n3. Enter Stream URL:")
    print("   Supported formats:")
    print("   - Webcam: 0, 1, 2")
    print("   - HLS stream: https://example.com/stream.m3u8")
    print("   - RTSP: rtsp://user:pass@192.168.1.100:554/live")
    print("   - Video file: /path/to/video.mp4")
    stream_url = input("   URL: ").strip()
    
    if not stream_url:
        print("   ❌ Stream URL is required!")
        return False
    
    # Test the URL
    test_stream_url(stream_url)
    
    # Get coordinates
    print("\n4. Enter Coordinates (or press Enter for defaults):")
    print("   Default: -6.9900, 110.4200 (Semarang center)")
    lat_input = input("   Latitude (e.g., -6.9902): ").strip()
    lng_input = input("   Longitude (e.g., 110.4229): ").strip()
    
    try:
        lat = float(lat_input) if lat_input else -6.9900
        lng = float(lng_input) if lng_input else 110.4200
    except ValueError:
        print("   ⚠️  Invalid coordinates, using defaults")
        lat = -6.9900
        lng = 110.4200
    
    # Confirm
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print("=" * 60)
    print(f"   ID:       {cctv_id}")
    print(f"   Name:     {name}")
    print(f"   Stream:   {stream_url[:50]}...")
    print(f"   Location: {lat}, {lng}")
    print("=" * 60)
    
    confirm = input("\nAdd this CCTV? (y/n): ").strip().lower()
    if confirm != 'y':
        print("   ❌ Cancelled")
        return False
    
    # Send to API
    data = {
        "id": cctv_id,
        "name": name,
        "latitude": lat,
        "longitude": lng,
        "stream_url": stream_url
    }
    
    try:
        print(f"\n   Sending to API: {API_URL}/api/cctvs")
        response = requests.post(
            f"{API_URL}/api/cctvs",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.ok:
            print(f"   ✅ CCTV added successfully!")
            
            # Start detection
            print(f"   Starting detection...")
            start_resp = requests.post(f"{API_URL}/api/cctvs/{cctv_id}/start", timeout=5)
            if start_resp.ok:
                print(f"   ✅ Detection started!")
                print(f"\n   🌐 View at: http://127.0.0.1:5000")
            else:
                print(f"   ⚠️  Detection start failed: {start_resp.text}")
            
            return True
        else:
            print(f"   ❌ Failed to add: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to {API_URL}")
        print(f"   Make sure the Flask server is running: python app.py")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def add_from_json():
    """Add CCTVs from a JSON file"""
    print("\n" + "=" * 60)
    print("📁 ADD FROM JSON FILE")
    print("=" * 60)
    
    filename = input("\nEnter JSON file path: ").strip()
    
    try:
        with open(filename, 'r') as f:
            cctvs = json.load(f)
        
        if not isinstance(cctvs, list):
            print("   ❌ JSON should be a list of CCTV objects")
            return
        
        print(f"\n   Found {len(cctvs)} CCTVs in file")
        
        success = 0
        for cctv in cctvs:
            try:
                print(f"\n   Adding: {cctv.get('name', 'Unknown')}")
                response = requests.post(
                    f"{API_URL}/api/cctvs",
                    json=cctv,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                if response.ok:
                    # Start detection
                    cctv_id = cctv.get('id')
                    requests.post(f"{API_URL}/api/cctvs/{cctv_id}/start", timeout=5)
                    print(f"   ✅ Added and started")
                    success += 1
                else:
                    print(f"   ❌ Failed: {response.text}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n   ✅ Successfully added {success}/{len(cctvs)} CCTVs")
        
    except FileNotFoundError:
        print(f"   ❌ File not found: {filename}")
    except json.JSONDecodeError:
        print(f"   ❌ Invalid JSON file")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def show_current_cctvs():
    """Show currently configured CCTVs"""
    print("\n" + "=" * 60)
    print("📹 CURRENT CCTVs")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_URL}/api/cctvs", timeout=5)
        if response.ok:
            data = response.json()
            cctvs = data.get('cctvs', [])
            
            if not cctvs:
                print("\n   No CCTVs configured yet")
                return
            
            print(f"\n   Total: {len(cctvs)} CCTVs\n")
            
            for cctv in cctvs:
                status = cctv.get('status', 'unknown')
                status_icon = '🟢' if status == 'active' else '🔴' if status == 'inactive' else '⚪'
                print(f"   {status_icon} {cctv['name']}")
                print(f"      ID: {cctv['id']}")
                print(f"      Location: {cctv['latitude']}, {cctv['longitude']}")
                print(f"      Status: {status}")
                print()
        else:
            print(f"   ❌ Failed to fetch CCTVs")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def main():
    """Main menu"""
    import time
    
    while True:
        print("\n" + "=" * 60)
        print("🚦 SMART TRAFFIC - CCTV MANAGEMENT")
        print("=" * 60)
        print("\n1. ➕ Add new CCTV (interactive)")
        print("2. 📁 Add CCTVs from JSON file")
        print("3. 📹 View current CCTVs")
        print("4. 🌐 Open browser dashboard")
        print("5. ❌ Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            add_cctv_interactive()
        elif choice == '2':
            add_from_json()
        elif choice == '3':
            show_current_cctvs()
        elif choice == '4':
            import webbrowser
            webbrowser.open('http://127.0.0.1:5000')
            print("\n   🌐 Opened browser at http://127.0.0.1:5000")
        elif choice == '5':
            print("\n   👋 Goodbye!")
            break
        else:
            print("\n   ❌ Invalid option")


if __name__ == "__main__":
    main()
