#!/usr/bin/env python3
"""
Add Real Semarang CCTVs to the System
Using the provided stream URLs
"""

import requests
import json

API_URL = "http://127.0.0.1:5005"

# Real CCTVs from Pantau Semarang
REAL_CCTVS = [
    {
        "id": "cctv_001",
        "name": "Indraprasta Imam Bonjol",
        "latitude": -6.9785713,
        "longitude": 110.411635,
        "stream_url": "https://livepantau.semarangkota.go.id/3cc2431b-3ee5-4c91-8330-251c021cd510/video1_stream.m3u8",
        "road_segment_id": "road_indraprasta"
    },
    {
        "id": "cctv_002",
        "name": "Kaligarang",
        "latitude": -6.9957663,
        "longitude": 110.4023126,
        "stream_url": "https://livepantau.semarangkota.go.id/e9203185-ee2e-4eb0-83a4-46b80c3bcc1a/video1_stream.m3u8",
        "road_segment_id": "road_kaligarang"
    },
    {
        "id": "cctv_003",
        "name": "Kalibanteng 2",
        "latitude": -6.9845739,
        "longitude": 110.3835144,
        "stream_url": "https://livepantau.semarangkota.go.id/b216444c-25db-4be2-bb30-fcb044f7c83f/video1_stream.m3u8",
        "road_segment_id": "road_kalibanteng"
    }
]


def test_stream(url):
    """Test if stream is accessible"""
    try:
        print(f"  Testing: {url[:70]}...")
        response = requests.head(url, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            print(f"  ✅ Stream accessible (Status: {response.status_code})")
            return True
        else:
            print(f"  ⚠️  Status: {response.status_code} - may need authentication")
            return True  # Still try to add
    except Exception as e:
        print(f"  ⚠️  Test failed: {e}")
        print(f"     Will still try to add")
        return True


def add_cctv(cctv):
    """Add a single CCTV"""
    try:
        print(f"\n📹 Adding: {cctv['name']}")
        print(f"   Location: {cctv['latitude']}, {cctv['longitude']}")
        
        # Test stream first
        test_stream(cctv['stream_url'])
        
        # Add CCTV
        response = requests.post(
            f"{API_URL}/api/cctvs",
            json=cctv,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.ok:
            print(f"   ✅ CCTV added to database")
            
            # Start detection
            print(f"   🎥 Starting detection (this may take a moment)...")
            start_resp = requests.post(
                f"{API_URL}/api/cctvs/{cctv['id']}/start",
                timeout=30  # Longer timeout for stream opening
            )
            
            if start_resp.ok:
                result = start_resp.json()
                print(f"   ✅ {result.get('message', 'Started')}")
            else:
                print(f"   ⚠️  Detection issue: {start_resp.status_code}")
                print(f"      This usually means the stream can't be opened.")
                print(f"      Check: 1) FFmpeg installed, 2) Network access, 3) Stream URL valid")
            
            return True
        else:
            print(f"   ❌ Failed to add: {response.status_code} - {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to {API_URL}")
        print(f"   Make sure: python app.py is running on port 5005")
        return False
    except requests.exceptions.Timeout:
        print(f"   ⚠️  Timeout - stream may be slow to open")
        print(f"      Check if stream is accessible in browser/VLC")
        return True  # Consider it added even if timeout
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    print("=" * 70)
    print("🚦 ADDING REAL SEMARANG CCTVs")
    print("=" * 70)
    print(f"\nAPI URL: {API_URL}")
    print(f"Total CCTVs to add: {len(REAL_CCTVS)}")
    print("\nMake sure the Flask server is running:")
    print("   python app.py")
    print()
    
    input("Press Enter to continue...")
    
    print("\n" + "=" * 70)
    
    success = 0
    for cctv in REAL_CCTVS:
        if add_cctv(cctv):
            success += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully added: {success}/{len(REAL_CCTVS)} CCTVs")
    print(f"\n🌐 View the map: http://127.0.0.1:5000")
    print(f"📊 Traffic status: http://127.0.0.1:5000/traffic-status")
    
    # Save to file
    with open('semarang_real_cctvs.json', 'w') as f:
        json.dump(REAL_CCTVS, f, indent=2)
    print(f"\n💾 Saved to: semarang_real_cctvs.json")


if __name__ == "__main__":
    main()
