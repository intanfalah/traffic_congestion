#!/usr/bin/env python3
"""
Manual CCTV Import Script
Add CCTVs manually since the scraper needs website inspection first
"""

import requests
import json

API_URL = "http://127.0.0.1:5005"

# Demo CCTVs - Replace these with real Semarang CCTVs when you have them
DEMO_CCTVS = [
    {
        "id": "semarang_01",
        "name": "Jl. Pahlawan - Simpang Lima",
        "latitude": -6.9902,
        "longitude": 110.4229,
        "stream_url": "0",  # Webcam for demo - replace with real stream
        "road_segment_id": "road_1"
    },
    {
        "id": "semarang_02", 
        "name": "Jl. MT Haryono - Pandanaran",
        "latitude": -6.9965,
        "longitude": 110.4310,
        "stream_url": "0",
        "road_segment_id": "road_2"
    },
    {
        "id": "semarang_03",
        "name": "Jl. Ahmad Yani - Kaligawe",
        "latitude": -6.9725,
        "longitude": 110.4450,
        "stream_url": "0",
        "road_segment_id": "road_3"
    },
    {
        "id": "semarang_04",
        "name": "Jl. Gajah Mada - Kudu",
        "latitude": -6.9830,
        "longitude": 110.4100,
        "stream_url": "0",
        "road_segment_id": "road_4"
    },
    {
        "id": "semarang_05",
        "name": "Jl. Pemuda - Balai Kota",
        "latitude": -6.9800,
        "longitude": 110.4080,
        "stream_url": "0",
        "road_segment_id": "road_5"
    }
]

def add_cctv(cctv_data):
    """Add a single CCTV"""
    try:
        response = requests.post(
            f"{API_URL}/api/cctvs",
            json=cctv_data,
            headers={"Content-Type": "application/json"}
        )
        if response.ok:
            print(f"✅ Added: {cctv_data['name']}")
            
            # Start detection
            start_resp = requests.post(f"{API_URL}/api/cctvs/{cctv_data['id']}/start")
            if start_resp.ok:
                print(f"   🎥 Detection started")
            return True
        else:
            print(f"❌ Failed: {cctv_data['name']} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {cctv_data['name']} - {e}")
        return False

def main():
    print("=" * 60)
    print("🚦 Adding Demo CCTVs to Traffic System")
    print("=" * 60)
    print(f"\nAPI URL: {API_URL}")
    print("\nMake sure the Flask server is running first!")
    print("Run: python app.py\n")
    
    input("Press Enter to continue...")
    
    print("\n" + "-" * 60)
    added = 0
    for cctv in DEMO_CCTVS:
        if add_cctv(cctv):
            added += 1
        print()
    
    print("-" * 60)
    print(f"\n✅ Successfully added {added}/{len(DEMO_CCTVS)} CCTVs")
    print(f"\nOpen http://127.0.0.1:5005 to view the map")
    
    # Save to file for reference
    with open('my_cctvs.json', 'w') as f:
        json.dump(DEMO_CCTVS, f, indent=2)
    print(f"📄 Saved CCTV list to my_cctvs.json")

if __name__ == "__main__":
    main()
