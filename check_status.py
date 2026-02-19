#!/usr/bin/env python3
"""
Check System Status
Verify CCTVs are added and working
"""

import requests
import json

API_URL = "http://127.0.0.1:5005"

def main():
    print("=" * 70)
    print("🔍 SYSTEM STATUS CHECK")
    print("=" * 70)
    
    try:
        # Check CCTVs
        print("\n📹 Checking CCTVs...")
        response = requests.get(f"{API_URL}/api/cctvs", timeout=5)
        if response.ok:
            data = response.json()
            cctvs = data.get('cctvs', [])
            print(f"   Total CCTVs: {len(cctvs)}")
            
            for cctv in cctvs:
                status = cctv.get('status', 'unknown')
                icon = '🟢' if status == 'active' else '🔴' if status == 'error' else '⚪'
                print(f"   {icon} {cctv['name']} ({status})")
        else:
            print(f"   ❌ Failed to get CCTVs: {response.status_code}")
        
        # Check traffic data
        print("\n📊 Checking traffic data...")
        response = requests.get(f"{API_URL}/api/traffic/status", timeout=5)
        if response.ok:
            data = response.json()
            print(f"   Active streams with data: {len(data)}")
        else:
            print(f"   ❌ Failed to get traffic data: {response.status_code}")
        
        # Check road segments
        print("\n🛣️  Checking road segments...")
        response = requests.get(f"{API_URL}/api/traffic/roads", timeout=5)
        if response.ok:
            data = response.json()
            roads = data.get('roads', [])
            print(f"   Total roads: {len(roads)}")
        else:
            print(f"   ❌ Failed to get roads: {response.status_code}")
        
        print("\n" + "=" * 70)
        print("✅ System check complete!")
        print("=" * 70)
        print(f"\n🌐 View dashboard: http://127.0.0.1:5005")
        print("\nNote: Detection runs in background.")
        print("      Streams may take 30-60 seconds to initialize.")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to server")
        print("   Make sure: python app.py is running")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
