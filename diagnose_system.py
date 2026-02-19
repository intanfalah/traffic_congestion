#!/usr/bin/env python3
"""
System Diagnostic Tool
Check if CCTVs are running and data is flowing
"""

import requests
import json
import time

API_URL = "http://127.0.0.1:5005"

def check_endpoint(endpoint, name):
    """Check an API endpoint"""
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=5)
        if response.ok:
            data = response.json()
            print(f"✅ {name}: OK")
            return data
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ {name}: {e}")
        return None

def main():
    print("=" * 70)
    print("🔍 SYSTEM DIAGNOSTIC")
    print("=" * 70)
    
    # Check if server is running
    print("\n1. Checking server...")
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        print(f"   ✅ Server is running (HTTP {response.status_code})")
    except:
        print(f"   ❌ Server not responding")
        print(f"   Run: python app.py")
        return
    
    # Check CCTVs
    print("\n2. Checking CCTVs...")
    cctv_data = check_endpoint("/api/cctvs", "CCTV List")
    if cctv_data:
        cctvs = cctv_data.get('cctvs', [])
        print(f"   Found {len(cctvs)} CCTVs:")
        for cctv in cctvs:
            status = cctv.get('status', 'unknown')
            icon = '🟢' if status == 'active' else '🔴' if status == 'error' else '⚪'
            print(f"   {icon} {cctv['name']} - Status: {status}")
    
    # Check traffic data
    print("\n3. Checking traffic data...")
    traffic_data = check_endpoint("/api/traffic/status", "Traffic Status")
    if traffic_data:
        print(f"   Data: {json.dumps(traffic_data, indent=2)[:500]}")
    
    # Check specific CCTV status
    if cctv_data and cctv_data.get('cctvs'):
        print("\n4. Checking individual CCTV status...")
        for cctv in cctv_data['cctvs']:
            cctv_id = cctv['id']
            status_data = check_endpoint(f"/api/cctvs/{cctv_id}/status", f"Status for {cctv_id}")
            if status_data:
                traffic = status_data.get('traffic', {})
                vcount = traffic.get('vehicle_count', 0)
                print(f"   Vehicles counted: {vcount}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    issues = []
    
    if not cctv_data or not cctv_data.get('cctvs'):
        issues.append("No CCTVs configured - run: python add_real_cctvs.py")
    else:
        active_cctvs = [c for c in cctv_data['cctvs'] if c.get('status') == 'active']
        if not active_cctvs:
            issues.append("CCTVs added but not active - check server logs for errors")
    
    if not traffic_data:
        issues.append("No traffic data - detection may not be running")
    elif traffic_data and not any(traffic_data.values()):
        issues.append("Traffic data is empty - detection running but not counting")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✅ System appears healthy")
        print("   If dashboard shows zeros, try refreshing the browser")
    
    print("\n💡 TIPS:")
    print("   - Check browser console (F12) for JavaScript errors")
    print("   - Check server terminal for Python errors")
    print("   - Try hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)")
    print("   - Clear browser cache and cookies")

if __name__ == "__main__":
    main()
