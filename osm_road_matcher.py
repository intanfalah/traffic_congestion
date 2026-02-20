#!/usr/bin/env python3
"""
OSM Road Matcher — Spatial join CCTVs to nearest OSM road segments.

Uses the Overpass API to find roads near each CCTV, then picks the nearest
road using haversine distance. Stores real OSM geometries in the database.
"""

import json
import math
import time
import requests
from database.db_manager import DatabaseManager

OVERPASS_URL = "https://lz4.overpass-api.de/api/interpreter"
OVERPASS_FALLBACK_URL = "https://overpass-api.de/api/interpreter"
SEARCH_RADIUS_M = 150  # meters
HIGHWAY_TYPES = "primary|secondary|tertiary|trunk"


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lon points."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Minimum distance from point (px,py) to line segment (ax,ay)-(bx,by) in lat/lon."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return haversine_distance(px, py, ax, ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return haversine_distance(px, py, proj_x, proj_y)


def min_distance_to_way(lat, lon, geometry):
    """Minimum distance from a point to a way's geometry (list of {lat, lon} dicts)."""
    min_dist = float("inf")
    for i in range(len(geometry) - 1):
        d = point_to_segment_distance(
            lat, lon,
            geometry[i]["lat"], geometry[i]["lon"],
            geometry[i + 1]["lat"], geometry[i + 1]["lon"],
        )
        min_dist = min(min_dist, d)
    return min_dist


def query_overpass(lat, lon, radius=SEARCH_RADIUS_M, retries=2):
    """Query Overpass API for roads near a coordinate."""
    query = f"""
[out:json][timeout:25];
(
  way["highway"~"{HIGHWAY_TYPES}"](around:{radius},{lat},{lon});
);
out geom;
"""
    for attempt in range(retries + 1):
        url = OVERPASS_URL if attempt == 0 else OVERPASS_FALLBACK_URL
        try:
            r = requests.get(url, params={"data": query}, timeout=30)
            if r.status_code == 200:
                return r.json().get("elements", [])
            print(f"  [Overpass] HTTP {r.status_code}, attempt {attempt + 1}")
        except requests.RequestException as e:
            print(f"  [Overpass] Request error: {e}, attempt {attempt + 1}")
        time.sleep(2)
    return []


def find_nearest_road(lat, lon, elements):
    """From a list of Overpass way elements, find the nearest one to (lat, lon)."""
    best = None
    best_dist = float("inf")
    for el in elements:
        geom = el.get("geometry", [])
        if len(geom) < 2:
            continue
        dist = min_distance_to_way(lat, lon, geom)
        if dist < best_dist:
            best_dist = dist
            best = el
    return best, best_dist


def match_cctvs_to_roads(db=None):
    """
    Main function: read CCTVs from DB, query OSM for each, pick nearest road,
    store road segment in DB, and link CCTV to it.
    
    Returns a list of {cctv_id, road_id, road_name, distance_m} dicts.
    """
    if db is None:
        db = DatabaseManager()

    cctvs = db.get_cctvs()
    if not cctvs:
        print("[RoadMatcher] No CCTVs in database")
        return []

    results = []
    for cctv in cctvs:
        cctv_id = cctv["id"]
        lat = cctv["latitude"]
        lon = cctv["longitude"]
        print(f"[RoadMatcher] Matching {cctv_id} ({cctv['name']}) at {lat},{lon}...")

        elements = query_overpass(lat, lon)
        if not elements:
            print(f"  No roads found within {SEARCH_RADIUS_M}m")
            continue

        best, dist = find_nearest_road(lat, lon, elements)
        if best is None:
            print(f"  No valid road geometry found")
            continue

        tags = best.get("tags", {})
        road_name = tags.get("name", f"OSM Way {best['id']}")
        road_type = tags.get("highway", "unknown")
        speed_limit = int(tags.get("maxspeed", "0").replace(" km/h", "").replace(" mph", "")) if tags.get("maxspeed") else 0
        osm_id = str(best["id"])
        road_id = f"osm_{osm_id}"

        # Build coordinate array [[lat, lon], ...]
        coords = [[pt["lat"], pt["lon"]] for pt in best.get("geometry", [])]

        print(f"  → {road_name} [{road_type}] ({len(coords)} pts, {dist:.0f}m away)")

        # Store road segment in database
        db.upsert_road_segment(
            road_id=road_id,
            name=road_name,
            road_type=road_type,
            speed_limit=speed_limit,
            capacity=0,
            coordinates=coords,
        )

        # Link CCTV to this road segment
        db.update_cctv_road_segment(cctv_id, road_id)

        results.append({
            "cctv_id": cctv_id,
            "road_id": road_id,
            "road_name": road_name,
            "distance_m": round(dist, 1),
        })

        # Rate-limit Overpass requests
        time.sleep(1.5)

    print(f"[RoadMatcher] Matched {len(results)}/{len(cctvs)} CCTVs to roads")
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("OSM Road Matcher")
    print("=" * 50)
    results = match_cctvs_to_roads()
    for r in results:
        print(f"  {r['cctv_id']} → {r['road_name']} ({r['distance_m']}m)")
