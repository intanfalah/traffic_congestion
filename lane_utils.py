#!/usr/bin/env python3
"""
Lane utilities: per-camera split-line config, lane assignment, and road-vector
offsetting for per-lane traffic on the map.

A "split line" is a straight line drawn in an image, stored in NORMALISED
coordinates (0..1 of frame width/height) so it is resolution independent. Each
detected vehicle's ground point (bottom-centre of its box) falls on one side of
the line -> lane 'A' or 'B'.
"""

import json
import math
import os

LANE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "lane_config.json")

# Perpendicular offset (metres) applied to a road centreline to draw each lane.
LANE_OFFSET_M = 6.0

# Default split line if a camera is not calibrated: vertical line down the middle.
DEFAULT_SPLIT_LINE = [[0.5, 0.0], [0.5, 1.0]]


def load_lane_config():
    """Load the whole lane config (cctv_id -> {split_line, flip})."""
    if os.path.exists(LANE_CONFIG_PATH):
        try:
            with open(LANE_CONFIG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_lane_config(cfg):
    with open(LANE_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_split_line(cctv_id, cfg=None):
    """Return (split_line, flip) for a camera; defaults to a centre vertical line."""
    cfg = cfg if cfg is not None else load_lane_config()
    entry = cfg.get(cctv_id)
    if entry and entry.get("split_line"):
        return entry["split_line"], bool(entry.get("flip", False))
    return DEFAULT_SPLIT_LINE, False


def set_split_line(cctv_id, split_line, flip=False):
    """Persist a camera's split line (normalised coords) and flip flag."""
    cfg = load_lane_config()
    cfg[cctv_id] = {"split_line": split_line, "flip": bool(flip)}
    save_lane_config(cfg)
    return cfg[cctv_id]


def is_calibrated(cctv_id, cfg=None):
    cfg = cfg if cfg is not None else load_lane_config()
    entry = cfg.get(cctv_id)
    return bool(entry and entry.get("split_line"))


def lane_of(px, py, w, h, split_line, flip=False):
    """Which lane a pixel point (px, py) is on given a normalised split line.

    Uses the sign of the 2D cross product of the line direction and the vector
    from the line start to the point. Returns 'A' or 'B'.
    """
    (nx1, ny1), (nx2, ny2) = split_line
    x1, y1 = nx1 * w, ny1 * h
    x2, y2 = nx2 * w, ny2 * h
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    side_a = cross >= 0
    if flip:
        side_a = not side_a
    return "A" if side_a else "B"


def split_line_pixels(split_line, w, h):
    """Return the split line endpoints in pixel coords for drawing."""
    (nx1, ny1), (nx2, ny2) = split_line
    return (int(nx1 * w), int(ny1 * h)), (int(nx2 * w), int(ny2 * h))


def classify_congestion(density, stopped_frac=None):
    """Classify congestion from vehicle DENSITY, refined by the stopped fraction.

    Flow (line crossings/min) is ambiguous: it is near zero both on an empty
    road and in a gridlock, so it must never drive the congestion label.
    Density (vehicles present in view) says how full the road is; the stopped
    fraction (share of tracked vehicles that are stationary) distinguishes
    busy-but-flowing from a jammed queue and bumps the level one step up.
    """
    levels = [("FREE_FLOW", "A"), ("MODERATE", "C"),
              ("CONGESTED", "D"), ("SEVERE", "F")]
    if density < 5:
        idx = 0
    elif density < 12:
        idx = 1
    elif density < 20:
        idx = 2
    else:
        idx = 3
    # A mostly-stopped road at meaningful density is one level worse
    if stopped_frac is not None and density >= 5 and stopped_frac >= 0.5:
        idx = min(idx + 1, 3)
    return levels[idx]


def lane_congestion(count):
    """Map a single lane's vehicle count to a congestion level (per-lane scale)."""
    if count < 3:
        return "FREE_FLOW", "A"
    if count < 7:
        return "MODERATE", "C"
    if count < 12:
        return "CONGESTED", "D"
    return "SEVERE", "F"


def offset_polyline(coords, meters):
    """Offset a [[lat, lng], ...] polyline perpendicular by `meters`.

    Positive/negative `meters` offset to opposite sides. Perpendicular direction
    at each vertex is derived from its neighbouring segment(s). Longitude is
    scaled by cos(lat) so the offset is a true metric distance.
    """
    n = len(coords)
    if n < 2:
        return list(coords)
    out = []
    for i in range(n):
        if i == 0:
            a, b = coords[0], coords[1]
        elif i == n - 1:
            a, b = coords[i - 1], coords[i]
        else:
            a, b = coords[i - 1], coords[i + 1]
        lat, lng = coords[i][0], coords[i][1]
        coslat = math.cos(math.radians(lat)) or 1e-6
        # Direction vector in metres (east, north)
        vx = (b[1] - a[1]) * 111320.0 * coslat
        vy = (b[0] - a[0]) * 110540.0
        norm = math.hypot(vx, vy) or 1.0
        # Left-perpendicular unit vector (rotate +90deg): (-vy, vx)
        px = -vy / norm
        py = vx / norm
        off_east = px * meters
        off_north = py * meters
        out.append([
            lat + off_north / 110540.0,
            lng + off_east / (111320.0 * coslat),
        ])
    return out
