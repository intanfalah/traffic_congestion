"""
Database Models for Traffic Management System
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum


class CongestionLevel(Enum):
    FREE_FLOW = "A"      # < 10 vehicles/min
    MODERATE = "B"       # 10-30 vehicles/min
    CONGESTED = "C"      # 30-60 vehicles/min
    SEVERE = "D"         # > 60 vehicles/min


@dataclass
class CCTV:
    id: str
    name: str
    latitude: float
    longitude: float
    stream_url: str
    road_segment_id: Optional[str] = None
    status: str = "inactive"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RoadSegment:
    id: str
    name: str
    road_type: str  # main, secondary, arterial
    geometry: Dict  # GeoJSON format
    speed_limit: int  # km/h
    capacity: int  # vehicles per hour
    cctv_ids: List[str] = field(default_factory=list)


@dataclass
class TrafficData:
    id: Optional[str] = None
    cctv_id: str = ""
    road_segment_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Vehicle counts
    vehicle_count: int = 0
    vehicles_per_minute: float = 0.0
    
    # Vehicle type breakdown
    cars: int = 0
    motorcycles: int = 0
    buses: int = 0
    trucks: int = 0
    
    # Traffic metrics
    average_speed: float = 0.0  # km/h (estimated)
    density: float = 0.0  # vehicles per km
    congestion_level: str = "UNKNOWN"
    los: str = "-"  # Level of Service A-F
    
    # Additional data
    metadata: Dict = field(default_factory=dict)


@dataclass
class TrafficSummary:
    """Aggregated traffic summary for a time period"""
    road_segment_id: str
    period_start: str
    period_end: str
    
    avg_vehicles_per_minute: float = 0.0
    max_vehicles_per_minute: float = 0.0
    min_vehicles_per_minute: float = 0.0
    
    dominant_congestion_level: str = "UNKNOWN"
    los_distribution: Dict = field(default_factory=dict)
