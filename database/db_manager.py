"""
Database Manager - Using SQLite with spatial support via GeoAlchemy
For production, replace with PostgreSQL + PostGIS
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional


class DatabaseManager:
    """Simple SQLite-based database manager for traffic data"""
    
    def __init__(self, db_path: str = "database/traffic.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.init_connection()
        
    def init_connection(self):
        """Initialize database connection"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
    def init_database(self):
        """Create database tables"""
        # CCTV table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS cctvs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                stream_url TEXT NOT NULL,
                road_segment_id TEXT,
                status TEXT DEFAULT 'inactive',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Road segments table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS road_segments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                road_type TEXT,
                geometry TEXT,  -- GeoJSON
                speed_limit INTEGER,
                capacity INTEGER,
                coordinates TEXT  -- JSON array of [lat, lng] points
            )
        ''')
        
        # Traffic data table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cctv_id TEXT NOT NULL,
                road_segment_id TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                vehicle_count INTEGER DEFAULT 0,
                vehicles_per_minute REAL DEFAULT 0,
                cars INTEGER DEFAULT 0,
                motorcycles INTEGER DEFAULT 0,
                buses INTEGER DEFAULT 0,
                trucks INTEGER DEFAULT 0,
                average_speed REAL DEFAULT 0,
                density REAL DEFAULT 0,
                congestion_level TEXT DEFAULT 'UNKNOWN',
                los TEXT DEFAULT '-',
                metadata TEXT  -- JSON
            )
        ''')
        
        # Create indexes
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_traffic_cctv ON traffic_data(cctv_id)
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_traffic_time ON traffic_data(timestamp)
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_traffic_road ON traffic_data(road_segment_id)
        ''')
        
        self.conn.commit()
        print(f"[Database] Initialized at {self.db_path}")
        
        # Insert demo road segments
        self.init_demo_roads()
    
    def init_demo_roads(self):
        """Initialize demo road segments for Semarang"""
        demo_roads = [
            {
                'id': 'road_1',
                'name': 'Jl. Pahlawan',
                'type': 'main',
                'speed_limit': 60,
                'capacity': 2000,
                'coords': [[-6.9902, 110.4229], [-6.9910, 110.4235]]
            },
            {
                'id': 'road_2',
                'name': 'Jl. MT Haryono',
                'type': 'main',
                'speed_limit': 50,
                'capacity': 1800,
                'coords': [[-6.9965, 110.4310], [-6.9975, 110.4320]]
            },
            {
                'id': 'road_3',
                'name': 'Jl. Ahmad Yani',
                'type': 'arterial',
                'speed_limit': 60,
                'capacity': 2500,
                'coords': [[-6.9725, 110.4450], [-6.9740, 110.4460]]
            },
            {
                'id': 'road_4',
                'name': 'Jl. Gajah Mada',
                'type': 'secondary',
                'speed_limit': 40,
                'capacity': 1200,
                'coords': [[-6.9830, 110.4100], [-6.9840, 110.4110]]
            },
            {
                'id': 'road_5',
                'name': 'Jl. Pemuda',
                'type': 'main',
                'speed_limit': 50,
                'capacity': 1500,
                'coords': [[-6.9800, 110.4080], [-6.9810, 110.4090]]
            }
        ]
        
        for road in demo_roads:
            self.cursor.execute('''
                INSERT OR REPLACE INTO road_segments 
                (id, name, road_type, speed_limit, capacity, coordinates)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                road['id'],
                road['name'],
                road['type'],
                road['speed_limit'],
                road['capacity'],
                json.dumps(road['coords'])
            ))
        
        self.conn.commit()
    
    def add_cctv(self, cctv_id: str, name: str, lat: float, lng: float, 
                 stream_url: str, road_segment_id: Optional[str] = None):
        """Add a new CCTV"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO cctvs 
            (id, name, latitude, longitude, stream_url, road_segment_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (cctv_id, name, lat, lng, stream_url, road_segment_id))
        self.conn.commit()
    
    def upsert_road_segment(self, road_id: str, name: str, road_type: str,
                            speed_limit: int, capacity: int, coordinates: list):
        """Insert or update a road segment with real OSM geometry"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO road_segments 
            (id, name, road_type, speed_limit, capacity, coordinates)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (road_id, name, road_type, speed_limit, capacity, json.dumps(coordinates)))
        self.conn.commit()
    
    def update_cctv_road_segment(self, cctv_id: str, road_segment_id: str):
        """Link a CCTV to its matched road segment"""
        self.cursor.execute('''
            UPDATE cctvs SET road_segment_id = ? WHERE id = ?
        ''', (road_segment_id, cctv_id))
        self.conn.commit()
    
    def add_traffic_data(self, cctv_id: str, data: dict):
        """Add traffic data point"""
        vehicle_types = data.get('vehicle_types', {})
        
        # Get road_segment_id from the CCTV
        road_segment_id = None
        self.cursor.execute('SELECT road_segment_id FROM cctvs WHERE id = ?', (cctv_id,))
        row = self.cursor.fetchone()
        if row:
            road_segment_id = row['road_segment_id']
        
        self.cursor.execute('''
            INSERT INTO traffic_data 
            (cctv_id, road_segment_id, vehicle_count, vehicles_per_minute, cars, motorcycles, 
             buses, trucks, congestion_level, los, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            cctv_id,
            road_segment_id,
            data.get('vehicle_count', 0),
            data.get('vehicles_per_minute', 0),
            vehicle_types.get('car', 0),
            vehicle_types.get('motorcycle', 0),
            vehicle_types.get('bus', 0),
            vehicle_types.get('truck', 0),
            data.get('congestion_level', 'UNKNOWN'),
            data.get('los', '-'),
            json.dumps(data)
        ))
        self.conn.commit()
    
    def get_cctvs(self) -> List[Dict]:
        """Get all CCTVs"""
        self.cursor.execute('SELECT * FROM cctvs')
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_road_segments(self) -> List[Dict]:
        """Get all road segments"""
        self.cursor.execute('SELECT * FROM road_segments')
        rows = self.cursor.fetchall()
        roads = []
        for row in rows:
            road = dict(row)
            road['coordinates'] = json.loads(road['coordinates'] or '[]')
            roads.append(road)
        return roads
    
    def get_road_segments_with_traffic(self) -> List[Dict]:
        """Get road segments with latest traffic data"""
        self.cursor.execute('''
            SELECT r.*, 
                   t.vehicle_count,
                   t.vehicles_per_minute,
                   t.congestion_level AS traffic_congestion_level,
                   t.los AS traffic_los,
                   t.timestamp AS traffic_timestamp,
                   c.id AS cctv_id,
                   c.name AS cctv_name
            FROM road_segments r
            LEFT JOIN cctvs c ON c.road_segment_id = r.id
            LEFT JOIN (
                SELECT cctv_id, MAX(timestamp) as max_ts
                FROM traffic_data
                GROUP BY cctv_id
            ) latest ON c.id = latest.cctv_id
            LEFT JOIN traffic_data t ON t.cctv_id = latest.cctv_id 
                AND t.timestamp = latest.max_ts
        ''')
        
        rows = self.cursor.fetchall()
        roads = []
        for row in rows:
            road = dict(row)
            road['coordinates'] = json.loads(road.get('coordinates') or '[]')
            # Remap aliased columns to expected names
            road['congestion_level'] = road.pop('traffic_congestion_level', None) or 'UNKNOWN'
            road['los'] = road.pop('traffic_los', None) or '-'
            road['timestamp'] = road.pop('traffic_timestamp', None)
            roads.append(road)
        return roads
    
    def get_traffic_history(self, cctv_id: str, hours: int = 24) -> List[Dict]:
        """Get traffic history for a CCTV"""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        self.cursor.execute('''
            SELECT * FROM traffic_data
            WHERE cctv_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        ''', (cctv_id, since))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_latest_traffic(self, cctv_id: str) -> Optional[Dict]:
        """Get latest traffic data for a CCTV"""
        self.cursor.execute('''
            SELECT * FROM traffic_data
            WHERE cctv_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (cctv_id,))
        
        row = self.cursor.fetchone()
        return dict(row) if row else None
    
    def get_congestion_summary(self) -> Dict:
        """Get congestion summary across all roads"""
        self.cursor.execute('''
            SELECT congestion_level, COUNT(*) as count
            FROM traffic_data
            WHERE timestamp > datetime('now', '-1 hour')
            GROUP BY congestion_level
        ''')
        
        summary = {'total': 0, 'levels': {}}
        for row in self.cursor.fetchall():
            summary['levels'][row['congestion_level']] = row['count']
            summary['total'] += row['count']
        
        return summary
