"""
==============================================================================
  Recall Memory — Persistent spatial knowledge base.
  Fixed: JSON-safe serialization, bounded growth, deduplication.
==============================================================================
"""
import os
import json
import math
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict

from .constants import MEMORY_DIR


class RecallMemory:
    """
    Long-term spatial memory for an agent.
    Tracks collision hotspots, off-road zones, red-light zones,
    and successful route waypoints. Persists to JSON.
    """

    def __init__(self, agent_id: str = "ego", capacity: int = 500):
        self.agent_id = agent_id
        self.fp = os.path.join(MEMORY_DIR, f"{agent_id}_memory.json")
        self.capacity = capacity

        self.collision_hotspots: List[Dict] = []
        self.offroad_zones: List[Dict] = []
        self.red_light_zones: List[Dict] = []
        self.successful_route: List[Dict] = []
        self.load()

    # =========================================================================
    # PERSISTENCE (JSON-safe — no numpy, no carla objects)
    # =========================================================================
    def load(self):
        if os.path.exists(self.fp):
            try:
                with open(self.fp) as f:
                    d = json.load(f)
                self.collision_hotspots = d.get("collision_hotspots", [])
                self.offroad_zones = d.get("offroad_zones", [])
                self.red_light_zones = d.get("red_light_zones", [])
                self.successful_route = d.get("successful_route", [])
            except Exception:
                pass

    def save(self):
        data = {
            "collision_hotspots": self.collision_hotspots[-self.capacity:],
            "offroad_zones": self.offroad_zones[-self.capacity:],
            "red_light_zones": self.red_light_zones[-self.capacity:],
            "successful_route": self.successful_route[-self.capacity * 5:],
        }
        try:
            with open(self.fp, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass

    # =========================================================================
    # ADD EVENTS (accepts x, y floats — NOT carla.Location)
    # =========================================================================
    def add_collision(self, x: float, y: float, z: float = 0.0):
        """Record collision. Merges with nearby existing hotspot."""
        for h in self.collision_hotspots:
            if math.sqrt((h["x"] - x)**2 + (h["y"] - y)**2) < 5.0:
                h["count"] += 1
                h["last_seen"] = datetime.now().isoformat()
                self.save()
                return
        self.collision_hotspots.append({
            "x": float(x), "y": float(y), "z": float(z),
            "count": 1, "last_seen": datetime.now().isoformat()
        })
        if len(self.collision_hotspots) > self.capacity:
            self.collision_hotspots = self.collision_hotspots[-self.capacity:]
        self.save()

    def add_offroad(self, x: float, y: float):
        self.offroad_zones.append({
            "x": float(x), "y": float(y), "radius": 8.0,
            "last_seen": datetime.now().isoformat()
        })
        if len(self.offroad_zones) > self.capacity:
            self.offroad_zones = self.offroad_zones[-self.capacity:]
        self.save()

    def add_red_light(self, x: float, y: float):
        self.red_light_zones.append({
            "x": float(x), "y": float(y), "danger": 1.0,
            "last_seen": datetime.now().isoformat()
        })
        if len(self.red_light_zones) > self.capacity:
            self.red_light_zones = self.red_light_zones[-self.capacity:]
        self.save()

    def add_waypoint(self, x: float, y: float, z: float, seq: int):
        self.successful_route.append({
            "x": float(x), "y": float(y), "z": float(z), "sequence": seq
        })

    # =========================================================================
    # QUERY
    # =========================================================================
    def get_danger_score(self, x: float, y: float) -> float:
        """Compute danger [0..1] at (x, y) from all memory banks."""
        danger = 0.0
        for h in self.collision_hotspots:
            d = math.sqrt((h["x"] - x)**2 + (h["y"] - y)**2)
            if d < 10.0:
                danger += h.get("count", 1) * (1.0 - d / 10.0) * 0.4
        for z in self.offroad_zones:
            d = math.sqrt((z["x"] - x)**2 + (z["y"] - y)**2)
            r = z.get("radius", 8.0)
            if d < r:
                danger += (1.0 - d / r) * 0.3
        for z in self.red_light_zones:
            d = math.sqrt((z["x"] - x)**2 + (z["y"] - y)**2)
            if d < 15.0:
                danger += 0.2
        return min(danger, 1.0)

    def get_route_guidance(self, x: float, y: float):
        """Find next waypoint on best route near (x, y). Returns (gx, gy) or None."""
        if not self.successful_route:
            return None
        best_seq, best_dist = -1, float('inf')
        for w in self.successful_route:
            d = math.sqrt((w["x"] - x)**2 + (w["y"] - y)**2)
            if d < best_dist:
                best_dist = d
                best_seq = w["sequence"]
        if best_seq >= 0 and best_dist < 20.0:
            for w in self.successful_route:
                if w["sequence"] == best_seq + 1:
                    return (w["x"], w["y"], w["z"])
        return None


def load_super_memory(target: RecallMemory):
    """Load merged knowledge from all previous training sessions."""
    super_path = os.path.join(MEMORY_DIR, "super_memory.json")
    if not os.path.exists(super_path):
        return
    try:
        with open(super_path) as f:
            sm = json.load(f)
        for h in sm.get("collision_hotspots", []):
            target.collision_hotspots.append(h)
        for z in sm.get("offroad_zones", []):
            target.offroad_zones.append(z)
        for z in sm.get("red_light_zones", []):
            target.red_light_zones.append(z)
        for w in sm.get("successful_route", []):
            target.successful_route.append(w)
        target.save()
        print(f"[MEMORY] Loaded super memory: "
              f"{len(sm.get('collision_hotspots', []))} hotspots, "
              f"{len(sm.get('offroad_zones', []))} offroad, "
              f"{len(sm.get('red_light_zones', []))} redlights, "
              f"{len(sm.get('successful_route', []))} waypoints")
    except Exception as e:
        print(f"[MEMORY] Failed to load super memory: {e}")
