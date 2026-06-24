"""
==============================================================================
  Neural Trajectory Planner — Tesla FSD / Waymo ChauffeurNet Inspired

  Tesla approach: Generate trajectory candidates → score with cost function
  Waymo approach: Predict other agents → plan ego trajectory

  Our simplified version:
  - Generate route waypoints along the road
  - Compute steering/speed targets toward next waypoint
  - Score trajectory quality for reward shaping

  This replaces the scattered waypoint logic across the old codebase.
==============================================================================
"""
import math
from typing import List, Optional, Tuple

import numpy as np


class RoutePlanner:
    """
    Generates and manages a route along the road network.
    Like Tesla's route planner: follow the road, hit waypoints.
    """

    def __init__(self, n_waypoints: int = 200, spacing: float = 10.0):
        self.n_waypoints = n_waypoints
        self.spacing = spacing
        self.waypoints: list = []     # list of (x, y, z)
        self.target_idx: int = 0
        self.prev_dist: float = 0.0

    def generate(self, start_wp, carla_module):
        """Generate route from a CARLA waypoint along the road."""
        self.waypoints = []
        self.target_idx = 0
        self.prev_dist = 0.0

        if start_wp is None:
            return

        current = start_wp
        for _ in range(self.n_waypoints):
            nexts = current.next(self.spacing)
            if not nexts:
                break
            # Prefer driving lanes
            chosen = nexts[0]
            for nw in nexts:
                if nw.lane_type in (carla_module.LaneType.Driving,
                                    carla_module.LaneType.Bidirectional):
                    chosen = nw
                    break
            loc = chosen.transform.location
            self.waypoints.append((loc.x, loc.y, loc.z))
            current = chosen

    def get_target(self) -> Optional[Tuple[float, float, float]]:
        """Get current target waypoint (x, y, z)."""
        if not self.waypoints or self.target_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.target_idx]

    def update(self, x: float, y: float) -> dict:
        """
        Update planner state with current ego position.
        Returns dict with route metrics for reward computation.
        """
        result = {
            "route_delta": 0.0,
            "route_waypoint_reached": False,
            "target_dx": 0.0,
            "target_dy": 0.0,
            "dist_to_target": 0.0,
        }

        if not self.waypoints or self.target_idx >= len(self.waypoints):
            return result

        tx, ty, tz = self.waypoints[self.target_idx]
        dx = tx - x
        dy = ty - y
        dist = math.sqrt(dx * dx + dy * dy)

        # Normalized direction to target
        if dist > 0.1:
            result["target_dx"] = dx / dist
            result["target_dy"] = dy / dist
        result["dist_to_target"] = min(dist / 30.0, 1.0)

        # Route progress delta (positive = getting closer)
        if self.prev_dist > 0:
            result["route_delta"] = self.prev_dist - dist
        self.prev_dist = dist

        # Advance if close enough
        if dist < 8.0 and self.target_idx < len(self.waypoints) - 1:
            self.target_idx += 1
            result["route_waypoint_reached"] = True
            # Reset prev_dist for new target
            if self.target_idx < len(self.waypoints):
                ntx, nty, _ = self.waypoints[self.target_idx]
                self.prev_dist = math.sqrt((ntx - x)**2 + (nty - y)**2)

        return result

    def get_steering_target(self, ego_x, ego_y, ego_fwd_x, ego_fwd_y):
        """
        Compute optimal steering direction toward route target.
        Used by anti-stall module to unstick the vehicle.

        Returns steer in [-1, 1].
        """
        target = self.get_target()
        if target is None:
            return 0.0

        tx, ty, _ = target
        dx = tx - ego_x
        dy = ty - ego_y

        # Cross product: fwd × target_dir → left/right
        cross = ego_fwd_x * dy - ego_fwd_y * dx
        # Dot product: facing target?
        dot = ego_fwd_x * dx + ego_fwd_y * dy

        if dot < 0:
            # Target behind us — hard turn
            return 1.0 if cross > 0 else -1.0
        else:
            # Target ahead — proportional steer
            return max(-1.0, min(1.0, cross * 0.05))

    def get_junction_ahead(self, wp) -> float:
        """Check if a junction is within 30m ahead."""
        if wp is None:
            return 0.0
        try:
            far = wp.next(30.0)
            if far and far[0].is_junction:
                return 1.0
        except Exception:
            pass
        return 0.0

    @property
    def progress(self) -> float:
        """Fraction of route completed [0..1]."""
        if not self.waypoints:
            return 0.0
        return self.target_idx / len(self.waypoints)
