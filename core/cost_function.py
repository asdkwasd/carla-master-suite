"""
==============================================================================
  Tesla/Waymo-Style Cost Function — Multi-Objective Reward Decomposition.

  Instead of one giant reward, we compute SEPARATE cost channels:
    1. PROGRESS  — Are we making forward progress along the route?
    2. SAFETY    — Are we collision-free and on-road?
    3. COMFORT   — Are we driving smoothly (no swerving, good speed)?
    4. RULES     — Are we following traffic rules (red lights, lanes)?

  Each channel has a clear, interpretable meaning.
  Final reward = weighted sum of channels.

  This is how Waymo ChauffeurNet and Tesla Planner decompose their costs.
==============================================================================
"""
import math
import numpy as np


class CostFunction:
    """
    Multi-objective cost function.
    Returns (total_reward, cost_breakdown) per step.
    """

    # --- Weight schedule (can be tuned per curriculum stage) ---
    WEIGHTS = {
        "progress": 3.0,      # Primary objective
        "safety": 2.0,        # Non-negotiable
        "comfort": 0.5,       # Nice to have
        "rules": 1.0,         # Traffic law compliance
    }

    def __init__(self, reward_clip=(-30, 30)):
        self.clip_lo, self.clip_hi = reward_clip
        self.episode_costs = {"progress": 0, "safety": 0, "comfort": 0, "rules": 0}

    def reset(self):
        self.episode_costs = {"progress": 0, "safety": 0, "comfort": 0, "rules": 0}

    def compute(self, ctx: dict) -> tuple:
        """
        Compute reward from context dict.

        Required ctx keys:
            step_dist, on_road, kmh, steer, reverse,
            collisions, lane_invasions, red_light_running,
            route_delta (positive = closer to next waypoint),
            route_waypoint_reached (bool),
            wp (carla waypoint or None),
            center_dist (distance to lane center),
            heading_alignment (dot product with road direction),
            at_junction,
            was_offroad, recovery_success,
        """
        costs = {}

        # =====================================================================
        # 1. PROGRESS — Tesla: "drive toward the next waypoint"
        # =====================================================================
        progress = 0.0

        # Route following: reward getting closer to target (like GPS nav)
        route_delta = ctx.get("route_delta", 0.0)
        progress += route_delta * 2.0  # +2.0 per meter closer

        # Reaching a waypoint
        if ctx.get("route_waypoint_reached", False):
            progress += 1.0

        # Forward movement on road
        step_dist = ctx.get("step_dist", 0.0)
        if ctx.get("on_road", False) and step_dist > 0.05 and not ctx.get("reverse", False):
            progress += min(ctx.get("kmh", 0) / 30.0, 1.0)  # Speed bonus [0..1]

        # Standing still penalty
        if step_dist < 0.05:
            progress -= 0.3

        # New waypoint bonus
        if ctx.get("new_waypoint", False):
            progress += 0.5

        costs["progress"] = progress

        # =====================================================================
        # 2. SAFETY — Waymo: "zero tolerance for crashes"
        # =====================================================================
        safety = 0.0

        # Collision
        if ctx.get("collisions", 0) > 0:
            safety -= 5.0
            if ctx.get("kmh", 0) > 5:
                safety -= 2.0  # High-speed crash extra

        # Off-road
        if not ctx.get("on_road", True):
            safety -= 0.5

        # Recovery success (rare, valuable)
        if ctx.get("recovery_success", False):
            safety += 3.0

        # Reverse penalty (unless recovering)
        if ctx.get("reverse", False) and ctx.get("on_road", True):
            safety -= 0.5

        costs["safety"] = safety

        # =====================================================================
        # 3. COMFORT — Smooth driving, appropriate speed
        # =====================================================================
        comfort = 0.0

        # Lane centering
        center_dist = ctx.get("center_dist", 999)
        if center_dist < 1.5:
            comfort += 0.5
        elif center_dist < 3.0:
            comfort += 0.2

        # Heading alignment with road
        alignment = ctx.get("heading_alignment", 0.0)
        if alignment > 0:
            weight = 0.4 if ctx.get("at_junction", False) else 0.2
            comfort += alignment * weight

        # Junction speed management
        if ctx.get("at_junction", False):
            kmh = ctx.get("kmh", 0)
            if 5.0 < kmh < 25.0:
                comfort += 0.3  # Good junction speed
            elif kmh >= 25.0:
                comfort -= 0.3  # Too fast

        costs["comfort"] = comfort

        # =====================================================================
        # 4. RULES — Traffic law compliance
        # =====================================================================
        rules = 0.0

        # Red light violation
        if ctx.get("red_light_running", False):
            rules -= 1.0

        # Lane invasion
        if ctx.get("lane_invasions", 0) > 0:
            rules -= 0.5

        costs["rules"] = rules

        # =====================================================================
        # FINAL: weighted sum + clip
        # =====================================================================
        total = sum(self.WEIGHTS[k] * v for k, v in costs.items())
        total = float(np.clip(total, self.clip_lo, self.clip_hi))

        # NaN guard
        if np.isnan(total) or np.isinf(total):
            total = 0.0

        # Accumulate for episode stats
        for k, v in costs.items():
            self.episode_costs[k] += v

        return total, costs
