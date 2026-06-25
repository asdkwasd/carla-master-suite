"""
==============================================================================
  Tesla/Waymo-Style Cost Function — Multi-Objective Reward Decomposition.

  FIXED v2 — BUG 2B fixes:
  - Strong anti-idle penalty: moving > staying always
  - Forward-progress bonus: reward speed along route
  - Reduced collision penalty: avoid reward hacking to idle
  - Waypoint reward: encourage exploration
  - Anti-stuck bonus: reward for recovering from stuck state

  Each channel has clear, interpretable meaning.
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

    FIXED v2 — Reward shaping to prevent idle behavior:
    - Anti-idle: -1.0 per step if barely moving
    - Progress: +speed_bonus when moving forward
    - Collision: reduced from -5 to -2 (avoid idle hacking)
    - Waypoint: +5.0 per new waypoint reached
    """

    # --- Weight schedule (tuned for Stage 1 = just drive forward) ---
    WEIGHTS = {
        "progress": 5.0,     # PRIMARY: encourage forward movement
        "safety": 2.0,       # Secondary: collision avoidance
        "comfort": 0.5,      # Nice to have: smooth driving
        "rules": 1.0,        # Traffic rules compliance
    }

    # === FIX v3: Strong moving reward to beat reverse ===
    # Anti-idle penalty: only when IDLE (not moving at all)
    ANTI_IDLE_PENALTY = -2.0   # Reduced from -5.0
    IDLE_THRESHOLD_MPS = 0.05   # < 0.05 m/s = considered idle (very lenient)
    MOVING_REWARD_PER_MPS = 10.0  # MUCH STRONGER: 10x (was 3.0)
    STEP_MOVING_BONUS = 1.0    # Extra bonus per step when moving

    # Collision penalty (reduced to avoid idle hacking)
    COLLISION_PENALTY = -2.0      # Was -5, reduced to -2
    HIGH_SPEED_CRASH_BONUS = -1.0 # Extra penalty for high-speed crash

    # Waypoint reward
    WAYPOINT_REWARD = 5.0         # Per new waypoint reached

    def __init__(self, reward_clip=(-50, 100)):  # Wider clip range
        self.clip_lo, self.clip_hi = reward_clip
        self.episode_costs = {"progress": 0, "safety": 0, "comfort": 0, "rules": 0}
        self._waypoints_reached = 0

    def reset(self):
        self.episode_costs = {"progress": 0, "safety": 0, "comfort": 0, "rules": 0}
        self._waypoints_reached = 0

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
            new_waypoint (bool),
        """
        costs = {}

        # =====================================================================
        # 1. PROGRESS — Tesla: "drive toward the next waypoint"
        # =====================================================================
        progress = 0.0

        # Route following: reward getting closer to target
        route_delta = ctx.get("route_delta", 0.0)
        progress += route_delta * 5.0  # Strong route-following reward

        # Reaching a waypoint (significant reward)
        if ctx.get("route_waypoint_reached", False):
            progress += 5.0

        # === FIX v3: MUCH stronger moving reward ===
        # Forward motion must ALWAYS be better than reverse or idle
        step_dist = ctx.get("step_dist", 0.0)
        speed_ms = ctx.get("kmh", 0) / 3.6

        if ctx.get("on_road", False):
            if not ctx.get("reverse", False) and speed_ms > self.IDLE_THRESHOLD_MPS:
                # Moving forward: HUGE reward that dominates all penalties
                # At 0.5 m/s: +5.0 — beats -5.0 idle penalty by 10x margin
                progress += min(speed_ms * self.MOVING_REWARD_PER_MPS, 10.0)
                progress += self.STEP_MOVING_BONUS
            elif speed_ms <= self.IDLE_THRESHOLD_MPS and ctx.get("step_dist", 0) < 0.02:
                # Truly stuck: moderate penalty
                progress += self.ANTI_IDLE_PENALTY

        # === FIX v3: Waypoint discovery bonus ===
        if ctx.get("new_waypoint", False):
            progress += self.WAYPOINT_REWARD
            self._waypoints_reached += 1
            progress += 1.0

        costs["progress"] = progress

        # =====================================================================
        # 2. SAFETY — Waymo: "zero tolerance for crashes"
        # =====================================================================
        safety = 0.0

        # === FIX v2: Reduced collision penalty ===
        # Original: -5 to -7 per collision (too punishing, encourages idle)
        # New: -2 to -3 per collision (still punishing but not game-breaking)
        n_collisions = ctx.get("collisions", 0)
        if n_collisions > 0:
            safety += self.COLLISION_PENALTY * n_collisions
            if ctx.get("kmh", 0) > 10:  # High-speed crash only
                safety += self.HIGH_SPEED_CRASH_BONUS * n_collisions

        # Off-road penalty (reduced)
        if not ctx.get("on_road", True):
            safety -= 0.3  # Was -0.5

        # Recovery success (rare, valuable)
        if ctx.get("recovery_success", False):
            safety += 5.0  # Increased from 3.0

        # Reverse penalty (unless recovering)
        if ctx.get("reverse", False) and ctx.get("on_road", True):
            safety -= 0.3  # Was -0.5

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
            rules -= 0.3  # Was -0.5

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
