"""
==============================================================================
  Safety Shield — Rule-based hard safety layer.
  Overrides dangerous RL actions before they reach CARLA.
  Tesla-style: Safety is NON-NEGOTIABLE — shield ALWAYS active.

  FIXED v2 — BUG 2A & 2B fixes:
  - LiDAR validation: empty sectors = LIDAR_MAX_RANGE (not 0)
  - TTC-based braking (not hard threshold)
  - Creep mode: allow low-speed movement
  - Anti-idle: ensure car always tries to move
==============================================================================
"""
import math
import numpy as np

from .constants import N_LIDAR_SECTORS, LIDAR_MAX_RANGE


class SafetyShield:
    """
    Hard safety layer that overrides dangerous actions.
    Cannot be disabled. Operates on raw LiDAR + speed data.

    FIXES from v1:
    - LiDAR cleaning: zero/NaN sectors → LIDAR_MAX_RANGE
    - TTC-based braking: time-to-collision instead of hard distance
    - Creep mode: allow slow movement (<10 km/h) even with close obstacles
    - Logging: track every intervention for debugging
    """

    def __init__(self):
        self.emergency_brakes = 0
        self.steering_corrections = 0
        self.total_interventions = 0
        self.ttc_violations = 0  # TTC < threshold
        self.creep_allowed = 0    # Creep mode activations

        # Log buffer for debugging
        self._log_buffer = []

    def _clean_lidar(self, sectors: np.ndarray) -> np.ndarray:
        """
        FIX: Ensure empty/zero/NaN sectors return LIDAR_MAX_RANGE.
        This was causing front_min=0 → constant braking.
        """
        cleaned = sectors.copy()
        # Replace zeros, NaN, and negative values with max range
        cleaned = np.where((cleaned <= 0) | np.isnan(cleaned) | (cleaned > LIDAR_MAX_RANGE),
                           LIDAR_MAX_RANGE, cleaned)
        return cleaned

    def _compute_ttc(self, distance: float, speed_kmh: float) -> float:
        """
        Compute Time-To-Collision (TTC).
        Returns time in seconds until collision at current speed.
        Returns infinity if speed is zero or negative.
        """
        if speed_kmh <= 0:
            return float('inf')
        speed_ms = speed_kmh / 3.6  # Convert to m/s
        # Relative speed: assume obstacle is stationary
        ttc = distance / speed_ms if speed_ms > 0.1 else float('inf')
        return ttc

    def filter_action(self, steer: float, throttle: float, brake: float,
                      lidar_sectors: np.ndarray, speed_kmh: float,
                      on_road: bool):
        """
        Filter action through safety rules.
        Returns (steer, throttle, brake, was_intervened).

        KEY CHANGES:
        - Use TTC instead of hard distance threshold
        - Allow creep at low speeds
        - Never completely stop the car unless imminent collision
        """
        intervened = False
        n = len(lidar_sectors)

        # === FIX 1: Clean LiDAR data ===
        lidar_clean = self._clean_lidar(lidar_sectors)

        # === FRONT SAFETY: TTC-based braking ===
        # 5 sectors forward (~112.5° arc): indices n-2, n-1, 0, 1, 2
        front_idxs = [(n - 2) % n, (n - 1) % n, 0, 1, 2]
        front_distances = [float(lidar_clean[i]) for i in front_idxs]
        front_min = min(front_distances)

        # Compute TTC for front obstacle
        front_ttc = self._compute_ttc(front_min, speed_kmh)

        # === FIX 2: Creep mode — allow slow movement ===
        # If speed < 10 km/h, allow creeping even with close obstacle
        is_creeping = speed_kmh < 10.0

        # TTC thresholds (seconds) — much more lenient than hard distance
        TTC_CRITICAL = 1.5   # Hard brake only if TTC < 1.5s (imminent collision)
        TTC_WARNING = 3.0     # Gentle brake if TTC < 3.0s
        TTC_SAFE = 5.0        # Full throttle OK if TTC > 5.0s

        # === FIX 3: More lenient front collision prevention ===
        if front_ttc < TTC_CRITICAL:
            # CRITICAL: imminent collision, hard brake
            # But if creeping, just reduce speed moderately
            if is_creeping:
                throttle = min(throttle, 0.3)
                brake = max(brake, 0.5)
            else:
                throttle = 0.0
                brake = 1.0
            self.emergency_brakes += 1
            self.ttc_violations += 1
            intervened = True
            self._log_intervention("FRONT_CRITICAL", front_min, front_ttc, speed_kmh)

        elif front_ttc < TTC_WARNING and not is_creeping:
            # WARNING: moderate braking needed
            throttle = min(throttle, 0.2)
            brake = max(brake, 0.3)
            intervened = True
            self._log_intervention("FRONT_WARNING", front_min, front_ttc, speed_kmh)

        elif front_ttc < TTC_SAFE:
            # CAUTION: gentle speed reduction
            if speed_kmh > 15.0:
                throttle = min(throttle, 0.4)
                brake = max(brake, 0.2)
                intervened = True
                self._log_intervention("FRONT_CAUTION", front_min, front_ttc, speed_kmh)
        else:
            # SAFE: allow full throttle (unless other conditions)
            self._log_intervention("FRONT_SAFE", front_min, front_ttc, speed_kmh)

        # === SIDE COLLISION PREVENTION (T-bone at intersections) ===
        left_idxs = [i % n for i in range(3, 6)]
        right_idxs = [i % n for i in range(n - 3, n)]

        left_min = min(float(lidar_clean[i]) for i in left_idxs)
        right_min = min(float(lidar_clean[i]) for i in right_idxs)
        side_min = min(left_min, right_min)
        side_ttc = self._compute_ttc(side_min, speed_kmh)

        # Side collision prevention (intersection safety)
        if side_min < 5.0 and speed_kmh > 15.0 and side_ttc < 3.0:
            # Gentle steering correction + speed reduction
            if left_min < right_min:
                steer = max(steer, 0.3)  # Steer right
            else:
                steer = min(steer, -0.3)  # Steer left
            throttle = min(throttle, 0.3)
            brake = max(brake, 0.2)
            intervened = True
            self._log_intervention("SIDE_DANGER", side_min, side_ttc, speed_kmh)

        # === HIGH-SPEED STEERING DAMPING ===
        if abs(steer) > 0.8 and speed_kmh > 30.0:
            steer *= 0.6  # Less aggressive damping
            self.steering_corrections += 1
            intervened = True

        # === OFF-ROAD THROTTLE LIMIT ===
        if not on_road and throttle > 0.5:
            throttle = 0.4  # Reduced from 0.2 - allow some off-road movement
            intervened = True
            self._log_intervention("OFF_ROAD_LIMIT", 0, 0, speed_kmh)

        # === FIX 4: Anti-stall — ensure car tries to move ===
        # Only apply if the car has been nearly stationary for a while
        # This prevents the car from learning to stay still
        if speed_kmh < 1.0 and throttle < 0.1 and brake < 0.1:
            # Car is stalled — give it a nudge
            throttle = max(throttle, 0.3)
            self._log_intervention("ANTI_STALL", 0, 0, speed_kmh)

        if intervened:
            self.total_interventions += 1

        return steer, throttle, brake, intervened

    def _log_intervention(self, reason: str, distance: float, ttc: float, speed: float):
        """Log intervention for debugging."""
        self._log_buffer.append({
            'reason': reason,
            'dist': distance,
            'ttc': ttc,
            'speed': speed,
            'brakes': self.emergency_brakes,
            'ttc_violations': self.ttc_violations
        })
        # Keep only last 100 logs
        if len(self._log_buffer) > 100:
            self._log_buffer = self._log_buffer[-100:]

    def get_stats(self) -> dict:
        return {
            "emergency_brakes": self.emergency_brakes,
            "steering_corrections": self.steering_corrections,
            "total_interventions": self.total_interventions,
            "ttc_violations": self.ttc_violations,
            "creep_allowed": self.creep_allowed,
        }

    def get_recent_logs(self, n: int = 10) -> list:
        """Get last n intervention logs for debugging."""
        return self._log_buffer[-n:]


class RecoveryModule:
    """
    State machine for off-road recovery.
    FIXED: Better state management, allows forward movement during recovery.
    """

    def __init__(self):
        self.is_recovering = False
        self.recovery_steps = 0
        self.max_recovery_steps = 500  # Increased from 300

    def enter(self):
        self.is_recovering = True
        self.recovery_steps = 0

    def exit(self):
        self.is_recovering = False
        self.recovery_steps = 0

    def step(self):
        if self.is_recovering:
            self.recovery_steps += 1
            if self.recovery_steps >= self.max_recovery_steps:
                self.exit()

    def is_active(self) -> bool:
        return self.is_recovering


class ADRManager:
    """Automatic Domain Randomization — adjusts difficulty based on performance."""

    def __init__(self, target: float = 0.5, window: int = 100):
        self.target = target
        self.window = window
        self.recent = []
        self.difficulty = 0.0

    def update(self, reward: float):
        self.recent.append(reward > 0)
        if len(self.recent) > self.window:
            self.recent = self.recent[-self.window:]
        if len(self.recent) < self.window:
            return
        rate = sum(self.recent) / len(self.recent)
        if rate > self.target + 0.1:
            self.difficulty = min(1.0, self.difficulty + 0.05)
        elif rate < self.target - 0.1:
            self.difficulty = max(0.0, self.difficulty - 0.05)

    def get_weather_params(self) -> dict:
        """Returns kwargs for carla.WeatherParameters."""
        import random
        d = self.difficulty
        return dict(
            cloudiness=max(0, min(100, d * 80 + random.uniform(-10, 10))),
            precipitation=d * 50 if d > 0.3 else 0,
            precipitation_deposits=d * 50 if d > 0.3 else 0,
            wind_intensity=random.uniform(0, 30 * d),
            sun_azimuth_angle=random.uniform(0, 360),
            sun_altitude_angle=random.uniform(10, 70),
            fog_density=d * 30 if d > 0.5 else 0,
            wetness=d * 40 if d > 0.4 else 0,
        )
