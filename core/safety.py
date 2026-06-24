"""
==============================================================================
  Safety Shield — Rule-based hard safety layer.
  Overrides dangerous RL actions before they reach CARLA.
  Tesla-style: Safety is NON-NEGOTIABLE — shield ALWAYS active.
==============================================================================
"""
import math
import numpy as np

from .constants import N_LIDAR_SECTORS


class SafetyShield:
    """
    Hard safety layer that overrides dangerous actions.
    Cannot be disabled. Operates on raw LiDAR + speed data.

    Sectors layout (N=16, 22.5° each):
      Sector 0    = forward (+X in sensor frame)
      Sector N/4  = left (+Y)
      Sector N/2  = behind
      Sector 3N/4 = right
    """

    def __init__(self):
        self.emergency_brakes = 0
        self.steering_corrections = 0
        self.total_interventions = 0

    def filter_action(self, steer: float, throttle: float, brake: float,
                      lidar_sectors: np.ndarray, speed_kmh: float,
                      on_road: bool):
        """
        Filter action through safety rules.
        Returns (steer, throttle, brake, was_intervened).
        """
        intervened = False
        n = len(lidar_sectors)

        # --- FRONT EMERGENCY BRAKE ---
        # ~67° forward arc: [n-2, n-1, 0, 1, 2]
        front_idxs = [(n - 2) % n, (n - 1) % n, 0, 1, 2]
        front_min = float(min(lidar_sectors[i] for i in front_idxs))

        if front_min < 5.0 and speed_kmh > 5.0:
            throttle = 0.0
            brake = 1.0
            self.emergency_brakes += 1
            intervened = True
        elif front_min < 10.0 and speed_kmh > 20.0:
            throttle = min(throttle, 0.2)
            brake = max(brake, 0.5)
            intervened = True

        # --- SIDE COLLISION PREVENTION (T-bone at intersections) ---
        left_idxs = [i % n for i in range(3, 6)]
        right_idxs = [i % n for i in range(n - 3, n)]
        left_min = float(min(lidar_sectors[i] for i in left_idxs))
        right_min = float(min(lidar_sectors[i] for i in right_idxs))
        side_min = min(left_min, right_min)

        if side_min < 3.0 and speed_kmh > 10.0:
            throttle = min(throttle, 0.3)
            brake = max(brake, 0.3)
            intervened = True

        # --- HIGH-SPEED STEERING DAMPING ---
        if abs(steer) > 0.8 and speed_kmh > 25.0:
            steer *= 0.5
            self.steering_corrections += 1
            intervened = True

        # --- OFF-ROAD THROTTLE LIMIT ---
        if not on_road and throttle > 0.5:
            throttle = 0.2
            intervened = True

        if intervened:
            self.total_interventions += 1

        return steer, throttle, brake, intervened

    def get_stats(self) -> dict:
        return {
            "emergency_brakes": self.emergency_brakes,
            "steering_corrections": self.steering_corrections,
            "total_interventions": self.total_interventions,
        }


class RecoveryModule:
    """State machine for off-road recovery."""

    def __init__(self):
        self.is_recovering = False
        self.recovery_steps = 0

    def enter(self):
        self.is_recovering = True
        self.recovery_steps = 0

    def exit(self):
        self.is_recovering = False
        self.recovery_steps = 0

    def step(self):
        if self.is_recovering:
            self.recovery_steps += 1


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
