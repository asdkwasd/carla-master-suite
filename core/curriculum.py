"""
==============================================================================
  Curriculum Manager — Per-scenario tracking with auto-advancement.
==============================================================================
"""
import numpy as np
from collections import deque, defaultdict

from .constants import CURRICULUM_STAGES, CURRICULUM_WINDOW


class CurriculumManager:
    """
    4-stage curriculum with per-scenario difficulty tracking.
    Auto-advances when rolling mean reward exceeds threshold.
    """

    def __init__(self, stage: int = 1):
        self.current_stage = max(1, min(stage, max(CURRICULUM_STAGES)))
        self._rewards = deque(maxlen=CURRICULUM_WINDOW)
        self.scenario_stats = defaultdict(
            lambda: {"success": deque(maxlen=50), "difficulty": 0.0}
        )

    def get_config(self) -> dict:
        return CURRICULUM_STAGES[self.current_stage]

    def detect_scenario(self, wp, kmh: float) -> str:
        if wp is None:
            return "off_road"
        if wp.is_junction:
            return "intersection"
        if kmh > 40:
            return "high_speed"
        return "straight_road"

    def record_scenario(self, scenario_type: str, success: bool):
        stats = self.scenario_stats[scenario_type]
        stats["success"].append(float(success))
        if len(stats["success"]) >= 20:
            rate = np.mean(stats["success"])
            if rate > 0.8:
                stats["difficulty"] = min(1.0, stats["difficulty"] + 0.05)
            elif rate < 0.4:
                stats["difficulty"] = max(0.0, stats["difficulty"] - 0.05)

    def record_episode(self, total_reward: float) -> bool:
        """Record episode reward. Returns True if stage advanced."""
        self._rewards.append(total_reward)
        if len(self._rewards) < CURRICULUM_WINDOW:
            return False
        mean = np.mean(self._rewards)
        threshold = CURRICULUM_STAGES[self.current_stage]["threshold"]
        if mean >= threshold and self.current_stage < max(CURRICULUM_STAGES):
            self.current_stage += 1
            self._rewards.clear()
            print(f"\n[CURRICULUM] >>> STAGE {self.current_stage}: "
                  f"{CURRICULUM_STAGES[self.current_stage]['name']} <<<\n")
            return True
        return False
