"""
==============================================================================
  Leaderboard — Competition tracking and agent records.
==============================================================================
"""
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict

from .constants import LEADERBOARD_DIR


@dataclass
class AgentRecord:
    agent_id: str
    total_distance: float = 0.0
    total_penalty: float = 0.0
    collisions: int = 0
    off_roads: int = 0
    ego_wins: int = 0
    rival_wins: int = 0
    recoveries: int = 0
    waypoints_visited: int = 0
    episodes_completed: int = 0
    best_single_distance: float = 0.0
    safety_score: float = 100.0
    avg_episode_reward: float = 0.0
    shield_interventions: int = 0
    last_updated: str = ""

    @property
    def score(self) -> float:
        return max(0, self.total_distance - self.total_penalty * 0.1
                   - (100 - self.safety_score) * 10
                   + self.ego_wins * 500 + self.recoveries * 200)


class Leaderboard:
    def __init__(self):
        self.fp = os.path.join(LEADERBOARD_DIR, "leaderboard.json")
        self.records: Dict[str, AgentRecord] = {}
        self.load()

    def load(self):
        if os.path.exists(self.fp):
            try:
                with open(self.fp) as f:
                    for k, v in json.load(f).items():
                        self.records[k] = AgentRecord(**v)
            except Exception:
                pass

    def save(self):
        try:
            with open(self.fp, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.records.items()},
                          f, indent=2)
        except Exception:
            pass

    def update(self, agent_id: str, **kw):
        if agent_id not in self.records:
            self.records[agent_id] = AgentRecord(agent_id=agent_id)
        r = self.records[agent_id]
        for k, v in kw.items():
            if hasattr(r, k):
                setattr(r, k, v)
        r.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def print_board(self, top_n: int = 10):
        sr = sorted(self.records.items(), key=lambda x: x[1].score, reverse=True)
        print(f"\n{'=' * 90}")
        print(f"  CARLA AI LEADERBOARD v15".center(90))
        print(f"{'=' * 90}")
        header = (f"{'Rank':<6}{'Agent':<20}{'Score':<10}{'Distance':<12}"
                  f"{'Wins':<8}{'Recov':<8}{'Safety':<10}{'Shield':<8}")
        print(header)
        print("-" * 90)
        for i, (aid, r) in enumerate(sr[:top_n]):
            m = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i+1}."
            print(f"{m:<6}{aid[:20]:<20}{r.score:<10.1f}{r.total_distance:<12.1f}"
                  f"{r.ego_wins:<8}{r.recoveries:<8}{r.safety_score:<10.1f}"
                  f"{r.shield_interventions:<8}")
        print(f"{'=' * 90}\n")
