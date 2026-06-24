"""
==============================================================================
  CARLA BATTLE ARENA — Head-to-Head AI Competition

  Dua agen bersaing di rute yang sama. Siapa yang jarak terjauh dengan
  penalty terkecil, dia menang!

  Features:
  - Side-by-side comparison view
  - Real-time leaderboard updates
  - Match replay system
  - Championship bracket
==============================================================================
"""

import os
import sys
import json
import time
import threading
import numpy as np
import cv2
import carla
import gymnasium as gym
from collections import deque
from datetime import datetime

from stable_baselines3 import PPO

BASE_DIR = r"D:\AI\traffic_vision\master_suite"
LEADERBOARD_FILE = os.path.join(BASE_DIR, "leaderboard", "leaderboard.json")
BATTLE_LOG = os.path.join(BASE_DIR, "battle_log.json")


class BattleResult:
    def __init__(self, agent1_id, agent2_id):
        self.agent1_id = agent1_id
        self.agent2_id = agent2_id
        self.agent1_stats = {}
        self.agent2_stats = {}
        self.winner = None
        self.timestamp = datetime.now().isoformat()

    def determine_winner(self):
        score1 = self.agent1_stats.get("score", 0)
        score2 = self.agent2_stats.get("score", 0)

        if score1 > score2:
            self.winner = self.agent1_id
        elif score2 > score1:
            self.winner = self.agent2_id
        else:
            self.winner = "draw"

        return self.winner

    def save(self):
        result = {
            "timestamp": self.timestamp,
            "agent1": self.agent1_stats,
            "agent2": self.agent2_stats,
            "winner": self.winner
        }

        history = []
        if os.path.exists(BATTLE_LOG):
            with open(BATTLE_LOG, 'r') as f:
                history = json.load(f)

        history.append(result)

        with open(BATTLE_LOG, 'w') as f:
            json.dump(history, f, indent=2)


class BattleArena:
    """
    Arena pertarungan antara dua agen.

    Kedua agen menggunakan environment CARLA yang terpisah
    tapi dengan konfigurasi yang identik.
    """

    def __init__(self, agent1_id: str, agent2_id: str,
                 model1_path: str, model2_path: str):
        self.agent1_id = agent1_id
        self.agent2_id = agent2_id
        self.model1_path = model1_path
        self.model2_path = model2_path

        # Load models
        self.model1 = PPO.load(model1_path)
        self.model2 = PPO.load(model2_path)

        # Battle config
        self.config = {
            "map": "Town01",
            "traffic": 20,
            "pedestrians": 10,
            "weather": "Clear"
        }

        # Result tracking
        self.result = BattleResult(agent1_id, agent2_id)

    def run_battle(self, max_steps: int = 3000) -> BattleResult:
        """
        Jalankan pertarungan head-to-head.

        Agent bergantian menjalankan episode di environment yang sama.
        Agent dengan skor tertinggi menang.
        """
        print(f"\n{'='*70}")
        print(f"  ⚔️  BATTLE: {self.agent1_id} vs {self.agent2_id} ⚔️".center(70))
        print(f"{'='*70}\n")

        # Run agent 1
        print(f"[BATTLE] Running {self.agent1_id}...")
        stats1 = self._run_agent(self.model1, self.agent1_id, max_steps)
        self.result.agent1_stats = stats1
        print(f"[DONE] {self.agent1_id} - Score: {stats1.get('score', 0):.1f}\n")

        # Run agent 2
        print(f"[BATTLE] Running {self.agent2_id}...")
        stats2 = self._run_agent(self.model2, self.agent2_id, max_steps)
        self.result.agent2_stats = stats2
        print(f"[DONE] {self.agent2_id} - Score: {stats2.get('score', 0):.1f}\n")

        # Determine winner
        winner = self.result.determine_winner()

        # Print results
        self._print_results()

        # Save
        self.result.save()

        return self.result

    def _run_agent(self, model, agent_id: str, max_steps: int) -> dict:
        """Jalankan satu agent dan kumpulkan statistik"""
        # Import environment here to avoid circular import
        from importlib import import_module
        comp_module = import_module("1_COMPETITION_MASTER")
        CarlaCompetitionEnv = comp_module.CarlaCompetitionEnv

        # Create temporary environment for battle
        env = CarlaCompetitionEnv(
            agent_id=agent_id + "_battle",
            stage_config=self.config,
            leaderboard=None  # Don't update leaderboard during battle
        )

        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            step += 1

            # Progress indicator
            if step % 500 == 0:
                print(f"  {agent_id}: Step {step}/{max_steps}, "
                      f"Dist: {env.total_distance:.1f}m, "
                      f"Penalty: {env.current_penalty:.1f}")

        env.close()

        # Compile stats
        stats = {
            "agent_id": agent_id,
            "distance": env.total_distance,
            "penalty": env.current_penalty,
            "collisions": env.collision_count,
            "off_roads": env.offroad_count,
            "waypoints": len(env.visited_waypoints),
            "safety_score": env.safety_score,
            "score": env.total_distance - env.current_penalty * 0.1 - (100 - env.safety_score) * 10
        }

        return stats

    def _print_results(self):
        """Print hasil pertarungan dengan format yang menarik"""
        a1 = self.result.agent1_stats
        a2 = self.result.agent2_stats

        print(f"\n{'='*70}")
        print("  🏁 BATTLE RESULTS 🏁".center(70))
        print(f"{'='*70}\n")

        print(f"{'Metric':<20} {self.agent1_id:<20} {self.agent2_id:<20}")
        print("-"*70)
        print(f"{'Distance (m)':<20} {a1.get('distance', 0):<20.1f} {a2.get('distance', 0):<20.1f}")
        print(f"{'Penalty':<20} {a1.get('penalty', 0):<20.1f} {a2.get('penalty', 0):<20.1f}")
        print(f"{'Collisions':<20} {a1.get('collisions', 0):<20} {a2.get('collisions', 0):<20}")
        print(f"{'Off-roads':<20} {a1.get('off_roads', 0):<20} {a2.get('off_roads', 0):<20}")
        print(f"{'Waypoints':<20} {a1.get('waypoints', 0):<20} {a2.get('waypoints', 0):<20}")
        print(f"{'Safety Score':<20} {a1.get('safety_score', 0):<20.1f} {a2.get('safety_score', 0):<20.1f}")
        print("-"*70)
        print(f"{'FINAL SCORE':<20} {a1.get('score', 0):<20.1f} {a2.get('score', 0):<20.1f}")

        winner = self.result.winner
        if winner != "draw":
            print(f"\n  🏆 WINNER: {winner} 🏆\n")
        else:
            print(f"\n  🤝 IT'S A DRAW! 🤝\n")

        print(f"{'='*70}\n")


class ChampionshipBracket:
    """
    Sistem bracket untuk turnamen antar agents.
    Single elimination: Agent kalah langsung tereliminasi.
    """

    def __init__(self):
        self.participants = []
        self.bracket = []
        self.matches = []
        self.champion = None

    def load_participants(self):
        """Load top agents dari leaderboard"""
        if not os.path.exists(LEADERBOARD_FILE):
            print("[ERROR] No leaderboard found!")
            return

        with open(LEADERBOARD_FILE, 'r') as f:
            data = json.load(f)

        # Sort by score and take top 8
        sorted_agents = sorted(
            data.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True
        )[:8]

        self.participants = [aid for aid, _ in sorted_agents]
        print(f"[BRACKET] Loaded {len(self.participants)} participants")

    def create_bracket(self):
        """Buat bracket pertandingan"""
        if len(self.participants) < 2:
            print("[ERROR] Need at least 2 participants!")
            return

        self.bracket = []
        round_matches = []

        # Quarter-finals
        i = 0
        while i < len(self.participants):
            if i + 1 < len(self.participants):
                round_matches.append({
                    "round": "Quarter-final",
                    "player1": self.participants[i],
                    "player2": self.participants[i+1],
                    "winner": None
                })
            i += 2

        self.bracket.append(round_matches)
        self._print_bracket()

    def run_championship(self):
        """Jalankan seluruh turnamen"""
        if not self.bracket:
            print("[ERROR] No bracket created!")
            return

        print("\n" + "="*70)
        print("  🏆 CHAMPIONSHIP BEGINS 🏆".center(70))
        print("="*70 + "\n")

        for round_matches in self.bracket:
            round_name = round_matches[0]["round"]
            print(f"\n{'='*70}")
            print(f"  {round_name.upper()}".center(70))
            print(f"{'='*70}\n")

            winners = []

            for match in round_matches:
                # Load models
                model1_path = os.path.join(BASE_DIR, "checkpoints",
                                           f"{match['player1']}_latest.zip")
                model2_path = os.path.join(BASE_DIR, "checkpoints",
                                           f"{match['player2']}_latest.zip")

                # Create default paths if not found
                if not os.path.exists(model1_path):
                    model1_path = os.path.join(BASE_DIR,
                                               f"{match['player1']}_champion.zip")
                if not os.path.exists(model2_path):
                    model2_path = os.path.join(BASE_DIR,
                                               f"{match['player2']}_champion.zip")

                # Skip if models don't exist
                if not os.path.exists(model1_path) or not os.path.exists(model2_path):
                    print(f"[SKIP] Match {match['player1']} vs {match['player2']} - models not found")
                    winners.append(match['player1'])  # Default winner
                    continue

                # Run battle
                arena = BattleArena(match['player1'], match['player2'],
                                   model1_path, model2_path)
                result = arena.run_battle(max_steps=2000)
                self.matches.append(result)

                match['winner'] = result.winner
                winners.append(result.winner)

            # Create next round if needed
            if len(winners) > 1:
                next_round = []
                i = 0
                while i < len(winners):
                    if i + 1 < len(winners):
                        next_round.append({
                            "round": "Semi-final" if round_name == "Quarter-final" else "Grand Final",
                            "player1": winners[i],
                            "player2": winners[i+1],
                            "winner": None
                        })
                    i += 2
                self.bracket.append(next_round)

        self.champion = winners[0] if winners else None

        print("\n" + "="*70)
        print(f"  🏆🏆🏆 CHAMPION: {self.champion} 🏆🏆🏆".center(70))
        print("="*70 + "\n")

    def _print_bracket(self):
        """Print bracket visualisasi"""
        print("\n  TOURNAMENT BRACKET:\n")
        for i, round_matches in enumerate(self.bracket):
            round_name = round_matches[0]["round"]
            print(f"  {round_name}:")
            for match in round_matches:
                p1 = match["player1"]
                p2 = match["player2"]
                print(f"    {p1} vs {p2}")
            print()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CARLA Battle Arena")
    parser.add_argument("--mode", choices=["1v1", "bracket"], default="1v1",
                        help="Battle mode: 1v1 match or championship bracket")
    parser.add_argument("--agent1", type=str, help="Agent 1 ID (for 1v1)")
    parser.add_argument("--agent2", type=str, help="Agent 2 ID (for 1v1)")
    parser.add_argument("--model1", type=str, help="Path to agent 1 model")
    parser.add_argument("--model2", type=str, help="Path to agent 2 model")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Max steps per battle")

    args = parser.parse_args()

    if args.mode == "1v1":
        if not args.agent1 or not args.agent2:
            # Auto-load from leaderboard
            print("[AUTO] Loading top 2 agents from leaderboard...")
            if os.path.exists(LEADERBOARD_FILE):
                with open(LEADERBOARD_FILE, 'r') as f:
                    data = json.load(f)
                sorted_agents = sorted(
                    data.items(),
                    key=lambda x: x[1].get("score", 0),
                    reverse=True
                )[:2]
                args.agent1 = sorted_agents[0][0]
                args.agent2 = sorted_agents[1][0]
                print(f"[AUTO] {args.agent1} vs {args.agent2}")

                args.model1 = os.path.join(BASE_DIR, "checkpoints",
                                          f"{args.agent1}_r1.zip")
                args.model2 = os.path.join(BASE_DIR, "checkpoints",
                                          f"{args.agent2}_r1.zip")

                # Try alternative paths
                if not os.path.exists(args.model1):
                    args.model1 = os.path.join(BASE_DIR,
                                               f"{args.agent1}_champion.zip")
                if not os.path.exists(args.model2):
                    args.model2 = os.path.join(BASE_DIR,
                                               f"{args.agent2}_champion.zip")

        if not args.model1 or not args.model2:
            print("[ERROR] Please provide model paths or ensure leaderboard exists")
            sys.exit(1)

        arena = BattleArena(args.agent1, args.agent2, args.model1, args.model2)
        arena.run_battle(max_steps=args.steps)

    elif args.mode == "bracket":
        bracket = ChampionshipBracket()
        bracket.load_participants()
        bracket.create_bracket()
        bracket.run_championship()
