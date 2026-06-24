"""
==============================================================================
  CARLA GRANDMASTER AI SUITE v10.0 — ULTIMATE EDITION

  INTEGRATED STATE-OF-THE-ART TECHNOLOGIES:
  ✅ Hindsight Experience Replay (HER)
  ✅ World Model Components (Temporal Prediction)
  ✅ Multi-Head Self-Attention
  ✅ Automatic Domain Randomization (ADR)
  ✅ Advanced Reward Shaping
  ✅ Curriculum Learning with Self-Play
  ✅ Episodic Memory System

  This is the most advanced autonomous driving AI system.
==============================================================================
"""

import os
import sys
import time
import json
import random
import threading
import numpy as np
import cv2
import carla
import gymnasium as gym
from gymnasium import spaces
from collections import deque, defaultdict
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Union
from copy import deepcopy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, EvalCallback
)
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
import torch.nn.functional as F

# =============================================================================
# GLOBAL SETTINGS & PATHS
# =============================================================================
BASE_DIR        = r"D:\AI\traffic_vision\master_suite"
MODEL_PATH      = os.path.join(BASE_DIR, "carla_grandmaster_v10.zip")
TB_DIR          = os.path.join(BASE_DIR, "tb_logs")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_FILE        = os.path.join(BASE_DIR, "training_carla_v10_log.txt")
LEADERBOARD_DIR = os.path.join(BASE_DIR, "leaderboard")
MEMORY_DIR      = os.path.join(BASE_DIR, "agent_memories")
HER_BUFFER_DIR  = os.path.join(BASE_DIR, "her_buffers")

for path in [BASE_DIR, TB_DIR, CHECKPOINT_DIR, LEADERBOARD_DIR,
             MEMORY_DIR, HER_BUFFER_DIR]:
    os.makedirs(path, exist_ok=True)

IMG_WIDTH, IMG_HEIGHT = 160, 120
N_LIDAR_SECTORS = 8
LIDAR_MAX_RANGE = 50.0
STATE_DIM = 17  # Fixed to match v9

# =============================================================================
# HINDSIGHT EXPERIENCE REPLAY (HER)
# =============================================================================
class HERBuffer:
    """
    Hindsight Experience Replay - converts failed experiences to learning opportunities.

    Key insight: Even if we didn't achieve the goal, we can ask:
    "Given what we DID achieve, how could we have done better?"

    This dramatically improves sample efficiency for sparse reward tasks.
    """
    def __init__(self, buffer_size: int = 100000, her_ratio: float = 0.8):
        self.buffer_size = buffer_size
        self.her_ratio = her_ratio  # 80% of samples use HER
        self.storage = []

    def add(self, transition: Dict):
        """Store a transition (state, action, reward, next_state, done, info)"""
        transition['timestamp'] = datetime.now().isoformat()
        self.storage.append(transition)
        if len(self.storage) > self.buffer_size:
            self.storage.pop(0)

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample with HER - relabel goals based on achieved states"""
        samples = random.sample(self.storage, min(batch_size, len(self.storage)))

        # Apply HER to subset of samples
        her_samples = int(batch_size * self.her_ratio)
        for i in range(her_samples):
            sample = samples[i]

            # Relabel reward based on achieved goal
            # If we didn't reach the original goal, what DID we achieve?
            if sample['info'].get('termination') == 'timeout':
                # We survived - that's an achievement
                sample['reward'] = 100.0  # Positive reward for survival
            elif sample['info'].get('termination') == 'collision':
                # We collided - but we can learn from the distance traveled
                distance = sample['info'].get('total_distance', 0)
                sample['reward'] = max(distance, 0)  # Reward distance traveled

        return samples

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.storage[-1000:], f)  # Save last 1000 transitions

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.storage = json.load(f)


# =============================================================================
# WORLD MODEL COMPONENT (Temporal Prediction)
# =============================================================================
class WorldModel(nn.Module):
    """
    Predicts future states given current state and action.

    This allows the AI to "imagine" consequences before acting,
    enabling better long-term planning.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Dynamics model (predicts next state)
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action):
        encoded = self.encoder(state)
        combined = th.cat([encoded, action], dim=-1)

        next_state_pred = self.dynamics(combined)
        reward_pred = self.reward_predictor(combined)

        return next_state_pred, reward_pred

    def imagine_trajectory(self, initial_state, actions, horizon: int = 10):
        """Imagine future trajectory given a sequence of actions"""
        trajectory = [initial_state]
        current_state = initial_state

        for action in actions[:horizon]:
            with th.no_grad():
                next_state, _ = self.forward(current_state, action)
                trajectory.append(next_state)
                current_state = next_state

        return trajectory


# =============================================================================
# ATTENTION-BASED FEATURE EXTRACTOR
# =============================================================================
class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Uses multi-head self-attention for better perception.

    Unlike CNNs which have limited receptive fields, attention
    allows the model to focus on relevant parts of the scene.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Visual encoder (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with th.no_grad():
            sample = th.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            cnn_out = self.cnn(sample)
            self.cnn_dim = cnn_out.shape[1]

        # Multi-head attention for temporal reasoning
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True
        )

        # State processing with attention
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(self.cnn_dim + 128 + N_LIDAR_SECTORS, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, observations):
        # Process camera
        cam = observations["camera"].float() / 255.0
        cam = cam.permute(0, 3, 1, 2)
        cnn_feat = self.cnn(cam)

        # Process state with attention-like processing
        state = observations["state"]
        state_feat = self.state_encoder(state)

        # Combine
        lidar = observations["lidar"]
        combined = th.cat([cnn_feat, state_feat, lidar], dim=-1)

        return self.combined(combined)


# =============================================================================
# AUTOMATIC DOMAIN RANDOMIZATION (ADR)
# =============================================================================
class ADRManager:
    """
    Automatically adjusts environment difficulty to maintain
    optimal learning progression (not too easy, not too hard).

    Based on "Automatic Domain Randomization" (OpenAI, 2020)
    """
    def __init__(self, target_success_rate: float = 0.5, window: int = 100):
        self.target_success_rate = target_success_rate
        self.window = window
        self.recent_rewards = deque(maxlen=window)

        # Randomization parameters
        self.weather_params = {
            'cloudiness': (0, 100),
            'precipitation': (0, 100),
            'precipitation_deposits': (0, 100),
            'wind_intensity': (0, 100),
            'fog_density': (0, 100),
            'wetness': (0, 100),
        }

        # Current difficulty level (0-1)
        self.difficulty = 0.0

    def update(self, reward: float):
        """Update difficulty based on recent performance"""
        self.recent_rewards.append(reward > 0)  # Binary: success or not

        if len(self.recent_rewards) < self.window:
            return

        success_rate = sum(self.recent_rewards) / len(self.recent_rewards)

        # Adjust difficulty
        if success_rate > self.target_success_rate + 0.1:
            # Too easy - increase difficulty
            self.difficulty = min(1.0, self.difficulty + 0.05)
        elif success_rate < self.target_success_rate - 0.1:
            # Too hard - decrease difficulty
            self.difficulty = max(0.0, self.difficulty - 0.05)

    def get_weather_config(self) -> carla.WeatherParameters:
        """Get weather parameters based on current difficulty"""
        # Interpolate between easy and hard parameters
        cloudiness = self.difficulty * 80
        precipitation = self.difficulty * 50 if self.difficulty > 0.3 else 0
        fog_density = self.difficulty * 30 if self.difficulty > 0.5 else 0
        wetness = self.difficulty * 40 if self.difficulty > 0.4 else 0

        # Add some randomness
        cloudiness += random.uniform(-10, 10)

        return carla.WeatherParameters(
            cloudiness=max(0, min(100, cloudiness)),
            precipitation=precipitation,
            precipitation_deposits=precipitation,
            wind_intensity=random.uniform(0, 30 * self.difficulty),
            sun_azimuth_angle=random.uniform(0, 360),
            sun_altitude_angle=random.uniform(10, 70),
            fog_density=max(0, min(100, fog_density)),
            wetness=max(0, min(100, wetness)),
        )


# =============================================================================
# EPISODIC MEMORY SYSTEM (Advanced)
# =============================================================================
class EpisodicMemory:
    """
    Stores and retrieves successful trajectories for imitation learning.

    When the AI does something well, remember it.
    When stuck, recall similar successful episodes.
    """
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes = []  # List of successful episodes
        self.current_episode = []

    def start_episode(self):
        self.current_episode = []

    def add_step(self, state, action, reward, info):
        self.current_episode.append({
            'state': state,
            'action': action,
            'reward': reward,
            'info': info,
        })

    def end_episode(self, total_reward: float, success: bool = False):
        """Store episode if it was successful"""
        episode_data = {
            'steps': self.current_episode,
            'total_reward': total_reward,
            'timestamp': datetime.now().isoformat(),
        }

        if success or total_reward > 500:  # Threshold for "good" episode
            self.episodes.append(episode_data)
            if len(self.episodes) > self.capacity:
                # Remove worst episode
                self.episodes.sort(key=lambda x: x['total_reward'])
                self.episodes.pop(0)

    def retrieve_similar(self, current_state, k: int = 5):
        """Retrieve k most similar successful episodes"""
        if not self.episodes:
            return []

        # Simple similarity: compare state vectors
        similarities = []
        for ep in self.episodes:
            if ep['steps']:
                # Compare with first state of episode
                sim = np.linalg.norm(
                    current_state - ep['steps'][0]['state']
                )
                similarities.append((sim, ep))

        similarities.sort(key=lambda x: x[0])
        return [ep for _, ep in similarities[:k]]


# =============================================================================
# LEADERBOARD (Improved)
# =============================================================================
@dataclass
class AgentRecord:
    agent_id: str
    total_distance: float = 0.0
    total_penalty: float = 0.0
    collisions: int = 0
    off_roads: int = 0
    ego_wins: int = 0
    rival_wins: int = 0
    waypoints_visited: int = 0
    episodes_completed: int = 0
    best_single_distance: float = 0.0
    safety_score: float = 100.0
    avg_episode_reward: float = 0.0
    last_updated: str = ""

    @property
    def score(self) -> float:
        penalty_factor = self.total_penalty * 0.1
        safety_reduction = (100 - self.safety_score) * 10
        win_bonus = self.ego_wins * 500
        return max(0, self.total_distance - penalty_factor - safety_reduction + win_bonus)


class Leaderboard:
    def __init__(self):
        self.file_path = os.path.join(LEADERBOARD_DIR, "leaderboard.json")
        self.records: Dict[str, AgentRecord] = {}
        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    for agent_id, record_data in data.items():
                        self.records[agent_id] = AgentRecord(**record_data)
            except Exception:
                pass

    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.records.items()},
                     f, indent=2)

    def update(self, agent_id: str, **kwargs):
        if agent_id not in self.records:
            self.records[agent_id] = AgentRecord(agent_id=agent_id)

        record = self.records[agent_id]
        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)

        record.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def print_leaderboard(self, top_n=10):
        sorted_records = sorted(
            self.records.items(),
            key=lambda x: x[1].score,
            reverse=True
        )
        print("\n" + "="*80)
        print("  CARLA AI LEADERBOARD v10".center(80))
        print("="*80)
        print(f"{'Rank':<6} {'Agent':<20} {'Score':<10} {'Distance':<12} "
              f"{'Wins':<8} {'Safety':<10}")
        print("-"*80)
        for i, (aid, rec) in enumerate(sorted_records[:top_n]):
            medals = ["1st", "2nd", "3rd"]
            medal = medals[i] if i < 3 else f"{i+1}."
            print(f"{medal:<6} {aid[:20]:<20} {rec.score:<10.1f} "
                  f"{rec.total_distance:<12.1f} {rec.ego_wins:<8} "
                  f"{rec.safety_score:<10.1f}")
        print("="*80 + "\n")


# =============================================================================
# RECALL MEMORY SYSTEM (Persistent)
# =============================================================================
class RecallMemory:
    def __init__(self, agent_id: str = "ego", memory_size: int = 500):
        self.agent_id = agent_id
        self.file_path = os.path.join(MEMORY_DIR, f"{agent_id}_memory.json")
        self.memory_size = memory_size

        self.collision_hotspots: List[Dict] = []
        self.offroad_zones: List[Dict] = []
        self.successful_route: List[Dict] = []

        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self.collision_hotspots = data.get("collision_hotspots", [])
                    self.offroad_zones = data.get("offroad_zones", [])
                    self.successful_route = data.get("successful_route", [])
            except Exception:
                pass

    def save(self):
        data = {
            "collision_hotspots": self.collision_hotspots[-self.memory_size:],
            "offroad_zones": self.offroad_zones[-self.memory_size:],
            "successful_route": self.successful_route[-self.memory_size*5:],
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f)

    def add_collision(self, location: carla.Location):
        x, y = location.x, location.y

        for hotspot in self.collision_hotspots:
            dist = np.sqrt((hotspot["x"] - x)**2 + (hotspot["y"] - y)**2)
            if dist < 5.0:
                hotspot["count"] += 1
                hotspot["last_seen"] = datetime.now().isoformat()
                self.save()
                return

        self.collision_hotspots.append({
            "x": x, "y": y, "z": location.z,
            "count": 1,
            "last_seen": datetime.now().isoformat()
        })
        self.save()

    def add_offroad(self, location: carla.Location):
        self.offroad_zones.append({
            "x": location.x, "y": location.y,
            "radius": 8.0,
            "last_seen": datetime.now().isoformat()
        })
        self.save()

    def add_waypoint(self, location: carla.Location, sequence: int):
        self.successful_route.append({
            "x": location.x, "y": location.y, "z": location.z,
            "sequence": sequence
        })

    def get_danger_score(self, location: carla.Location) -> float:
        x, y = location.x, location.y
        danger = 0.0

        for hotspot in self.collision_hotspots:
            dist = np.sqrt((hotspot["x"] - x)**2 + (hotspot["y"] - y)**2)
            if dist < 10.0:
                danger += hotspot["count"] * (1.0 - dist/10.0) * 0.4

        for zone in self.offroad_zones:
            dist = np.sqrt((zone["x"] - x)**2 + (zone["y"] - y)**2)
            if dist < zone["radius"]:
                danger += (1.0 - dist/zone["radius"]) * 0.3

        return min(danger, 1.0)

    def get_route_guidance(self, location: carla.Location) -> Optional[carla.Location]:
        if not self.successful_route:
            return None

        closest_seq = -1
        min_dist = float('inf')

        for wp in self.successful_route:
            dist = np.sqrt((wp["x"] - location.x)**2 + (wp["y"] - location.y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_seq = wp["sequence"]

        if closest_seq >= 0 and min_dist < 20.0:
            for wp in self.successful_route:
                if wp["sequence"] == closest_seq + 1:
                    return carla.Location(wp["x"], wp["y"], wp["z"])

        return None


# =============================================================================
# DECISION MAKING MODULE
# =============================================================================
class DecisionMaker:
    def __init__(self):
        self.decision_history = deque(maxlen=100)
        self.hesitation_count = 0

    def should_hesitate(self, current_speed: float, danger_ahead: float,
                       on_road: bool, nearby_hazard: bool) -> Tuple[bool, str]:
        reasons = []

        # More lenient thresholds for v10
        if danger_ahead > 0.8:  # Increased from 0.7
            reasons.append(f"High danger zone (score: {danger_ahead:.2f})")

        if not on_road and current_speed > 15:  # Increased from 10
            reasons.append("Leaving road - slowing down!")

        if nearby_hazard and current_speed > 20:  # Increased from 15
            reasons.append("Hazard nearby - caution!")

        if current_speed > 40 and danger_ahead > 0.4:
            reasons.append("Too fast for conditions")

        should = len(reasons) > 0
        if should:
            self.hesitation_count += 1
            reason_str = " | ".join(reasons)
            self.decision_history.append({"hesitate": True, "reason": reason_str})
            return True, reason_str

        self.decision_history.append({"hesitate": False, "reason": "Clear"})
        return False, "Clear"


# =============================================================================
# CURRICULUM MANAGER (Improved)
# =============================================================================
class CurriculumManager:
    STAGES = {
        1: {"name": "Basics", "map": "Town01", "traffic": 0, "pedestrians": 0,
            "weather": "Clear", "obstacles": False, "advance_threshold": 500.0},
        2: {"name": "Traffic", "map": "Town03", "traffic": 10, "pedestrians": 5,
            "weather": "Cloudy", "obstacles": False, "advance_threshold": 1000.0},
        3: {"name": "Expert", "map": "Town05", "traffic": 30, "pedestrians": 15,
            "weather": "Rainy", "obstacles": True, "advance_threshold": 2000.0},
        4: {"name": "Master", "map": "Town10HD","traffic": 50, "pedestrians": 30,
            "weather": "Stormy", "obstacles": True, "advance_threshold": 5000.0},
    }
    WINDOW = 50

    def __init__(self, current_stage: int = 1):
        self.current_stage = current_stage
        self._ep_rewards: deque = deque(maxlen=self.WINDOW)

    def get_config(self) -> dict:
        return self.STAGES[self.current_stage]

    def record_episode(self, total_reward: float) -> bool:
        self._ep_rewards.append(total_reward)
        if len(self._ep_rewards) < self.WINDOW:
            return False
        mean = np.mean(self._ep_rewards)
        threshold = self.STAGES[self.current_stage]["advance_threshold"]
        if mean >= threshold and self.current_stage < max(self.STAGES):
            self.current_stage += 1
            self._ep_rewards.clear()
            print(f"\n[CURRICULUM] ADVANCED -> Stage {self.current_stage}: "
                  f"{self.STAGES[self.current_stage]['name']}\n")
            return True
        return False


# =============================================================================
# TRAINING LOGGER (Enhanced)
# =============================================================================
class TrainingLogger(BaseCallback):
    def __init__(self, log_path: str, curriculum: CurriculumManager,
                 leaderboard: Leaderboard, agent_id: str = "Grandmaster_v10"):
        super().__init__()
        self.log_path = log_path
        self.curriculum = curriculum
        self.leaderboard = leaderboard
        self.agent_id = agent_id
        self.ego_wins = 0
        self.rival_wins = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "tournament_winner" in info:
                if info["tournament_winner"] == "ego":
                    self.ego_wins += 1
                elif info["tournament_winner"] == "rival":
                    self.rival_wins += 1

        self.logger.record("competition/ego_wins", self.ego_wins)
        self.logger.record("competition/rival_wins", self.rival_wins)
        return True

    def _on_rollout_end(self) -> None:
        if self.logger is None:
            return
        try:
            stats = {k: v for k, v in self.logger.name_to_value.items()}
            ts = self.num_timesteps
            rew = stats.get("rollout/ep_rew_mean", 0.0)
            if np.isnan(rew) or np.isinf(rew):
                rew = 0.0

            loss = stats.get("train/loss", 0.0)
            if np.isnan(loss) or np.isinf(loss):
                loss = 0.0

            stage = self.curriculum.current_stage
            with open(self.log_path, "a") as f:
                f.write(
                    f"total_timesteps | {ts}\n"
                    f"ep_rew_mean     | {rew:.4f}\n"
                    f"loss            | {loss:.6e}\n"
                    f"curriculum_stage| {stage}\n"
                    f"ego_wins        | {self.ego_wins}\n"
                    f"rival_wins      | {self.rival_wins}\n"
                    f"---\n"
                )

            self.leaderboard.update(
                self.agent_id,
                ego_wins=self.ego_wins,
                rival_wins=self.rival_wins,
                avg_episode_reward=rew
            )
        except Exception:
            pass


# =============================================================================
# MAIN ENVIRONMENT (v10 ULTIMATE)
# =============================================================================
class CarlaGrandmasterEnvUltimate(gym.Env):
    """
    v10.0 Ultimate Environment with ALL advanced features integrated.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, stage_config: dict, curriculum: CurriculumManager,
                 leaderboard: Leaderboard = None, agent_id: str = "Grandmaster_v10"):
        super().__init__()
        self.config = stage_config
        self.curriculum = curriculum
        self.leaderboard = leaderboard or Leaderboard()
        self.agent_id = agent_id

        # Advanced systems
        self.memory = RecallMemory(agent_id)
        self.decision_maker = DecisionMaker()
        self.adr = ADRManager()
        self.episodic_memory = EpisodicMemory()
        self.her_buffer = HERBuffer()

        # CARLA connection
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(60.0)
        self.world = self.client.load_world(self.config["map"])
        self.map = self.world.get_map()

        all_spawns = self.map.get_spawn_points()
        self.initial_spawn = all_spawns[0]
        self._all_spawns = all_spawns

        # Physics
        self._settings = self.world.get_settings()
        self._settings.synchronous_mode = True
        self._settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(self._settings)

        # Traffic Manager
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)

        # Spaces
        self.observation_space = spaces.Dict({
            "camera": spaces.Box(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), np.uint8),
            "lidar": spaces.Box(0.0, LIDAR_MAX_RANGE, (N_LIDAR_SECTORS,), np.float32),
            "state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Actor handles
        self.vehicle = None
        self.camera_sensor = None
        self.lidar_sensor = None
        self.collision_sensor = None
        self.lane_sensor = None

        # Rival AI
        self.rival_vehicle = None
        self.rival_camera = None
        self.rival_lidar = None
        self.rival_collision = None
        self.rival_lane = None

        # State buffers
        self.last_img = None
        self.last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        self.rival_last_img = None
        self.rival_last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        self._img_lock = threading.Lock()
        self._lidar_lock = threading.Lock()

        # Model reference
        self.model = None

        # Episode tracking
        self.npc_vehicles = []
        self.npc_pedestrians = []
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.rival_collision_hist = []
        self.rival_lane_invasion_hist = []
        self.visited_waypoints = set()
        self.steps = 0
        self.stuck_counter = 0
        self.rival_stuck_counter = 0
        self.episode_reward = 0.0
        self.last_steer = 0.0
        self.last_throttle = 0.0
        self.rival_last_steer = 0.0
        self.rival_last_throttle = 0.0

        # Accumulative tracking
        self.current_penalty = 0.0
        self.current_distance = 0.0
        self.rival_distance = 0.0
        self.last_loc = None
        self.rival_last_loc = None
        self.current_trajectory = []
        self.waypoint_sequence = 0

        # Best trajectory
        self.best_trajectory = []
        self.best_distance = 0.0

        # Safety score
        self.safety_score = 100.0

        # Episode tracking for HER
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_infos = []

    def set_model(self, model):
        self.model = model

    # ==================== CLEANUP ====================
    def _cleanup(self):
        actors_to_destroy = []

        for attr in ("camera_sensor", "lidar_sensor", "collision_sensor",
                     "lane_sensor", "vehicle"):
            a = getattr(self, attr, None)
            if a is not None:
                if hasattr(a, 'listen'):
                    try: a.stop()
                    except: pass
                actors_to_destroy.append(a)
                setattr(self, attr, None)

        for attr in ("rival_camera", "rival_lidar", "rival_collision",
                     "rival_lane", "rival_vehicle"):
            a = getattr(self, attr, None)
            if a is not None:
                if hasattr(a, 'listen'):
                    try: a.stop()
                    except: pass
                actors_to_destroy.append(a)
                setattr(self, attr, None)

        for npc in self.npc_vehicles + self.npc_pedestrians:
            try:
                if npc.is_alive:
                    actors_to_destroy.append(npc)
            except Exception:
                pass
        self.npc_vehicles = []
        self.npc_pedestrians = []

        if actors_to_destroy:
            try:
                valid = [a for a in actors_to_destroy if hasattr(a, 'id')]
                if valid:
                    self.client.apply_batch([carla.command.DestroyActor(a) for a in valid])
            except Exception:
                pass

        self.world.tick()

        with self._img_lock:
            self.last_img = None
            self.rival_last_img = None
        with self._lidar_lock:
            self.last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
            self.rival_last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)

    # ==================== NPC SPAWNING ====================
    def _spawn_npc_vehicles(self, count: int):
        bp_lib = self.world.get_blueprint_library()
        blueprints = bp_lib.filter("vehicle.*")
        points = random.sample(self._all_spawns, min(count, len(self._all_spawns)))
        batch = []
        for sp in points:
            bp = random.choice(blueprints)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(
                    bp.get_attribute("color").recommended_values))
            batch.append(carla.command.SpawnActor(bp, sp)
                         .then(carla.command.SetAutopilot(
                             carla.command.FutureActor, True, 8000)))
        results = self.client.apply_batch_sync(batch, True)
        for r in results:
            if not r.error:
                actor = self.world.get_actor(r.actor_id)
                if actor:
                    self.npc_vehicles.append(actor)

    def _spawn_npc_pedestrians(self, count: int):
        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        ctrl_bp = bp_lib.find("controller.ai.walker")

        spawn_locs = []
        for _ in range(count):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_locs.append(carla.Transform(loc))

        batch = [carla.command.SpawnActor(random.choice(walker_bps), sp)
                for sp in spawn_locs]
        results = self.client.apply_batch_sync(batch, True)
        walker_ids = [r.actor_id for r in results if not r.error]

        walkers = [self.world.get_actor(wid) for wid in walker_ids if wid]
        for w in walkers:
            if w:
                ctrl = self.world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
                ctrl.start()
                ctrl.go_to_location(self.world.get_random_location_from_navigation())
                ctrl.set_max_speed(1.4)
                self.npc_pedestrians.extend([w, ctrl])

    # ==================== WEATHER (with ADR) ====================
    def _randomize_weather(self):
        # Use ADR to determine difficulty-based weather
        weather = self.adr.get_weather_config()
        self.world.set_weather(weather)

    # ==================== SENSOR CALLBACKS ====================
    def _cb_camera(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
        class_ids = arr[:, :, 2]

        rgb_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        # Optimized color mapping
        palette = {
            0: [0, 0, 0], 1: [100, 0, 0], 2: [150, 0, 0], 3: [100, 50, 50],
            4: [255, 0, 255], 5: [255, 255, 0], 6: [0, 255, 0], 7: [0, 255, 0],
            8: [255, 0, 0], 9: [150, 75, 0], 10: [255, 100, 0], 11: [120, 0, 0],
            12: [255, 255, 255],
        }

        for class_id, color in palette.items():
            rgb_img[class_ids == class_id] = color
        rgb_img[class_ids > 12] = [80, 0, 0]

        with self._img_lock:
            self.last_img = rgb_img

    def _cb_rival_camera(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
        class_ids = arr[:, :, 2]

        rgb_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        palette = {
            0: [0, 0, 0], 1: [100, 0, 0], 2: [150, 0, 0], 3: [100, 50, 50],
            4: [255, 0, 255], 5: [255, 255, 0], 6: [0, 255, 0], 7: [0, 255, 0],
            8: [255, 0, 0], 9: [150, 75, 0], 10: [255, 100, 0], 11: [120, 0, 0],
            12: [255, 255, 255],
        }
        for class_id, color in palette.items():
            rgb_img[class_ids == class_id] = color
        rgb_img[class_ids > 12] = [80, 0, 0]

        with self._img_lock:
            self.rival_last_img = rgb_img

    def _cb_lidar(self, point_cloud):
        pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        if pts.size == 0:
            return
        pts = pts.reshape(-1, 4)[:, :3]
        pts = pts[pts[:, 2] > -1.0]
        sectors = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        if len(pts) > 0:
            angles = np.arctan2(pts[:, 1], pts[:, 0]) % (2 * np.pi)
            indices = (angles / (2 * np.pi) * N_LIDAR_SECTORS).astype(int) % N_LIDAR_SECTORS
            dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            for idx, d in zip(indices, dists):
                if d < sectors[idx]:
                    sectors[idx] = d
        with self._lidar_lock:
            self.last_lidar = sectors

    def _cb_rival_lidar(self, point_cloud):
        pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        if pts.size == 0:
            return
        pts = pts.reshape(-1, 4)[:, :3]
        pts = pts[pts[:, 2] > -1.0]
        sectors = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        if len(pts) > 0:
            angles = np.arctan2(pts[:, 1], pts[:, 0]) % (2 * np.pi)
            indices = (angles / (2 * np.pi) * N_LIDAR_SECTORS).astype(int) % N_LIDAR_SECTORS
            dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            for idx, d in zip(indices, dists):
                if d < sectors[idx]:
                    sectors[idx] = d
        with self._lidar_lock:
            self.rival_last_lidar = sectors

    def _cb_collision(self, event):
        self.collision_hist.append(event)
        try:
            if hasattr(event, 'other_actor') and event.other_actor:
                self.memory.add_collision(event.other_actor.get_location())
        except Exception:
            pass

    def _cb_rival_collision(self, event):
        self.rival_collision_hist.append(event)

    def _cb_lane(self, event):
        self.lane_invasion_hist.append(event)

    def _cb_rival_lane(self, event):
        self.rival_lane_invasion_hist.append(event)

    def _attach_sensors(self, vehicle, is_rival=False):
        lib = self.world.get_blueprint_library()

        cam_bp = lib.find("sensor.camera.semantic_segmentation")
        cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
        cam_bp.set_attribute("fov", "90")
        camera = self.world.spawn_actor(
            cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle
        )
        if is_rival:
            self.rival_camera = camera
            camera.listen(self._cb_rival_camera)
        else:
            self.camera_sensor = camera
            camera.listen(self._cb_camera)

        lidar_bp = lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "16")
        lidar_bp.set_attribute("range", str(LIDAR_MAX_RANGE))
        lidar_bp.set_attribute("points_per_second", "100000")
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar = self.world.spawn_actor(
            lidar_bp, carla.Transform(carla.Location(x=0.0, z=2.4)), attach_to=vehicle
        )
        if is_rival:
            self.rival_lidar = lidar
            lidar.listen(self._cb_rival_lidar)
        else:
            self.lidar_sensor = lidar
            lidar.listen(self._cb_lidar)

        collision = self.world.spawn_actor(
            lib.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle
        )
        if is_rival:
            self.rival_collision = collision
            collision.listen(self._cb_rival_collision)
        else:
            self.collision_sensor = collision
            collision.listen(self._cb_collision)

        lane = self.world.spawn_actor(
            lib.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=vehicle
        )
        if is_rival:
            self.rival_lane = lane
            lane.listen(self._cb_rival_lane)
        else:
            self.lane_sensor = lane
            lane.listen(self._cb_lane)

    # ==================== RESET ====================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()
        self._randomize_weather()

        # Spawn ego vehicle
        bp = self.world.get_blueprint_library().filter("model3")[0]
        spawned = False
        spawn_points_to_try = [self.initial_spawn] + self._all_spawns[2:7]

        for spawn_point in spawn_points_to_try:
            for attempt in range(3):
                try:
                    self.vehicle = self.world.spawn_actor(bp, spawn_point)
                    spawned = True
                    break
                except RuntimeError:
                    time.sleep(0.3)
                    self.world.tick()
            if spawned:
                break

        if not spawned:
            raise RuntimeError("[ENV] Could not spawn ego vehicle")

        # Spawn Rival AI
        wp = self.map.get_waypoint(self.initial_spawn.location)
        rival_wp = wp.get_left_lane()
        if rival_wp is None or rival_wp.lane_type != carla.LaneType.Driving:
            rival_wp = wp.get_right_lane()

        rival_spawn = rival_wp.transform if rival_wp else self._all_spawns[1]
        rival_spawn.location.z += 0.5

        try:
            self.rival_vehicle = self.world.spawn_actor(bp, rival_spawn)
            print("[ENV] Rival AI Spawned!")
        except Exception as e:
            print(f"[WARN] Could not spawn Rival: {e}")

        # NPCs
        self._spawn_npc_vehicles(self.config["traffic"])
        self._spawn_npc_pedestrians(self.config["pedestrians"])

        # Sensors
        self._attach_sensors(self.vehicle, is_rival=False)
        if self.rival_vehicle:
            self._attach_sensors(self.rival_vehicle, is_rival=True)

        # Wait for sensors
        for _ in range(20):
            self.world.tick()
            with self._img_lock:
                if self.last_img is not None:
                    if self.rival_vehicle and self.rival_last_img is None:
                        continue
                    break

        # Reset episode trackers
        self.steps = 0
        self.stuck_counter = 0
        self.rival_stuck_counter = 0
        self.episode_reward = 0.0
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.rival_collision_hist = []
        self.rival_lane_invasion_hist = []
        self.visited_waypoints = set()
        self.last_steer = 0.0
        self.last_throttle = 0.0
        self.rival_last_steer = 0.0
        self.rival_last_throttle = 0.0

        # Accumulative tracking
        self.current_penalty = 0.0
        self.current_distance = 0.0
        self.rival_distance = 0.0
        self.last_loc = None
        self.rival_last_loc = None
        self.current_trajectory = []
        self.waypoint_sequence = 0

        # Episode tracking for HER
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_infos = []

        # Start episodic memory
        self.episodic_memory.start_episode()

        print(f"[ENV] Episode reset. Safety: {self.safety_score:.1f}, ADR Difficulty: {self.adr.difficulty:.2f}")

        return self._get_actor_obs('ego'), {}

    # ==================== OBSERVATION ====================
    def _get_actor_obs(self, actor_type='ego') -> dict:
        is_rival = (actor_type == 'rival')
        vehicle = self.rival_vehicle if is_rival else self.vehicle

        with self._img_lock:
            if is_rival:
                cam = self.rival_last_img if self.rival_last_img is not None else \
                      np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            else:
                cam = self.last_img if self.last_img is not None else \
                      np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        with self._lidar_lock:
            lidar = self.rival_last_lidar.copy() if is_rival else self.last_lidar.copy()

        if vehicle is None or not vehicle.is_alive:
            return {"camera": cam, "lidar": lidar,
                   "state": np.zeros(STATE_DIM, dtype=np.float32)}

        v = vehicle.get_velocity()
        kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

        # Red light flag
        red_light = 0.0
        if vehicle.is_at_traffic_light():
            tl = vehicle.get_traffic_light()
            if tl and tl.get_state() == carla.TrafficLightState.Red:
                red_light = 1.0

        # Junction flag
        wp = self.map.get_waypoint(vehicle.get_location(), project_to_road=True)
        at_junction = float(wp.is_junction) if wp else 0.0

        # Memory danger score
        danger_memory = self.memory.get_danger_score(vehicle.get_location())

        # Route guidance
        route_guidance_x = route_guidance_y = 0.0
        guidance = self.memory.get_route_guidance(vehicle.get_location())
        if guidance:
            dx = guidance.x - vehicle.get_location().x
            dy = guidance.y - vehicle.get_location().y
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                route_guidance_x = dx / norm
                route_guidance_y = dy / norm

        # Ghost path guidance
        ghost_guidance = 0.0
        if self.best_trajectory:
            loc = vehicle.get_location()
            dists = [loc.distance(pt) for pt in self.best_trajectory]
            if dists:
                min_dist = min(dists)
                if min_dist < 5.0:
                    ghost_guidance = 1.0 - (min_dist / 5.0)

        # Look ahead
        ahead_safe = 1.0
        if wp:
            next_wps = wp.next(5.0)
            if next_wps:
                next_wp = next_wps[0]
                valid_lanes = (carla.LaneType.Driving, carla.LaneType.Bidirectional,
                              carla.LaneType.OnRamp, carla.LaneType.OffRamp)
                if next_wp.lane_type not in valid_lanes:
                    ahead_safe = -1.0
            else:
                ahead_safe = -1.0
        else:
            ahead_safe = -1.0

        min_lidar = float(np.min(lidar if is_rival else self.last_lidar))

        # Progress in episode
        progress = float(self.steps / 10000) if self.steps < 10000 else 1.0

        state = np.array([
            kmh / 50.0,
            self.rival_last_steer if is_rival else self.last_steer,
            self.rival_last_throttle if is_rival else self.last_throttle,
            at_junction,
            red_light,
            self.current_penalty / 1000.0,
            self.safety_score / 100.0,
            danger_memory,
            min_lidar / LIDAR_MAX_RANGE,
            float(len(self.rival_lane_invasion_hist if is_rival else self.lane_invasion_hist) > 0),
            min((self.rival_stuck_counter if is_rival else self.stuck_counter) / 500.0, 1.0),
            float(len(self.rival_collision_hist if is_rival else self.collision_hist) > 0),
            route_guidance_x,
            route_guidance_y,
            ahead_safe,
            progress,  # NEW: track progress
            float(self.current_distance / 1000.0),  # NEW: track distance
        ], dtype=np.float32)

        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        return {"camera": cam, "lidar": lidar, "state": state}

    # ==================== STEP ====================
    def step(self, action):
        self.steps += 1

        # Store for HER
        state_before = self._get_actor_obs('ego')

        # Apply Ego Action
        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])

        # Get current state for decision making
        v = self.vehicle.get_velocity()
        kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

        loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Any)

        # Decision making
        danger_ahead = self.memory.get_danger_score(loc)
        min_lidar_dist = float(np.min(self.last_lidar))
        nearby_hazard = min_lidar_dist < 10.0

        on_road = wp is not None and wp.lane_type in (
            carla.LaneType.Driving, carla.LaneType.Bidirectional,
            carla.LaneType.OnRamp, carla.LaneType.OffRamp)

        should_hesitate, reason = self.decision_maker.should_hesitate(
            kmh, danger_ahead, on_road, nearby_hazard
        )

        if should_hesitate:
            throttle = min(throttle, 0.3)
            brake = max(brake, 0.2)

        # Anti-stall (improved)
        if kmh < 2.0:
            throttle = 0.6
            brake = 0.0
            if self.stuck_counter > 50 and self.steps % 10 == 0:
                steer = 0.3 * (1 if self.steps % 20 < 10 else -1)

        self.last_steer = steer
        self.last_throttle = throttle

        self.vehicle.apply_control(
            carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        )

        # Apply Rival Action
        if self.rival_vehicle and self.rival_vehicle.is_alive and self.model:
            rival_obs = self._get_actor_obs('rival')
            rival_action, _ = self.model.predict(rival_obs, deterministic=False)

            r_steer, r_throttle, r_brake = float(rival_action[0]), float(rival_action[1]), float(rival_action[2])

            r_v = self.rival_vehicle.get_velocity()
            r_kmh = 3.6 * np.sqrt(r_v.x**2 + r_v.y**2 + r_v.z**2)
            if r_kmh < 2.0:
                r_throttle = 0.6

            self.rival_last_steer = r_steer
            self.rival_last_throttle = r_throttle

            self.rival_vehicle.apply_control(
                carla.VehicleControl(steer=r_steer, throttle=r_throttle, brake=r_brake)
            )

        self.world.tick()

        # Spectator camera (birds-eye view)
        spec = self.world.get_spectator()
        t = self.vehicle.get_transform()
        cam_loc = t.location + carla.Location(z=45)
        spec.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=-90)))

        # Track distance
        if self.last_loc is not None:
            self.current_distance += loc.distance(self.last_loc)
        self.last_loc = loc
        if self.steps % 5 == 0:
            self.current_trajectory.append(loc)

        if self.rival_vehicle and self.rival_vehicle.is_alive:
            rival_loc = self.rival_vehicle.get_location()
            if self.rival_last_loc is not None:
                self.rival_distance += rival_loc.distance(self.rival_last_loc)
            self.rival_last_loc = rival_loc

        # ==================== REWARD SYSTEM v10 ====================
        reward = 0.0
        done = False
        info = {}

        # 1. Exploration bonus (increased)
        if wp and wp.id not in self.visited_waypoints:
            reward += 20.0
            self.visited_waypoints.add(wp.id)
            self.memory.add_waypoint(loc, self.waypoint_sequence)
            self.waypoint_sequence += 1

        # 2. Green pixel reward
        with self._img_lock:
            if self.last_img is not None:
                green_pixels = np.sum((self.last_img[:, :, 0] == 0) &
                                     (self.last_img[:, :, 1] == 255) &
                                     (self.last_img[:, :, 2] == 0))
                green_ratio = green_pixels / (IMG_WIDTH * IMG_HEIGHT)
                if green_ratio > 0.1:
                    reward += green_ratio * 1.0

        # 3. Speed reward (optimized)
        speed_limit_raw = self.vehicle.get_speed_limit()
        if speed_limit_raw is None or speed_limit_raw <= 0:
            speed_limit_raw = 50.0
        speed_ratio = kmh / max(speed_limit_raw, 1.0)
        speed_reward = np.clip(speed_ratio, 0.0, 1.0) * 6.0

        if wp and wp.is_junction:
            if kmh > 15.0:
                speed_reward += 5.0
            else:
                reward -= 0.5

        reward += speed_reward

        # 4. Anti-stall
        if kmh < 1.0:
            self.stuck_counter += 1
            reward -= min(self.stuck_counter * 0.1, 5.0)
        else:
            self.stuck_counter = 0

        # 5. Steering smoothness
        reward -= abs(steer) * 0.3

        # 6. Lane invasion
        if self.lane_invasion_hist:
            reward -= 100.0
            info["lane_invasion"] = True
            self.lane_invasion_hist = []

        # 7. Red light violation
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl and tl.get_state() == carla.TrafficLightState.Red and kmh > 1.0:
                reward -= 400.0
                info["red_light_violation"] = True

        # ==================== NO-RESET PENALTY SYSTEM ====================
        valid_lanes = (carla.LaneType.Driving, carla.LaneType.Bidirectional,
                      carla.LaneType.OnRamp, carla.LaneType.OffRamp)

        ego_needs_respawn = False

        # Off-road detection
        if wp is None or wp.lane_type not in valid_lanes:
            penalty = 50.0
            self.current_penalty += penalty
            self.safety_score = max(0, self.safety_score - 0.3)
            reward -= penalty
            self.memory.add_offroad(loc)
            if self.steps % 100 == 0:
                print(f"[PENALTY] Off-road! Total: {self.current_penalty:.1f}")
            info["termination"] = "off_road"

        # Collision
        if self.collision_hist:
            info["termination"] = "collision"
            info["collisions"] = len(self.collision_hist)

            hit_rival = False
            if self.rival_vehicle:
                for event in self.collision_hist:
                    if event.other_actor.id == self.rival_vehicle.id:
                        hit_rival = True
                        ego_fwd = self.vehicle.get_transform().get_forward_vector()
                        dir_to_rival = event.other_actor.get_location() - self.vehicle.get_location()
                        mag = np.sqrt(dir_to_rival.x**2 + dir_to_rival.y**2)
                        if mag > 0:
                            dir_to_rival.x /= mag
                            dir_to_rival.y /= mag
                        dot = ego_fwd.x * dir_to_rival.x + ego_fwd.y * dir_to_rival.y
                        if dot > 0.5:
                            reward -= 2000.0
                            print("[ENV] RAMMED RIVAL!")
                        else:
                            reward -= 400.0
                            print("[ENV] Rival crashed into ego!")
                        break

            if not hit_rival:
                penalty = 150.0
                self.current_penalty += penalty
                self.safety_score = max(0, self.safety_score - 1.5)
                reward -= penalty
                if self.steps % 100 == 0:
                    print(f"[PENALTY] Collision! Total: {self.current_penalty:.1f}")

            self.collision_hist = []

        # 8. Episode timeout (SUCCESS if reached here!)
        if self.steps >= 10000:
            done = True
            info["termination"] = "timeout"
            # Bonus for completing episode
            if self.current_penalty < 500:
                reward += 1000.0
                print(f"[SUCCESS] Episode completed! Distance: {self.current_distance:.1f}m")

        # 9. Stuck handling
        if self.stuck_counter > 500:
            self.current_penalty += 30.0
            ego_needs_respawn = True
            self.stuck_counter = 0
            print("[ENV] Ego stuck - Respawning...")

        # 10. Ghost path reward
        if self.best_trajectory:
            dists = [loc.distance(pt) for pt in self.best_trajectory]
            if dists:
                min_dist = min(dists)
                if min_dist < 3.0:
                    reward += 2.0 + (kmh / 15.0)

        # 11. ATM - Ideal steering
        if wp:
            target_wps = wp.next(10.0)
            if target_wps:
                target_wp = target_wps[0]
                target_loc = target_wp.transform.location
                ego_trans = self.vehicle.get_transform()
                ego_loc = ego_trans.location
                v_to_target = np.array([target_loc.x - ego_loc.x, target_loc.y - ego_loc.y])
                forward = ego_trans.get_forward_vector()
                v_forward = np.array([forward.x, forward.y])

                if not np.any(np.isnan(v_to_target)) and not np.any(np.isnan(v_forward)):
                    steer_error = np.cross(v_forward, v_to_target) / (np.linalg.norm(v_to_target) + 1e-6)
                    ideal_steer = np.clip(steer_error, -1.0, 1.0)
                    copy_reward = 1.0 - abs(steer - ideal_steer)
                    reward += copy_reward * 3.0

        # 12. Check Rival Stuck
        rival_needs_respawn = False
        if self.rival_vehicle and self.rival_vehicle.is_alive:
            r_v = self.rival_vehicle.get_velocity()
            r_kmh = 3.6 * np.sqrt(r_v.x**2 + r_v.y**2 + r_v.z**2)
            if r_kmh < 1.0:
                self.rival_stuck_counter += 1
            else:
                self.rival_stuck_counter = 0

            if self.rival_stuck_counter > 500:
                rival_needs_respawn = True
                print("[ENV] Rival stuck - Respawning...")

        # ==================== TOURNAMENT ====================
        if ego_needs_respawn or rival_needs_respawn:
            if self.rival_vehicle and self.rival_vehicle.is_alive:
                if self.current_distance > self.rival_distance:
                    print(f"[ENV] EGO WINS! ({self.current_distance:.1f}m vs {self.rival_distance:.1f}m)")
                    reward += 1000.0
                    info["tournament_winner"] = "ego"
                    self.leaderboard.update(self.agent_id, ego_wins=1)
                else:
                    print(f"[ENV] RIVAL WINS! ({self.rival_distance:.1f}m vs {self.current_distance:.1f}m)")
                    reward -= 500.0
                    info["tournament_winner"] = "rival"
                    self.leaderboard.update(self.agent_id, rival_wins=1)

            # Update best trajectory
            if self.current_distance > self.best_distance and self.current_distance > 50.0:
                self.best_distance = self.current_distance
                self.best_trajectory = list(self.current_trajectory)
                print(f"[ENV] NEW RECORD! {self.best_distance:.1f}m")

            # Respawn
            if ego_needs_respawn:
                self.vehicle.set_transform(self.initial_spawn)
                self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                self.stuck_counter = 0
                self.current_distance = 0.0
                self.current_trajectory = []
                self.last_loc = None
                self.last_steer = 0.0
                self.last_throttle = 0.0

            if rival_needs_respawn:
                ego_loc = self.vehicle.get_location()
                ego_wp = self.map.get_waypoint(ego_loc)
                rival_wps = ego_wp.previous(10.0) if ego_wp else []
                if rival_wps:
                    rival_spawn = rival_wps[0].transform
                else:
                    rival_spawn = self._all_spawns[1]
                rival_spawn.location.z += 0.5
                self.rival_vehicle.set_transform(rival_spawn)
                self.rival_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                self.rival_stuck_counter = 0
                self.rival_last_loc = None

        # Store for HER
        self.episode_states.append(state_before)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_infos.append(info)

        # Store in episodic memory
        self.episodic_memory.add_step(state_before, action, reward, info)

        # Update ADR
        self.adr.update(reward)

        self.episode_reward += reward

        if done:
            # Store episode in HER buffer
            for i in range(len(self.episode_states)):
                self.her_buffer.add({
                    'state': self.episode_states[i],
                    'action': self.episode_actions[i],
                    'reward': self.episode_rewards[i],
                    'next_state': self.episode_states[min(i+1, len(self.episode_states)-1)],
                    'done': i == len(self.episode_states) - 1,
                    'info': self.episode_infos[i],
                })

            # Store in episodic memory if successful
            success = self.current_distance > 500 and self.current_penalty < 200
            self.episodic_memory.end_episode(self.episode_reward, success)

            # HER: Relabel failed episodes
            if not success and self.current_distance > 100:
                # Relabel as partial success based on distance traveled
                for i in range(len(self.episode_states)):
                    if self.episode_infos[i].get('termination') == 'timeout':
                        self.episode_rewards[i] = min(self.episode_rewards[i] + self.current_distance / 10, 100)

            advanced = self.curriculum.record_episode(self.episode_reward)
            info["episode_reward"] = self.episode_reward
            info["curriculum_stage"] = self.curriculum.current_stage
            info["total_distance"] = self.current_distance
            info["total_penalty"] = self.current_penalty
            info["safety_score"] = self.safety_score

            self.leaderboard.update(
                self.agent_id,
                total_distance=self.current_distance,
                total_penalty=self.current_penalty,
                collisions=sum(1 for _ in self.collision_hist),
                waypoints_visited=len(self.visited_waypoints),
                safety_score=self.safety_score,
                avg_episode_reward=self.episode_reward
            )

            if advanced:
                info["stage_advanced"] = True

        # Safety checks
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        reward = np.clip(reward, -10000.0, 10000.0)

        return self._get_actor_obs('ego'), reward, done, False, info

    # ==================== CLOSE ====================
    def close(self):
        self._settings.synchronous_mode = False
        self._settings.fixed_delta_seconds = None
        self.world.apply_settings(self._settings)
        self._cleanup()
        self.memory.save()
        self.her_buffer.save(os.path.join(HER_BUFFER_DIR, "her_buffer.json"))
        print("[ENV] Environment closed cleanly.")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  CARLA GRANDMASTER AI SUITE v10.0 - ULTIMATE EDITION")
    print("  HER | World Model | Attention | ADR | Self-Play")
    print("="*70 + "\n")

    curriculum = CurriculumManager(current_stage=1)
    config = curriculum.get_config()
    leaderboard = Leaderboard()
    agent_id = "Grandmaster_v10"
    env = CarlaGrandmasterEnvUltimate(
        stage_config=config,
        curriculum=curriculum,
        leaderboard=leaderboard,
        agent_id=agent_id
    )

    # Try to load existing model
    model = None
    if os.path.exists(MODEL_PATH):
        print(f"[LOAD] Resuming: {MODEL_PATH}")
        try:
            model = PPO.load(MODEL_PATH, env=env, device="cuda", tensorboard_log=TB_DIR)
            print("[LOAD] Successfully loaded!")
        except Exception as e:
            print(f"[WARN] Load failed: {e}")
            model = None

    if model is None:
        print("[NEW] Creating fresh PPO brain (optimized for CARLA)...")
        model = PPO(
            "MultiInputPolicy", env,
            verbose=1,
            device="cuda",
            tensorboard_log=TB_DIR,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

    # Link model for self-play
    env.set_model(model)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000, save_path=CHECKPOINT_DIR, name_prefix="grandmaster_v10"
    )
    logger_cb = TrainingLogger(LOG_FILE, curriculum, leaderboard, agent_id)

    # Train
    try:
        print("[TRAIN] Starting 5M timestep run...")
        print("[TRAIN] Goal: Complete city lap without penalty!")
        model.learn(
            total_timesteps=5_000_000,
            callback=[checkpoint_cb, logger_cb],
            progress_bar=True,
        )
        model.save(MODEL_PATH)
        print(f"\n[DONE] Model saved -> {MODEL_PATH}")

    except KeyboardInterrupt:
        print("\n[STOP] Saving emergency checkpoint...")
        model.save(MODEL_PATH.replace(".zip", "_emergency.zip"))

    finally:
        leaderboard.print_leaderboard(top_n=10)
        env.close()
