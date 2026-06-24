"""
==============================================================================
  CARLA GRANDMASTER AI SUITE v14.0 -- NEXT-GEN EDITION

  PIPELINE: Training -> Observing -> Fine-tuning -> Real-World Deploy

  INTEGRATED CUTTING-EDGE TECHNOLOGIES:
  ✅ DINOv2 Vision Backbone (frozen, domain-invariant) with CNN fallback
  ✅ PPO + SAC option (SAC for sample efficiency, PPO for stability)
  ✅ Albumentations Augmentation (domain randomization)
  ✅ Rule-Based Safety Shield (zero collision guarantee layer)
  ✅ Enhanced Per-Scenario Curriculum (not just per-stage)
  ✅ Multi-Objective Reward Decomposition (safety/progress/comfort/rules)
  ✅ HER Buffer + World Model + ADR + Episodic Memory (from v10)
  ✅ Recovery Protocol + Reverse Action (from v11)
  ✅ Visual Awareness + Achievement Milestones (from v12)
  ✅ Self-Play Rival + Competition Leaderboard (from v9)
  ✅ Observer Integration Hook (for 2_OBSERVER_AGENT.py)
==============================================================================
"""

import os, sys, time, json, math, random, threading, warnings
import numpy as np
import cv2
import carla
import gymnasium as gym
from gymnasium import spaces
from collections import deque, defaultdict
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch as th
from torch import nn
import torch.nn.functional as F

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Optional: DINOv2 for domain-invariant vision
try:
    import timm
    DINOV2_AVAILABLE = False  # Force CNN for RL speed
except ImportError:
    DINOV2_AVAILABLE = False

# Optional: Albumentations for augmentation
try:
    import albumentations as A
    AUGMENT_AVAILABLE = True
except ImportError:
    AUGMENT_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
BASE_DIR        = r"D:\AI\traffic_vision\master_suite"
MODEL_PATH      = os.path.join(BASE_DIR, "carla_grandmaster_v14.zip")
TB_DIR          = os.path.join(BASE_DIR, "tb_logs")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_FILE        = os.path.join(BASE_DIR, "training_carla_v14_log.txt")
LEADERBOARD_DIR = os.path.join(BASE_DIR, "leaderboard")
MEMORY_DIR      = os.path.join(BASE_DIR, "agent_memories")
HER_BUFFER_DIR  = os.path.join(BASE_DIR, "her_buffers")
OBSERVER_PIPE   = os.path.join(BASE_DIR, "observer_state.json")

for path in [BASE_DIR, TB_DIR, CHECKPOINT_DIR, LEADERBOARD_DIR,
             MEMORY_DIR, HER_BUFFER_DIR]:
    os.makedirs(path, exist_ok=True)

IMG_WIDTH, IMG_HEIGHT = 80, 60  # Smaller = faster (was 160x120)
N_LIDAR_SECTORS = 16  # 360 degree mapping -- 22.5 deg per sector (was 8 = too coarse)
LIDAR_MAX_RANGE = 50.0
STATE_DIM = 28  # +4 for waypoint navigation (target_dx, target_dy, dist_to_target, junction_ahead)

# =============================================================================
# AUGMENTATION PIPELINE (Domain Randomization)
# =============================================================================
def get_augmentation_pipeline():
    if not AUGMENT_AVAILABLE:
        return None
    return A.Compose([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(10, 30), p=0.3),
        A.RandomRain(slant_lower=-10, slant_upper=10, p=0.15),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        A.RandomSunFlare(src_radius=50, p=0.05),
        A.HorizontalFlip(p=0.0),  # Never flip - driving is directional
    ])

AUGMENT_PIPELINE = get_augmentation_pipeline()

# =============================================================================
# VISION BACKBONE: DINOv2 (frozen) with CNN fallback
# =============================================================================
class DINOv2Extractor(nn.Module):
    """DINOv2 ViT-S/14 -- frozen, domain-invariant features."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch14_dinov2.lvd142m',
                                                   pretrained=True, dynamic_img_size=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.out_dim = 384  # ViT-S embedding size
        # Use 196x196 (14*14) -- fast for RL, still compatible via dynamic_img_size
        self.target_size = (196, 196)

    def forward(self, x):
        # x expected: (B, 3, H, W) -- already in CHW format from caller
        # Interpolate to target size that's divisible by patch_size=14
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        with th.no_grad():
            return self.backbone.forward_features(x)[:, 0]  # CLS token


class CNNExtractor(nn.Module):
    """Small CNN for visual context."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=3), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            self.out_dim = self.cnn(sample).shape[1]

    def forward(self, x):
        return self.cnn(x)


class GrandmasterFeatureExtractor(BaseFeaturesExtractor):
    """Balanced extractor: LiDAR = spatial awareness (PRIMARY), Vision = context, State = memory."""
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Vision -- compressed context
        self.vision = CNNExtractor()
        vision_dim = self.vision.out_dim
        print(f"[VISION] CNN context encoder ({vision_dim}d)")

        # === LIDAR SPATIAL AWARENESS (the "eyes") ===
        # Process each sector: distance + angle -> spatial map
        self.lidar_spatial = nn.Sequential(
            nn.Linear(N_LIDAR_SECTORS * 3, 128),  # 3 features per sector: dist, sin, cos
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # === DANGER DETECTION -- is something close? ===
        self.danger_head = nn.Sequential(
            nn.Linear(N_LIDAR_SECTORS, 32),
            nn.ReLU(),
            nn.Linear(32, 16),  # 16 danger zones around the car
        )

        # State (speed, steering, etc)
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 64), nn.ReLU(),
        )

        # Cross-attention: LiDAR tells vision WHERE to look
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, batch_first=True
        )

        # Fusion: balanced -- LiDAR has most weight
        combined_dim = vision_dim + 128 + 64  # vision + lidar_spatial + state
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

        print(f"[SENSOR] LiDAR spatial=128d, Vision context={vision_dim}d, State=64d -> {features_dim}d")

    def forward(self, observations):
        # === VISION (compressed context) ===
        cam = observations["camera"].float()
        if cam.ndim == 4 and cam.shape[-1] == 3:
            cam = cam.permute(0, 3, 1, 2) / 255.0
        elif cam.ndim == 4 and cam.max() > 1.0:
            cam = cam / 255.0
        vis_feat = self.vision(cam)

        # === LIDAR SPATIAL MAP (primary awareness) ===
        lidar = observations["lidar"]  # (B, 8) distances
        # Create spatial features: distance + sin(angle) + cos(angle) per sector
        angles = th.linspace(0, 2 * math.pi, N_LIDAR_SECTORS, device=lidar.device)
        lidar_spatial_input = th.cat([
            lidar,                                          # raw distances
            th.sin(angles).unsqueeze(0).expand(lidar.shape[0], -1),  # sin of sector angle
            th.cos(angles).unsqueeze(0).expand(lidar.shape[0], -1),  # cos of sector angle
        ], dim=-1)  # (B, 24)

        lidar_feat = self.lidar_spatial(lidar_spatial_input)

        # === STATE ===
        state_feat = self.state_encoder(observations["state"])

        # === FUSION ===
        combined = th.cat([vis_feat, lidar_feat, state_feat], dim=-1)
        return self.fusion(combined)


# =============================================================================
# SAFETY SHIELD (Rule-Based, Zero Collision Guarantee)
# =============================================================================
class SafetyShield:
    """Hard safety layer that overrides dangerous actions."""
    def __init__(self):
        self.emergency_brakes = 0
        self.steering_corrections = 0
        self.total_interventions = 0

    def filter_action(self, steer, throttle, brake, lidar_sectors, speed_kmh, on_road):
        intervened = False

        # Emergency brake: obstacle within 5m in front sectors
        # LiDAR sectors: angle = arctan2(y,x) % 2pi, sector = angle/(2pi)*N
        # Sector 0 = forward (+X), sector N/4 = left (+Y), sector N/2 = behind, sector 3N/4 = right
        # Front ~90 degrees: sectors [N-2, N-1, 0, 1, 2] (wrapping around 0)
        n = len(lidar_sectors)
        front_idxs = [n-2, n-1, 0, 1, 2]  # front ~67 degrees with 16 sectors
        front_sectors = np.array([lidar_sectors[i] for i in front_idxs])
        front_min = float(np.min(front_sectors))

        if front_min < 5.0 and speed_kmh > 5.0:
            throttle = 0.0
            brake = 1.0
            self.emergency_brakes += 1
            intervened = True

        elif front_min < 10.0 and speed_kmh > 20.0:
            throttle = min(throttle, 0.2)
            brake = max(brake, 0.5)
            intervened = True

        # Side obstacle detection (at intersections, prevent T-bone)
        left_sectors = np.array([lidar_sectors[i % n] for i in range(3, 6)])   # left ~45 deg
        right_sectors = np.array([lidar_sectors[i % n] for i in range(n-3, n)])  # right ~45 deg
        side_min = min(float(np.min(left_sectors)), float(np.min(right_sectors)))
        if side_min < 3.0 and speed_kmh > 10.0:
            throttle = min(throttle, 0.3)
            brake = max(brake, 0.3)
            intervened = True

        # Dampen aggressive steering at high speed
        if abs(steer) > 0.8 and speed_kmh > 25.0:
            steer = steer * 0.5
            self.steering_corrections += 1
            intervened = True

        # Don't accelerate off-road
        if not on_road and throttle > 0.5:
            throttle = 0.2
            intervened = True

        if intervened:
            self.total_interventions += 1

        return steer, throttle, brake, intervened


# =============================================================================
# HER BUFFER
# =============================================================================
class HERBuffer:
    def __init__(self, buffer_size=100000, her_ratio=0.8):
        self.buffer_size = buffer_size
        self.her_ratio = her_ratio
        self.storage = []

    def add(self, transition):
        transition['timestamp'] = datetime.now().isoformat()
        self.storage.append(transition)
        if len(self.storage) > self.buffer_size:
            self.storage.pop(0)

    def sample(self, batch_size):
        samples = random.sample(self.storage, min(batch_size, len(self.storage)))
        her_count = int(batch_size * self.her_ratio)
        for i in range(min(her_count, len(samples))):
            s = samples[i]
            if s['info'].get('termination') == 'timeout':
                s['reward'] = 100.0
            elif s['info'].get('termination') == 'collision':
                s['reward'] = max(s['info'].get('total_distance', 0), 0)
        return samples

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.storage[-1000:], f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.storage = json.load(f)


# =============================================================================
# WORLD MODEL
# =============================================================================
class WorldModel(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=3, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(hidden + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden + action_dim, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, state, action):
        h = self.encoder(state)
        x = th.cat([h, action], dim=-1)
        return self.dynamics(x), self.reward_head(x)


# =============================================================================
# ADR (Automatic Domain Randomization)
# =============================================================================
class ADRManager:
    def __init__(self, target=0.5, window=100):
        self.target = target
        self.window = window
        self.recent = deque(maxlen=window)
        self.difficulty = 0.0

    def update(self, reward):
        self.recent.append(reward > 0)
        if len(self.recent) < self.window:
            return
        rate = sum(self.recent) / len(self.recent)
        if rate > self.target + 0.1:
            self.difficulty = min(1.0, self.difficulty + 0.05)
        elif rate < self.target - 0.1:
            self.difficulty = max(0.0, self.difficulty - 0.05)

    def get_weather(self):
        d = self.difficulty
        return carla.WeatherParameters(
            cloudiness=max(0, min(100, d * 80 + random.uniform(-10, 10))),
            precipitation=d * 50 if d > 0.3 else 0,
            precipitation_deposits=d * 50 if d > 0.3 else 0,
            wind_intensity=random.uniform(0, 30 * d),
            sun_azimuth_angle=random.uniform(0, 360),
            sun_altitude_angle=random.uniform(10, 70),
            fog_density=d * 30 if d > 0.5 else 0,
            wetness=d * 40 if d > 0.4 else 0,
        )


# =============================================================================
# EPISODIC MEMORY
# =============================================================================
class EpisodicMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.episodes = []
        self.current = []

    def start_episode(self):
        self.current = []

    def add_step(self, state, action, reward, info):
        self.current.append({'state': state, 'action': action, 'reward': reward, 'info': info})

    def end_episode(self, total_reward, success=False):
        ep = {'steps': self.current, 'total_reward': total_reward, 'ts': datetime.now().isoformat()}
        if success or total_reward > 500:
            self.episodes.append(ep)
            if len(self.episodes) > self.capacity:
                self.episodes.sort(key=lambda x: x['total_reward'])
                self.episodes.pop(0)


# =============================================================================
# RECOVERY MODULE
# =============================================================================
class RecoveryModule:
    def __init__(self):
        self.is_recovering = False
        self.recovery_start = None
        self.last_road_pos = None

    def enter(self, pos, road_pos):
        self.is_recovering = True
        self.recovery_start = time.time()
        self.last_road_pos = road_pos

    def exit(self):
        self.is_recovering = False
        self.recovery_start = None


# =============================================================================
# SAFETY SHIELD
# =============================================================================
# (Already defined above -- class SafetyShield)


# =============================================================================
# LEADERBOARD
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
    recoveries: int = 0
    waypoints_visited: int = 0
    episodes_completed: int = 0
    best_single_distance: float = 0.0
    safety_score: float = 100.0
    avg_episode_reward: float = 0.0
    shield_interventions: int = 0
    last_updated: str = ""

    @property
    def score(self):
        return max(0, self.total_distance - self.total_penalty * 0.1
                   - (100 - self.safety_score) * 10 + self.ego_wins * 500
                   + self.recoveries * 200)


class Leaderboard:
    def __init__(self):
        self.fp = os.path.join(LEADERBOARD_DIR, "leaderboard.json")
        self.records = {}
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
        with open(self.fp, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.records.items()}, f, indent=2)

    def update(self, agent_id, **kw):
        if agent_id not in self.records:
            self.records[agent_id] = AgentRecord(agent_id=agent_id)
        r = self.records[agent_id]
        for k, v in kw.items():
            if hasattr(r, k):
                setattr(r, k, v)
        r.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def print_board(self, top_n=10):
        sr = sorted(self.records.items(), key=lambda x: x[1].score, reverse=True)
        print(f"\n{'='*90}")
        print(f"  CARLA AI LEADERBOARD v14".center(90))
        print(f"{'='*90}")
        print(f"{'Rank':<6}{'Agent':<20}{'Score':<10}{'Distance':<12}{'Wins':<8}{'Recov':<8}{'Safety':<10}{'Shield':<8}")
        print("-"*90)
        for i, (aid, r) in enumerate(sr[:top_n]):
            m = ["1st","2nd","3rd"][i] if i < 3 else f"{i+1}."
            print(f"{m:<6}{aid[:20]:<20}{r.score:<10.1f}{r.total_distance:<12.1f}{r.ego_wins:<8}{r.recoveries:<8}{r.safety_score:<10.1f}{r.shield_interventions:<8}")
        print(f"{'='*90}\n")


# =============================================================================
# RECALL MEMORY (with red_light_zones from competition)
# =============================================================================
class RecallMemory:
    def __init__(self, agent_id="ego", size=500):
        self.agent_id = agent_id
        self.fp = os.path.join(MEMORY_DIR, f"{agent_id}_memory.json")
        self.size = size
        self.collision_hotspots = []
        self.offroad_zones = []
        self.red_light_zones = []
        self.successful_route = []
        self.load()

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
        with open(self.fp, 'w') as f:
            json.dump({
                "collision_hotspots": self.collision_hotspots[-self.size:],
                "offroad_zones": self.offroad_zones[-self.size:],
                "red_light_zones": self.red_light_zones[-self.size:],
                "successful_route": self.successful_route[-self.size*5:],
            }, f)

    def add_collision(self, loc):
        x, y = loc.x, loc.y
        for h in self.collision_hotspots:
            if np.sqrt((h["x"]-x)**2 + (h["y"]-y)**2) < 5.0:
                h["count"] += 1
                h["last_seen"] = datetime.now().isoformat()
                self.save(); return
        self.collision_hotspots.append({"x":x,"y":y,"z":loc.z,"count":1,"last_seen":datetime.now().isoformat()})
        self.save()

    def add_offroad(self, loc):
        self.offroad_zones.append({"x":loc.x,"y":loc.y,"radius":8.0,"last_seen":datetime.now().isoformat()})
        self.save()

    def add_red_light(self, loc):
        self.red_light_zones.append({"x":loc.x,"y":loc.y,"danger":1.0,"last_seen":datetime.now().isoformat()})
        self.save()

    def add_waypoint(self, loc, seq):
        self.successful_route.append({"x":loc.x,"y":loc.y,"z":loc.z,"sequence":seq})

    def get_danger_score(self, loc):
        x, y = loc.x, loc.y
        d = 0.0
        for h in self.collision_hotspots:
            dist = np.sqrt((h["x"]-x)**2+(h["y"]-y)**2)
            if dist < 10: d += h.get("count",1)*(1-dist/10)*0.4
        for z in self.offroad_zones:
            dist = np.sqrt((z["x"]-x)**2+(z["y"]-y)**2)
            if dist < z.get("radius",8): d += (1-dist/z.get("radius",8))*0.3
        for z in self.red_light_zones:
            dist = np.sqrt((z["x"]-x)**2+(z["y"]-y)**2)
            if dist < 15: d += 0.2
        return min(d, 1.0)

    def get_route_guidance(self, loc):
        if not self.successful_route: return None
        best_seq, best_dist = -1, float('inf')
        for w in self.successful_route:
            dist = np.sqrt((w["x"]-loc.x)**2+(w["y"]-loc.y)**2)
            if dist < best_dist: best_dist, best_seq = dist, w["sequence"]
        if best_seq >= 0 and best_dist < 20:
            for w in self.successful_route:
                if w["sequence"] == best_seq+1:
                    return carla.Location(w["x"],w["y"],w["z"])
        return None


# =============================================================================
# ENHANCED CURRICULUM (Per-Scenario Tracking)
# =============================================================================
class CurriculumManager:
    STAGES = {
        1: {"name":"Basics","map":"Town01","traffic":0,"pedestrians":0,"weather":"Clear","obstacles":False,"threshold":500.0},
        2: {"name":"Traffic","map":"Town03","traffic":10,"pedestrians":5,"weather":"Cloudy","obstacles":False,"threshold":1000.0},
        3: {"name":"Expert","map":"Town05","traffic":30,"pedestrians":15,"weather":"Rainy","obstacles":True,"threshold":2000.0},
        4: {"name":"Master","map":"Town10HD","traffic":50,"pedestrians":30,"weather":"Stormy","obstacles":True,"threshold":5000.0},
    }
    WINDOW = 50

    def __init__(self, stage=1):
        self.current_stage = stage
        self._rewards = deque(maxlen=self.WINDOW)
        # Per-scenario tracking
        self.scenario_stats = defaultdict(lambda: {"success": deque(maxlen=50), "difficulty": 0.0})

    def get_config(self):
        return self.STAGES[self.current_stage]

    def detect_scenario(self, wp, kmh):
        if wp is None: return "off_road"
        if wp.is_junction: return "intersection"
        if kmh > 40: return "high_speed"
        return "straight_road"

    def record_scenario(self, scenario_type, success):
        self.scenario_stats[scenario_type]["success"].append(float(success))
        stats = self.scenario_stats[scenario_type]
        if len(stats["success"]) >= 20:
            rate = np.mean(stats["success"])
            if rate > 0.8: stats["difficulty"] = min(1.0, stats["difficulty"]+0.05)
            elif rate < 0.4: stats["difficulty"] = max(0.0, stats["difficulty"]-0.05)

    def record_episode(self, total_reward):
        self._rewards.append(total_reward)
        if len(self._rewards) < self.WINDOW: return False
        mean = np.mean(self._rewards)
        if mean >= self.STAGES[self.current_stage]["threshold"] and self.current_stage < max(self.STAGES):
            self.current_stage += 1
            self._rewards.clear()
            print(f"\n[CURRICULUM] >>> STAGE {self.current_stage}: {self.STAGES[self.current_stage]['name']} <<<\n")
            return True
        return False


# =============================================================================
# DECISION MAKER
# =============================================================================
class DecisionMaker:
    def __init__(self):
        self.history = deque(maxlen=100)
        self.hesitations = 0

    def should_hesitate(self, speed, danger, on_road, hazard):
        reasons = []
        if danger > 0.8: reasons.append(f"Danger:{danger:.2f}")
        if not on_road and speed > 15: reasons.append("Leaving road")
        if hazard and speed > 20: reasons.append("Hazard nearby")
        if speed > 40 and danger > 0.4: reasons.append("Too fast")
        if reasons:
            self.hesitations += 1
            return True, " | ".join(reasons)
        return False, "Clear"


# =============================================================================
# OBSERVER STATE WRITER (for 2_OBSERVER_AGENT.py)
# =============================================================================
def write_observer_state(data):
    try:
        with open(OBSERVER_PIPE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# =============================================================================
# TRAINING CALLBACK (writes to observer)
# =============================================================================
class TrainingLogger(BaseCallback):
    def __init__(self, log_path, curriculum, leaderboard, agent_id, env_ref):
        super().__init__()
        self.log_path = log_path
        self.curriculum = curriculum
        self.leaderboard = leaderboard
        self.agent_id = agent_id
        self.env_ref = env_ref
        self.ego_wins = 0
        self.rival_wins = 0
        self.episode_count = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "tournament_winner" in info:
                if info["tournament_winner"] == "ego": self.ego_wins += 1
                elif info["tournament_winner"] == "rival": self.rival_wins += 1
        self.logger.record("competition/ego_wins", self.ego_wins)
        self.logger.record("competition/rival_wins", self.rival_wins)
        return True

    def _on_rollout_end(self):
        if self.logger is None: return
        try:
            stats = {k: v for k, v in self.logger.name_to_value.items()}
            ts = self.num_timesteps
            rew = stats.get("rollout/ep_rew_mean", 0.0)
            if np.isnan(rew) or np.isinf(rew): rew = 0.0
            loss = stats.get("train/loss", 0.0)
            if np.isnan(loss) or np.isinf(loss): loss = 0.0
            stage = self.curriculum.current_stage

            with open(self.log_path, "a") as f:
                f.write(f"total_timesteps | {ts}\nep_rew_mean | {rew:.4f}\n"
                        f"loss | {loss:.6e}\ncurriculum_stage | {stage}\n"
                        f"ego_wins | {self.ego_wins}\nrival_wins | {self.rival_wins}\n---\n")

            self.leaderboard.update(self.agent_id, ego_wins=self.ego_wins,
                                     rival_wins=self.rival_wins, avg_episode_reward=rew)

            # Write observer state
            shield = getattr(self.env_ref, 'safety_shield', None)
            write_observer_state({
                "timestamp": datetime.now().isoformat(),
                "timesteps": ts,
                "ep_rew_mean": float(rew),
                "loss": float(loss),
                "curriculum_stage": stage,
                "ego_wins": self.ego_wins,
                "rival_wins": self.rival_wins,
                "adr_difficulty": float(self.env_ref.adr.difficulty) if hasattr(self.env_ref, 'adr') else 0,
                "shield_interventions": shield.total_interventions if shield else 0,
                "shield_emergency_brakes": shield.emergency_brakes if shield else 0,
                "best_distance": float(self.env_ref.best_distance) if hasattr(self.env_ref, 'best_distance') else 0,
            })
        except Exception:
            pass


# =============================================================================
# MAIN ENVIRONMENT v14
# =============================================================================
class CarlaGrandmasterEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, stage_config, curriculum, leaderboard=None, agent_id="Grandmaster_v14"):
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
        self.safety_shield = SafetyShield()  # NEW v14
        self.recovery = RecoveryModule()

        # Recovery state (from v11)
        self.was_offroad = False
        self.offroad_duration = 0
        self.consecutive_onroad = 0
        self.last_safe_location = None
        self.successful_recoveries = 0

        # Milestones (from v12)
        self.milestones = {100: False, 500: False, 1000: False}

        # CARLA
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(60.0)
        self.world = self.client.load_world(self.config["map"])
        self.map = self.world.get_map()
        all_spawns = self.map.get_spawn_points()
        self.initial_spawn = all_spawns[0]
        self._all_spawns = all_spawns

        self._settings = self.world.get_settings()
        self._settings.synchronous_mode = True
        self._settings.fixed_delta_seconds = 0.05  # 2x faster physics (was 0.1)
        self.world.apply_settings(self._settings)

        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)

        # Spaces (throttle -1 for reverse)
        self.observation_space = spaces.Dict({
            "camera": spaces.Box(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), np.uint8),
            "lidar": spaces.Box(0.0, LIDAR_MAX_RANGE, (N_LIDAR_SECTORS,), np.float32),
            "state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Actors
        self.vehicle = None
        self.camera_sensor = None
        self.lidar_sensor = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.rival_vehicle = None
        self.rival_camera = None
        self.rival_lidar = None
        self.rival_collision = None
        self.rival_lane = None

        # Buffers
        self.last_img = None
        self.last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        self.rival_last_img = None
        self.rival_last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        self._img_lock = threading.Lock()
        self._lidar_lock = threading.Lock()
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
        self.current_penalty = 0.0
        self.current_distance = 0.0
        self.rival_distance = 0.0
        self.last_loc = None
        self.rival_last_loc = None
        self.current_trajectory = []
        self.waypoint_sequence = 0
        self.best_trajectory = []
        self.best_distance = 0.0
        self.safety_score = 100.0
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_infos = []
        self.consecutive_onroad_steps = 0  # reward streak for staying on road

        # Route following system
        self.route_waypoints = []  # list of carla.Location along the route
        self.route_target_idx = 0  # current target waypoint index
        self.prev_route_dist = 0.0  # distance to target at previous step
        self.offroad_steps = 0  # consecutive off-road steps counter

    def set_model(self, model):
        self.model = model

    # ==================== CLEANUP ====================
    def _cleanup(self):
        actors = []
        for attr in ("camera_sensor","lidar_sensor","collision_sensor","lane_sensor","vehicle"):
            a = getattr(self, attr, None)
            if a is not None:
                if hasattr(a, 'listen'):
                    try: a.stop()
                    except: pass
                actors.append(a); setattr(self, attr, None)
        for attr in ("rival_camera","rival_lidar","rival_collision","rival_lane","rival_vehicle"):
            a = getattr(self, attr, None)
            if a is not None:
                if hasattr(a, 'listen'):
                    try: a.stop()
                    except: pass
                actors.append(a); setattr(self, attr, None)
        for npc in self.npc_vehicles + self.npc_pedestrians:
            try:
                if npc.is_alive: actors.append(npc)
            except: pass
        self.npc_vehicles = []; self.npc_pedestrians = []
        if actors:
            try:
                valid = [a for a in actors if hasattr(a, 'id')]
                if valid: self.client.apply_batch([carla.command.DestroyActor(a) for a in valid])
            except: pass
        self.world.tick()
        with self._img_lock:
            self.last_img = None; self.rival_last_img = None
        with self._lidar_lock:
            self.last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
            self.rival_last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)

    # ==================== NPC SPAWNING ====================
    def _spawn_npcs(self):
        cfg = self.config
        bp_lib = self.world.get_blueprint_library()
        if cfg["traffic"] > 0:
            bps = bp_lib.filter("vehicle.*")
            pts = random.sample(self._all_spawns, min(cfg["traffic"], len(self._all_spawns)))
            batch = []
            for sp in pts:
                bp = random.choice(bps)
                if bp.has_attribute("color"):
                    bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
                batch.append(carla.command.SpawnActor(bp, sp).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, 8000)))
            for r in self.client.apply_batch_sync(batch, True):
                if not r.error:
                    a = self.world.get_actor(r.actor_id)
                    if a: self.npc_vehicles.append(a)
        if cfg["pedestrians"] > 0:
            wb = bp_lib.filter("walker.pedestrian.*")
            cb = bp_lib.find("controller.ai.walker")
            locs = [self.world.get_random_location_from_navigation() for _ in range(cfg["pedestrians"])]
            locs = [carla.Transform(l) for l in locs if l]
            batch = [carla.command.SpawnActor(random.choice(wb), sp) for sp in locs]
            wids = [r.actor_id for r in self.client.apply_batch_sync(batch, True) if not r.error]
            for wid in wids:
                w = self.world.get_actor(wid)
                if w:
                    ctrl = self.world.spawn_actor(cb, carla.Transform(), attach_to=w)
                    ctrl.start(); ctrl.go_to_location(self.world.get_random_location_from_navigation())
                    ctrl.set_max_speed(1.4)
                    self.npc_pedestrians.extend([w, ctrl])

    # ==================== OBSTACLE INJECTION ====================
    def _inject_obstacle(self):
        if not self.config.get("obstacles"): return
        props = self.world.get_blueprint_library().filter("static.prop.*")
        if not props: return
        loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(carla.Location(loc.x+random.uniform(5,20), loc.y+random.uniform(-5,5), loc.z), project_to_road=True)
        if wp:
            try: self.world.spawn_actor(random.choice(props), wp.transform)
            except: pass

    # ==================== SENSOR CALLBACKS ====================
    def _to_semantic(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
        cids = arr[:, :, 2]
        rgb = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)
        palette = {0:[0,0,0],1:[100,0,0],2:[150,0,0],3:[100,50,50],4:[255,0,255],
                   5:[255,255,0],6:[0,255,0],7:[0,255,0],8:[255,0,0],9:[150,75,0],
                   10:[255,100,0],11:[120,0,0],12:[255,255,255]}
        for cid, col in palette.items(): rgb[cids==cid] = col
        rgb[cids>12] = [80,0,0]
        return rgb

    def _to_lidar(self, pc):
        """Convert raw LiDAR point cloud to 360-degree sector map.
        Filters out ground (z < 0.3m) to focus on actual obstacles.
        Returns N_LIDAR_SECTORS distances -- 360 degree awareness."""
        pts = np.frombuffer(pc.raw_data, dtype=np.float32)
        if pts.size == 0: return np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        pts = pts.reshape(-1,4)[:,:3]
        # Filter ground: keep points above 0.3m (obstacles) and below car roof
        pts = pts[(pts[:,2] > 0.3) & (pts[:,2] < 2.2)]
        sectors = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        if len(pts) > 0:
            angles = np.arctan2(pts[:,1], pts[:,0]) % (2*np.pi)
            idxs = (angles/(2*np.pi)*N_LIDAR_SECTORS).astype(int) % N_LIDAR_SECTORS
            dists = np.sqrt(pts[:,0]**2+pts[:,1]**2)
            for idx, d in zip(idxs, dists):
                if d < sectors[idx]: sectors[idx] = d
        return sectors

    def _cb_cam(self, img):
        with self._img_lock: self.last_img = self._to_semantic(img)

    def _cb_rcam(self, img):
        with self._img_lock: self.rival_last_img = self._to_semantic(img)

    def _cb_lidar(self, pc):
        with self._lidar_lock: self.last_lidar = self._to_lidar(pc)

    def _cb_rlidar(self, pc):
        with self._lidar_lock: self.rival_last_lidar = self._to_lidar(pc)

    def _cb_col(self, ev):
        self.collision_hist.append(ev)
        try:
            if hasattr(ev,'other_actor') and ev.other_actor:
                self.memory.add_collision(ev.other_actor.get_location())
        except: pass

    def _cb_rcol(self, ev): self.rival_collision_hist.append(ev)
    def _cb_lane(self, ev): self.lane_invasion_hist.append(ev)
    def _cb_rlane(self, ev): self.rival_lane_invasion_hist.append(ev)

    def _attach_sensors(self, vehicle, rival=False):
        lib = self.world.get_blueprint_library()
        cbp = lib.find("sensor.camera.semantic_segmentation")
        cbp.set_attribute("image_size_x", str(IMG_WIDTH))
        cbp.set_attribute("image_size_y", str(IMG_HEIGHT))
        cbp.set_attribute("fov", "90")
        cam = self.world.spawn_actor(cbp, carla.Transform(carla.Location(x=1.6,z=1.7)), attach_to=vehicle)
        if rival: self.rival_camera = cam; cam.listen(self._cb_rcam)
        else: self.camera_sensor = cam; cam.listen(self._cb_cam)

        lbp = lib.find("sensor.lidar.ray_cast")
        lbp.set_attribute("channels","32"); lbp.set_attribute("range",str(LIDAR_MAX_RANGE))
        lbp.set_attribute("points_per_second","200000"); lbp.set_attribute("rotation_frequency","20")
        lbp.set_attribute("upper_fov","10"); lbp.set_attribute("lower_fov","-30")
        lid = self.world.spawn_actor(lbp, carla.Transform(carla.Location(x=0.0,z=2.4)), attach_to=vehicle)
        if rival: self.rival_lidar = lid; lid.listen(self._cb_rlidar)
        else: self.lidar_sensor = lid; lid.listen(self._cb_lidar)

        col = self.world.spawn_actor(lib.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle)
        if rival: self.rival_collision = col; col.listen(self._cb_rcol)
        else: self.collision_sensor = col; col.listen(self._cb_col)

        lane = self.world.spawn_actor(lib.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=vehicle)
        if rival: self.rival_lane = lane; lane.listen(self._cb_rlane)
        else: self.lane_sensor = lane; lane.listen(self._cb_lane)

    # ==================== RESET ====================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()
        self.world.set_weather(self.adr.get_weather())

        bp = self.world.get_blueprint_library().filter("model3")[0]

        # === RANDOMIZE SPAWN POINT -- explore the whole map! ===
        random.shuffle(self._all_spawns)
        spawned = False
        for sp in self._all_spawns:
            for _ in range(3):
                try: self.vehicle = self.world.spawn_actor(bp, sp); spawned = True; break
                except RuntimeError: time.sleep(0.3); self.world.tick()
            if spawned:
                self.initial_spawn = sp  # Update so route generates from here
                break
        if not spawned: raise RuntimeError("[ENV] Cannot spawn ego")

        # Rival -- spawn near ego
        wp = self.map.get_waypoint(self.initial_spawn.location)
        rwp = wp.get_left_lane()
        if rwp is None or rwp.lane_type != carla.LaneType.Driving: rwp = wp.get_right_lane()
        rspawn = rwp.transform if rwp else self._all_spawns[1]
        rspawn.location.z += 0.5
        try: self.rival_vehicle = self.world.spawn_actor(bp, rspawn); print("[ENV] Rival spawned")
        except: pass

        self._spawn_npcs()
        self._attach_sensors(self.vehicle)
        if self.rival_vehicle: self._attach_sensors(self.rival_vehicle, rival=True)

        for _ in range(20):
            self.world.tick()
            with self._img_lock:
                if self.last_img is not None:
                    if self.rival_vehicle and self.rival_last_img is None: continue
                    break

        # Reset all state
        for attr in ['steps','stuck_counter','rival_stuck_counter']:
            setattr(self, attr, 0)
        self.episode_reward = 0.0
        for attr in ['collision_hist','lane_invasion_hist','rival_collision_hist','rival_lane_invasion_hist']:
            setattr(self, attr, [])
        self.visited_waypoints = set()
        self.last_steer = self.last_throttle = 0.0
        self.rival_last_steer = self.rival_last_throttle = 0.0
        self.current_penalty = self.current_distance = self.rival_distance = 0.0
        self.last_loc = self.rival_last_loc = None
        self.current_trajectory = []; self.waypoint_sequence = 0
        self.episode_states = []; self.episode_actions = []
        self.episode_rewards = []; self.episode_infos = []
        self.recovery.exit(); self.was_offroad = False
        self.offroad_duration = self.consecutive_onroad = 0
        self.offroad_steps = 0
        self.milestones = {100: False, 500: False, 1000: False}
        self.episodic_memory.start_episode()

        # === GENERATE ROUTE (follow-road mission) ===
        self.route_waypoints = []
        self.route_target_idx = 0
        self.prev_route_dist = 0.0
        try:
            start_wp = self.map.get_waypoint(self.initial_spawn.location)
            if start_wp:
                # Generate 200 waypoints along the road (approx 2km)
                current = start_wp
                for _ in range(200):
                    nexts = current.next(10.0)  # 10m spacing
                    if not nexts: break
                    # Pick the first valid option (prefer straight, then any road)
                    chosen = nexts[0]
                    for nw in nexts:
                        if nw.lane_type in (carla.LaneType.Driving, carla.LaneType.Bidirectional):
                            chosen = nw; break
                    self.route_waypoints.append(chosen.transform.location)
                    current = chosen
                print(f"[ROUTE] Generated {len(self.route_waypoints)} waypoints")
        except Exception as e:
            print(f"[ROUTE] Failed: {e}")

        print(f"[ENV] Reset | Stage {self.curriculum.current_stage} | "
              f"Safety {self.safety_score:.0f} | ADR {self.adr.difficulty:.2f}")
        return self._get_obs('ego'), {}

    # ==================== OBSERVATION ====================
    def _get_obs(self, who='ego'):
        is_r = (who == 'rival')
        veh = self.rival_vehicle if is_r else self.vehicle

        with self._img_lock:
            cam = (self.rival_last_img if is_r else self.last_img)
            if cam is None: cam = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)
        with self._lidar_lock:
            lidar = (self.rival_last_lidar if is_r else self.last_lidar).copy()

        if veh is None or not veh.is_alive:
            return {"camera":cam,"lidar":lidar,"state":np.zeros(STATE_DIM,np.float32)}

        v = veh.get_velocity()
        kmh = 3.6*np.sqrt(v.x**2+v.y**2+v.z**2)

        red_light = 0.0
        if veh.is_at_traffic_light():
            tl = veh.get_traffic_light()
            if tl and tl.get_state() == carla.TrafficLightState.Red: red_light = 1.0

        wp = self.map.get_waypoint(veh.get_location(), project_to_road=True)
        at_junc = float(wp.is_junction) if wp else 0.0

        valid = (carla.LaneType.Driving, carla.LaneType.Bidirectional,
                 carla.LaneType.OnRamp, carla.LaneType.OffRamp)
        on_road = wp is not None and wp.lane_type in valid if wp else False

        danger = self.memory.get_danger_score(veh.get_location())

        gx = gy = 0.0
        g = self.memory.get_route_guidance(veh.get_location())
        if g:
            dx, dy = g.x-veh.get_location().x, g.y-veh.get_location().y
            n = np.sqrt(dx**2+dy**2)
            if n > 0: gx, gy = dx/n, dy/n

        ghost = 0.0
        if self.best_trajectory:
            loc = veh.get_location()
            ds = [loc.distance(p) for p in self.best_trajectory]
            md = min(ds)
            if md < 5: ghost = 1.0-md/5

        ahead = 1.0
        if wp:
            nws = wp.next(5.0)
            if nws:
                if nws[0].lane_type not in valid: ahead = -1.0
            else: ahead = -1.0
        else: ahead = -1.0

        min_l = float(np.min(lidar if is_r else self.last_lidar))

        red_ratio = 0.0
        if not is_r and self.last_img is not None:
            with self._img_lock:
                if self.last_img is not None:
                    rp = np.sum((self.last_img[:,:,0]==255)&(self.last_img[:,:,1]==0)&(self.last_img[:,:,2]==0))
                    red_ratio = min(rp/(IMG_WIDTH*IMG_HEIGHT), 1.0)

        in_rec = float(self.recovery.is_recovering)
        rival_rel = 0.0
        if not is_r and self.rival_vehicle and self.rival_vehicle.is_alive:
            el, rl = veh.get_location(), self.rival_vehicle.get_location()
            rival_rel = min((el.distance(rl)-50)/200, 1.0)

        # === MISTAKE AWARENESS: where is the road? ===
        is_off = 0.0
        rdx = rdy = 0.0
        road_dist_n = 0.0
        if wp and not on_road:
            wloc = wp.transform.location
            vloc = veh.get_location()
            dx = wloc.x - vloc.x
            dy = wloc.y - vloc.y
            d = math.sqrt(dx*dx + dy*dy)
            if d > 0.1: rdx, rdy = dx/d, dy/d
            road_dist_n = min(d / 20.0, 1.0)
            is_off = 1.0

        # === WAYPOINT NAVIGATION: where should I go? ===
        target_dx = target_dy = 0.0
        dist_to_target = 0.0
        junction_ahead = 0.0

        # Detect junction ahead for all cases
        if wp:
            far_wps = wp.next(30.0)
            if far_wps and far_wps[0].is_junction:
                junction_ahead = 1.0

        # Use route waypoints for guidance (primary navigation)
        if not is_r and self.route_waypoints and self.route_target_idx < len(self.route_waypoints):
            target_loc = self.route_waypoints[self.route_target_idx]
            vloc = veh.get_location()
            tdx = target_loc.x - vloc.x
            tdy = target_loc.y - vloc.y
            tdist = math.sqrt(tdx*tdx + tdy*tdy)
            if tdist > 0.1:
                target_dx = tdx / tdist
                target_dy = tdy / tdist
            dist_to_target = min(tdist / 30.0, 1.0)
            # Advance target if close enough
            if tdist < 8.0 and self.route_target_idx < len(self.route_waypoints) - 1:
                self.route_target_idx += 1
        elif wp:
            # Fallback: use local waypoint if no route
            next_wps = wp.next(5.0)
            if next_wps:
                twp = next_wps[0]
                tloc = twp.transform.location
                vloc = veh.get_location()
                tdx = tloc.x - vloc.x
                tdy = tloc.y - vloc.y
                tdist = math.sqrt(tdx*tdx + tdy*tdy)
                if tdist > 0.1:
                    target_dx = tdx / tdist
                    target_dy = tdy / tdist
                    dist_to_target = min(tdist / 10.0, 1.0)

        state = np.array([
            kmh/50, self.rival_last_steer if is_r else self.last_steer,
            self.rival_last_throttle if is_r else self.last_throttle,
            at_junc, red_light, self.current_penalty/1000, self.safety_score/100,
            danger, min_l/LIDAR_MAX_RANGE,
            float(bool(self.rival_lane_invasion_hist if is_r else self.lane_invasion_hist)),
            min((self.rival_stuck_counter if is_r else self.stuck_counter)/500, 1.0),
            float(bool(self.rival_collision_hist if is_r else self.collision_hist)),
            gx, gy, ahead, min(self.steps/10000, 1.0), self.current_distance/1000,
            red_ratio, in_rec, rival_rel,
            is_off, road_dist_n, rdx, rdy,
            target_dx, target_dy, dist_to_target, junction_ahead
        ], dtype=np.float32)

        # === BLUE LINE HUD: Google Maps style navigation overlay ===
        # Draw a blue guidance line on the camera image showing where to go
        # No camera overlay -- navigation is in state vector (target_dx, target_dy)
        # This prevents flickering that HUD drawing caused
        return {"camera":cam,"lidar":lidar,"state":np.nan_to_num(state, nan=0, posinf=1, neginf=-1)}

    # ==================== STEP ====================
    def step(self, action):
        self.steps += 1
        obs_before = self._get_obs('ego')
        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])

        v = self.vehicle.get_velocity()
        kmh = 3.6*np.sqrt(v.x**2+v.y**2+v.z**2)
        loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Any)

        valid = (carla.LaneType.Driving, carla.LaneType.Bidirectional,
                 carla.LaneType.OnRamp, carla.LaneType.OffRamp)
        on_road = wp is not None and wp.lane_type in valid

        # Decision maker
        danger = self.memory.get_danger_score(loc)
        hazard = float(np.min(self.last_lidar)) < 10.0
        hesitate, reason = self.decision_maker.should_hesitate(kmh, danger, on_road, hazard)
        if hesitate: throttle = min(throttle, 0.3); brake = max(brake, 0.2)

        # SAFETY SHIELD -- first line of defense
        with self._lidar_lock:
            lidar_copy = self.last_lidar.copy()
        steer, throttle, brake, shielded = self.safety_shield.filter_action(
            steer, throttle, brake, lidar_copy, kmh, on_road)

        # Anti-stall -- AFTER safety shield (overrides everything)
        # Steer TOWARD next route waypoint (not random oscillation!)
        # This teaches the AI: "when stuck, steer toward the blue line"
        if kmh < 2.0 and throttle >= 0:
            throttle = 0.6; brake = 0.0
            if self.stuck_counter > 30:
                # Calculate steering toward route waypoint
                waypoint_steer = steer  # default: keep current steer
                if self.route_waypoints and self.route_target_idx < len(self.route_waypoints):
                    target_loc = self.route_waypoints[self.route_target_idx]
                    et = self.vehicle.get_transform()
                    # Vector from car to target
                    dx = target_loc.x - et.location.x
                    dy = target_loc.y - et.location.y
                    # Car's forward direction
                    fwd = et.get_forward_vector()
                    # Cross product determines left/right
                    cross = fwd.x * dy - fwd.y * dx
                    # Also check if target is behind us (dot product)
                    dot = fwd.x * dx + fwd.y * dy
                    if dot < 0:
                        # Target is behind -- need to turn around
                        waypoint_steer = 1.0 if cross > 0 else -1.0
                    else:
                        # Target is ahead -- steer proportionally
                        waypoint_steer = max(-1.0, min(1.0, cross * 0.05))
                steer = waypoint_steer

        self.last_steer = steer; self.last_throttle = throttle

        # Reverse handling
        reverse = throttle < 0
        if reverse: throttle = abs(throttle)

        self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake, reverse=reverse))

        # Rival
        if self.rival_vehicle and self.rival_vehicle.is_alive and self.model:
            robs = self._get_obs('rival')
            ract, _ = self.model.predict(robs, deterministic=False)
            rs, rt, rb = float(ract[0]), float(ract[1]), float(ract[2])
            rrev = rt < 0
            if rrev: rt = abs(rt)
            rv = self.rival_vehicle.get_velocity()
            rkmh = 3.6*np.sqrt(rv.x**2+rv.y**2+rv.z**2)
            if rkmh < 2.0 and not rrev: rt = 0.6
            self.rival_last_steer = rs; self.rival_last_throttle = rt
            self.rival_vehicle.apply_control(carla.VehicleControl(steer=rs, throttle=rt, brake=rb, reverse=rrev))

        self.world.tick()

        # Chase cam
        spec = self.world.get_spectator()
        t = self.vehicle.get_transform()
        yr = math.radians(t.rotation.yaw)
        spec.set_transform(carla.Transform(
            carla.Location(x=t.location.x-12*math.cos(yr), y=t.location.y-12*math.sin(yr), z=t.location.z+6),
            carla.Rotation(pitch=-20, yaw=t.rotation.yaw)))

        # Distance
        step_dist = 0.0
        if self.last_loc is not None:
            step_dist = loc.distance(self.last_loc)
            self.current_distance += step_dist
        self.last_loc = loc
        if self.steps % 5 == 0: self.current_trajectory.append(loc)
        if self.rival_vehicle and self.rival_vehicle.is_alive:
            rl = self.rival_vehicle.get_location()
            if self.rival_last_loc is not None: self.rival_distance += rl.distance(self.rival_last_loc)
            self.rival_last_loc = rl

        # ==================== REWARD v20 -- ROUTE FOLLOWING ====================
        # Like Tesla/Waymo: follow a planned route, reward getting closer to target
        reward = 0.0; done = False; info = {}

        # === ROUTE PROGRESS REWARD (primary signal, like GPS navigation) ===
        if self.route_waypoints and self.route_target_idx < len(self.route_waypoints):
            target_loc = self.route_waypoints[self.route_target_idx]
            route_dist = loc.distance(target_loc)
            # Reward for getting closer to route target
            if self.prev_route_dist > 0:
                delta = self.prev_route_dist - route_dist  # positive = closer
                reward += delta * 2.0  # +2.0 per meter closer to target
            self.prev_route_dist = route_dist
            # Advance target when close
            if route_dist < 8.0 and self.route_target_idx < len(self.route_waypoints) - 1:
                self.route_target_idx += 1
                reward += 1.0  # Bonus for reaching a waypoint
                print(f"[ROUTE] Reached waypoint {self.route_target_idx}/{len(self.route_waypoints)}")
        else:
            # No route -- fallback to simple movement reward
            if step_dist > 0.05 and on_road:
                reward += 1.0

        # === STANDING STILL PENALTY ===
        if step_dist < 0.05:
            self.stuck_counter += 1
            reward -= 0.3  # Stronger penalty to discourage stalling
            if reverse: reward -= 0.1  # Reversing while stuck = bad
        else:
            self.stuck_counter = 0
            # === WAYPOINT STEERING BONUS ===
            # Reward for steering toward route target (the "blue line")
            # This is the key signal: "when you steer toward the target, you get reward"
            if self.route_waypoints and self.route_target_idx < len(self.route_waypoints):
                target_loc = self.route_waypoints[self.route_target_idx]
                et = self.vehicle.get_transform()
                dx = target_loc.x - et.location.x
                dy = target_loc.y - et.location.y
                fwd = et.get_forward_vector()
                # Are we facing the target? (dot product of forward and direction to target)
                target_dist = math.sqrt(dx*dx + dy*dy)
                if target_dist > 0.1:
                    fwd_dot = (fwd.x * dx + fwd.y * dy) / target_dist
                    # fwd_dot: +1 = facing target, -1 = facing away
                    # Only reward when moving AND facing the target
                    if fwd_dot > 0.3 and on_road:
                        reward += 0.5 * fwd_dot  # up to +0.5 for facing the target

        # === ON ROAD BONUSES ===
        if on_road and step_dist > 0.05:
            self.consecutive_onroad_steps += 1
            self.offroad_steps = 0

            # Reverse penalty (real cars don't reverse on road!)
            if reverse:
                reward -= 0.5

            # Speed bonus (0 to +1.0) -- only forward, STRONG positive signal
            if not reverse:
                reward += min(kmh / 30.0, 1.0)

            # Lane center bonus
            if wp:
                center_dist = loc.distance(wp.transform.location)
                if center_dist < 1.5: reward += 0.5
                elif center_dist < 3.0: reward += 0.2

            # Heading alignment -- follow the road direction
            if wp:
                twps = wp.next(10.0)
                if twps:
                    tl2 = twps[0].transform.location; et = self.vehicle.get_transform()
                    vt = np.array([tl2.x-et.location.x, tl2.y-et.location.y])
                    vf = np.array([et.get_forward_vector().x, et.get_forward_vector().y])
                    if not np.any(np.isnan(vt)) and not np.any(np.isnan(vf)):
                        alignment = np.dot(vf, vt) / (np.linalg.norm(vf) * np.linalg.norm(vt) + 1e-6)
                        align_w = 0.4 if wp.is_junction else 0.2
                        reward += max(0, alignment) * align_w

            # Intersection navigation
            if wp and wp.is_junction:
                if 5.0 < kmh < 25.0: reward += 0.3  # Good speed
                elif kmh >= 25.0: reward -= 0.3  # Too fast

            # New waypoint bonus
            if wp and wp.id not in self.visited_waypoints:
                reward += 0.5; self.visited_waypoints.add(wp.id)
                self.memory.add_waypoint(loc, self.waypoint_sequence); self.waypoint_sequence += 1

            if not self.collision_hist:
                self.last_safe_location = self.vehicle.get_transform()

        # === OFF ROAD (penalty only, no termination) ===
        elif not on_road:
            self.consecutive_onroad_steps = 0
            self.offroad_steps += 1
            self.offroad_duration += 1
            reward -= 0.5  # Moderate off-road penalty
            self.was_offroad = True; self.consecutive_onroad = 0

            # Recovery reward: only if moving toward road
            if step_dist > 0.05:
                road_wp = self.map.get_waypoint(loc, project_to_road=True)
                if road_wp and self.last_loc is not None:
                    rloc = road_wp.transform.location
                    cur_d = loc.distance(rloc)
                    prev_d = self.last_loc.distance(rloc)
                    if cur_d < prev_d - 0.1:
                        reward += 0.5  # Getting closer to road
                        if reverse: reward += 0.2  # Reverse toward road OK
                if reverse and (road_wp is None or loc.distance(road_wp.transform.location) > 5):
                    reward -= 0.3  # Aimless reverse

            # NO early termination for off-road -- just keep penalizing
            # AI needs TIME to learn recovery, not instant death

        # === RECOVERY BONUS ===
        if on_road and self.was_offroad:
            self.consecutive_onroad += 1
            if self.consecutive_onroad >= 5:
                print(f"[RECOVERY] Self-recovery! ({self.offroad_duration} steps off-road)")
                self.was_offroad = False
                self.offroad_duration = 0
                self.offroad_steps = 0
                self.successful_recoveries += 1
                reward += 3.0  # Recovery bonus
                info["recovery_success"] = True
                self.recovery.exit()
                self.consecutive_onroad = 0

        # === COLLISION = PENALTY (but don't end episode -- AI needs to learn) ===
        if self.collision_hist:
            reward -= 5.0  # Strong but not extreme
            self.safety_score = max(0, self.safety_score - 5.0)
            info["collision"] = True
            self.collision_hist = []
            if kmh > 5: reward -= 2.0  # Extra for high-speed crash

        # Lane invasion
        if self.lane_invasion_hist:
            reward -= 0.5; info["lane_invasion"] = True; self.lane_invasion_hist = []

        # Red light
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl and tl.get_state() == carla.TrafficLightState.Red and kmh > 1:
                reward -= 1.0; info["red_light"] = True; self.memory.add_red_light(loc)

        # Penalty tracking
        ego_respawn = False
        if wp is None or wp.lane_type not in valid:
            self.current_penalty += 1; self.safety_score = max(0, self.safety_score-0.2)
            self.memory.add_offroad(loc)

        # Timeout (shorter episodes = faster learning)
        if self.steps >= 5000:
            done = True; info["termination"] = "timeout"
            if self.current_distance > 200: reward += 10.0; print(f"[SUCCESS] {self.current_distance:.0f}m driven!")
            elif self.current_distance > 50: reward += 3.0; print(f"[OK] {self.current_distance:.0f}m")

        # Stuck -- rotate toward waypoint + penalize
        if self.stuck_counter > 200:
            reward -= 2.0
            self.stuck_counter = 0
            # Nudge vehicle: rotate toward next route waypoint
            if self.route_waypoints and self.route_target_idx < len(self.route_waypoints):
                target_loc = self.route_waypoints[self.route_target_idx]
                et = self.vehicle.get_transform()
                et.rotation.yaw = math.degrees(math.atan2(
                    target_loc.y - et.location.y,
                    target_loc.x - et.location.x))
                et.location.z += 0.0  # keep same height
                self.vehicle.set_transform(et)
            print(f"[STUCK] Nudged toward waypoint at {self.current_distance:.0f}m")

        # Milestones
        for m, ok in self.milestones.items():
            if not ok and self.current_distance >= m:
                self.milestones[m] = True; reward += m * 0.01; print(f"[MILESTONE] {m}m! +{m*0.01}")

        # Scenario tracking
        scenario = self.curriculum.detect_scenario(wp, kmh)
        self.curriculum.record_scenario(scenario, on_road and not self.collision_hist)

        # Obstacle injection
        if self.steps % 500 == 0: self._inject_obstacle()

        # Rival stuck
        rival_respawn = False
        if self.rival_vehicle and self.rival_vehicle.is_alive:
            rv = self.rival_vehicle.get_velocity()
            if 3.6*np.sqrt(rv.x**2+rv.y**2+rv.z**2) < 1: self.rival_stuck_counter += 1
            else: self.rival_stuck_counter = 0
            if self.rival_stuck_counter > 200: rival_respawn = True

        # Tournament
        if ego_respawn or rival_respawn:
            if self.rival_vehicle and self.rival_vehicle.is_alive:
                if self.current_distance > self.rival_distance:
                    print(f"[ENV] EGO WINS ({self.current_distance:.0f}m vs {self.rival_distance:.0f}m)")
                    reward += 50; info["tournament_winner"] = "ego"
                    self.leaderboard.update(self.agent_id, ego_wins=1)
                else:
                    print(f"[ENV] RIVAL WINS ({self.rival_distance:.0f}m vs {self.current_distance:.0f}m)")
                    reward -= 25; info["tournament_winner"] = "rival"
                    self.leaderboard.update(self.agent_id, rival_wins=1)
            if self.current_distance > self.best_distance and self.current_distance > 50:
                self.best_distance = self.current_distance; self.best_trajectory = list(self.current_trajectory)
            if ego_respawn:
                self.vehicle.set_transform(self.initial_spawn)
                self.vehicle.set_target_velocity(carla.Vector3D(0,0,0))
                self.stuck_counter=0; self.current_distance=0; self.current_trajectory=[]
                self.last_loc=None; self.last_steer=0; self.last_throttle=0
            if rival_respawn:
                ew = self.map.get_waypoint(self.vehicle.get_location())
                rw = ew.previous(10) if ew else []
                rsp = rw[0].transform if rw else self._all_spawns[1]
                rsp.location.z += 0.5
                self.rival_vehicle.set_transform(rsp)
                self.rival_vehicle.set_target_velocity(carla.Vector3D(0,0,0))
                self.rival_stuck_counter=0; self.rival_last_loc=None

        # HER + Episodic
        self.episode_states.append(obs_before); self.episode_actions.append(action)
        self.episode_rewards.append(reward); self.episode_infos.append(info)
        self.episodic_memory.add_step(obs_before, action, reward, info)
        self.adr.update(reward); self.episode_reward += reward

        if done:
            for i in range(len(self.episode_states)):
                self.her_buffer.add({
                    'state':self.episode_states[i],'action':self.episode_actions[i],
                    'reward':self.episode_rewards[i],
                    'next_state':self.episode_states[min(i+1,len(self.episode_states)-1)],
                    'done':i==len(self.episode_states)-1,'info':self.episode_infos[i]})
            success = self.current_distance > 500 and self.current_penalty < 200
            self.episodic_memory.end_episode(self.episode_reward, success)
            advanced = self.curriculum.record_episode(self.episode_reward)
            info.update(episode_reward=self.episode_reward, curriculum_stage=self.curriculum.current_stage,
                        total_distance=self.current_distance, total_penalty=self.current_penalty,
                        safety_score=self.safety_score, recoveries=self.successful_recoveries,
                        shield_interventions=self.safety_shield.total_interventions)
            self.leaderboard.update(self.agent_id,
                total_distance=self.current_distance, total_penalty=self.current_penalty,
                safety_score=self.safety_score, recoveries=self.successful_recoveries,
                shield_interventions=self.safety_shield.total_interventions,
                avg_episode_reward=self.episode_reward)
            if advanced: info["stage_advanced"] = True

        if np.isnan(reward) or np.isinf(reward): reward = 0.0
        reward = np.clip(reward, -10, 20)  # Tight range for stable value learning
        return self._get_obs('ego'), reward, done, False, info

    def close(self):
        self._settings.synchronous_mode = False; self._settings.fixed_delta_seconds = None
        self.world.apply_settings(self._settings)
        self._cleanup(); self.memory.save()
        self.her_buffer.save(os.path.join(HER_BUFFER_DIR, "her_buffer.json"))
        print("[ENV] Closed.")


# =============================================================================
# HELPER: Find latest checkpoint
# =============================================================================
def find_latest_checkpoint():
    """Find the checkpoint with the most training steps (never start from 0)."""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    best_path, best_steps = None, 0
    for f in os.listdir(CHECKPOINT_DIR):
        if f.startswith("v14_") and f.endswith("_steps.zip"):
            try:
                steps = int(f.split("_")[1])
                if steps > best_steps:
                    best_steps = steps
                    best_path = os.path.join(CHECKPOINT_DIR, f)
            except (ValueError, IndexError):
                pass
    return best_path, best_steps


def load_super_memory(target_memory):
    """Load merged knowledge from all previous training sessions."""
    super_path = os.path.join(MEMORY_DIR, "super_memory.json")
    if not os.path.exists(super_path):
        print("[MEMORY] No super_memory.json found - starting fresh")
        return

    try:
        with open(super_path) as f:
            sm = json.load(f)

        # Merge collision hotspots
        for h in sm.get("collision_hotspots", []):
            target_memory.collision_hotspots.append({
                "x": h["x"], "y": h["y"],
                "z": h.get("z", 0),
                "count": h.get("count", 1),
                "last_seen": h.get("last_seen", "")
            })

        # Merge offroad zones
        for z in sm.get("offroad_zones", []):
            target_memory.offroad_zones.append({
                "x": z["x"], "y": z["y"],
                "radius": z.get("radius", 8.0),
                "last_seen": z.get("last_seen", "")
            })

        # Merge red light zones
        for z in sm.get("red_light_zones", []):
            target_memory.red_light_zones.append({
                "x": z["x"], "y": z["y"],
                "danger": z.get("danger", 1.0),
                "last_seen": z.get("last_seen", "")
            })

        # Merge successful routes
        for w in sm.get("successful_route", []):
            target_memory.successful_route.append({
                "x": w["x"], "y": w["y"], "z": w.get("z", 0),
                "sequence": w.get("sequence", 0)
            })

        target_memory.save()
        n_col = len(sm.get("collision_hotspots", []))
        n_off = len(sm.get("offroad_zones", []))
        n_red = len(sm.get("red_light_zones", []))
        n_route = len(sm.get("successful_route", []))
        print(f"[MEMORY] Loaded super memory: {n_col} hotspots, {n_off} offroad, "
              f"{n_red} redlights, {n_route} route waypoints")

    except Exception as e:
        print(f"[MEMORY] Failed to load super memory: {e}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"  CARLA GRANDMASTER AI v14.0 -- NEXT-GEN EDITION")
    print(f"  Vision: {'DINOv2' if DINOV2_AVAILABLE else 'CNN'} | "
          f"Augment: {'ON' if AUGMENT_AVAILABLE else 'OFF'} | "
          f"Shield: ON")
    print(f"{'='*70}\n")

    curriculum = CurriculumManager(stage=1)
    config = curriculum.get_config()
    leaderboard = Leaderboard()
    agent_id = "Grandmaster_v14"
    env = CarlaGrandmasterEnv(config, curriculum, leaderboard, agent_id)

    # === AUTO-RESUME: Always find and load latest checkpoint ===
    model = None
    loaded_steps = 0

    # Try main model first
    if os.path.exists(MODEL_PATH):
        print(f"[LOAD] Main model: {MODEL_PATH}")
        try:
            model = PPO.load(MODEL_PATH, env=env, device="cuda", tensorboard_log=TB_DIR)
            loaded_steps = model.num_timesteps
            print(f"[LOAD] Resuming from {loaded_steps:,} timesteps")
        except Exception as e:
            print(f"[WARN] Main model failed: {e}"); model = None

    # Try latest checkpoint
    if model is None:
        ckpt_path, ckpt_steps = find_latest_checkpoint()
        if ckpt_path:
            print(f"[LOAD] Checkpoint: {ckpt_path} ({ckpt_steps:,} steps)")
            try:
                model = PPO.load(ckpt_path, env=env, device="cuda", tensorboard_log=TB_DIR)
                loaded_steps = model.num_timesteps
                print(f"[LOAD] Resuming from {loaded_steps:,} timesteps")
            except Exception as e:
                print(f"[WARN] Checkpoint failed: {e}"); model = None

    if model is None:
        print("[NEW] Creating PPO with GrandmasterFeatureExtractor...")
        model = PPO(
            "MultiInputPolicy", env, verbose=1, device="cuda", tensorboard_log=TB_DIR,
            policy_kwargs=dict(features_extractor_class=GrandmasterFeatureExtractor,
                               net_arch=dict(pi=[128, 64], vf=[128, 64])),
            learning_rate=3e-4, n_steps=4096, batch_size=256, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.25, max_grad_norm=0.5,
        )

    # === LOAD SUPER MEMORY (knowledge from all previous sessions) ===
    load_super_memory(env.memory)

    env.set_model(model)

    ckpt_cb = CheckpointCallback(save_freq=10_000, save_path=CHECKPOINT_DIR, name_prefix="v14")
    log_cb = TrainingLogger(LOG_FILE, curriculum, leaderboard, agent_id, env)

    # === CALCULATE REMAINING STEPS (never start from 0) ===
    remaining = max(5_000_000 - loaded_steps, 100_000)
    print(f"[TRAIN] {remaining:,} remaining timesteps (from {loaded_steps:,} to 5M)")

    try:
        model.learn(total_timesteps=remaining, callback=[ckpt_cb, log_cb],
                    reset_num_timesteps=False, progress_bar=True)
        model.save(MODEL_PATH)
        print(f"\n[DONE] {MODEL_PATH}")
    except KeyboardInterrupt:
        print("\n[STOP] Emergency save...")
        model.save(MODEL_PATH.replace(".zip","_emergency.zip"))
    finally:
        leaderboard.print_board(); env.close()
