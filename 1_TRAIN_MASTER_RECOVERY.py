"""
==============================================================================
  CARLA AI v11.0 - RECOVERY EDITION

  FITUR BARU: "Recovery Protocol"

  Masalah yang diselesaikan:
  - AI saat off-road terus maju (tidak mundur/berbalik)
  - Tidak belajar cari jalan kembali
  - Hanya teleport saat stuck

  Solusi:
  ✅ RECOVERY STATE - Otomatis reverse saat off-road
  ✅ ROAD FINDER - Cari arah ke jalan terdekat
  ✅ RECOVERY REWARD - Bonus besar saat berhasil kembali
  ✅ REVERSE ACTION - Belajar bahwa mundur itu valid
==============================================================================
"""

import os
import sys
import time
import json
import random
import numpy as np
import cv2
import carla
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import torch as th
import threading

BASE_DIR        = r"D:\AI\traffic_vision\master_suite"
MODEL_PATH      = os.path.join(BASE_DIR, "carla_grandmaster_v11.zip")
TB_DIR          = os.path.join(BASE_DIR, "tb_logs")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_FILE        = os.path.join(BASE_DIR, "training_carla_v11_log.txt")
LEADERBOARD_DIR = os.path.join(BASE_DIR, "leaderboard")
MEMORY_DIR      = os.path.join(BASE_DIR, "agent_memories")

for path in [BASE_DIR, TB_DIR, CHECKPOINT_DIR, LEADERBOARD_DIR, MEMORY_DIR]:
    os.makedirs(path, exist_ok=True)

IMG_WIDTH, IMG_HEIGHT = 160, 120
N_LIDAR_SECTORS = 8
LIDAR_MAX_RANGE = 50.0
STATE_DIM = 18  # Includes recovery state + rival_relative

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
    recoveries: int = 0  # NEW: track successful recoveries
    waypoints_visited: int = 0
    best_single_distance: float = 0.0
    safety_score: float = 100.0
    last_updated: str = ""

    @property
    def score(self) -> float:
        bonus = self.ego_wins * 500 + self.recoveries * 200  # Bonus for recovery
        return max(0, self.total_distance - self.total_penalty * 0.1
                  - (100 - self.safety_score) * 10 + bonus)


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
        print("  CARLA AI LEADERBOARD v11".center(80))
        print("="*80)
        print(f"{'Rank':<6} {'Agent':<20} {'Score':<10} {'Distance':<12} "
              f"{'Wins':<8} {'Recoveries':<12}")
        print("-"*80)
        for i, (aid, rec) in enumerate(sorted_records[:top_n]):
            medals = ["1st", "2nd", "3rd"]
            medal = medals[i] if i < 3 else f"{i+1}."
            print(f"{medal:<6} {aid[:20]:<20} {rec.score:<10.1f} "
                  f"{rec.total_distance:<12.1f} {rec.ego_wins:<8} "
                  f"{rec.recoveries:<12}")
        print("="*80 + "\n")


# =============================================================================
# RECOVERY MODULE - Fitur Utama v11
# =============================================================================
class RecoveryModule:
    """
    Modul RECOVERY - Belajar AI untuk:
    1. Mendeteksi saat off-road
    2. Mencari arah balik ke jalan
    3. Melakukan REVERSE atau U-TURN
    4. Memberi reward saat berhasil recovery
    """

    def __init__(self):
        self.is_recovering = False
        self.recovery_start_time = None
        self.last_road_position = None
        self.recovery_timeout = 300  # max steps for recovery attempt

    def enter_recovery_mode(self, current_position, road_position):
        """Mulai mode recovery saat off-road terdeteksi"""
        self.is_recovering = True
        self.recovery_start_time = time.time()
        self.last_road_position = road_position
        return True

    def exit_recovery_mode(self):
        """Keluar dari mode recovery saat kembali ke jalan"""
        self.is_recovering = False
        self.recovery_start_time = None

    def get_recovery_action(self, current_location, current_yaw, map_ways):
        """
        Hitung aksi recovery: REVERSE atau U-TURN ke arah jalan

        Returns:
            (steer, throttle, brake) - Aksi untuk recovery
        """
        if not map_ways:
            return 0.0, -0.5, 0.0  # Reverse full if no road data

        # Cari waypoint jalan terdekat
        closest_wp = None
        min_dist = float('inf')

        for wp in map_ways:
            dist = current_location.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                closest_wp = wp

        if closest_wp is None:
            return 0.0, -0.5, 0.0  # Reverse full

        # Hitung arah ke waypoint
        to_road = closest_wp.transform.location - current_location
        angle_to_road = np.arctan2(to_road.y, to_road.x)

        # Normalisasi ke -1 s/d 1
        current_heading = np.radians(current_yaw)
        angle_diff = angle_to_road - current_heading

        # Normalize ke -pi s/d pi
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Strategi Recovery:
        if abs(angle_diff) > np.pi / 2:  # 90 derajat
            # U-Turn lebih cepat
            steer = 1.0 if angle_diff > 0 else -1.0
            throttle = 0.3
            brake = 0.0
        else:
            # Sedikit belok + reverse
            steer = np.clip(angle_diff / np.pi, -1.0, 1.0)
            throttle = -0.3  # REVERSE!
            brake = 0.0

        return steer, throttle, brake


# =============================================================================
# MEMORY SYSTEM
# =============================================================================
class RecallMemory:
    def __init__(self, agent_id: str = "ego"):
        self.agent_id = agent_id
        self.file_path = os.path.join(MEMORY_DIR, f"{agent_id}_memory.json")
        self.collision_hotspots = []
        self.offroad_zones = []
        self.successful_route = []
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
            "collision_hotspots": self.collision_hotspots[-500:],
            "offroad_zones": self.offroad_zones[-500:],
            "successful_route": self.successful_route[-2500:],
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f)

    def add_offroad(self, location: carla.Location):
        self.offroad_zones.append({
            "x": location.x, "y": location.y,
            "radius": 8.0,
            "timestamp": datetime.now().isoformat()
        })
        self.save()

    def get_danger_score(self, location: carla.Location) -> float:
        x, y = location.x, location.y
        danger = 0.0
        for hotspot in self.collision_hotspots:
            dist = np.sqrt((hotspot["x"] - x)**2 + (hotspot["y"] - y)**2)
            if dist < 10.0:
                danger += hotspot.get("count", 1) * (1.0 - dist/10.0) * 0.3
        return min(danger, 1.0)


# =============================================================================
# CURRICULUM
# =============================================================================
class CurriculumManager:
    STAGES = {
        1: {"name": "Basics", "map": "Town01", "traffic": 0, "pedestrians": 0,
            "weather": "Clear", "advance_threshold": 500.0},
        2: {"name": "Traffic", "map": "Town03", "traffic": 10, "pedestrians": 5,
            "weather": "Cloudy", "advance_threshold": 1200.0},
        3: {"name": "Expert", "map": "Town05", "traffic": 30, "pedestrians": 15,
            "weather": "Rainy", "advance_threshold": 2500.0},
    }
    WINDOW = 50

    def __init__(self, current_stage: int = 1):
        self.current_stage = current_stage
        self._ep_rewards = deque(maxlen=self.WINDOW)

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
            print(f"\n[CURRICULUM] ADVANCED -> Stage {self.current_stage}!")
            return True
        return False


# =============================================================================
# TRAINING LOGGER
# =============================================================================
class TrainingLogger(BaseCallback):
    def __init__(self, log_path: str, curriculum: CurriculumManager,
                 leaderboard: Leaderboard, agent_id: str = "Grandmaster_v11"):
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
            stage = self.curriculum.current_stage
            with open(self.log_path, "a") as f:
                f.write(f"timesteps: {ts}, reward: {rew:.2f}, stage: {stage}\n")
        except Exception:
            pass


# =============================================================================
# MAIN ENVIRONMENT v11 - WITH RECOVERY
# =============================================================================
class CarlaRecoveryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, stage_config: dict, curriculum: CurriculumManager,
                 leaderboard: Leaderboard = None, agent_id: str = "Grandmaster_v11"):
        super().__init__()
        self.config = stage_config
        self.curriculum = curriculum
        self.leaderboard = leaderboard or Leaderboard()
        self.agent_id = agent_id

        # Recovery Module - NEW!
        self.recovery = RecoveryModule()
        self.was_offroad = False
        self.offroad_duration = 0
        self.consecutive_onroad = 0  # For hysteresis in recovery detection
        self.last_safe_location = None
        self.successful_recoveries = 0

        # Memory
        self.memory = RecallMemory(agent_id)

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
        self._settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(self._settings)

        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)

        # Spaces
        self.observation_space = spaces.Dict({
            "camera": spaces.Box(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), np.uint8),
            "lidar": spaces.Box(0.0, LIDAR_MAX_RANGE, (N_LIDAR_SECTORS,), np.float32),
            "state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),  # throttle can be negative for REVERSE!
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

        # Episode state
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
        self.successful_recoveries = 0

    def set_model(self, model):
        self.model = model

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

    def _randomize_weather(self):
        w = carla.WeatherParameters(
            cloudiness=random.uniform(0, 30),
            precipitation=random.uniform(0, 20),
            precipitation_deposits=random.uniform(0, 20),
            wind_intensity=random.uniform(0, 20),
            sun_azimuth_angle=random.uniform(0, 360),
            sun_altitude_angle=random.uniform(20, 70),
            fog_density=random.uniform(0, 5),
            wetness=random.uniform(0, 20),
        )
        self.world.set_weather(w)

    def _cb_camera(self, image):
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
                self.memory.add_offroad(event.other_actor.get_location())
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()
        self._randomize_weather()

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

        # Spawn Rival
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

        self._spawn_npc_vehicles(self.config["traffic"])
        self._spawn_npc_pedestrians(self.config["pedestrians"])

        self._attach_sensors(self.vehicle, is_rival=False)
        if self.rival_vehicle:
            self._attach_sensors(self.rival_vehicle, is_rival=True)

        for _ in range(20):
            self.world.tick()
            with self._img_lock:
                if self.last_img is not None:
                    if self.rival_vehicle and self.rival_last_img is None:
                        continue
                    break

        # Reset episode state
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

        self.current_penalty = 0.0
        self.current_distance = 0.0
        self.rival_distance = 0.0
        self.last_loc = None
        self.rival_last_loc = None
        self.current_trajectory = []
        self.waypoint_sequence = 0

        # Reset recovery state
        self.recovery.is_recovering = False
        self.was_offroad = False
        self.offroad_duration = 0

        print(f"[ENV] Reset complete. Safety: {self.safety_score:.1f}")

        return self._get_actor_obs('ego'), {}

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

        red_light = 0.0
        if vehicle.is_at_traffic_light():
            tl = vehicle.get_traffic_light()
            if tl and tl.get_state() == carla.TrafficLightState.Red:
                red_light = 1.0

        wp = self.map.get_waypoint(vehicle.get_location(), project_to_road=True)
        at_junction = float(wp.is_junction) if wp else 0.0

        # Check if on road
        on_road = False
        if wp:
            valid_lanes = (carla.LaneType.Driving, carla.LaneType.Bidirectional,
                          carla.LaneType.OnRamp, carla.LaneType.OffRamp)
            on_road = wp.lane_type in valid_lanes

        danger_memory = self.memory.get_danger_score(vehicle.get_location())

        # Recovery state (NEW)
        in_recovery = float(self.recovery.is_recovering)

        min_lidar = float(np.min(lidar if is_rival else self.last_lidar))

        # Rival relative info (for ego only, 0 for rival)
        rival_relative = 0.0
        if not is_rival and self.rival_vehicle and self.rival_vehicle.is_alive:
            ego_loc = vehicle.get_location()
            rival_loc = self.rival_vehicle.get_location()
            dist = ego_loc.distance(rival_loc)
            # Rival ahead = positive, behind = negative
            rival_relative = min((dist - 50.0) / 200.0, 1.0)

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
            float(on_road),
            in_recovery,
            float(self.offroad_duration / 300.0),
            float(self.steps / 10000),
            float(self.current_distance / 1000.0),
            rival_relative,  # CRITICAL: rival proximity awareness
        ], dtype=np.float32)

        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        return {"camera": cam, "lidar": lidar, "state": state}

    def step(self, action):
        self.steps += 1

        # Store state before action
        state_before = self._get_actor_obs('ego')
        on_road_before = state_before["state"][12] > 0.5

        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])

        # Get current info
        v = self.vehicle.get_velocity()
        kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

        loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(loc, project_to_road=False,
                                   lane_type=carla.LaneType.Any)

        valid_lanes = (carla.LaneType.Driving, carla.LaneType.Bidirectional,
                      carla.LaneType.OnRamp, carla.LaneType.OffRamp)
        on_road = wp is not None and wp.lane_type in valid_lanes

        # Initialize reward early (needed for punishment awareness)
        reward = 0.0
        done = False
        info = {}

        # ==================== PUNISHMENT AWARENESS (NEW) ====================
        # Let AI learn to recover by itself - no forced actions!
        # AI gets awareness through observation state and learns through rewards

        if not on_road:
            self.was_offroad = True
            self.offroad_duration += 1
            self.consecutive_onroad = 0  # Reset on-road counter

            # CRUSHING punishment for off-road - AI MUST learn to avoid
            reward -= 100.0  # Huge penalty every step off-road (increased from 50)

            # Exponentially increasing penalty the longer off-road
            reward -= self.offroad_duration * 10.0  # Increased from 5

            # Update recovery state for awareness (NOT for forcing actions)
            self.recovery.enter_recovery_mode(loc, wp)

            # MASSIVE penalty for moving forward while off-road (wrong direction!)
            if throttle > 0.3:
                reward -= 100.0  # Increased from 30

            # Good reward for reversing (correct behavior!)
            if throttle < -0.2:
                reward += 50.0  # Increased from 20

            info["offroad_duration"] = self.offroad_duration
            info["punishment_reason"] = "off_road"

            # Faster respawn - don't waste time on hopeless off-road
            if self.offroad_duration > 100:  # Reduced from 300
                print(f"[AWARENESS] Off-road timeout - Respawning after {self.offroad_duration} steps")
                self.was_offroad = False
                self.offroad_duration = 0
                self.consecutive_onroad = 0
                self.recovery.exit_recovery_mode()
                # Respawn at last good position, not initial spawn
                if self.last_safe_location:
                    self.vehicle.set_transform(self.last_safe_location)
                    self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                else:
                    self.vehicle.set_transform(self.initial_spawn)
                    self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))

        else:  # ON ROAD - bonus + recovery check
            reward += 50.0  # Large bonus for being on road

            if self.was_offroad:
                self.consecutive_onroad += 1

                # Require 10+ consecutive frames on-road before declaring recovery
                # (prevents edge flickering from triggering recovery)
                if self.consecutive_onroad >= 10 and self.offroad_duration > 5:
                    print(f"[AWARENESS] AI recovered from off-road! Steps: {self.offroad_duration}")
                    self.was_offroad = False
                    self.offroad_duration = 0
                    self.successful_recoveries += 1
                    reward += 1000.0  # HUGE bonus for recovering by itself!
                    info["recovery_success"] = True
                    self.recovery.exit_recovery_mode()
                    self.consecutive_onroad = 0
                elif self.consecutive_onroad >= 10 and self.offroad_duration <= 5:
                    # Was just edge flicker, not real off-road
                    self.was_offroad = False
                    self.offroad_duration = 0
                    self.consecutive_onroad = 0

            # Track safe location
            if not self.collision_hist:
                self.safe_loc = self.vehicle.get_transform()

        # ==================== COLLISION AWARENESS (NEW) ====================
        if self.collision_hist:
            # AI needs to learn to reverse after collision
            reward -= 100.0  # Big penalty for collision

            info["punishment_reason"] = "collision"

            # Reward reversing behavior after collision
            if throttle < -0.3:
                reward += 30.0  # Reward backing up from collision
                info["correct_action"] = "reversing_from_collision"

        # Track last safe location for smart respawn
        if on_road and not self.collision_hist:
            self.last_safe_location = self.vehicle.get_transform()

        # ==================== NORMAL DRIVING MODE ====================
        # (Recovery handled by hysteresis logic above)

        self.last_steer = steer
        self.last_throttle = throttle

        # Handle reverse gear - negative throttle means reverse!
        reverse = False
        if throttle < 0:
            reverse = True
            throttle = abs(throttle)  # Convert to positive for CARLA

        # Anti-stall (only for forward movement)
        if kmh < 2.0 and not reverse:
            throttle = 0.6
            brake = 0.0

        self.vehicle.apply_control(
            carla.VehicleControl(steer=steer, throttle=throttle, brake=brake, reverse=reverse)
        )

        # Rival AI
        if self.rival_vehicle and self.rival_vehicle.is_alive and self.model:
            rival_obs = self._get_actor_obs('rival')
            rival_action, _ = self.model.predict(rival_obs, deterministic=False)

            r_steer, r_throttle, r_brake = float(rival_action[0]), float(rival_action[1]), float(rival_action[2])

            r_v = self.rival_vehicle.get_velocity()
            r_kmh = 3.6 * np.sqrt(r_v.x**2 + r_v.y**2 + r_v.z**2)

            # Handle reverse for rival too
            r_reverse = False
            if r_throttle < 0:
                r_reverse = True
                r_throttle = abs(r_throttle)

            if r_kmh < 2.0 and not r_reverse:
                r_throttle = 0.6

            self.rival_last_steer = r_steer
            self.rival_last_throttle = r_throttle

            self.rival_vehicle.apply_control(
                carla.VehicleControl(steer=r_steer, throttle=r_throttle, brake=r_brake, reverse=r_reverse)
            )

        self.world.tick()

        # Spectator - Smooth follow camera (chase view)
        spec = self.world.get_spectator()
        t = self.vehicle.get_transform()

        # Chase camera position - behind and above the car
        # Calculate offset based on car's rotation
        import math
        yaw_rad = math.radians(t.rotation.yaw)
        offset_x = -12 * math.cos(yaw_rad)  # 12m behind
        offset_y = -12 * math.sin(yaw_rad)
        cam_location = carla.Location(x=t.location.x + offset_x,
                                       y=t.location.y + offset_y,
                                       z=t.location.z + 6)  # 6m above

        spec.set_transform(carla.Transform(cam_location, carla.Rotation(pitch=-20, yaw=t.rotation.yaw)))

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

        # ==================== REWARD SYSTEM v11 ====================
        # (reward, done, info already initialized above)

        # 1. Exploration bonus (reduced - focus on staying on road first!)
        if wp and wp.id not in self.visited_waypoints:
            reward += 5.0  # Reduced from 25
            self.visited_waypoints.add(wp.id)

        # 2. Green pixel bonus (increased - reward for seeing road!)
        with self._img_lock:
            if self.last_img is not None:
                green_pixels = np.sum((self.last_img[:, :, 0] == 0) &
                                     (self.last_img[:, :, 1] == 255) &
                                     (self.last_img[:, :, 2] == 0))
                green_ratio = green_pixels / (IMG_WIDTH * IMG_HEIGHT)
                if green_ratio > 0.2:
                    reward += green_ratio * 20.0  # Increased from 2.0

        # 3. Speed reward (only when safely on road)
        speed_limit_raw = self.vehicle.get_speed_limit()
        if speed_limit_raw is None or speed_limit_raw <= 0:
            speed_limit_raw = 50.0
        speed_ratio = kmh / max(speed_limit_raw, 1.0)
        if on_road and not self.collision_hist:
            reward += np.clip(speed_ratio, 0.0, 1.0) * 10.0  # Increased from 8.0

        # 4. Anti-stuck
        if kmh < 1.0:
            self.stuck_counter += 1
            reward -= self.stuck_counter * 0.1
        else:
            self.stuck_counter = 0

        # 5. Steering smoothness
        reward -= abs(steer) * 0.3

        # 6. Lane invasion
        if self.lane_invasion_hist:
            reward -= 100.0
            self.lane_invasion_hist = []

        # 7. Red light
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl and tl.get_state() == carla.TrafficLightState.Red and kmh > 1.0:
                reward -= 300.0

        # ==================== PENALTY (No-Reset) ====================
        ego_needs_respawn = False

        # Off-road penalty (REDUCED - let recovery handle it)
        if not on_road and not self.recovery.is_recovering:
            # Small penalty per step, but recovery module handles the fix
            self.current_penalty += 20.0
            reward -= 20.0
            self.memory.add_offroad(loc)
            if self.steps % 100 == 0:
                print(f"[PENALTY] Off-road! Total: {self.current_penalty:.1f}")
            info["termination"] = "off_road"

        # Collision
        if self.collision_hist:
            self.current_penalty += 100.0
            reward -= 100.0
            info["termination"] = "collision"
            self.collision_hist = []

        # Episode timeout
        if self.steps >= 10000:
            done = True
            info["termination"] = "timeout"
            if self.current_penalty < 200:
                reward += 1000.0  # Bonus for clean finish

        # Stuck handling
        if self.stuck_counter > 500:
            ego_needs_respawn = True
            self.stuck_counter = 0
            print("[ENV] Ego stuck - Respawning...")

        # Tournament check
        if ego_needs_respawn:
            if self.rival_vehicle and self.rival_vehicle.is_alive:
                if self.current_distance > self.rival_distance:
                    print(f"[ENV] EGO WINS! ({self.current_distance:.1f}m vs {self.rival_distance:.1f}m)")
                    reward += 500.0
                    info["tournament_winner"] = "ego"
                    self.leaderboard.update(self.agent_id, ego_wins=1)
                else:
                    print(f"[ENV] RIVAL WINS! ({self.rival_distance:.1f}m vs {self.current_distance:.1f}m)")
                    reward -= 200.0
                    info["tournament_winner"] = "rival"
                    self.leaderboard.update(self.agent_id, rival_wins=1)

            # Update best
            if self.current_distance > self.best_distance and self.current_distance > 50.0:
                self.best_distance = self.current_distance
                self.best_trajectory = list(self.current_trajectory)
                print(f"[ENV] NEW RECORD! {self.best_distance:.1f}m")

            # Respawn
            self.vehicle.set_transform(self.initial_spawn)
            self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            self.stuck_counter = 0
            self.current_distance = 0.0
            self.current_trajectory = []
            self.last_loc = None
            self.last_steer = 0.0
            self.last_throttle = 0.0

        self.episode_reward += reward

        if done:
            self.curriculum.record_episode(self.episode_reward)
            info["episode_reward"] = self.episode_reward
            info["total_distance"] = self.current_distance
            info["total_penalty"] = self.current_penalty
            info["safety_score"] = self.safety_score
            info["recoveries"] = self.successful_recoveries

            self.leaderboard.update(
                self.agent_id,
                total_distance=self.current_distance,
                total_penalty=self.current_penalty,
                recoveries=self.successful_recoveries
            )

        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        reward = np.clip(reward, -10000.0, 10000.0)

        return self._get_actor_obs('ego'), reward, done, False, info

    def close(self):
        self._settings.synchronous_mode = False
        self._settings.fixed_delta_seconds = None
        self.world.apply_settings(self._settings)
        self._cleanup()
        self.memory.save()
        print("[ENV] Closed.")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  CARLA AI v11.0 - RECOVERY EDITION")
    print("  Fitur: Recovery Protocol | Reverse Action | Road Finder")
    print("="*70 + "\n")

    curriculum = CurriculumManager(current_stage=1)
    config = curriculum.get_config()
    leaderboard = Leaderboard()
    agent_id = "Grandmaster_v11"
    env = CarlaRecoveryEnv(
        stage_config=config,
        curriculum=curriculum,
        leaderboard=leaderboard,
        agent_id=agent_id
    )

    # Try to load existing
    model = None
    if os.path.exists(MODEL_PATH):
        print(f"[LOAD] Attempting to load: {MODEL_PATH}")
        try:
            model = PPO.load(MODEL_PATH, env=env, device="cuda", tensorboard_log=TB_DIR)
            print("[LOAD] Success!")
        except Exception as e:
            print(f"[WARN] Load failed: {e}")
            model = None

    if model is None:
        print("[NEW] Creating fresh PPO brain...")
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
            ent_coef=0.05,  # High exploration for learning recovery
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

    env.set_model(model)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000, save_path=CHECKPOINT_DIR, name_prefix="grandmaster_v11"
    )
    logger_cb = TrainingLogger(LOG_FILE, curriculum, leaderboard, agent_id)

    try:
        print("[TRAIN] Starting 5M timestep run...")
        model.learn(
            total_timesteps=5_000_000,
            callback=[checkpoint_cb, logger_cb],
            progress_bar=True,
        )
        model.save(MODEL_PATH)
        print(f"\n[DONE] Saved -> {MODEL_PATH}")

    except KeyboardInterrupt:
        print("\n[STOP] Emergency save...")
        model.save(MODEL_PATH.replace(".zip", "_emergency.zip"))

    finally:
        leaderboard.print_leaderboard()
        env.close()
