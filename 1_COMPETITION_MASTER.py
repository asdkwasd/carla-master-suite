"""
==============================================================================
  CARLA COMPETITION MASTER v1.0 — Multi-Agent AI Racing League

  FITUR UTAMA:
  1. NO-RESET PENALTY - Agent tidak reset saat kena punish, tetap lanjut
  2. DECISION MAKING - Agent "mikir 2x" sebelum keluar aspal/nabrak
  3. RECALL MEMORY - Mengingat semua lokasi kecelakaan & menghindarinya
  4. COMPETITIVE MODE - Multiple agents bersaing di leaderboard

  Algorithm: PPO dengan Modified Reward & Experience Replay
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
from typing import List, Dict, Tuple, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch as th
from torch import nn

# =============================================================================
# 1. GLOBAL SETTINGS & PATHS
# =============================================================================
BASE_DIR       = r"D:\AI\traffic_vision\master_suite"
LEADERBOARD_DIR = os.path.join(BASE_DIR, "leaderboard")
MEMORY_DIR     = os.path.join(BASE_DIR, "agent_memories")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_FILE       = os.path.join(BASE_DIR, "competition_log.txt")

for path in [BASE_DIR, LEADERBOARD_DIR, MEMORY_DIR, CHECKPOINT_DIR]:
    os.makedirs(path, exist_ok=True)

IMG_WIDTH, IMG_HEIGHT = 160, 120
N_LIDAR_SECTORS = 8
LIDAR_MAX_RANGE = 50.0
STATE_DIM = 15  # Extended for memory features

# =============================================================================
# 2. LEADERBOARD & AGENT REGISTRY
# =============================================================================
@dataclass
class AgentRecord:
    """Record performa agent untuk leaderboard"""
    agent_id: str
    total_distance: float = 0.0
    total_penalty: float = 0.0
    collisions: int = 0
    off_roads: int = 0
    red_light_violations: int = 0
    waypoints_visited: int = 0
    episodes_completed: int = 0
    best_single_distance: float = 0.0
    safety_score: float = 100.0  # Starts at 100, decreases with violations
    last_updated: str = ""

    @property
    def score(self) -> float:
        """Composite score: distance - penalties - safety_reduction"""
        penalty_factor = self.total_penalty * 0.1
        safety_reduction = (100 - self.safety_score) * 10
        return max(0, self.total_distance - penalty_factor - safety_reduction)


class Leaderboard:
    """Manage kompetisi antar agents"""

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
            except Exception as e:
                print(f"[LEADERBOARD] Failed to load: {e}")

    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.records.items()}, f, indent=2)

    def update(self, agent_id: str, **kwargs):
        if agent_id not in self.records:
            self.records[agent_id] = AgentRecord(agent_id=agent_id)

        record = self.records[agent_id]
        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)

        record.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def get_ranking(self) -> List[Tuple[int, str, AgentRecord]]:
        """Return sorted leaderboard: (rank, agent_id, record)"""
        sorted_records = sorted(
            self.records.items(),
            key=lambda x: x[1].score,
            reverse=True
        )
        return [(i+1, aid, rec) for i, (aid, rec) in enumerate(sorted_records)]

    def print_leaderboard(self, top_n=10):
        ranking = self.get_ranking()
        print("\n" + "="*80)
        print("  🏆 CARLA AI COMPETITION LEADERBOARD 🏆".center(80))
        print("="*80)
        print(f"{'Rank':<6} {'Agent ID':<20} {'Score':<10} {'Distance':<12} {'Safety':<10}")
        print("-"*80)
        for rank, aid, rec in ranking[:top_n]:
            print(f"{rank:<6} {aid:<20} {rec.score:<10.1f} {rec.total_distance:<12.1f} {rec.safety_score:<10.1f}")
        print("="*80 + "\n")

    def get_best_agent(self) -> Optional[str]:
        if not self.records:
            return None
        return self.get_ranking()[0][1]


# =============================================================================
# 3. RECALL MEMORY SYSTEM
# =============================================================================
class RecallMemory:
    """
    Sistem memori jangka panjang untuk mengingat:
    - Lokasi kecelakaan (collision hotspots)
    - Area off-road berbahaya
    - Pelanggaran traffic light
    - Route terbaik yang pernah dilalui
    """

    def __init__(self, agent_id: str, memory_size: int = 1000):
        self.agent_id = agent_id
        self.file_path = os.path.join(MEMORY_DIR, f"{agent_id}_memory.json")
        self.memory_size = memory_size

        # Memory banks
        self.collision_hotspots: List[Dict] = []      # {x, y, z, count, last_seen}
        self.offroad_zones: List[Dict] = []           # {x, y, radius, severity}
        self.red_light_zones: List[Dict] = []         # {x, y, danger_level}
        self.successful_route: List[Dict] = []        # {x, y, z, sequence}

        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self.collision_hotspots = data.get("collision_hotspots", [])
                    self.offroad_zones = data.get("offroad_zones", [])
                    self.red_light_zones = data.get("red_light_zones", [])
                    self.successful_route = data.get("successful_route", [])
            except Exception as e:
                print(f"[MEMORY] Failed to load memory for {self.agent_id}: {e}")

    def save(self):
        data = {
            "collision_hotspots": self.collision_hotspots[-self.memory_size:],
            "offroad_zones": self.offroad_zones[-self.memory_size:],
            "red_light_zones": self.red_light_zones[-self.memory_size:],
            "successful_route": self.successful_route[-self.memory_size*5:],
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f)

    def add_collision(self, location: carla.Location):
        """Catat lokasi tabrakan"""
        x, y, z = location.x, location.y, location.z

        # Check if near existing hotspot
        for hotspot in self.collision_hotspots:
            dist = np.sqrt((hotspot["x"] - x)**2 + (hotspot["y"] - y)**2)
            if dist < 5.0:  # Within 5 meters
                hotspot["count"] += 1
                hotspot["last_seen"] = datetime.now().isoformat()
                self.save()
                return

        # New hotspot
        self.collision_hotspots.append({
            "x": x, "y": y, "z": z,
            "count": 1,
            "last_seen": datetime.now().isoformat()
        })
        self.save()

    def add_offroad(self, location: carla.Location, severity: float = 1.0):
        """Catat area off-road"""
        self.offroad_zones.append({
            "x": location.x,
            "y": location.y,
            "radius": 10.0,
            "severity": severity,
            "last_seen": datetime.now().isoformat()
        })
        self.save()

    def add_red_light_violation(self, location: carla.Location):
        """Catat pelanggaran traffic light"""
        self.red_light_zones.append({
            "x": location.x,
            "y": location.y,
            "danger_level": 1.0,
            "last_seen": datetime.now().isoformat()
        })
        self.save()

    def add_waypoint(self, location: carla.Location, sequence: int):
        """Catat waypoint yang berhasil dilalui"""
        self.successful_route.append({
            "x": location.x,
            "y": location.y,
            "z": location.z,
            "sequence": sequence
        })

    def get_danger_score(self, location: carla.Location) -> float:
        """
        Hitung skor bahaya berdasarkan memori (0 = aman, 1 = sangat berbahaya)
        Dipakai untuk decision making
        """
        x, y = location.x, location.y
        danger = 0.0

        # Check collision hotspots
        for hotspot in self.collision_hotspots:
            dist = np.sqrt((hotspot["x"] - x)**2 + (hotspot["y"] - y)**2)
            if dist < 10.0:
                danger += hotspot["count"] * (1.0 - dist/10.0) * 0.5

        # Check offroad zones
        for zone in self.offroad_zones:
            dist = np.sqrt((zone["x"] - x)**2 + (zone["y"] - y)**2)
            if dist < zone["radius"]:
                danger += zone["severity"] * (1.0 - dist/zone["radius"]) * 0.3

        # Check red light zones
        for zone in self.red_light_zones:
            dist = np.sqrt((zone["x"] - x)**2 + (zone["y"] - y)**2)
            if dist < 15.0:
                danger += 0.2

        return min(danger, 1.0)

    def get_route_guidance(self, location: carla.Location) -> Optional[carla.Location]:
        """Dapatkan lokasi berikutnya dari route terbaik"""
        if not self.successful_route:
            return None

        # Find closest waypoint in memory
        closest_seq = -1
        min_dist = float('inf')

        for wp in self.successful_route:
            dist = np.sqrt((wp["x"] - location.x)**2 + (wp["y"] - location.y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_seq = wp["sequence"]

        if closest_seq >= 0 and min_dist < 20.0:
            # Find next waypoint
            for wp in self.successful_route:
                if wp["sequence"] == closest_seq + 1:
                    return carla.Location(wp["x"], wp["y"], wp["z"])

        return None


# =============================================================================
# 4. DECISION MAKING MODULE
# =============================================================================
class DecisionMaker:
    """
    Module yang memaksa agent untuk "mikir 2x" sebelum mengambil keputusan berbahaya.
    Menambah delay consideration saat agent akan keluar dari aspal atau menabrak.
    """

    def __init__(self):
        self.decision_history = deque(maxlen=100)
        self.hesitation_count = 0

    def should_hesitate(self,
                        current_speed: float,
                        danger_ahead: float,
                        on_road: bool,
                        nearby_hazard: bool) -> Tuple[bool, str]:
        """
        Return (should_hesitate, reason)

        Agent akan "ragu-ragu" (mengurangi kecepatan secara signifikan)
        saat situasi berbahaya terdeteksi.
        """
        reasons = []

        # 1. High danger area from memory
        if danger_ahead > 0.5:
            reasons.append(f"High danger zone detected (score: {danger_ahead:.2f})")

        # 2. About to leave road
        if not on_road and current_speed > 10:
            reasons.append("About to leave road - slowing down!")

        # 3. Nearby hazard (obstacle, vehicle, pedestrian)
        if nearby_hazard and current_speed > 15:
            reasons.append("Hazard nearby - caution needed!")

        # 4. High speed in dangerous area
        if current_speed > 30 and danger_ahead > 0.3:
            reasons.append("Too fast for current conditions")

        should = len(reasons) > 0
        if should:
            self.hesitation_count += 1
            reason_str = " | ".join(reasons)
            self.decision_history.append({"hesitate": True, "reason": reason_str})
            return True, reason_str

        self.decision_history.append({"hesitate": False, "reason": "Clear"})
        return False, "Clear"


# =============================================================================
# 5. COMPETITION ENVIRONMENT
# =============================================================================
class CarlaCompetitionEnv(gym.Env):
    """
    Environment dengan NO-RESET PENALTY SYSTEM.

    Perbedaan utama dengan environment biasa:
    - Episode tidak berakhir saat collision/off-road
    - Penalty terus bertambah (akumulatif)
    - Agent harus memutuskan: lanjut dengan penalty atau cari jalan aman
    - Memory dipakai untuk pengambilan keputusan
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, agent_id: str, stage_config: dict, leaderboard: Leaderboard):
        super().__init__()
        self.agent_id = agent_id
        self.config = stage_config
        self.leaderboard = leaderboard

        # Memory & Decision systems
        self.memory = RecallMemory(agent_id)
        self.decision_maker = DecisionMaker()

        # CARLA connection
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(60.0)
        self.world = self.client.load_world(self.config["map"])
        self.map = self.world.get_map()

        # Spawn points
        all_spawns = self.map.get_spawn_points()
        self.initial_spawn = all_spawns[0]
        self._all_spawns = all_spawns

        # Physics
        self._settings = self.world.get_settings()
        self._settings.synchronous_mode = True
        self._settings.fixed_delta_seconds = 0.05
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

        # Sensor buffers
        self.last_img = None
        self.last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        self._img_lock = threading.Lock()
        self._lidar_lock = threading.Lock()

        # Episode state (NO-RESET: accumulative penalties)
        self.npc_vehicles = []
        self.npc_pedestrians = []
        self.collision_count = 0
        self.offroad_count = 0
        self.red_light_violations = 0
        self.visited_waypoints = set()
        self.waypoint_sequence = 0

        self.steps = 0
        self.episode_reward = 0.0
        self.total_distance = 0.0
        self.last_steer = 0.0
        self.last_throttle = 0.0
        self.last_loc = None

        # Penalty tracking
        self.current_penalty = 0.0
        self.penalty_history = []
        self.safety_score = 100.0

    def _cleanup(self):
        """Destroy all actors"""
        actors_to_destroy = []
        for attr in ("camera_sensor", "lidar_sensor", "collision_sensor",
                     "lane_sensor", "vehicle"):
            a = getattr(self, attr, None)
            if a is not None:
                actors_to_destroy.append(a)
                setattr(self, attr, None)

        for npc_list in (self.npc_vehicles, self.npc_pedestrians):
            for npc in npc_list[:]:
                try:
                    if npc and hasattr(npc, 'is_alive') and npc.is_alive:
                        actors_to_destroy.append(npc)
                except Exception:
                    pass
        self.npc_vehicles = []
        self.npc_pedestrians = []

        if actors_to_destroy:
            try:
                valid_actors = [a for a in actors_to_destroy if a and hasattr(a, 'id')]
                if valid_actors:
                    self.client.apply_batch_sync([
                        carla.command.DestroyActor(a) for a in valid_actors
                    ], True)
            except Exception:
                pass

    def _spawn_npcs(self):
        bp_lib = self.world.get_blueprint_library()

        # Vehicles
        if self.config["traffic"] > 0:
            blueprints = bp_lib.filter("vehicle.*")
            points = random.sample(self._all_spawns,
                                   min(self.config["traffic"], len(self._all_spawns)))
            batch = []
            for sp in points:
                bp = random.choice(blueprints)
                batch.append(carla.command.SpawnActor(bp, sp)
                             .then(carla.command.SetAutopilot(
                                 carla.command.FutureActor, True, 8000)))
            results = self.client.apply_batch_sync(batch, True)
            for r in results:
                if not r.error:
                    actor = self.world.get_actor(r.actor_id)
                    if actor:
                        self.npc_vehicles.append(actor)

        # Pedestrians
        if self.config["pedestrians"] > 0:
            walker_bps = bp_lib.filter("walker.pedestrian.*")
            ctrl_bp = bp_lib.find("controller.ai.walker")

            spawn_locs = []
            for _ in range(self.config["pedestrians"]):
                loc = self.world.get_random_location_from_navigation()
                if loc:
                    spawn_locs.append(carla.Transform(loc))

            for sp in spawn_locs:
                try:
                    w = self.world.spawn_actor(random.choice(walker_bps), sp)
                    ctrl = self.world.spawn_actor(ctrl_bp, carla.Transform(),
                                                   attach_to=w)
                    ctrl.start()
                    ctrl.go_to_location(
                        self.world.get_random_location_from_navigation())
                    ctrl.set_max_speed(1.4)
                    self.npc_pedestrians.extend([w, ctrl])
                except Exception:
                    pass

    def _randomize_weather(self):
        w = carla.WeatherParameters(
            cloudiness=random.uniform(0, 30),
            precipitation=random.uniform(0, 10),
            precipitation_deposits=random.uniform(0, 10),
            sun_azimuth_angle=random.uniform(0, 360),
            sun_altitude_angle=random.uniform(20, 70),
            fog_density=random.uniform(0, 5),
            wetness=random.uniform(0, 20),
        )
        self.world.set_weather(w)

    # Sensor callbacks
    def _cb_camera(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3]
        with self._img_lock:
            self.last_img = arr.copy()

    def _cb_lidar(self, point_cloud):
        pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        if pts.size == 0:
            return
        pts = pts.reshape(-1, 4)[:, :3]
        pts = pts[pts[:, 2] > -1.0]  # Remove ground
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

    def _cb_collision(self, event):
        self.collision_count += 1
        self.current_penalty += 100.0
        self.safety_score = max(0, self.safety_score - 5.0)
        self.memory.add_collision(event.location)

    def _cb_lane(self, event):
        pass  # Lane invasion handled in step()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()
        self._randomize_weather()

        # Spawn vehicle
        bp = self.world.get_blueprint_library().filter("model3")[0]
        for attempt in range(5):
            try:
                self.vehicle = self.world.spawn_actor(bp, self.initial_spawn)
                break
            except RuntimeError:
                time.sleep(0.5)
                if attempt == 4:
                    raise RuntimeError("Could not spawn vehicle")

        self._spawn_npcs()

        # Sensors
        lib = self.world.get_blueprint_library()

        cam_bp = lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
        cam_bp.set_attribute("fov", "90")
        self.camera_sensor = self.world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            attach_to=self.vehicle
        )
        self.camera_sensor.listen(self._cb_camera)

        lidar_bp = lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "16")
        lidar_bp.set_attribute("range", str(LIDAR_MAX_RANGE))
        lidar_bp.set_attribute("points_per_second", "100000")
        lidar_bp.set_attribute("rotation_frequency", "20")
        self.lidar_sensor = self.world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=0.0, z=2.4)),
            attach_to=self.vehicle
        )
        self.lidar_sensor.listen(self._cb_lidar)

        self.collision_sensor = self.world.spawn_actor(
            lib.find("sensor.other.collision"),
            carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(self._cb_collision)

        self.lane_sensor = self.world.spawn_actor(
            lib.find("sensor.other.lane_invasion"),
            carla.Transform(), attach_to=self.vehicle
        )
        self.lane_sensor.listen(self._cb_lane)

        # Wait for sensors
        for _ in range(20):
            self.world.tick()
            with self._img_lock:
                if self.last_img is not None:
                    break

        # Reset episode state (but keep memory!)
        self.steps = 0
        self.episode_reward = 0.0
        self.current_penalty = 0.0
        self.penalty_history = []
        self.total_distance = 0.0
        self.last_loc = None
        self.waypoint_sequence = 0

        # NOTE: We DON'T reset these - they accumulate across episodes!
        # self.collision_count = 0  # Keep counting!
        # self.offroad_count = 0
        # self.safety_score - decays over time

        print(f"[ENV] {self.agent_id} - Episode reset. Safety Score: {self.safety_score:.1f}")

        return self._get_obs(), {}

    def _get_obs(self) -> dict:
        with self._img_lock:
            cam = self.last_img if self.last_img is not None else \
                  np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        with self._lidar_lock:
            lidar = self.last_lidar.copy()

        v = self.vehicle.get_velocity()
        kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

        loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(loc, project_to_road=True)

        # Red light
        red_light = 0.0
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl and tl.get_state() == carla.TrafficLightState.Red:
                red_light = 1.0

        # Junction
        at_junction = float(wp.is_junction) if wp else 0.0

        # Danger from memory
        danger_memory = self.memory.get_danger_score(loc)

        # Route guidance
        route_guidance_x = route_guidance_y = 0.0
        guidance = self.memory.get_route_guidance(loc)
        if guidance:
            dx = guidance.x - loc.x
            dy = guidance.y - loc.y
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                route_guidance_x = dx / norm
                route_guidance_y = dy / norm

        # LiDAR closest obstacle
        min_lidar = float(np.min(lidar))

        state = np.array([
            kmh / 50.0,
            self.last_steer,
            self.last_throttle,
            at_junction,
            red_light,
            self.current_penalty / 1000.0,  # Normalized penalty
            self.safety_score / 100.0,
            danger_memory,
            min_lidar / LIDAR_MAX_RANGE,
            float(self.collision_count > 0),
            float(self.offroad_count > 0),
            float(self.steps % 1000 / 1000),  # Time in episode
            route_guidance_x,
            route_guidance_y,
            float(len(self.visited_waypoints) / 1000),  # Progress
        ], dtype=np.float32)

        return {"camera": cam, "lidar": lidar, "state": state}

    def step(self, action):
        self.steps += 1
        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])
        self.last_steer = steer
        self.last_throttle = throttle

        # Get current state for decision making
        v = self.vehicle.get_velocity()
        kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

        loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(loc, project_to_road=False,
                                   lane_type=carla.LaneType.Driving)
        wp_near = self.map.get_waypoint(loc, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        on_road = wp is not None or wp_near is not None

        # Check danger ahead from memory
        danger_ahead = self.memory.get_danger_score(loc)

        # Check nearby hazards from LiDAR
        min_lidar_dist = float(np.min(self.last_lidar))
        nearby_hazard = min_lidar_dist < 10.0

        # DECISION MAKING: Should the agent hesitate?
        should_hesitate, reason = self.decision_maker.should_hesitate(
            kmh, danger_ahead, on_road, nearby_hazard
        )

        if should_hesitate:
            # Force slow down for "thinking"
            throttle = min(throttle, 0.2)
            brake = max(brake, 0.3)
            if self.steps % 50 == 0:
                print(f"[DECISION] {self.agent_id} hesitating: {reason}")

        # Anti-stall
        if kmh < 2.0:
            throttle = max(throttle, 0.4)

        self.vehicle.apply_control(
            carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        )
        self.world.tick()

        # Calculate distance
        if self.last_loc is not None:
            dist = loc.distance(self.last_loc)
            self.total_distance += dist
        self.last_loc = loc

        # =========================================================================
        # REWARD SYSTEM - NO RESET, ACCUMULATIVE PENALTIES
        # =========================================================================
        reward = 0.0
        done = False
        info = {}
        grace_period = self.steps < 50

        # 1. Distance reward (primary goal - keep moving!)
        reward += 1.0  # Per step

        # 2. Waypoint discovery (exploration bonus)
        active_wp = wp if wp else wp_near
        if active_wp and active_wp.id not in self.visited_waypoints:
            reward += 20.0
            self.visited_waypoints.add(active_wp.id)
            self.memory.add_waypoint(active_wp.transform.location, self.waypoint_sequence)
            self.waypoint_sequence += 1

        # 3. Speed reward (but not too fast in dangerous areas)
        if danger_ahead > 0.5:
            # Safer to go slow in dangerous areas
            optimal_speed = 20.0
            if kmh < optimal_speed:
                reward += 2.0
        else:
            optimal_speed = 40.0
            speed_ratio = min(kmh / optimal_speed, 1.0)
            reward += speed_ratio * 3.0

        # 4. Following memory route bonus
        guidance = self.memory.get_route_guidance(loc)
        if guidance:
            dist_to_guidance = loc.distance(guidance)
            if dist_to_guidance < 5.0:
                reward += 5.0 * (1.0 - dist_to_guidance/5.0)

        # 5. Center-line bonus
        if wp is not None:
            reward += 2.0
        elif wp_near is not None:
            reward += 0.5

        # =========================================================================
        # PENALTIES - ACCUMULATIVE, NO RESET
        # =========================================================================

        # Off-road penalty - NO RESET, just accumulate
        if not grace_period and not on_road:
            if wp_near and wp_near.lane_type in (carla.LaneType.Sidewalk,
                                                  carla.LaneType.NONE):
                penalty = 50.0  # Per step while off-road
                self.current_penalty += penalty
                self.safety_score = max(0, self.safety_score - 0.5)
                self.offroad_count += 1
                reward -= penalty
                self.memory.add_offroad(loc)
                if self.steps % 25 == 0:
                    print(f"[PENALTY] {self.agent_id} OFF-ROAD! Total penalty: {self.current_penalty:.1f}")

        # Red light violation - accumulative
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl and tl.get_state() == carla.TrafficLightState.Red and kmh > 1.0:
                penalty = 100.0
                self.current_penalty += penalty
                self.safety_score = max(0, self.safety_score - 2.0)
                self.red_light_violations += 1
                reward -= penalty
                self.memory.add_red_light_violation(loc)
                print(f"[PENALTY] {self.agent_id} RED LIGHT! Total: {self.current_penalty:.1f}")

        # Lane change penalty (minor)
        # if self.lane_sensor: (handled via callback)

        # Collision already added via callback

        # =========================================================================
        # EPISODE TERMINATION - Only on extreme cases or timeout
        # =========================================================================
        if self.current_penalty > 1000.0:  # Too many mistakes
            done = True
            info["termination"] = "max_penalty_reached"

        if self.steps >= 5000:  # Episode timeout
            done = True
            info["termination"] = "timeout"

        # =========================================================================
        # UPDATE LEADERBOARD
        # =========================================================================
        self.episode_reward += reward

        if done:
            # Update leaderboard record
            self.leaderboard.update(
                self.agent_id,
                total_distance=self.total_distance,
                total_penalty=self.current_penalty,
                collisions=self.collision_count,
                off_roads=self.offroad_count,
                red_light_violations=self.red_light_violations,
                waypoints_visited=len(self.visited_waypoints),
                episodes_completed=self.leaderboard.records.get(
                    self.agent_id, AgentRecord(agent_id=self.agent_id)
                ).episodes_completed + 1,
                best_single_distance=max(
                    self.leaderboard.records.get(
                        self.agent_id, AgentRecord(agent_id=self.agent_id)
                    ).best_single_distance,
                    self.total_distance
                ),
                safety_score=self.safety_score
            )

            info["episode_reward"] = self.episode_reward
            info["total_distance"] = self.total_distance
            info["total_penalty"] = self.current_penalty
            info["safety_score"] = self.safety_score

            print(f"[DONE] {self.agent_id} - Distance: {self.total_distance:.1f}m, "
                  f"Penalty: {self.current_penalty:.1f}, Safety: {self.safety_score:.1f}")

        return self._get_obs(), reward, done, False, info

    def close(self):
        self._settings.synchronous_mode = False
        self._settings.fixed_delta_seconds = None
        self.world.apply_settings(self._settings)
        self._cleanup()
        self.memory.save()
        print(f"[ENV] {self.agent_id} environment closed.")


# =============================================================================
# 6. MULTI-AGENT COMPETITION ORCHESTRATOR
# =============================================================================
class CompetitionOrchestrator:
    """
    Mengelola kompetisi antar multiple AI agents.
    Setiap agent training dengan environment mereka sendiri.
    """

    STAGES = {
        1: {"name": "Basics", "map": "Town01", "traffic": 0, "pedestrians": 0},
        2: {"name": "Traffic", "map": "Town03", "traffic": 20, "pedestrians": 10},
        3: {"name": "Expert", "map": "Town05", "traffic": 50, "pedestrians": 30},
    }

    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.leaderboard = Leaderboard()
        self.agents = []
        self.envs = []

    def create_agents(self):
        """Buat multiple agents untuk kompetisi"""
        for i in range(self.num_agents):
            agent_id = f"Racer_v1_{i+1}"
            stage = self.STAGES[1]  # All start at stage 1

            env = CarlaCompetitionEnv(
                agent_id=agent_id,
                stage_config=stage,
                leaderboard=self.leaderboard
            )

            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=0,
                device="cuda",
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                ent_coef=0.05,  # High exploration for competition
                tensorboard_log=os.path.join(BASE_DIR, "tb_logs"),
            )

            self.agents.append({"id": agent_id, "model": model, "env": env})
            print(f"[INIT] Created agent: {agent_id}")

    def train_round(self, rounds: int = 5, steps_per_round: int = 10000):
        """Training session dengan leaderboard update tiap round"""

        for round_num in range(1, rounds + 1):
            print(f"\n{'='*60}")
            print(f"  ROUND {round_num}/{rounds} - COMPETITION TRAINING")
            print(f"{'='*60}\n")

            # Train all agents
            for agent in self.agents:
                print(f"[TRAIN] Training {agent['id']}...")
                try:
                    agent['model'].learn(
                        total_timesteps=steps_per_round,
                        progress_bar=True,
                    )
                    # Save checkpoint
                    ckpt_path = os.path.join(CHECKPOINT_DIR,
                                             f"{agent['id']}_r{round_num}.zip")
                    agent['model'].save(ckpt_path)
                except Exception as e:
                    print(f"[ERROR] Training failed for {agent['id']}: {e}")

            # Print leaderboard
            self.leaderboard.print_leaderboard()

            # Save best agent
            best_id = self.leaderboard.get_best_agent()
            if best_id:
                print(f"[INFO] Current champion: {best_id}")
                best_model_path = os.path.join(BASE_DIR, f"{best_id}_champion.zip")
                for agent in self.agents:
                    if agent['id'] == best_id:
                        agent['model'].save(best_model_path)
                        break

    def cleanup(self):
        for agent in self.agents:
            agent['env'].close()


# =============================================================================
# 7. MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CARLA AI Competition Training")
    parser.add_argument("--mode", choices=["train", "rank", "battle"], default="train",
                        help="Mode: train new agents, show ranking, or battle")
    parser.add_argument("--agents", type=int, default=3,
                        help="Number of agents to compete")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of training rounds")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Steps per round")

    args = parser.parse_args()

    leaderboard = Leaderboard()

    if args.mode == "rank":
        # Just show current leaderboard
        leaderboard.print_leaderboard(top_n=20)

    elif args.mode == "train":
        # Train multiple competing agents
        print("\n" + "="*60)
        print("  🏎️  CARLA AI COMPETITION LEAGUE 🏎️".center(60))
        print("="*60 + "\n")

        orchestrator = CompetitionOrchestrator(num_agents=args.agents)
        orchestrator.create_agents()

        try:
            orchestrator.train_round(
                rounds=args.rounds,
                steps_per_round=args.steps
            )
        except KeyboardInterrupt:
            print("\n[STOP] Training interrupted. Saving...")
        finally:
            orchestrator.cleanup()
            leaderboard.print_leaderboard()

    elif args.mode == "battle":
        # Load best agents and let them compete
        print("[BATTLE] Loading top 2 agents for head-to-head competition...")
        ranking = leaderboard.get_ranking()
        if len(ranking) < 2:
            print("Need at least 2 trained agents!")
        else:
            agent1_id = ranking[0][1]
            agent2_id = ranking[1][1]
            print(f"[BATTLE] {agent1_id} vs {agent2_id}")
            # TODO: Implement head-to-head battle mode
