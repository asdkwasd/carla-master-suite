"""
==============================================================================
  CARLA GRANDMASTER AI SUITE v8.0 — 4_COLLECT_HUMAN.py (FULL REWRITE)
  Data collection for Behavioral Cloning (BC):
    • Manual keyboard control
    • Records: Semantic Camera, LiDAR (8-sector), and State Vector
    • Saves to: human_demos/observations.npy and actions.npy
    • Format: Dict of arrays (matching Gym observation space)
==============================================================================
"""
import os
import time
import random
import threading
import numpy as np
import cv2
import carla
import pygame

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = r"D:\AI\traffic_vision\master_suite"
HUMAN_DATA_DIR  = os.path.join(BASE_DIR, "human_demos")
os.makedirs(HUMAN_DATA_DIR, exist_ok=True)

IMG_WIDTH, IMG_HEIGHT = 160, 120
N_LIDAR_SECTORS  = 8
LIDAR_MAX_RANGE  = 50.0
STATE_DIM        = 11
TARGET_FPS       = 20   # 20Hz for demo collection is usually enough

# ── Palette for Semantic segmentation ─────────────────────────────────────────
PALETTE = {
    0: [0, 0, 0], 1: [100, 0, 0], 2: [150, 0, 0], 3: [100, 50, 50],
    4: [255, 0, 255], 5: [255, 255, 0], 6: [0, 255, 0], 7: [0, 255, 0],
    8: [255, 0, 0], 9: [150, 75, 0], 10: [255, 100, 0], 11: [120, 0, 0],
    12: [255, 255, 255],
}

class DataCollector:
    def __init__(self):
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        self.vehicle = None
        self.camera = None
        self.lidar = None
        
        self.last_img = None
        self.last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        self._lock = threading.Lock()
        
        self.observations = {"camera": [], "lidar": [], "state": []}
        self.actions = []

    def _cb_camera(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
        class_ids = arr[:, :, 2]
        rgb_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        for cid, color in PALETTE.items():
            rgb_img[class_ids == cid] = color
        rgb_img[class_ids > 12] = [80, 0, 0]
        with self._lock:
            self.last_img = rgb_img

    def _cb_lidar(self, point_cloud):
        pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        if pts.size == 0: return
        pts = pts.reshape(-1, 4)[:, :3]
        pts = pts[pts[:, 2] > -1.0]
        sectors = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        if len(pts) > 0:
            angles  = np.arctan2(pts[:, 1], pts[:, 0]) % (2 * np.pi)
            indices = (angles / (2 * np.pi) * N_LIDAR_SECTORS).astype(int) % N_LIDAR_SECTORS
            dists   = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            for idx, d in zip(indices, dists):
                if d < sectors[idx]: sectors[idx] = d
        with self._lock:
            self.last_lidar = sectors

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("CARLA Human Data Collector")
        clock = pygame.time.Clock()

        # Spawn
        bp = self.world.get_blueprint_library().filter("model3")[0]
        spawn_point = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        
        # Sensors
        cam_bp = self.world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        cam_bp.set_attribute("image_size_x", str(IMG_WIDTH)); cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
        self.camera = self.world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=self.vehicle)
        self.camera.listen(self._cb_camera)
        
        lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "16"); lidar_bp.set_attribute("range", str(LIDAR_MAX_RANGE))
        self.lidar = self.world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0, z=2.4)), attach_to=self.vehicle)
        self.lidar.listen(self._cb_lidar)

        print("[REC] DRIVE using ARROW KEYS. Press 'S' to save and exit. ESC to cancel.")
        
        recording = False
        last_steer, last_throttle = 0.0, 0.0

        try:
            while True:
                clock.tick(TARGET_FPS)
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                
                if keys[pygame.K_ESCAPE]: break
                if keys[pygame.K_s]:
                    self.save_data()
                    break
                
                # Controls
                steer = 0.0; throttle = 0.0; brake = 0.0
                if keys[pygame.K_LEFT]: steer = -1.0
                elif keys[pygame.K_RIGHT]: steer = 1.0
                if keys[pygame.K_UP]: throttle = 1.0
                if keys[pygame.K_DOWN]: brake = 1.0
                
                self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))
                
                # Spectator follow
                loc = self.vehicle.get_location()
                self.world.get_spectator().set_transform(carla.Transform(loc + carla.Location(z=30), carla.Rotation(pitch=-90)))

                # Start recording after moving
                if not recording and (throttle > 0 or abs(steer) > 0.1):
                    recording = True; print("[REC] Recording started...")

                if recording:
                    with self._lock:
                        if self.last_img is not None:
                            v = self.vehicle.get_velocity(); kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
                            
                            # Simple state vector for BC (normalized speed + previous actions)
                            state = np.zeros(STATE_DIM, dtype=np.float32)
                            state[0] = kmh / 50.0
                            state[1] = last_steer
                            state[2] = last_throttle
                            
                            self.observations["camera"].append(self.last_img.copy())
                            self.observations["lidar"].append(self.last_lidar.copy())
                            self.observations["state"].append(state)
                            self.actions.append([steer, throttle, brake])
                            
                            last_steer, last_throttle = steer, throttle
                            
                            # Small preview
                            cv2.imshow("Semantic View", cv2.cvtColor(self.last_img, cv2.COLOR_RGB2BGR))
                            cv2.waitKey(1)

        finally:
            cv2.destroyAllWindows()
            for actor in [self.camera, self.lidar, self.vehicle]:
                if actor: actor.destroy()
            pygame.quit()

    def save_data(self):
        if not self.actions:
            print("[WARN] No data recorded."); return
            
        print(f"[SAVE] Saving {len(self.actions)} frames...")
        # Convert list of arrays to dict of single arrays
        obs_array = {
            "camera": np.array(self.observations["camera"], dtype=np.uint8),
            "lidar":  np.array(self.observations["lidar"], dtype=np.float32),
            "state":  np.array(self.observations["state"], dtype=np.float32)
        }
        act_array = np.array(self.actions, dtype=np.float32)
        
        np.save(os.path.join(HUMAN_DATA_DIR, "observations.npy"), obs_array)
        np.save(os.path.join(HUMAN_DATA_DIR, "actions.npy"), act_array)
        print(f"[DONE] Data saved to {HUMAN_DATA_DIR}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
