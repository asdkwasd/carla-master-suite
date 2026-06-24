"""
==============================================================================
  CARLA GRANDMASTER AI SUITE v8.0 — 3_TEST_LIVE.py (PATCHED)
  Live evaluation with:
    • Smooth async physics (60 FPS async mode)
    • Semantic Segmentation Eye (matches training palette)
    • CarlaFeaturesExtractor support for v8.0 PPO models
    • Human-in-the-Loop override (Pillar 7)
      – Hold SPACE → switch to keyboard control
      – Release SPACE → AI resumes
    • On-screen HUD: speed, steer, mode, session stats
==============================================================================
"""
import os
import sys
import time
import threading
import numpy as np
import carla
import pygame
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = r"D:\AI\traffic_vision\master_suite"
MODEL_PATH    = os.path.join(BASE_DIR, "carla_grandmaster_v8.zip")
CARLA_HOST    = "127.0.0.1"
CARLA_PORT    = 2000
IMG_WIDTH     = 160
IMG_HEIGHT    = 120
DISPLAY_W     = 900
DISPLAY_H     = 600
TARGET_FPS    = 60
STEER_STEP    = 0.04   # keyboard steering increment per frame
STATE_DIM     = 11     # Matches v8.0 Trainer
N_LIDAR_SECTORS = 8
LIDAR_MAX_RANGE = 50.0

# ── Colour palette for HUD ────────────────────────────────────────────────────
HUD_BG      = (13,  17,  23,  180)    # dark translucent
HUD_GREEN   = (63,  185, 80)
HUD_RED     = (248, 81,  73)
HUD_WHITE   = (230, 237, 243)
HUD_GRAY    = (139, 148, 158)
HUD_ORANGE  = (210, 153, 34)

# ==============================================================================
# CUSTOM FEATURE EXTRACTOR (REQUIRED FOR v8.0 MODELS)
# ==============================================================================
class CarlaFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=512)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Linear(64 * 7 * 7 + STATE_DIM + N_LIDAR_SECTORS, 512)

    def forward(self, observations):
        cam = observations["camera"].float() / 255.0
        cam = cam.permute(0, 3, 1, 2)
        cnn_feat = self.cnn(cam)
        combined = th.cat([cnn_feat, observations["state"], observations["lidar"]], dim=1)
        return self.linear(combined)

# ==============================================================================
# HUD RENDERER
# ==============================================================================
class HUD:
    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        self.font_lg = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_sm = pygame.font.SysFont("consolas", 14)
        self.font_xs = pygame.font.SysFont("consolas", 11)

    def _text(self, text: str, x: int, y: int, color=HUD_WHITE, font=None):
        font = font or self.font_sm
        surf = font.render(text, True, color)
        self.surface.blit(surf, (x, y))

    def draw(self, info: dict):
        panel = pygame.Surface((280, 210), pygame.SRCALPHA)
        panel.fill(HUD_BG)
        self.surface.blit(panel, (10, 10))

        mode_color = HUD_ORANGE if info.get("mode") == "HUMAN" else HUD_GREEN
        self._text(f"MODE : {info.get('mode','AI')}", 20, 18, mode_color, self.font_lg)
        self._text(f"Speed   : {info.get('speed',0.0):6.1f} km/h", 20, 48)
        self._text(f"Steer   : {info.get('steer',0.0):+.3f}", 20, 66)
        self._text(f"Throttle: {info.get('throttle',0.0):.3f}", 20, 84)
        self._text(f"Brake   : {info.get('brake',0.0):.3f}", 20, 102)
        pygame.draw.line(self.surface, HUD_GRAY, (20, 120), (280, 120), 1)
        self._text(f"Distance: {info.get('distance',0.0):7.1f} m", 20, 128)
        self._text(f"Avg Speed: {info.get('avg_speed',0.0):7.1f} km/h", 20, 146)
        col_color = HUD_RED if info.get("collisions", 0) > 0 else HUD_GREEN
        self._text(f"Collisions: {info.get('collisions',0):7d}", 20, 164, col_color)
        self._text(f"FPS: {info.get('fps',0.0):.0f} | Elapsed: {info.get('elapsed',0):.0f}s", 20, 190, HUD_GRAY, self.font_xs)
        self._text("Hold [SPACE] = Override  [ESC] = Quit", 10, DISPLAY_H - 22, HUD_GRAY, self.font_xs)

# ==============================================================================
# SEMANTIC CAMERA BUFFER (Matches Trainer Palette)
# ==============================================================================
class CameraBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._palette = {
            0: [0, 0, 0], 1: [100, 0, 0], 2: [150, 0, 0], 3: [100, 50, 50],
            4: [255, 0, 255], 5: [255, 255, 0], 6: [0, 255, 0], 7: [0, 255, 0],
            8: [255, 0, 0], 9: [150, 75, 0], 10: [255, 100, 0], 11: [120, 0, 0],
            12: [255, 255, 255],
        }

    def write(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        class_ids = arr[:, :, 2]
        rgb_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        for cid, color in self._palette.items():
            rgb_img[class_ids == cid] = color
        rgb_img[class_ids > 12] = [80, 0, 0]
        with self._lock:
            self._frame = rgb_img

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

# ==============================================================================
# MAIN TEST FUNCTION
# ==============================================================================
def run_live_test():
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption("CARLA Grandmaster v8.0 — Live Evaluation")
    clock = pygame.time.Clock()
    hud = HUD(screen)

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(15.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        pygame.quit(); return

    print(f"[LOAD] Loading brain: {MODEL_PATH}")
    # Custom extractor must be defined or passed during load if dimension changes occurred
    model = PPO.load(MODEL_PATH, device="cuda")

    lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(lib.filter("model3")[0], spawn_points[0])

    cam_bp = lib.find("sensor.camera.semantic_segmentation")
    cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
    cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
    camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)        
    cam_buf = CameraBuffer()
    camera.listen(cam_buf.write)

    col_sensor = world.spawn_actor(lib.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle)    
    collision_count = [0]
    col_sensor.listen(lambda _: collision_count.__setitem__(0, collision_count[0] + 1))

    session_start = time.time()
    distance_m = 0.0
    speed_samples = []
    prev_loc = vehicle.get_location()
    steer_k = 0.0
    last_steer = 0.0
    last_throttle = 0.0

    print("[LIVE] Running. Hold SPACE for override. ESC to quit.")

    try:
        while True:
            clock.tick(TARGET_FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    raise KeyboardInterrupt

            keys = pygame.key.get_pressed()
            human_mode = bool(keys[pygame.K_SPACE])

            cur_loc = vehicle.get_location()
            v = vehicle.get_velocity()
            kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

            if human_mode:
                if keys[pygame.K_LEFT]: steer_k = max(steer_k - STEER_STEP, -1.0)
                elif keys[pygame.K_RIGHT]: steer_k = min(steer_k + STEER_STEP, 1.0)
                else: steer_k *= 0.85
                t_k = 1.0 if keys[pygame.K_UP] else 0.0
                b_k = 1.0 if keys[pygame.K_DOWN] else 0.0
                vehicle.apply_control(carla.VehicleControl(steer=steer_k, throttle=t_k, brake=b_k))
                s_disp, t_disp, b_disp = steer_k, t_k, b_k
                last_steer, last_throttle = steer_k, t_k
            else:
                frame = cam_buf.read()
                if frame is not None:
                    # Construct v8.0 state vector
                    state = np.zeros(STATE_DIM, dtype=np.float32)
                    state[0] = kmh / 50.0
                    state[1] = last_steer
                    state[2] = last_throttle
                    
                    obs = {
                        "camera": frame,
                        "lidar":  np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, dtype=np.float32),
                        "state":  state
                    }
                    action, _ = model.predict(obs, deterministic=True)
                    s_ai, t_ai, b_ai = float(action[0]), float(action[1]), float(action[2])
                    vehicle.apply_control(carla.VehicleControl(steer=s_ai, throttle=t_ai, brake=b_ai))
                    s_disp, t_disp, b_disp = s_ai, t_ai, b_ai
                    last_steer, last_throttle = s_ai, t_ai
                else:
                    s_disp = t_disp = b_disp = 0.0

            # Spectator follow (Top-down bird's eye)
            world.get_spectator().set_transform(carla.Transform(cur_loc + carla.Location(z=45), carla.Rotation(pitch=-90)))

            distance_m += cur_loc.distance(prev_loc) if cur_loc.distance(prev_loc) < 10 else 0
            prev_loc = cur_loc
            speed_samples.append(kmh)
            avg_speed = np.mean(speed_samples[-200:]) if speed_samples else 0.0

            frame_view = cam_buf.read()
            if frame_view is not None:
                surf = pygame.surfarray.make_surface(np.transpose(frame_view, (1, 0, 2)))
                screen.blit(pygame.transform.scale(surf, (DISPLAY_W, DISPLAY_H)), (0, 0))

            hud.draw({"mode": "HUMAN" if human_mode else "AI", "speed": kmh, "steer": s_disp, "throttle": t_disp, "brake": b_disp,
                      "distance": distance_m, "avg_speed": avg_speed, "collisions": collision_count[0], "fps": clock.get_fps(), "elapsed": time.time()-session_start})
            pygame.display.flip()

    except KeyboardInterrupt:
        print("\n[STOP] Live test ended.")
    finally:
        for actor in (col_sensor, camera, vehicle):
            try: actor.destroy()
            except: pass
        pygame.quit()

if __name__ == "__main__":
    run_live_test()
