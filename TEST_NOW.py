"""
Quick Test — Load v14 checkpoint and drive live in CARLA.
Uses v14's GrandmasterFeatureExtractor (compatible with checkpoint).
"""
# NumPy 2.x compat shim (checkpoint saved with numpy 2.x, we have 1.x)
import numpy as np
import numpy.core
if not hasattr(np, '_core'):
    import types
    np._core = types.ModuleType('numpy._core')
    np._core.numeric = numpy.core.numeric
    np._core.multiarray = numpy.core.multiarray
    sys_modules = __import__('sys').modules
    sys_modules['numpy._core'] = np._core
    sys_modules['numpy._core.numeric'] = numpy.core.numeric
    sys_modules['numpy._core.multiarray'] = numpy.core.multiarray

import os, sys, time, math, threading
import numpy as np
import carla
import pygame
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import glob, re

BASE_DIR = r"E:\AI\traffic_vision\master_suite"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

# v14 constants
IMG_WIDTH, IMG_HEIGHT = 80, 60
N_LIDAR_SECTORS = 16
LIDAR_MAX_RANGE = 50.0
STATE_DIM = 28
DISPLAY_W, DISPLAY_H = 1280, 720
TARGET_FPS = 60

# Semantic palette
PALETTE = {
    0:[0,0,0], 1:[100,0,0], 2:[150,0,0], 3:[100,50,50],
    4:[255,0,255], 5:[255,255,0], 6:[0,255,0], 7:[0,255,0],
    8:[255,0,0], 9:[150,75,0], 10:[255,100,0], 11:[120,0,0],
    12:[255,255,255],
}

# === v14 Feature Extractor (MUST match training) ===
class CNNExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=3), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            self.out_dim = self.cnn(th.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)).shape[1]
    def forward(self, x):
        return self.cnn(x)

class GrandmasterFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.vision = CNNExtractor()
        vd = self.vision.out_dim
        self.lidar_spatial = nn.Sequential(
            nn.Linear(N_LIDAR_SECTORS*3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.LayerNorm(128))
        self.danger_head = nn.Sequential(
            nn.Linear(N_LIDAR_SECTORS, 32), nn.ReLU(), nn.Linear(32, 16))
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, 64), nn.ReLU())
        self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(vd + 128 + 64, features_dim), nn.ReLU())

    def forward(self, obs):
        cam = obs["camera"].float()
        if cam.ndim == 4 and cam.shape[-1] == 3:
            cam = cam.permute(0, 3, 1, 2) / 255.0
        elif cam.max() > 1.0:
            cam = cam / 255.0
        vis = self.vision(cam)
        lidar = obs["lidar"]
        angles = th.linspace(0, 2*math.pi, N_LIDAR_SECTORS, device=lidar.device)
        li = th.cat([lidar, th.sin(angles).unsqueeze(0).expand(lidar.shape[0],-1),
                     th.cos(angles).unsqueeze(0).expand(lidar.shape[0],-1)], dim=-1)
        lf = self.lidar_spatial(li)
        sf = self.state_encoder(obs["state"])
        return self.fusion(th.cat([vis, lf, sf], dim=-1))

# === Camera Buffer ===
class CamBuf:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
    def write(self, img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
        ids = arr[:,:,2]
        rgb = np.zeros((img.height, img.width, 3), dtype=np.uint8)
        for cid, col in PALETTE.items():
            rgb[ids == cid] = col
        rgb[ids > 12] = [80, 0, 0]
        with self._lock:
            self._frame = rgb
    def read(self):
        with self._lock:
            return self._frame

# === LiDAR Buffer ===
class LidarBuf:
    def __init__(self):
        self._lock = threading.Lock()
        self._sectors = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
    def write(self, data):
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        if len(pts) == 0: return
        x, y = pts[:,0], pts[:,1]
        angles = np.arctan2(y, x) % (2*np.pi)
        dists = np.sqrt(x**2 + y**2)
        sectors = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        for i in range(N_LIDAR_SECTORS):
            lo = i * 2*np.pi / N_LIDAR_SECTORS
            hi = (i+1) * 2*np.pi / N_LIDAR_SECTORS
            mask = (angles >= lo) & (angles < hi)
            if mask.any():
                sectors[i] = min(float(np.min(dists[mask])), LIDAR_MAX_RANGE)
        with self._lock:
            self._sectors = sectors
    def read(self):
        with self._lock:
            return self._sectors.copy()

# HUD colors
C_BG = (13,17,23,180); C_GREEN = (63,185,80); C_RED = (248,81,73)
C_WHITE = (230,237,243); C_GRAY = (139,148,158); C_ORANGE = (210,153,34)
C_BLUE = (88,166,255); C_CYAN = (0,230,230)

class HUD:
    def __init__(self, surf):
        self.s = surf
        self.fl = pygame.font.SysFont("consolas", 22, bold=True)
        self.fm = pygame.font.SysFont("consolas", 15)
        self.fs = pygame.font.SysFont("consolas", 12)
    def t(self, txt, x, y, c=C_WHITE, f=None):
        self.s.blit((f or self.fm).render(txt, True, c), (x, y))
    def draw(self, info):
        p = pygame.Surface((320, 280), pygame.SRCALPHA); p.fill(C_BG)
        self.s.blit(p, (10, 10))
        mc = C_ORANGE if info.get("mode")=="HUMAN" else C_GREEN
        self.t(f"MODE: {info.get('mode','AI')}", 20, 18, mc, self.fl)
        self.t(f"Speed   : {info.get('speed',0):6.1f} km/h", 20, 50)
        self.t(f"Steer   : {info.get('steer',0):+.3f}", 20, 70)
        self.t(f"Throttle: {info.get('throttle',0):.3f}", 20, 90)
        self.t(f"Brake   : {info.get('brake',0):.3f}", 20, 110)
        pygame.draw.line(self.s, C_GRAY, (20,130), (310,130), 1)
        self.t(f"Distance : {info.get('dist',0):8.1f} m", 20, 138)
        self.t(f"Avg Speed: {info.get('avg',0):8.1f} km/h", 20, 158)
        cc = C_RED if info.get("col",0)>0 else C_GREEN
        self.t(f"Collisions: {info.get('col',0):5d}", 20, 178, cc)
        self.t(f"LiDAR min : {info.get('lidar_min',50):5.1f} m", 20, 198)
        self.t(f"Model: v14 @ {info.get('steps','?')} steps", 20, 224, C_CYAN, self.fs)
        self.t(f"FPS: {info.get('fps',0):.0f} | {info.get('elapsed',0):.0f}s", 20, 244, C_GRAY, self.fs)
        self.t("SPACE=Override  ESC=Quit  Arrows=Steer", 10, DISPLAY_H-22, C_GRAY, self.fs)

def find_best_ckpt():
    ckpts = glob.glob(os.path.join(CKPT_DIR, "v14_*_steps.zip"))
    if not ckpts: return None, 0
    best, bs = None, 0
    for c in ckpts:
        m = re.search(r'(\d+)_steps', c)
        if m and int(m.group(1)) > bs:
            bs = int(m.group(1)); best = c
    return best, bs

def main():
    ckpt, steps = find_best_ckpt()
    if not ckpt:
        print("[ERROR] No v14 checkpoint found!"); return
    print(f"[LOAD] {ckpt} ({steps:,} steps)")

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption(f"CARLA Grandmaster v14 — Live Test ({steps:,} steps)")
    clock = pygame.time.Clock()
    hud = HUD(screen)

    # Load model with custom extractor
    custom = {"features_extractor_class": GrandmasterFeatureExtractor,
              "features_extractor_kwargs": {"features_dim": 256}}
    model = PPO.load(ckpt, device="cuda", custom_objects={"policy_kwargs": {
        "features_extractor_class": GrandmasterFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),
    }})
    print("[LOAD] Model loaded!")

    # CARLA
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    lib = world.get_blueprint_library()
    vehicle = world.spawn_actor(lib.filter("model3")[0], world.get_map().get_spawn_points()[0])

    # Sensors
    cam_bp = lib.find("sensor.camera.semantic_segmentation")
    cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
    cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
    camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)
    cam_buf = CamBuf()
    camera.listen(cam_buf.write)

    lidar_bp = lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", "32")
    lidar_bp.set_attribute("range", str(LIDAR_MAX_RANGE))
    lidar_bp.set_attribute("points_per_second", "200000")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("upper_fov", "10")
    lidar_bp.set_attribute("lower_fov", "-30")
    lidar_sensor = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=2.5)), attach_to=vehicle)
    lidar_buf = LidarBuf()
    lidar_sensor.listen(lidar_buf.write)

    col_sensor = world.spawn_actor(lib.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle)
    col_count = [0]
    col_sensor.listen(lambda _: col_count.__setitem__(0, col_count[0]+1))

    t0 = time.time(); dist = 0.0; speeds = []
    prev_loc = vehicle.get_location()
    steer_k = 0.0; last_s = 0.0; last_t = 0.0

    print("[LIVE] Running! Hold SPACE=override, ESC=quit")
    try:
        while True:
            clock.tick(TARGET_FPS)
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    raise KeyboardInterrupt

            keys = pygame.key.get_pressed()
            human = bool(keys[pygame.K_SPACE])
            loc = vehicle.get_location()
            v = vehicle.get_velocity()
            kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            lidar_data = lidar_buf.read()

            if human:
                if keys[pygame.K_LEFT]: steer_k = max(steer_k-0.04, -1)
                elif keys[pygame.K_RIGHT]: steer_k = min(steer_k+0.04, 1)
                else: steer_k *= 0.85
                tk = 1.0 if keys[pygame.K_UP] else 0.0
                bk = 1.0 if keys[pygame.K_DOWN] else 0.0
                vehicle.apply_control(carla.VehicleControl(steer=steer_k, throttle=tk, brake=bk))
                sd, td, bd = steer_k, tk, bk
            else:
                frame = cam_buf.read()
                if frame is not None:
                    state = np.zeros(STATE_DIM, np.float32)
                    state[0] = kmh/50.0; state[1] = last_s; state[2] = last_t
                    state[8] = float(np.min(lidar_data))/LIDAR_MAX_RANGE
                    obs = {"camera": frame, "lidar": lidar_data, "state": state}
                    action, _ = model.predict(obs, deterministic=True)
                    sd, td, bd = float(action[0]), float(action[1]), float(action[2])
                    vehicle.apply_control(carla.VehicleControl(
                        steer=sd, throttle=max(0,td), brake=bd, reverse=(td<0)))
                    last_s, last_t = sd, td
                else:
                    sd = td = bd = 0.0

            world.get_spectator().set_transform(
                carla.Transform(loc + carla.Location(z=45), carla.Rotation(pitch=-90)))

            d = loc.distance(prev_loc)
            if d < 10: dist += d
            prev_loc = loc; speeds.append(kmh)
            avg = np.mean(speeds[-200:]) if speeds else 0

            frame_view = cam_buf.read()
            if frame_view is not None:
                surf = pygame.surfarray.make_surface(np.transpose(frame_view, (1,0,2)))
                screen.blit(pygame.transform.scale(surf, (DISPLAY_W, DISPLAY_H)), (0,0))

            hud.draw({"mode": "HUMAN" if human else "AI", "speed": kmh,
                      "steer": sd, "throttle": td, "brake": bd,
                      "dist": dist, "avg": avg, "col": col_count[0],
                      "lidar_min": float(np.min(lidar_data)),
                      "steps": f"{steps:,}", "fps": clock.get_fps(),
                      "elapsed": time.time()-t0})
            pygame.display.flip()

    except KeyboardInterrupt:
        print("\n[STOP] Test ended.")
    finally:
        for a in (col_sensor, lidar_sensor, camera, vehicle):
            try: a.destroy()
            except: pass
        pygame.quit()

if __name__ == "__main__":
    main()
