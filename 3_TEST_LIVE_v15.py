"""
==============================================================================
  3_TEST_LIVE.py v15 — Live Evaluation with TensorRT

  Supports:
  ✅ SB3 PyTorch inference
  ✅ ONNX Runtime inference
  ✅ TensorRT FP16/INT8 inference (FASTEST)
  ✅ Human override (SPACE key)
  ✅ Full HUD with safety shield status
  ✅ ALL dimensions from core.constants (zero mismatch)
==============================================================================
"""
import os
import sys
import time
import math
import numpy as np
import carla
import pygame

# ── Apply GPU optimizations ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.constants import apply_gpu_opts
apply_gpu_opts()

import torch as th
if th.cuda.is_available():
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
    th.backends.cudnn.benchmark = True

from core import (
    BASE_DIR, MODEL_PATH, DEPLOY_DIR,
    IMG_WIDTH, IMG_HEIGHT, N_LIDAR_SECTORS, LIDAR_MAX_RANGE, STATE_DIM,
    CARLA_HOST, CARLA_PORT,
    SensorSuite, SafetyShield,
)

DISPLAY_W = 900
DISPLAY_H = 600
TARGET_FPS = 60
STEER_STEP = 0.04

# HUD colors
C_BG    = (13, 17, 23, 180)
C_GREEN = (63, 185, 80)
C_RED   = (248, 81, 73)
C_WHITE = (230, 237, 243)
C_GRAY  = (139, 148, 158)
C_ORANGE = (210, 153, 34)
C_BLUE   = (88, 166, 255)


class HUD:
    def __init__(self, surface):
        self.surface = surface
        self.f_lg = pygame.font.SysFont("consolas", 20, bold=True)
        self.f_sm = pygame.font.SysFont("consolas", 14)
        self.f_xs = pygame.font.SysFont("consolas", 11)

    def _text(self, txt, x, y, color=C_WHITE, font=None):
        s = (font or self.f_sm).render(txt, True, color)
        self.surface.blit(s, (x, y))

    def draw(self, info):
        panel = pygame.Surface((300, 250), pygame.SRCALPHA)
        panel.fill(C_BG)
        self.surface.blit(panel, (10, 10))

        # Mode
        mc = C_ORANGE if info.get("mode") == "HUMAN" else C_GREEN
        self._text(f"MODE: {info.get('mode', 'AI')}", 20, 18, mc, self.f_lg)

        # Backend
        backend = info.get("backend", "PyTorch")
        bc = C_BLUE if "TRT" in backend else C_WHITE
        self._text(f"Backend: {backend}", 20, 44, bc, self.f_xs)

        self._text(f"Speed   : {info.get('speed',0):6.1f} km/h", 20, 62)
        self._text(f"Steer   : {info.get('steer',0):+.3f}", 20, 80)
        self._text(f"Throttle: {info.get('throttle',0):.3f}", 20, 98)
        self._text(f"Brake   : {info.get('brake',0):.3f}", 20, 116)

        pygame.draw.line(self.surface, C_GRAY, (20, 134), (300, 134), 1)

        self._text(f"Distance : {info.get('dist',0):7.1f} m", 20, 142)
        self._text(f"Avg Speed: {info.get('avg_speed',0):7.1f} km/h", 20, 160)

        cc = C_RED if info.get("collisions", 0) > 0 else C_GREEN
        self._text(f"Collisions: {info.get('collisions',0):5d}", 20, 178, cc)

        sc = C_RED if info.get("shield", 0) > 0 else C_GREEN
        self._text(f"Shield:     {info.get('shield',0):5d}", 20, 196, sc)

        self._text(
            f"FPS: {info.get('fps',0):.0f} | {info.get('elapsed',0):.0f}s",
            20, 222, C_GRAY, self.f_xs
        )
        self._text(
            "SPACE=Override  ESC=Quit",
            10, DISPLAY_H - 22, C_GRAY, self.f_xs
        )


def run_live_test():
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption("CARLA Grandmaster v15 — Live Eval")
    clock = pygame.time.Clock()
    hud = HUD(screen)

    # CARLA
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(15.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    # Spawn
    lib = world.get_blueprint_library()
    vehicle = world.spawn_actor(
        lib.filter("model3")[0],
        world.get_map().get_spawn_points()[0]
    )

    # Sensors (using unified SensorSuite!)
    sensors = SensorSuite()
    sensors.attach(vehicle, world)

    # Collision counter
    collision_count = [0]
    col_sensor = world.spawn_actor(
        lib.find("sensor.other.collision"),
        carla.Transform(), attach_to=vehicle
    )
    col_sensor.listen(lambda _: collision_count.__setitem__(0, collision_count[0] + 1))

    # Safety Shield
    shield = SafetyShield()

    # Load inference engine — auto-selects best backend
    backend_name = "None"
    engine = None

    # Try TensorRT / ONNX first
    try:
        from core.tensorrt_engine import load_inference_engine
        engine = load_inference_engine(DEPLOY_DIR)
        backend_name = type(engine).__name__
        print(f"[LIVE] Using {backend_name}")
    except Exception as e:
        print(f"[LIVE] No TRT/ONNX engine: {e}")

    # Fallback to SB3
    if engine is None:
        if os.path.exists(MODEL_PATH):
            from stable_baselines3 import PPO
            engine = PPO.load(MODEL_PATH, device="cuda")
            backend_name = "PyTorch (SB3)"
            print(f"[LIVE] Using SB3 PyTorch model")
        else:
            print(f"[ERROR] No model found. Run training first.")
            vehicle.destroy()
            pygame.quit()
            return

    # Main loop state
    t0 = time.time()
    dist = 0.0
    speeds = []
    prev_loc = vehicle.get_location()
    steer_k = 0.0
    last_steer = 0.0
    last_throttle = 0.0

    print("[LIVE] Running. Hold SPACE to override. ESC to quit.")

    try:
        while True:
            clock.tick(TARGET_FPS)
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (
                    e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE
                ):
                    raise KeyboardInterrupt

            keys = pygame.key.get_pressed()
            human = bool(keys[pygame.K_SPACE])
            loc = vehicle.get_location()
            v = vehicle.get_velocity()
            kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

            if human:
                if keys[pygame.K_LEFT]:
                    steer_k = max(steer_k - STEER_STEP, -1.0)
                elif keys[pygame.K_RIGHT]:
                    steer_k = min(steer_k + STEER_STEP, 1.0)
                else:
                    steer_k *= 0.85
                t_k = 1.0 if keys[pygame.K_UP] else 0.0
                b_k = 1.0 if keys[pygame.K_DOWN] else 0.0
                vehicle.apply_control(carla.VehicleControl(
                    steer=steer_k, throttle=t_k, brake=b_k
                ))
                sd, td, bd = steer_k, t_k, b_k
            else:
                cam = sensors.get_camera()
                lidar = sensors.get_lidar()

                # Build state (using correct STATE_DIM!)
                state = np.zeros(STATE_DIM, dtype=np.float32)
                state[0] = kmh / 50.0
                state[1] = last_steer
                state[2] = last_throttle

                # Predict
                if hasattr(engine, 'predict') and callable(engine.predict):
                    if hasattr(engine, 'policy'):
                        # SB3 model
                        obs = {"camera": cam, "lidar": lidar, "state": state}
                        action, _ = engine.predict(obs, deterministic=True)
                    else:
                        # TRT / ONNX engine
                        action = engine.predict(cam, lidar, state)

                    sd = float(action[0])
                    td = float(action[1])
                    bd = float(action[2]) if len(action) > 2 else 0.0

                    # Safety shield
                    sd, td, bd, _ = shield.filter_action(
                        sd, td, bd, lidar, kmh, True
                    )

                    vehicle.apply_control(carla.VehicleControl(
                        steer=sd, throttle=max(0, td), brake=bd,
                        reverse=(td < 0),
                    ))
                    last_steer, last_throttle = sd, td
                else:
                    sd = td = bd = 0.0

            # Spectator
            world.get_spectator().set_transform(carla.Transform(
                loc + carla.Location(z=45), carla.Rotation(pitch=-90)
            ))

            d = loc.distance(prev_loc)
            if d < 10:
                dist += d
            prev_loc = loc
            speeds.append(kmh)
            avg = np.mean(speeds[-200:]) if speeds else 0

            # Render
            frame = sensors.get_camera()
            surf = pygame.surfarray.make_surface(
                np.transpose(frame, (1, 0, 2))
            )
            screen.blit(pygame.transform.scale(surf, (DISPLAY_W, DISPLAY_H)), (0, 0))

            hud.draw({
                "mode": "HUMAN" if human else "AI",
                "backend": backend_name,
                "speed": kmh, "steer": sd, "throttle": td, "brake": bd,
                "dist": dist, "avg_speed": avg,
                "collisions": collision_count[0],
                "shield": shield.total_interventions,
                "fps": clock.get_fps(),
                "elapsed": time.time() - t0,
            })
            pygame.display.flip()

    except KeyboardInterrupt:
        print("\n[STOP] Live test ended.")
    finally:
        actors = sensors.detach()
        actors.extend([col_sensor, vehicle])
        for a in actors:
            try:
                a.destroy()
            except:
                pass
        pygame.quit()


if __name__ == "__main__":
    run_live_test()
