"""
Quick CARLA physics diagnostic - test if vehicle moves.
"""
import sys
sys.path.insert(0, "E:/AI/traffic_vision/master_suite")
sys.path.insert(0, "E:/AI/traffic_vision")

import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

import time, math
import numpy as np

# Test 1: Basic CARLA physics
print("=" * 60)
print("DIAGNOSTIC 1: Basic CARLA physics test")
print("=" * 60)
import carla
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(15)
world = client.get_world()

# Load Town01 (simpler)
try:
    world = client.load_world('Town01')
    print("[OK] Loaded Town01")
except Exception as e:
    print(f"[WARN] Could not load Town01: {e}")

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
print(f"[OK] Sync mode: dt={settings.fixed_delta_seconds}")

tm = client.get_trafficmanager(8000)
tm.set_synchronous_mode(True)
print("[OK] Traffic Manager sync ON")

# Spawn vehicle
bp_lib = world.get_blueprint_library()
bp = bp_lib.filter("model3")[0]
spawns = world.get_map().get_spawn_points()
spawn = spawns[0]

vehicle = world.spawn_actor(bp, spawn)
print(f"[OK] Vehicle spawned at {spawn.location}")

# Spawn camera for debugging
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "80")
cam_bp.set_attribute("image_size_y", "60")
cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)
print("[OK] Camera spawned")

# Spawn LiDAR
lid_bp = bp_lib.find("sensor.lidar.ray_cast")
lid_bp.set_attribute("channels", "32")
lid_bp.set_attribute("range", "50")
lid_bp.set_attribute("points_per_second", "200000")
lid = world.spawn_actor(lid_bp, carla.Transform(carla.Location(x=0.0, z=2.4)), attach_to=vehicle)
print("[OK] LiDAR spawned")

# Tick to initialize
for _ in range(10):
    world.tick()

print("\n[TEST] Applying full throttle for 5 seconds...")
start_loc = vehicle.get_location()
start_time = time.time()

for step in range(100):  # 5 seconds at 20 Hz
    # Full throttle, no steer
    vehicle.apply_control(carla.VehicleControl(steer=0.0, throttle=1.0, brake=0.0))
    world.tick()

    v = vehicle.get_velocity()
    speed_ms = math.sqrt(v.x**2 + v.y**2 + v.z**2)
    speed_kmh = speed_ms * 3.6

    if step % 20 == 0:
        loc = vehicle.get_location()
        dist = loc.distance(start_loc)
        elapsed = time.time() - start_time
        print(f"  Step {step}: speed={speed_kmh:.1f} km/h, dist_from_start={dist:.2f}m, time={elapsed:.1f}s")

    if speed_kmh > 1.0:
        print(f"[PASS] Car moving at {speed_kmh:.1f} km/h after {step} steps!")
        break
else:
    print("[FAIL] Car still not moving after 5 seconds!")
    loc = vehicle.get_location()
    print(f"  Final location: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")

print("\n[TEST 2] Check LiDAR data...")
lid_data = []
for _ in range(5):
    world.tick()
    v = vehicle.get_velocity()
    speed_ms = math.sqrt(v.x**2 + v.y**2 + v.z**2)
    print(f"  LiDAR check: speed={speed_ms:.2f} m/s")

print("\n[TEST 3] Check stuck detection...")
for _ in range(20):
    world.tick()
    vehicle.apply_control(carla.VehicleControl(steer=0.0, throttle=1.0, brake=0.0))
    v = vehicle.get_velocity()
    speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    loc = vehicle.get_location()
    print(f"  Speed={speed_kmh:.1f} km/h, loc=({loc.x:.1f}, {loc.y:.1f})")

# Cleanup
vehicle.destroy()
cam.destroy()
lid.destroy()
print("\n[OK] Cleanup done")
print("DIAGNOSTICS COMPLETE")
