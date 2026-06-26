"""Test env loading like 1_TRAIN_MASTER_v15.py does"""
import sys, os
sys.path.insert(0, "E:/AI/traffic_vision/master_suite")

os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

import carla
from core.constants import CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT, TM_PORT, PHYSICS_DT

print(f"Connecting to {CARLA_HOST}:{CARLA_PORT}...")
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(CARLA_TIMEOUT)
print(f"Connected! Server version: {client.get_server_version()}")

print("Loading Town01...")
world = client.load_world("Town01")
print(f"World loaded: {world.get_map().name}")

print("Setting synchronous mode...")
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = PHYSICS_DT
world.apply_settings(settings)
print(f"  dt={settings.fixed_delta_seconds}")

print("Setting up Traffic Manager...")
tm = client.get_trafficmanager(TM_PORT)
tm.set_synchronous_mode(True)
print("  TM sync ON")

print("Getting spawn points...")
spawns = world.get_map().get_spawn_points()
print(f"  {len(spawns)} spawn points")

print("Spawning vehicle...")
bp_lib = world.get_blueprint_library()
bp = bp_lib.filter("model3")[0]
spawn = spawns[0]
vehicle = world.spawn_actor(bp, spawn)
print(f"  Spawned at {spawn.location}")

print("Attaching sensors...")
from core.sensors import SensorSuite
sensors = SensorSuite()
sensors.attach(vehicle, world)
print("  Sensors attached")

print("Ticking to initialize (30 steps)...")
for i in range(30):
    world.tick()
    if i % 10 == 0:
        print(f"  Tick {i}/30")

print("Checking sensor data...")
print(f"  Camera data: {sensors.has_camera_data()}")
lidar = sensors.get_lidar()
print(f"  LiDAR: min={lidar.min():.1f}m, max={lidar.max():.1f}m")

print("\n=== TEST: Apply throttle and check movement ===")
start_loc = vehicle.get_location()
start_time = 0

for step in range(100):
    vehicle.apply_control(carla.VehicleControl(steer=0.0, throttle=1.0, brake=0.0))
    world.tick()

    v = vehicle.get_velocity()
    speed_ms = (v.x**2 + v.y**2 + v.z**2) ** 0.5
    speed_kmh = speed_ms * 3.6
    loc = vehicle.get_location()

    if step % 20 == 0:
        dist = loc.distance(start_loc)
        print(f"  Step {step}: speed={speed_kmh:.1f} km/h, dist={dist:.1f}m")

    if speed_kmh > 1.0:
        print(f"[PASS] Car moving at {speed_kmh:.1f} km/h!")
        break
else:
    print("[FAIL] Car not moving after 100 ticks!")
    print(f"  Final location: ({loc.x:.2f}, {loc.y:.2f})")

print("\n=== Environment loading test PASSED ===")
sensors.detach()
actors = [vehicle]
valid = [a for a in actors if a and hasattr(a, 'id')]
if valid:
    client.apply_batch_sync([carla.command.DestroyActor(a) for a in valid], True)
vehicle.destroy()
print("Cleanup done")
