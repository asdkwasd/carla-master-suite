"""Minimal diagnostic: use default map (skip load_world), test if car moves."""
import carla, time, math

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(15.0)
world = client.get_world()
print(f"Map: {world.get_map().name}")

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

bp = world.get_blueprint_library().filter("model3")[0]
sp = world.get_map().get_spawn_points()[0]
print(f"Spawn: ({sp.location.x:.0f}, {sp.location.y:.0f}, {sp.location.z:.0f})")

vehicle = world.spawn_actor(bp, sp)
vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))

for _ in range(20):
    world.tick()

print("\nApplying throttle=0.8 for 100 ticks...")
for i in range(100):
    vehicle.apply_control(carla.VehicleControl(
        throttle=0.8, steer=0.0, brake=0.0, hand_brake=False
    ))
    world.tick()
    if i % 10 == 0:
        vel = vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2)
        loc = vehicle.get_location()
        print(f"  t={i:3d} kmh={kmh:6.1f} loc=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f})")

vel = vehicle.get_velocity()
kmh_final = 3.6 * math.sqrt(vel.x**2 + vel.y**2)
if kmh_final > 1.0:
    print(f"\nSUCCESS: Car moves at {kmh_final:.1f} km/h")
else:
    print(f"\nFAIL: Car stuck at {kmh_final:.1f} km/h")

vehicle.destroy()
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)
