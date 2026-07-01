import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.cost_function import CostFunction

cost_fn = CostFunction()
ctx = {
    "step_dist": 0.0,
    "on_road": True,
    "kmh": 0.0,
    "steer": 0.0,
    "reverse": False,
    "collisions": 0,
    "lane_invasions": 0,
    "red_light_running": False,
    "route_delta": 0.0,
    "route_waypoint_reached": False,
    "wp": None,
    "center_dist": 0.0,
    "heading_alignment": 1.0,
    "at_junction": False,
    "was_offroad": False,
    "recovery_success": False,
    "new_waypoint": False,
}

total, breakdown = cost_fn.compute(ctx)
print("Stuck Reward:", total)
print("Breakdown:", breakdown)
