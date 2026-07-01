"""
==============================================================================
  Expert Data Collector (ATM Pre-training)
  Runs the CarlaGrandmasterEnv (v16) with Traffic Manager autopilot.
  Records (state, action) pairs into JSON for Behavioral Cloning.
==============================================================================
"""
import os
import time
import json
import uuid
import carla
from tqdm import tqdm

from core.constants import HUMAN_DATA_DIR
from core.curriculum import CurriculumManager
from core.leaderboard import Leaderboard

import sys
import sys

def collect_data(episodes=5, max_steps=2000):
    try:
        from iter_v16 import CarlaGrandmasterEnv
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("v16", "1_TRAIN_MASTER_v16.py")
        v16 = importlib.util.module_from_spec(spec)
        sys.modules["v16"] = v16
        spec.loader.exec_module(v16)
        CarlaGrandmasterEnv = v16.CarlaGrandmasterEnv

    os.makedirs(HUMAN_DATA_DIR, exist_ok=True)
    
    curriculum = CurriculumManager(stage=1)
    env = CarlaGrandmasterEnv(curriculum.get_config(), curriculum, Leaderboard(), "DataCollector")
    
    print("======================================================")
    print("  ATM Data Collector: Recording Expert Demonstrations ")
    print("======================================================")

    for ep in range(episodes):
        obs, _ = env.reset()
        
        # Enable Traffic Manager Autopilot
        env.vehicle.set_autopilot(True, env.tm.get_port() if env.tm else 8000)
        
        # Configure TM for safe, expert-like driving
        if env.tm:
            env.tm.ignore_lights_percentage(env.vehicle, 0)
            env.tm.ignore_signs_percentage(env.vehicle, 0)
            env.tm.distance_to_leading_vehicle(env.vehicle, 5.0)
            env.tm.vehicle_percentage_speed_difference(env.vehicle, -10.0) # Drive 10% faster than limit
            
        dataset = {"episode": ep, "steps": []}
        
        print(f"\n[EPISODE {ep+1}/{episodes}] Recording...")
        
        for step in tqdm(range(max_steps)):
            # Step environment with dummy action (TM is driving anyway)
            # We must pass valid action array shape
            dummy_action = [0.0, 0.0, 0.0] 
            next_obs, reward, done, truncated, info = env.step(dummy_action)
            
            # Read actual applied control from TM
            ctrl = env.vehicle.get_control()
            actual_action = [ctrl.steer, ctrl.throttle, ctrl.brake]
            
            # Record state (before action) and the expert action
            dataset["steps"].append({
                "state": obs["state"].tolist(),
                "action": actual_action
            })
            
            obs = next_obs
            
            # If stuck or collision, end recording for this episode early
            if env.stuck_counter > 50 or env.collision_count > 0:
                print(f"Ending episode early (Stuck/Collision). Steps: {step}")
                break
                
        # Save episode data
        file_id = str(uuid.uuid4())[:8]
        filepath = os.path.join(HUMAN_DATA_DIR, f"expert_{file_id}.json")
        with open(filepath, 'w') as f:
            json.dump(dataset, f)
            
        print(f"Saved {len(dataset['steps'])} frames to {filepath}")

    print("\n[COLLECTION COMPLETE]")

if __name__ == "__main__":
    # Ensure this is only run when training is NOT running, 
    # to avoid CARLA synchronous mode tick collisions.
    print("NOTE: Do not run this concurrently with 1_TRAIN_MASTER if both use sync mode!")
    collect_data(episodes=5)
