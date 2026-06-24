"""
==============================================================================
  4_DEPLOY_MASTER.py — v15 Deployment Pipeline

  Full pipeline: Validate → ONNX → TensorRT FP16/INT8 → Package
  Uses core.tensorrt_engine for all export logic.
==============================================================================
"""
import os
import sys
import json
import time
import argparse
import numpy as np

# ── Apply GPU optimizations ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.constants import apply_gpu_opts
apply_gpu_opts()

from core import (
    BASE_DIR, MODEL_PATH, DEPLOY_DIR, REPORT_DIR,
    IMG_WIDTH, IMG_HEIGHT, N_LIDAR_SECTORS, LIDAR_MAX_RANGE, STATE_DIM,
    CurriculumManager, Leaderboard,
)
from core.tensorrt_engine import (
    ONNXExporter, TensorRTBuilder, load_inference_engine,
    full_export_pipeline, TRT_AVAILABLE, ORT_AVAILABLE,
)


# =============================================================================
# MODEL VALIDATOR
# =============================================================================
class ModelValidator:
    """Run evaluation episodes to validate model quality."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def validate(self, n_episodes: int = 10, stage: int = 1):
        print(f"\n[VALIDATE] Running {n_episodes} eval episodes...")
        print(f"[VALIDATE] Model: {self.model_path}")

        from stable_baselines3 import PPO
        from core.feature_extractor import GrandmasterExtractor

        # Create env
        curriculum = CurriculumManager(stage=stage)
        config = curriculum.get_config()
        leaderboard = Leaderboard()

        # Import environment from trainer
        try:
            from importlib import import_module
            tm = import_module("1_TRAIN_MASTER_v15")
            env = tm.CarlaGrandmasterEnv(
                config, curriculum, leaderboard, "Validator"
            )
        except ImportError:
            print("[ERROR] Cannot import 1_TRAIN_MASTER_v15.py")
            return None

        try:
            model = PPO.load(self.model_path, env=env, device="cpu")
        except Exception as e:
            print(f"[ERROR] Cannot load model: {e}")
            env.close()
            return None

        results = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            total_dist = 0
            collisions = 0
            steps = 0
            done = False

            while not done and steps < 10000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                total_dist = info.get("total_distance", total_dist)
                if info.get("termination") == "collision":
                    collisions += 1
                steps += 1

            results.append({
                "episode": ep + 1,
                "reward": float(total_reward),
                "distance": float(total_dist),
                "collisions": collisions,
                "steps": steps,
            })
            status = "PASS" if collisions == 0 and total_dist > 100 else "FAIL"
            print(f"  EP {ep+1}: reward={total_reward:.0f} "
                  f"dist={total_dist:.0f}m col={collisions} [{status}]")

        env.close()

        summary = {
            "mean_reward": float(np.mean([r["reward"] for r in results])),
            "mean_distance": float(np.mean([r["distance"] for r in results])),
            "total_collisions": sum(r["collisions"] for r in results),
            "pass_rate": sum(1 for r in results
                           if r["collisions"] == 0 and r["distance"] > 100) / len(results) * 100,
            "episodes": results,
        }
        print(f"\n[RESULTS] Mean reward: {summary['mean_reward']:.0f}")
        print(f"[RESULTS] Mean distance: {summary['mean_distance']:.0f}m")
        print(f"[RESULTS] Pass rate: {summary['pass_rate']:.0f}%")
        return summary


# =============================================================================
# BENCHMARK — Compare SB3 vs ONNX vs TensorRT inference speed
# =============================================================================
def benchmark(model_path: str, n_iters: int = 100):
    """Compare inference speeds across backends."""
    print(f"\n{'=' * 60}")
    print(f"  INFERENCE BENCHMARK ({n_iters} iterations)")
    print(f"{'=' * 60}\n")

    # Dummy inputs
    cam = np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    lidar = np.random.rand(N_LIDAR_SECTORS).astype(np.float32) * LIDAR_MAX_RANGE
    state = np.random.randn(STATE_DIM).astype(np.float32)

    results = {}

    # 1. SB3 (PyTorch)
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cuda")
        obs = {
            "camera": cam[np.newaxis],
            "lidar": lidar[np.newaxis],
            "state": state[np.newaxis],
        }
        # Warmup
        for _ in range(10):
            model.predict(obs, deterministic=True)
        t0 = time.time()
        for _ in range(n_iters):
            model.predict(obs, deterministic=True)
        dt = (time.time() - t0) / n_iters * 1000
        results["PyTorch (SB3)"] = dt
        print(f"  PyTorch (SB3):  {dt:.2f} ms/inference")
    except Exception as e:
        print(f"  PyTorch: SKIP ({e})")

    # 2. ONNX Runtime
    try:
        engine = load_inference_engine(DEPLOY_DIR)
        if "ONNX" in type(engine).__name__:
            for _ in range(10):
                engine.predict(cam, lidar, state)
            t0 = time.time()
            for _ in range(n_iters):
                engine.predict(cam, lidar, state)
            dt = (time.time() - t0) / n_iters * 1000
            results["ONNX Runtime"] = dt
            print(f"  ONNX Runtime:   {dt:.2f} ms/inference")
    except Exception as e:
        print(f"  ONNX Runtime: SKIP ({e})")

    # 3. TensorRT
    try:
        from core.tensorrt_engine import TensorRTInference
        for prec in ["fp16", "int8"]:
            engine_path = os.path.join(DEPLOY_DIR, f"carla_v15_{prec}.engine")
            if os.path.exists(engine_path):
                eng = TensorRTInference(engine_path)
                for _ in range(10):
                    eng.predict(cam, lidar, state)
                t0 = time.time()
                for _ in range(n_iters):
                    eng.predict(cam, lidar, state)
                dt = (time.time() - t0) / n_iters * 1000
                results[f"TensorRT {prec.upper()}"] = dt
                print(f"  TensorRT {prec.upper()}: {dt:.2f} ms/inference")
    except Exception as e:
        print(f"  TensorRT: SKIP ({e})")

    print(f"\n{'=' * 60}")
    if results:
        fastest = min(results, key=results.get)
        print(f"  FASTEST: {fastest} ({results[fastest]:.2f} ms)")
    print(f"{'=' * 60}\n")
    return results


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CARLA AI Deployment Pipeline v15"
    )
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--action",
                        choices=["validate", "export", "benchmark", "all"],
                        default="all")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--stage", type=int, default=1)

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  CARLA AI DEPLOYMENT PIPELINE v15")
    print(f"  TensorRT: {'AVAILABLE' if TRT_AVAILABLE else 'NOT INSTALLED'}")
    print(f"  ONNX Runtime: {'AVAILABLE' if ORT_AVAILABLE else 'NOT INSTALLED'}")
    print(f"{'=' * 60}\n")

    # Validate
    if args.action in ["validate", "all"]:
        validator = ModelValidator(args.model)
        results = validator.validate(args.episodes, args.stage)
        if results:
            report_path = os.path.join(REPORT_DIR, "validation_v15.json")
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"[SAVED] {report_path}")

    # Export
    if args.action in ["export", "all"]:
        from stable_baselines3 import PPO
        if os.path.exists(args.model):
            model = PPO.load(args.model, device="cpu")
            full_export_pipeline(model)
        else:
            print(f"[ERROR] Model not found: {args.model}")

    # Benchmark
    if args.action in ["benchmark", "all"]:
        benchmark(args.model)

    print(f"\n{'=' * 60}")
    print(f"  DEPLOYMENT PIPELINE COMPLETE")
    print(f"{'=' * 60}\n")
