"""
==============================================================================
  RTX 4080 OPTIMIZED TRAINING LAUNCHER
  Target: 90%+ GPU Utilization, Max Throughput

  Changes from default:
  - batch_size: 256 -> 1024 (4x)
  - n_steps: 2048 -> 8192 (4x)
  - n_epochs: 5 -> 10 (2x)
  - Larger network: 512-256
==============================================================================
"""

import os
import sys
import torch as th

# Apply RTX 4080 optimizations FIRST
print("\n" + "="*70)
print(" RTX 4080 MAXIMUM PERFORMANCE MODE ACTIVATED")
print("="*70)

# 1. Enable TF32 for RTX 40xx (Ada Lovelace)
if th.cuda.is_available():
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
    th.backends.cudnn.benchmark = True
    th.backends.cudnn.deterministic = False
    print("[GPU] TF32 Enabled: 2-3x faster matmul")
    print("[GPU] cuDNN Benchmark: Auto-optimized algorithms")
    print(f"[GPU] Device: {th.cuda.get_device_name(0)}")

    # Show VRAM
    total_gb = th.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[GPU] VRAM: {total_gb:.1f} GB")

# 2. Memory allocator optimization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
print("[GPU] Memory Allocator: expandable_segments enabled")

# 3. CPU optimization
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
print("[CPU] Threads: 4")

print("="*70 + "\n")

# Import from existing training file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_optimized_training():
    """Run training with maximum GPU utilization."""
    from 1_TRAIN_MASTER_v15 import CarlaGrandmasterEnv, CurriculumManager, Leaderboard
    from 1_TRAIN_MASTER_v15 import GrandmasterExtractor
    from 1_TRAIN_MASTER_v15 import CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT, TM_PORT
    from 1_TRAIN_MASTER_v15 import BASE_DIR, TB_DIR, CHECKPOINT_DIR, MODEL_PATH
    from 1_TRAIN_MASTER_v15 import PHYSICS_DT, MAX_EPISODE_STEPS
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from core import CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT, TM_PORT
    from core import BASE_DIR, TB_DIR, CHECKPOINT_DIR, LOG_FILE, MODEL_PATH
    from core import PHYSICS_DT, MAX_EPISODE_STEPS
    import carla

    print("\n[INIT] Connecting to CARLA...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"[CARLA] Connected: {client.get_server_version()}")

    # Setup
    curriculum = CurriculumManager(stage=1)
    config = curriculum.get_config()
    leaderboard = Leaderboard()

    print("\n[ENV] Creating CARLA environment...")
    env = CarlaGrandmasterEnv(config, curriculum, leaderboard, "v15_RTX4080")

    print("\n[MODEL] Creating RTX 4080 optimized PPO...")

    # === RTX 4080 OPTIMIZED SETTINGS ===
    model = PPO(
        "MultiInputPolicy", env, verbose=1, device="cuda",
        tensorboard_log=TB_DIR,
        policy_kwargs=dict(
            features_extractor_class=GrandmasterExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[512, 256], vf=[512, 256]),  # Larger network
        ),
        # === OPTIMIZED FOR RTX 4080 ===
        learning_rate=5e-4,
        n_steps=8192,       # 4x from default (more data per update)
        batch_size=1024,      # 4x from default (fill VRAM)
        n_epochs=10,         # 2x from default (more updates)
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    print("\n" + "="*70)
    print(" RTX 4080 OPTIMIZED CONFIG:")
    print(f"   batch_size: 1024 (4x increase)")
    print(f"   n_steps: 8192 (4x increase)")
    print(f"   n_epochs: 10 (2x increase)")
    print(f"   network: 512-256 (larger)")
    print("="*70 + "\n")

    # Check VRAM usage after model creation
    if th.cuda.is_available():
        allocated = th.cuda.memory_allocated(0) / 1024**3
        reserved = th.cuda.memory_reserved(0) / 1024**3
        print(f"[GPU] VRAM after model: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    env.set_model(model)

    # Callbacks
    ckpt_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="v15_rtx",
    )

    # Training
    total_steps = 10_000_000
    print(f"\n[TRAIN] Starting with {total_steps:,} steps...")
    print(f"[TRAIN] RTX 4080 optimized - expecting 90%+ GPU utilization")

    try:
        model.learn(
            total_timesteps=total_steps,
            callback=ckpt_cb,
            progress_bar=True,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\n[STOP] Saving emergency checkpoint...")

    # Final save
    model.save(MODEL_PATH.replace(".zip", "_rtx4080.zip"))
    env.close()

    print("\n[DONE] RTX 4080 optimized training complete!")


if __name__ == "__main__":
    print("""
    +======================================================================+
    |     RTX 4080 MAXIMUM PERFORMANCE TRAINING                            |
    |     GPU Target: 90%+ Utilization                                      |
    |     VRAM Target: 10GB+                                                |
    +======================================================================+
    """)

    run_optimized_training()
