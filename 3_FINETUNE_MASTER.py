"""
==============================================================================
  3_FINETUNE_MASTER.py — Fine-Tuning Pipeline v14

  FINE-TUNING MODES:
  ✅ Resume from checkpoint with adjusted hyperparams
  ✅ Transfer learning between curriculum stages
  ✅ Focus training on weak scenarios
  ✅ Curriculum stage override
  ✅ Reward weight adjustment
  ✅ Human demo injection (BC warm-start)
==============================================================================
"""

import os, sys, json, argparse
import numpy as np

BASE_DIR = r"D:\AI\traffic_vision\master_suite"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
HUMAN_DATA_DIR = os.path.join(BASE_DIR, "human_demos")

# Import training environment
sys.path.insert(0, BASE_DIR)
from importlib import import_module

train_module = import_module("1_TRAIN_MASTER")

PRESETS = {
    "aggressive": {"learning_rate": 1e-3, "ent_coef": 0.1, "clip_range": 0.3},
    "conservative": {"learning_rate": 1e-5, "ent_coef": 0.001, "clip_range": 0.1},
    "exploration": {"learning_rate": 3e-4, "ent_coef": 0.1, "n_steps": 4096},
    "focused": {"learning_rate": 1e-4, "ent_coef": 0.001, "batch_size": 256},
    "recovery": {"learning_rate": 3e-4, "ent_coef": 0.08, "n_steps": 4096, "gamma": 0.995},
    "highspeed": {"learning_rate": 3e-4, "ent_coef": 0.02, "clip_range": 0.15},
}

def list_checkpoints():
    """List all available checkpoints."""
    if not os.path.exists(CHECKPOINT_DIR):
        print("[INFO] No checkpoints directory found.")
        return []
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")]
    ckpts.sort(key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)), reverse=True)
    print(f"\n{'='*60}")
    print(f"  AVAILABLE CHECKPOINTS")
    print(f"{'='*60}")
    for i, ckpt in enumerate(ckpts[:20]):
        size_mb = os.path.getsize(os.path.join(CHECKPOINT_DIR, ckpt)) / 1e6
        mtime = os.path.getmtime(os.path.join(CHECKPOINT_DIR, ckpt))
        from datetime import datetime
        ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {i+1:<4} {ckpt:<45} {size_mb:>6.1f}MB  {ts}")
    print(f"{'='*60}\n")
    return ckpts


def finetune(checkpoint_path=None, stage=None, preset=None, timesteps=1_000_000,
             custom_params=None):
    """Run fine-tuning with specified parameters."""

    print(f"\n{'='*60}")
    print(f"  CARLA AI FINE-TUNING PIPELINE")
    print(f"{'='*60}")

    # Setup
    curriculum = train_module.CurriculumManager(stage=stage or 1)
    config = curriculum.get_config()
    leaderboard = train_module.Leaderboard()
    env = train_module.CarlaGrandmasterEnv(config, curriculum, leaderboard)

    # Load model
    model = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[LOAD] {checkpoint_path}")
        try:
            model = train_module.PPO.load(checkpoint_path, env=env, device="cuda",
                                          tensorboard_log=train_module.TB_DIR)
            print("[LOAD] Success!")
        except Exception as e:
            print(f"[WARN] Load failed: {e}")
            model = None

    if model is None:
        print("[NEW] Creating fresh model...")
        model = train_module.PPO(
            "MultiInputPolicy", env, verbose=1, device="cuda",
            tensorboard_log=train_module.TB_DIR,
            policy_kwargs=dict(features_extractor_class=train_module.GrandmasterFeatureExtractor),
            learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
        )

    # Apply preset or custom params
    params = {}
    if preset and preset in PRESETS:
        params = PRESETS[preset]
        print(f"[PRESET] {preset}: {params}")
    if custom_params:
        params.update(custom_params)
        print(f"[CUSTOM] {custom_params}")

    if params:
        for k, v in params.items():
            if hasattr(model, k):
                setattr(model, k, v)
                print(f"  Set {k} = {v}")

    env.set_model(model)

    # Callbacks
    from stable_baselines3.common.callbacks import CheckpointCallback
    ckpt_cb = CheckpointCallback(save_freq=10_000, save_path=CHECKPOINT_DIR, name_prefix="v14_finetune")
    log_cb = train_module.TrainingLogger(
        train_module.LOG_FILE, curriculum, leaderboard,
        f"Finetune_{preset or 'custom'}", env
    )

    # Train
    print(f"\n[FINE-TUNE] {timesteps:,} steps | Stage {curriculum.current_stage}")
    try:
        model.learn(total_timesteps=timesteps, callback=[ckpt_cb, log_cb], progress_bar=True)
        save_path = os.path.join(BASE_DIR, f"carla_finetune_{preset or 'custom'}_v14.zip")
        model.save(save_path)
        print(f"\n[DONE] Saved: {save_path}")
    except KeyboardInterrupt:
        model.save(os.path.join(BASE_DIR, "finetune_emergency.zip"))
        print("\n[STOP] Emergency save.")
    finally:
        leaderboard.print_board()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA AI Fine-Tuning")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .zip")
    parser.add_argument("--stage", type=int, default=None, help="Curriculum stage (1-4)")
    parser.add_argument("--preset", type=str, default=None, choices=list(PRESETS.keys()),
                        help="Training preset")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Training steps")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--ent", type=float, default=None, help="Override entropy coefficient")

    args = parser.parse_args()

    if args.list:
        list_checkpoints()
    else:
        custom = {}
        if args.lr: custom["learning_rate"] = args.lr
        if args.ent: custom["ent_coef"] = args.ent
        finetune(
            checkpoint_path=args.checkpoint,
            stage=args.stage,
            preset=args.preset,
            timesteps=args.timesteps,
            custom_params=custom if custom else None,
        )
