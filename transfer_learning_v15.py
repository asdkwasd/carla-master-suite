"""
==============================================================================
  TRANSFER LEARNING PIPELINE v15 — Grandmaster AI
  Mengambil pengetahuan dari v1-v14 ke v15

  Strategy:
  1. Load best checkpoint v14 (1.76M steps) — rich spatial knowledge
  2. Load intermediate checkpoints — curriculum progression
  3. Transfer compatible weights to v15's GrandmasterExtractor
  4. Fine-tune with v15 improvements + bug fixes
  5. Curriculum-aware learning rate scheduling
==============================================================================
"""

import os
import sys
import torch as th
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Setup
CHECKPOINT_DIR = r"D:\AI\traffic_vision\master_suite\checkpoints"
BASE_DIR = r"D:\AI\traffic_vision"
V15_MODEL_PATH = os.path.join(BASE_DIR, "grandmaster_v15_transfer.zip")

# Best checkpoint from v14
BEST_V14_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "v14_1760000_steps.zip")
V14_INTERMEDIATE = os.path.join(CHECKPOINT_DIR, "v14_1000000_steps.zip")

def load_checkpoint_info(path):
    """Load and inspect checkpoint structure."""
    try:
        checkpoint = th.load(path, map_location='cpu')
        print(f"\n{'='*60}")
        print(f"CHECKPOINT: {os.path.basename(path)}")
        print(f"{'='*60}")

        # Extract info
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            print(f"Keys: {keys[:10]}...")  # First 10 keys

            # Policy network info
            if 'policy_state_dict' in checkpoint:
                sd = checkpoint['policy_state_dict']
                print(f"\nPolicy state dict ({len(sd)} tensors):")
                for k, v in list(sd.items())[:5]:
                    print(f"  {k}: {v.shape}")

            if 'policy' in checkpoint:
                print(f"\nPolicy type: {type(checkpoint['policy'])}")

            # Training info
            if 'train_step' in checkpoint:
                print(f"Train step: {checkpoint['train_step']}")
            if 'episode_count' in checkpoint:
                print(f"Episode: {checkpoint['episode_count']}")

        return checkpoint
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def extract_transferable_weights(v14_state_dict):
    """
    Extract weights from v14 that can transfer to v15.

    v15 GrandmasterExtractor structure:
    - vision.net (CNN backbone)
    - lidar_encoder.encoder (MLP)
    - state_encoder (MLP)
    - fusion (MLP)
    """
    transferable = {}
    failed = []

    for key, tensor in v14_state_dict.items():
        # Vision CNN weights
        if 'cnn' in key.lower() or 'vision' in key.lower() or 'features' in key.lower():
            new_key = key.replace('cnn.', 'vision.net.').replace('features.', 'vision.net.')
            transferable[new_key] = tensor

        # LiDAR encoder
        elif 'lidar' in key.lower() or 'spatial' in key.lower():
            new_key = key.replace('lidar_encoder.', 'lidar_encoder.encoder.')
            transferable[new_key] = tensor

        # State encoder
        elif 'state' in key.lower() or 'mlp' in key.lower():
            new_key = key.replace('mlp.', 'state_encoder.')
            transferable[new_key] = tensor

        # Value network
        elif 'value' in key.lower() or 'vf' in key.lower():
            transferable[key] = tensor

        # Action network (pi)
        elif 'pi' in key.lower() or 'action' in key.lower():
            transferable[key] = tensor

        else:
            failed.append(key)

    return transferable, failed

def create_transfer_report(checkpoint_path, report_path):
    """Create detailed transfer learning report."""
    print(f"\n[TRANSFER] Analyzing checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint_info(checkpoint_path)
    if checkpoint is None:
        return False

    # Extract state dict
    if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
        sd = checkpoint['policy_state_dict']
    elif isinstance(checkpoint, dict) and 'policy' in checkpoint:
        # SB3 PPO model
        sd = checkpoint['policy'].state_dict() if hasattr(checkpoint['policy'], 'state_dict') else {}
    else:
        sd = checkpoint if isinstance(checkpoint, dict) else {}

    transferable, failed = extract_transferable_weights(sd)

    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        "total_parameters": len(sd) if sd else 0,
        "transferable": len(transferable),
        "failed": len(failed),
        "failed_keys": failed[:20],  # First 20 failed
        "transferable_preview": {k: str(v.shape) for k, v in list(transferable.items())[:10]}
    }

    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[TRANSFER REPORT]")
    print(f"  Total params: {report['total_parameters']}")
    print(f"  Transferable: {report['transferable']}")
    print(f"  Failed: {report['failed']}")
    print(f"  Report saved: {report_path}")

    return True

def analyze_all_versions():
    """Analyze all checkpoints from v1-v14 for transfer readiness."""
    print("\n" + "="*70)
    print("ANALYZING ALL CHECKPOINTS FOR TRANSFER LEARNING")
    print("="*70)

    checkpoints = []

    # v14 checkpoints (most valuable)
    v14_files = list(Path(CHECKPOINT_DIR).glob("v14_*.zip"))
    if v14_files:
        # Get latest v14
        latest_v14 = sorted(v14_files, key=lambda x: x.stat().st_mtime)[-1]
        checkpoints.append(("v14 (latest)", str(latest_v14)))

        # Get milestone checkpoints
        milestones = ["500k", "1000k", "1500k"]
        for m in milestones:
            matches = [f for f in v14_files if m in f.name]
            if matches:
                checkpoints.append((f"v14 ({m})", str(matches[0])))

    # v11 checkpoint
    v11_file = Path(CHECKPOINT_DIR) / "grandmaster_v11_10000_steps.zip"
    if v11_file.exists():
        checkpoints.append(("v11 (baseline)", str(v11_file)))

    # Analyze each
    reports = []
    for name, path in checkpoints:
        print(f"\n>>> Analyzing: {name}")
        report_path = os.path.join(BASE_DIR, f"transfer_report_{name.replace(' ', '_').replace('(', '').replace(')', '')}.json")
        if create_transfer_report(path, report_path):
            reports.append(report_path)

    # Summary
    print("\n" + "="*70)
    print("TRANSFER LEARNING SUMMARY")
    print("="*70)

    for report_path in reports:
        with open(report_path) as f:
            r = json.load(f)
            print(f"\n{r['checkpoint'].split('\\')[-1]}:")
            print(f"  Transferable: {r['transferable']}/{r['total_parameters']} params")

    return reports

def prepare_transfer_training():
    """
    Create optimized training script that uses transfer learning.
    """
    print("\n" + "="*70)
    print("PREPARING TRANSFER LEARNING TRAINING")
    print("="*70)

    # Find best checkpoint
    best_checkpoint = BEST_V14_CHECKPOINT
    if not os.path.exists(best_checkpoint):
        # Fall back to any v14 checkpoint
        v14_files = list(Path(CHECKPOINT_DIR).glob("v14_*.zip"))
        if v14_files:
            best_checkpoint = str(sorted(v14_files, key=lambda x: x.stat().st_mtime)[-1])
            print(f"[FALLBACK] Using latest v14: {best_checkpoint}")

    if not os.path.exists(best_checkpoint):
        print("[ERROR] No v14 checkpoint found!")
        return False

    print(f"\n[INFO] Best checkpoint: {best_checkpoint}")
    print(f"[INFO] Output model: {V15_MODEL_PATH}")

    # Create training config
    config = {
        "transfer_learning": {
            "enabled": True,
            "source_checkpoint": best_checkpoint,
            "target_model": V15_MODEL_PATH,
            "strategy": "fine_tune_with_warmup",
            "warmup_steps": 10000,
            "initial_lr": 1e-5,  # Lower LR for fine-tuning
            "final_lr": 3e-4,
        },
        "architecture": {
            "feature_extractor": "GrandmasterExtractor",
            "policy": "CnnPolicy",
            "features_dim": 256
        },
        "v15_improvements": [
            "Fixed action space (throttle/brake conflict)",
            "SafetyShield TTC thresholds",
            "Anti-stall one-time nudge",
            "No watchdog crashes",
            "Better reward shaping"
        ]
    }

    config_path = os.path.join(BASE_DIR, "transfer_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n[CONFIG] Saved to: {config_path}")
    print("\n[NEXT STEPS]")
    print("1. Run: python 1_TRAIN_MASTER_v15.py --transfer")
    print("2. The training will load v14 weights and fine-tune")
    print("3. Expected improvement: Faster convergence, better base behavior")

    return True

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║            CARLA GRANDMASTER AI — TRANSFER LEARNING                  ║
    ║           Knowledge Transfer from v1-v14 to v15                     ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Analyze all checkpoints
    analyze_all_versions()

    # Step 2: Prepare training config
    prepare_transfer_training()

    print("\n[DONE] Transfer learning analysis complete!")
