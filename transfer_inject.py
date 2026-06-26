"""
==============================================================================
  TRANSFER LEARNING CALLBACK v15 — On-The-Fly Knowledge Injection
  Transfers v14 knowledge WITHOUT stopping current training

  This can be run while training is active to boost the model with v14
  knowledge. Safe to use at any time - uses soft transfer (blending).
==============================================================================
"""

import os
import sys
import torch as th
import numpy as np
from pathlib import Path

# Setup
CHECKPOINT_DIR = r"D:\AI\traffic_vision\master_suite\checkpoints"
BASE_DIR = r"D:\AI\traffic_vision"

# Best v14 checkpoint
V14_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "v14_1760000_steps.zip")


def soft_transfer(source_state_dict, target_state_dict, alpha=0.7):
    """
    Soft transfer: blend source and target weights.
    new_weight = alpha * source + (1-alpha) * target

    This is safer than hard transfer because:
    - Doesn't completely overwrite current learning
    - Preserves recent v15 improvements
    - Gradually incorporates v14 knowledge
    """
    blended = {}
    for key in target_state_dict:
        if key in source_state_dict:
            if target_state_dict[key].shape == source_state_dict[key].shape:
                if target_state_dict[key].dtype in [th.float32, th.float64]:
                    # Blend the weights
                    blended[key] = alpha * source_state_dict[key].float() + \
                                   (1 - alpha) * target_state_dict[key].float()
                else:
                    blended[key] = target_state_dict[key]
            else:
                blended[key] = target_state_dict[key]
        else:
            blended[key] = target_state_dict[key]
    return blended


def inject_v14_knowledge(model_path, v14_path=None, alpha=0.5, output_path=None):
    """
    Inject v14 knowledge into a v15 model checkpoint.

    Args:
        model_path: Path to current v15 model checkpoint
        v14_path: Path to v14 checkpoint (auto-detected if None)
        alpha: Blend factor (0.7 = 70% v14, 30% v15)
        output_path: Save to new path (overwrites if None)

    Returns:
        Number of layers transferred
    """
    import io
    import zipfile
    import torch as th

    if v14_path is None:
        v14_path = V14_CHECKPOINT

    if not os.path.exists(v14_path):
        print(f"[TRANSFER] v14 checkpoint not found: {v14_path}")
        return 0

    if not os.path.exists(model_path):
        print(f"[TRANSFER] Model checkpoint not found: {model_path}")
        return 0

    print(f"\n{'='*60}")
    print(f"[TRANSFER] Injecting v14 knowledge into current model")
    print(f"[TRANSFER] v14 source: {os.path.basename(v14_path)}")
    print(f"[TRANSFER] v15 target: {os.path.basename(model_path)}")
    print(f"[TRANSFER] Blend factor: alpha={alpha}")
    print(f"{'='*60}")

    # Load v14 policy weights directly from zip
    print("[TRANSFER] Loading v14 checkpoint...")
    with zipfile.ZipFile(v14_path, 'r') as z:
        with z.open('policy.pth') as f:
            v14_sd = th.load(io.BytesIO(f.read()), map_location='cpu', weights_only=False)

    # Name mapping from v14 to v15
    name_mapping = {
        'features_extractor.vision.cnn': 'features_extractor.vision.net',
        'features_extractor.vision.': 'features_extractor.vision.net.',
    }

    # Map v14 names to v15 names
    v14_sd_mapped = {}
    for key, value in v14_sd.items():
        new_key = key
        for old, new in name_mapping.items():
            if old in new_key:
                new_key = new_key.replace(old, new)
        v14_sd_mapped[new_key] = value

    print(f"[TRANSFER] v14 state dict (mapped): {len(v14_sd_mapped)} parameters")

    # Load v15 policy weights directly from zip
    print("[TRANSFER] Loading v15 checkpoint...")
    with zipfile.ZipFile(model_path, 'r') as z:
        with z.open('policy.pth') as f:
            v15_sd = th.load(io.BytesIO(f.read()), map_location='cpu', weights_only=False)

    print(f"[TRANSFER] v15 state dict: {len(v15_sd)} parameters")

    # Perform soft transfer
    print("[TRANSFER] Blending weights...")
    blended_sd = soft_transfer(v14_sd_mapped, v15_sd, alpha=alpha)

    # Count matches
    matched = sum(1 for k in v15_sd if k in v14_sd_mapped and v15_sd[k].shape == v14_sd_mapped.get(k, v15_sd[k]).shape)
    total = len(v15_sd)

    print(f"[TRANSFER] Matched: {matched}/{total} layers")

    # Save updated zip file
    if output_path is None:
        output_path = model_path  # Overwrite

    print(f"[TRANSFER] Saving to: {os.path.basename(output_path)}")

    # Copy original zip and replace policy.pth
    import shutil
    temp_path = output_path + '.tmp'
    shutil.copy(model_path, temp_path)

    # Update the zip with new policy
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as z_out:
        with zipfile.ZipFile(temp_path, 'r') as z_in:
            for item in z_in.infolist():
                if item.filename == 'policy.pth':
                    # Write blended weights
                    buf = io.BytesIO()
                    th.save(blended_sd, buf, _use_new_zipfile_serialization=True)
                    z_out.writestr(item, buf.getvalue())
                else:
                    # Copy original
                    z_out.writestr(item, z_in.read(item.filename))

    # Clean up
    os.remove(temp_path)

    print(f"\n{'='*60}")
    print(f"[TRANSFER] Complete!")
    print(f"[TRANSFER] v14 knowledge injected successfully")
    print(f"[TRANSFER] alpha={alpha} means {int(alpha*100)}pct v14, {int((1-alpha)*100)}pct v15 preserved")
    print(f"{'='*60}\n")

    return matched


def create_emergency_rescue():
    """
    Create a script that can rescue training by injecting v14 knowledge
    if training gets stuck or reward drops.
    """
    script = '''@echo off
chcp 65001 >nul
echo.
echo +======================================================================+
echo +            EMERGENCY RESCUE - Inject v14 Knowledge                +
echo +======================================================================+
echo.
echo This will inject v14 knowledge into the latest v15 checkpoint
echo WITHOUT stopping current training!
echo.

cd /d D:\\AI\\traffic_vision\\master_suite

REM Find latest v15 checkpoint
for /f "tokens=*" %%f in ('dir /b /o-n checkpoints\\\\v15_*.zip ^| head -1') do set LATEST=%%f

echo Latest v15 checkpoint: %LATEST%
echo.
echo Press any key to inject v14 knowledge...
pause >nul

REM Run transfer injection
call ..\\venv\\Scripts\\activate.bat
python -c "
import sys
sys.path.insert(0, '.')
from transfer_inject import inject_v14_knowledge
inject_v14_knowledge(
    'checkpoints\\\\%LATEST%',
    alpha=0.6  # 60%% v14, 40%% preserved v15
)
"

echo.
echo [DONE] v14 knowledge injected!
echo [TIP] Training will automatically use new weights on next checkpoint load
echo.
pause
'''

    with open(os.path.join(BASE_DIR, "master_suite", "EMERGENCY_RESCUE.bat"), 'w') as f:
        f.write(script)

    print("[RESCUE] Emergency rescue script created: EMERGENCY_RESCUE.bat")


def create_periodic_transfer_script():
    """
    Create a script that runs transfer learning periodically
    to keep v15 model refreshed with v14 knowledge.
    """
    script = '''"""
PERIODIC TRANSFER - Run this every hour during training
to keep v15 fresh with v14 knowledge
"""
import os
import sys
import glob
from pathlib import Path

CHECKPOINT_DIR = r"D:\\AI\\traffic_vision\\master_suite\\checkpoints"
BASE_DIR = r"D:\\AI\\traffic_vision"

# Find latest v15 checkpoint
v15_files = glob.glob(os.path.join(CHECKPOINT_DIR, "v15_*.zip"))
if not v15_files:
    print("[PERIODIC] No v15 checkpoints found")
    sys.exit(0)

latest_v15 = max(v15_files, key=lambda x: os.path.getmtime(x))
print(f"[PERIODIC] Latest v15: {os.path.basename(latest_v15)}")

# Import and run transfer
sys.path.insert(0, os.path.join(BASE_DIR, "master_suite"))
from transfer_inject import inject_v14_knowledge

# Soft transfer with alpha=0.4 (40pct v14, 60pct v15 preserved)
transferred = inject_v14_knowledge(
    latest_v15,
    alpha=0.4,
    output_path=latest_v15  # Overwrite with blended weights
)

if transferred > 0:
    print(f"[PERIODIC] Transferred {transferred} layers from v14")
else:
    print("[PERIODIC] Transfer failed")
'''

    with open(os.path.join(BASE_DIR, "master_suite", "periodic_transfer.py"), 'w') as f:
        f.write(script)

    print("[PERIODIC] Periodic transfer script created: periodic_transfer.py")


if __name__ == "__main__":
    print("""
    +======================================================================+
    |          TRANSFER LEARNING INJECTOR - v14 to v15                     |
    |                                                                      |
    |  This tool injects v14 knowledge into current v15 checkpoint        |
    |  WITHOUT stopping training!                                          |
    |                                                                      |
    |  Usage:                                                              |
    |    python transfer_inject.py                    # Auto-detect latest|
    |    python transfer_inject.py --checkpoint=X      # Specific checkpoint|
    |    python transfer_inject.py --alpha=0.7         # More v14 influence |
    +======================================================================+
    """)

    import argparse
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--create-scripts', action='store_true')
    args = parser.parse_args()

    if args.create_scripts:
        create_emergency_rescue()
        create_periodic_transfer_script()
        print("\n[INFO] Rescue scripts created!")
        print("  - EMERGENCY_RESCUE.bat: Run when training gets stuck")
        print("  - periodic_transfer.py: Run every hour during training")
    else:
        # Find latest v15 checkpoint
        if args.checkpoint:
            model_path = args.checkpoint
        else:
            v15_files = glob.glob(os.path.join(CHECKPOINT_DIR, "v15_*.zip"))
            if not v15_files:
                print("[ERROR] No v15 checkpoints found!")
                sys.exit(1)
            model_path = max(v15_files, key=lambda x: os.path.getmtime(x))

        # Run transfer
        inject_v14_knowledge(
            model_path,
            alpha=args.alpha,
            output_path=model_path
        )
