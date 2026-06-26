"""
PERIODIC TRANSFER - Run this every hour during training
to keep v15 fresh with v14 knowledge
"""
import os
import sys
import glob
from pathlib import Path

CHECKPOINT_DIR = r"D:\AI\traffic_vision\master_suite\checkpoints"
BASE_DIR = r"D:\AI\traffic_vision"

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
