"""
==============================================================================
  RTX 4080 MAXIMIZER — GPU Utilization Booster
  Target: 95%+ GPU utilization, full VRAM usage

  RTX 4080 Laptop Specs:
  - 12GB VRAM (Ada Lovelace)
  - 7424 CUDA cores
  - 16GB/s memory bandwidth

  Problem: Current training uses:
  - batch_size=256 (too small for 12GB VRAM)
  - n_steps=2048 (single env)
  - No mixed precision
  - CPU bottleneck

  Solution: Increase all GPU-bound parameters
==============================================================================
"""

import os
import sys
import torch as th

# =============================================================================
# RTX 4080 OPTIMIZATION SETTINGS
# =============================================================================

# GPU Settings
GPU_CONFIG = {
    # Memory settings
    "batch_size": 512,           # UP from 256 (12GB VRAM can handle 1024+)
    "n_steps": 4096,             # UP from 2048
    "n_envs": 4,                 # NEW: Parallel environments
    "n_epochs": 10,              # UP from 5

    # Mixed precision (FP16) - RTX 4080 Ampere/Ada supports native FP16
    "use_mixed_precision": True,

    # Gradient settings
    "max_grad_norm": 0.5,

    # Learning rate
    "learning_rate": 5e-4,

    # VRAM optimization
    "pin_memory": True,          # Faster CPU->GPU transfer
    "non_blocking": True,        # Async data transfer
}

# Check current VRAM
def check_vram():
    """Check available VRAM on RTX 4080."""
    if th.cuda.is_available():
        total = th.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = th.cuda.memory_allocated(0) / 1024**3
        reserved = th.cuda.memory_reserved(0) / 1024**3
        free = total - reserved

        print(f"\n{'='*60}")
        print(f"RTX 4080 GPU STATUS:")
        print(f"  Total VRAM:  {total:.1f} GB")
        print(f"  Allocated:   {allocated:.1f} GB")
        print(f"  Reserved:    {reserved:.1f} GB")
        print(f"  Free:       {free:.1f} GB")
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"  GPU Utilization: {util.gpu}%")
            pynvml.nvmlShutdown()
        except:
            print(f"  GPU Utilization: (pynvml not available)")
        print(f"{'='*60}\n")

        return free
    return 0

def apply_max_performance():
    """
    Apply maximum performance settings for RTX 4080.
    Call this BEFORE training starts.
    """
    print("\n" + "="*60)
    print("RTX 4080 MAXIMUM PERFORMANCE MODE")
    print("="*60)

    # 1. Check GPU
    check_vram()

    # 2. Apply CUDA optimizations
    if th.cuda.is_available():
        # TF32 for faster matmul (RTX 40xx)
        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True
        th.backends.cudnn.benchmark = True
        th.backends.cudnn.deterministic = False

        print("[GPU] TF32 enabled: Faster matmul with no accuracy loss")
        print("[GPU] cuDNN benchmark: Auto-optimized conv algorithms")

        # 3. Set memory allocator
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

        print("[GPU] Memory allocator: expandable_segments enabled")
        print("[GPU] Max split size: 512MB")

    # 4. CPU optimization (reduce fork overhead)
    if sys.platform == 'win32':
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        print("[CPU] OMP threads: 4")
        print("[CPU] MKL threads: 4")

    print("\n[READY] RTX 4080 optimized for maximum throughput")
    print(f"[CONFIG] Recommended batch_size: {GPU_CONFIG['batch_size']}")
    print(f"[CONFIG] Recommended n_steps: {GPU_CONFIG['n_steps']}")
    print(f"[CONFIG] Recommended n_envs: {GPU_CONFIG['n_envs']}")
    print("="*60 + "\n")

    return GPU_CONFIG


def estimate_optimal_batch_size():
    """
    Estimate optimal batch size for RTX 4080.
    Based on model size and VRAM.
    """
    # Model estimate: ~50MB for GrandmasterExtractor + PPO
    model_size_mb = 50

    # Sensor data: 80x60 RGB camera = 80*60*3 = 14,400 bytes per step
    # LiDAR: 16 sectors * 4 bytes = 64 bytes per step
    # State: 28 * 4 bytes = 112 bytes per step
    per_step_mb = (14400 + 64 + 112) / (1024 * 1024)  # ~0.014 MB per observation

    # Overhead for gradients, activations, etc. (rough estimate: 5x)
    overhead = 5

    # Available VRAM (use 80% to leave room)
    if th.cuda.is_available():
        available_gb = (th.cuda.get_device_properties(0).total_memory / 1024**3) * 0.7
    else:
        available_gb = 8  # Conservative estimate

    # Calculate batch size
    optimal_batch = int((available_gb * 1024) / (per_step_mb * overhead))

    # Clamp to reasonable range
    optimal_batch = max(256, min(optimal_batch, 4096))

    print(f"\n[ESTIMATE] Optimal batch size: {optimal_batch}")
    print(f"[ESTIMATE] Based on: {available_gb:.1f}GB available VRAM")

    return optimal_batch


if __name__ == "__main__":
    # Check current GPU status
    free_vram = check_vram()

    # Apply max performance
    config = apply_max_performance()

    # Estimate optimal
    estimate_optimal_batch_size()

    print("\n" + "="*60)
    print("TO APPLY OPTIMIZATIONS TO TRAINING:")
    print("="*60)
    print("1. Stop current training (Ctrl+C)")
    print("2. Run: python rtx_maximizer.py")
    print("3. Restart training with new settings")
    print("")
    print("OR add these to your training script:")
    print("  from rtx_maximizer import apply_max_performance")
    print("  apply_max_performance()")
    print("="*60 + "\n")
