"""
==============================================================================
  RTX 4080 LAUNCHER - Simple & Fast
  Patches existing training to use RTX 4080 optimized settings
==============================================================================
"""
import os
import sys
import torch as th

# Apply RTX 4080 optimizations
print("\n" + "="*70)
print(" RTX 4080 MAXIMUM PERFORMANCE MODE")
print("="*70)

if th.cuda.is_available():
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
    th.backends.cudnn.benchmark = True
    total_gb = th.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[GPU] {th.cuda.get_device_name(0)}")
    print(f"[GPU] {total_gb:.1f} GB VRAM")
    print("[GPU] TF32 + cuDNN Benchmark ENABLED")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
print("[MEM] expandable_segments ENABLED")

print("="*70 + "\n")

# Now import and run training
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Patch PPO before creating model
from stable_baselines3 import PPO

# Store original PPO.__init__
_orig_ppo_init = PPO.__init__

def _patched_ppo_init(self, *args, **kwargs):
    # Force RTX 4080 optimized settings
    kwargs.setdefault('n_steps', 8192)      # 4x increase
    kwargs.setdefault('batch_size', 1024)     # 4x increase
    kwargs.setdefault('n_epochs', 10)         # 2x increase

    # Larger network
    if 'policy_kwargs' not in kwargs:
        kwargs['policy_kwargs'] = {}
    kwargs['policy_kwargs'].setdefault('net_arch', dict(pi=[512, 256], vf=[512, 256]))

    print(f"\n[RTX4080] Patched PPO settings:")
    print(f"         n_steps: {kwargs['n_steps']}")
    print(f"         batch_size: {kwargs['batch_size']}")
    print(f"         n_epochs: {kwargs['n_epochs']}")
    print(f"         net_arch: {kwargs['policy_kwargs']['net_arch']}")

    return _orig_ppo_init(self, *args, **kwargs)

PPO.__init__ = _patched_ppo_init

# Import and run training
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", "1_TRAIN_MASTER_v15.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

print("\n[START] Running training with RTX 4080 optimizations...")
train_module.main()
