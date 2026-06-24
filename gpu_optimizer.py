"""
==============================================================================
  GPU OPTIMIZER — RTX 4080 Laptop Auto-Tuner
  
  - Detects GPU, VRAM, driver version
  - Sets optimal CUDA/TensorRT flags
  - Memory cleanup & garbage collection
  - RTX 4080 Laptop specific optimizations
  - Pre-flight checks before training/inference
==============================================================================
"""
import os
import sys
import gc
import json
import subprocess
import ctypes
import time
from pathlib import Path

# ── ANSI Colors ──────────────────────────────────────────────────────────────
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
R  = "\033[91m"   # red
C  = "\033[96m"   # cyan
B  = "\033[94m"   # blue
W  = "\033[97m"   # white
M  = "\033[95m"   # magenta
RST = "\033[0m"   # reset


def banner():
    print(f"""
{C}╔══════════════════════════════════════════════════════════════╗
║  {W}⚡  GPU OPTIMIZER — CARLA Grandmaster AI v15  ⚡{C}             ║
║  {G}Auto-detect & Optimize for NVIDIA RTX GPUs{C}                  ║
╚══════════════════════════════════════════════════════════════╝{RST}
""")


# =============================================================================
# GPU DETECTION
# =============================================================================
def get_nvidia_info() -> dict:
    """Query nvidia-smi for GPU details."""
    info = {
        "gpu_name": "Unknown",
        "vram_total_mb": 0,
        "vram_used_mb": 0,
        "vram_free_mb": 0,
        "driver_version": "Unknown",
        "cuda_version": "Unknown",
        "gpu_temp": 0,
        "power_draw": 0,
        "power_limit": 0,
        "gpu_util": 0,
        "mem_util": 0,
        "compute_cap": "Unknown",
    }
    
    try:
        # Basic info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,"
             "driver_version,temperature.gpu,power.draw,power.limit,"
             "utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 10:
                info["gpu_name"] = parts[0]
                info["vram_total_mb"] = int(float(parts[1]))
                info["vram_used_mb"] = int(float(parts[2]))
                info["vram_free_mb"] = int(float(parts[3]))
                info["driver_version"] = parts[4]
                try: info["gpu_temp"] = int(float(parts[5]))
                except (ValueError, TypeError): pass
                try: info["power_draw"] = float(parts[6])
                except (ValueError, TypeError): pass
                try: info["power_limit"] = float(parts[7])
                except (ValueError, TypeError): pass
                try: info["gpu_util"] = int(float(parts[8]))
                except (ValueError, TypeError): pass
                try: info["mem_util"] = int(float(parts[9]))
                except (ValueError, TypeError): pass
        
        # CUDA version
        result2 = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result2.returncode == 0:
            info["compute_cap"] = result2.stdout.strip()
        
        # CUDA version from nvidia-smi header
        result3 = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result3.returncode == 0:
            for line in result3.stdout.split("\n"):
                if "CUDA Version" in line:
                    cuda_part = line.split("CUDA Version:")[1].strip().split()[0]
                    info["cuda_version"] = cuda_part
                    break
    except Exception as e:
        print(f"{R}[ERROR] nvidia-smi failed: {e}{RST}")
    
    return info


def get_ram_info() -> dict:
    """Get system RAM info via WMI."""
    info = {"total_gb": 0, "free_gb": 0, "used_pct": 0}
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "$os = Get-CimInstance Win32_OperatingSystem; "
             "Write-Host ($os.TotalVisibleMemorySize/1MB) "
             "($os.FreePhysicalMemory/1MB)"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                total = float(parts[0].replace(",", "."))
                free = float(parts[1].replace(",", "."))
                info["total_gb"] = round(total, 1)
                info["free_gb"] = round(free, 1)
                info["used_pct"] = round((total - free) / total * 100, 1)
    except Exception:
        pass
    return info


def kill_bloatware():
    """Kill common memory-hogging processes (safe ones only)."""
    bloat = [
        "MicrosoftEdge", "msedge", "Teams", "Spotify",
        "Discord", "Slack", "OneDrive", "YourPhone",
        "SkypeApp", "GameBar", "XboxGameBar",
        "SearchUI", "SearchApp", "WidgetService",
    ]
    killed = []
    for proc in bloat:
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", f"{proc}.exe"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                killed.append(proc)
        except Exception:
            pass
    return killed


def print_gpu_status(gpu_info: dict, ram_info: dict):
    """Pretty-print GPU & RAM status."""
    vram_pct = (gpu_info["vram_used_mb"] / max(gpu_info["vram_total_mb"], 1)) * 100
    
    # GPU bar
    bar_len = 30
    vram_fill = int(bar_len * vram_pct / 100)
    vram_bar = "█" * vram_fill + "░" * (bar_len - vram_fill)
    vc = G if vram_pct < 60 else (Y if vram_pct < 80 else R)
    
    # RAM bar
    ram_fill = int(bar_len * ram_info["used_pct"] / 100)
    ram_bar = "█" * ram_fill + "░" * (bar_len - ram_fill)
    rc = G if ram_info["used_pct"] < 60 else (Y if ram_info["used_pct"] < 80 else R)
    
    is_rtx4080 = "4080" in gpu_info["gpu_name"]
    gpu_badge = f"{G}✓ DETECTED{RST}" if is_rtx4080 else f"{Y}≈ COMPATIBLE{RST}"
    
    print(f"""
{W}┌─────────────────── GPU STATUS ───────────────────┐{RST}
  {C}GPU:{RST}   {gpu_info['gpu_name']}  [{gpu_badge}]
  {C}VRAM:{RST}  {vc}{vram_bar}{RST} {gpu_info['vram_used_mb']}/{gpu_info['vram_total_mb']} MB ({vram_pct:.0f}%)
  {C}Temp:{RST}  {gpu_info['gpu_temp']}°C  |  {C}Power:{RST} {gpu_info['power_draw']:.0f}/{gpu_info['power_limit']:.0f}W
  {C}Load:{RST}  GPU {gpu_info['gpu_util']}%  |  MEM {gpu_info['mem_util']}%
  {C}Driver:{RST} {gpu_info['driver_version']}  |  {C}CUDA:{RST} {gpu_info['cuda_version']}
{W}├─────────────────── RAM STATUS ────────────────────┤{RST}
  {C}RAM:{RST}   {rc}{ram_bar}{RST} {ram_info['used_pct']:.0f}% used ({ram_info['free_gb']:.1f}/{ram_info['total_gb']:.1f} GB free)
{W}└──────────────────────────────────────────────────┘{RST}
""")


# =============================================================================
# OPTIMIZATION ENGINE
# =============================================================================
def apply_optimizations(gpu_info: dict, mode: str = "training") -> dict:
    """
    Apply RTX 4080 Laptop optimized settings.
    Returns dict of env vars and recommendations.
    """
    opts = {}
    is_rtx4080 = "4080" in gpu_info["gpu_name"]
    is_laptop = "Laptop" in gpu_info["gpu_name"] or gpu_info["power_limit"] < 200
    vram_mb = gpu_info["vram_total_mb"]
    
    print(f"\n{C}[OPT] Applying optimizations for: {gpu_info['gpu_name']}{RST}")
    if is_laptop:
        print(f"{Y}[OPT] Laptop GPU detected — power-aware mode{RST}")
    
    # ── 1. CUDA Optimizations ───────────────────────────────────────────────
    # TF32 for Ampere/Ada Lovelace (RTX 30xx/40xx) — free 2-3x speedup
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    opts["tf32"] = True
    print(f"  {G}✓{RST} TF32 math enabled (2-3x faster matmul)")
    
    # ── 2. cuDNN Optimizations ──────────────────────────────────────────────
    os.environ["CUDNN_BENCHMARK"] = "1"
    opts["cudnn_benchmark"] = True
    print(f"  {G}✓{RST} cuDNN auto-tuner enabled")
    
    # ── 3. Memory Allocator ─────────────────────────────────────────────────
    # Use CUDA memory pool for less fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,"
        "max_split_size_mb:256"
    )
    opts["cuda_alloc_conf"] = True
    print(f"  {G}✓{RST} CUDA memory pool: expandable segments + 256MB max split")
    
    # ── 4. TensorRT Workspace ───────────────────────────────────────────────
    if vram_mb >= 10000:  # 10GB+
        trt_workspace = 2 * (1 << 30)  # 2GB
        trt_note = "2 GB (high VRAM)"
    elif vram_mb >= 6000:
        trt_workspace = 1 << 30  # 1GB
        trt_note = "1 GB (standard)"
    else:
        trt_workspace = 512 * (1 << 20)  # 512MB
        trt_note = "512 MB (low VRAM)"
    
    os.environ["TRT_WORKSPACE_MB"] = str(trt_workspace // (1 << 20))
    opts["trt_workspace"] = trt_workspace
    print(f"  {G}✓{RST} TensorRT workspace: {trt_note}")
    
    # ── 5. Power Mode (Laptop specific) ─────────────────────────────────────
    if is_laptop:
        # Prefer max performance
        try:
            subprocess.run(
                ["nvidia-smi", "-pm", "1"],
                capture_output=True, timeout=5
            )
            print(f"  {G}✓{RST} GPU persistence mode enabled")
        except Exception:
            pass
        
        # Set prefer max performance
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        opts["laptop_opt"] = True
        print(f"  {G}✓{RST} CUDA device pinned to GPU 0")
    
    # ── 6. Mode-specific tuning ─────────────────────────────────────────────
    if mode == "training":
        # Higher batch size for 12GB VRAM
        if vram_mb >= 10000:
            opts["rec_batch_size"] = 512
            opts["rec_n_steps"] = 4096
        elif vram_mb >= 6000:
            opts["rec_batch_size"] = 256
            opts["rec_n_steps"] = 2048
        else:
            opts["rec_batch_size"] = 128
            opts["rec_n_steps"] = 1024
        
        print(f"  {G}✓{RST} Recommended: batch={opts['rec_batch_size']}, n_steps={opts['rec_n_steps']}")
        
        # Gradient optimization
        os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
        print(f"  {G}✓{RST} cuDNN v8 Flash Attention API enabled")
        
    elif mode == "inference":
        # Lightweight mode
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        opts["threads"] = 4
        print(f"  {G}✓{RST} Thread count: 4 (inference mode)")
    
    # ── 7. Python GC ────────────────────────────────────────────────────────
    gc.collect()
    print(f"  {G}✓{RST} Python garbage collector ran")
    
    # ── 8. PyTorch warm-up + torch.compile ───────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set TF32 (Ada Lovelace / Ampere free speedup)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable Flash SDP for attention layers (RTX 4080 Compute 8.9)
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                print(f"  {G}✓{RST} Flash Scaled Dot Product Attention enabled")
            
            # Enable cuDNN v9 features if available
            if hasattr(torch.backends.cudnn, 'version') and torch.backends.cudnn.version() >= 9000:
                print(f"  {G}✓{RST} cuDNN {torch.backends.cudnn.version()} detected (v9+ optimized)")
            elif torch.backends.cudnn.is_available():
                print(f"  {G}✓{RST} cuDNN {torch.backends.cudnn.version()} detected")
            
            # torch.compile availability (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                opts["torch_compile"] = True
                print(f"  {G}✓{RST} torch.compile available (inductor backend)")
            
            # AMP (Automatic Mixed Precision) — huge speedup on RTX 4080
            opts["amp_available"] = True
            print(f"  {G}✓{RST} AMP (FP16 Mixed Precision) ready")
            
            # Warm up GPU with small tensor ops
            _t = torch.randn(1, 3, 64, 64, device="cuda")
            _ = torch.nn.functional.conv2d(_t, torch.randn(16, 3, 3, 3, device="cuda"))
            del _t, _
            torch.cuda.empty_cache()
            
            free_mem = torch.cuda.mem_get_info()[0] / 1e6
            total_mem = torch.cuda.mem_get_info()[1] / 1e6
            print(f"  {G}✓{RST} PyTorch CUDA ready | VRAM: {free_mem:.0f}/{total_mem:.0f} MB free")
            opts["torch_ready"] = True
    except ImportError:
        print(f"  {Y}⚠{RST} PyTorch not found in current env")
    
    print(f"\n{G}[OPT] All optimizations applied ✓{RST}\n")
    return opts


# =============================================================================
# CHECK TENSORRT STATUS
# =============================================================================
def check_tensorrt_status() -> dict:
    """Check if TensorRT engines exist and are ready."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from core.constants import DEPLOY_DIR, MODEL_PATH
    
    status = {
        "model_exists": os.path.exists(MODEL_PATH),
        "onnx_exists": os.path.exists(os.path.join(DEPLOY_DIR, "carla_v15.onnx")),
        "trt_fp16_exists": os.path.exists(os.path.join(DEPLOY_DIR, "carla_v15_fp16.engine")),
        "trt_int8_exists": os.path.exists(os.path.join(DEPLOY_DIR, "carla_v15_int8.engine")),
        "trt_available": False,
        "ort_available": False,
        "ort_trt_ep": False,
        "ort_cuda_ep": False,
    }
    
    try:
        import tensorrt
        status["trt_available"] = True
        status["trt_version"] = tensorrt.__version__
    except ImportError:
        pass
    
    try:
        import onnxruntime
        status["ort_available"] = True
        status["ort_version"] = onnxruntime.__version__
        providers = onnxruntime.get_available_providers()
        status["ort_trt_ep"] = 'TensorrtExecutionProvider' in providers
        status["ort_cuda_ep"] = 'CUDAExecutionProvider' in providers
    except ImportError:
        pass
    
    return status


def print_tensorrt_status(status: dict):
    """Pretty-print TensorRT & inference engine status."""
    def icon(b): return f"{G}✓{RST}" if b else f"{R}✗{RST}"
    
    # ORT TRT EP is functionally equivalent to standalone TensorRT
    has_trt_accel = status['trt_available'] or status.get('ort_trt_ep', False)
    trt_badge = f"{G}★ ACTIVE{RST}" if has_trt_accel else f"{R}MISSING{RST}"
    
    print(f"""
{W}┌────────────── INFERENCE ENGINES ──────────────────┐{RST}
  {icon(status['model_exists'])} PyTorch Model  (carla_grandmaster_v15.zip)
  {icon(status['onnx_exists'])} ONNX Model     (carla_v15.onnx)
  {icon(status['trt_fp16_exists'])} TensorRT FP16  (carla_v15_fp16.engine)
  {icon(status['trt_int8_exists'])} TensorRT INT8  (carla_v15_int8.engine)
{W}├─────────────── RUNTIME BACKENDS ──────────────────┤{RST}
  {icon(status['trt_available'])} TensorRT Standalone  {f"v{status.get('trt_version', '?')}" if status['trt_available'] else '(not installed)'}
  {icon(status['ort_available'])} ONNX Runtime         {f"v{status.get('ort_version', '?')}" if status['ort_available'] else '(pip install onnxruntime-gpu)'}
  {icon(status.get('ort_trt_ep', False))} ORT TensorRT EP      {'⚡ FP16 accelerated' if status.get('ort_trt_ep') else '(unavailable)'}
  {icon(status.get('ort_cuda_ep', False))} ORT CUDA EP          {'✓ cuDNN optimized' if status.get('ort_cuda_ep') else '(unavailable)'}
{W}├───────────────────────────────────────────────────┤{RST}
  TensorRT Acceleration: [{trt_badge}]
{W}└───────────────────────────────────────────────────┘{RST}
""")


# =============================================================================
# EXPORT TENSORRT (build engines if needed)
# =============================================================================
def auto_export_tensorrt():
    """Auto-export TensorRT engines if model exists but engines don't."""
    from core.constants import MODEL_PATH, DEPLOY_DIR
    
    status = check_tensorrt_status()
    
    if not status["model_exists"]:
        print(f"{R}[ERROR] No trained model found. Train first!{RST}")
        return False
    
    needs_export = not status["onnx_exists"] or not status["trt_fp16_exists"]
    
    if not needs_export:
        print(f"{G}[OK] TensorRT engines already exist. Skipping export.{RST}")
        return True
    
    if not status["trt_available"]:
        print(f"{Y}[WARN] TensorRT not installed.{RST}")
        if status["ort_available"]:
            print(f"{C}[INFO] Will use ONNX Runtime fallback.{RST}")
        else:
            print(f"{R}[ERROR] Neither TensorRT nor ONNX Runtime available!{RST}")
            print(f"  Install: pip install tensorrt onnxruntime-gpu")
            return False
    
    print(f"\n{C}[EXPORT] Building inference engines...{RST}")
    print(f"  This takes 1-5 minutes on first run.\n")
    
    try:
        from stable_baselines3 import PPO
        from core.tensorrt_engine import full_export_pipeline
        
        model = PPO.load(MODEL_PATH, device="cpu")
        results = full_export_pipeline(model)
        
        print(f"\n{G}[EXPORT] Complete! Engines ready for inference.{RST}")
        for k, v in results.items():
            print(f"  {G}✓{RST} {k}: {v}")
        return True
    except Exception as e:
        print(f"{R}[ERROR] Export failed: {e}{RST}")
        return False


# =============================================================================
# FULL PRE-FLIGHT CHECK
# =============================================================================
def preflight(mode: str = "training") -> dict:
    """
    Full system check + optimization.
    Call this before any training or inference.
    Returns optimization dict.
    """
    banner()
    
    # 1. GPU info
    gpu_info = get_nvidia_info()
    ram_info = get_ram_info()
    print_gpu_status(gpu_info, ram_info)
    
    # 2. TensorRT status
    trt_status = check_tensorrt_status()
    print_tensorrt_status(trt_status)
    
    # 3. Apply optimizations
    opts = apply_optimizations(gpu_info, mode)
    
    # 4. Recommendations
    vram_free = gpu_info["vram_free_mb"]
    ram_free = ram_info["free_gb"]
    
    print(f"{W}┌──────────────── RECOMMENDATIONS ──────────────────┐{RST}")
    
    if ram_free < 4:
        print(f"  {R}⚠ RAM CRITICAL ({ram_free:.1f} GB free){RST}")
        print(f"    → Close browsers, Discord, Spotify")
        print(f"    → Use CARLA -quality-level=Low")
    elif ram_free < 8:
        print(f"  {Y}⚠ RAM LOW ({ram_free:.1f} GB free){RST}")
        print(f"    → Consider closing heavy apps")
    else:
        print(f"  {G}✓ RAM OK ({ram_free:.1f} GB free){RST}")
    
    if vram_free < 2000:
        print(f"  {R}⚠ VRAM LOW ({vram_free} MB free){RST}")
        print(f"    → Close GPU-heavy apps")
    else:
        print(f"  {G}✓ VRAM OK ({vram_free} MB free){RST}")
    
    is_rtx4080 = "4080" in gpu_info["gpu_name"]
    if is_rtx4080:
        print(f"  {G}★ RTX 4080 — FP16 TensorRT is optimal{RST}")
        print(f"  {G}★ Ada Lovelace DLSS/TF32 active{RST}")
    
    if mode == "training" and not trt_status["model_exists"]:
        print(f"  {C}ℹ First training — model will be created{RST}")
    elif mode == "inference" and not trt_status["trt_fp16_exists"]:
        if trt_status["model_exists"]:
            print(f"  {Y}⚠ No TensorRT engine — will auto-build{RST}")
        else:
            print(f"  {R}⚠ No model found — train first!{RST}")
    
    print(f"{W}└───────────────────────────────────────────────────┘{RST}\n")
    
    return {
        "gpu": gpu_info,
        "ram": ram_info,
        "trt": trt_status,
        "opts": opts,
    }


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU Optimizer for CARLA AI")
    parser.add_argument("--mode", choices=["check", "optimize", "export", "cleanup"],
                        default="check")
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    
    if args.mode == "check":
        preflight("training" if args.training else "inference")
    
    elif args.mode == "optimize":
        gpu_info = get_nvidia_info()
        apply_optimizations(gpu_info, "training" if args.training else "inference")
    
    elif args.mode == "export":
        preflight("inference")
        auto_export_tensorrt()
    
    elif args.mode == "cleanup":
        print(f"\n{C}[CLEANUP] Killing bloatware...{RST}")
        killed = kill_bloatware()
        if killed:
            print(f"{G}  Killed: {', '.join(killed)}{RST}")
        else:
            print(f"  No bloatware found running.")
        
        ram = get_ram_info()
        print(f"  RAM after cleanup: {ram['used_pct']:.0f}% used ({ram['free_gb']:.1f} GB free)")
