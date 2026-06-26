# HANDOFF.md — CARLA Grandmaster AI

## Who Am I
- **Agent:** Claude Code (Autonomous Mode)
- **Machine:** Windows Laptop RTX 4080 12GB
- **Date:** 2026-06-26 15:27
- **Status:** PAUSED - Save State

## What I Did

### Session Summary
1. Found repo at D:\AI\traffic_vision\master_suite
2. Implemented Transfer Learning v14→v15 (12/85 layers matched)
3. Fixed reward shaping (MOVING_REWARD=10, WAYPOINT=50, ANTI_IDLE=-1)
4. Made SafetyShield less aggressive (TTC thresholds increased)
5. Deleted redundant trainers (1_TRAIN_MASTER.py, etc.)
6. Training ran but GPU not utilized (CPU only issue)

### Current Training Metrics
| Metric | Value |
|--------|-------|
| Total timesteps | 64,096 |
| Iterations | 2 |
| Reward | ~36,800 |
| Distance | ~7-11m |
| GPU utilization | 0-33% (PROBLEM) |

### Bugs Fixed This Session
1. Reward shaping: forward reward +10/m/s, idle penalty -1/step, waypoint +50
2. SafetyShield: TTC_CRITICAL 1.0→0.8s, creep speed 20 km/h
3. Deleted: 1_TRAIN_MASTER.py, 1_TRAIN_MASTER_ULTIMATE.py, 1_TRAIN_MASTER_RECOVERY.py, 3_TEST_LIVE.py, TEST_NOW.py

### Files Modified
- D:\AI\traffic_vision\master_suite\STATE.md (updated)
- D:\AI\traffic_vision\master_suite\core\cost_function.py (reward fixes)
- D:\AI\traffic_vision\master_suite\core\safety.py (SafetyShield less aggressive)
- D:\AI\traffic_vision\master_suite\1_TRAIN_MASTER_v15.py (various fixes)
- D:\AI\traffic_vision\master_suite\CHANGELOG.md (updated)

### Scripts Created
- D:\AI\traffic_vision\master_suite\transfer_inject.py (on-the-fly transfer learning)
- D:\AI\traffic_vision\master_suite\rtx_maximizer.py (GPU optimizer)
- D:\AI\traffic_vision\master_suite\START_TRANSFER.bat (transfer launcher)
- D:\AI\traffic_vision\master_suite\EMERGENCY_RESCUE.bat (rescue script)

## Active Issues (Need Fix)

### CRITICAL: GPU Not Utilized
- Training runs on CPU, not GPU
- GPU utilization: 0-33%
- Fix: Add `os.environ['CUDA_VISIBLE_DEVICES']='0'` at top of training script
- Ensure PPO uses device='cuda'

### CRITICAL: CARLA May Have Crashed
- Training frozen at step 2150
- Symptoms: "streaming client: connection failed"
- Fix: Restart CARLA headless before training

### Car Won't Move
- Stuck/sloth behavior persists
- Reward shaping fix applied but needs more iterations
- Waypoint reward +50 should help

## What You MUST Do Next

### 1. RESTART CARLA (if crashed)
```batch
cd D:\AI\traffic_vision\CARLA_0.9.16
start CarlaUE4.exe -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
```

### 2. FIX GPU TRAINING
Add at TOP of D:\AI\traffic_vision\master_suite\1_TRAIN_MASTER_v15.py (before torch import):
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

### 3. RESUME TRAINING
```batch
cd D:\AI\traffic_vision\master_suite
call ..\venv\Scripts\activate
python 1_TRAIN_MASTER_v15.py
```

## Checkpoints Available
- D:\AI\traffic_vision\master_suite\checkpoints\v15_20000_steps.zip
- D:\AI\traffic_vision\master_suite\checkpoints\v15_40000_steps.zip
- D:\AI\traffic_vision\master_suite\checkpoints\v15_60000_steps.zip
- D:\AI\traffic_vision\master_suite\checkpoints\v14_1760000_steps.zip (1.76M steps!)

## Exact Resume Commands

### Resume Training
```batch
cd D:\AI\traffic_vision\master_suite
call ..\venv\Scripts\activate
python 1_TRAIN_MASTER_v15.py
```

### Check GPU Status
```powershell
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
```

### Check CARLA
```powershell
Test-NetConnection localhost -Port 2000
```

## Open Risks
1. GPU not utilized - training slow on CPU
2. CARLA may crash during training
3. Car still oscillates forward/reverse
4. Need headless mode for better GPU utilization

## Phase Status
**PHASE -1: STABILIZE** — In progress, not yet passed
