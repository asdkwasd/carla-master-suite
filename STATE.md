# STATE.md — CARLA Grandmaster AI

## Repo Info
- **Location:** D:\AI\traffic_vision\master_suite
- **Venv:** D:\AI\traffic_vision\venv
- **CARLA:** D:\AI\traffic_vision\CARLA_0.9.16
- **GPU:** RTX 4080 Laptop (12GB)
- **Updated:** 2026-06-26 15:27

## Phase
**PHASE -1: STABILIZE BEFORE SCALE** — IN PROGRESS

## Session Status
**PAUSED** — Ready to resume with "lanjutkan carla ue4"

## Checkpoints Available
| File | Steps | Date |
|------|-------|------|
| v15_20000_steps.zip | 20K | Jun 25 |
| v15_40000_steps.zip | 40K | Jun 25 |
| v15_60000_steps.zip | 60K | Jun 26 |
| v14_1760000_steps.zip | 1.76M | Apr 7 |

## Current Metrics (Last Known)
| Metric | Value |
|--------|-------|
| Total timesteps | 64,096 |
| Iterations | 2 |
| Reward | ~36,800 |
| Distance | ~7-11m |
| GPU utilization | 0-33% (PROBLEM!) |
| Curriculum stage | 1 |

## Bugs Fixed This Session
- [x] Reward shaping: MOVING_REWARD=10, IDLE_PENALTY=-1, WAYPOINT=50
- [x] SafetyShield: TTC_CRITICAL 0.8s, creep 20 km/h
- [x] Deleted 5 redundant trainers

## CRITICAL Issues (Need Fix)
1. **GPU Not Utilized** — Add `CUDA_VISIBLE_DEVICES=0` at top of training script
2. **CARLA May Be Crashed** — Need to restart headless
3. **Car Won't Move** — Stuck/oscillates (reward fix needs iterations)

## Quick Resume Commands
```batch
cd D:\AI\traffic_vision\master_suite
call ..\venv\Scripts\activate

REM Start CARLA headless
cd ..\CARLA_0.9.16
start CarlaUE4.exe -RenderOffScreen -quality-level=Low -carla-rpc-port=2000

REM Resume training
cd ..\master_suite
python 1_TRAIN_MASTER_v15.py
```

## Scripts Created
- transfer_inject.py — On-the-fly transfer learning
- rtx_maximizer.py — GPU optimizer
- START_TRANSFER.bat — Transfer launcher
- EMERGENCY_RESCUE.bat — Rescue script

## Next Agent Actions (Priority)
1. Fix GPU training (add CUDA_VISIBLE_DEVICES)
2. Start CARLA headless
3. Resume training
4. Monitor for stuck behavior
