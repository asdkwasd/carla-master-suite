# HANDOFF.md — CARLA Grandmaster AI

## Who Am I
- **Agent:** Claude Code (Autonomous Mode v2)
- **Machine:** Windows Laptop RTX 4080 12GB
- **Date:** 2026-07-01
- **Status:** ACTIVE - Running Phase -1 Stabilization

## What I Did

### Session Summary
1. Located repo at D:\AI\traffic_vision\master_suite (moved from E:\)
2. Read all state files (STATE.md, HANDOFF.md, PROGRESS.jsonl, CHANGELOG.md)
3. Analyzed training log: reward=0.00 indicates car NOT moving
4. Found available CARLA maps: Town01-Town10HD
5. Fixed curriculum map: "Town10HD_Opt" → "Town01" (Stage 1)
6. Added explicit map loading in reset()

### Current Training Metrics
| Metric | Value |
|--------|-------|
| Total timesteps | 64,096 (last known) |
| Iterations | 2 (last known) |
| Reward | 0.00 (PROBLEM - car not moving) |
| Distance | ~7-11m (last known) |
| GPU utilization | 0-33% (PROBLEM) |

### Bugs Fixed This Session
1. **Curriculum map fix**: Town10HD_Opt → Town01 (valid map)
2. **Map loading**: Added explicit map loading in reset()

### Files Modified
- D:\AI\traffic_vision\master_suite\core\constants.py (curriculum map)
- D:\AI\traffic_vision\master_suite\1_TRAIN_MASTER_v15.py (map loading)

## Active Issues (Need Fix)

### CRITICAL: Car Not Moving (reward=0.00)
- Training log shows reward=0.00 consistently
- Possible causes:
  1. RL not learning (needs more steps)
  2. Environment step() returning 0 reward
  3. CARLA not running
  4. Map/sensor issues

### CRITICAL: GPU Not Utilized
- GPU utilization: 0-33%
- Fix: Already have CUDA_VISIBLE_DEVICES=0 set
- Need to verify training actually uses GPU

## What You MUST Do Next

### 1. CHECK CARLA STATUS
```powershell
Test-NetConnection localhost -Port 2000
```

### 2. VERIFY GPU UTILIZATION
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```
Should show > 50% utilization during training.

### 3. RESTART TRAINING IF NEEDED
```batch
cd D:\AI\traffic_vision\master_suite
call ..\venv\Scripts\activate
python 1_TRAIN_MASTER_v15.py
```

### 4. MONITOR TRAINING LOG
```bash
tail -f training_carla_v15_log.txt
```
Look for:
- `[EP] dist=Xm` (X should increase over time)
- Non-zero reward values
- GPU utilization in training output

## Checkpoints Available
- D:\AI\traffic_vision\master_suite\checkpoints\v15_60000_steps.zip (60K steps)
- D:\AI\traffic_vision\master_suite\checkpoints\v15_40000_steps.zip (40K steps)
- D:\AI\traffic_vision\master_suite\checkpoints\v15_20000_steps.zip (20K steps)
- D:\AI\traffic_vision\master_suite\checkpoints\v14_1760000_steps.zip (1.76M steps!)

## Phase Status
**PHASE -1: STABILIZE** — In progress, not yet passed
- Acceptance criteria: 3 seeds reach mean episode distance > 200m on Stage 1
- Current status: Not verified (car not moving)

## Open Risks
1. CARLA may not be running
2. GPU not utilized during training
3. Car still oscillates/stuck behavior
4. Need to verify training actually produces non-zero rewards
