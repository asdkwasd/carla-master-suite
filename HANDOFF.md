# HANDOFF.md — CARLA Grandmaster AI

## Who Am I
- **Agent:** Claude Code (Opus 4.8)
- **Machine:** Windows Laptop (RTX 4080 12GB)
- **Date:** 2026-06-25
- **Session:** Started ~4 hours ago, pausing now

## What I Did
1. Located repo at E:\AI\traffic_vision\master_suite (moved from original location)
2. Found venv at E:\AI\traffic_vision\venv (PyTorch 2.5.1+cu121)
3. Found CARLA at E:\AI\traffic_vision\CARLA_0.9.16
4. Fixed BUG 1: Action space — throttle/brake conflict (RL outputs both positive = cancel!)
5. Fixed BUG 2A: SafetyShield TTC too aggressive (car's own roof at 2m)
6. Fixed BUG 2B: Anti-stall one-time nudge (was overriding RL every step)
7. Fixed BUG 3: load_world hangs → use get_world
8. Removed watchdog (was causing cascading timeouts)
9. Added CARLA pre-flight check (world + TM both verified)
10. Verified physics works (car moves fine in diagnostic test)
11. Training runs but RL oscillates between forward/reverse

## What You MUST Do Next
1. **CARLA may be dead** — launch with:
   ```
   E:\AI\traffic_vision\CARLA_0.9.16\CarlaUE4.exe -quality-level=Low -prefernvidia -carla-rpc-port=2000
   ```
2. **Verify CARLA:**
   ```python
   import carla
   c = carla.Client('127.0.0.1', 2000)
   c.set_timeout(10)
   w = c.get_world(); tm = c.get_trafficmanager(8000)
   print(w.get_map().name)  # Should be Town10HD_Opt
   ```
3. **Resume training:**
   ```
   cd E:\AI\traffic_vision\master_suite
   E:\AI\traffic_vision\venv\Scripts\python.exe 1_TRAIN_MASTER_v15.py
   ```
4. **Latest checkpoint:** `checkpoints/v15_40000_steps.zip` (auto-resumes)

## Key Metrics
| Metric | Value |
|--------|-------|
| episode_distance | ~25-62m (best) |
| ep_rew_mean | +4k to +24k |
| shield_interventions | 0 (FIXED) |
| top_speed | 19.7 km/h |
| iterations done | 11 (22,528 steps) |

## Root Cause of RL Oscillation
RL learns to output negative throttle (= reverse direction) because:
- Collision penalty -5.0 > forward progress reward +5.0
- Reverse keeps car moving (progress reward) without collision risk
- FIX APPLIED: MOVING_REWARD_PER_MPS = 10.0 (much stronger)
- Need more iterations to see if fix works

## Open Risks
1. RL may need 100k+ steps to learn forward-only
2. CARLA crashes ~10-20 min in (memory leak suspected)
3. Need auto-restart script for overnight runs
4. v14 checkpoint has numpy version mismatch (can't load)

## Files Modified This Session
- 1_TRAIN_MASTER_v15.py (many fixes)
- core/safety.py (TTC thresholds, 5m ignore)
- core/cost_function.py (strong moving reward v3)
- core/constants.py (Town10HD_Opt map)
- core/feature_extractor.py (Unicode fix)
- Checkpoints: v15_20k, v15_40k
- TB logs: tb_logs/PPO_102/

## Exact Resume Command
```
cd E:\AI\traffic_vision\master_suite
git pull
E:\AI\traffic_vision\venv\Scripts\python.exe 1_TRAIN_MASTER_v15.py
```
