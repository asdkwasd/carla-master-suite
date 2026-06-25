# STATE.md — CARLA Grandmaster AI Project

## Project Info
- **Name:** CARLA Grandmaster AI v15
- **Repo:** E:\AI\traffic_vision\master_suite
- **Venv:** E:\AI\traffic_vision\venv
- **Last Updated:** 2026-06-25
- **Agent:** Claude Code (Opus 4.8)

## Current Phase
**PHASE -1: STABILIZE BEFORE SCALE** — PAUSED

## Bugs Fixed
1. ✅ Action space: throttle/brake conflict fixed (RL outputs both positive = cancel)
2. ✅ SafetyShield TTC: ignore <5m (car's own roof at 2m = false positive)
3. ✅ Anti-stall: one-time nudge, then let RL learn
4. ✅ min throttle floor: 0.2 (ensure forward motion)
5. ✅ Stuck threshold: 0.3 km/h (lenient for fresh policy)
6. ✅ load_world hangs → use get_world
7. ✅ Removed watchdog (was causing cascading timeouts)
8. ✅ CARLA pre-flight check (world + TM both verified)

## Training Results
| Metric | Value |
|--------|-------|
| episode_distance | ~25-62m |
| ep_rew_mean | +4,000 to +24,000 (improving) |
| shield_interventions | 0 (fixed!) |
| top_speed | 19.7 km/h |
| iterations | 11 (22,528 steps before crash) |
| checkpoints | v15_20k, v15_40k |

## Known Issues
- RL oscillates between forward/reverse (untrained policy)
- reward shaping: forward motion > reverse > idle
- CARLA crashes ~10-20 min into training (after ~20-40k steps)
- Need auto-restart script for overnight runs

## Hardware
- **GPU:** RTX 4080 Laptop (12GB)
- **CARLA:** localhost:2000 (Town10HD_Opt)

## Files Modified This Session
- core/safety.py (TTC thresholds + 5m ignore)
- core/cost_function.py (strong moving reward v3)
- 1_TRAIN_MASTER_v15.py (safe_tick, action mapping, anti-stall, init fixes)
- core/constants.py (Town10HD_Opt map)
