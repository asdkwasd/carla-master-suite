# STATE.md — CARLA Grandmaster AI Project

## Project Info
- **Name:** CARLA Grandmaster AI v15
- **Repo:** E:\AI\traffic_vision\master_suite
- **Last Updated:** 2026-06-24
- **Agent:** Claude Code (Opus 4.8)
- **Git Commit:** bb3857f (initial recall)

## Current Phase
**PHASE -1: STABILIZE BEFORE SCALE**

## Problem Statement
AI won't move reliably. Two critical bugs:
1. **BUG 1:** world.tick() deadlock (no watchdog)
2. **BUG 2:** SafetyShield over-braking + reward hacking to idle

## Last Training Run
- **Date:** ~2026-04-09
- **Timesteps:** ~12K (stuck early)
- **Reward:** ep_rew_mean = -701 to -741
- **Issue:** "[STUCK] Didn't move for 500 steps" spam

## Metrics
| Metric | Value | Target |
|--------|-------|--------|
| episode_distance | ~0-32m | >200m |
| ep_rew_mean | -741 | >0 |
| shield_emergency_brakes | ~98 | <10 |
| stuck_count | 500/episode | 0 |

## Hardware
- **GPU:** RTX 4080 Laptop (12GB)
- **CPU:** i9-13900HX
- **CARLA:** localhost:2000

## Active Blockers
1. SafetyShield over-braking (BUG 2A)
2. Reward hacking to idle (BUG 2B)
3. No world.tick() watchdog (BUG 1)

## Checkpoint
- **Path:** E:\AI\traffic_vision\master_suite\checkpoints\
- **Latest:** v14_160000_steps.zip (dim mismatch, cannot load)
- **Action:** Start fresh with fixed v15

## Next Action
Fix BUG 1 + BUG 2 in core modules (safety.py, cost_function.py, 1_TRAIN_MASTER_v15.py)

## Git SHA
bb3857f
