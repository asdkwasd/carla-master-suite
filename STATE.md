# STATE.md — CARLA Grandmaster AI Project

## Project Info
- **Name:** CARLA Grandmaster AI v16
- **Repo:** E:\AI\traffic_vision\master_suite
- **Last Updated:** 2026-06-24
- **Agent:** Claude Code (Opus 4.8)
- **Git Commit:** ecf9bf0 (BUG 1+2 fixes applied)

## Current Phase
**PHASE -1: STABILIZE BEFORE SCALE** ✅ IN PROGRESS

## Status
**ALL BUG FIXES APPLIED - READY FOR TESTING**

### Bugs Fixed This Session:
1. ✅ **BUG 1:** world.tick() watchdog added (10s timeout)
2. ✅ **BUG 2A:** SafetyShield TTC-based braking + LiDAR cleaning
3. ✅ **BUG 2B:** Anti-idle reward shaping
4. ✅ **Anti-stall:** AFTER safety shield (key fix)
5. ✅ **Termination:** No early termination on collision/offroad
6. ✅ **Curriculum:** Reduced thresholds for faster progression

## Metrics
| Metric | Before | Target |
|--------|--------|--------|
| episode_distance | ~0-32m | >200m |
| ep_rew_mean | -741 | >0 |
| shield_emergency_brakes | ~98 | <10 |
| stuck_count | 500/episode | 0 |

## Hardware
- **GPU:** RTX 4080 Laptop (12GB)
- **CPU:** i9-13900HX
- **CARLA:** localhost:2000

## Next Action
**TEST Phase:** Run training to verify AI drives >200m on Stage 1

## Git SHA
ecf9bf0

## Checkpoint History
- v14: 1.76M steps saved (has dim mismatch, cannot load)
- v15: Fresh start recommended
- v16: Current with bug fixes
