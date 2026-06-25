# CHANGELOG.md

## 2026-06-25 — Session 2: Stabilization + Testing

### Bugs Fixed
1. **Action Space Conflict:** RL outputs throttle+brake both positive = cancel out
   - Fix: `if throttle > 0.1: brake = 0.0` (throttle wins)
   - Car now moves at 0-19.7 km/h

2. **SafetyShield Over-Braking:** TTC thresholds too aggressive
   - Car's own roof at z=2.4m detected at 2m → constant braking
   - Fix: Ignore <5m, TTC thresholds 1.0/2.0/3.0s

3. **Anti-Stall Override:** Was overriding RL every step (stuck=51→0→51...)
   - Fix: One-time nudge (throttle=0.9), then `_anti_stall_nudged` flag

4. **Watchdog Crash:** 10s timeout < 100s rollout time = cascading timeouts
   - Fix: Removed watchdog, use try/except on world.tick()

5. **load_world Hang:** Reloading world hangs indefinitely
   - Fix: Use get_world() instead

6. **CARLA Pre-flight:** No check before training → crashes on reset
   - Fix: Test both get_world() AND get_trafficmanager()

7. **Unicode Encoding:** `→` character → cp1252 crash
   - Fix: Replaced with ASCII arrows

8. **PPO Config:** LR=3e-4, n_steps=4096 too slow for debugging
   - Fix: LR=5e-4, n_steps=2048, ent_coef=0.03

### Training Results
- Episode distance: 2m → 25m → 62m (best)
- Shield interventions: 2,500 → 0 (FIXED)
- Reward: -25,000 → +24,000
- Top speed: 19.7 km/h
- Steps trained: ~40,000 (2 iterations of checkpoints)

### Checkpoints Saved
- checkpoints/v15_20000_steps.zip (Jun 25 17:04)
- checkpoints/v15_40000_steps.zip (Jun 25 17:26)

## 2026-06-24 — Session 1: Initial Bug Fixes
- Commit: ecf9bf0
- TickWatchdog, SafetyShield TTC, anti-idle reward, no early termination

## [Earlier Sessions]
- v14: GrandmasterFeatureExtractor, stuck at ~1.76M steps (numpy mismatch, can't load)
- v11: RecoveryModule
- v10: HER buffer, WorldModel
- v9: ADR (Automatic Domain Randomization)
