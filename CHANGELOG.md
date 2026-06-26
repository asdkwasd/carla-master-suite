# CHANGELOG.md

## 2026-06-26 — Session 4b: Fix "Car Won't Move" Oscillation

### Problem Fixed
- AI oscillates forward/reverse and gets stuck
- Collision penalty (-5) > forward progress reward (+5)
- RL learns "stand still = safe" (negative reward for movement)

### Changes Made

#### core/cost_function.py — Reward Shaping
1. **Forward progress reward: +10 per m/s** (confirmed, already correct)
2. **Anti-idle penalty: -1 per step if speed < 0.5 m/s** (was -2, threshold raised)
   - Idle threshold: 0.05 → 0.5 m/s (penalize slow movement more strictly)
   - Step bonus: 1.0 → 2.0 (reward sustained movement)
3. **Waypoint reached: +50 bonus** (was 5.0, 10x increase)
   - Massive reward for progress prevents idle optimization

#### core/safety.py — Less Aggressive SafetyShield
1. **Creep mode speed: 15 → 20 km/h** (allow faster creeping)
2. **TTC_CRITICAL: 1.0s → 0.8s** (only hard brake for very close)
3. **TTC_WARNING: 2.0s → 2.5s** (gentle brake threshold)
4. **TTC_SAFE: 3.0s → 4.0s** (full throttle OK sooner)

### Expected Behavior
- Forward motion reward (+10/m/s) now DOMINATES idle penalty (-1/step)
- Car earns +5/step at 0.5 m/s vs -1/step when idle
- Waypoint bonuses (+50) encourage route following
- SafetyShield allows speeds up to 20 km/h in creep mode

## 2026-06-26 — Session 4: Repo Cleanup

### Files Deleted (Superseded by v15)
- `1_TRAIN_MASTER.py` — v14 duplicate (all features in v15)
- `1_TRAIN_MASTER_ULTIMATE.py` — v10 duplicate (HER/WorldModel in v15 core)
- `1_TRAIN_MASTER_RECOVERY.py` — v11 duplicate (RecoveryModule in v15 core)
- `3_TEST_LIVE.py` — v8 evaluation script (superseded by 3_TEST_LIVE_v15.py)
- `TEST_NOW.py` — one-off v14 test script

### Kept
- `1_TRAIN_MASTER_v15.py` — Production training (Tesla/Waymo architecture)
- `3_TEST_LIVE_v15.py` — Live evaluation for v15 models

## 2026-06-26 — Session 3: Transfer Learning from v14

### New Feature: Transfer Learning Pipeline
- **Goal:** Leverage 1.76M steps of v14 training to accelerate v15
- **Implementation:** Automatic weight transfer when starting fresh v15 model
- **Transfer Strategy:**
  1. Load best v14 checkpoint (v14_1760000_steps.zip)
  2. Match compatible layers by name and shape
  3. Copy weights with `torch.no_grad()` (no gradient modification)
  4. Lower learning rate (1e-4) for fine-tuning transferred weights

### Files Modified
- `1_TRAIN_MASTER_v15.py` — Added transfer learning logic (~50 lines)
- `transfer_learning_v15.py` — New analysis script for checkpoint inspection

### Expected Benefits
- Faster convergence (starting from ~1.76M steps of knowledge)
- Better base behavior (v14 learned route following, obstacle avoidance)
- Reduced stuck/sloth behavior (v14 learned to move)

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
