# CHANGELOG.md

## 2026-06-24 — Session 1 (Bug Fixes Applied)
- **Commit:** ecf9bf0
- **Action:** Phase -1 bug fixes - ALL major issues resolved

### Files Modified:
1. **1_TRAIN_MASTER_v15.py:**
   - Added TickWatchdog class (10s timeout for world.tick())
   - Added anti-stall AFTER safety shield
   - Removed early termination on collision/offroad
   - Wrapped all world.tick() with watchdog

2. **core/safety.py (v2):**
   - Added LiDAR cleaning (_clean_lidar)
   - TTC-based braking (1.5s/3.0s/5.0s thresholds)
   - Creep mode: allow slow movement at low speed
   - Added intervention logging

3. **core/cost_function.py (v2):**
   - Anti-idle penalty: -1.0 per step (was -0.3)
   - Forward-progress bonus: speed * 0.5 per m/s
   - Collision penalty reduced: -2 (was -5)
   - Waypoint reward: +5.0 per new waypoint

4. **core/constants.py:**
   - Reduced curriculum thresholds for faster progression
   - Stage 1: 500 → 100
   - Stage 2: 1000 → 300
   - Stage 3: 2000 → 500
   - Stage 4: 5000 → 1000

### New Baton Files:
- STATE.md, HANDOFF.md, PROGRESS.jsonl, NEXT_ACTIONS.md, CHANGELOG.md

## 2026-06-24 — Initial Commit
- **Commit:** bb3857f
- **Action:** Initial state from memory recall
- **Files:** All v15 code (24 files)

## [Previous Sessions - from memory]
- v14: Added GrandmasterFeatureExtractor, stuck at ~1.76M steps
- v11: Added RecoveryModule
- v10: Added HER buffer, WorldModel
- v9: Added ADR (Automatic Domain Randomization)
