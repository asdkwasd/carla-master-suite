# NEXT_ACTIONS.md — Ordered Task Queue

## Phase -1: Stabilize (IN PROGRESS)
All bug fixes applied. Now TESTING.

### Task 1: ✅ BUG 1 - world.tick() Watchdog
- **Status:** DONE
- **Added:** TickWatchdog class with 10s timeout
- **Files:** 1_TRAIN_MASTER_v15.py

### Task 2: ✅ BUG 2A - SafetyShield Over-Braking
- **Status:** DONE
- **Fixed:** TTC-based braking, LiDAR cleaning, creep mode
- **Files:** core/safety.py

### Task 3: ✅ BUG 2B - Reward Hacking to Idle
- **Status:** DONE
- **Fixed:** Anti-idle penalty -1.0, forward-progress bonus, waypoint reward +5
- **Files:** core/cost_function.py

### Task 4: ✅ Anti-Stall AFTER Safety Shield
- **Status:** DONE
- **Fixed:** Nudges car forward when stuck >100 steps
- **Files:** 1_TRAIN_MASTER_v15.py

### Task 5: ✅ No Early Termination
- **Status:** DONE
- **Fixed:** Removed collision/offroad early termination
- **Files:** 1_TRAIN_MASTER_v15.py

### Task 6: TESTING (NEXT)
- **Action:** Run training to verify AI drives >200m on Stage 1
- **Command:** `python 1_TRAIN_MASTER_v15.py`
- **Expected:** No more "[STUCK]" logs, car moves forward
- **Acceptance:** Mean episode distance >200m over 3 seeds

## Phase 0-7: Future (After Phase -1)
See MASTER_PROMPT.txt for full roadmap.
- Phase 0: Smoke-test no-op agent
- Phase 1: Expert data collection
- Phase 2: Reproduce TransFuser++ baseline
- Phase 3+: Improve perception and add System 1/2
