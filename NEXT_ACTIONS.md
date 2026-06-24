# NEXT_ACTIONS.md — Ordered Task Queue

## Phase -1: Stabilize (CURRENT)
Priority order - fix one at a time, test between each.

### Task 1: BUG 1 - world.tick() Watchdog (HIGHEST)
- **File:** 1_TRAIN_MASTER_v15.py
- **Action:** Add watchdog timer around world.tick()
- **Timeout:** 10 seconds
- **On timeout:** Log, reconnect client, reload world, continue
- **Status:** TODO

### Task 2: BUG 2A - SafetyShield Over-Braking (HIGH)
- **File:** core/safety.py
- **Root cause:** front_min<5 triggers brake=1.0 constantly
- **Fixes needed:**
  - [ ] Verify LiDAR sectors return LIDAR_MAX_RANGE when empty
  - [ ] Add TTC (Time-To-Collision) based braking
  - [ ] Allow low-speed creep (<5 km/h)
  - [ ] Only hard-brake on verified close obstacle
  - [ ] Log every intervention for debugging
- **Status:** TODO

### Task 3: BUG 2B - Reward Hacking to Idle (HIGH)
- **File:** core/cost_function.py
- **Root cause:** standing penalty (-0.3) << collision penalty (-100)
- **Fixes needed:**
  - [ ] Increase anti-idle penalty to -1.0 per step
  - [ ] Add forward-progress bonus: +speed_along_route * 0.5
  - [ ] Reduce collision penalty: -5 → -2
  - [ ] Add waypoint reward: +5.0 per waypoint
  - [ ] Ensure moving > staying
- **Status:** TODO

### Task 4: BUG 2C - Curriculum Stage 1 Too Hard (MEDIUM)
- **File:** core/curriculum.py
- **Fix:** Stage 1 = empty straight road, zero NPCs, goal = drive forward
- **Status:** TODO

### Task 5: Validation Test
- **Action:** Run 3 seeds, verify mean episode distance >200m
- **Acceptance:** Zero freezes, car drives forward consistently
- **Status:** TODO

## Phase 0-7: Future (After Phase -1)
See MASTER_PROMPT.txt for full roadmap.
