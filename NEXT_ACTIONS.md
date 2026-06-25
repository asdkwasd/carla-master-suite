# NEXT_ACTIONS.md — Ordered Task Queue

## Phase -1: Stabilize (IN PROGRESS)

### DONE
- ✅ Action space fix (throttle/brake conflict)
- ✅ SafetyShield TTC fix (ignore <5m)
- ✅ Anti-stall one-time nudge
- ✅ min throttle floor = 0.2
- ✅ Stuck threshold = 0.3 km/h
- ✅ load_world → get_world
- ✅ Removed watchdog
- ✅ CARLA pre-flight check
- ✅ Shield interventions = 0

### IN PROGRESS: RL Learning
- 🔄 RL oscillates between forward/reverse
- 🔄 MOVING_REWARD_PER_MPS = 10.0 fix applied (need 50k+ steps to see effect)
- 🔄 Need to monitor if distance improves over iterations

### TODO After RL Learns Forward
1. Verify episode_distance > 200m (Stage 1 acceptance)
2. Add auto-restart script for overnight training
3. Investigate CARLA memory leak (crashes ~10-20 min)
4. Create TensorRT checkpoint at 100k steps

## Phase 0-7: Future
See MASTER_PROMPT.txt for full roadmap.
- Phase 0: Smoke-test no-op agent
- Phase 1: Expert data collection (PDM-lite/Roach)
- Phase 2: Reproduce TransFuser++ baseline
- Phase 3+: Improve perception and add System 1/2
