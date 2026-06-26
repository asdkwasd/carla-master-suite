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
- ✅ Reward shaping fix (MOVING_REWARD=10, WAYPOINT=50, IDLE=-1)
- ✅ Deleted redundant trainers (5 files removed)
- ✅ Transfer learning v14→v15 implemented

### IN PROGRESS: Fix GPU Training
- 🔄 GPU not utilized (CPU only) — ADD CUDA_VISIBLE_DEVICES=0
- 🔄 CARLA may be crashed — restart headless
- 🔄 RL oscillates (reward fix needs iterations)

### TODO (Priority Order)
1. **FIX GPU** — Add `os.environ['CUDA_VISIBLE_DEVICES']='0'` at top of 1_TRAIN_MASTER_v15.py
2. **START CARLA** — CarlaUE4.exe -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
3. **RESUME** — python 1_TRAIN_MASTER_v15.py
4. **MONITOR** — nvidia-smi should show 50-90% utilization
5. **VERIFY** — episode_distance > 200m after 100K steps

## Phase 0-7: Future
See MASTER_PROMPT.txt for full roadmap.
- Phase 0: Smoke-test no-op agent
- Phase 1: Expert data collection (PDM-lite/Roach)
- Phase 2: Reproduce TransFuser++ baseline
- Phase 3+: Improve perception and add System 1/2
