# CARLA AI GRANDMASTER - Learning Curriculum Documentation

> **Document Purpose**: This file serves as both human documentation AND machine-readable curriculum guide.
> The AI system can parse this file to understand its learning stages, goals, and evaluation criteria.

---

## 🎯 Mission Objective

**Primary Goal**: Navigate a complete city loop (Town01, Town03, Town05, or Town10HD) **without any penalties**.

**Success Criteria**:
- ✅ Complete full circuit around the city
- ✅ Zero (0) collisions
- ✅ Zero (0) off-road violations
- ✅ Zero (0) red light violations
- ✅ Follow all traffic rules
- ✅ Drive smoothly (speed appropriate for conditions)

**Ultimate Benchmark**: `Score = Distance - (Penalty × 0.1) - Safety_Reducton + Win_Bonus`

---

## 📚 Learning Curriculum (4 Stages)

### Stage 1: BASICS - "Learn to Drive"

**Environment**: Town01 (simplest map)
- Traffic: 0 vehicles
- Pedestrians: 0
- Weather: Clear (0% rain, 0% fog)
- Obstacles: None

**Learning Objectives**:
1. Stay on the road (green asphalt in semantic segmentation)
2. Basic steering control
3. Acceleration and braking
4. Navigate simple turns
5. Understand the reward function

**Reward Shaping**:
```python
# Positive Reinforcement
+20.0   # New waypoint discovery (exploration)
+6.0    # Speed reward (drive at optimal speed)
+2.0    # Ghost path following (follow best known route)
+1.0    # Green pixel detection (on asphalt)

# Negative Reinforcement (No Reset - Accumulative)
-50.0/step  # Off-road penalty (continues, doesn't reset)
-150.0      # Collision (continues, learn to recover)
-400.0      # Red light violation
```

**Advancement Threshold**: `Mean Episode Reward >= 500.0` over 50 episodes

**Key Behaviors to Learn**:
- Green = Safe (asphalt), Red = Danger (sidewalk/lava)
- Center of lane is safest
- Smooth steering = better reward
- Keep moving (stuck = penalty)

---

### Stage 2: RULES - "Traffic Laws & Basic Traffic"

**Environment**: Town03 (more complex intersections)
- Traffic: 10 vehicles (NPC)
- Pedestrians: 5
- Weather: Cloudy (20% cloudiness)
- Obstacles: None

**Learning Objectives**:
1. Traffic light compliance
2. Junction navigation
3. Sharing road with other vehicles
4. Pedestrian avoidance
5. Lane discipline

**New Challenges**:
- Traffic lights require stopping (red) and going (green/yellow)
- Other vehicles move unpredictably
- Pedestrians may cross the road
- More complex intersections

**Reward Changes**:
```python
# Additional penalties
-100.0      # Lane invasion (changing lanes unsafely)
-400.0      # Red light violation (increased severity)
```

**Advancement Threshold**: `Mean Episode Reward >= 1000.0` over 50 episodes

**Key Behaviors to Learn**:
- Stop at red lights, go on green
- Yield to other traffic when appropriate
- Maintain safe distance from pedestrians
- Use turn signals (implied by smooth lane changes)

---

### Stage 3: EXPERT - "Complex Traffic & Weather"

**Environment**: Town05 (highway & complex urban)
- Traffic: 30 vehicles
- Pedestrians: 15
- Weather: Rainy (wet roads, reduced visibility)
- Obstacles: Yes (random props on road)

**Learning Objectives**:
1. Highway driving (higher speeds)
2. Wet road handling
3. Dynamic obstacle avoidance
4. Dense traffic navigation
5. Decision making under uncertainty

**New Challenges**:
- Reduced traction (wet roads)
- Lower visibility (rain on camera)
- Obstacles appear randomly
- Higher speed requirements

**Advanced Features Activated**:
```python
# Decision Making Module
if danger_ahead > 0.8:
    throttle = min(throttle, 0.3)  # Slow down for caution
    brake = max(brake, 0.2)

# Memory System
danger_memory = recall_memory.get_danger_score(location)
# Avoid areas with past collisions
```

**Advancement Threshold**: `Mean Episode Reward >= 2000.0` over 50 episodes

**Key Behaviors to Learn**:
- Slow down for hazards
- Use memory to avoid dangerous areas
- Recover from skids (wet road)
- Plan ahead for obstacles

---

### Stage 4: MASTER - "Ultimate Challenge"

**Environment**: Town10HD (most complex, high detail)
- Traffic: 50 vehicles (very dense)
- Pedestrians: 30
- Weather: Stormy (heavy rain, fog, wind)
- Obstacles: Yes (frequent)

**Learning Objectives**:
1. Perfect driving in ALL conditions
2. Zero penalty completion
3. Championship-level racing against rival AI
4. Adaptive behavior (adjust to changing conditions)

**Advanced Systems Fully Operational**:
```python
# Hindsight Experience Replay (HER)
# "Even if I failed, what did I achieve?"
failed_episode.relabel_goals(achieved_distance)

# Automatic Domain Randomization (ADR)
# Auto-adjust difficulty to maintain optimal learning
if success_rate > 0.6:
    difficulty.increase()
elif success_rate < 0.3:
    difficulty.decrease()

# World Model Prediction
# "Imagine what will happen if I take this action"
future_state, predicted_reward = world_model.predict(state, action)

# Self-Play Competition
# Compete against a copy of yourself
if ego_distance > rival_distance:
    ego_wins += 1
    reward += 1000  # Victory bonus!
```

**Graduation Requirements**: `Complete city lap (5km+) with < 100 penalty points`

---

## 🧠 Neural Architecture

### Observation Space (What the AI "Sees")

```python
{
    "camera": (120, 160, 3)  # Semantic segmentation
        # Channel 0: Unlabeled/Building (black/dark red)
        # Channel 1: 0 (unused)
        # Channel 2: Class ID (see palette below)

    "lidar": (8,)  # 8-sector polar distance map
        # [front, front-right, right, back-right, back, back-left, left, front-left]

    "state": (17,)  # Telemetry vector
        # [0] speed_norm        # Normalized speed (0-1)
        # [1] steer             # Last steering command
        # [2] throttle          # Last throttle command
        # [3] at_junction       # Is at junction?
        # [4] red_light         # Red light ahead?
        # [5] penalty_accum     # Accumulated penalty
        # [6] safety_score      # Current safety (0-1)
        # [7] danger_memory     # Memory-based danger score
        # [8] min_lidar         # Closest obstacle distance
        # [9] lane_invasion     # Lane invaded recently?
        # [10] stuck_counter    # How long stuck?
        # [11] collision_recent # Collision recently?
        # [12] route_x          # Route guidance X
        # [13] route_y          # Route guidance Y
        # [14] ahead_safe        # Is path ahead safe?
        # [15] progress         # Episode progress (0-1)
        # [16] distance_norm    # Normalized distance
}
```

### Semantic Segmentation Palette

The AI sees the world in specific colors:

| Color (RGB) | Class | Meaning |
|-------------|-------|---------|
| (0, 255, 0) | Road/RoadLine | **SAFE** - Drive here! |
| (255, 0, 0) | Sidewalk | **DANGER** - Lava! Avoid! |
| (255, 0, 255) | Pedestrian | **HAZARD** - Don't hit! |
| (255, 100, 0) | Vehicle | **OBSTACLE** - Avoid collision |
| (255, 255, 0) | Pole | **OBSTACLE** - Static hazard |
| (0, 0, 0) | Building/Void | **BOUNDARY** - Cannot pass |
| (100, 0, 0) | Fence/Wall | **BOUNDARY** |
| (255, 255, 255) | Traffic Sign | **INFO** - Read if possible |

### Action Space

```python
[steer, throttle, brake]
# steer:   -1.0 (left) to 1.0 (right)
# throttle: 0.0 (none) to 1.0 (full)
# brake:    0.0 (none) to 1.0 (full)
```

---

## 🎓 Learning Algorithm

### PPO (Proximal Policy Optimization)

**Hyperparameters**:
```python
learning_rate   = 3e-4     # How fast to learn
n_steps         = 2048     # Samples per iteration
batch_size      = 128      # Updates per iteration
n_epochs        = 10       # How many times to reuse data
gamma           = 0.99     # Discount factor (care about future)
gae_lambda      = 0.95     # Temporal difference smoothing
clip_range      = 0.2      # Prevent too large policy updates
ent_coef        = 0.05     # Exploration bonus
vf_coef         = 0.5      # Value function importance
max_grad_norm   = 0.5      # Gradient clipping (stable training)
```

### Advanced Learning Techniques

#### 1. Hindsight Experience Replay (HER)
```python
# Convert failures to learning opportunities
for episode in failed_episodes:
    for transition in episode:
        original_goal = "reach_destination"
        achieved_goal = transition.achieved_state
        # Relabel: "Given I reached X, was my action good for getting to X?"
        new_reward = compute_reward(achieved_goal, transition.action)
```

#### 2. Self-Play Competition
```python
# Ego vs Rival (same neural network, different instances)
rival_action = model.predict(rival_observation)
ego_reward = compete(ego_action, rival_action)
if ego_wins:
    ego_wins += 1
    # Learn from winning
else:
    # Learn from what rival did differently
```

#### 3. Recall Memory
```python
# Remember dangerous locations
if collision_occurred:
    memory.add_collision(location)
# Later
danger_score = memory.get_danger_score(current_location)
if danger_score > threshold:
    slow_down()
```

#### 4. Decision Making
```python
# "Think twice" before dangerous actions
if leaving_road and speed > 10:
    # Hesitate - reduce speed and evaluate
    throttle = 0.2
    brake = 0.3
```

---

## 📊 Evaluation Metrics

### Primary Metrics

1. **Success Rate**: Episodes completed without major penalties
2. **Average Distance**: Mean meters traveled per episode
3. **Penalty Accumulation**: Total penalty points per episode
4. **Win Rate vs Rival**: Percentage of self-play victories

### Secondary Metrics

5. **Safety Score**: 0-100, decreases with violations
6. **Waypoints Discovered**: Unique road segments explored
7. **Episode Duration**: Steps before termination
8. **FPS**: Training speed (frames per second)

### Success Stages

| Stage | Distance | Penalty | Description |
|-------|----------|---------|-------------|
| 🌱 Beginner | < 100m | High | Random exploration, frequent errors |
| 🌿 Learner | 100-500m | Medium | Can navigate, some mistakes |
| 🚗 Driver | 500-2000m | Low | Competent driving, rare errors |
| 🏆 Master | 2000m+ | < 100 | Expert, near-perfect driving |
| 👑 Legend | 5000m+ | 0 | Perfect city loop, zero penalties |

---

## 🔧 Troubleshooting Guide

### Problem: AI keeps going off-road
**Cause**: Not detecting road boundaries properly
**Solution**: Increase green pixel reward weight, check semantic segmentation

### Problem: AI is too hesitant/slow
**Cause**: Danger threshold too low
**Solution**: Increase `danger_ahead` threshold from 0.7 to 0.9

### Problem: AI gets stuck frequently
**Cause**: Anti-stall not aggressive enough
**Solution**: Increase minimum throttle to 0.7

### Problem: Not progressing through curriculum
**Cause**: Threshold too high or exploration too low
**Solution**: Lower `advance_threshold`, increase `ent_coef`

### Problem: High variance in performance
**Cause**: Learning rate too high or batch size too small
**Solution**: Decrease `learning_rate`, increase `batch_size`

---

## 🚀 Quick Start

```bash
# 1. Start CARLA
cd /path/to/CARLA_0.9.16
./CarlaUE4.exe -windowed -world-port=2000

# 2. Train AI
cd /path/to/traffic_vision/master_suite
python 1_TRAIN_MASTER_ULTIMATE.py

# 3. Monitor training (another terminal)
tensorboard --logdir tb_logs

# 4. View leaderboard
python 3_COMPETITION_VIEWER.py --mode live
```

---

## 📖 Learning Progress Checklist

- [ ] Stage 1: Achieve 500+ reward consistently
- [ ] Stage 2: Complete 1km without off-road
- [ ] Stage 3: Handle wet roads without incidents
- [ ] Stage 4: Full city loop under 100 penalty
- [ ] Championship: Beat rival AI > 70% win rate
- [ ] Legend: Perfect city loop, zero penalties

---

**Remember**: The goal is not just to survive, but to drive **perfectly**. Every penalty is a lesson. Every collision is data. Every success is progress.

> "The AI that learns from its mistakes while remembering its successes will become the ultimate driver."
