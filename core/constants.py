"""
==============================================================================
  CARLA GRANDMASTER AI — SINGLE SOURCE OF TRUTH
  All dimensional constants, paths, and configuration in ONE place.
  Every file imports from here. Zero mismatch guaranteed.
==============================================================================
"""
import os
import numpy as np

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH      = os.path.join(BASE_DIR, "carla_grandmaster_v15.zip")
TB_DIR          = os.path.join(BASE_DIR, "tb_logs")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_FILE        = os.path.join(BASE_DIR, "training_carla_v15_log.txt")
LEADERBOARD_DIR = os.path.join(BASE_DIR, "leaderboard")
MEMORY_DIR      = os.path.join(BASE_DIR, "agent_memories")
HER_BUFFER_DIR  = os.path.join(BASE_DIR, "her_buffers")
OBSERVER_PIPE   = os.path.join(BASE_DIR, "observer_state.json")
DEPLOY_DIR      = os.path.join(BASE_DIR, "deploy")
REPORT_DIR      = os.path.join(BASE_DIR, "reports")
HUMAN_DATA_DIR  = os.path.join(BASE_DIR, "human_demos")

for _p in [BASE_DIR, TB_DIR, CHECKPOINT_DIR, LEADERBOARD_DIR,
           MEMORY_DIR, HER_BUFFER_DIR, DEPLOY_DIR, REPORT_DIR, HUMAN_DATA_DIR]:
    os.makedirs(_p, exist_ok=True)

# =============================================================================
# SENSOR DIMENSIONS — THE SINGLE SOURCE OF TRUTH
# =============================================================================
IMG_WIDTH       = 80
IMG_HEIGHT      = 60
CAMERA_TYPE     = "sensor.camera.semantic_segmentation"
CAMERA_FOV      = "90"

N_LIDAR_SECTORS = 16        # 360° / 16 = 22.5° per sector
LIDAR_MAX_RANGE = 50.0
LIDAR_CHANNELS  = 32
LIDAR_PPS       = 200000    # points per second
LIDAR_RPM       = 20        # rotation frequency
LIDAR_UPPER_FOV = 10
LIDAR_LOWER_FOV = -30

# =============================================================================
# STATE VECTOR LAYOUT (28 dimensions)
# =============================================================================
# [0]  speed_norm          — kmh / 50
# [1]  last_steer          — previous steer action
# [2]  last_throttle       — previous throttle action
# [3]  at_junction         — 1.0 if at junction
# [4]  red_light           — 1.0 if red light ahead
# [5]  penalty_norm        — current_penalty / 1000
# [6]  safety_norm         — safety_score / 100
# [7]  danger_score        — from recall memory [0..1]
# [8]  min_lidar_norm      — min(lidar) / LIDAR_MAX_RANGE
# [9]  has_lane_invasion   — 1.0 if recent lane invasion
# [10] stuck_norm          — stuck_counter / 500
# [11] has_collision       — 1.0 if recent collision
# [12] guidance_x          — memory route guidance dx (normalized)
# [13] guidance_y          — memory route guidance dy (normalized)
# [14] road_ahead          — +1 road ahead, -1 no road
# [15] time_norm           — steps / max_steps
# [16] distance_norm       — current_distance / 1000
# [17] red_pixel_ratio     — semantic camera red ratio
# [18] is_recovering       — 1.0 if in recovery mode
# [19] rival_relative      — distance to rival (normalized)
# [20] is_offroad          — 1.0 if currently off-road
# [21] road_dist_norm      — distance to nearest road / 20
# [22] road_dx             — direction to road (x, normalized)
# [23] road_dy             — direction to road (y, normalized)
# [24] target_dx           — waypoint nav target dx (normalized)
# [25] target_dy           — waypoint nav target dy (normalized)
# [26] dist_to_target      — distance to route target / 30
# [27] junction_ahead      — 1.0 if junction within 30m
STATE_DIM       = 28

# =============================================================================
# ACTION SPACE
# =============================================================================
ACTION_DIM      = 3         # [steer, throttle, brake]
STEER_RANGE     = (-1.0, 1.0)
THROTTLE_RANGE  = (-1.0, 1.0)   # negative = reverse
BRAKE_RANGE     = (0.0, 1.0)

# =============================================================================
# TRAINING
# =============================================================================
PHYSICS_DT      = 0.05      # 20 Hz physics
MAX_EPISODE_STEPS = 5000
ROUTE_WAYPOINTS = 200       # waypoints to generate per episode
ROUTE_SPACING   = 10.0      # meters between route waypoints

# =============================================================================
# SEMANTIC SEGMENTATION PALETTE (CARLA class_id → RGB)
# =============================================================================
SEMANTIC_PALETTE = {
    0:  [0, 0, 0],          # Unlabeled
    1:  [100, 0, 0],        # Building
    2:  [150, 0, 0],        # Fence
    3:  [100, 50, 50],      # Other
    4:  [255, 0, 255],      # Pedestrian
    5:  [255, 255, 0],      # Pole
    6:  [0, 255, 0],        # RoadLine
    7:  [0, 255, 0],        # Road
    8:  [255, 0, 0],        # Sidewalk
    9:  [150, 75, 0],       # Vegetation
    10: [255, 100, 0],      # Vehicle
    11: [120, 0, 0],        # Wall
    12: [255, 255, 255],    # TrafficSign
}
SEMANTIC_DEFAULT = [80, 0, 0]

# =============================================================================
# CURRICULUM STAGES
# =============================================================================
CURRICULUM_STAGES = {
    1: {"name": "Basics",  "map": "Town01", "traffic": 0,  "pedestrians": 0,
        "weather": "Clear",  "obstacles": False, "threshold": 100.0},  # Reduced from 500
    2: {"name": "Traffic", "map": "Town03", "traffic": 10, "pedestrians": 5,
        "weather": "Cloudy", "obstacles": False, "threshold": 300.0},  # Reduced from 1000
    3: {"name": "Expert",  "map": "Town05", "traffic": 30, "pedestrians": 15,
        "weather": "Rainy",  "obstacles": True,  "threshold": 500.0},  # Reduced from 2000
    4: {"name": "Master",  "map": "Town10HD", "traffic": 50, "pedestrians": 30,
        "weather": "Stormy", "obstacles": True,  "threshold": 1000.0},  # Reduced from 5000
}
CURRICULUM_WINDOW = 50

# =============================================================================
# VALID LANE TYPES (for on-road detection)
# =============================================================================
# These are set at runtime after carla import, but we define a helper
def get_valid_lane_types():
    """Call after carla is imported."""
    import carla
    return (carla.LaneType.Driving, carla.LaneType.Bidirectional,
            carla.LaneType.OnRamp, carla.LaneType.OffRamp)

# =============================================================================
# CARLA CONNECTION
# =============================================================================
CARLA_HOST      = "127.0.0.1"
CARLA_PORT      = 2000
CARLA_TIMEOUT   = 60.0
TM_PORT         = 8000

# =============================================================================
# TENSORRT — Optimized for RTX 4080 Laptop (12GB VRAM, Ada Lovelace)
# =============================================================================
TRT_PRECISION   = "fp16"    # fp16 = best for RTX 40xx (Ada Lovelace)
TRT_MAX_BATCH   = 1
TRT_WORKSPACE   = 2 << 30   # 2 GB — RTX 4080 has plenty VRAM

# =============================================================================
# GPU PERFORMANCE FLAGS
# =============================================================================
# These are applied by gpu_optimizer.py and START.bat for RTX 4080 Laptop.
# TF32: ~2-3x speedup on Ampere/Ada matmul (zero accuracy loss for RL)
# cuDNN Benchmark: auto-selects fastest conv algorithm per input shape
# CUDA Alloc: expandable segments reduces memory fragmentation
GPU_OPTS = {
    "NVIDIA_TF32_OVERRIDE": "1",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
    "CUDNN_BENCHMARK": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0",
    "TORCH_CUDNN_V8_API_ENABLED": "1",
}

def apply_gpu_opts():
    """Apply GPU optimization env vars. Call before torch import."""
    for k, v in GPU_OPTS.items():
        os.environ.setdefault(k, v)
