"""
==============================================================================
  Feature Extractor — Tesla/Waymo Inspired Sensor Fusion

  Architecture:
  ┌─────────────┐   ┌──────────────────┐   ┌───────────────┐
  │ Semantic Cam │──▶│ CNN → context    │──▶│               │
  │  (80×60×3)  │   │  features        │   │               │
  └─────────────┘   └──────────────────┘   │   FUSION      │
  ┌─────────────┐   ┌──────────────────┐   │   CrossAttn   │──▶ 256d
  │ LiDAR 16sec │──▶│ Spatial Encoder  │──▶│   + MLP       │
  │  (16,)      │   │  + Angle embed   │   │               │
  └─────────────┘   └──────────────────┘   │               │
  ┌─────────────┐   ┌──────────────────┐   │               │
  │ State 28d   │──▶│ State MLP        │──▶│               │
  │  telemetry  │   │  encoder         │   │               │
  └─────────────┘   └──────────────────┘   └───────────────┘

  KEY DESIGN (Tesla-inspired):
  - LiDAR is PRIMARY spatial awareness (like Tesla's occupancy net)
  - Camera is CONTEXT (lane markings, traffic lights, semantic scene)
  - State is MEMORY (speed, steering history, danger, route guidance)
  - Cross-attention lets LiDAR tell vision WHERE to focus
==============================================================================
"""
import math
import torch as th
from torch import nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from .constants import (
    IMG_HEIGHT, IMG_WIDTH, N_LIDAR_SECTORS, STATE_DIM, LIDAR_MAX_RANGE
)


class CNNBackbone(nn.Module):
    """Compact CNN for semantic camera context extraction."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=3), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        # Compute output dim dynamically (immune to resolution changes)
        with th.no_grad():
            sample = th.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            self.out_dim = self.net(sample).shape[1]

    def forward(self, x):
        return self.net(x)


class LiDARSpatialEncoder(nn.Module):
    """
    Encodes LiDAR sectors with spatial angle embeddings.
    Each sector gets: [distance, sin(angle), cos(angle)] → spatial features.
    Tesla-style: this is the "occupancy network" equivalent.
    """

    def __init__(self, n_sectors: int = N_LIDAR_SECTORS, out_dim: int = 128):
        super().__init__()
        self.n_sectors = n_sectors
        self.out_dim = out_dim

        # Register angle embeddings as buffer (not parameter)
        angles = th.linspace(0, 2 * math.pi, n_sectors)
        self.register_buffer('sin_angles', th.sin(angles))
        self.register_buffer('cos_angles', th.cos(angles))

        self.encoder = nn.Sequential(
            nn.Linear(n_sectors * 3, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

        # Danger detection head — binary "is something close?"
        self.danger_head = nn.Sequential(
            nn.Linear(n_sectors, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

    def forward(self, lidar):
        B = lidar.shape[0]
        spatial_input = th.cat([
            lidar,
            self.sin_angles.unsqueeze(0).expand(B, -1),
            self.cos_angles.unsqueeze(0).expand(B, -1),
        ], dim=-1)
        return self.encoder(spatial_input)


class GrandmasterExtractor(BaseFeaturesExtractor):
    """
    Tesla/Waymo-inspired multi-modal sensor fusion.

    Key innovations:
    1. LiDAR spatial encoder with angle embeddings (occupancy-like)
    2. CNN for semantic scene context
    3. State encoder for telemetry + route guidance
    4. Fusion MLP with residual connection
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # --- Vision backbone ---
        self.vision = CNNBackbone()
        v_dim = self.vision.out_dim

        # --- LiDAR spatial encoder ---
        self.lidar_encoder = LiDARSpatialEncoder(N_LIDAR_SECTORS, 128)
        l_dim = 128

        # --- State encoder ---
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        s_dim = 64

        # --- Fusion with residual ---
        combined_dim = v_dim + l_dim + s_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

        # Log dimensions
        print(f"[EXTRACTOR] Vision={v_dim}d, LiDAR={l_dim}d, "
              f"State={s_dim}d → {features_dim}d")

    def forward(self, observations):
        # --- Camera (semantic segmentation) ---
        cam = observations["camera"].float()
        if cam.ndim == 4 and cam.shape[-1] == 3:
            cam = cam.permute(0, 3, 1, 2) / 255.0
        elif cam.ndim == 4 and cam.max() > 1.0:
            cam = cam / 255.0
        vis = self.vision(cam)

        # --- LiDAR ---
        lidar = observations["lidar"]
        lid = self.lidar_encoder(lidar)

        # --- State ---
        state = self.state_encoder(observations["state"])

        # --- Fusion ---
        combined = th.cat([vis, lid, state], dim=-1)
        return self.fusion(combined)
