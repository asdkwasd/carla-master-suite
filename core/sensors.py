"""
==============================================================================
  Unified Sensor Processing — Thread-safe, vectorized, no duplication.
  Handles: Semantic Camera, LiDAR sector mapping, Collision, Lane Invasion.
==============================================================================
"""
import threading
import numpy as np
import carla

from .constants import (
    IMG_WIDTH, IMG_HEIGHT, N_LIDAR_SECTORS, LIDAR_MAX_RANGE,
    SEMANTIC_PALETTE, SEMANTIC_DEFAULT, CAMERA_TYPE, CAMERA_FOV,
    LIDAR_CHANNELS, LIDAR_PPS, LIDAR_RPM, LIDAR_UPPER_FOV, LIDAR_LOWER_FOV,
    N_RADAR_SECTORS, RADAR_RANGE, RADAR_FOV, RADAR_POINTS
)


class SensorSuite:
    """
    Manages all CARLA sensors for one vehicle.
    Thread-safe buffers with proper locking.
    """

    def __init__(self):
        # --- Buffers ---
        self.last_img: np.ndarray | None = None
        self.last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        self.last_radar = np.zeros(N_RADAR_SECTORS, np.float32)
        self.collision_events: list = []
        self.lane_invasion_events: list = []

        # --- Locks ---
        self._img_lock = threading.Lock()
        self._lidar_lock = threading.Lock()
        self._radar_lock = threading.Lock()
        self._col_lock = threading.Lock()
        self._lane_lock = threading.Lock()

        # --- Actor handles (set after attach) ---
        self.camera = None
        self.lidar = None
        self.radar = None
        self.collision = None
        self.lane = None

        # --- Pre-build palette LUT for fast semantic conversion ---
        self._palette_lut = np.zeros((256, 3), dtype=np.uint8)
        for cid, color in SEMANTIC_PALETTE.items():
            self._palette_lut[cid] = color
        # Default for unknown classes
        for i in range(max(SEMANTIC_PALETTE.keys()) + 1, 256):
            self._palette_lut[i] = SEMANTIC_DEFAULT

    # =========================================================================
    # ATTACH / DETACH
    # =========================================================================
    def attach(self, vehicle, world):
        """Spawn and attach all sensors to vehicle."""
        lib = world.get_blueprint_library()

        # --- Semantic Camera ---
        cam_bp = lib.find(CAMERA_TYPE)
        cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
        cam_bp.set_attribute("fov", CAMERA_FOV)
        self.camera = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            attach_to=vehicle
        )
        self.camera.listen(self._on_camera)

        # --- LiDAR ---
        lid_bp = lib.find("sensor.lidar.ray_cast")
        lid_bp.set_attribute("channels", str(LIDAR_CHANNELS))
        lid_bp.set_attribute("range", str(LIDAR_MAX_RANGE))
        lid_bp.set_attribute("points_per_second", str(LIDAR_PPS))
        lid_bp.set_attribute("rotation_frequency", str(LIDAR_RPM))
        lid_bp.set_attribute("upper_fov", str(LIDAR_UPPER_FOV))
        lid_bp.set_attribute("lower_fov", str(LIDAR_LOWER_FOV))
        self.lidar = world.spawn_actor(
            lid_bp,
            carla.Transform(carla.Location(x=0.0, z=2.4)),
            attach_to=vehicle
        )
        self.lidar.listen(self._on_lidar)

        # --- Radar ---
        rad_bp = lib.find("sensor.other.radar")
        rad_bp.set_attribute("horizontal_fov", str(RADAR_FOV))
        rad_bp.set_attribute("vertical_fov", str(RADAR_FOV))
        rad_bp.set_attribute("points_per_second", str(RADAR_POINTS))
        rad_bp.set_attribute("range", str(RADAR_RANGE))
        self.radar = world.spawn_actor(
            rad_bp,
            carla.Transform(carla.Location(x=2.0, z=1.0)), # Front bumper
            attach_to=vehicle
        )
        self.radar.listen(self._on_radar)

        # --- Collision ---
        self.collision = world.spawn_actor(
            lib.find("sensor.other.collision"),
            carla.Transform(), attach_to=vehicle
        )
        self.collision.listen(self._on_collision)

        # --- Lane Invasion ---
        self.lane = world.spawn_actor(
            lib.find("sensor.other.lane_invasion"),
            carla.Transform(), attach_to=vehicle
        )
        self.lane.listen(self._on_lane)

    def detach(self):
        """Stop and mark sensors for destruction. Returns list of actors."""
        actors = []
        for attr in ("camera", "lidar", "radar", "collision", "lane"):
            s = getattr(self, attr, None)
            if s is not None:
                try:
                    s.stop()
                except Exception:
                    pass
                actors.append(s)
                setattr(self, attr, None)
        self.reset_buffers()
        return actors

    def reset_buffers(self):
        with self._img_lock:
            self.last_img = None
        with self._lidar_lock:
            self.last_lidar = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        with self._radar_lock:
            self.last_radar = np.zeros(N_RADAR_SECTORS, np.float32)
        with self._col_lock:
            self.collision_events = []
        with self._lane_lock:
            self.lane_invasion_events = []

    # =========================================================================
    # CALLBACKS — fast, thread-safe
    # =========================================================================
    def _on_camera(self, image):
        """Convert semantic segmentation → RGB via LUT (vectorized)."""
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        raw = raw.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
        class_ids = raw[:, :, 2]                       # CARLA semantic channel
        rgb = self._palette_lut[class_ids]             # Vectorized LUT — 50x faster than loop
        with self._img_lock:
            self.last_img = rgb

    def _on_lidar(self, point_cloud):
        """Convert LiDAR → N-sector min-distance array (vectorized)."""
        pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        if pts.size == 0:
            return
        pts = pts.reshape(-1, 4)[:, :3]                # x, y, z
        # Filter: keep obstacles above ground, below roof
        mask = (pts[:, 2] > 0.3) & (pts[:, 2] < 2.2)
        pts = pts[mask]

        sectors = np.full(N_LIDAR_SECTORS, LIDAR_MAX_RANGE, np.float32)
        if len(pts) > 0:
            angles = np.arctan2(pts[:, 1], pts[:, 0]) % (2 * np.pi)
            indices = (angles / (2 * np.pi) * N_LIDAR_SECTORS).astype(np.intp) % N_LIDAR_SECTORS
            dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            # VECTORIZED min-reduce — replaces slow Python loop
            np.minimum.at(sectors, indices, dists)

        with self._lidar_lock:
            self.last_lidar = sectors

    def _on_radar(self, radar_data):
        """Convert Radar → N-sector velocity array."""
        # raw_data is an array of carla.RadarDetection
        # We can extract velocity and azimuth
        pts = np.frombuffer(radar_data.raw_data, dtype=np.float32)
        if pts.size == 0:
            return
        
        # radar format per point: velocity, azimuth, altitude, depth
        pts = pts.reshape(-1, 4)
        velocities = pts[:, 0]
        azimuths = pts[:, 1]
        
        sectors = np.zeros(N_RADAR_SECTORS, np.float32)
        if len(pts) > 0:
            # Map azimuth (-FOV/2 to FOV/2) to sector bins
            # shift to [0, FOV], then normalize
            fov_rad = math.radians(RADAR_FOV)
            shifted = azimuths + (fov_rad / 2)
            indices = (shifted / fov_rad * N_RADAR_SECTORS).astype(np.intp)
            indices = np.clip(indices, 0, N_RADAR_SECTORS - 1)
            
            # Use max velocity approaching us (negative velocity = moving toward ego)
            np.minimum.at(sectors, indices, velocities)

        with self._radar_lock:
            self.last_radar = sectors

    def _on_collision(self, event):
        with self._col_lock:
            self.collision_events.append(event)

    def _on_lane(self, event):
        with self._lane_lock:
            self.lane_invasion_events.append(event)

    # =========================================================================
    # THREAD-SAFE READERS
    # =========================================================================
    def get_camera(self) -> np.ndarray:
        with self._img_lock:
            if self.last_img is not None:
                return self.last_img.copy()
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    def get_lidar(self) -> np.ndarray:
        with self._lidar_lock:
            return self.last_lidar.copy()

    def get_radar(self) -> np.ndarray:
        with self._radar_lock:
            return self.last_radar.copy()

    def get_and_clear_collisions(self) -> list:
        with self._col_lock:
            events = self.collision_events
            self.collision_events = []
            return events

    def get_and_clear_lane_invasions(self) -> list:
        with self._lane_lock:
            events = self.lane_invasion_events
            self.lane_invasion_events = []
            return events

    def has_camera_data(self) -> bool:
        with self._img_lock:
            return self.last_img is not None

    def get_red_pixel_ratio(self) -> float:
        """Fraction of red (sidewalk) pixels in semantic view."""
        with self._img_lock:
            if self.last_img is None:
                return 0.0
            img = self.last_img
        red_mask = (img[:, :, 0] == 255) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
        return float(np.sum(red_mask) / (IMG_WIDTH * IMG_HEIGHT))
