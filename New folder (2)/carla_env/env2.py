import carla
import numpy as np
import cv2
import math
from collections import deque

from carla_env.vehicle import Vehicle
from carla_env.sensors import CollisionSensor

import gymnasium as gym
from gymnasium import spaces


# Semantic segmentation tag IDs in CARLA
# https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
SEG_ROAD      = 1   # drivable road surface
SEG_ROADLINE  = 6   # lane markings


class RGBCamera:
    """Higher resolution front-facing RGB camera for the agent's observation."""
    def __init__(self, world, vehicle):
        self.image = None
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '160')
        bp.set_attribute('image_size_y', '80')
        bp.set_attribute('fov', '100')
        transform = carla.Transform(carla.Location(x=2.5, z=1.0),
                                    carla.Rotation(pitch=-5))
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.sensor.listen(self._on_image)

    def _on_image(self, data):
        img = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((80, 160, 4))
        self.image = img[:, :, :3]   # RGB, drop alpha

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()


class SegCamera:
    """
    Semantic segmentation camera — used ONLY for reward computation.
    The agent never sees this. It tells us how much road/lane is ahead
    without using any CARLA map API.
    """
    def __init__(self, world, vehicle):
        self.image = None
        bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', '160')
        bp.set_attribute('image_size_y', '80')
        bp.set_attribute('fov', '100')
        transform = carla.Transform(carla.Location(x=2.5, z=1.0),
                                    carla.Rotation(pitch=-5))
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.sensor.listen(self._on_image)

    def _on_image(self, data):
        # Raw semantic image: R channel = semantic tag
        img = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((80, 160, 4))
        self.image = img[:, :, 2]   # Red channel holds the tag

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()


class CarlaEnvCamera(gym.Env):
    """
    Camera-only RL environment.
    - Observation : stacked RGB frames (no CARLA map data whatsoever)
    - Reward      : derived from semantic segmentation (road visible, centered)
                    + collision sensor
    - No waypoints, no lateral_offset, no heading_error from CARLA APIs
    """

    def __init__(self):
        super().__init__()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world  = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode   = False
        self.world.apply_settings(settings)

        self.vehicle          = None
        self.rgb_cam          = None
        self.seg_cam          = None
        self.collision_sensor = None

        # 7 symmetric discrete actions
        self.actions = [
            (0.5, -0.4),
            (0.5, -0.2),
            (0.5, -0.1),
            (0.5,  0.0),
            (0.5,  0.1),
            (0.5,  0.2),
            (0.5,  0.4),
        ]
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation: 4 stacked grayscale frames, each 80×160
        # Grayscale reduces channels while keeping spatial detail
        self.n_frames    = 4
        self.frame_stack = deque(maxlen=self.n_frames)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(80, 160, self.n_frames),
            dtype=np.uint8
        )

        self.prev_steer = 0.0
        self.step_count = 0
        self.MAX_STEPS  = 1000

    # ================================================================
    # RESET
    # ================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy_actors()

        self.vehicle      = Vehicle(self.world)
        vehicle_actor     = self.vehicle.spawn()
        self.prev_steer   = 0.0
        self.step_count   = 0

        spectator = self.world.get_spectator()
        tf = vehicle_actor.get_transform()
        spectator.set_transform(carla.Transform(
            tf.location + carla.Location(z=20),
            carla.Rotation(pitch=-90)
        ))

        self.rgb_cam          = RGBCamera(self.world, vehicle_actor)
        self.seg_cam          = SegCamera(self.world, vehicle_actor)
        self.collision_sensor = CollisionSensor(self.world, vehicle_actor)
        self.frame_stack.clear()

        # Wait until both cameras have a frame
        while self.rgb_cam.image is None or self.seg_cam.image is None:
            self.world.tick()

        first = self._preprocess(self.rgb_cam.image)
        for _ in range(self.n_frames):
            self.frame_stack.append(first)

        return self._get_obs(), {}

    # ================================================================
    # PREPROCESSING — RGB → grayscale
    # ================================================================
    def _preprocess(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)   # (80, 160)
        return gray

    # ================================================================
    # OBSERVATION
    # ================================================================
    def _get_obs(self):
        # Stack along channel axis → (80, 160, 4)
        return np.stack(list(self.frame_stack), axis=2)

    # ================================================================
    # STEP
    # ================================================================
    def step(self, action_idx):
        throttle, target_steer = self.actions[action_idx]

        steer           = 0.6 * target_steer + 0.4 * self.prev_steer
        self.prev_steer = steer

        self.vehicle.apply_control(throttle=throttle, steer=steer)

        spectator = self.world.get_spectator()
        tf = self.vehicle.vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            tf.location + carla.Location(z=15),
            carla.Rotation(pitch=-90)
        ))

        self.world.tick()
        self.step_count += 1

        rgb_frame = self.rgb_cam.image.copy()
        seg_frame = self.seg_cam.image.copy()

        self.frame_stack.append(self._preprocess(rgb_frame))

        reward, done = self._compute_reward(steer, seg_frame)

        if self.step_count >= self.MAX_STEPS:
            done = True

        self._show_debug(rgb_frame, seg_frame)

        return self._get_obs(), reward, done, False, {}

    # ================================================================
    # REWARD — derived purely from camera + collision sensor
    # ================================================================
    def _compute_reward(self, steer, seg):
        if self.collision_sensor.collision:
            return -100.0, True

        h, w = seg.shape

        # Focus on the bottom 60% of the image (near field, most relevant)
        roi = seg[int(h * 0.4):, :]
        roi_h, roi_w = roi.shape
        total_pixels = roi_h * roi_w

        road_mask = (roi == SEG_ROAD) | (roi == SEG_ROADLINE)
        road_pixels = np.sum(road_mask)

        # If almost no road visible — car is off road
        road_ratio = road_pixels / total_pixels
        if road_ratio < 0.10:
            return -10.0, True

        # --- Road coverage reward ---
        # More road visible ahead = good
        reward = road_ratio * 3.0

        # --- Road centering reward ---
        # Find the horizontal center of mass of road pixels
        # If it's near the image center, the car is centered on the road
        col_indices = np.where(road_mask)
        if len(col_indices[1]) > 0:
            road_cx = np.mean(col_indices[1])          # horizontal centroid
            img_cx  = roi_w / 2.0
            offset  = abs(road_cx - img_cx) / img_cx   # 0=centred, 1=edge
            reward += (1.0 - offset) * 2.0

            # Penalise if road center is far from image center
            if offset > 0.35:
                reward -= 1.5

        # --- Speed reward (from vehicle physics, not CARLA map) ---
        velocity = self.vehicle.vehicle.get_velocity()
        speed    = math.sqrt(velocity.x**2 + velocity.y**2)
        reward  += speed * 0.1

        # --- Steer jitter penalty ---
        reward -= abs(steer - self.prev_steer) * 0.05

        return reward, False

    # ================================================================
    # DEBUG WINDOW
    # ================================================================
    def _show_debug(self, rgb, seg):
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Overlay road pixels in green, lane lines in yellow
        road_overlay = np.zeros_like(vis)
        road_overlay[seg == SEG_ROAD]     = [0, 180, 0]
        road_overlay[seg == SEG_ROADLINE] = [0, 220, 220]
        vis = cv2.addWeighted(vis, 0.7, road_overlay, 0.3, 0)

        velocity = self.vehicle.vehicle.get_velocity()
        speed_kmh = math.sqrt(velocity.x**2 + velocity.y**2) * 3.6
        cv2.putText(vis, f"spd={speed_kmh:.0f}km/h  steer={self.prev_steer:.2f}",
                    (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.imshow("AutoCar Camera-Only", vis)
        cv2.waitKey(1)

    # ================================================================
    # CLEANUP
    # ================================================================
    def _destroy_actors(self):
        for attr in ('rgb_cam', 'seg_cam', 'collision_sensor', 'vehicle'):
            obj = getattr(self, attr)
            if obj is not None:
                obj.destroy()
                setattr(self, attr, None)

    def close(self):
        self._destroy_actors()
        cv2.destroyAllWindows()
        settings = self.world.get_settings()
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        settings.no_rendering_mode   = False
        self.world.apply_settings(settings)
