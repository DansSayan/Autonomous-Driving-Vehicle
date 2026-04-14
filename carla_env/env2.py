import math
import carla
import numpy as np
import cv2
from collections import deque

import gymnasium as gym
from gymnasium import spaces

from carla_env.vehicle import Vehicle
from carla_env.sensors import CollisionSensor
from models.cnn_model import ImagePreprocessor, N_FRAMES, IMG_H, IMG_W, ROUTE_VEC_DIM, N_MAJOR

TRAFFIC_LIGHTS = [
    {"id": "id22", "loc": carla.Location(x=-113.80, y=  5.16, z=0.5)},
    {"id": "id18", "loc": carla.Location(x=-113.96, y= 19.06, z=0.5)},
    {"id": "id21", "loc": carla.Location(x= -32.90, y= 28.40, z=0.5)},
    {"id": "id19", "loc": carla.Location(x= -58.52, y=140.35, z=0.5)},
    {"id": "id11", "loc": carla.Location(x= 109.70, y= 21.20, z=0.5)},
    {"id": "id9",  "loc": carla.Location(x= -46.25, y=-68.48, z=0.5)},
]

_SEGMENTS = [
    [(-113.80, 10.0), (-113.90, 14.5), (-113.96, 19.06)],
    [(-105.0, 22.0), (-85.0, 27.0), (-65.0, 29.0), (-45.0, 28.5), (-32.90, 28.40)],
    [(-45.0, 60.0), (-50.0, 90.0), (-55.0, 120.0), (-58.52, 140.35)],
    [(10.0, 140.0), (60.04, 69.83), (100.0, 55.0), (109.70, 21.20)],
    [(79.95, 13.41), (30.01, -57.41), (-10.0, -55.0), (-46.25, -68.48)],
    [(-69.92, -58.22), (-95.0, -25.0), (-110.0, -8.0), (-113.80, 5.16)],
]

ROUTE = []
for major_idx, segment in enumerate(_SEGMENTS):
    tl_id = TRAFFIC_LIGHTS[major_idx]["id"]
    for (x, y) in segment:
        ROUTE.append({
            "loc":       carla.Location(x=x, y=y, z=0.5),
            "major_idx": major_idx,
            "id":        tl_id,
        })

N_WAYPOINTS       = len(ROUTE)
CHECKPOINT_RADIUS = 12.0
TL_RADIUS         = 14.0

SPAWN_TRANSFORM = carla.Transform(
    carla.Location(x=-113.80, y=2.0, z=0.5),
    carla.Rotation(yaw=90.0)
)

SEG_ROAD     = 1
SEG_ROADLINE = 6


class RGBCamera:
    def __init__(self, world, vehicle_actor):
        self.image = None
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(IMG_W))
        bp.set_attribute('image_size_y', str(IMG_H))
        bp.set_attribute('fov', '100')
        transform = carla.Transform(carla.Location(x=2.5, z=1.0), carla.Rotation(pitch=-5))
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle_actor)
        self.sensor.listen(self._cb)

    def _cb(self, data):
        img = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((IMG_H, IMG_W, 4))
        self.image = img[:, :, :3]

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()


class SegCamera:
    def __init__(self, world, vehicle_actor):
        self.image = None
        bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(IMG_W))
        bp.set_attribute('image_size_y', str(IMG_H))
        bp.set_attribute('fov', '100')
        transform = carla.Transform(carla.Location(x=2.5, z=1.0), carla.Rotation(pitch=-5))
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle_actor)
        self.sensor.listen(self._cb)

    def _cb(self, data):
        img = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((IMG_H, IMG_W, 4))
        self.image = img[:, :, 2]

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()


class CarlaEnvCamera(gym.Env):
    """
    Camera-only RL env. Observation is a Dict:
      "image"     : (N_FRAMES, IMG_H, IMG_W) uint8  — channel-first for DummyVecEnv
      "route_vec" : (8,) float32  — one-hot(6) + cos_bearing + sin_bearing

    Action: Discrete(7) — fixed (throttle, steer) pairs.
    """

    metadata = {"render_modes": []}

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
        self.preprocessor     = ImagePreprocessor()

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

        self.frame_stack = deque(maxlen=N_FRAMES)

        # Channel-first: (C, H, W) = (N_FRAMES, IMG_H, IMG_W)
        # DummyVecEnv keeps this layout — CnnFeatureExtractor receives (B, C, H, W).
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(N_FRAMES, IMG_H, IMG_W),
                dtype=np.uint8,
            ),
            "route_vec": spaces.Box(
                low=-1.0, high=1.0,
                shape=(ROUTE_VEC_DIM,),
                dtype=np.float32,
            ),
        })

        self.prev_steer      = 0.0
        self.step_count      = 0
        self.MAX_STEPS       = 4000
        self.wp_idx          = 0
        self.loops_completed = 0
        self.total_wps       = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy_actors()

        vehicle_actor = self._spawn_vehicle()
        self.prev_steer = 0.0
        self.step_count = 0
        self.wp_idx     = 0

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

        while self.rgb_cam.image is None or self.seg_cam.image is None:
            self.world.tick()

        first_gray = self.preprocessor.to_gray(self.rgb_cam.image)
        for _ in range(N_FRAMES):
            self.frame_stack.append(first_gray)

        return self._get_obs(), {}

    def _spawn_vehicle(self):
        bp_lib     = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        for _ in range(10):
            actor = self.world.try_spawn_actor(vehicle_bp, SPAWN_TRANSFORM)
            if actor is not None:
                self.vehicle            = Vehicle.__new__(Vehicle)
                self.vehicle.world      = self.world
                self.vehicle.vehicle    = actor
                self.vehicle.vehicle_bp = vehicle_bp
                return actor
            self.world.tick()
        raise RuntimeError("Could not spawn vehicle. Check SPAWN_TRANSFORM coordinates.")

    def _get_obs(self):
        # Stack as (N_FRAMES, H, W) — channel-first
        image     = np.stack(list(self.frame_stack), axis=0)
        route_vec = self._make_route_vec()
        return {"image": image, "route_vec": route_vec}

    def _make_route_vec(self) -> np.ndarray:
        wp        = ROUTE[self.wp_idx]
        major_idx = wp["major_idx"]

        one_hot = np.zeros(N_MAJOR, dtype=np.float32)
        one_hot[major_idx] = 1.0

        loc    = self.vehicle.vehicle.get_location()
        target = wp["loc"]
        dx     = target.x - loc.x
        dy     = target.y - loc.y
        angle_to_target  = math.atan2(dy, dx)
        yaw_rad          = math.radians(self.vehicle.vehicle.get_transform().rotation.yaw)
        relative_bearing = (angle_to_target - yaw_rad + math.pi) % (2 * math.pi) - math.pi

        bearing_feat = np.array([
            math.cos(relative_bearing),
            math.sin(relative_bearing),
        ], dtype=np.float32)

        return np.concatenate([one_hot, bearing_feat])

    def step(self, action_idx):
        throttle, target_steer = self.actions[action_idx]
        steer           = 0.6 * target_steer + 0.4 * self.prev_steer
        self.prev_steer = steer

        self.vehicle.apply_control(throttle=throttle, steer=steer)

        spectator = self.world.get_spectator()
        tf        = self.vehicle.vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            tf.location + carla.Location(z=20),
            carla.Rotation(pitch=-90)
        ))

        self.world.tick()
        self.step_count += 1

        rgb_frame = self.rgb_cam.image.copy()
        seg_frame = self.seg_cam.image.copy()
        self.frame_stack.append(self.preprocessor.to_gray(rgb_frame))

        wp_reward, loop_complete = self._check_waypoint()
        reward, done             = self._compute_reward(steer, seg_frame, wp_reward)

        if loop_complete or self.step_count >= self.MAX_STEPS:
            done = True

        velocity  = self.vehicle.vehicle.get_velocity()
        speed_kmh = math.sqrt(velocity.x**2 + velocity.y**2) * 3.6
        self.preprocessor.debug_overlay(
            rgb_frame, seg_frame, speed_kmh,
            cp_idx=ROUTE[self.wp_idx]["major_idx"],
            n_cp=N_MAJOR,
            runs=self.loops_completed,
            next_cp_name=ROUTE[self.wp_idx]["id"],
        )

        return self._get_obs(), reward, done, False, {}

    def _check_waypoint(self):
        loc    = self.vehicle.vehicle.get_location()
        wp     = ROUTE[self.wp_idx]
        target = wp["loc"]
        dist   = math.sqrt((loc.x - target.x)**2 + (loc.y - target.y)**2)

        tl_loc   = TRAFFIC_LIGHTS[wp["major_idx"]]["loc"]
        is_tl_wp = abs(target.x - tl_loc.x) < 0.1 and abs(target.y - tl_loc.y) < 0.1
        radius   = TL_RADIUS if is_tl_wp else CHECKPOINT_RADIUS

        if dist > radius:
            return 0.0, False

        self.total_wps += 1
        collected_idx   = self.wp_idx
        self.wp_idx     = (self.wp_idx + 1) % N_WAYPOINTS

        if collected_idx == N_WAYPOINTS - 1:
            self.loops_completed += 1
            print(f"[env] Loop {self.loops_completed} complete! Total wps: {self.total_wps}")
            return 50.0, True

        if is_tl_wp:
            print(f"[env] TL {TRAFFIC_LIGHTS[wp['major_idx']]['id']} passed!")
            return 10.0, False

        return 2.0, False

    def _compute_reward(self, steer, seg, wp_reward):
        if self.collision_sensor.collision:
            return -100.0, True

        road_ratio, offset, _ = self.preprocessor.road_stats(seg)

        if road_ratio < 0.10:
            return -10.0, True

        reward  = road_ratio * 3.0
        reward += (1.0 - offset) * 2.0
        if offset > 0.35:
            reward -= 1.5

        velocity = self.vehicle.vehicle.get_velocity()
        speed    = math.sqrt(velocity.x**2 + velocity.y**2)
        reward  += speed * 0.1
        reward  -= abs(steer - self.prev_steer) * 0.05

        route_vec    = self._make_route_vec()
        cos_bearing  = float(route_vec[6])
        reward      += 0.3 * max(0.0, cos_bearing)

        reward += wp_reward
        return reward, False

    def _destroy_actors(self):
        for attr in ('rgb_cam', 'seg_cam', 'collision_sensor'):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.destroy()
                setattr(self, attr, None)
        if self.vehicle is not None:
            if self.vehicle.vehicle is not None:
                self.vehicle.vehicle.destroy()
            self.vehicle = None

    def close(self):
        self._destroy_actors()
        cv2.destroyAllWindows()
        settings = self.world.get_settings()
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        settings.no_rendering_mode   = False
        self.world.apply_settings(settings)