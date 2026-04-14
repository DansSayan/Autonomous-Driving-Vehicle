import carla
import numpy as np
import cv2
import math
from collections import deque

from carla_env.vehicle import Vehicle
from carla_env.sensors import CameraSensor, CollisionSensor

import gymnasium as gym
from gymnasium import spaces


# Lookahead distances in metres
LOOKAHEAD_DISTANCES = [3.0, 7.0, 14.0]

# How many numeric state features we pass alongside the image
# [lateral_offset, heading_err, wp1_angle, wp2_angle, wp3_angle, speed]
N_STATE_FEATURES = 6


class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map   = self.world.get_map()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

        self.vehicle          = None
        self.camera           = None
        self.collision_sensor = None

        # 7 discrete actions — symmetric left/right + straight
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

        # Dict observation: camera frames + numeric road state
        self.n_frames = 4
        self.frame_stack = deque(maxlen=self.n_frames)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(84, 84, 3 * self.n_frames),
                dtype=np.uint8
            ),
            # All values normalised to roughly [-1, 1]
            # [lateral_offset, heading_err, wp1_angle, wp2_angle, wp3_angle, speed]
            "state": spaces.Box(
                low=-1.0, high=1.0,
                shape=(N_STATE_FEATURES,),
                dtype=np.float32
            ),
        })

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

        self.camera           = CameraSensor(self.world, vehicle_actor)
        self.collision_sensor = CollisionSensor(self.world, vehicle_actor)
        self.frame_stack.clear()

        while self.camera.image is None:
            self.world.tick()

        first = self.camera.image.copy()
        for _ in range(self.n_frames):
            self.frame_stack.append(first)

        return self._get_obs(), {}

    # ================================================================
    # OBSERVATION
    # ================================================================
    def _get_obs(self):
        image = np.concatenate(list(self.frame_stack), axis=2)
        state = self._get_state_vector()
        return {"image": image, "state": state}

    # ================================================================
    # ROAD STATE VECTOR
    # ================================================================
    def _get_state_vector(self):
        """
        Returns a normalised float32 vector:
          [lateral_offset, heading_err, wp1_angle, wp2_angle, wp3_angle, speed]

        wp_angle at distance d = angle (radians) between the car's current
        heading and the road direction at the waypoint d metres ahead.
        Positive = road turns right, negative = road turns left.
        This is the key signal that lets the agent anticipate curves.
        """
        vehicle_tf = self.vehicle.vehicle.get_transform()
        location   = vehicle_tf.location
        vehicle_yaw = math.radians(vehicle_tf.rotation.yaw)

        waypoint = self.map.get_waypoint(
            location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            return np.zeros(N_STATE_FEATURES, dtype=np.float32)

        # --- Lateral offset (metres, clamp to ±lane_width) ---
        wp_loc  = waypoint.transform.location
        wp_fwd  = waypoint.transform.get_forward_vector()
        dx = location.x - wp_loc.x
        dy = location.y - wp_loc.y
        right_x = -wp_fwd.y
        right_y =  wp_fwd.x
        lateral_offset = dx * right_x + dy * right_y
        lane_half = waypoint.lane_width / 2.0
        lat_norm  = np.clip(lateral_offset / lane_half, -1.0, 1.0)

        # --- Heading error (current wp) ---
        road_yaw    = math.radians(waypoint.transform.rotation.yaw)
        heading_err = vehicle_yaw - road_yaw
        heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
        hdg_norm    = np.clip(heading_err / math.pi, -1.0, 1.0)

        # --- Lookahead waypoint angles ---
        wp_angles = []
        for dist in LOOKAHEAD_DISTANCES:
            future_wps = waypoint.next(dist)
            if future_wps:
                future_yaw = math.radians(future_wps[0].transform.rotation.yaw)
                angle = future_yaw - road_yaw
                angle = (angle + math.pi) % (2 * math.pi) - math.pi
                # Normalise: max meaningful curve angle ~90 deg = pi/2
                wp_angles.append(np.clip(angle / (math.pi / 2), -1.0, 1.0))
            else:
                wp_angles.append(0.0)

        # --- Speed (normalise to ~30 km/h = 8.3 m/s max) ---
        velocity = self.vehicle.vehicle.get_velocity()
        speed    = math.sqrt(velocity.x**2 + velocity.y**2)
        speed_norm = np.clip(speed / 8.3, 0.0, 1.0)

        return np.array(
            [lat_norm, hdg_norm] + wp_angles + [speed_norm],
            dtype=np.float32
        )

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

        image = self.camera.image.copy()
        self.frame_stack.append(image)

        state = self._get_state_vector()

        # Recover real-unit values for reward (state is normalised)
        wp = self.map.get_waypoint(
            self.vehicle.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        lane_half      = (wp.lane_width / 2.0) if wp is not None else 1.75
        lateral_offset = state[0] * lane_half
        heading_err    = state[1] * math.pi
        on_road        = abs(state[0]) < 1.0

        self._show_debug(image, state)

        reward, done = self._compute_reward(steer, lateral_offset, heading_err, on_road, state)

        if self.step_count >= self.MAX_STEPS:
            done = True

        obs = {"image": np.concatenate(list(self.frame_stack), axis=2), "state": state}
        return obs, reward, done, False, {}

    # ================================================================
    # REWARD
    # ================================================================
    def _compute_reward(self, steer, lateral_offset, heading_err, on_road, state):
        if self.collision_sensor.collision:
            return -100.0, True

        if not on_road:
            return -10.0, True

        speed_norm = state[5]
        speed_ms   = speed_norm * 8.3

        # Base: reward for moving forward
        reward = speed_ms * 0.15

        # Lane centering: penalise lateral deviation
        reward -= abs(lateral_offset) * 2.0

        # Heading alignment: penalise pointing away from road
        reward -= abs(heading_err) * 1.5

        # Lookahead alignment bonus:
        # If the agent's steer direction matches the upcoming road curve,
        # give a small bonus. This directly rewards anticipating turns.
        wp1_angle = state[2]   # road curve at 3m ahead (normalised)
        # steer is in [-0.4, 0.4], wp1_angle in [-1, 1]
        # same sign = turning in the right direction
        if wp1_angle * steer > 0:
            reward += 0.3
        elif abs(wp1_angle) > 0.1 and wp1_angle * steer < 0:
            # Turning the wrong way on a curve
            reward -= 0.5

        # Small jitter penalty
        reward -= abs(steer - self.prev_steer) * 0.05

        return reward, False

    # ================================================================
    # DEBUG WINDOW
    # ================================================================
    def _show_debug(self, image, state):
        vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lat   = state[0]
        hdg   = math.degrees(state[1] * math.pi)
        wp1   = math.degrees(state[2] * (math.pi / 2))
        wp2   = math.degrees(state[3] * (math.pi / 2))
        spd   = state[5] * 8.3 * 3.6   # km/h
        on    = "ON " if abs(lat) < 1.0 else "OFF"
        cv2.putText(vis, f"{on} lat={lat:.2f} hdg={hdg:.1f} wp1={wp1:.1f} spd={spd:.0f}kmh",
                    (2, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
        cv2.imshow("AutoCar Debug", vis)
        cv2.waitKey(1)

    # ================================================================
    # CLEANUP
    # ================================================================
    def _destroy_actors(self):
        if self.camera is not None:
            self.camera.destroy()
            self.camera = None
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

    def close(self):
        self._destroy_actors()
        cv2.destroyAllWindows()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
