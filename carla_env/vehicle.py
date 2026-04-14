import carla
import random

class Vehicle:
    def __init__(self, world):
        self.world = world
        self.vehicle = None

        blueprint_library = world.get_blueprint_library()
        self.vehicle_bp = blueprint_library.filter('vehicle')[0]

    def spawn(self):
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(
            self.vehicle_bp,
            random.choice(spawn_points)
        )
        return self.vehicle

    def apply_control(self, throttle=0.5, steer=0.0, brake=0.0):
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake
        )
        self.vehicle.apply_control(control)

    def destroy(self):
        if self.vehicle is not None:
            self.vehicle.destroy()