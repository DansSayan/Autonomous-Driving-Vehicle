import carla
import numpy as np


class CameraSensor:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.image = None

        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        camera_bp.set_attribute('image_size_x', '84')
        camera_bp.set_attribute('image_size_y', '84')
        camera_bp.set_attribute('fov', '110')

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.sensor = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        self.sensor.listen(lambda data: self.process_image(data))

    def process_image(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((84, 84, 4))[:, :, :3]  # remove alpha channel
        self.image = img

    def destroy(self):
        self.sensor.destroy()


class CollisionSensor:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.collision = False

        blueprint_library = world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')

        self.sensor = world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=vehicle
        )

        self.sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        self.collision = True

    def destroy(self):
        self.sensor.destroy()