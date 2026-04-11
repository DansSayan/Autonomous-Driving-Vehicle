import carla
import random
import time
from carla_env.sensors import CameraSensor

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)

world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Choose a vehicle
vehicle_bp = blueprint_library.filter('vehicle')[0]

# Get spawn points
spawn_points = world.get_map().get_spawn_points()

# Spawn vehicle
vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
camera = CameraSensor(world, vehicle)

print("Vehicle spawned!")

# Let it drive forward a bit
vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

import time

for _ in range(50):
    if camera.image is not None:
        print(camera.image.shape)  # should print (84, 84, 3)
    time.sleep(0.1)

camera.destroy()
# Destroy vehicle after
vehicle.destroy()
print("Vehicle destroyed!")