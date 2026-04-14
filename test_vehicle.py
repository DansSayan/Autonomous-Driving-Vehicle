import carla
import time
from carla_env.vehicle import Vehicle

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)

world = client.get_world()

# Create vehicle object
vehicle_obj = Vehicle(world)
vehicle = vehicle_obj.spawn()

print("Vehicle spawned via module!")

# Move forward
for _ in range(50):
    vehicle_obj.apply_control(throttle=0.5, steer=0.0)
    time.sleep(0.1)

vehicle_obj.destroy()
print("Vehicle destroyed!")