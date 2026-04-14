"""
spawn_helper.py
---------------
Run this ONCE to find your desired spawn point and route checkpoints.

Usage:
    python spawn_helper.py --mode spawn     → prints all available spawn points with index
    python spawn_helper.py --mode watch     → prints the car's live location every second
                                              so you can drive it manually and record waypoints

After running, copy the spawn transform and checkpoint coordinates into env2.py.
"""

import carla
import argparse
import time


def get_client():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    return client


def list_spawn_points():
    client = get_client()
    world  = client.get_world()
    spawn_points = world.get_map().get_spawn_points()
    print(f"\nTotal spawn points: {len(spawn_points)}\n")
    for i, sp in enumerate(spawn_points):
        print(f"[{i:3d}]  x={sp.location.x:8.2f}  y={sp.location.y:8.2f}  z={sp.location.z:6.2f}"
              f"  yaw={sp.rotation.yaw:7.2f}")
    print("\nCopy the index you want and set SPAWN_POINT_INDEX in env2.py")


def watch_live():
    """
    Spawns a vehicle you can control with the CARLA manual_control.py,
    then prints its location every second so you can record checkpoints
    along your desired route.
    Press Ctrl+C to stop.
    """
    client = get_client()
    world  = client.get_world()

    # Find the first existing vehicle (assumes you launched manual_control.py)
    vehicles = world.get_actors().filter('vehicle.*')
    if not vehicles:
        print("No vehicle found. Launch manual_control.py first, then run this.")
        return

    vehicle = vehicles[0]
    print(f"Tracking vehicle: {vehicle.type_id}")
    print("Drive along your desired route and note the coordinates below.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            tf  = vehicle.get_transform()
            loc = tf.location
            rot = tf.rotation
            print(f"carla.Transform(carla.Location(x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}), "
                  f"carla.Rotation(yaw={rot.yaw:.2f}))  ← checkpoint")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nDone. Paste the printed transforms as ROUTE_CHECKPOINTS in env2.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["spawn", "watch"], required=True)
    args = parser.parse_args()

    if args.mode == "spawn":
        list_spawn_points()
    elif args.mode == "watch":
        watch_live()
