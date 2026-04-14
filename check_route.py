"""
check_route.py
==============
Verifies that all 6 traffic light checkpoints and dense sub-waypoints
are on drivable roads.

Usage:
    python check_route.py

This helps you verify the route is valid before training.
"""

import carla
import math


TRAFFIC_LIGHTS = [
    {"id": "id22", "x": -113.80, "y":   5.16},
    {"id": "id18", "x": -113.96, "y":  19.06},
    {"id": "id21", "x":  -32.90, "y":  28.40},
    {"id": "id19", "x":  -58.52, "y": 140.35},
    {"id": "id11", "x":  109.70, "y":  21.20},
    {"id": "id9",  "x":  -46.25, "y": -68.48},
]

SEGMENTS = [
    [(-113.80, 10.0), (-113.90, 14.5), (-113.96, 19.06)],
    [(-105.0, 22.0), (-85.0, 27.0), (-65.0, 29.0), (-45.0, 28.5), (-32.90, 28.40)],
    [(-45.0, 60.0), (-50.0, 90.0), (-55.0, 120.0), (-58.52, 140.35)],
    [(10.0, 140.0), (60.04, 69.83), (100.0, 55.0), (109.70, 21.20)],
    [(79.95, 13.41), (30.01, -57.41), (-10.0, -55.0), (-46.25, -68.48)],
    [(-69.92, -58.22), (-95.0, -25.0), (-110.0, -8.0), (-113.80, 5.16)],
]


def check_route():
    client     = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world      = client.get_world()
    carla_map  = world.get_map()

    print("\n" + "=" * 70)
    print("TRAFFIC LIGHT VALIDATION")
    print("=" * 70)

    all_valid = True

    for i, tl in enumerate(TRAFFIC_LIGHTS):
        loc = carla.Location(x=tl["x"], y=tl["y"], z=0.5)
        wp  = carla_map.get_waypoint(loc, project_to_road=True,
                                     lane_type=carla.LaneType.Driving)

        if wp is None:
            print(f"\n[{i}] {tl['id']:>6}  ({tl['x']:>7.1f}, {tl['y']:>7.1f})")
            print("     ❌ NOT on a drivable road!")
            all_valid = False
            continue

        snapped = wp.transform.location
        dist    = math.sqrt((loc.x - snapped.x)**2 + (loc.y - snapped.y)**2)

        status = "✓" if dist < 5.0 else "⚠"
        print(f"\n[{i}] {tl['id']:>6}  ({tl['x']:>7.1f}, {tl['y']:>7.1f})")
        print(f"     {status} Snapped to ({snapped.x:>7.1f}, {snapped.y:>7.1f})")
        print(f"     Distance: {dist:.2f}m  |  Lane width: {wp.lane_width:.2f}m")

        if dist > 5.0:
            print(f"     ⚠ WARNING: {dist:.1f}m from nearest road.")
            print(f"     Consider: carla.Location(x={snapped.x:.2f}, y={snapped.y:.2f}, z=0.5)")

    print("\n" + "=" * 70)
    print("SUB-WAYPOINT VALIDATION")
    print("=" * 70)

    total_wps = 0
    bad_wps   = 0

    for seg_idx, segment in enumerate(SEGMENTS):
        tl_from = TRAFFIC_LIGHTS[seg_idx]["id"]
        tl_to   = TRAFFIC_LIGHTS[(seg_idx + 1) % len(TRAFFIC_LIGHTS)]["id"]
        print(f"\n{tl_from} → {tl_to}  ({len(segment)} waypoints)")

        for i, (x, y) in enumerate(segment):
            loc = carla.Location(x=x, y=y, z=0.5)
            wp  = carla_map.get_waypoint(loc, project_to_road=True,
                                         lane_type=carla.LaneType.Driving)

            total_wps += 1

            if wp is None:
                print(f"  [{i}] ({x:>7.1f}, {y:>7.1f})  ❌ NOT on road")
                bad_wps += 1
                continue

            snapped = wp.transform.location
            dist    = math.sqrt((loc.x - snapped.x)**2 + (loc.y - snapped.y)**2)

            if dist > 8.0:
                print(f"  [{i}] ({x:>7.1f}, {y:>7.1f})  ⚠ {dist:.1f}m from road")
                print(f"       Suggest: ({snapped.x:.2f}, {snapped.y:.2f})")
                bad_wps += 1

    print("\n" + "=" * 70)
    print(f"Total sub-waypoints: {total_wps}")
    print(f"Problematic waypoints: {bad_wps}")

    if bad_wps == 0:
        print("✓ All waypoints are on drivable roads.")
    else:
        print(f"⚠ {bad_wps} waypoints need adjustment in env2.py → _SEGMENTS")

    print("=" * 70)


if __name__ == "__main__":
    check_route()
