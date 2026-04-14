"""
route_planner.py
----------------
Draws debug dots in the CARLA world between the two traffic lights
so you can visually verify the route and adjust checkpoints in env2.py.

Usage:
    python route_planner.py

- Green dot  = start (traffic light ID=10)
- Red dot    = end   (traffic light ID=23)
- Yellow dots = intermediate checkpoints
- White dots  = interpolated path points (every 2m along the road)

The script also prints the waypoint coordinates so you can copy
the ones that look right into ROUTE_CHECKPOINTS in env2.py.

Keep CARLA running and look at the spectator view to see the dots.
Press Ctrl+C to exit.
"""

import carla
import time

# ── Traffic light coordinates ──────────────────────────────────────
START = carla.Location(x=115.45, y=35.04, z=0.5)
END   = carla.Location(x=-119.24, y=19.0, z=0.5)

# ── Current checkpoints from env2.py (shown in cyan) ───────────────
CURRENT_CHECKPOINTS = [
    carla.Location(x= 90.0,  y=34.5,  z=0.5),
    carla.Location(x= 60.0,  y=34.5,  z=0.5),
    carla.Location(x= 30.0,  y=34.5,  z=0.5),
    carla.Location(x=  0.0,  y=34.0,  z=0.5),
    carla.Location(x=-31.64, y=33.59, z=0.5),
]

DOT_SIZE      = 0.3    # metres radius of each debug sphere
DOT_LIFETIME  = 20.0   # seconds each dot stays visible (redrawn in loop)
STEP_METRES   = 2.0    # spacing between interpolated road waypoints


def draw_route(world, carla_map, debug):
    # ── Snap START and END to nearest drivable waypoint ────────────
    wp_start = carla_map.get_waypoint(START, project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    wp_end   = carla_map.get_waypoint(END,   project_to_road=True,
                                      lane_type=carla.LaneType.Driving)

    if wp_start is None or wp_end is None:
        print("ERROR: Could not snap start/end to a drivable waypoint.")
        print("Check that your CARLA map matches the coordinates.")
        return

    # ── Walk waypoints from start toward end ───────────────────────
    # We follow the road topology using waypoint.next() until we get
    # close to the end point or run out of road.
    MAX_STEPS   = 500
    path_wps    = [wp_start]
    current_wp  = wp_start

    for _ in range(MAX_STEPS):
        loc = current_wp.transform.location
        dist_to_end = loc.distance(wp_end.transform.location)

        if dist_to_end < STEP_METRES * 2:
            path_wps.append(wp_end)
            break

        nexts = current_wp.next(STEP_METRES)
        if not nexts:
            print("WARNING: Road ended before reaching the end point.")
            break

        # Pick the next waypoint closest to the end point
        current_wp = min(nexts, key=lambda w: w.transform.location.distance(
                                              wp_end.transform.location))
        path_wps.append(current_wp)

    print(f"\nFound {len(path_wps)} waypoints along the route.\n")

    # ── Print snapped coordinates ───────────────────────────────────
    print("=" * 60)
    print("Snapped route waypoints (copy into ROUTE_CHECKPOINTS):")
    print("=" * 60)

    # Print every ~15m as a suggested checkpoint
    step = max(1, int(15.0 / STEP_METRES))
    suggested = []
    for i, wp in enumerate(path_wps):
        if i % step == 0 or i == len(path_wps) - 1:
            loc = wp.transform.location
            suggested.append(loc)
            print(f"  carla.Location(x={loc.x:7.2f}, y={loc.y:7.2f}, z=0.5),")

    print("=" * 60)
    print(f"\nSuggested ROUTE_CHECKPOINTS ({len(suggested)} points):")
    print("Paste these into env2.py → ROUTE_CHECKPOINTS\n")

    # ── Draw dots in CARLA world ────────────────────────────────────
    # Start = green
    debug.draw_point(
        wp_start.transform.location + carla.Location(z=0.5),
        size=DOT_SIZE * 2,
        color=carla.Color(0, 255, 0),
        life_time=DOT_LIFETIME
    )
    debug.draw_string(
        wp_start.transform.location + carla.Location(z=1.5),
        "START (TL #10)",
        color=carla.Color(0, 255, 0),
        life_time=DOT_LIFETIME
    )

    # End = red
    debug.draw_point(
        wp_end.transform.location + carla.Location(z=0.5),
        size=DOT_SIZE * 2,
        color=carla.Color(255, 0, 0),
        life_time=DOT_LIFETIME
    )
    debug.draw_string(
        wp_end.transform.location + carla.Location(z=1.5),
        "END (TL #23)",
        color=carla.Color(255, 0, 0),
        life_time=DOT_LIFETIME
    )

    # Interpolated path = white dots
    for wp in path_wps[1:-1]:
        debug.draw_point(
            wp.transform.location + carla.Location(z=0.3),
            size=DOT_SIZE * 0.5,
            color=carla.Color(200, 200, 200),
            life_time=DOT_LIFETIME
        )

    # Current env2.py checkpoints = cyan
    for i, loc in enumerate(CURRENT_CHECKPOINTS):
        debug.draw_point(
            loc + carla.Location(z=0.5),
            size=DOT_SIZE,
            color=carla.Color(0, 220, 220),
            life_time=DOT_LIFETIME
        )
        debug.draw_string(
            loc + carla.Location(z=1.2),
            f"CP{i}",
            color=carla.Color(0, 220, 220),
            life_time=DOT_LIFETIME
        )

    # Suggested checkpoints = yellow
    for i, loc in enumerate(suggested):
        debug.draw_point(
            loc + carla.Location(z=0.8),
            size=DOT_SIZE,
            color=carla.Color(255, 220, 0),
            life_time=DOT_LIFETIME
        )

    return path_wps, suggested


def main():
    client     = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world      = client.get_world()
    carla_map  = world.get_map()
    debug      = world.debug

    # Move spectator to see the full route from above
    mid_x = (START.x + END.x) / 2
    mid_y = (START.y + END.y) / 2
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(x=mid_x, y=mid_y, z=80),
        carla.Rotation(pitch=-90)
    ))

    print("Drawing route in CARLA world...")
    print(f"  Start : x={START.x}, y={START.y}  (traffic light ID=10)")
    print(f"  End   : x={END.x},  y={END.y}  (traffic light ID=23)")
    print()
    print("Legend:")
    print("  Green  = start point")
    print("  Red    = end point")
    print("  White  = road path (every 2m)")
    print("  Cyan   = current checkpoints in env2.py")
    print("  Yellow = suggested new checkpoints (every ~15m)")
    print()

    path_wps, suggested = draw_route(world, carla_map, debug)

    print("\nDots drawn. Check the CARLA spectator window.")
    print("Dots refresh every 15 seconds. Press Ctrl+C to exit.\n")

    # Keep redrawing so dots don't expire
    try:
        while True:
            time.sleep(DOT_LIFETIME - 5)
            draw_route(world, carla_map, debug)
    except KeyboardInterrupt:
        print("Exited.")


if __name__ == "__main__":
    main()
