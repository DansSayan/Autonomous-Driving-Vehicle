import argparse

def main():
    parser = argparse.ArgumentParser(description="AutoCar CARLA RL Agent")
    parser.add_argument(
        "--mode",
        choices=["train", "run", "train2", "run2"],
        required=True,
        help="train/run = waypoint-assisted agent | train2/run2 = camera-only agent"
    )
    args = parser.parse_args()

    if args.mode == "train":
        import train_rl
    elif args.mode == "run":
        import run_model
    elif args.mode == "train2":
        import train2
        train2.main()
    elif args.mode == "run2":
        import run2

if __name__ == "__main__":
    main()
