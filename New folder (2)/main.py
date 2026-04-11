import argparse
import sys
import os

# Add the project root to the Python path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_model():
    """Run the training script."""
    print("Starting training...")
    import train_rl
    # The training code is already in train_rl.py, so this just imports it to run

def run_model():
    """Run the trained model."""
    print("Running trained model...")
    import run_model
    # The running code is already in run_model.py

def train2_model():
    """Run the second training script."""
    print("Starting training2...")
    import train2
    # Assuming train2.py exists

def run2_model():
    """Run the second model."""
    print("Running trained model2...")
    import run2
    # Assuming run2.py exists

def spawn_vehicle():
    """Spawn a vehicle in CARLA."""
    print("Spawning vehicle...")
    import spawn
    # Assuming spawn.py has the spawning logic

def main():
    parser = argparse.ArgumentParser(description="Autonomous Car Project Main Script")
    parser.add_argument(
        "mode",
        choices=["train", "run", "train2", "run2", "spawn"],
        help="Choose what to do: train/run = waypoint-assisted agent | train2/run2 = camera-only agent | spawn = spawn vehicle"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model()
    elif args.mode == "run":
        run_model()
    elif args.mode == "train2":
        train2_model()
    elif args.mode == "run2":
        run2_model()
    elif args.mode == "spawn":
        spawn_vehicle()
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()
