# 🚗 Autonomous Driving Vehicle — CARLA RL Agent

An end-to-end **Reinforcement Learning** system that trains and runs an autonomous vehicle agent inside the [CARLA](https://carla.org/) simulator. The agent learns to navigate a fixed circular urban route through 6 traffic-light checkpoints using raw camera images and a lightweight route vector, powered by **Proximal Policy Optimization (PPO)** and a custom CNN feature extractor.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Running the Trained Agent](#running-the-trained-agent)
  - [Helper & Utility Scripts](#helper--utility-scripts)
- [Environment Details](#environment-details)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Reward Function](#reward-function)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Route & Checkpoints](#route--checkpoints)
- [Logs & Checkpoints](#logs--checkpoints)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)

---

## Overview

This project implements a **camera-only RL agent** that drives a Tesla Model 3 in the CARLA simulator. The vehicle must:

1. Follow a **predefined circular route** through 6 traffic-light intersections.
2. Stay on the road and avoid collisions.
3. Navigate the full loop repeatedly without human intervention.

The agent never uses GPS, LIDAR, or explicit map data during inference — only visual input from an RGB camera and a compact directional bearing vector.

---

## Architecture

```
Camera Frame (RGB)
       │
       ▼
 ImagePreprocessor
 (Grayscale + Road Stats)
       │
       ▼
 Frame Stack (4 frames)         Route Vector (8-dim)
 [N_FRAMES × H × W]             [one-hot(6) + cos/sin bearing]
       │                                │
       ▼                                ▼
  CNN Branch                     Route MLP Branch
  Conv(4→32) → Conv(32→64)       Linear(8→64) → Linear(64→32)
  → Conv(64→64) → Linear(256)
       │                                │
       └──────────── cat ───────────────┘
                      │
                  [288-dim feature]
                      │
                   PPO Policy
                  (MultiInputPolicy)
                      │
                  Discrete(7) action
              (throttle=0.5, steer ∈ {-0.4,-0.2,-0.1,0,0.1,0.2,0.4})
```

---

## Features

- **Camera-only perception** — no GPS, LIDAR, or HD maps at inference time.
- **Semantic segmentation overlay** — used during training for road-ratio reward shaping; visual debug overlay rendered live.
- **Stacked grayscale frames** (4 frames) for temporal context.
- **Smooth steering** via exponential moving average (`steer = 0.6 × target + 0.4 × prev`).
- **Circular route with 6 traffic-light checkpoints** in the CARLA Town map.
- **Graceful training interruption** — press Enter to stop training and auto-save the model.
- **Route validation tooling** — `check_route.py` and `route_planner.py` for verifying and visualising waypoints before training.
- **TensorBoard logs** included across 5 PPO runs.
- **Pre-trained model** included (`best_model_camera.zip`).

---

## Project Structure

```
Autonomous-Driving-Vehicle/
│
├── main.py                    # Unified CLI entry point (--mode train2 / run2)
├── train2.py                  # PPO training loop (camera-only agent)
├── run2.py                    # Inference script — runs the saved model
├── check_route.py             # Validates all route waypoints against CARLA map
├── route_planner.py           # Draws route visualisation in CARLA spectator view
├── spawn.py                   # Quick vehicle spawn & camera sensor test
├── spawn_helper.py            # Tool to find spawn points & record live waypoints
├── test_env.py                # Basic environment smoke test
├── test_vehicle.py            # Vehicle module unit test
├── traffic_lights.json        # Traffic light IDs and metadata
├── best_model_camera.zip      # Pre-trained PPO model weights
│
├── carla_env/
│   ├── __init__.py
│   ├── env2.py                # Main Gymnasium env (CarlaEnvCamera)
│   ├── sensors.py             # RGBCamera sensor & CollisionSensor wrappers
│   └── vehicle.py             # Vehicle spawning and control abstraction
│
├── models/
│   └── cnn_model.py           # CnnFeatureExtractor + ImagePreprocessor
│
├── utils/
│   └── config.py              # Shared configuration constants
│
└── logs/
    ├── PPO_1/ … PPO_5/        # TensorBoard training logs
    └── (PPO_5 contains latest TFEvent file)
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| **CARLA Simulator** | 0.9.14+ recommended |
| **Python** | 3.8 – 3.12 |
| **PyTorch** | ≥ 2.0 |
| **Stable-Baselines3** | ≥ 2.0 |
| **Gymnasium** | ≥ 0.26 |
| **OpenCV** (`cv2`) | ≥ 4.5 |
| **NumPy** | ≥ 1.21 |

> The CARLA Python API (`carla` package) must match your simulator version. Install it via the `.egg` or `.whl` provided in your CARLA release.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DansSayan/Autonomous-Driving-Vehicle.git
cd Autonomous-Driving-Vehicle
```

### 2. Install Python dependencies

```bash
pip install stable-baselines3[extra] gymnasium opencv-python numpy torch
```

### 3. Install the CARLA Python API

Find the `.whl` file in your CARLA installation (e.g. `PythonAPI/carla/dist/`):

```bash
pip install <path-to-carla>/PythonAPI/carla/dist/carla-*.whl
```

### 4. Launch CARLA

```bash
# On Linux
./CarlaUE4.sh

# On Windows
CarlaUE4.exe
```

Ensure CARLA is running and accepting connections on `localhost:2000` before running any scripts.

---

## Usage

All modes are accessible through the unified `main.py` entry point, or by running scripts directly.

### Training

Train the camera-only PPO agent from scratch:

```bash
python main.py --mode train2
```

Or directly:

```bash
python train2.py
```

- Training runs for up to **500,000 timesteps** by default.
- Press **Enter** at any time to stop training gracefully and save the model.
- The model is saved as `best_model_camera.zip`.

**PPO Hyperparameters used:**

| Parameter | Value |
|---|---|
| Learning rate | `1e-4` |
| Steps per rollout (`n_steps`) | `2048` |
| Batch size | `64` |
| Discount factor (`gamma`) | `0.99` |
| GAE lambda | `0.95` |
| Clip range | `0.1` |
| Entropy coefficient | `0.01` |

---

### Running the Trained Agent

Run inference using the saved (or pre-trained) model:

```bash
python main.py --mode run2
```

Or directly:

```bash
python run2.py
```

The agent will loop the circular route indefinitely, printing episode resets as it completes laps. A live debug window (`AutoCar | Camera RL`) shows the camera feed with road-segmentation overlay, speed, checkpoint index, and loop count.

---

### Helper & Utility Scripts

#### `spawn_helper.py` — Discover spawn points and record route waypoints

```bash
# List all available spawn points in the loaded CARLA map
python spawn_helper.py --mode spawn

# Track a manually-driven vehicle and record checkpoint coordinates
python spawn_helper.py --mode watch
```

#### `route_planner.py` — Visualise the route in CARLA's spectator view

```bash
python route_planner.py
```

Draws coloured debug dots in the CARLA world:
- 🟢 Green — start point  
- 🔴 Red — end point  
- ⚪ White — dense road path (every 2 m)  
- 🩵 Cyan — current checkpoints from `env2.py`  
- 🟡 Yellow — suggested new checkpoints (every ~15 m)

#### `check_route.py` — Validate all waypoints against the CARLA map

```bash
python check_route.py
```

Confirms that every traffic-light waypoint and sub-waypoint in `env2.py` snaps to a drivable road lane. Run this after editing the route before training.

#### `spawn.py` — Quick camera sensor smoke test

```bash
python spawn.py
```

Spawns a random vehicle, attaches an RGB camera sensor, and prints the image shape.

#### `test_env.py` — Environment smoke test

```bash
python test_env.py
```

Resets the environment and runs 250 steps with a fixed action to verify the env is working.

---

## Environment Details

**Class:** `CarlaEnvCamera` (`carla_env/env2.py`)  
**Interface:** [Gymnasium](https://gymnasium.farama.org/) `gym.Env`

### Observation Space

The observation is a `Dict` with two keys:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `image` | `(4, 80, 160)` | `uint8` | 4 stacked grayscale camera frames (channel-first) |
| `route_vec` | `(8,)` | `float32` | One-hot encoding of current segment (6 dims) + `cos(bearing)` + `sin(bearing)` to next waypoint |

- **Image size:** 160 × 80 pixels, field of view 100°, mounted at `(x=2.5, z=1.0)` with a −5° pitch.
- **Segmentation camera:** same transform, used for reward computation and debug overlay only.

### Action Space

`Discrete(7)` — each action maps to a fixed `(throttle=0.5, steer)` pair:

| Action | Steer |
|---|---|
| 0 | −0.40 (hard left) |
| 1 | −0.20 |
| 2 | −0.10 |
| 3 |  0.00 (straight) |
| 4 | +0.10 |
| 5 | +0.20 |
| 6 | +0.40 (hard right) |

Steering is smoothed: `steer = 0.6 × target + 0.4 × prev_steer`.

### Reward Function

Each step reward is composed of:

| Component | Value | Condition |
|---|---|---|
| **Collision** | −100 | Episode terminates |
| **Off-road** (road_ratio < 10%) | −10 | Episode terminates |
| **Road coverage** | `road_ratio × 3.0` | Per step |
| **Lane centering** | `(1 − offset) × 2.0` | Per step |
| **Lane centering penalty** | −1.5 | When `offset > 0.35` |
| **Speed reward** | `speed_m/s × 0.1` | Per step |
| **Steering smoothness** | `−|Δsteer| × 0.05` | Per step |
| **Heading alignment** | `0.3 × max(0, cos_bearing)` | Per step |
| **Sub-waypoint reached** | +2.0 | Per waypoint |
| **Traffic-light waypoint** | +10.0 | Per TL checkpoint |
| **Full loop completed** | +50.0 | Episode ends (success) |

**Episode termination conditions:**
- Collision detected
- Road coverage drops below 10%
- Step limit reached (`MAX_STEPS = 4000`)
- Full loop completed

**Simulator settings:** synchronous mode, fixed delta = 0.05 s (20 Hz), rendering enabled.

---

## Model Architecture

**Class:** `CnnFeatureExtractor` (`models/cnn_model.py`)  
**Base:** `stable_baselines3.common.torch_layers.BaseFeaturesExtractor`

```
CNN Branch (processes image):
  Conv2d(4, 32, kernel=5, stride=2, padding=2)  → ReLU
  Conv2d(32, 64, kernel=3, stride=2, padding=1) → ReLU
  Conv2d(64, 64, kernel=3, stride=2, padding=1) → ReLU
  Flatten → [12800]
  Linear(12800, 256) → ReLU                     → [256]

Route MLP Branch (processes route_vec):
  Linear(8, 64)  → ReLU
  Linear(64, 32) → ReLU                         → [32]

Output: Concatenate → [288-dim feature vector]
```

**Spatial resolution after convolutions:**  
- Height: 80 → 40 → 20 → 10  
- Width: 160 → 80 → 40 → 20  
- Flat CNN size: `64 × 10 × 20 = 12,800`

---

## Route & Checkpoints

The agent navigates a **closed circular loop** through 6 named traffic-light intersections:

```
id22 → id18 → id21 → id19 → id11 → id9 → id22 → ...
```

Each segment is defined by dense intermediate sub-waypoints in `env2.py → _SEGMENTS`. The full route contains ~21 waypoints total.

**Waypoint collection radii:**
- Normal sub-waypoints: 12 m
- Traffic-light waypoints: 14 m

**Vehicle:** `vehicle.tesla.model3`  
**Spawn transform:** `x=−113.80, y=2.0, z=0.5, yaw=90°`

---

## Logs & Checkpoints

TensorBoard training logs are stored in `logs/PPO_1` through `logs/PPO_5`.

To view training curves:

```bash
tensorboard --logdir logs/
```

A pre-trained model is included at the root of the repository:

```
best_model_camera.zip   (~40 MB)
```

---

## Known Limitations

- **CARLA map dependency:** The hardcoded route coordinates are calibrated for a specific CARLA town map. Loading a different map will require re-running `route_planner.py` and updating `_SEGMENTS` in `env2.py`.
- **Single vehicle only:** The environment spawns one Tesla Model 3 at a fixed location. No NPC traffic or pedestrians are configured by default.
- **No traffic-light state detection:** The agent currently receives a bonus for *reaching* traffic-light waypoints but does not observe the actual signal state (red/green).
- **Fixed throttle:** All 7 actions share `throttle=0.5`. Variable speed control is not implemented.
- **Synchronous mode required:** The env forces CARLA into synchronous mode. Running multiple environments in parallel requires careful handling.

---

## Contributing

Contributions, issues, and feature requests are welcome. Some areas open for improvement:

- Add traffic-light state to the observation space.
- Support variable throttle actions.
- Add NPC traffic for more realistic training scenarios.
- Implement a curriculum that gradually increases route complexity.
- Package dependencies into a `requirements.txt` or `pyproject.toml`.

---

*Built with [CARLA](https://carla.org/), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), and [Gymnasium](https://gymnasium.farama.org/).*
