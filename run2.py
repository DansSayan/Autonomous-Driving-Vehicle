from stable_baselines3 import PPO
from carla_env.env2 import CarlaEnvCamera
from models.cnn_model import CnnFeatureExtractor

env   = CarlaEnvCamera()
model = PPO.load(
    "best_model_camera",
    env=env,
    custom_objects={
        "features_extractor_class": CnnFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 288},
    }
)

obs, _ = env.reset()
print("Running agent on circular route: id22->id18->id21->id19->id11->id9->id22->...")

try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        if done:
            print("Episode ended — resetting...")
            obs, _ = env.reset()
except KeyboardInterrupt:
    print("Stopped.")
finally:
    env.close()