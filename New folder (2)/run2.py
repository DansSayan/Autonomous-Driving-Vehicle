from stable_baselines3 import PPO
from carla_env.env2 import CarlaEnvCamera

env   = CarlaEnvCamera()
model = PPO.load("best_model_camera", env=env)

obs, _ = env.reset()
print("Running camera-only agent...")

for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    if done:
        print("Episode ended, resetting...")
        obs, _ = env.reset()

env.close()
