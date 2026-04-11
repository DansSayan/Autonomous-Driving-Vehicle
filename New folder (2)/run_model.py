from stable_baselines3 import PPO
from carla_env.env import CarlaEnv

env   = CarlaEnv()
model = PPO.load("best_model", env=env)

obs, _ = env.reset()
print("Running trained agent...")

for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    if done:
        print("Episode ended, resetting...")
        obs, _ = env.reset()

env.close()
