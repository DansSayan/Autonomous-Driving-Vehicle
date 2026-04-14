from stable_baselines3 import PPO
from carla_env.env import CarlaEnv

env = CarlaEnv()

model = PPO(
    "MultiInputPolicy",   # handles Dict observation (image + state vector)
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,        # slightly higher entropy to prevent early convergence
)

model.learn(total_timesteps=200_000)
model.save("best_model")
env.close()
