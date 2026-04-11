from stable_baselines3 import PPO
from carla_env.env2 import CarlaEnvCamera

env = CarlaEnvCamera()

model = PPO(
    "CnnPolicy",   # pure CNN — observation is stacked grayscale frames only
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,
)

model.learn(total_timesteps=200_000)
model.save("best_model_camera")
env.close()
