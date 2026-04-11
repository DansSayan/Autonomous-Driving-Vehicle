from carla_env.env import CarlaEnv
import time

env = CarlaEnv()

state = env.reset()
print("Initial state shape:", state.shape)

for _ in range(250):
    action = 0
    state, reward, done = env.step(action)

    print("Step state:", state.shape, "Reward:", reward)

    if done:
        print("Episode ended (collision)")
        break

env.close()