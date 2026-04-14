import threading
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from carla_env.env2 import CarlaEnvCamera
from models.cnn_model import CnnFeatureExtractor


class StopTrainingCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self._stop_requested = False
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        print("\n[train2] Training started. Press Enter to stop and save.\n")
        try:
            input()
        except EOFError:
            pass
        self._stop_requested = True
        print("\n[train2] Stop requested — finishing rollout then saving...")

    def _on_step(self) -> bool:
        return not self._stop_requested


def main():
    env = DummyVecEnv([lambda: CarlaEnvCamera()])

    policy_kwargs = dict(
        features_extractor_class=CnnFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=288),
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
    )

    callback = StopTrainingCallback()

    try:
        model.learn(total_timesteps=500_000, callback=callback)
    except KeyboardInterrupt:
        print("\n[train2] Ctrl+C received.")
    finally:
        print("[train2] Saving model...")
        model.save("best_model_camera")
        print("[train2] Saved to best_model_camera.zip")
        env.close()


if __name__ == "__main__":
    main()