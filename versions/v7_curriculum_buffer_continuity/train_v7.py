from pathlib import Path
from stable_baselines3 import SAC
from src.config import EnvConfig, TrainConfig
from src.training import make_env

ROOT = Path(__file__).resolve().parents[2]
model = None
for stage, nobs in enumerate([2, 3, 5], 1):
    env_cfg = EnvConfig(n_obstacles=nobs, seed=700+stage)
    train_cfg = TrainConfig(total_timesteps=100_000, seed=700+stage,
                            train_filter=True)
    env = make_env(env_cfg, True)
    if model is None:
        model = SAC("MlpPolicy", env, learning_rate=3e-4,
                    buffer_size=100_000, learning_starts=1000,
                    batch_size=256, gamma=.99, tau=.005,
                    seed=train_cfg.seed,
                    policy_kwargs={"net_arch":[256,256]}, verbose=1)
    else:
        model.set_env(env)
    model.learn(total_timesteps=100_000, reset_num_timesteps=False)
    out = ROOT/"checkpoints"/"v7_curriculum_buffer_continuity"
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out/f"stage_{stage}_{nobs}_obstacles"))
    env.close()
    print(f"Completed stage {stage}: {nobs} obstacles; replay buffer retained.")
