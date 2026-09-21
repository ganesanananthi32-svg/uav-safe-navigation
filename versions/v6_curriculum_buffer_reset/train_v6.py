from pathlib import Path
from stable_baselines3 import SAC
from src.config import EnvConfig, TrainConfig
from src.training import make_env

ROOT = Path(__file__).resolve().parents[2]
for stage, nobs in enumerate([2, 3, 5], 1):
    env_cfg = EnvConfig(n_obstacles=nobs, seed=600+stage)
    train_cfg = TrainConfig(total_timesteps=100_000, seed=600+stage,
                            train_filter=True)
    env = make_env(env_cfg, True)
    model = SAC("MlpPolicy", env, learning_rate=3e-4,
                buffer_size=100_000, learning_starts=1000, batch_size=256,
                gamma=.99, tau=.005, seed=train_cfg.seed,
                policy_kwargs={"net_arch":[256,256]}, verbose=1)
    model.learn(total_timesteps=100_000)
    out = ROOT/"checkpoints"/"v6_curriculum_buffer_reset"
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out/f"stage_{stage}_{nobs}_obstacles"))
    env.close()
    print(f"Completed stage {stage}: {nobs} obstacles; replay buffer reset.")
