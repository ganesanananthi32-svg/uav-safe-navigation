from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from .env import DynamicUAVEnv
from .wrappers import TrainingShield

def make_env(env_cfg, training_filter=False):
    env = DynamicUAVEnv(env_cfg)
    env = Monitor(env)
    if training_filter:
        env = TrainingShield(env, env_cfg.safe_distance,
                             env_cfg.gamma, env_cfg.max_accel)
    return env

def train(env_cfg, train_cfg, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(env_cfg, train_cfg.train_filter)
    model = SAC(
        "MlpPolicy", env,
        learning_rate=train_cfg.learning_rate,
        buffer_size=train_cfg.buffer_size,
        learning_starts=train_cfg.learning_starts,
        batch_size=train_cfg.batch_size,
        gamma=train_cfg.gamma, tau=train_cfg.tau,
        seed=train_cfg.seed,
        policy_kwargs={"net_arch": list(train_cfg.network_arch)},
        verbose=1)
    model.learn(total_timesteps=train_cfg.total_timesteps, log_interval=10)
    model.save(str(output_dir / "sac_model"))
    env.close()
    return output_dir / "sac_model.zip"
