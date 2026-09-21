from src.config import EnvConfig, TrainConfig
ENV = EnvConfig(n_obstacles=5, seed=42, spacing_min=0.0)
TRAIN = TrainConfig(total_timesteps=150000, seed=42,
                    train_filter=True)
