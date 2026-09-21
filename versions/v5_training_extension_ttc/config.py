from src.config import EnvConfig, TrainConfig
ENV = EnvConfig(n_obstacles=5, seed=52, spacing_min=0.0)
TRAIN = TrainConfig(total_timesteps=300000, seed=52,
                    train_filter=True)
