from src.config import EnvConfig, TrainConfig
ENV = EnvConfig(n_obstacles=5, seed=1038, spacing_min=0.45)
TRAIN = TrainConfig(total_timesteps=1000000, seed=1038,
                    train_filter=False)
