from src.config import EnvConfig, TrainConfig
ENV = EnvConfig(n_obstacles=5, seed=82, spacing_min=0.45)
TRAIN = TrainConfig(total_timesteps=150000, seed=82,
                    train_filter=True)
