from pathlib import Path
from src.config import EnvConfig, TrainConfig
from src.training import train

ROOT = Path(__file__).resolve().parents[2]
ENV = EnvConfig(n_obstacles=5, spacing_min=0.45, seed=82)
TRAIN = TrainConfig(total_timesteps=150_000, seed=82, train_filter=True)

if __name__ == "__main__":
    print("v8 minimum obstacle spacing = 0.45 m")
    print(train(ENV, TRAIN, ROOT/"checkpoints"/"v8_spacing_fix"))
