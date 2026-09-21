from pathlib import Path
from src.config import EnvConfig, TrainConfig
from src.training import train

ROOT = Path(__file__).resolve().parents[2]
ENV = EnvConfig(n_obstacles=5, spacing_min=0.45, seed=1038)
TRAIN = TrainConfig(total_timesteps=1_000_000, seed=1038,
                    train_filter=False, network_arch=(256,256))

if __name__ == "__main__":
    print("v9: no continuous CBF action substitution during SAC learning.")
    print("CBF filter is applied at deployment/evaluation.")
    print(train(ENV, TRAIN, ROOT/"checkpoints"/"v9_corrected_training"))
