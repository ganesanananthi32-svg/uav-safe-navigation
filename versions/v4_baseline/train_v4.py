from pathlib import Path
from config import ENV, TRAIN
from src.training import train

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    print(train(ENV, TRAIN, ROOT / "checkpoints" / "v4_baseline"))
