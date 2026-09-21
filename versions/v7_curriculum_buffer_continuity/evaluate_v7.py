from pathlib import Path
from config import ENV
from src.evaluation import evaluate, summarize
from src.plots import plot_evaluation

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "checkpoints" / "v7_curriculum_buffer_continuity" / "sac_model.zip"
OUT = ROOT / "results" / "v7_curriculum_buffer_continuity"

if __name__ == "__main__":
    if not MODEL.exists():
        raise FileNotFoundError(f"{MODEL} not found. Train this version first.")
    csv = OUT / "episodes.csv"
    rows = evaluate(MODEL, ENV, 50, 1000, True, csv)
    print(summarize(rows))
    plot_evaluation(csv, OUT)
