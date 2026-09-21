import argparse
from pathlib import Path
from src.config import EnvConfig
from src.evaluation import evaluate, summarize
from src.plots import plot_evaluation

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--episodes", type=int, default=50)
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--out", default="results/custom")
a = p.parse_args()

out = Path(a.out)
csv_path = out / "episodes.csv"
rows = evaluate(a.model, EnvConfig(seed=a.seed), a.episodes, a.seed, True, csv_path)
print(summarize(rows))
plot_evaluation(csv_path, out)
