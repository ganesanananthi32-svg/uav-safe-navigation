from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_evaluation(csv_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    for col, ylabel, title, filename in [
        ("final_distance", "Final distance (m)", "Episode final distance",
         "final_distance.png"),
        ("min_obstacle_distance", "Minimum obstacle distance (m)",
         "Obstacle clearance", "obstacle_clearance.png")]:
        plt.figure()
        plt.plot(df["episode"], df[col])
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=160)
        plt.close()
