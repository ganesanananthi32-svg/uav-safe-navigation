from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path("results")
rows = []
for csv in root.glob("v*/episodes.csv"):
    df = pd.read_csv(csv)
    rows.append({
        "version": csv.parent.name,
        "goal_rate": df.goal.mean(),
        "collision_rate": df.collision.mean(),
        "timeout_rate": df.timeout.mean(),
        "mean_final_distance": df.final_distance.mean(),
        "mean_min_obstacle_distance": df.min_obstacle_distance.mean(),
    })
if not rows:
    raise SystemExit("No fresh evaluation files found.")
summary = pd.DataFrame(rows).sort_values("version")
print(summary.to_string(index=False))
summary.to_csv(root/"fresh_version_summary.csv", index=False)
plt.figure()
plt.plot(summary["version"], summary["goal_rate"]*100, marker="o")
plt.ylabel("Goal completion (%)")
plt.xlabel("Version")
plt.title("Fresh experiment goal completion")
plt.tight_layout()
plt.savefig(root/"fresh_goal_completion.png", dpi=160)
plt.close()
