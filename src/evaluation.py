from pathlib import Path
import csv
import numpy as np
from stable_baselines3 import SAC
from .env import DynamicUAVEnv
from .wrappers import DeploymentShield

def evaluate(model_path, env_cfg, episodes=50, seed=1000,
             deployment_filter=True, output_csv=None):
    env = DynamicUAVEnv(env_cfg)
    if deployment_filter:
        env = DeploymentShield(env, env_cfg.safe_distance,
                               env_cfg.gamma, env_cfg.max_accel)
    model = SAC.load(model_path)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total_reward, last_info, steps = False, 0.0, {}, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            last_info, steps = info, steps + 1
            done = term or trunc
        rows.append({
            "episode": ep + 1, "steps": steps, "reward": total_reward,
            "goal": int(last_info.get("goal", False)),
            "collision": int(last_info.get("collision", False)),
            "timeout": int(last_info.get("timeout", False)),
            "final_distance": last_info.get("final_distance", np.nan),
            "min_obstacle_distance": last_info.get(
                "min_obstacle_distance", np.nan)})
    env.close()
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return rows

def summarize(rows):
    n = len(rows)
    return {
        "episodes": n,
        "goal_rate": sum(r["goal"] for r in rows) / n,
        "collision_rate": sum(r["collision"] for r in rows) / n,
        "timeout_rate": sum(r["timeout"] for r in rows) / n,
        "mean_final_distance": float(np.mean([r["final_distance"] for r in rows])),
        "mean_min_obstacle_distance": float(
            np.mean([r["min_obstacle_distance"] for r in rows])),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
    }
