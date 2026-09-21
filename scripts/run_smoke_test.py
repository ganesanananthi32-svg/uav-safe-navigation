from src.config import EnvConfig
from src.env import DynamicUAVEnv

env = DynamicUAVEnv(EnvConfig(n_obstacles=2, seed=123))
obs, _ = env.reset()
for _ in range(10):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        break
print("Smoke test passed.")
print("Observation shape:", obs.shape)
print("Last info:", info)
env.close()
