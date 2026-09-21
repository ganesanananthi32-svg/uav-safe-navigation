from src.config import EnvConfig
from src.env import DynamicUAVEnv

def test_reset_and_step():
    env = DynamicUAVEnv(EnvConfig(n_obstacles=5))
    obs, _ = env.reset(seed=7)
    assert obs.shape == env.observation_space.shape
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape
