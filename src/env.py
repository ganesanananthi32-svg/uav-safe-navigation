import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DynamicUAVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_cfg):
        super().__init__()
        self.cfg = env_cfg
        self.rng = np.random.default_rng(env_cfg.seed)
        self.action_space = spaces.Box(
            low=-env_cfg.max_accel, high=env_cfg.max_accel,
            shape=(3,), dtype=np.float32)
        self.obs_dim = 9 + 6 * env_cfg.n_obstacles
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.bounds_low = np.array([-5.0, -5.0, 0.3], dtype=np.float32)
        self.bounds_high = np.array([5.0, 5.0, 4.0], dtype=np.float32)
        self.goal = np.array([4.0, 0.0, 1.5], dtype=np.float32)

    def _sample_obstacles(self):
        obs, attempts = [], 0
        while len(obs) < self.cfg.n_obstacles and attempts < 5000:
            p = self.rng.uniform(self.bounds_low + 0.8, self.bounds_high - 0.8)
            if np.linalg.norm(p - self.pos) < 1.5:
                attempts += 1
                continue
            if self.cfg.spacing_min > 0 and any(
                np.linalg.norm(p-q) < self.cfg.spacing_min for q in obs):
                attempts += 1
                continue
            obs.append(p)
            attempts += 1
        while len(obs) < self.cfg.n_obstacles:
            obs.append(self.rng.uniform(
                self.bounds_low + 0.8, self.bounds_high - 0.8))
        self.obstacles = np.asarray(obs, dtype=np.float32)
        self.obstacle_vel = self.rng.uniform(
            -self.cfg.obstacle_speed, self.cfg.obstacle_speed,
            size=(self.cfg.n_obstacles, 3)).astype(np.float32)
        self.obstacle_vel[:, 2] *= 0.25

    def _get_obs(self):
        parts = [self.pos, self.vel, self.goal - self.pos]
        for p, v in zip(self.obstacles, self.obstacle_vel):
            parts.extend([p - self.pos, v])
        return np.concatenate(parts).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = np.array([-4.0, 0.0, 1.5], dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self._sample_obstacles()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32),
                         -self.cfg.max_accel, self.cfg.max_accel)
        old_dist = float(np.linalg.norm(self.goal - self.pos))

        self.vel += action * self.cfg.dt
        speed = np.linalg.norm(self.vel)
        if speed > self.cfg.max_speed:
            self.vel *= self.cfg.max_speed / speed
        self.pos += self.vel * self.cfg.dt
        self.pos = np.clip(self.pos, self.bounds_low, self.bounds_high)

        self.obstacles += self.obstacle_vel * self.cfg.dt
        for i in range(self.cfg.n_obstacles):
            for j in range(3):
                if (self.obstacles[i, j] <= self.bounds_low[j] or
                        self.obstacles[i, j] >= self.bounds_high[j]):
                    self.obstacle_vel[i, j] *= -1
                    self.obstacles[i, j] = np.clip(
                        self.obstacles[i, j],
                        self.bounds_low[j], self.bounds_high[j])

        dists = np.linalg.norm(self.obstacles - self.pos, axis=1)
        min_dist = float(np.min(dists))
        new_dist = float(np.linalg.norm(self.goal - self.pos))
        collision = min_dist <= self.cfg.safe_distance * 0.65
        goal = new_dist <= self.cfg.goal_radius
        timeout = (self.steps + 1) >= self.cfg.max_steps

        reward = 4.0 * (old_dist - new_dist) - 0.01
        reward -= max(0.0, self.cfg.safe_distance - min_dist) * 1.5
        if collision:
            reward -= 50.0
        if goal:
            reward += 100.0

        self.steps += 1
        terminated = bool(collision or goal)
        truncated = bool(timeout and not terminated)
        info = {
            "goal": goal, "collision": collision, "timeout": truncated,
            "final_distance": new_dist,
            "min_obstacle_distance": min_dist
        }
        return self._get_obs(), float(reward), terminated, truncated, info
