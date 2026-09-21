import gymnasium as gym
from .cbf import safe_action
from .trigger import EventTrigger

class DeploymentShield(gym.Wrapper):
    def __init__(self, env, safe_distance=0.55, gamma=3.0,
                 max_accel=3.0, ttc_threshold=1.2):
        super().__init__(env)
        self.safe_distance = safe_distance
        self.gamma = gamma
        self.max_accel = max_accel
        self.trigger = EventTrigger(ttc_threshold=ttc_threshold)

    def step(self, action):
        e = self.env.unwrapped
        if self.trigger.should_trigger(
            e.pos, e.vel, e.obstacles, e.obstacle_vel):
            action, _ = safe_action(
                e.pos, e.vel, action, e.obstacles, e.obstacle_vel,
                self.max_accel, self.safe_distance, self.gamma, e.cfg.dt)
        return self.env.step(action)

class TrainingShield(gym.Wrapper):
    # Diagnostic wrapper: filter the action before the environment executes it.
    def __init__(self, env, safe_distance=0.55, gamma=3.0, max_accel=3.0):
        super().__init__(env)
        self.safe_distance = safe_distance
        self.gamma = gamma
        self.max_accel = max_accel

    def step(self, action):
        e = self.env.unwrapped
        filtered, active = safe_action(
            e.pos, e.vel, action, e.obstacles, e.obstacle_vel,
            self.max_accel, self.safe_distance, self.gamma, e.cfg.dt)
        obs, reward, terminated, truncated, info = self.env.step(filtered)
        info["shield_active"] = active
        return obs, reward, terminated, truncated, info
