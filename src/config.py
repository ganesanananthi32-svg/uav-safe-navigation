from dataclasses import dataclass

@dataclass
class EnvConfig:
    n_obstacles: int = 5
    dt: float = 0.05
    max_steps: int = 300
    max_accel: float = 3.0
    max_speed: float = 2.0
    goal_radius: float = 0.20
    safe_distance: float = 0.55
    gamma: float = 3.0
    obstacle_speed: float = 0.30
    spacing_min: float = 0.0
    seed: int = 42

@dataclass
class TrainConfig:
    total_timesteps: int = 150_000
    learning_starts: int = 1_000
    batch_size: int = 256
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    seed: int = 42
    train_filter: bool = False
    network_arch: tuple = (256, 256)
