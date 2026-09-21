import numpy as np

class EventTrigger:
    def __init__(self, distance_threshold=0.8, ttc_threshold=1.2):
        self.distance_threshold = distance_threshold
        self.ttc_threshold = ttc_threshold

    def should_trigger(self, position, velocity, obstacles, obstacle_velocities):
        if len(obstacles) == 0:
            return False
        rel = obstacles - position
        dist = np.linalg.norm(rel, axis=1)
        if np.min(dist) < self.distance_threshold:
            return True
        rel_v = obstacle_velocities - velocity
        closing = -np.sum(rel * rel_v, axis=1) / np.maximum(dist, 1e-6)
        closing = np.maximum(closing, 0.0)
        ttc = dist / np.maximum(closing, 1e-6)
        return bool(np.min(ttc) < self.ttc_threshold)
