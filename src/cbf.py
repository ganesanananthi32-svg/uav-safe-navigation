import numpy as np

def safe_action(position, velocity, nominal_action, obstacles,
                obstacle_velocities, max_accel=3.0,
                safe_distance=0.55, gamma=3.0, dt=0.05):
    # Acceleration-space CBF-style safety correction.
    u = np.clip(np.asarray(nominal_action, dtype=float),
                -max_accel, max_accel)
    active = False
    for p, vo in zip(obstacles, obstacle_velocities):
        r = np.asarray(p) - np.asarray(position)
        rv = np.asarray(vo) - np.asarray(velocity)
        d = np.linalg.norm(r)
        if d < 1e-9:
            continue
        closing = max(0.0, -float(np.dot(r, rv)) / d)
        if d < safe_distance or closing > 0:
            n = r / d
            required = gamma * max(0.0, safe_distance - d) + closing
            u = np.clip(u - required * n, -max_accel, max_accel)
            active = True
    return u.astype(np.float32), active
