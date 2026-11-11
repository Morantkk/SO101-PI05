"""Joint-space speed and range guards for real-robot policy execution."""

import numpy as np

GRIPPERS = [5, 11]
FULL_TASK_JOINT_SPEED_DEG_S = 45.63
FULL_TASK_GRIPPER_SPEED_UNITS_S = 36.0


def full_task_target(target, measured, previous, low, high, dt):
    """Limit full-task commands by speed, calibration and tracking bounds."""
    arrays = [np.asarray(x, dtype=float) for x in (target, measured, previous, low, high)]
    if any(x.shape != (12,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Invalid position array")
    target, measured, previous, low, high = arrays
    if not 0 < dt <= 0.1 or np.any(low > high):
        raise ValueError("Control loop stalled or invalid limits")
    if np.any(measured < low) or np.any(measured > high):
        raise ValueError("Measured position outside calibration")
    speed = np.full(12, FULL_TASK_JOINT_SPEED_DEG_S)
    speed[GRIPPERS] = FULL_TASK_GRIPPER_SPEED_UNITS_S
    tracking = np.full(12, 8.0)
    tracking[GRIPPERS] = 4.0
    follow = np.full(12, 6.0)
    follow[GRIPPERS] = 2.0
    if np.any(np.abs(previous - measured) > tracking):
        raise ValueError("Joint or gripper tracking error exceeded")
    lower = np.maximum.reduce([low, previous - speed * dt, measured - follow])
    upper = np.minimum.reduce([high, previous + speed * dt, measured + follow])
    if np.any(lower > upper):
        raise ValueError("No target satisfies movement limits")
    return np.clip(target, lower, upper)
