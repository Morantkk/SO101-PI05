"""Establish measured-position targets before enabling motor torque."""

import numpy as np


def establish_hold(arms, joints, read, send, should_stop):
    if should_stop():
        raise RuntimeError("Stop requested before position hold")
    measured = np.asarray(read(), dtype=float)
    if measured.shape != (12,) or not np.isfinite(measured).all():
        raise ValueError("Invalid measured position for hold")
    send(measured)
    goals = np.asarray([values[j] for arm in arms
                        for values in [arm.bus.sync_read("Goal_Position", num_retry=1)] for j in joints])
    if goals.shape != (12,) or not np.isfinite(goals).all() or np.max(np.abs(goals - measured)) > 0.2:
        raise RuntimeError("Current-position hold target verification failed; torque was not enabled")
    current = np.asarray(read(), dtype=float)
    if current.shape != (12,) or not np.isfinite(current).all() or np.max(np.abs(current - measured)) > 0.5:
        raise RuntimeError("Pose moved while preparing hold; torque was not enabled")
    for arm in arms:
        if should_stop():
            raise RuntimeError("Stop requested before torque enable")
        arm.bus.enable_torque(num_retry=1)
    for arm in arms:
        values = arm.bus.sync_read("Torque_Enable", normalize=False, num_retry=1)
        if set(values) != set(joints) or any(v != 1 for v in values.values()):
            raise RuntimeError("Position hold torque verification failed")
    return measured
