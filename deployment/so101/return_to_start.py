"""Bounded joint-space return to the pose measured at the beginning of a run."""

import time

import numpy as np

GRIPPERS = [5, 11]


def return_to_start(
    initial,
    low,
    high,
    *,
    read,
    send,
    should_stop,
    on_progress,
    period=1 / 30,
    timeout=60,
    clock=time.monotonic,
    sleep=time.sleep,
):
    target, low, high = [np.asarray(x, dtype=float).copy() for x in (initial, low, high)]
    if any(x.shape != (12,) or not np.isfinite(x).all() for x in (target, low, high)):
        raise ValueError("Invalid return pose or calibration limits")
    if np.any(low > high) or np.any(target < low) or np.any(target > high):
        raise ValueError("Start pose is outside calibration limits")
    if not np.isfinite(period) or not 0 < period <= 0.1 or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid return timing")
    speed = np.full(12, 8.0)
    speed[GRIPPERS] = 5.0
    tracking = np.full(12, 8.0)
    tracking[GRIPPERS] = 4.0
    follow = np.full(12, 6.0)
    follow[GRIPPERS] = 2.0
    started = clock()
    last_tick = None
    previous = None
    settled_since = None
    cycles = 0
    while True:
        if should_stop():
            return dict(
                status="stopped", returned_to_start=False, return_cycles=cycles, return_elapsed_s=clock() - started
            )
        tick = clock()
        if tick - started >= timeout:
            raise TimeoutError("Return to start timed out")
        dt = period if last_tick is None else tick - last_tick
        if not 0 < dt <= 0.1:
            raise RuntimeError("Return control loop stalled")
        measured = np.asarray(read(), dtype=float).copy()
        if measured.shape != (12,) or not np.isfinite(measured).all():
            raise ValueError("Invalid measured position during return")
        if np.any(measured < low) or np.any(measured > high):
            raise ValueError("Measured position outside calibration during return")
        if previous is None:
            previous = measured.copy()
        if np.any(np.abs(previous - measured) > tracking):
            raise RuntimeError("Return tracking error exceeded")
        error = np.abs(target - measured)
        if np.all(error <= 1.0):
            if settled_since is None:
                settled_since = tick
        else:
            settled_since = None
        if should_stop():
            return dict(
                status="stopped", returned_to_start=False, return_cycles=cycles, return_elapsed_s=clock() - started
            )
        if settled_since is not None and tick - settled_since >= 0.2:
            return dict(
                status="completed",
                returned_to_start=True,
                return_cycles=cycles,
                return_elapsed_s=tick - started,
                return_final_error=error.tolist(),
            )
        lower = np.maximum.reduce([low, previous - speed * dt, measured - follow])
        upper = np.minimum.reduce([high, previous + speed * dt, measured + follow])
        if np.any(lower > upper):
            raise RuntimeError("No return target satisfies movement limits")
        command = np.clip(target, lower, upper)
        if should_stop():
            return dict(
                status="stopped", returned_to_start=False, return_cycles=cycles, return_elapsed_s=clock() - started
            )
        send(command)
        previous = command
        cycles += 1
        on_progress(tick - started, measured, command, error, cycles)
        last_tick = tick
        sleep(max(0, period - (clock() - tick)))
