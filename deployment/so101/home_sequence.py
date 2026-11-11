"""Half-open both grippers, pause, then return to the recorded home pose."""

import time

import numpy as np

GRIPPERS = [5, 11]
OPEN_POSITION = 50.0
OPEN_HOLD_SECONDS = 0.5


def open_then_home(
    target, low, high, *, move, read, send, should_stop, on_progress,
    clock=time.monotonic, sleep=time.sleep,
):
    target, low, high = [np.asarray(value, dtype=float).copy() for value in (target, low, high)]
    current = np.asarray(read(), dtype=float).copy()
    if any(value.shape != (12,) or not np.isfinite(value).all() for value in (target, low, high, current)):
        raise ValueError("Invalid home sequence positions")
    if np.any(low > high) or np.any(target < low) or np.any(target > high):
        raise ValueError("Home pose outside calibration")
    if np.any(current < low) or np.any(current > high):
        raise ValueError("Measured pose outside calibration")
    opening = current.copy()
    opening[GRIPPERS] = OPEN_POSITION
    if np.any(opening < low) or np.any(opening > high):
        raise ValueError("Open gripper target outside calibration")
    home = target.copy()
    for phase, destination in (("opening_grippers", opening), ("returning", home)):
        if should_stop():
            return dict(status="stopped", returned_to_start=False, phase=phase)
        result = move(
            destination, low, high, read=read, send=send, should_stop=should_stop,
            on_progress=lambda *args, phase=phase: on_progress(phase, *args), timeout=60,
        )
        if not result.get("returned_to_start"):
            return dict(result, returned_to_start=False, phase=phase)
        measured = np.asarray(read(), dtype=float)
        if measured.shape != (12,) or not np.isfinite(measured).all():
            raise RuntimeError(f"Invalid measured pose after {phase}")
        if np.any(np.abs(measured - destination) > 1.0):
            raise RuntimeError(f"Position verification failed after {phase}")
        if phase == "opening_grippers":
            started = clock()
            while True:
                if should_stop():
                    return dict(status="stopped", returned_to_start=False, phase="holding_grippers")
                measured = np.asarray(read(), dtype=float)
                if measured.shape != (12,) or not np.isfinite(measured).all():
                    raise RuntimeError("Invalid measured pose while holding grippers")
                error = np.abs(measured - opening)
                if np.any(measured < low) or np.any(measured > high) or np.any(error > 1.0):
                    raise RuntimeError("Position verification failed while holding grippers")
                elapsed = clock() - started
                on_progress("holding_grippers", elapsed, measured, opening, error, 0)
                if elapsed >= OPEN_HOLD_SECONDS:
                    break
                sleep(min(0.02, OPEN_HOLD_SECONDS - elapsed))
    return dict(result, phase="completed", grippers_opened=True, target=home.tolist())
