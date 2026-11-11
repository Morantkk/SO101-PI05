"""Exercise staged homing without hardware, using the bounded return controller."""

import sys
import runpy
from pathlib import Path

import numpy as np
import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deployment/so101"
sys.path.insert(0, str(DEPLOY))
from home_sequence import open_then_home

MOVE = runpy.run_path(str(DEPLOY / "return_to_start.py"))["return_to_start"]


def simulate(*, cancel_open=False, cancel_hold=False, stuck=False, drift_hold=False):
    measured = np.zeros(12)
    target = np.full(12, 10.0)
    target[[5, 11]] = 0
    low, high = np.full(12, -180.0), np.full(12, 180.0)
    low[[5, 11]], high[[5, 11]] = 0, 100
    ticks = [0.0]
    phases, samples = [], []
    hold_started = []

    def send(command):
        samples.append(command.copy())
        if not stuck:
            measured[:] = command

    def sleep(seconds):
        ticks[0] += seconds
        if drift_hold and hold_started:
            measured[5] += 2

    def move(*args, **kwargs):
        if hold_started:
            assert ticks[0] - hold_started[0] == pytest.approx(0.5, abs=0.001)
        return MOVE(*args, **kwargs, clock=lambda: ticks[0], sleep=sleep)

    def progress(phase, *_):
        phases.append(phase)
        if phase in ("opening_grippers", "holding_grippers"):
            np.testing.assert_array_equal(measured[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]], 0)
        if phase == "holding_grippers":
            np.testing.assert_allclose(measured[[5, 11]], 50, atol=1)
            if not hold_started:
                hold_started.append(ticks[0])

    result = open_then_home(
        target, low, high, move=move, read=lambda: measured.copy(), send=send,
        should_stop=lambda: (cancel_open and bool(phases)) or (cancel_hold and bool(hold_started)),
        on_progress=progress, clock=lambda: ticks[0], sleep=sleep,
    )
    return result, measured, phases, samples


def test_half_open_and_half_second_hold_precede_original_home_pose():
    result, measured, phases, _ = simulate()
    assert result["returned_to_start"] and result["grippers_opened"]
    assert phases[0] == "opening_grippers" and phases[-1] == "returning"
    assert "holding_grippers" in phases
    np.testing.assert_allclose(measured[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]], 10, atol=1)
    np.testing.assert_allclose(measured[[5, 11]], 0, atol=1)


def test_cancel_during_half_second_hold_never_starts_return():
    result, _, phases, _ = simulate(cancel_hold=True)
    assert not result["returned_to_start"]
    assert result["phase"] == "holding_grippers"
    assert "returning" not in phases


def test_drift_during_hold_aborts_return():
    with pytest.raises(RuntimeError, match="while holding grippers"):
        simulate(drift_hold=True)


def test_cancel_during_opening_never_starts_return():
    result, _, phases, _ = simulate(cancel_open=True)
    assert not result["returned_to_start"]
    assert set(phases) == {"opening_grippers"}


def test_stuck_gripper_aborts_on_timeout():
    with pytest.raises(TimeoutError):
        simulate(stuck=True)


def test_false_success_readback_prevents_arm_return():
    calls = []

    def fake_move(*args, **kwargs):
        calls.append(args[0])
        return {"returned_to_start": True}

    with pytest.raises(RuntimeError, match="verification failed"):
        open_then_home(
            np.zeros(12), np.zeros(12), np.full(12, 100), move=fake_move,
            read=lambda: np.zeros(12), send=lambda _: None, should_stop=lambda: False,
            on_progress=lambda *_: None,
        )
    assert len(calls) == 1
