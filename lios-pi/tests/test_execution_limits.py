"""Verify real-robot speed increases retain calibration and tracking guards."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deployment/so101"))
from execution_limits import (
    FULL_TASK_JOINT_SPEED_DEG_S,
    FULL_TASK_GRIPPER_SPEED_UNITS_S,
    full_task_target,
)


def test_full_task_speeds_match_original_limits():
    assert FULL_TASK_JOINT_SPEED_DEG_S == 45.63
    assert FULL_TASK_GRIPPER_SPEED_UNITS_S == 36.0


def test_full_task_target_remains_inside_calibration_limits():
    measured = np.zeros(12)
    previous = np.zeros(12)
    low = np.full(12, -20.0)
    high = np.full(12, 20.0)
    low[[5, 11]] = 0
    high[[5, 11]] = 100
    target = np.full(12, 1000.0)

    command = full_task_target(target, measured, previous, low, high, 0.1)

    assert np.all(command >= low)
    assert np.all(command <= high)
    assert command[0] == pytest.approx(FULL_TASK_JOINT_SPEED_DEG_S * 0.1)
    assert command[5] == pytest.approx(2.0)


def test_full_task_target_retains_tracking_error_guard():
    with pytest.raises(ValueError, match="tracking error"):
        full_task_target(
            np.zeros(12), np.zeros(12), np.full(12, 9.0), np.full(12, -20.0), np.full(12, 20.0), 0.05
        )
