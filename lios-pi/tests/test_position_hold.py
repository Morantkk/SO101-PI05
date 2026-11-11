"""Exercise measured-position hold with fake motor buses only."""

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deployment/so101"))
from position_hold import establish_hold

JOINTS = tuple("abcdef")


def setup_hold(*, bad_goal=False, bad_torque=False):
    events = []
    measured = np.arange(12, dtype=float)
    arms = []
    for i in range(2):
        def read(register, i=i, **_):
            events.append(("read", i, register))
            values = measured[i*6:(i+1)*6] + (1 if bad_goal else 0) if register == "Goal_Position" else [0 if bad_torque else 1]*6
            return dict(zip(JOINTS, values, strict=True))
        arms.append(SimpleNamespace(bus=SimpleNamespace(sync_read=read,
                    enable_torque=lambda i=i, **_: events.append(("enable", i)))))
    return arms, measured, events


def test_hold_verifies_both_goals_before_enabling_either_arm():
    arms, measured, events = setup_hold()
    establish_hold(arms, JOINTS, lambda: measured, lambda value: events.append(("send", value.copy())), lambda: False)
    assert events[0][0] == "send"
    assert events[1:5] == [("read", 0, "Goal_Position"), ("read", 1, "Goal_Position"), ("enable", 0), ("enable", 1)]


def test_hold_rejects_goal_mismatch_without_enabling():
    arms, measured, events = setup_hold(bad_goal=True)
    with pytest.raises(RuntimeError, match="target verification"):
        establish_hold(arms, JOINTS, lambda: measured, lambda _: None, lambda: False)
    assert not any(event[0] == "enable" for event in events)


def test_hold_cancels_and_reports_failed_torque_readback():
    arms, measured, events = setup_hold(bad_torque=True)
    with pytest.raises(RuntimeError, match="Stop requested"):
        establish_hold(arms, JOINTS, lambda: measured, lambda _: pytest.fail("Must not write"), lambda: True)
    with pytest.raises(RuntimeError, match="torque verification"):
        establish_hold(arms, JOINTS, lambda: measured, lambda _: None, lambda: False)


def test_hold_does_not_enable_if_pose_moved():
    arms, measured, events = setup_hold()
    reads = iter([measured, measured + 2])
    with pytest.raises(RuntimeError, match="Pose moved"):
        establish_hold(arms, JOINTS, lambda: next(reads), lambda _: None, lambda: False)
    assert not any(event[0] == "enable" for event in events)
