"""Verify torque release without loading real serial drivers or touching hardware."""

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deployment/so101/manual_release.py"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def load_release(tmp_path, *, fail_left=False):
    import json
    (tmp_path / "config.json").write_text(json.dumps(dict(robot_id="test", left_port="left", right_port="right")))
    buses, reports = [], []

    class Bus:
        def __init__(self, side):
            self.side, self.is_connected = side, False
            self.torque = dict.fromkeys(JOINTS, 1)
            self.writes = []

        def connect(self):
            self.is_connected = True

        def sync_read(self, register, **kwargs):
            assert register == "Torque_Enable"
            assert kwargs["normalize"] is False
            return self.torque.copy()

        def sync_write(self, register, values, **kwargs):
            assert register == "Torque_Enable"
            assert kwargs["normalize"] is False
            assert values == dict.fromkeys(JOINTS, 0)
            self.writes.append(values)
            if not (fail_left and self.side == "left"):
                self.torque.update(values)

        def disconnect(self, disable_torque):
            assert disable_torque is False
            self.is_connected = False

    def robot(config):
        bus = Bus(config.port)
        buses.append(bus)
        return SimpleNamespace(bus=bus)

    modules = {
        "state_client": SimpleNamespace(ROOT=tmp_path, JOINTS=JOINTS,
                                        save_status=lambda name, report: reports.append(report)),
        "lerobot.robots.so_follower": SimpleNamespace(SOFollower=robot, SOFollowerRobotConfig=SimpleNamespace),
    }
    with patch.dict(sys.modules, modules):
        main = runpy.run_path(str(SCRIPT))["main"]
    return main, buses, reports


def test_release_both_arms_and_verify_readback(tmp_path):
    main, buses, reports = load_release(tmp_path)
    (tmp_path / "STOP_MOTION").touch()
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=3)):
        assert main() == 0
    assert len(buses) == 2
    assert all(len(bus.writes) == 1 and not bus.is_connected for bus in buses)
    assert reports[-1]["status"] == "released"
    assert reports[-1]["holding"] is False


def test_failed_readback_is_not_reported_as_released(tmp_path):
    main, buses, reports = load_release(tmp_path, fail_left=True)
    (tmp_path / "STOP_MOTION").touch()
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=3)):
        assert main() == 1
    assert all(len(bus.writes) == 1 and not bus.is_connected for bus in buses)
    assert reports[-1]["status"] == "failed"
    assert reports[-1]["holding"] is None


def test_active_controller_or_missing_latch_prevents_bus_access(tmp_path):
    main, buses, _ = load_release(tmp_path)
    with pytest.raises(RuntimeError, match="Stop latch"):
        main()
    (tmp_path / "STOP_MOTION").touch()
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
        with pytest.raises(RuntimeError, match="Stop so101"):
            main()
    assert buses == []
