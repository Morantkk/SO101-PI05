"""Return to the recorded pre-task pose using the existing bounded return controller."""

import sys
import json
import time
import runpy
import signal
import argparse
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE
sys.path.insert(0, str(ROOT))
from state_client import JOINTS, save_status
from home_sequence import open_then_home
from position_hold import establish_hold
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "home_pose.json")
    args = parser.parse_args()
    return_to_start = runpy.run_path(str(ROOT / "return_to_start.py"))["return_to_start"]
    source = args.source
    previous = json.loads(source.read_text())
    config = json.loads((ROOT / "config.json").read_text())
    target = np.asarray(previous["start"], dtype=float)
    stop_file = ROOT / "STOP_MOTION"
    stopped = False
    touched = False
    arms = []
    report = dict(status="preparing_return", source=str(source), target=target.tolist(), motor_commands_sent=0, executed=False)

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def read():
        return np.asarray([values[joint] for arm in arms for values in [arm.bus.sync_read("Present_Position", num_retry=1)] for joint in JOINTS], dtype=float)

    def send(values):
        nonlocal touched
        for index, arm in enumerate(arms):
            if stopped or stop_file.exists():
                raise RuntimeError("Return stopped")
            touched = True
            arm.bus.sync_write("Goal_Position", dict(zip(JOINTS, map(float, values[6*index:6*index+6]), strict=True)))
            report["motor_commands_sent"] += 1

    def progress(phase, elapsed, measured, command, error, cycles):
        report.update(status=phase, executed=True, elapsed_s=elapsed, measured=measured.tolist(), command=command.tolist(), max_error=float(error.max()), cycles=cycles, updated_at=time.time())
        save_status("execution_status.json", report)

    try:
        if stop_file.exists():
            raise RuntimeError("Stop latch set")
        low, high = [], []
        for side in ("left", "right"):
            arm = SOFollower(SOFollowerRobotConfig(id=f"{config['robot_id']}_{side}", port=config[f"{side}_port"], use_degrees=True, disable_torque_on_disconnect=False))
            arms.append(arm)
            arm.bus.connect()
            if not arm.calibration or not arm.bus.is_calibrated:
                raise RuntimeError(f"Calibration mismatch: {side}")
            for joint in JOINTS:
                if arm.bus.read("Operating_Mode", joint) != 0:
                    raise RuntimeError("Expected position mode")
                calibration = arm.calibration[joint]
                resolution = arm.bus.model_resolution_table[arm.bus.motors[joint].model] - 1
                span = (calibration.range_max - calibration.range_min) * 180 / resolution
                low.append(0 if joint == "gripper" else -span)
                high.append(100 if joint == "gripper" else span)
            protections = {name: arm.bus.read(name, "gripper")
                           for name in ("Max_Torque_Limit", "Protection_Current", "Overload_Torque")}
            if not (0 < protections["Max_Torque_Limit"] <= 500
                    and 0 < protections["Protection_Current"] <= 250
                    and 0 < protections["Overload_Torque"] <= 25):
                raise RuntimeError(f"Gripper protection settings exceed driver defaults: {side}")
        current = read()
        if target.shape != (12,) or not np.isfinite(target).all() or np.any(target < low) or np.any(target > high):
            raise ValueError("Invalid recorded return pose")
        if not np.isfinite(current).all() or np.any(current < low) or np.any(current > high):
            raise ValueError("Invalid measured pose")
        report["start"] = current.tolist()
        report["torque_before"] = [arm.bus.sync_read("Torque_Enable", normalize=False) for arm in arms]
        establish_hold(arms, JOINTS, read, send, lambda: stopped or stop_file.exists())
        report.update(open_then_home(target, low, high, move=return_to_start, read=read, send=send,
                                     should_stop=lambda: stopped or stop_file.exists(), on_progress=progress))
    except Exception as error:
        report.update(status="failed", error=str(error))
    finally:
        hold_errors = []
        if touched:
            for arm in arms:
                if arm.bus.is_connected:
                    try:
                        held = arm.bus.sync_read("Present_Position", num_retry=1)
                        arm.bus.sync_write("Goal_Position", held, num_retry=1)
                        actual = arm.bus.sync_read("Goal_Position", num_retry=1)
                        if max(abs(actual[j]-held[j]) for j in JOINTS) > 0.2:
                            raise RuntimeError("Hold verification failed")
                    except Exception as error:
                        hold_errors.append(str(error))
        if len(arms) == 2 and all(arm.bus.is_connected for arm in arms):
            try:
                report["end"] = read().tolist()
            except Exception as error:
                hold_errors.append(str(error))
        torque_confirmed = False
        if touched and len(arms) == 2 and all(arm.bus.is_connected for arm in arms):
            try:
                torque_readbacks = [arm.bus.sync_read("Torque_Enable", normalize=False) for arm in arms]
                torque_confirmed = all(set(values) == set(JOINTS) and all(v == 1 for v in values.values())
                                       for values in torque_readbacks)
                if not torque_confirmed:
                    hold_errors.append("Position hold torque was not confirmed on all joints")
            except Exception as error:
                hold_errors.append(str(error))
        for arm in arms:
            if arm.bus.is_connected:
                try:
                    arm.bus.disconnect(disable_torque=False)
                except Exception as error:
                    hold_errors.append(str(error))
        report.update(holding=touched and torque_confirmed and not hold_errors, hold_errors=hold_errors, finished_at=time.time())
        save_status("execution_status.json", report)
        (PACKAGE / "return_home_result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report.get("returned_to_start") and report.get("holding") and not hold_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
