"""Execute SO101 joint trials or a full task using SO101-PI05 state/action WebSockets."""

import json
import time
import signal
import logging
import argparse
from collections import deque

import numpy as np
from rtc_protocol import RTCConsumer
from state_client import ROOT, JOINTS, save_status
from position_hold import establish_hold
from action_protocol import select_action
from execution_limits import FULL_TASK_JOINT_SPEED_DEG_S, FULL_TASK_GRIPPER_SPEED_UNITS_S, full_task_target
from execution_options import resolve_duration
from websockets.sync.client import connect
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

STOP_FILE = ROOT / "STOP_MOTION"
GRIPPERS = [5, 11]


def bounded_target(target, measured, initial, previous, low, high, dt):
    """Limit speed, measured tracking error, calibration range and displacement."""
    arrays = [np.asarray(x, dtype=float) for x in (target, measured, initial, previous, low, high)]
    if any(x.shape != (12,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Invalid position array")
    target, measured, initial, previous, low, high = arrays
    if not 0 < dt <= 0.1 or np.any(low > high):
        raise ValueError("Control loop stalled or invalid limits")
    if np.any(measured < low) or np.any(measured > high):
        raise ValueError("Measured position outside calibration")
    if np.max(np.abs(measured - initial)) > 5.5 or np.max(np.abs(previous - measured)) > 2:
        raise ValueError("Displacement or tracking error exceeded")
    lower = np.maximum.reduce([low, initial - 5, previous - 3 * dt, measured - 1])
    upper = np.minimum.reduce([high, initial + 5, previous + 3 * dt, measured + 1])
    if np.any(lower > upper):
        raise ValueError("No target satisfies movement limits")
    command = np.clip(target, lower, upper)
    command[GRIPPERS] = initial[GRIPPERS]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--full-task", action="store_true", help="Allow calibrated joint travel and gripper motion")
    parser.add_argument("--duration", type=float, default=None,
                        help="Full-task seconds; 0 runs continuously; defaults to deployment config")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    try:
        duration = resolve_duration(args.full_task, args.duration, config.get("execution_duration_s", 60))
    except ValueError as error:
        parser.error(str(error))
    result_name = "full_task_result.json" if args.full_task else "joint_trial_result.json"
    period = 1 / config["fps"]
    rtc = RTCConsumer(config) if config.get("rtc", {}).get("enabled", False) else None
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    arms = []
    touched = False
    started = None
    report = dict(status="preparing", updated_at=time.time(), executed=False, motor_commands_sent=0,
                  checkpoint_step=config["checkpoint_step"], model_sha256=config["model_sha256"],
                  rtc=config.get("rtc", {"enabled": False}), duration_limit_s=duration,
                  mode="full_task" if args.full_task else "joint_trial",
                  speed_limit_deg_s=FULL_TASK_JOINT_SPEED_DEG_S if args.full_task else 3,
                  gripper_speed_limit_units_s=FULL_TASK_GRIPPER_SPEED_UNITS_S if args.full_task else 0,
                  displacement_limit_deg=None if args.full_task else 5,
                  grippers="model_controlled" if args.full_task else "held",
                  sample_capacity=9000, samples=deque(maxlen=9000))

    def publish():
        report["updated_at"] = time.time()
        save_status("execution_status.json", {k:v for k,v in report.items() if k != "samples"})

    def limit(target, measured, initial, previous, low, high, dt):
        if args.full_task:
            return full_task_target(target, measured, previous, low, high, dt)
        return bounded_target(target, measured, initial, previous, low, high, dt)

    def read():
        values = np.array([value[j] for arm in arms for value in [arm.bus.sync_read("Present_Position", num_retry=1)] for j in JOINTS])
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError("Invalid measured state")
        return values

    def send(values, holding=False):
        nonlocal touched
        if not holding and (stopped or STOP_FILE.exists()):
            raise RuntimeError("Stop requested")
        for index, arm in enumerate(arms):
            if not holding and (stopped or STOP_FILE.exists()):
                raise RuntimeError("Stop requested before arm write")
            touched = True
            arm.bus.sync_write("Goal_Position", dict(zip(JOINTS, map(float, values[6*index:6*index+6]), strict=True)))
            report["motor_commands_sent"] += 1

    try:
        if args.execute and STOP_FILE.exists():
            raise RuntimeError("Stop latch set")
        publish()
        low, high = [], []
        for side in ("left", "right"):
            arm = SOFollower(SOFollowerRobotConfig(id=f"{config['robot_id']}_{side}", port=config[f"{side}_port"], use_degrees=True, disable_torque_on_disconnect=False))
            arms.append(arm)
            arm.bus.connect()
            if not arm.calibration or not arm.bus.is_calibrated:
                raise RuntimeError(f"Calibration mismatch: {side}")
            for joint in JOINTS:
                if arm.bus.read("Operating_Mode", joint) != 0:
                    raise RuntimeError(f"Not in position mode: {side}/{joint}")
                c = arm.calibration[joint]
                resolution = arm.bus.model_resolution_table[arm.bus.motors[joint].model] - 1
                span = (c.range_max - c.range_min) * 180 / resolution
                low.append(0 if joint == "gripper" else -span)
                high.append(100 if joint == "gripper" else span)
        initial = previous = read()
        low, high = np.array(low), np.array(high)
        limit(initial, initial, initial, previous, low, high, period)
        report["start"] = initial.tolist()
        report["torque_before"] = [arm.bus.sync_read("Torque_Enable", normalize=False) for arm in arms]
        if args.full_task:
            report["gripper_protection"] = [
                {name: arm.bus.read(name, "gripper") for name in ("Max_Torque_Limit", "Protection_Current", "Overload_Torque")}
                for arm in arms
            ]
            if any(not 0 < p["Max_Torque_Limit"] <= 500 or not 0 < p["Protection_Current"] <= 250
                   or not 0 < p["Overload_Torque"] <= 25 for p in report["gripper_protection"]):
                raise RuntimeError("Gripper protection settings exceed SO101 driver defaults")
        earliest = time.time()
        latest = None
        count = 0
        samples = 0
        last_tick = time.monotonic()
        with connect(config["state_url"], proxy=None, open_timeout=3, close_timeout=1) as states, connect(config["action_url"], proxy=None, open_timeout=3, close_timeout=1) as actions:
            while not stopped and not (args.execute and STOP_FILE.exists()):
                tick = time.monotonic()
                if started is not None and duration is not None and tick - started >= duration:
                    report["status"] = "time_limit" if args.full_task else "completed"
                    break
                measured = read()
                message = {"timestamp/ms": time.time()*1000, "state_names": config.get("state_names"), "left_arm": {"positions": measured[:6].tolist()},
                           "right_arm": {"positions": measured[6:].tolist()}, "joint_units": "degrees",
                           "gripper_units": "0_to_100", "execution_enabled": started is not None and args.execute}
                if rtc is not None:
                    message["rtc"] = rtc.state()
                states.send(json.dumps(message))
                samples += 1
                save_status("state_status.json", dict(updated_at=time.time(), samples_sent=samples, last_state=message, motor_commands_sent=report["motor_commands_sent"]))
                for _ in range(10):
                    try:
                        candidate = json.loads(actions.recv(timeout=0.001))
                    except TimeoutError:
                        break
                    if "left_arm" not in candidate:
                        continue
                    if float(candidate["timestamp/ms"])/1000 < earliest:
                        continue
                    if rtc is not None:
                        if not rtc.accept(candidate, time.time(), earliest):
                            continue
                    else:
                        select_action(candidate, config, time.time(), earliest)
                    latest = candidate
                    count += 1
                    save_status("received_action.json", dict(received_at=time.time(), received_chunks=count, executed=args.execute and started is not None, motor_commands_sent=report["motor_commands_sent"], message=latest))
                if latest is None:
                    if time.time()-earliest > 5:
                        raise RuntimeError("No fresh model prediction within five seconds")
                else:
                    target = rtc.peek() if rtc is not None else select_action(latest, config, time.time(), earliest)
                    if started is None:
                        if np.max(np.abs(measured-initial)) > 0.5:
                            raise RuntimeError("Pose moved during preparation")
                        if not args.execute:
                            command = limit(target, measured, initial, previous, low, high, period)
                            report.update(status="dry_run_completed", first_bounded_target=command.tolist())
                            break
                        if args.execute:
                            initial = establish_hold(arms, JOINTS, read, send,
                                                     lambda: stopped or STOP_FILE.exists())
                            previous = initial.copy()
                        started = time.monotonic()
                        last_tick = started
                        report.update(status="running" if args.execute else "dry_run", executed=args.execute)
                        publish()
                    else:
                        command = limit(target, measured, initial, previous, low, high, tick-last_tick)
                        if args.execute:
                            send(command)
                        if rtc is not None:
                            rtc.pop()
                        previous = command
                        report["samples"].append(dict(elapsed_s=tick-started, measured=measured.tolist(), command=command.tolist(), raw_action=target.tolist()))
                        report["command_cycles"] = report.get("command_cycles", 0) + 1
                        report["samples_dropped"] = report["command_cycles"] - len(report["samples"])
                        report["elapsed_s"] = tick-started
                        publish()
                last_tick = tick if started is None else max(tick,last_tick)
                time.sleep(max(0, period-(time.monotonic()-tick)))
            else:
                report["status"] = "stopped"
    except Exception as error:
        STOP_FILE.touch()
        report.update(status="failed", error=str(error))
        logging.exception("Joint trial stopped")
    finally:
        if touched:
            hold_errors = []
            for arm in arms:
                if arm.bus.is_connected:
                    try:
                        held = arm.bus.sync_read("Present_Position", num_retry=1)
                        arm.bus.sync_write("Goal_Position", held, num_retry=1)
                        report["motor_commands_sent"] += 1
                        confirmed = arm.bus.sync_read("Goal_Position", num_retry=1)
                        if max(abs(confirmed[j]-held[j]) for j in JOINTS) > 0.2:
                            raise RuntimeError("Final hold readback mismatch")
                    except Exception as error:
                        hold_errors.append(str(error))
            report["hold_errors"] = hold_errors
            try:
                torque = [arm.bus.sync_read("Torque_Enable", normalize=False, num_retry=1) for arm in arms]
                confirmed_torque = len(torque) == 2 and all(set(values) == set(JOINTS) and all(v == 1 for v in values.values()) for values in torque)
                if not confirmed_torque:
                    hold_errors.append("Holding torque could not be confirmed on all joints")
            except Exception as error:
                hold_errors.append(str(error))
            report["holding"] = not hold_errors
        try:
            if len(arms)==2 and all(a.bus.is_connected for a in arms):
                report["end"] = read().tolist()
        except Exception as error:
            report["final_read_error"] = str(error)
        for arm in arms:
            if arm.bus.is_connected:
                try:
                    arm.bus.disconnect(disable_torque=False)
                except Exception as error:
                    report.setdefault("hold_errors", []).append(str(error))
                    report["holding"] = False
        report["finished_at"] = time.time()
        publish()
        report["samples"] = list(report["samples"])
        save_status(result_name, report)
        print(json.dumps({k:v for k,v in report.items() if k!="samples"},indent=2),flush=True)
    return 1 if report["status"] == "failed" or report.get("hold_errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
