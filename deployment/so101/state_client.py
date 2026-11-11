"""Read SO101 joints and consume LiOS JSON actions without commanding motors."""

import json
import time
import signal
import logging
import threading
from pathlib import Path

import numpy as np
from rtc_protocol import RTCConsumer
from action_protocol import validate_action
from websockets.sync.client import connect
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

ROOT = Path(__file__).resolve().parent
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def save_status(name, value):
    target = ROOT / name
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(target)


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    config = json.loads((ROOT / "config.json").read_text())
    if config["execution_enabled"]:
        raise ValueError("This client supports observation only; motor execution is not implemented")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    arms = []

    def consume_actions():
        count = 0
        while not stop.is_set():
            try:
                with connect(config["action_url"], proxy=None, open_timeout=3) as ws:
                    while not stop.is_set():
                        try:
                            message = json.loads(ws.recv(timeout=1))
                        except TimeoutError:
                            continue
                        if "left_arm" not in message:
                            continue
                        validate_action(message, config)
                        age = time.time() - float(message["timestamp/ms"]) / 1000
                        if not np.isfinite(age) or not 0 <= age <= 0.4:
                            raise ValueError("Expired received action")
                        count += 1
                        save_status(
                            "received_action.json",
                            {
                                "received_at": time.time(),
                                "received_chunks": count,
                                "executed": False,
                                "motor_commands_sent": 0,
                                "observation_age_s": age,
                                "message": message,
                            },
                        )
                        if count == 1 or count % 100 == 0:
                            logging.info("Received %s finite action chunks; motor commands sent: 0", count)
            except Exception as error:
                logging.warning("Action channel: %s", error)
                stop.wait(2)

    try:
        for side in ("left", "right"):
            arm = SOFollower(
                SOFollowerRobotConfig(
                    id=f"{config['robot_id']}_{side}",
                    port=config[f"{side}_port"],
                    use_degrees=True,
                    disable_torque_on_disconnect=False,
                )
            )
            arms.append(arm)
            if not arm.calibration:
                raise RuntimeError(f"Missing calibration: {side}")
            # Bypass Robot.connect(): it configures motors. The bus handshake only reads.
            arm.bus.connect()
            if not arm.bus.is_calibrated:
                raise RuntimeError(f"Calibration mismatch: {side}")
            if any(arm.bus.read("Operating_Mode", joint) != 0 for joint in JOINTS):
                raise RuntimeError(f"Expected position mode: {side}")
        if config.get("rtc", {}).get("enabled", False):
            observe_rtc(config, arms, stop)
            return
        threading.Thread(target=consume_actions, daemon=True).start()
        samples = 0
        while not stop.is_set():
            try:
                with connect(config["state_url"], proxy=None, open_timeout=3) as ws:
                    while not stop.is_set():
                        started = time.monotonic()
                        timestamp = time.time() * 1000
                        positions = []
                        for arm in arms:
                            values = arm.bus.sync_read("Present_Position", num_retry=2)
                            positions.append([float(values[j]) for j in JOINTS])
                        if not np.isfinite(positions).all():
                            raise ValueError("Non-finite robot state")
                        message = {
                            "timestamp/ms": timestamp,
                            "left_arm": {"positions": positions[0]},
                            "right_arm": {"positions": positions[1]},
                            "joint_units": "degrees",
                            "gripper_units": "0_to_100",
                            "execution_enabled": False,
                            "state_names": config.get("state_names"),
                        }
                        ws.send(json.dumps(message))
                        samples += 1
                        if samples % 50 == 0:
                            save_status(
                                "state_status.json",
                                {
                                    "updated_at": time.time(),
                                    "samples_sent": samples,
                                    "last_state": message,
                                    "motor_commands_sent": 0,
                                },
                            )
                        stop.wait(max(0, 1 / config["state_hz"] - (time.monotonic() - started)))
            except Exception as error:
                logging.warning("State channel: %s", error)
                stop.wait(2)
    finally:
        stop.set()
        for arm in arms:
            if arm.bus.is_connected:
                arm.bus.disconnect(disable_torque=False)


def observe_rtc(config, arms, stop):
    """Run the actual RTC queue at control frequency without any motor writes."""
    samples, chunks, resets = 0, 0, 0
    while not stop.is_set():
        consumer = RTCConsumer(config)
        earliest = time.time()
        try:
            with connect(config["state_url"], proxy=None, open_timeout=3) as states, connect(
                config["action_url"], proxy=None, open_timeout=3
            ) as actions:
                while not stop.is_set():
                    started = time.monotonic()
                    positions = [[float(v[j]) for j in JOINTS] for arm in arms
                                 for v in [arm.bus.sync_read("Present_Position", num_retry=2)]]
                    if not np.isfinite(positions).all():
                        raise ValueError("Non-finite state")
                    message = {
                        "timestamp/ms": time.time() * 1000, "rtc": consumer.state(),
                        "left_arm": {"positions": positions[0]}, "right_arm": {"positions": positions[1]},
                        "joint_units": "degrees", "gripper_units": "0_to_100",
                        "execution_enabled": False, "state_names": config["state_names"],
                    }
                    states.send(json.dumps(message))
                    samples += 1
                    for _ in range(10):
                        try:
                            candidate = json.loads(actions.recv(timeout=0.001))
                        except TimeoutError:
                            break
                        if "left_arm" in candidate and consumer.accept(candidate, time.time(), earliest):
                            chunks += 1
                            save_status("received_action.json", dict(
                                received_at=time.time(), received_chunks=chunks, executed=False,
                                motor_commands_sent=0, message=candidate, rtc=consumer.state(),
                            ))
                    consumer.pop()
                    if samples % 30 == 0:
                        save_status("state_status.json", dict(updated_at=time.time(), samples_sent=samples,
                            last_state=message, motor_commands_sent=0, rtc_resets=resets, rtc=consumer.state()))
                    stop.wait(max(0, 1 / config["fps"] - (time.monotonic() - started)))
        except Exception as error:
            resets += 1
            logging.warning("RTC observation session reset (no motor writes): %s", error)
            stop.wait(1)


if __name__ == "__main__":
    main()
