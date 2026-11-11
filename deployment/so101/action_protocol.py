"""Shared validation for observation and execution consumers of SO101-PI05 actions."""

import numpy as np


def validate_action(message, config):
    if "rtc" in config and bool(config["rtc"].get("enabled")) != ("rtc" in message):
        raise ValueError("Action RTC mode does not match the selected deployment")
    for key in ("policy", "checkpoint_step", "model_sha256"):
        if key in config and message.get(key) != config[key]:
            raise ValueError(f"Wrong model identity: {key}")
    if (message.get("control_mode"), message.get("joint_units"), message.get("gripper_units")) != (
        "joint",
        "degrees",
        "0_to_100",
    ):
        raise ValueError("Wrong action mode or units")
    horizon = int(config.get("horizon", 30))
    fps = float(config.get("fps", 30))
    delta = float(message["delta/ms"]) / 1000
    if fps <= 0 or not np.isfinite(delta) or abs(delta - 1 / fps) > 1e-6:
        raise ValueError("Unexpected action interval")
    arms = [np.asarray(message[f"{side}_arm"]["actions"], dtype=float) for side in ("left", "right")]
    if any(arm.shape != (horizon, 6) or not np.isfinite(arm).all() for arm in arms):
        raise ValueError("Invalid action chunk")
    return np.concatenate(arms, axis=1), delta


def select_action(message, config, now, earliest):
    """Select by observation time, never replay an expired or warmup action."""
    actions, delta = validate_action(message, config)
    stamp = float(message["timestamp/ms"]) / 1000
    age = now - stamp
    if not np.isfinite(stamp) or not np.isfinite(now) or stamp < earliest or not 0 <= age <= 0.4:
        raise ValueError("Expired action or action predating this trial")
    index = int(age / delta)
    if index >= len(actions):
        raise ValueError("Action horizon exhausted")
    return actions[index].copy()
