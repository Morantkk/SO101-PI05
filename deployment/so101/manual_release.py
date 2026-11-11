"""Release SO101 holding torque for supported, manual repositioning."""

import json
import subprocess
import time

from state_client import ROOT, JOINTS, save_status
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig


def main():
    if not (ROOT / "STOP_MOTION").exists():
        raise RuntimeError("Stop latch must be set before releasing torque")
    for unit in ("joint-trial", "return-home", "state-client"):
        result = subprocess.run(["systemctl", "--user", "is-active", "--quiet",
                                 f"so101-deploy-{unit}"], check=False)
        if result.returncode == 0:
            raise RuntimeError(f"Stop so101-deploy-{unit} before releasing torque")
    config = json.loads((ROOT / "config.json").read_text())
    arms = []
    report = dict(status="releasing", holding=None, torque_readback={}, errors=[])
    try:
        # Establish both bus connections before changing either arm.
        for side in ("left", "right"):
            arm = SOFollower(SOFollowerRobotConfig(
                id=f"{config['robot_id']}_{side}", port=config[f"{side}_port"],
                use_degrees=True, disable_torque_on_disconnect=False,
            ))
            arms.append((side, arm))
            arm.bus.connect()
            arm.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
        for side, arm in arms:
            try:
                # Only change torque enable; do not unlock EEPROM or write positions.
                arm.bus.sync_write("Torque_Enable", {j: 0 for j in JOINTS},
                                   normalize=False, num_retry=2)
                actual = arm.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
                report["torque_readback"][side] = actual
                if set(actual) != set(JOINTS) or any(value != 0 for value in actual.values()):
                    raise RuntimeError("Torque-off readback failed")
            except Exception as error:
                report["errors"].append(f"{side}: {error}")
        report["status"] = "failed" if report["errors"] else "released"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append(str(error))
    finally:
        for side, arm in arms:
            if arm.bus.is_connected:
                try:
                    arm.bus.disconnect(disable_torque=False)
                except Exception as error:
                    report["errors"].append(f"{side} disconnect: {error}")
                    report["status"] = "failed"
        report["updated_at"] = time.time()
        if report["status"] == "released":
            report["holding"] = False
        save_status("manual_release_status.json", report)
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "released" else 1


if __name__ == "__main__":
    raise SystemExit(main())
