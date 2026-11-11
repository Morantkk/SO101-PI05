"""Stop inference processes and archive transient state for a fresh next run."""

import json
import time
import uuid
import socket
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
UNITS = tuple(f"lios-deploy-{name}" for name in
              ("policy-task", "joint-trial", "return-home", "manual-release", "state-client", "inference",
               "receiver", "sender"))
TRANSIENT_FILES = (
    "inference_status.json", "latest_prediction.json", "received_action.json",
    "state_status.json", "execution_status.json",
)


def unit_state(unit):
    result = subprocess.run(
        ["systemctl", "--user", "show", unit,
         "--property=LoadState,ActiveState,MainPID", "--no-pager"],
        capture_output=True, text=True, check=False, timeout=10,
    )
    state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if state.get("LoadState") == "not-found":
        return state
    if result.returncode or not {"LoadState", "ActiveState", "MainPID"} <= state.keys():
        raise RuntimeError(f"Cannot inspect {unit}: {result.stderr.strip()}")
    return state


def stop_unit(unit):
    inference_only = unit.removesuffix(".service") == "lios-deploy-inference"
    if unit_state(unit).get("LoadState") != "not-found":
        print(f"Stopping {unit}...", flush=True)
        try:
            if inference_only:
                subprocess.run(["systemctl", "--user", "stop", "--no-block", unit], check=True, timeout=10)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if is_stopped(unit_state(unit)):
                        break
                    time.sleep(0.2)
                else:
                    # This service only predicts; motor controllers must stop normally.
                    print("Inference did not exit within 10s; ending its process group (no motor control).", flush=True)
                    subprocess.run(
                        ["systemctl", "--user", "kill", "--signal=SIGKILL", "--kill-whom=all", unit],
                        check=True, timeout=10,
                    )
                    deadline = time.monotonic() + 5
                    while not is_stopped(unit_state(unit)) and time.monotonic() < deadline:
                        time.sleep(0.2)
            else:
                # Allow systemd's 90s default stop timeout to finish before our caller times out.
                subprocess.run(["systemctl", "--user", "stop", unit], check=True, timeout=100)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Timed out stopping {unit}; STOP_MOTION remains set, retry clear-cache") from error
    state = unit_state(unit)
    if not is_stopped(state):
        raise RuntimeError(f"{unit} has not fully stopped: {state}")


def is_stopped(state):
    return state.get("LoadState") == "not-found" or (
        state.get("ActiveState") in ("inactive", "failed") and state.get("MainPID") == "0"
    )


def ensure_endpoint_stopped():
    try:
        connection = socket.create_connection(("127.0.0.1", 18081), timeout=1)
    except ConnectionRefusedError:
        return
    except OSError as error:
        raise RuntimeError(f"Cannot verify inference endpoint shutdown: {error}") from error
    else:
        connection.close()
        raise RuntimeError("Port 18081 is still listening; stop any foreground inference process first")


def initialize(root=ROOT, stop_service=stop_unit, verify_stopped=ensure_endpoint_stopped):
    lock = root / ".reset_in_progress"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError("Another reset is in progress; initialization cancelled") from error
    report = dict(status="initializing", stopped_units=[], archived_files=[],
                  motors_started=False, torque_released=False)
    try:
        (root / "STOP_MOTION").touch()
        generation = uuid.uuid4().hex
        (root / "INFERENCE_RESET_GENERATION").write_text(generation + "\n")
        report["generation"] = generation
        for unit in UNITS:
            print(f"Checking {unit}...", flush=True)
            stop_service(unit)
            report["stopped_units"].append(unit)
        verify_stopped()
        archive = root / "runtime_archive" / (datetime.now().strftime("%Y%m%d_%H%M%S_") + generation[:8])
        archive.mkdir(parents=True)
        report["archive"] = str(archive)
        for name in TRANSIENT_FILES:
            source = root / name
            if source.exists():
                source.rename(archive / name)
                report["archived_files"].append(name)
        report["status"] = "initialized"
        report["next_start"] = "bash infer_robot.sh"
    except BaseException as error:
        report.update(status="failed", error=str(error) or type(error).__name__)
        raise
    finally:
        report["updated_at"] = time.time()
        try:
            target = root / "inference_reset_status.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(target)
        finally:
            lock.rmdir()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without stopping or writing anything")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(stop_units=UNITS, archive_files=TRANSIENT_FILES,
                              preserve=["weights", "config.json", "home pose", "calibration", "logs"],
                              motors_started=False, torque_released=False), indent=2))
        return 0
    try:
        print(json.dumps(initialize(), indent=2), flush=True)
        print("Inference state cleared. Next run: bash infer_robot.sh", flush=True)
        return 0
    except Exception as error:
        print(f"Initialization failed; do not start motors until resolved: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
