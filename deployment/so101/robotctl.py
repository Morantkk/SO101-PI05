"""Select a validated SO101 model, control execution, home, or clear runtime state."""

import os
import re
import json
import time
import uuid
import fcntl
import shutil
import hashlib
import argparse
import tempfile
import subprocess
from pathlib import Path
from contextlib import contextmanager

from init_inference import stop_unit, initialize, unit_state
from model_registry import load_profiles

ROOT = Path(__file__).resolve().parent
TASK_UNIT = "so101-deploy-policy-task"
RTC = dict(
    enabled=True,
    num_steps=5,
    max_guidance_weight=5.0,
    execution_horizon=15,
    initial_delay_steps=5,
    delay_buffer=10,
    prefix_schedule="exp",
    paper="2506.07339v1",
)
PROFILES = {
    "local100000": dict(
        checkpoint="checkpoints/pi05_so101_fold_81plus/fold81_197_b8_e10_20261108/step100000",
        checkpoint_step=100000,
        model_sha256="042e6ea3ddbcad174dcacdab4fef2c1d6b848e72fb3cf2cc5ddfdb4fcbd2c454",
        metadata_sha256="cf129832922185b1a1489dc21c1d71a3889c398b4379227f9ff633b7378dc1ce",
        report_dir="local_step100000",
    ),
}
NORM_SHA256 = "02bcec98c2125383b9b2fd266b00384f4c5137768d8682b38eecd1dc6389cce0"


def read_json(path):
    return json.loads(path.read_text())


def atomic_text(path, text):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def command_lock(root):
    with (root / ".robotctl.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another control command is preparing; use stop to cancel it") from error
        yield


def select_config(model, rtc, root=ROOT):
    config = read_json(root / "config.json")
    if rtc not in ("on", "off"):
        raise ValueError("RTC must be on or off")
    profiles = load_profiles(root, PROFILES)
    profile = profiles.get(model)
    if profile is None:
        path = Path(model).expanduser().resolve()
        profile = next((p for p in profiles.values() if path == (root.parents[1] / p["checkpoint"]).resolve()), None)
        if profile is None:
            raise ValueError("Unknown model; use models or register-model before selecting a new checkpoint")
    # Drop model-specific overrides left by a previously selected custom profile.
    for key in ("assets_base_dir", "asset_id", "frame_ranges_manifest", "split_manifest",
                "train_episodes", "validation_episodes", "train_episode_range", "dataset_root"):
        config.pop(key, None)
    config.update(
        policy="pi05_so101_fold_81plus",
        norm_stats_sha256=NORM_SHA256,
        horizon=30,
        fps=30,
        num_steps=40,
        weights_variant="raw",
        execution_enabled=False,
        rtc=RTC.copy() if rtc == "on" else {"enabled": False},
        inference_backend="rtc" if rtc == "on" else "ordinary",
    )
    config.update(profile)
    machine_path = root / "machine.json"
    if machine_path.exists():
        machine = read_json(machine_path)
        allowed = {"dataset_root", "assets_base_dir", "asset_id", "frame_ranges_manifest", "split_manifest"}
        if set(machine) - allowed:
            raise ValueError("Unexpected machine-specific configuration fields")
        for key, value in machine.items():
            if key not in profile:
                config[key] = value
    checkpoint = (root.parents[1] / config["checkpoint"]).resolve()
    config["checkpoint"] = str(checkpoint)
    for filename in ("__0_0.distcp", ".metadata"):
        if not (checkpoint / filename).is_file():
            raise ValueError(f"Missing checkpoint file: {checkpoint / filename}")
    actual = hashlib.sha256((checkpoint / ".metadata").read_bytes()).hexdigest()
    if actual != config["metadata_sha256"]:
        raise ValueError("Checkpoint metadata does not match the selected model")
    return config


def cancel_startup(root):
    (root / "STOP_MOTION").touch()
    atomic_text(root / "INFERENCE_RESET_GENERATION", uuid.uuid4().hex + "\n")
    stop_unit(TASK_UNIT)
    (root / "STOP_MOTION").touch()


def reset_robot(mode, root):
    subprocess.run(["bash", str(root / "reset_robot.sh"), mode], check=True)


def stop_inference_stack():
    for component in ("inference", "receiver", "sender"):
        stop_unit(f"so101-deploy-{component}")


def validate_profile(command, project):
    with tempfile.TemporaryDirectory(prefix="so101-model-check-") as directory:
        output = Path(directory) / "profile.json"
        result = subprocess.run(command + ["--output", str(output)], cwd=project, capture_output=True, text=True)
        if result.returncode:
            raise ValueError(result.stderr.strip() or result.stdout.strip())
        if not output.is_file():
            raise ValueError("Model validator did not write a profile")
        return read_json(output)


def register_model(args, root):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.name):
        raise ValueError("Use letters, numbers, underscores or hyphens for the model name")
    if args.name in load_profiles(root, PROFILES):
        raise ValueError("Model name already exists; choose a new name")
    if unit_state(TASK_UNIT).get("ActiveState") in ("active", "activating", "deactivating"):
        raise RuntimeError("Stop a running robot command before registering a model")
    project = root.parents[1] / "so101-policy"
    command = [os.environ.get("PIXI_BIN", str(Path.home() / ".pixi/bin/pixi")), "run", "--as-is", "-e", "dev", "python",
               "scripts/deployment/register_so101_model.py", "--checkpoint", str(args.checkpoint.expanduser().resolve()),
               "--policy", args.policy, "--step", str(args.step)]
    for key in ("assets", "dataset"):
        if getattr(args, key):
            command += [f"--{key}", str(getattr(args, key).expanduser().resolve())]
    print("Checking tensor interface, dataset/statistics and checkpoint checksums; no motor access...", flush=True)
    profile = validate_profile(command, project)
    profile["report_dir"] = "registered_models/" + args.name
    with command_lock(root):
        profiles = load_profiles(root, PROFILES)
        if args.name in profiles:
            raise ValueError("Model was registered concurrently")
        custom = {k: v for k, v in profiles.items() if k not in PROFILES}
        custom[args.name] = profile
        atomic_text(root / "models.json", json.dumps(custom, indent=2) + "\n")
    print(f"Registered {args.name}. Use: bash robotctl.sh run --model {args.name} --rtc off (or on)")


def check_model(args, root):
    """Check registered identity on CPU; optionally run device-free GPU inference."""
    config = select_config(args.model, args.rtc, root)
    project = root.parents[1] / "so101-policy"
    pixi = [os.environ.get("PIXI_BIN", str(Path.home() / ".pixi/bin/pixi")), "run", "--as-is", "-e", "dev", "python"]
    command = pixi + ["scripts/deployment/register_so101_model.py", "--checkpoint", config["checkpoint"],
                      "--policy", config["policy"], "--step", str(config["checkpoint_step"])]
    if "assets_base_dir" in config:
        command += ["--assets", str(Path(config["assets_base_dir"]) / config["policy"] / config["asset_id"])]
    if "dataset_root" in config:
        command += ["--dataset", config["dataset_root"]]
    verified = validate_profile(command, project)
    for key in ("model_sha256", "metadata_sha256", "norm_stats_sha256"):
        if verified[key] != config[key]:
            raise ValueError(f"Registered identity mismatch: {key}")
    print(f"Model interface, weights, statistics and data verified: {args.model}; RTC {args.rtc}; no motor access")
    if args.offline:
        if unit_state("so101-deploy-inference").get("ActiveState") in ("active", "activating", "deactivating"):
            raise RuntimeError("Stop the inference service before running a separate offline GPU check")
        report = root / "validation" / (time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
        report.mkdir(parents=True)
        config["report_dir"] = str(report)
        path = report / "config.json"
        atomic_text(path, json.dumps(config, indent=2))
        subprocess.run(pixi + ["scripts/deployment/serve_so101.py", "--offline", "--config", str(path)],
                       cwd=project, check=True)
        print(f"Offline predictions saved: {report / 'offline_prediction.json'}; motor commands sent: 0")


def launch(generation, root):
    (root / "logs").mkdir(exist_ok=True)
    log = root / "logs/policy-task.log"
    subprocess.run(
        [
            "systemd-run",
            "--user",
            "--collect",
            f"--unit={TASK_UNIT}",
            f"--working-directory={root}",
            f"--property=StandardOutput=append:{log}",
            f"--property=StandardError=append:{log}",
            f"--setenv=SO101_EXPECTED_GENERATION={generation}",
            "/usr/bin/bash",
            str(root / "infer_robot.sh"),
        ],
        check=True,
    )


def wait_running(config, generation, root, timeout=600):
    deadline = time.monotonic() + timeout
    next_progress = 0
    while time.monotonic() < deadline:
        if (root / "INFERENCE_RESET_GENERATION").read_text().strip() != generation:
            raise RuntimeError("Startup cancelled by a stop or reset command")
        if unit_state(TASK_UNIT).get("ActiveState") not in ("active", "activating"):
            raise RuntimeError(f"Startup service exited; inspect {root / 'logs/policy-task.log'}")
        if time.monotonic() >= next_progress:
            try:
                progress = read_json(root / "inference_status.json")
                stage = progress.get("status", "starting")
                detail = progress.get("detail", "") if stage == "waiting_for_observations" else ""
            except (OSError, ValueError):
                stage, detail = "loading checkpoint and verifying checksums", ""
            print(f"Waiting for step{config['checkpoint_step']}: {stage} {detail}".rstrip(), flush=True)
            next_progress = time.monotonic() + 5
        try:
            execution = read_json(root / "execution_status.json")
            inference = read_json(root / "inference_status.json")
            action = read_json(root / "received_action.json")["message"]
        except (OSError, ValueError, KeyError):
            time.sleep(0.5)
            continue
        if execution.get("status") == "failed":
            raise RuntimeError(execution.get("error", "Controller failed"))
        if (
            execution.get("status") == "running"
            and execution.get("checkpoint_step") == config["checkpoint_step"]
            and execution.get("model_sha256") == config["model_sha256"]
            and execution.get("rtc") == config["rtc"]
            and execution.get("command_cycles", 0) > 0
            and time.time() - execution.get("updated_at", 0) < 1
            and inference.get("model_sha256") == config["model_sha256"]
            and inference.get("rtc") == config["rtc"]
            and action.get("model_sha256") == config["model_sha256"]
            and ("rtc" in action) == config["rtc"]["enabled"]
            and not (root / "STOP_MOTION").exists()
        ):
            return execution
        time.sleep(0.5)
    raise RuntimeError(f"Startup timed out; inspect {root / 'logs/policy-task.log'}")


def run_model(args, root=ROOT):
    # Resolve identity before stopping a working deployment or changing its config.
    config = select_config(args.model, args.rtc, root)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    checkpoint=config["checkpoint"],
                    checkpoint_step=config["checkpoint_step"],
                    rtc=config["rtc"],
                    home_first=args.home,
                    background=True,
                    actions=["stop old runtime", "archive runtime", "load model", "execute"],
                ),
                indent=2,
            )
        )
        return
    with command_lock(root):
        print(f"Preparing step{config['checkpoint_step']}, RTC {args.rtc}; stopping old runtime...", flush=True)
        report = initialize(root)
        archive = Path(report["archive"])
        shutil.copy2(root / "config.json", archive / "config.json")
        atomic_text(root / "config.json", json.dumps(config, indent=2) + "\n")
        if args.home:
            print("Returning home before inference...", flush=True)
            reset_robot("--home", root)
            returned = read_json(root / "execution_status.json")
            if not returned.get("returned_to_start") or returned.get("hold_errors"):
                raise RuntimeError("Home was not verified; inference was not started")
        generation = report["generation"]
        if (root / "INFERENCE_RESET_GENERATION").read_text().strip() != generation:
            raise RuntimeError("Startup cancelled during preparation")
        launch(generation, root)
    print(f"Loading step{config['checkpoint_step']}; RTC {args.rtc}. Waiting for real-robot control...", flush=True)
    try:
        wait_running(config, generation, root)
    except BaseException:
        # Never stop a newer run that replaced this startup while it was waiting.
        if (root / "INFERENCE_RESET_GENERATION").read_text().strip() == generation:
            cancel_startup(root)
            reset_robot("--stop", root)
            stop_inference_stack()
        raise
    print("Real-robot inference is running in the background. Use: bash robotctl.sh stop", flush=True)


def main(argv=None, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Load the selected model and start continuous real-robot control")
    run.add_argument("--model", required=True, help="A name from `models`, or a registered checkpoint path")
    run.add_argument("--rtc", required=True, choices=("on", "off"))
    run.add_argument("--home", action="store_true", help="Return home before starting control")
    run.add_argument("--dry-run", action="store_true", help="Validate selection without changes or motor access")
    for name, help_text in (
        ("stop", "Stop movement and retain position hold"),
        ("home", "Stop movement, half-open grippers, hold 0.5s, then return home; do not start inference"),
        ("clear-cache", "Stop and archive inference queues/state; preserve weights and calibration"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--dry-run", action="store_true", help="Print the action without changing anything")
    commands.add_parser("models", help="List supported model names and checkpoint paths")
    commands.add_parser("status", help="Show configuration, service states and execution status")
    check = commands.add_parser("check-model", help="Validate model/data on CPU without motors; optionally test offline GPU inference")
    check.add_argument("--model", required=True)
    check.add_argument("--rtc", required=True, choices=("on", "off"))
    check.add_argument("--offline", action="store_true", help="Also run GPU inference on a recorded dataset sample, without cameras/motors")
    register = commands.add_parser("register-model", help="Validate and register a compatible SO101-PI05 SO101 checkpoint; no motors")
    register.add_argument("--name", required=True)
    register.add_argument("--checkpoint", type=Path, required=True)
    register.add_argument("--step", type=int, required=True)
    register.add_argument("--policy", default="pi05_so101_fold_81plus")
    register.add_argument("--assets", type=Path, help="Prepared asset directory with norm_stats.json, split.json and provenance.json")
    register.add_argument("--dataset", type=Path, help="Dataset matching the prepared assets")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            run_model(args, root)
        elif args.command == "register-model":
            register_model(args, root)
        elif args.command == "check-model":
            check_model(args, root)
        elif args.command == "models":
            for name, profile in load_profiles(root, PROFILES).items():
                print(f"{name}: {root.parents[1] / profile['checkpoint']}")
        elif args.command == "status":
            config = read_json(root / "config.json")
            result = dict(
                checkpoint=config["checkpoint"],
                rtc=config.get("rtc"),
                stop_latch=(root / "STOP_MOTION").exists(),
                services={
                    name: unit_state(f"so101-deploy-{name}")
                    for name in ("policy-task", "joint-trial", "return-home", "inference", "receiver", "sender")
                },
            )
            try:
                result["execution"] = read_json(root / "execution_status.json")
            except FileNotFoundError:
                result["execution"] = None
            print(json.dumps(result, indent=2))
        elif args.dry_run:
            print(f"Would {args.command}; no services, files or motors changed")
        elif args.command == "stop":
            cancel_startup(root)
            reset_robot("--stop", root)
            stop_inference_stack()
        else:
            with command_lock(root):
                if args.command == "clear-cache":
                    initialize(root)
                    print(
                        "Inference queues and process caches cleared. Weights/calibration/compiler disk caches preserved."
                    )
                else:
                    cancel_startup(root)
                    reset_robot("--stop", root)
                    stop_inference_stack()
                    reset_robot("--home", root)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Check: bash robotctl.sh status", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Control command failed: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
