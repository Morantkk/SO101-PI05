"""Verify CLI orchestration without starting services or accessing motor buses."""

import sys
import json
import hashlib
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deployment/so101"))
import robotctl


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    monkeypatch.setitem(robotctl.PROFILES, "cloud27010", dict(robotctl.PROFILES["local100000"], checkpoint_step=27010))
    metadata = b"test checkpoint metadata"
    for name, original in robotctl.PROFILES.items():
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / ".metadata").write_bytes(metadata)
        (checkpoint / "__0_0.distcp").write_bytes(b"test shard")
        monkeypatch.setitem(
            robotctl.PROFILES,
            name,
            dict(original, checkpoint=str(checkpoint), metadata_sha256=hashlib.sha256(metadata).hexdigest()),
        )
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                left_port="left serial",
                right_port="right serial",
                prompt="Fold clothing",
                checkpoint="old model",
                rtc={"enabled": True},
            )
        )
    )
    return tmp_path


def arguments(**kwargs):
    return SimpleNamespace(**dict(dict(model="cloud27010", rtc="on", home=False, dry_run=False), **kwargs))


@pytest.mark.parametrize("model,step", [("cloud27010", 27010), ("local100000", 100000)])
@pytest.mark.parametrize("rtc", ["on", "off"])
def test_selection_switches_identity_and_rtc_together(deployment, model, step, rtc):
    config = robotctl.select_config(model, rtc, deployment)
    assert config["checkpoint_step"] == step
    assert config["model_sha256"] == robotctl.PROFILES[model]["model_sha256"]
    assert config["rtc"]["enabled"] is (rtc == "on")
    assert config["num_steps"] == 40
    if rtc == "on":
        assert config["rtc"]["num_steps"] == 5
        assert config["rtc"]["execution_horizon"] == 15
    assert config["left_port"] == "left serial"
    assert config["execution_enabled"] is False
    by_path = robotctl.select_config(config["checkpoint"], rtc, deployment)
    assert by_path == config


def test_unknown_or_corrupt_model_never_stops_services(deployment, monkeypatch):
    monkeypatch.setattr(robotctl, "initialize", lambda *_: pytest.fail("Must validate before stopping"))
    with pytest.raises(ValueError, match="Unknown model"):
        robotctl.run_model(arguments(model="typo"), deployment)
    (deployment / "cloud27010/.metadata").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="metadata"):
        robotctl.run_model(arguments(), deployment)


def test_dry_runs_do_not_write_files_or_start_services(deployment, monkeypatch):
    monkeypatch.setattr(robotctl, "initialize", lambda *_: pytest.fail("No mutation in dry-run"))
    monkeypatch.setattr(robotctl, "cancel_startup", lambda *_: pytest.fail("No mutation in dry-run"))
    before = {p: p.read_bytes() for p in deployment.rglob("*") if p.is_file()}
    assert robotctl.main(["run", "--model", "local100000", "--rtc", "off", "--home", "--dry-run"], deployment) == 0
    for command in ("home", "clear-cache", "stop"):
        assert robotctl.main([command, "--dry-run"], deployment) == 0
    assert before == {p: p.read_bytes() for p in deployment.rglob("*") if p.is_file()}


def test_old_backend_marker_does_not_prevent_explicit_rtc_selection(deployment, monkeypatch):
    path = deployment / "config.json"
    config = json.loads(path.read_text())
    config["inference_backend"] = "pre_rtc"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(robotctl, "initialize", lambda *_: pytest.fail("Must reject before stopping"))
    selected = robotctl.select_config("local100000", "on", deployment)
    assert selected["rtc"]["enabled"] is True
    assert selected["inference_backend"] == "rtc"
    selected = robotctl.select_config("local100000", "off", deployment)
    assert selected["num_steps"] == 40
    assert selected["rtc"] == {"enabled": False}


def test_custom_model_profile_and_switch_back_do_not_leak_overrides(deployment):
    profile = dict(robotctl.PROFILES["local100000"], policy="pi05_so101_fold_81plus", norm_stats_sha256="custom",
                   assets_base_dir="custom-assets", asset_id="custom", dataset_root="custom-data")
    (deployment / "models.json").write_text(json.dumps({"cloth_v2": profile}))
    selected = robotctl.select_config("cloth_v2", "on", deployment)
    assert selected["norm_stats_sha256"] == "custom"
    assert selected["rtc"]["enabled"] is True
    (deployment / "config.json").write_text(json.dumps(selected))
    ordinary = robotctl.select_config("local100000", "off", deployment)
    assert "assets_base_dir" not in ordinary
    assert ordinary["norm_stats_sha256"] == robotctl.NORM_SHA256
    assert robotctl.main(["models"], deployment) == 0


def test_registry_cannot_override_motor_configuration(deployment):
    (deployment / "models.json").write_text(json.dumps({"evil": dict(left_port="wrong-device")}))
    with pytest.raises(ValueError, match="Invalid fields"):
        robotctl.select_config("evil", "on", deployment)


def prepare_start(deployment, monkeypatch):
    events = []

    def initialize(root):
        events.append("initialize")
        (root / "STOP_MOTION").touch()
        (root / "INFERENCE_RESET_GENERATION").write_text("generation\n")
        (root / "archive").mkdir()
        return dict(archive=str(root / "archive"), generation="generation")

    def launch(generation, root):
        assert json.loads((root / "config.json").read_text())["checkpoint_step"] == 27010
        assert (root / "STOP_MOTION").exists()
        assert generation == "generation"
        events.append("launch")

    monkeypatch.setattr(robotctl, "initialize", initialize)
    monkeypatch.setattr(robotctl, "launch", launch)
    monkeypatch.setattr(robotctl, "wait_running", lambda *_: events.append("wait"))
    return events


def test_start_archives_previous_config_and_homes_before_launch(deployment, monkeypatch):
    events = prepare_start(deployment, monkeypatch)

    def home(mode, root):
        assert mode == "--home"
        events.append("home")
        (root / "execution_status.json").write_text(json.dumps(dict(returned_to_start=True, hold_errors=[])))

    monkeypatch.setattr(robotctl, "reset_robot", home)
    robotctl.run_model(arguments(home=True), deployment)
    assert events == ["initialize", "home", "launch", "wait"]
    assert json.loads((deployment / "archive/config.json").read_text())["checkpoint"] == "old model"


def test_failed_home_cannot_launch_inference(deployment, monkeypatch):
    events = prepare_start(deployment, monkeypatch)
    monkeypatch.setattr(
        robotctl,
        "reset_robot",
        lambda *_: (deployment / "execution_status.json").write_text(json.dumps(dict(returned_to_start=False))),
    )
    with pytest.raises(RuntimeError, match="Home was not verified"):
        robotctl.run_model(arguments(home=True), deployment)
    assert events == ["initialize"]
    assert (deployment / "STOP_MOTION").exists()


def test_failed_start_cancels_controller_and_holds(deployment, monkeypatch):
    events = prepare_start(deployment, monkeypatch)

    def fail(*_):
        raise RuntimeError("No predictions")

    monkeypatch.setattr(robotctl, "wait_running", fail)
    monkeypatch.setattr(robotctl, "cancel_startup", lambda *_: events.append("cancel"))
    monkeypatch.setattr(robotctl, "reset_robot", lambda mode, _: events.append(mode))
    monkeypatch.setattr(robotctl, "stop_unit", lambda unit: events.append(unit))
    with pytest.raises(RuntimeError, match="No predictions"):
        robotctl.run_model(arguments(), deployment)
    assert events == [
        "initialize", "launch", "cancel", "--stop", "so101-deploy-inference", "so101-deploy-receiver",
        "so101-deploy-sender",
    ]


def test_home_stops_runtime_stack_before_returning(deployment, monkeypatch):
    events = []
    monkeypatch.setattr(robotctl, "cancel_startup", lambda *_: events.append("cancel"))
    monkeypatch.setattr(robotctl, "reset_robot", lambda mode, _: events.append(mode))
    monkeypatch.setattr(robotctl, "stop_unit", lambda unit: events.append(unit))
    assert robotctl.main(["home"], deployment) == 0
    assert events == [
        "cancel", "--stop", "so101-deploy-inference", "so101-deploy-receiver", "so101-deploy-sender", "--home",
    ]


def test_stop_cancels_pending_startup_without_waiting_for_lock(deployment, monkeypatch):
    (deployment / "INFERENCE_RESET_GENERATION").write_text("old")
    stopped = []
    monkeypatch.setattr(robotctl, "stop_unit", stopped.append)
    monkeypatch.setattr(robotctl, "reset_robot", lambda mode, _: stopped.append(mode))
    with robotctl.command_lock(deployment):
        assert robotctl.main(["stop"], deployment) == 0
    assert stopped == [robotctl.TASK_UNIT, "--stop", "so101-deploy-inference", "so101-deploy-receiver",
                       "so101-deploy-sender"]
    assert (deployment / "STOP_MOTION").exists()
    assert (deployment / "INFERENCE_RESET_GENERATION").read_text() != "old"


def test_wait_rejects_cancelled_generation(deployment):
    (deployment / "INFERENCE_RESET_GENERATION").write_text("new")
    with pytest.raises(RuntimeError, match="cancelled"):
        robotctl.wait_running({}, "old", deployment)


@pytest.mark.parametrize("wrong_rtc", [False, True])
def test_startup_confirmation_requires_matching_rtc_and_fresh_controller(deployment, monkeypatch, wrong_rtc):
    config = robotctl.select_config("cloud27010", "on", deployment)
    (deployment / "INFERENCE_RESET_GENERATION").write_text("generation")
    (deployment / "execution_status.json").write_text(
        json.dumps(dict(status="running", checkpoint_step=27010, model_sha256=config["model_sha256"], rtc=config["rtc"],
                        command_cycles=3, updated_at=robotctl.time.time()))
    )
    (deployment / "inference_status.json").write_text(
        json.dumps(dict(model_sha256=config["model_sha256"], rtc=config["rtc"]))
    )
    message = dict(model_sha256=config["model_sha256"])
    if not wrong_rtc:
        message["rtc"] = dict(session="session", chunk_id="chunk")
    (deployment / "received_action.json").write_text(json.dumps(dict(message=message)))
    monkeypatch.setattr(robotctl, "unit_state", lambda _: {"ActiveState": "active"})
    ticks = iter([0, 0, 0, 0, 2])
    monkeypatch.setattr(robotctl.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(robotctl.time, "sleep", lambda _: None)
    if wrong_rtc:
        with pytest.raises(RuntimeError, match="timed out"):
            robotctl.wait_running(config, "generation", deployment, timeout=1)
    else:
        assert robotctl.wait_running(config, "generation", deployment, timeout=1)["command_cycles"] == 3


def test_dead_startup_fails_without_waiting_for_timeout(deployment, monkeypatch):
    (deployment / "INFERENCE_RESET_GENERATION").write_text("generation")
    monkeypatch.setattr(robotctl, "unit_state", lambda _: {"ActiveState": "failed"})
    with pytest.raises(RuntimeError, match="service exited"):
        robotctl.wait_running({}, "generation", deployment)


def test_clear_cache_uses_full_runtime_initialization(deployment, monkeypatch):
    calls = []
    monkeypatch.setattr(robotctl, "initialize", lambda root: calls.append(root))
    assert robotctl.main(["clear-cache"], deployment) == 0
    assert calls == [deployment]


def test_cache_clear_stops_pending_startup_before_controller():
    import init_inference

    assert init_inference.UNITS[:2] == (robotctl.TASK_UNIT, "so101-deploy-joint-trial")
    assert init_inference.UNITS[-3:] == ("so101-deploy-inference", "so101-deploy-receiver", "so101-deploy-sender")


def test_model_and_rtc_must_be_explicit(deployment):
    for argv in (["run", "--model", "cloud27010"], ["run", "--rtc", "on"]):
        with pytest.raises(SystemExit) as error:
            robotctl.main(argv, deployment)
        assert error.value.code == 2


def test_subprocess_timeout_reported_without_traceback(deployment, monkeypatch, capsys):
    import subprocess

    def timeout(_root):
        raise subprocess.TimeoutExpired(["systemctl", "stop"], 10)

    monkeypatch.setattr(robotctl, "initialize", timeout)
    assert robotctl.main(["clear-cache"], deployment) == 1
    assert "Control command failed" in capsys.readouterr().out


def test_registration_reads_structured_output_and_reuses_new_model(deployment, monkeypatch):
    profile = dict(robotctl.PROFILES["local100000"], policy="pi05_so101_fold_81plus",
                   norm_stats_sha256=robotctl.NORM_SHA256)
    monkeypatch.setattr(robotctl, "unit_state", lambda _: {"ActiveState": "inactive"})

    def validate(command, **kwargs):
        Path(command[command.index("--output") + 1]).write_text(json.dumps(profile))
        return SimpleNamespace(returncode=0, stdout="Library informational message\n", stderr="")

    monkeypatch.setattr(robotctl.subprocess, "run", validate)
    args = ["register-model", "--name", "new_model", "--checkpoint", profile["checkpoint"], "--step", "100000"]
    assert robotctl.main(args, deployment) == 0
    assert robotctl.select_config("new_model", "on", deployment)["rtc"]["enabled"] is True
    assert robotctl.select_config("new_model", "off", deployment)["rtc"] == {"enabled": False}
    before = (deployment / "models.json").read_bytes()
    assert robotctl.main(args, deployment) == 1
    assert (deployment / "models.json").read_bytes() == before


def test_failed_registration_leaves_registry_and_config_unchanged(deployment, monkeypatch):
    monkeypatch.setattr(robotctl, "unit_state", lambda _: {"ActiveState": "inactive"})
    monkeypatch.setattr(robotctl.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=1, stderr="Tensor shape mismatch", stdout=""))
    before = (deployment / "config.json").read_bytes()
    assert robotctl.main(["register-model", "--name", "bad_model", "--checkpoint", str(deployment),
                          "--step", "1"], deployment) == 1
    assert not (deployment / "models.json").exists()
    assert (deployment / "config.json").read_bytes() == before
