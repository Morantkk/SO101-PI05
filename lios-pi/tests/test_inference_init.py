"""Check initialization with fake service management and no hardware access."""

import json
import runpy
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deployment/so101/init_inference.py"
MODULE = runpy.run_path(str(SCRIPT))


def test_init_archives_only_runtime_and_preserves_configuration(tmp_path):
    for name in MODULE["TRANSIENT_FILES"]:
        (tmp_path / name).write_text(name)
    preserved = ["config.json", "model.distcp", "full_task_result.json", "manual_release_status.json"]
    for name in preserved:
        (tmp_path / name).write_text("preserved")
    stopped = []
    report = MODULE["initialize"](tmp_path, stopped.append, lambda: None)
    assert stopped == list(MODULE["UNITS"])
    assert report["status"] == "initialized"
    archive = Path(report["archive"])
    for name in MODULE["TRANSIENT_FILES"]:
        assert not (tmp_path / name).exists()
        assert (archive / name).read_text() == name
    assert all((tmp_path / name).read_text() == "preserved" for name in preserved)
    assert (tmp_path / "STOP_MOTION").exists()
    assert not (tmp_path / ".reset_in_progress").exists()
    first_generation = report["generation"]
    again = MODULE["initialize"](tmp_path, stopped.append, lambda: None)
    assert again["generation"] != first_generation
    assert again["archive"] != report["archive"]


@pytest.mark.parametrize("fail_endpoint", [False, True])
def test_failed_shutdown_keeps_old_status_and_stop_latch(tmp_path, fail_endpoint):
    (tmp_path / "inference_status.json").write_text("old")
    def fail(*_args):
        raise RuntimeError("shutdown failed")
    stop = (lambda _unit: None) if fail_endpoint else fail
    verify = fail if fail_endpoint else (lambda: None)
    with pytest.raises(RuntimeError, match="shutdown failed"):
        MODULE["initialize"](tmp_path, stop, verify)
    assert (tmp_path / "inference_status.json").read_text() == "old"
    assert (tmp_path / "STOP_MOTION").exists()
    assert not (tmp_path / ".reset_in_progress").exists()
    assert json.loads((tmp_path / "inference_reset_status.json").read_text())["status"] == "failed"


def test_concurrent_reset_rejected_without_stopping_anything(tmp_path):
    (tmp_path / ".reset_in_progress").mkdir()
    stopped = []
    with pytest.raises(RuntimeError, match="Another reset"):
        MODULE["initialize"](tmp_path, stopped.append, lambda: None)
    assert not stopped
    assert (tmp_path / ".reset_in_progress").exists()


@pytest.mark.parametrize("stuck", [False, True])
def test_inference_stop_has_bounded_fallback_only_when_needed(monkeypatch, stuck):
    module = runpy.run_path(str(SCRIPT))
    calls = []
    stopped = False
    clock = 0

    def state(_unit):
        return dict(LoadState="loaded", ActiveState="inactive" if stopped else "active",
                    MainPID="0" if stopped else "123")

    def run(command, **_kwargs):
        nonlocal stopped
        calls.append(command)
        if "kill" in command or not stuck:
            stopped = True
        return SimpleNamespace(returncode=0)

    def now():
        nonlocal clock
        clock += 1
        return clock

    module["stop_unit"].__globals__["unit_state"] = state
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(module["time"], "monotonic", now)
    monkeypatch.setattr(module["time"], "sleep", lambda _: None)
    module["stop_unit"]("lios-deploy-inference")
    assert stopped
    assert "--no-block" in calls[0]
    assert any("kill" in command for command in calls) is stuck
    if stuck:
        assert "--signal=SIGKILL" in calls[-1]
        assert "--kill-whom=all" in calls[-1]


def test_controller_timeout_never_uses_forced_termination(monkeypatch):
    module = runpy.run_path(str(SCRIPT))
    calls = []
    module["stop_unit"].__globals__["unit_state"] = lambda _: dict(
        LoadState="loaded", ActiveState="active", MainPID="123")

    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Timed out stopping"):
        module["stop_unit"]("lios-deploy-joint-trial")
    assert len(calls) == 1
    assert not any("kill" in command for command in calls)


def test_forced_inference_exit_must_be_verified(monkeypatch):
    module = runpy.run_path(str(SCRIPT))
    module["stop_unit"].__globals__["unit_state"] = lambda _: dict(
        LoadState="loaded", ActiveState="deactivating", MainPID="123")
    clock = 0

    def now():
        nonlocal clock
        clock += 5
        return clock

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(module["time"], "monotonic", now)
    monkeypatch.setattr(module["time"], "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="has not fully stopped"):
        module["stop_unit"]("lios-deploy-inference")
