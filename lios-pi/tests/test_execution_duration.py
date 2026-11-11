"""Duration selection must preserve bounded trials and explicit stop semantics."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deployment/so101"))
from execution_options import resolve_duration


def test_continuous_full_task_and_explicit_override():
    assert resolve_duration(True, configured=0) is None
    assert resolve_duration(True, requested=0, configured=90) is None
    assert resolve_duration(True, requested=90, configured=0) == 90
    assert resolve_duration(True) == 60


def test_small_trial_remains_bounded_in_continuous_deployment():
    assert resolve_duration(False, configured=0) == 3
    assert resolve_duration(False, requested=0) == 3


@pytest.mark.parametrize("duration", [-1, 121, float("inf"), float("nan")])
def test_invalid_duration_rejected(duration):
    with pytest.raises(ValueError):
        resolve_duration(True, requested=duration)
