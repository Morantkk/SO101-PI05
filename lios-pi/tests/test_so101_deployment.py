"""Check deployment action decoding independently of robot hardware."""

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/deployment"))
from serve_so101 import decode_actions, select_offline_episode


def test_action_decoding_restores_absolute_joints_and_keeps_grippers():
    state = np.arange(12, dtype=np.float32) * 3
    mask = np.array([True] * 5 + [False] + [True] * 5 + [False])
    absolute = np.tile(np.arange(12, dtype=np.float32) + 10, (30, 1))
    targets = absolute - np.where(mask, state, 0)
    stats = SimpleNamespace(q01=np.full(12, -50.0), q99=np.full(12, 100.0))
    normalized = 2 * (targets - stats.q01) / (stats.q99 - stats.q01 + 1e-6) - 1
    padded = np.pad(normalized, ((0, 0), (0, 20)), constant_values=1234)
    original = padded.copy()
    decoded = decode_actions(padded, state, stats, mask)
    np.testing.assert_allclose(decoded, absolute, atol=1e-5)
    np.testing.assert_array_equal(padded, original)
    assert decoded.shape == (30, 12)


def test_action_decoding_rejects_nonfinite_values():
    stats = SimpleNamespace(q01=np.zeros(12), q99=np.ones(12))
    with pytest.raises(ValueError):
        decode_actions(np.full((30, 32), np.nan), np.zeros(12), stats, np.ones(12, dtype=bool))


def test_offline_sample_selection_marks_training_only_dataset():
    assert select_offline_episode(list(range(81, 198)), []) == (81, "train")
    assert select_offline_episode([0, 1], [2]) == (2, "validation")
    with pytest.raises(ValueError, match="No episodes"):
        select_offline_episode([], [])

