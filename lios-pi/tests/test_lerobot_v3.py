"""CPU coverage for trimming, episode splits, action boundaries and SO101 semantics."""

import json

import numpy as np
import pandas as pd
import pytest

from pi.data import _make_bool_mask, _repack_transform, _pad_state_actions, _apply_delta_actions
from pi.lerobot_v3 import LeRobotV3Dataset, file_sha256, load_episode_split


@pytest.fixture
def v3_root(tmp_path):
    root = tmp_path / "source"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": 2,
        "total_frames": 12,
        "features": {},
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["fold"], name="task")).to_parquet(root / "meta/tasks.parquet")
    rows = [
        {
            "index": ep * 6 + frame,
            "episode_index": ep,
            "frame_index": frame,
            "timestamp": frame / 30,
            "task_index": 0,
            "observation.state": np.arange(12, dtype=np.float32) + 10,
            "action": np.full(12, ep * 100 + frame, dtype=np.float32),
        }
        for ep in range(2)
        for frame in range(6)
    ]
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet")
    pd.DataFrame({"episode_index": [0, 1]}).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    manifest = tmp_path / "ranges.json"
    manifest.write_text(
        json.dumps(
            {
                "data_root": str(root),
                "dataset_info_sha256": file_sha256(root / "meta/info.json"),
                "frame_ranges": {"0": [1, 5], "1": [2, 6]},
            }
        )
    )
    return root, manifest


def test_trim_preserves_source_frame_indices_and_pads_retained_boundaries(v3_root):
    root, manifest = v3_root
    dataset = LeRobotV3Dataset(
        root,
        frame_ranges_manifest=manifest,
        decode_images=False,
        delta_timestamps={"action": [0, 1 / 30, 2 / 30], "observation.state": [0, -1 / 30]},
    )
    assert len(dataset) == 8
    assert dataset[0]["frame_index"] == 1
    assert dataset[0]["task"] == "fold"
    np.testing.assert_array_equal(dataset[2]["action"][:, 0], [3, 4, 4])
    np.testing.assert_array_equal(dataset[3]["action"][:, 0], [4, 4, 4])
    np.testing.assert_array_equal(dataset[4]["action"][:, 0], [102, 103, 104])
    np.testing.assert_array_equal(dataset[0]["observation.state"][0], dataset[0]["observation.state"][1])
    # Numeric transforms must not mutate the source table or later reads.
    item = dataset[0]
    item["action"][:] = -1000
    assert dataset[0]["action"][0, 0] == 1


def test_episode_selection_does_not_leak_other_episodes(v3_root):
    root, manifest = v3_root
    train = LeRobotV3Dataset(root, episodes=[1], frame_ranges_manifest=manifest, decode_images=False)
    test = LeRobotV3Dataset(root, episodes=[0], frame_ranges_manifest=manifest, decode_images=False)
    assert set(train.table.episode_index) == {1}
    assert set(test.table.episode_index) == {0}
    assert train[0]["frame_index"] == 2


def test_split_rejects_overlap_and_changed_sources(v3_root, tmp_path):
    root, manifest = v3_root
    split = {
        "data_root": str(root),
        "dataset_info_sha256": file_sha256(root / "meta/info.json"),
        "frame_ranges_sha256": file_sha256(manifest),
        "train": [1],
        "test": [0],
        "source_files": {"meta/tasks.parquet": file_sha256(root / "meta/tasks.parquet")},
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    assert load_episode_split(root, path, manifest) == ([1], [0])
    split["test"] = [1]
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="overlap"):
        load_episode_split(root, path, manifest)
    split["test"] = [0]
    path.write_text(json.dumps(split))
    (root / "meta/tasks.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source file changed"):
        load_episode_split(root, path, manifest)


def test_split_supports_excluded_episodes_without_creating_a_test_set(v3_root, tmp_path):
    root, manifest = v3_root
    split = {
        "data_root": str(root),
        "dataset_info_sha256": file_sha256(root / "meta/info.json"),
        "frame_ranges_sha256": file_sha256(manifest),
        "train": [1],
        "test": [],
        "excluded": [0],
        "source_files": {},
    }
    path = tmp_path / "train_only_split.json"
    path.write_text(json.dumps(split))

    assert load_episode_split(root, path, manifest) == ([1], [])


def test_so101_delta_preserves_both_grippers_and_pads_only_at_end():
    step = {"state": np.arange(12, dtype=float) + 10, "actions": np.full((30, 12), 50.0)}
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    repacked = _repack_transform(
        "pi05_so101_fold",
        {
            "observation.state": step["state"],
            "action": step["actions"],
            "task": "fold",
            "observation.images.front": image,
            "observation.images.left_wrist": image,
            "observation.images.right_wrist": image,
        },
    )
    assert repacked["observation/cam_env"] is image
    result = _apply_delta_actions(step, _make_bool_mask(5, -1, 5, -1))
    np.testing.assert_array_equal(result["actions"][:, [5, 11]], 50.0)
    np.testing.assert_array_equal(result["actions"][0, [0, 4, 6, 10]], [40.0, 36.0, 34.0, 30.0])
    padded = _pad_state_actions(result, 32)
    assert padded["actions"].shape == (30, 32)
    np.testing.assert_array_equal(padded["actions"][:, 12:], 0.0)
    np.testing.assert_array_equal(padded["state"][12:], 0.0)


def test_trim_rejects_mismatched_dataset(v3_root):
    root, manifest = v3_root
    content = json.loads(manifest.read_text())
    content["dataset_info_sha256"] = "bad"
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="metadata changed"):
        LeRobotV3Dataset(root, frame_ranges_manifest=manifest, decode_images=False)
