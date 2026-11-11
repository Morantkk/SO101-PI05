"""Prepare a fixed split and train-only statistics without changing source data."""

import json
from pathlib import Path

import tyro
import numpy as np

from pi.data import _make_bool_mask, _apply_delta_actions
from pi.shared import normalize
from pi.lerobot_v3 import LeRobotV3Dataset, file_sha256
from pi.training.instance_config import get_config


def prepare(config_name: str = "pi05_so101_fold"):
    """Compute numeric statistics on the exact retained training action chunks."""
    config = get_config(config_name)
    dc = config.data
    root = Path(dc.repo_id)
    ranges_path = Path(dc.frame_ranges_manifest)
    split_path = Path(dc.split_manifest)
    output = config.assets_dirs / dc.asset_id
    if (output / "norm_stats.json").exists() or split_path.exists():
        raise FileExistsError("Prepared artifacts already exist; use a new asset version instead of overwriting")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Prepared artifacts must be outside the source dataset")
    before = {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    old_snapshot = json.loads((ranges_path.parent / "source_snapshot_before.json").read_text())
    if before != {name: value["sha256"] for name, value in old_snapshot.items()}:
        raise ValueError("Source files no longer match the verified preprocessing snapshot")
    old_norm_path = Path(json.loads(ranges_path.read_text())["assets_dir"]) / "norm_stats.json"
    old_artifacts = {str(p): file_sha256(p) for p in [ranges_path, old_norm_path]}
    info = json.loads((root / "meta/info.json").read_text())
    if info["features"]["action"]["names"] != info["features"]["observation.state"]["names"]:
        raise ValueError("State and action joint order differ")
    ordered = np.random.RandomState(42).permutation(info["total_episodes"])
    train = ordered[: -dc.test_ep_num].tolist()
    test = ordered[-dc.test_ep_num :].tolist()
    split = {
        "version": "so101_fold_h30_train90_v1",
        "seed": 42,
        "data_root": str(root.resolve()),
        "dataset_info_sha256": before["meta/info.json"],
        "frame_ranges_sha256": file_sha256(ranges_path),
        "source_files": before,
        "train": train,
        "test": test,
    }
    horizon = config.model.action_horizon
    offsets = {key: [i / info["fps"] for i in range(horizon)] for key in dc.action_sequence_keys}
    train_ds = LeRobotV3Dataset(
        root,
        episodes=train,
        delta_timestamps=offsets,
        frame_ranges_manifest=ranges_path,
        decode_images=False,
    )
    test_ds = LeRobotV3Dataset(root, episodes=test, frame_ranges_manifest=ranges_path, decode_images=False)
    state = np.stack(train_ds.table["observation.state"]).astype(np.float32)
    actions = np.stack(train_ds.table["action"]).astype(np.float32)
    ends = np.empty(len(train_ds), dtype=int)
    for lo, hi in train_ds._bounds.values():
        ends[lo:hi] = hi - 1
    state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
    mask = _make_bool_mask(*dc.delta_action_dims)
    for lo in range(0, len(train_ds), 512):
        indices = np.arange(lo, min(lo + 512, len(train_ds)))
        future = np.minimum(indices[:, None] + np.arange(horizon), ends[indices, None])
        chunks = actions[future].copy()
        # Call the same delta helper used in training for each block start.
        for i, current in enumerate(indices):
            chunks[i] = _apply_delta_actions({"state": state[current], "actions": chunks[i]}, mask)["actions"]
        state_stats.update(state[indices].astype(np.float64))
        action_stats.update(chunks.astype(np.float64))
    stats = {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    for value in stats.values():
        if not all(np.isfinite(getattr(value, key)).all() for key in ["mean", "std", "q01", "q99"]):
            raise ValueError("Non-finite normalization statistics")
        if not (np.all(value.std > 0) and np.all(value.q99 > value.q01)):
            raise ValueError("Degenerate normalization statistics")
    after = {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if before != after or any(file_sha256(path) != sha for path, sha in old_artifacts.items()):
        raise ValueError("Source data or old preprocessing artifacts changed")
    output.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2) + "\n")
    normalize.save(output, stats)
    provenance = {
        "config_name": config_name,
        "data_root": str(root.resolve()),
        "train_episodes": train,
        "test_episodes": test,
        "train_frames": len(train_ds),
        "test_frames": len(test_ds),
        "action_horizon": horizon,
        "fps": info["fps"],
        "action_vectors": len(train_ds) * horizon,
        "state_dim": state.shape[-1],
        "delta_action_dims": list(dc.delta_action_dims),
        "quantiles": "SO101-PI05 RunningStats, 5000 bins",
        "split_sha256": file_sha256(split_path),
        "norm_stats_sha256": file_sha256(output / "norm_stats.json"),
        "old_artifacts_sha256": old_artifacts,
        "source_files_unchanged": True,
        "statistics_include_validation": False,
        "training_started": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    tyro.cli(prepare)
