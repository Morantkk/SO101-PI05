"""Prepare and verify a train-only LiOS view of episodes 81 through the current end."""

from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from pi.data import _make_bool_mask, _apply_delta_actions
from pi.shared import normalize
from pi.lerobot_v3 import LeRobotV3Dataset, file_sha256

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/path/to/so101_dataset")
OLD_MANIFEST = Path("/home/YOUR_USER/projects/openpi_laundry/finetuning/fold100_clean_v1_20261106/manifest.json")
OLD_SPLIT = PROJECT / "assets/pi05_so101_fold/fold100_h30_train90_wrist_fixed_v1/split.json"
OUTPUT = PROJECT / "assets/pi05_so101_fold_81plus/fold81_197_h30_trainonly_wristfixed_20261108"
TRAIN_START = 81
HORIZON = 30
FPS = 30
JOINTS = np.asarray([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
GRIPPERS = np.asarray([5, 11])
CAMERAS = (
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def snapshot(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        str(path.relative_to(root)): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True))


def sustained_motion(values: np.ndarray) -> list[tuple[int, int]]:
    movement = np.abs(values[6:] - values[:-6])
    active = (movement[:, JOINTS].max(axis=1) >= 1.0) | (movement[:, GRIPPERS].max(axis=1) >= 2.0)
    return [(int(start), int(end + 6)) for start, end in runs(active) if end - start >= 3]


def make_ranges(table: pd.DataFrame, old_manifest: dict) -> tuple[dict[str, list[int]], list[dict]]:
    ranges: dict[str, list[int]] = {}
    records: list[dict] = []
    for episode, group in table.groupby("episode_index", sort=True):
        episode = int(episode)
        count = len(group)
        if episode < TRAIN_START:
            ranges[str(episode)] = [0, count]
            continue
        if not np.array_equal(group["frame_index"].to_numpy(), np.arange(count)):
            raise ValueError(f"Non-contiguous source frame indices in episode {episode}")
        state = np.stack(group["observation.state"]).astype(np.float32)
        action = np.stack(group["action"]).astype(np.float32)
        state_motion = sustained_motion(state)
        action_motion = sustained_motion(action)
        if not state_motion or not action_motion:
            raise ValueError(f"No sustained state/action movement in episode {episode}")
        start = max(0, min(state_motion[0][0], action_motion[0][0]) - 3)
        end = min(count, max(state_motion[-1][1], action_motion[-1][1]) + 15)
        if end - start < HORIZON:
            raise ValueError(f"Trim leaves fewer than {HORIZON} frames in episode {episode}")
        if str(episode) in old_manifest["frame_ranges"]:
            if [start, end] != old_manifest["frame_ranges"][str(episode)]:
                raise ValueError(f"Previously verified trim changed in episode {episode}")
        ranges[str(episode)] = [start, end]
        records.append(
            {
                "episode": episode,
                "source_frames": count,
                "start_inclusive": start,
                "end_exclusive": end,
                "kept_frames": end - start,
                "removed_start": start,
                "removed_end": count - end,
                "state_motion_start": state_motion[0][0],
                "action_motion_start": action_motion[0][0],
                "state_motion_end": state_motion[-1][1],
                "action_motion_end": action_motion[-1][1],
            }
        )
    return ranges, records


def check_camera_metadata(root: Path, info: dict, episodes: pd.DataFrame) -> None:
    for episode in episodes.to_dict("records"):
        for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            values = [episode[f"videos/{camera}/{field}"] for camera in CAMERAS]
            if not np.allclose(values, values[0], rtol=0, atol=1e-6):
                raise ValueError(f"Camera {field} does not align in episode {episode['episode_index']}")
        for camera in CAMERAS:
            video_path = root / info["video_path"].format(
                video_key=camera,
                chunk_index=int(episode[f"videos/{camera}/chunk_index"]),
                file_index=int(episode[f"videos/{camera}/file_index"]),
            )
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing camera video for episode {episode['episode_index']}: {video_path}")


def compute_stats(
    table: pd.DataFrame,
    ranges: dict[str, list[int]],
    train_episodes: list[int],
) -> tuple[dict, int]:
    state = np.stack(table["observation.state"]).astype(np.float32)
    actions = np.stack(table["action"]).astype(np.float32)
    if state.shape != actions.shape or state.shape[1] != 12:
        raise ValueError(f"Expected matching SO101 state/action dimensions of 12, got {state.shape}/{actions.shape}")

    train_indices: list[np.ndarray] = []
    episode_end = np.full(len(table), -1, dtype=np.int64)
    for episode in train_episodes:
        group = table[table["episode_index"] == episode]
        global_indices = group.index.to_numpy(dtype=np.int64)
        start, end = ranges[str(episode)]
        retained = global_indices[start:end]
        if len(retained) != end - start or len(retained) < HORIZON:
            raise ValueError(f"Invalid selected range for episode {episode}")
        train_indices.append(retained)
        episode_end[retained] = retained[-1]

    selected = np.concatenate(train_indices)
    if np.any(episode_end[selected] < selected):
        raise ValueError("Retained rows do not have an episode-local end index")
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    delta_mask = _make_bool_mask(5, -1, 5, -1)
    for offset in range(0, len(selected), 512):
        indices = selected[offset : offset + 512]
        future = np.minimum(indices[:, None] + np.arange(HORIZON), episode_end[indices, None])
        if not np.all(
            table.iloc[future.ravel()]["episode_index"].to_numpy().reshape(future.shape)
            == table.iloc[indices]["episode_index"].to_numpy()[:, None]
        ):
            raise ValueError("Action horizon crosses an episode boundary")
        chunks = actions[future].copy()
        for batch_index, source_index in enumerate(indices):
            chunks[batch_index] = _apply_delta_actions(
                {"state": state[source_index], "actions": chunks[batch_index]}, delta_mask
            )["actions"]
        state_stats.update(state[indices].astype(np.float64))
        action_stats.update(chunks.astype(np.float64))

    stats = {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    for name, values in stats.items():
        for field in ("mean", "std", "q01", "q99"):
            array = np.asarray(getattr(values, field))
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite {name}.{field} normalization statistic")
        if np.any(values.std <= 0) or np.any(values.q99 <= values.q01):
            raise ValueError(f"Degenerate {name} normalization statistic")
    return stats, len(selected) * HORIZON


def decode_alignment_samples(root: Path, ranges_path: Path, train_episodes: list[int], report_dir: Path) -> dict:
    dataset = LeRobotV3Dataset(
        root,
        episodes=train_episodes,
        frame_ranges_manifest=ranges_path,
        decode_images=True,
    )
    checks = []
    visual_samples = []
    ratios = (0.0, 0.5, 1.0)
    visual_episodes = [episode for episode in (100, 120, 150, 180, 197) if episode in train_episodes]
    for episode, (start, end) in dataset._bounds.items():
        if episode not in train_episodes:
            raise ValueError(f"Unexpected episode loaded: {episode}")
        for ratio in ratios:
            local_index = start + round((end - start - 1) * ratio)
            sample = dataset[local_index]
            if int(sample["episode_index"]) != episode:
                raise ValueError(f"Dataset index leaked across episodes at {episode}/{ratio}")
            original_frame = int(sample["frame_index"])
            if not ranges_path.is_file():
                raise FileNotFoundError(ranges_path)
            image_shapes = {}
            image_sizes = {}
            images = []
            for camera in CAMERAS:
                image = sample[camera]
                if image.dtype != np.uint8 or image.shape != (480, 640, 3):
                    raise ValueError(
                        f"Invalid decoded image in episode {episode}: {camera} {image.shape}/{image.dtype}"
                    )
                if float(image.std()) < 1.0 or float(image.mean()) < 1.0 or float(image.mean()) > 254.0:
                    raise ValueError(f"Blank/invalid decoded frame in episode {episode}: {camera}")
                image_shapes[camera] = list(image.shape)
                image_sizes[camera] = int(image.nbytes)
                if episode in visual_episodes and ratio in (0.0, 0.5, 1.0):
                    thumb = Image.fromarray(image)
                    thumb.thumbnail((320, 240))
                    images.append((camera, thumb.copy()))
            checks.append(
                {
                    "episode": episode,
                    "retained_position": ratio,
                    "original_frame_index": original_frame,
                    "decoded_cameras": list(CAMERAS),
                    "image_shapes": image_shapes,
                }
            )
            if images:
                visual_samples.append((episode, ratio, images))

    width, height = 3 * 330, len(visual_samples) * 270
    montage = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(montage)
    camera_names = {
        "observation.images.front": "front",
        "observation.images.left_wrist": "left_wrist",
        "observation.images.right_wrist": "right_wrist",
    }
    for row_index, (episode, ratio, images) in enumerate(visual_samples):
        top = row_index * 270
        draw.text((4, top + 2), f"episode {episode}, retained position {ratio:.1f}", fill="black")
        for column, (camera, image) in enumerate(images):
            x = column * 330
            draw.text((x + 4, top + 24), camera_names[camera], fill="black")
            montage.paste(image, (x, top + 45))
    montage_path = report_dir / "camera_sample_montage.jpg"
    montage.save(montage_path, quality=90)
    return {
        "status": "passed",
        "sampled_episodes": len(train_episodes),
        "sample_positions_per_episode": list(ratios),
        "decoded_image_count": len(checks) * len(CAMERAS),
        "frame_count": len(checks),
        "exact_pts_lookup": "passed; LeRobotV3Dataset raises when the exact timestamped RGB frame is missing",
        "camera_metadata_timestamps_and_file_ids_aligned": True,
        "image_dimensions": [480, 640, 3],
        "camera_mapping_visual_sample_episodes": visual_episodes,
        "camera_sample_montage": montage_path.name,
        "checks": checks,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Prepared output already exists; use a new version: {OUTPUT}")
    temporary = OUTPUT.with_name(f"{OUTPUT.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)
    if not OLD_MANIFEST.is_file() or not OLD_SPLIT.is_file():
        raise FileNotFoundError("Previously verified 100-episode trim/split artifacts are required")

    info = json.loads((DATA_ROOT / "meta/info.json").read_text())
    episodes = int(info["total_episodes"])
    frames = int(info["total_frames"])
    if info["fps"] != FPS or episodes <= TRAIN_START or not info["codebase_version"].startswith("v3."):
        raise ValueError(
            f"Unexpected LeRobot dataset metadata: {info['fps']=}, {episodes=}, {info['codebase_version']=}"
        )
    train_episodes = list(range(TRAIN_START, episodes))
    excluded_episodes = list(range(TRAIN_START))
    old_manifest = json.loads(OLD_MANIFEST.read_text())
    old_split = json.loads(OLD_SPLIT.read_text())

    try:
        temporary.mkdir(parents=True)
        before = snapshot(DATA_ROOT)
        save_json(temporary / "source_snapshot_before.json", before)
        old_data_hashes = {
            name: digest for name, digest in old_split["source_files"].items() if name.startswith("data/")
        }
        if not old_data_hashes or any(before[name]["sha256"] != digest for name, digest in old_data_hashes.items()):
            raise ValueError("Existing episode 0-99 numeric parquet files changed since the previous verified run")

        data_files = sorted((DATA_ROOT / "data").glob("*/*.parquet"))
        table = pd.concat([pd.read_parquet(path) for path in data_files], ignore_index=True)
        table = table.sort_values("index").reset_index(drop=True)
        if len(table) != frames or not np.array_equal(table["index"].to_numpy(), np.arange(frames)):
            raise ValueError("LeRobot global frame indices or total frame count do not match metadata")
        if table["episode_index"].nunique() != episodes or not np.array_equal(
            np.sort(table["episode_index"].unique()), np.arange(episodes)
        ):
            raise ValueError("Source episodes are missing, duplicated, or outside the expected range")
        state = np.stack(table["observation.state"]).astype(np.float32)
        action = np.stack(table["action"]).astype(np.float32)
        if state.shape != action.shape or state.shape[1] != 12:
            raise ValueError(f"Expected 12-dimensional SO101 state/action arrays, got {state.shape}/{action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("State/action table contains NaN or infinite values")

        expected_names = [
            "left_shoulder_pan.pos",
            "left_shoulder_lift.pos",
            "left_elbow_flex.pos",
            "left_wrist_flex.pos",
            "left_wrist_roll.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_shoulder_lift.pos",
            "right_elbow_flex.pos",
            "right_wrist_flex.pos",
            "right_wrist_roll.pos",
            "right_gripper.pos",
        ]
        if info["features"]["action"]["names"] != expected_names:
            raise ValueError("Action feature order is not the expected bimanual SO101 order")
        if info["features"]["observation.state"]["names"] != expected_names:
            raise ValueError("State feature order is not the expected bimanual SO101 order")
        if not set(CAMERAS).issubset(info["features"]):
            raise ValueError("One or more required RGB camera channels are absent")

        task_table = pd.read_parquet(DATA_ROOT / "meta/tasks.parquet").reset_index()
        if len(task_table) != 1 or task_table.iloc[0]["task"] != "Fold the clothing neatly using both arms":
            raise ValueError(f"Unexpected task definitions in the source dataset: {task_table.to_dict('records')}")
        episode_metadata = pd.concat(
            [pd.read_parquet(path) for path in sorted((DATA_ROOT / "meta/episodes").glob("*/*.parquet"))],
            ignore_index=True,
        )
        if len(episode_metadata) != episodes:
            raise ValueError("Episode metadata row count does not match dataset metadata")
        check_camera_metadata(DATA_ROOT, info, episode_metadata)

        ranges, range_records = make_ranges(table, old_manifest)
        retained_frames = sum(record["kept_frames"] for record in range_records)
        trimmed_start = sum(record["removed_start"] for record in range_records)
        trimmed_end = sum(record["removed_end"] for record in range_records)
        if len(range_records) != len(train_episodes):
            raise ValueError(
                f"Expected trim records for {len(train_episodes)} training episodes, got {len(range_records)}"
            )

        selected = table[table["episode_index"] >= TRAIN_START]
        timestamps_bad = 0
        timestamps_max_abs_error_frames = 0.0
        for _episode, group in selected.groupby("episode_index", sort=True):
            timestamp = group["timestamp"].to_numpy(dtype=np.float64)
            frame_diff = np.diff(timestamp) * FPS
            bad = np.abs(frame_diff - 1.0) > 0.05
            timestamps_bad += int(bad.sum())
            if frame_diff.size:
                timestamps_max_abs_error_frames = max(
                    timestamps_max_abs_error_frames, float(np.max(np.abs(frame_diff - 1.0)))
                )
        if timestamps_bad:
            raise ValueError(f"Found {timestamps_bad} non-30-FPS numeric frame intervals in selected episodes")

        stats, action_vectors = compute_stats(table, ranges, train_episodes)
        manifest = {
            "version": "so101_fold_81plus_h30_trainonly_wristfixed_20261108",
            "status": "prepared_and_verified_training_not_started",
            "data_root": str(DATA_ROOT.resolve()),
            "dataset_info_sha256": before["meta/info.json"]["sha256"],
            "source_snapshot_sha256": file_sha256(temporary / "source_snapshot_before.json"),
            "fps": FPS,
            "original_frames": frames,
            "total_episodes": episodes,
            "frame_ranges": ranges,
            "retained_frames": retained_frames,
            "frames": {"train": retained_frames, "validation": 0},
            "split": {"train": train_episodes, "validation": [], "excluded": excluded_episodes},
            "asset_id": OUTPUT.name,
            "assets_dir": str(OUTPUT),
            "action_horizon": HORIZON,
            "trimming": {
                "method": "state OR action motion across 6 frames; at least 3 consecutive active windows",
                "joint_threshold_degrees": 1.0,
                "gripper_threshold_points": 2.0,
                "pre_onset_context_frames": 3,
                "post_motion_context_frames": 15,
                "interior_frames_removed": 0,
                "end_exclusive": True,
            },
            "action_alignment": {
                "fps": FPS,
                "horizon": HORIZON,
                "start_offset_frames": 0,
                "end_policy": "repeat the last retained action within this episode",
                "preserve_original_frame_index": True,
                "interior_splicing": False,
            },
            "camera_mapping": {
                "base_0_rgb": "observation.images.front",
                "left_wrist_0_rgb": "observation.images.left_wrist",
                "right_wrist_0_rgb": "observation.images.right_wrist",
            },
        }
        save_json(temporary / "original_frame_ranges.json", manifest)

        split = {
            "version": "so101_fold_81plus_trainonly_h30_v1",
            "seed": 42,
            "data_root": str(DATA_ROOT.resolve()),
            "dataset_info_sha256": before["meta/info.json"]["sha256"],
            "frame_ranges_sha256": file_sha256(temporary / "original_frame_ranges.json"),
            "source_files": {name: value["sha256"] for name, value in before.items()},
            "train": train_episodes,
            "test": [],
            "excluded": excluded_episodes,
            "validation_enabled": False,
            "excluded_reason": "Episodes 0-80 are intentionally not used for this training run; they are not validation data.",
        }
        save_json(temporary / "split.json", split)
        normalize.save(temporary, stats)

        alignment = decode_alignment_samples(
            DATA_ROOT, temporary / "original_frame_ranges.json", train_episodes, temporary
        )
        old_retained = sum(
            old_manifest["frame_ranges"][str(ep)][1] - old_manifest["frame_ranges"][str(ep)][0]
            for ep in range(TRAIN_START, 100)
        )
        if retained_frames < old_retained:
            raise ValueError("Retained frame count for episodes 81-99 should be included in the new range")

        after = snapshot(DATA_ROOT)
        if before != after:
            raise ValueError("Source dataset changed while preprocessing; generated artifacts are not trusted")
        save_json(temporary / "trim_records.json", range_records)
        verification = {
            "status": "passed",
            "episodes_total": episodes,
            "episodes_train": len(train_episodes),
            "train_episode_range": [train_episodes[0], train_episodes[-1]],
            "excluded_episodes": len(excluded_episodes),
            "validation_episodes": 0,
            "source_frames_total": frames,
            "selected_source_frames": int(len(selected)),
            "retained_train_frames": retained_frames,
            "trimmed_leading_frames": trimmed_start,
            "trimmed_trailing_frames": trimmed_end,
            "interior_frames_removed": 0,
            "existing_81_99_boundaries_unchanged": True,
            "old_numeric_data_parquet_hashes_match": True,
            "state_action_shape": list(state.shape[1:]),
            "nonfinite_state_action_values": 0,
            "action_horizon": HORIZON,
            "normalized_action_vectors": action_vectors,
            "delta_action_mask": list(_make_bool_mask(5, -1, 5, -1)),
            "validation_used_for_statistics": False,
            "frame_periods_outside_30fps_tolerance": timestamps_bad,
            "max_frame_period_error_in_frame_units": timestamps_max_abs_error_frames,
            "camera_metadata_alignment": "passed",
            "camera_image_alignment": alignment,
            "source_files_unchanged_during_preparation": True,
            "training_started": False,
        }
        save_json(temporary / "verification.json", verification)
        provenance = {
            "config_name": "pi05_so101_fold_81plus",
            "data_root": str(DATA_ROOT.resolve()),
            "train_episodes": train_episodes,
            "excluded_episodes": excluded_episodes,
            "validation_episodes": [],
            "train_frames": retained_frames,
            "action_horizon": HORIZON,
            "fps": FPS,
            "action_vectors": action_vectors,
            "state_dim": 12,
            "delta_action_dims": [5, -1, 5, -1],
            "quantiles": "LiOS RunningStats, 5000 bins",
            "split_sha256": file_sha256(temporary / "split.json"),
            "norm_stats_sha256": file_sha256(temporary / "norm_stats.json"),
            "source_files_unchanged": True,
            "statistics_include_validation": False,
            "training_started": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "trimmed_leading_frames": trimmed_start,
            "trimmed_trailing_frames": trimmed_end,
            "interior_frames_removed": 0,
            "wrist_camera_review": {
                "episodes_sampled": alignment["camera_mapping_visual_sample_episodes"],
                "review_image": "camera_sample_montage.jpg",
                "hardware_camera_accessed": False,
            },
        }
        save_json(temporary / "provenance.json", provenance)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(OUTPUT)
        print(json.dumps(verification, indent=2, ensure_ascii=False))
        print(f"Prepared assets: {OUTPUT}")
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
