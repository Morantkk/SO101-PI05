"""Read-only local LeRobot v3 data with explicit episode splits and frame ranges."""

import json
import hashlib
from pathlib import Path

import av
import numpy as np
import pandas as pd


def file_sha256(path):
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_local_v3(repo_id):
    """Detect a local v3 dataset without a Hub lookup."""
    info_path = Path(repo_id).expanduser() / "meta/info.json"
    return info_path.is_file() and json.loads(info_path.read_text())["codebase_version"].startswith("v3.")


class LeRobotV3Metadata:
    """Metadata fields used by LiOS loaders."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.fps = float(self.info["fps"])
        self.total_episodes = int(self.info["total_episodes"])
        self.video_keys = [key for key, value in self.info["features"].items() if value["dtype"] == "video"]


def load_episode_split(root, path, frame_ranges_manifest):
    """Validate a saved split against its source dataset and trimming manifest."""
    split = json.loads(Path(path).read_text())
    metadata = LeRobotV3Metadata(root)
    if Path(split["data_root"]).resolve() != metadata.root:
        raise ValueError("Split points to a different dataset")
    if split["dataset_info_sha256"] != file_sha256(metadata.root / "meta/info.json"):
        raise ValueError("Dataset metadata changed after split preparation")
    if frame_ranges_manifest is None or split["frame_ranges_sha256"] != file_sha256(frame_ranges_manifest):
        raise ValueError("Trimming manifest does not match the saved split")
    train, test = split["train"], split["test"]
    excluded = split.get("excluded", [])
    groups = [set(train), set(test), set(excluded)]
    if any(len(items) != len(values) for items, values in zip(groups, [train, test, excluded], strict=True)):
        raise ValueError("Episode split contains duplicates or overlap")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Episode split contains duplicates or overlap")
    if groups[0] | groups[1] | groups[2] != set(range(metadata.total_episodes)):
        raise ValueError("Train, test, and excluded splits must cover every source episode")
    for name, expected in split["source_files"].items():
        if file_sha256(metadata.root / name) != expected:
            raise ValueError(f"Source file changed after split preparation: {name}")
    return train, test


class LeRobotV3Dataset:
    """Return independent numeric arrays and timestamp-aligned RGB images."""

    def __init__(self, root, *, episodes=None, delta_timestamps=None, frame_ranges_manifest=None, decode_images=True):
        self.meta = LeRobotV3Metadata(root)
        self.delta_timestamps = delta_timestamps or {}
        self.decode_images = decode_images
        self._videos = {}
        self._frame_cache = {}
        table = (
            pd.concat(
                [pd.read_parquet(p) for p in sorted((self.meta.root / "data").glob("*/*.parquet"))],
                ignore_index=True,
            )
            .sort_values("index")
            .reset_index(drop=True)
        )
        if len(table) != self.meta.info["total_frames"] or not np.array_equal(table["index"], np.arange(len(table))):
            raise ValueError("Source frame count or global indices do not match metadata")
        ranges = None
        if frame_ranges_manifest is not None:
            manifest = json.loads(Path(frame_ranges_manifest).read_text())
            if Path(manifest["data_root"]).resolve() != self.meta.root:
                raise ValueError("Trimming manifest points to a different dataset")
            if manifest["dataset_info_sha256"] != file_sha256(self.meta.root / "meta/info.json"):
                raise ValueError("Dataset metadata changed after trimming")
            ranges = manifest["frame_ranges"]
            if set(ranges) != {str(int(ep)) for ep in table.episode_index.unique()}:
                raise ValueError("Trimming manifest must cover every source episode")
        selected = []
        wanted = set(episodes) if episodes is not None else set(table.episode_index.unique())
        for ep, group in table.groupby("episode_index", sort=True):
            if ep not in wanted:
                continue
            if not np.array_equal(group.frame_index, np.arange(len(group))):
                raise ValueError(f"Source frame indices are not contiguous in episode {ep}")
            lo, hi = ranges[str(int(ep))] if ranges else (0, len(group))
            if not 0 <= lo < hi <= len(group):
                raise ValueError(f"Invalid retained range for episode {ep}")
            selected.append(group.iloc[lo:hi])
        if len(selected) != len(wanted) or not selected:
            raise ValueError("Selected episodes are missing or empty")
        self.table = pd.concat(selected, ignore_index=True)
        self._bounds = {}
        for ep, group in self.table.groupby("episode_index", sort=False):
            self._bounds[int(ep)] = (int(group.index[0]), int(group.index[-1]) + 1)
        self._episodes = pd.concat(
            [pd.read_parquet(p) for p in sorted((self.meta.root / "meta/episodes").glob("*/*.parquet"))],
            ignore_index=True,
        ).set_index("episode_index")
        tasks = pd.read_parquet(self.meta.root / "meta/tasks.parquet")
        if "task_index" not in tasks.columns or "task" not in tasks.columns:
            tasks = tasks.reset_index()
        self._tasks = dict(zip(tasks.task_index, tasks.task, strict=True))

    def __len__(self):
        return len(self.table)

    def _image(self, ep, key, frame_index):
        row = self._episodes.loc[ep]
        prefix = f"videos/{key}"
        relative = self.meta.info["video_path"].format(
            video_key=key,
            chunk_index=int(row[f"{prefix}/chunk_index"]),
            file_index=int(row[f"{prefix}/file_index"]),
        )
        path = str(self.meta.root / relative)
        target = float(row[f"{prefix}/from_timestamp"]) + frame_index / self.meta.fps
        cache_key = (path, round(target * self.meta.fps))
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key].copy()
        if path not in self._videos:
            if len(self._videos) >= 6:
                self._videos.pop(next(iter(self._videos))).close()
            self._videos[path] = av.open(path)
        container = self._videos[path]
        stream = container.streams.video[0]
        container.seek(max(0, int((target - 2 / self.meta.fps) / stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is not None and abs(float(frame.pts * stream.time_base) - target) < 1e-4:
                image = frame.to_ndarray(format="rgb24")
                if len(self._frame_cache) >= 24:
                    self._frame_cache.pop(next(iter(self._frame_cache)))
                self._frame_cache[cache_key] = image
                return image.copy()
            if frame.pts is not None and float(frame.pts * stream.time_base) > target + 1 / self.meta.fps:
                break
        raise ValueError(f"No exact video frame at {target}: {path}")

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.table.iloc[index]
        ep = int(row.episode_index)
        item = {
            key: np.array(value, copy=True) if isinstance(value, (np.ndarray, list)) else value
            for key, value in row.items()
        }
        item["task"] = self._tasks[int(row.task_index)]
        lo, hi = self._bounds[ep]
        for key, offsets in self.delta_timestamps.items():
            indices = np.clip(index + np.rint(np.asarray(offsets) * self.meta.fps).astype(int), lo, hi - 1)
            item[key] = np.stack(self.table.iloc[indices][key]).astype(np.float32, copy=True)
        if self.decode_images:
            for key in self.meta.video_keys:
                item[key] = self._image(ep, key, int(row.frame_index))
        return item

    def close(self):
        """Release decoder handles before passing a dataset to workers."""
        for container in self._videos.values():
            container.close()
        self._videos.clear()
        self._frame_cache.clear()
