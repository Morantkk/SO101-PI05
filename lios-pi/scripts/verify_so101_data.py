"""Independently verify prepared statistics and real SO101 dataset samples on CPU."""

import sys
import json
import dataclasses
from pathlib import Path

import av
import numpy as np
import torch
import pandas as pd

from pi.data import MultiLeRobotLoader, SimpleLeRobotLoader
from pi.lerobot_v3 import file_sha256, load_episode_split
from pi.training.instance_config import get_config


def verify():
    """Check all numeric training targets and sample real image preprocessing."""
    config = get_config("pi05_so101_fold")
    dc = config.data
    root = Path(dc.repo_id)
    output = config.assets_dirs / dc.asset_id
    train, test = load_episode_split(root, dc.split_manifest, dc.frame_ranges_manifest)
    stats = dc.load_norm_stats(config.assets_dirs, config.model)
    ranges = json.loads(Path(dc.frame_ranges_manifest).read_text())["frame_ranges"]
    source = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").glob("*/*.parquet"))], ignore_index=True)
    sums = np.zeros(12)
    squares = np.zeros(12)
    count = 0
    retained_states = []
    joints = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    # Recompute each episode independently from the original parquet rows.
    for ep in train:
        group = source[source.episode_index == ep].sort_values("frame_index")
        lo, hi = ranges[str(ep)]
        state = np.stack(group["observation.state"])[lo:hi].astype(np.float32)
        actions = np.stack(group.action)[lo:hi].astype(np.float32)
        retained_states.append(state)
        future = np.minimum(np.arange(len(state))[:, None] + np.arange(30), len(state) - 1)
        targets = actions[future].copy()
        targets[:, :, joints] -= state[:, None, :][:, :, joints]
        flat = targets.reshape(-1, 12).astype(np.float64)
        sums += flat.sum(axis=0)
        squares += np.square(flat).sum(axis=0)
        count += len(flat)
    state = np.concatenate(retained_states).astype(np.float64)
    errors = {}
    for key, mean, std in [
        ("state", state.mean(axis=0), state.std(axis=0)),
        ("actions", sums / count, np.sqrt(squares / count - (sums / count) ** 2)),
    ]:
        errors[key] = {"mean": float(abs(mean - stats[key].mean).max()), "std": float(abs(std - stats[key].std).max())}
        assert max(errors[key].values()) < 1e-7
    for incompatible in [dataclasses.replace(config.model, action_horizon=50)]:
        try:
            dc.load_norm_stats(config.assets_dirs, incompatible)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched horizon was not rejected")
    loaders = {mode: SimpleLeRobotLoader(config, batch_size=1, mode=mode) for mode in ["train", "test"]}
    checked = 0
    exact_rgb = 0
    for mode, loader in loaders.items():
        dataset = loader.dataset
        expected_episodes = train if mode == "train" else test
        assert set(dataset.table.episode_index) == set(expected_episodes)
        for ep, (_lo, hi) in dataset._bounds.items():
            sample = dataset[hi - 1]
            # Every episode's final chunk must repeat its last retained action.
            expected = np.asarray(
                source[(source.episode_index == ep) & (source.frame_index == ranges[str(ep)][1] - 1)].iloc[0].action
            )
            np.testing.assert_array_equal(sample["action"], np.repeat(expected[None, :], 30, axis=0))
            checked += 1
        for index in [0, len(dataset) // 2, len(dataset) - 1]:
            raw = dataset[index]
            ep = int(raw["episode_index"])
            row = dataset._episodes.loc[ep]
            for key in dataset.meta.video_keys:
                prefix = f"videos/{key}"
                relative = dataset.meta.info["video_path"].format(
                    video_key=key,
                    chunk_index=int(row[f"{prefix}/chunk_index"]),
                    file_index=int(row[f"{prefix}/file_index"]),
                )
                target = float(row[f"{prefix}/from_timestamp"]) + int(raw["frame_index"]) / dataset.meta.fps
                with av.open(str(root / relative)) as container:
                    stream = container.streams.video[0]
                    container.seek(max(0, int((target - 1) / stream.time_base)), stream=stream, backward=True)
                    for frame in container.decode(stream):
                        if frame.pts is not None and abs(float(frame.pts * stream.time_base) - target) < 1e-4:
                            np.testing.assert_array_equal(raw[key], frame.to_ndarray(format="rgb24"))
                            exact_rgb += 1
                            break
                    else:
                        raise AssertionError("Independent RGB decode failed")
            transformed = loader[index]
            assert transformed["state"].shape == (1, 32)
            assert transformed["actions"].shape == (30, 32)
            assert transformed["tokenized_prompt"].shape == (200,)
            assert all(image.shape == (224, 224, 3) for image in transformed["image"].values())
            np.testing.assert_array_equal(transformed["actions"][:, 12:], 0.0)
            x = transformed["actions"][:, :12]
            absolute = (x + 1) / 2 * (stats["actions"].q99 - stats["actions"].q01 + 1e-6) + stats["actions"].q01
            absolute[:, joints] += raw["observation.state"][joints]
            np.testing.assert_allclose(absolute, raw["action"], atol=2e-5, rtol=1e-6)
        dataset.close()
    ds_config = dataclasses.replace(dc, norm_stats=stats, policy_name=config.name, use_quantile_norm=True)
    multi = MultiLeRobotLoader(
        datasets=[ds_config],
        batch_size=1,
        action_horizon=30,
        action_dim=32,
        max_token_len=200,
        discrete_state_input=True,
        mode="train",
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))
    from data_loader import _build_dataset_configs, collate_and_preprocess

    assert _build_dataset_configs([config])[0].delta_action_dims == (5, -1, 5, -1)
    batch = next(
        iter(
            torch.utils.data.DataLoader(
                multi,
                batch_size=1,
                num_workers=2,
                collate_fn=collate_and_preprocess,
            )
        )
    )
    assert batch["actions"].shape == (1, 30, 32)
    assert batch["image"]["base_0_rgb"].shape == (1, 3, 224, 224)
    assert torch.isfinite(batch["image"]["base_0_rgb"]).all()
    split = json.loads(Path(dc.split_manifest).read_text())
    assert all(file_sha256(root / name) == sha for name, sha in split["source_files"].items())
    provenance = json.loads((output / "provenance.json").read_text())
    assert all(file_sha256(path) == sha for path, sha in provenance["old_artifacts_sha256"].items())
    report = {
        "train_episodes": len(train),
        "test_episodes": len(test),
        "train_frames": len(loaders["train"]),
        "test_frames": len(loaders["test"]),
        "train_action_vectors": count,
        "normalization_errors": errors,
        "episode_end_padding_checks": checked,
        "independent_exact_rgb_checks": exact_rgb,
        "transformed_samples_checked": 6,
        "multiprocess_loader_passed": True,
        "source_and_old_artifacts_unchanged": True,
        "training_started": False,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(4)
    verify()
