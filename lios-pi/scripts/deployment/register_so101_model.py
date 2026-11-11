"""Describe a compatible LiOS checkpoint on CPU, without starting robot services."""

import json
import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
from normalizer import normalize_state_dict_key
from torch.distributed.checkpoint import FileSystemReader

from pi.lerobot_v3 import file_sha256, load_episode_split
from pi.training.instance_config import get_config
from pi.models_pytorch.pi0_pytorch import PI0Pytorch


def describe(checkpoint, policy, step, assets=None, dataset=None):
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if step < 0:
        raise ValueError("Checkpoint step must be non-negative")
    cfg = get_config(policy)
    # This deployment adapter has three RGB cameras, 12 physical joints and a 30-frame chunk.
    if not cfg.model.pi05 or cfg.model.action_dim != 32 or cfg.model.action_horizon != 30:
        raise ValueError("This adapter requires LiOS Pi0.5, action_dim=32 and horizon=30")
    if not cfg.model.discrete_state_input or cfg.model.state_history_frames != 1:
        raise ValueError("This adapter requires discrete state and one history frame")
    overrides = {}
    if assets:
        assets = assets.expanduser().resolve(strict=True)
        if assets.parent.name != cfg.name:
            raise ValueError(f"Assets must be under <assets-base>/{cfg.name}/<asset-id>")
        ranges = assets / "frame_ranges.json"
        if not ranges.exists():
            ranges = assets / "original_frame_ranges.json"
        cfg = dataclasses.replace(cfg, assets_base_dir=str(assets.parent.parent), data=dataclasses.replace(
            cfg.data, asset_id=assets.name, split_manifest=str(assets / "split.json"),
            frame_ranges_manifest=str(ranges)))
        overrides.update(assets_base_dir=str(assets.parent.parent), asset_id=assets.name,
                         split_manifest=str(assets / "split.json"), frame_ranges_manifest=str(ranges))
    if dataset:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=str(dataset.resolve(strict=True))))
    load_episode_split(cfg.data.repo_id, cfg.data.split_manifest, cfg.data.frame_ranges_manifest)
    stats = cfg.data.load_norm_stats(cfg.assets_dirs, cfg.model)
    if tuple(cfg.data.delta_action_dims) != (5, -1, 5, -1):
        raise ValueError("SO101 requires the 12D joint/gripper delta mask")
    for field in ("state", "actions"):
        if stats is None or np.asarray(stats[field].q01).shape != (12,) or np.asarray(stats[field].q99).shape != (12,):
            raise ValueError("Normalization must describe 12 physical state/action dimensions")
        if not np.isfinite(stats[field].q01).all() or not np.isfinite(stats[field].q99).all():
            raise ValueError("Normalization contains non-finite values")
        if np.any(np.asarray(stats[field].q99) < np.asarray(stats[field].q01)):
            raise ValueError("Normalization quantiles are reversed")
    shards = list(checkpoint.glob("*.distcp"))
    if len(shards) != 1 or shards[0].name != "__0_0.distcp":
        raise ValueError("Export the model as a single-shard LiOS DCP checkpoint before registration")
    metadata = FileSystemReader(checkpoint).read_metadata()
    stored = {normalize_state_dict_key(k): v for k, v in metadata.state_dict_metadata.items()}
    with torch.device("meta"):
        model = PI0Pytorch(cfg.model)
    live = model.state_dict()
    if len(stored) != len(metadata.state_dict_metadata) or set(stored) != set(live):
        raise ValueError("Checkpoint tensor keys do not match the selected LiOS model interface")
    for name, tensor in live.items():
        if tuple(stored[name].size) != tuple(tensor.shape):
            raise ValueError(f"Checkpoint tensor shape mismatch: {name}")
    shard_size = shards[0].stat().st_size
    for item in metadata.storage_data.values():
        if item.relative_path != "__0_0.distcp" or item.offset < 0 or item.length < 0 or item.offset + item.length > shard_size:
            raise ValueError("Incomplete or unsupported checkpoint storage")
    return dict(overrides, checkpoint=str(checkpoint), checkpoint_step=step, policy=policy,
                model_sha256=file_sha256(shards[0]), metadata_sha256=file_sha256(checkpoint / ".metadata"),
                norm_stats_sha256=file_sha256(cfg.assets_dirs / cfg.data.asset_id / "norm_stats.json"),
                dataset_root=cfg.data.repo_id, horizon=30, fps=30, num_steps=40, weights_variant="raw")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, help="Write the structured profile independently of library messages")
    args = parser.parse_args()
    profile = json.dumps(describe(args.checkpoint, args.policy, args.step, args.assets, args.dataset))
    if args.output:
        args.output.write_text(profile + "\n")
    else:
        print(profile)


if __name__ == "__main__":
    main()
