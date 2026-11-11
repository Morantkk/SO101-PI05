"""Validated model profiles; registration never opens motor or camera devices."""

import re
import json


def load_profiles(root, builtins):
    path = root / "models.json"
    custom = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(custom, dict) or set(custom) & set(builtins):
        raise ValueError("Invalid model registry or reserved model name")
    allowed = {"checkpoint", "checkpoint_step", "model_sha256", "metadata_sha256", "report_dir",
               "policy", "norm_stats_sha256", "assets_base_dir", "asset_id", "frame_ranges_manifest",
               "split_manifest", "dataset_root", "horizon", "fps", "num_steps", "weights_variant"}
    for name, profile in custom.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("Invalid registered model name")
        if not isinstance(profile, dict) or set(profile) - allowed:
            raise ValueError(f"Invalid fields in model profile: {name}")
        for key in ("checkpoint", "checkpoint_step", "model_sha256", "metadata_sha256", "policy", "norm_stats_sha256"):
            if key not in profile:
                raise ValueError(f"Missing {key} in model profile {name}")
    return {**builtins, **custom}
