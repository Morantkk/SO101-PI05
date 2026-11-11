"""Fully load a pretrained SafeTensors checkpoint on CPU and verify every tensor."""

import json
from pathlib import Path

import tyro
import torch
from safetensors import safe_open
from safetensors.torch import load_model

from pi.lerobot_v3 import file_sha256
from pi.training.instance_config import get_config
from pi.models_pytorch.pi0_pytorch import PI0Pytorch


def verify(config_name: str = "pi05_so101_fold"):
    """Check shape compatibility, full loading, finiteness and loaded values."""
    config = get_config(config_name)
    path = Path(config.pytorch_weight_path) / "model.safetensors"
    sha = file_sha256(path)
    expected = (path.parent / "SHA256SUMS").read_text().split()[0]
    if sha != expected:
        raise ValueError("Pretrained checkpoint SHA-256 mismatch")
    print("Checkpoint SHA-256 verified; constructing full CPU model", flush=True)
    model = PI0Pytorch(config.model)
    state = model.state_dict()
    with safe_open(path, framework="pt", device="cpu") as reader:
        shape_mismatches = {
            key: [list(reader.get_slice(key).get_shape()), list(state[key].shape)]
            for key in reader.keys()
            if key in state and list(reader.get_slice(key).get_shape()) != list(state[key].shape)
        }
    if shape_mismatches:
        raise ValueError(f"Weight shapes do not match: {shape_mismatches}")
    missing, unexpected = load_model(model, str(path), strict=True, device="cpu")
    checked, finite = 0, True
    with torch.no_grad(), safe_open(path, framework="pt", device="cpu") as reader:
        for key in reader.keys():
            loaded = state[key]
            expected_tensor = reader.get_tensor(key).to(loaded.dtype)
            finite = finite and bool(torch.isfinite(loaded).all())
            if not torch.equal(loaded, expected_tensor):
                raise ValueError(f"Loaded tensor values differ: {key}")
            checked += 1
    if not finite:
        raise ValueError("Loaded checkpoint contains non-finite tensors")
    report = {
        "config_name": config_name,
        "weight_path": str(path),
        "sha256": sha,
        "strict_loading_passed": True,
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "shape_mismatches": shape_mismatches,
        "tensors_value_checked": checked,
        "all_tensors_finite": finite,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "device": "cpu",
        "training_started": False,
        "forward_pass_tested": False,
    }
    out = Path(__file__).resolve().parents[2] / "pretrained_loading_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(4)
    tyro.cli(verify)
