"""Relocated assets retain source integrity and the original data definition."""

import json
import importlib.util
from pathlib import Path

import pytest

from pi.lerobot_v3 import file_sha256, load_episode_split

_SPEC = importlib.util.spec_from_file_location(
    "relocate_so101_assets", Path(__file__).parents[1] / "scripts/train/relocate_so101_assets.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "uploaded_data"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"fps": 30, "total_episodes": 2, "features": {}}))
    info_hash = file_sha256(root / "meta/info.json")
    assets = tmp_path / "original_assets"
    assets.mkdir()
    trim = tmp_path / "manifest.json"
    trim.write_text(
        json.dumps(
            {
                "data_root": "/old/server/data",
                "dataset_info_sha256": info_hash,
                "frame_ranges": {"0": [10, 20], "1": [5, 25]},
            }
        )
    )
    (assets / "split.json").write_text(
        json.dumps(
            {
                "data_root": "/old/server/data",
                "dataset_info_sha256": info_hash,
                "frame_ranges_sha256": file_sha256(trim),
                "source_files": {"meta/info.json": info_hash},
                "train": [0],
                "test": [1],
            }
        )
    )
    (assets / "norm_stats.json").write_text('{"norm_stats": {}}\n')
    (assets / "provenance.json").write_text(
        json.dumps(
            {
                "split_sha256": file_sha256(assets / "split.json"),
                "norm_stats_sha256": file_sha256(assets / "norm_stats.json"),
            }
        )
    )
    return assets, trim, root, tmp_path / "output"


def test_relocation_preserves_inputs_and_statistics(prepared):
    assets, trim, root, output = prepared
    originals = {path: path.read_bytes() for path in [*assets.iterdir(), trim, root / "meta/info.json"]}
    _MODULE.relocate_assets(*prepared)
    assert load_episode_split(root, output / "split.json", output / "frame_ranges.json") == ([0], [1])
    assert (output / "norm_stats.json").read_bytes() == originals[assets / "norm_stats.json"]
    assert json.loads((output / "frame_ranges.json").read_text())["frame_ranges"] == {
        "0": [10, 20],
        "1": [5, 25],
    }
    assert all(path.read_bytes() == content for path, content in originals.items())
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["split_sha256"] == file_sha256(output / "split.json")
    assert provenance["relocated_from"]["frame_ranges_sha256"] == file_sha256(trim)
    with pytest.raises(FileExistsError):
        _MODULE.relocate_assets(*prepared)


@pytest.mark.parametrize("damaged", ["split.json", "norm_stats.json", "manifest.json", "meta/info.json"])
def test_relocation_rejects_changed_inputs(prepared, damaged):
    assets, trim, root, output = prepared
    path = trim if damaged == "manifest.json" else root / damaged if damaged.startswith("meta/") else assets / damaged
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        _MODULE.relocate_assets(*prepared)
    assert not output.exists()
