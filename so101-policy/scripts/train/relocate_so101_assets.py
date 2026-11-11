"""Copy prepared SO101 assets to a new dataset location without recomputing statistics."""

import json
import shutil
import argparse
import tempfile
from pathlib import Path

from pi.lerobot_v3 import file_sha256, load_episode_split


def relocate_assets(source_assets: Path, trim_manifest: Path, data_root: Path, output: Path):
    source_assets = source_assets.resolve()
    data_root = data_root.resolve(strict=True)
    output = output.absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared assets: {output}")
    split_path = source_assets / "split.json"
    stats_path = source_assets / "norm_stats.json"
    provenance = json.loads((source_assets / "provenance.json").read_text())
    split = json.loads(split_path.read_text())
    manifest = json.loads(trim_manifest.read_text())
    if provenance["split_sha256"] != file_sha256(split_path):
        raise ValueError("Prepared split checksum mismatch")
    if provenance["norm_stats_sha256"] != file_sha256(stats_path):
        raise ValueError("Prepared statistics checksum mismatch")
    if split["frame_ranges_sha256"] != file_sha256(trim_manifest):
        raise ValueError("Original trimming manifest checksum mismatch")
    if manifest["dataset_info_sha256"] != split["dataset_info_sha256"]:
        raise ValueError("Trimming manifest and split metadata mismatch")

    original_hashes = {
        "split_sha256": file_sha256(split_path),
        "frame_ranges_sha256": file_sha256(trim_manifest),
        "provenance_sha256": file_sha256(source_assets / "provenance.json"),
        "data_root": split["data_root"],
    }
    split["data_root"] = manifest["data_root"] = provenance["data_root"] = str(data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".so101-relocate-", dir=output.parent) as temporary:
        staged = Path(temporary) / "assets"
        staged.mkdir()
        ranges_path = staged / "frame_ranges.json"
        ranges_path.write_text(json.dumps(manifest, indent=2) + "\n")
        split["frame_ranges_sha256"] = file_sha256(ranges_path)
        relocated_split = staged / "split.json"
        relocated_split.write_text(json.dumps(split, indent=2) + "\n")
        # Verify every source file at the new location before publishing the copy.
        load_episode_split(data_root, relocated_split, ranges_path)
        shutil.copyfile(stats_path, staged / "norm_stats.json")
        provenance["split_sha256"] = file_sha256(relocated_split)
        provenance["relocated_from"] = original_hashes
        (staged / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        staged.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-assets", type=Path, required=True)
    parser.add_argument("--trim-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(relocate_assets(args.source_assets, args.trim_manifest, args.data_root, args.output))


if __name__ == "__main__":
    main()
