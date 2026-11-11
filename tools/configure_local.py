"""Prepare local paths and verified assets; never connect to robot devices."""
import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / 'deployment/so101'
ASSET_ID = 'fold81_197_h30_trainonly_wristfixed_20261108'
POLICY = 'pi05_so101_fold_81plus'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Original, matching LeRobot dataset (not included).')
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve(strict=True)
    config_path, machine_path = DEPLOY / 'config.json', DEPLOY / 'machine.json'
    if config_path.exists() or machine_path.exists():
        raise FileExistsError('Local config already exists. Back it up and edit it explicitly.')
    source = ROOT / 'lios-pi/assets' / POLICY / ASSET_ID
    output = ROOT / 'local_assets' / POLICY / ASSET_ID
    pixi = os.environ.get('PIXI_BIN', shutil.which('pixi') or str(Path.home() / '.pixi/bin/pixi'))
    subprocess.run([pixi, 'run', '-e', 'dev', 'python', 'scripts/train/relocate_so101_assets.py',
                    '--source-assets', str(source), '--trim-manifest', str(source / 'original_frame_ranges.json'),
                    '--data-root', str(dataset), '--output', str(output)], cwd=ROOT / 'lios-pi', check=True)
    config = json.loads((DEPLOY / 'config.example.json').read_text())
    machine = dict(dataset_root=str(dataset), assets_base_dir=str(ROOT / 'local_assets'), asset_id=ASSET_ID,
                   frame_ranges_manifest=str(output / 'frame_ranges.json'), split_manifest=str(output / 'split.json'))
    config.update(machine)
    config['checkpoint'] = str(ROOT / config['checkpoint'])
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    machine_path.write_text(json.dumps(machine, indent=2) + '\n')
    (DEPLOY / 'STOP_MOTION').touch()
    print('Created config.json and machine.json; source-data hashes verified.')
    print('Fill YOUR_* device/calibration values and your own home_pose.json before any robot run.')
    print('No services or hardware were accessed.')

if __name__ == '__main__':
    main()
