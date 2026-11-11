"""Verify or restore the losslessly split, original local100000 checkpoint."""
import json
import shutil
import hashlib
import tarfile
import argparse
import tempfile
import threading
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def process(weights_dir, destination=None):
    manifest = json.loads((ROOT / 'models/local100000/manifest.json').read_text())
    parts = manifest['parts']
    for part in parts:
        path = weights_dir / part['name']
        if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
            raise ValueError('Part checksum mismatch: ' + part['name'])
        print('Verified part: ' + part['name'], flush=True)
    members = {item['archive_path']: item for item in manifest['files']}
    temporary = None
    if destination is not None:
        if destination.exists():
            raise FileExistsError('Refusing to overwrite checkpoint: ' + str(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        needed = sum(item['bytes'] for item in members.values()) + 1024**3
        if shutil.disk_usage(destination.parent).free < needed:
            raise RuntimeError('Insufficient free disk space: allow at least 17 GiB for extraction.')
        temporary = tempfile.TemporaryDirectory(prefix='.restore-model-', dir=destination.parent)
    errors = []
    proc = subprocess.Popen(['zstd', '-dc'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    def feed():
        try:
            for part in parts:
                with (weights_dir / part['name']).open('rb') as stream:
                    shutil.copyfileobj(stream, proc.stdin, length=4*1024*1024)
        except Exception as error:
            errors.append(error)
        finally:
            proc.stdin.close()
    thread = threading.Thread(target=feed, daemon=True)
    thread.start()
    seen = set()
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for item in archive:
                if not item.isfile() or item.name not in members or item.name in seen:
                    raise ValueError('Unexpected archive member: ' + item.name)
                expected = members[item.name]
                if item.size != expected['bytes']:
                    raise ValueError('Unexpected member size: ' + item.name)
                stream = archive.extractfile(item)
                sha = hashlib.sha256()
                output = None
                if temporary:
                    output = (Path(temporary.name) / Path(item.name).name).open('xb')
                try:
                    for block in iter(lambda stream=stream: stream.read(4*1024*1024), b''):
                        sha.update(block)
                        if output:
                            output.write(block)
                finally:
                    if output:
                        output.close()
                    stream.close()
                if sha.hexdigest() != expected['sha256']:
                    raise ValueError('Restored checksum mismatch: ' + item.name)
                seen.add(item.name)
                print('Original checkpoint verified: ' + item.name, flush=True)
        while proc.stdout.read(4*1024*1024):
            pass
        thread.join()
        code = proc.wait()
        if errors or code or seen != set(members):
            raise RuntimeError('Incomplete or invalid archive')
        if temporary:
            Path(temporary.name).rename(destination)
        print('RESTORED ' + str(destination) if destination else 'VERIFIED: all original bytes match; no extraction performed.')
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        thread.join(timeout=5)
        proc.wait()
        if temporary:
            temporary.cleanup()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'restore'])
    parser.add_argument('--weights-dir', type=Path, default=ROOT.parent / 'model_weights')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'models/local100000/manifest.json').read_text())
    destination = ROOT / manifest['restore_path'] if args.action == 'restore' else None
    process(args.weights_dir.expanduser().resolve(), destination)

if __name__ == '__main__':
    main()
