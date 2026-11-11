"""Supervise a local LiOS run and serve its read-only progress panel on localhost."""

import os
import re
import sys
import json
import time
import argparse
import threading
import subprocess
import dataclasses
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import torch

from pi.training.instance_config import get_config


def read_records(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def log_tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 24000))
        return stream.read().decode("utf-8", errors="replace").replace("\r", "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_so101_fold")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--save-step-interval", type=int, default=20000)
    parser.add_argument("--frozen", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fused-adam", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--attention-backend", choices=["eager", "sdpa"])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--data-assets", type=Path)
    parser.add_argument("--weight-dir", type=Path)
    parser.add_argument("--checkpoint-base-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_epochs <= 0:
        parser.error("Batch size and number of epochs must be positive")
    if args.save_step_interval is not None and args.save_step_interval <= 0:
        parser.error("Save step interval must be positive")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.exp_name):
        parser.error("Use letters, digits, underscores or hyphens for the experiment name")
    project = Path(__file__).resolve().parents[2]
    base_config = get_config(args.config_name)
    data = base_config.data
    if args.data_root:
        data = dataclasses.replace(data, repo_id=str(args.data_root.resolve(strict=True)))
    if args.data_assets:
        data_assets = args.data_assets.resolve(strict=True)
        if data_assets.parent.name != base_config.name:
            parser.error("Data assets must be inside a directory named after the selected configuration")
        data = dataclasses.replace(
            data,
            asset_id=data_assets.name,
            frame_ranges_manifest=str(data_assets / "frame_ranges.json"),
            split_manifest=str(data_assets / "split.json"),
        )
    config = dataclasses.replace(
        base_config,
        data=data,
        frozen=base_config.frozen if args.frozen is None else args.frozen,
        gradient_checkpointing=base_config.gradient_checkpointing
        if args.gradient_checkpointing is None
        else args.gradient_checkpointing,
        compile_model=base_config.compile_model if args.compile_model is None else args.compile_model,
        fused_adam=base_config.fused_adam if args.fused_adam is None else args.fused_adam,
        attention_backend=args.attention_backend or base_config.attention_backend,
        assets_base_dir=str(data_assets.parent.parent) if args.data_assets else base_config.assets_base_dir,
        pytorch_weight_path=str(args.weight_dir.resolve(strict=True))
        if args.weight_dir
        else base_config.pytorch_weight_path,
        checkpoint_base_dir=str(args.checkpoint_base_dir.resolve())
        if args.checkpoint_base_dir
        else base_config.checkpoint_base_dir,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_step_interval=args.save_step_interval or base_config.save_step_interval,
        resume=args.resume,
        overwrite=not args.resume,
    )
    provenance = json.loads((config.assets_dirs / config.data.asset_id / "provenance.json").read_text())
    steps_per_epoch = provenance["train_frames"] // config.batch_size
    if not steps_per_epoch:
        parser.error("Batch size exceeds the number of training frames")
    initial_step = 0
    metrics_offset = 0
    if config.checkpoint_dir.exists() and not args.resume:
        raise FileExistsError(config.checkpoint_dir)
    run_dir = project.parent / "training_runs" / args.exp_name
    if args.resume:
        checkpoints = [p for p in config.checkpoint_dir.glob("step*") if p.is_dir() and p.name[4:].isdigit()]
        if not checkpoints:
            raise FileNotFoundError("No step checkpoint available for resume")
        latest = max(checkpoints, key=lambda p: int(p.name[4:]))
        for path in (latest / ".metadata", latest / "ema/.metadata", config.checkpoint_dir / "last_optim/.metadata"):
            if not path.is_file():
                raise FileNotFoundError(path)
        initial_step = torch.load(latest / "train_state.pt", map_location="cpu", weights_only=True)["completed_steps"]
        metrics_offset = len(read_records(config.checkpoint_dir / "train_metrics.jsonl"))
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / f"resume_{time.time_ns()}.log"
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        log_path = run_dir / "train.log"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=1",
        "scripts/train/train_pytorch_fsdp.py",
        config.name,
        "--exp-name",
        args.exp_name,
        "--batch-size",
        str(config.batch_size),
        "--num-epochs",
        str(config.num_epochs),
        "--save-step-interval",
        str(config.save_step_interval),
    ]
    command.extend(
        [
            "--frozen" if config.frozen else "--no-frozen",
            "--gradient-checkpointing" if config.gradient_checkpointing else "--no-gradient-checkpointing",
            "--compile-model" if config.compile_model else "--no-compile-model",
            "--fused-adam" if config.fused_adam else "--no-fused-adam",
            "--attention-backend",
            config.attention_backend,
            "--data.repo-id",
            config.data.repo_id,
            "--data.asset-id",
            config.data.asset_id,
            "--data.frame-ranges-manifest",
            config.data.frame_ranges_manifest,
            "--data.split-manifest",
            config.data.split_manifest,
            "--assets-base-dir",
            config.assets_base_dir,
            "--pytorch-weight-path",
            config.pytorch_weight_path,
            "--checkpoint-base-dir",
            config.checkpoint_base_dir,
        ]
    )
    if args.resume:
        command.extend(["--resume", "--no-overwrite"])
    started = time.time()
    state = {
        "experiment": args.exp_name,
        "status": "starting",
        "started_at": started,
        "batch_size": config.batch_size,
        "epochs": config.num_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": config.num_epochs * steps_per_epoch,
        "frozen": config.frozen,
        "checkpoint_dir": str(config.checkpoint_dir),
        "log_path": str(log_path),
        "wandb_mode": os.environ.get("WANDB_MODE", "online"),
        "command": command,
        "config": dataclasses.asdict(config),
        "initial_step": initial_step,
        "resumed": args.resume,
    }
    state_path = run_dir / "run.json"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/api/status":
                records = read_records(config.checkpoint_dir / "train_metrics.jsonl")[metrics_offset:]
                tail = log_tail(log_path)
                result = {**state, "elapsed_s": time.time() - started, "metrics": records, "log": tail[-12000:]}
                result["validation_metrics"] = read_records(config.checkpoint_dir / "validation_metrics.jsonl")
                matches = re.findall(
                    r"FSDP-Train:.*?(\d+)/" + str(state["total_steps"]) + r".*?loss=([\d.eE+-]+)", tail
                )
                result["latest_step"] = (
                    int(matches[-1][0]) if matches else records[-1]["step"] if records else initial_step
                )
                try:
                    gpu = (
                        subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=3,
                        )
                        .strip()
                        .split(", ")
                    )
                    result["gpu"] = dict(
                        zip(["name", "used_mib", "total_mib", "utilization", "temperature"], gpu, strict=True)
                    )
                except (OSError, subprocess.SubprocessError):
                    result["gpu"] = None
                body = json.dumps(result).encode()
                content_type = "application/json"
            elif self.path == "/":
                body = Path(__file__).with_name("training_panel.html").read_bytes()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if args.resume and state_path.exists():
        state_path.replace(run_dir / f"run_before_resume_{time.time_ns()}.json")
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=project, stdout=log, stderr=subprocess.STDOUT)
    state.update(status="running", pid=process.pid)
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    def watch():
        code = process.wait()
        state.update(status="completed" if code == 0 else "failed", exit_code=code, finished_at=time.time())
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    threading.Thread(target=watch, daemon=True).start()
    print(f"Training panel: http://127.0.0.1:{args.port} | log: {log_path}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if process.poll() is None:
            process.terminate()
            process.wait()


if __name__ == "__main__":
    main()
