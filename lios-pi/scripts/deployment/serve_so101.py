"""SO101 adaptation of inference.ipynb using the original LiOS IPC and JSON APIs."""

import sys
import json
import time
import queue
import signal
import logging
import argparse
import threading
import dataclasses
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from normalizer import MetadataNormalizingPlanner, normalize_state_dict_key
from torch.distributed.checkpoint import FileSystemReader, load

from pi.data import _make_bool_mask, _normalize_array, _tokenize_prompt
from pi.lerobot_v3 import LeRobotV3Dataset, file_sha256, load_episode_split
from pi.models.model import Observation
from pi.models.tokenizer import PaligemmaTokenizer
from pi.models_pytorch.rtc import encode_previous, sample_actions_rtc
from pi.shared.image_tools import resize_with_pad
from pi.inference_buffer_v2 import InferenceBufferV2
from pi.services.ws_json_api import JSONWebSocketAPIServer
from pi.training.instance_config import get_config
from pi.models_pytorch.pi0_pytorch import PI0Pytorch

ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT = ROOT / "deployment/so101"
sys.path.insert(0, str(DEPLOYMENT))
from rtc_protocol import RTCPlanner

IMAGE_MAP = {"mid": "base_0_rgb", "left": "left_wrist_0_rgb", "right": "right_wrist_0_rgb"}


def decode_actions(predictions, original_state, stats, mask):
    """Undo training quantile normalization and delta targets, preserving grippers."""
    result = np.asarray(predictions, dtype=np.float32)[..., :12].copy()
    result = (result + 1) / 2 * (stats.q99 - stats.q01 + 1e-6) + stats.q01
    result += np.where(mask, original_state, 0)[None, :]
    if result.shape != (30, 12) or not np.isfinite(result).all():
        raise ValueError("Expected finite 30x12 absolute SO101 actions")
    return result


def save_json(name, value):
    target = DEPLOYMENT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(target)


def select_offline_episode(train_ids, test_ids):
    """Prefer validation data; label a training sample explicitly when none exists."""
    if test_ids:
        return test_ids[0], "validation"
    if train_ids:
        return train_ids[0], "train"
    raise ValueError("No episodes available for offline inference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Predict one dataset sample and exit without devices")
    parser.add_argument("--no-compile", action="store_true", help="Use eager inference for diagnosis")
    parser.add_argument("--config", type=Path, default=DEPLOYMENT / "config.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    deploy = json.loads(args.config.read_text())
    rtc = deploy.get("rtc", {})
    rtc_enabled = rtc.get("enabled", False)
    planner = RTCPlanner(deploy) if rtc_enabled else None
    cfg = get_config(deploy["policy"])
    data_overrides = {k: deploy[k] for k in ("asset_id", "frame_ranges_manifest", "split_manifest") if k in deploy}
    if "dataset_root" in deploy:
        data_overrides["repo_id"] = deploy["dataset_root"]
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, **data_overrides),
                              assets_base_dir=deploy.get("assets_base_dir", cfg.assets_base_dir))
    checkpoint = Path(deploy["checkpoint"]).expanduser().resolve()
    if cfg.model.action_horizon != deploy["horizon"]:
        raise ValueError("Deployment action horizon differs from the training preset")
    if file_sha256(checkpoint / "__0_0.distcp") != deploy["model_sha256"]:
        raise ValueError("Checkpoint shard checksum differs from deployment identity")
    if file_sha256(checkpoint / ".metadata") != deploy["metadata_sha256"]:
        raise ValueError("Checkpoint metadata checksum differs from deployment")
    stats_hash = file_sha256(cfg.assets_dirs / cfg.data.asset_id / "norm_stats.json")
    if stats_hash != deploy["norm_stats_sha256"]:
        raise ValueError("Normalization statistics differ from deployment")
    stats = cfg.data.load_norm_stats(cfg.assets_dirs, cfg.model)
    if stats is None:
        raise ValueError("Training normalization statistics are required")
    train_ids, test_ids = load_episode_split(cfg.data.repo_id, cfg.data.split_manifest, cfg.data.frame_ranges_manifest)
    for key, ids in (("train_episodes", train_ids), ("validation_episodes", test_ids)):
        if key in deploy and len(ids) != deploy[key]:
            raise ValueError(f"Deployment {key} differs from the prepared dataset")
    if deploy.get("train_episode_range") is not None:
        first, last = deploy["train_episode_range"]
        if sorted(train_ids) != list(range(first, last + 1)):
            raise ValueError("Deployment training episode range differs from the prepared dataset")
    mask = np.asarray(_make_bool_mask(*cfg.data.delta_action_dims))
    tokenizer = PaligemmaTokenizer(cfg.model.max_token_len)
    status = {
        "status": "loading",
        "checkpoint": str(checkpoint),
        "num_steps": rtc["num_steps"] if rtc_enabled else deploy["num_steps"],
        "rtc": rtc,
        "inference_backend": "rtc" if rtc_enabled else "ordinary",
        "policy": cfg.name,
        "execution_enabled": False,
        "predictions": 0,
        "published": 0,
        "checkpoint_step": deploy["checkpoint_step"],
        "model_sha256": deploy["model_sha256"],
        "norm_stats_sha256": stats_hash,
        "runtime_executable": sys.executable,
        "runtime_environment": "lios-pi Pixi dev",
        "updated_at": time.time(),
        "compile_model": not args.no_compile and deploy.get("compile_model", True),
    }
    if not args.offline:
        save_json("inference_status.json", status)
    device = "cuda:0"
    with torch.device(device):
        model = PI0Pytorch(cfg.model)
    metadata = FileSystemReader(checkpoint).read_metadata()
    stored = {normalize_state_dict_key(k): v for k, v in metadata.state_dict_metadata.items()}
    live = model.state_dict()
    if set(stored) != set(live):
        raise ValueError(f"Checkpoint key mismatch: missing={set(live) - set(stored)}, extra={set(stored) - set(live)}")
    for name, tensor in live.items():
        if tuple(stored[name].size) != tuple(tensor.shape):
            raise ValueError(f"Checkpoint shape mismatch: {name}")
    load(live, checkpoint_id=checkpoint, planner=MetadataNormalizingPlanner())
    for name, tensor in live.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
    model.eval()
    model.requires_grad_(False)
    status["loaded_tensors"] = len(live)
    logging.info("Loaded %s matching, finite model tensors from %s", len(live), checkpoint)
    sample = model.sample_actions
    denoise = model.denoise_step
    if status["compile_model"]:
        if rtc_enabled:
            denoise = torch.compile(denoise, mode=deploy.get("compile_mode", "default"))
        else:
            sample = torch.compile(sample, mode=deploy.get("compile_mode", "default"))

    def predict(images, state, previous=None, delay=0):
        normalized = _normalize_array(state, stats["state"], True)
        tokenized = _tokenize_prompt(
            {"state": normalized, "prompt": deploy["prompt"]}, tokenizer, discrete_state_input=True
        )
        data = {
            "image": {IMAGE_MAP[k]: v.unsqueeze(0) for k, v in images.items()},
            "image_mask": {name: torch.ones(1, dtype=torch.bool, device=device) for name in IMAGE_MAP.values()},
            "state": torch.tensor(np.pad(normalized, (0, 20)), dtype=torch.float32, device=device)[None, :],
            "tokenized_prompt": torch.as_tensor(tokenized["tokenized_prompt"], device=device)[None, :],
            "tokenized_prompt_mask": torch.as_tensor(tokenized["tokenized_prompt_mask"], device=device)[None, :],
        }
        if rtc_enabled:
            observation = Observation.from_dict(data)
            noise = model.sample_noise((1, 30, 32), device)
            encoded = None if previous is None else torch.as_tensor(
                encode_previous(previous, state, stats["actions"], mask, 30, 32), device=device
            )[None]
            output = sample_actions_rtc(
                model, device, observation, previous=encoded, delay=delay,
                overlap=0 if previous is None else len(previous), num_steps=rtc["num_steps"],
                beta=rtc["max_guidance_weight"], noise=noise, denoise=denoise,
            )
        else:
            with torch.inference_mode():
                output = sample(device=device, observation=Observation.from_dict(data),
                                noise=model.sample_noise((1, 30, 32), device), num_steps=deploy["num_steps"])
        prediction = output[0].detach().float().cpu().numpy()
        return decode_actions(prediction, state, stats["actions"], mask)

    if args.offline:
        episode, sample_split = select_offline_episode(train_ids, test_ids)
        dataset = LeRobotV3Dataset(
            cfg.data.repo_id, episodes=[episode], frame_ranges_manifest=cfg.data.frame_ranges_manifest
        )
        raw = dataset[len(dataset) // 2]
        names = {"mid": "front", "left": "left_wrist", "right": "right_wrist"}
        images = {
            k: torch.from_numpy(resize_with_pad(raw[f"observation.images.{v}"], 224, 224)).to(device)
            for k, v in names.items()
        }
        timings = []
        previous = None
        for _ in range(6 if rtc_enabled else 4):
            started = time.perf_counter()
            actions = predict(images, raw["observation.state"], previous, rtc.get("initial_delay_steps", 0))
            if rtc_enabled:
                previous = actions[rtc["execution_horizon"]:].copy()
            timings.append(time.perf_counter() - started)
        save_json(
            str(Path(deploy.get("report_dir", ".")) / "offline_prediction.json"),
            {
                "checkpoint": str(checkpoint),
                "episode": int(raw["episode_index"]),
                "sample_split": sample_split,
                "rtc": rtc,
                "compile_model": status["compile_model"],
                "frame": int(raw["frame_index"]),
                "state": raw["observation.state"].tolist(),
                "actions": actions.tolist(),
                "shape": list(actions.shape),
                "finite": True,
                "elapsed_s": timings[-1],
                "inference_seconds": timings,
                "loaded_tensors": len(live),
                "executed": False,
                "motor_commands_sent": 0,
                "runtime_executable": sys.executable,
                "model_sha256": deploy["model_sha256"],
                "norm_stats_sha256": stats_hash,
            },
        )
        dataset.close()
        logging.info("Offline inference passed: %s", actions.shape)
        return

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    api = JSONWebSocketAPIServer(host="127.0.0.1", port=18081, max_queue_size=1)
    api.start()
    buffer = None
    previous_stamp = None
    rtc_warmed = False
    status["status"] = "waiting_for_observations"
    save_json("inference_status.json", status)
    try:
        while not stop.is_set():
            try:
                if buffer is None:
                    with urlopen("http://127.0.0.1:5082/api/v1/infer-buffer/base64", timeout=3) as response:
                        candidate = InferenceBufferV2.from_base64(response.read().decode("ascii"))
                    if set(IMAGE_MAP) != set(candidate.images):
                        candidate.close_semaphore()
                        raise ValueError(f"Waiting for all three camera streams: {list(candidate.images)}")
                    buffer = candidate
                message = api.recv_state(timeout=2)
                if message.get("state_names") != deploy["state_names"]:
                    raise ValueError("Robot joint order differs from deployment")
                ticket = planner.plan(message) if rtc_enabled else None
                if rtc_enabled and ticket is None:
                    stop.wait(0.002)
                    continue
                if message["timestamp/ms"] == previous_stamp:
                    stop.wait(0.005)
                    continue
                if message.get("joint_units") != "degrees" or message.get("gripper_units") != "0_to_100":
                    raise ValueError("SO101 state must specify degrees and gripper 0_to_100")
                state = np.asarray(
                    message["left_arm"]["positions"] + message["right_arm"]["positions"], dtype=np.float32
                )
                if state.shape != (12,) or not np.isfinite(state).all():
                    raise ValueError("Expected finite 12D SO101 state")
                state_age = time.time() - message["timestamp/ms"] / 1000
                if not 0 <= state_age <= 0.2:
                    raise ValueError(f"Stale state: {state_age:.3f}s")
                with buffer.hold_lock(timeout=2):
                    images = {k: buffer.images[k].clone() for k in IMAGE_MAP}
                    frame_info = {k: buffer.states[k].cpu().tolist() for k in IMAGE_MAP}
                    torch.cuda.synchronize()
                if any(image.dtype != torch.uint8 or tuple(image.shape) != (224, 224, 3) for image in images.values()):
                    raise ValueError("LiOS inference expects resized RGB uint8 224x224x3 frames")
                frame_ages = {k: time.time() - v[0] / 1e9 for k, v in frame_info.items()}
                if any(not 0 <= age <= 0.5 for age in frame_ages.values()):
                    raise ValueError(f"Stale received camera frame: {frame_ages}")
            except Exception as error:
                status.update(status="waiting_for_observations", updated_at=time.time(), detail=str(error))
                save_json("inference_status.json", status)
                stop.wait(1)
                continue
            status.update(status="warming_up" if not status["predictions"] else "predicting", updated_at=time.time())
            save_json("inference_status.json", status)
            started = time.perf_counter()
            actions = predict(images, state, ticket["previous"], ticket["delay_steps"]) if rtc_enabled else predict(images, state)
            if rtc_enabled and not rtc_warmed:
                # Compile both no-gradient bootstrap and full VJP before queue execution.
                predict(images, state, actions[rtc["execution_horizon"]:], rtc["initial_delay_steps"])
                rtc_warmed = True
            elapsed = time.perf_counter() - started
            previous_stamp = message["timestamp/ms"]
            status["predictions"] += 1
            action_message = {
                "control_mode": "joint",
                "timestamp/ms": message["timestamp/ms"],
                "delta/ms": 1000 / deploy["fps"],
                "left_arm": {"actions": actions[:, :6].tolist()},
                "right_arm": {"actions": actions[:, 6:].tolist()},
                "send_time": time.time(),
                "joint_units": "degrees",
                "gripper_units": "0_to_100",
                "policy": cfg.name,
                "checkpoint_step": deploy["checkpoint_step"],
                "model_sha256": deploy["model_sha256"],
                "inference_seconds": elapsed,
                "execution_enabled": False,
                "observation_only": True,
            }
            # Compilation and warmup may outlive the action horizon; do not publish those chunks.
            age = time.time() - message["timestamp/ms"] / 1000
            fresh = 0 <= age < 0.35
            if fresh and rtc_enabled:
                try:
                    latest_state = api.recv_state(timeout=0)
                except (TimeoutError, queue.Empty):
                    fresh = False
                else:
                    fresh = latest_state.get("rtc", {}).get("session") == ticket["session"]
            if fresh:
                if rtc_enabled:
                    action_message["rtc"] = planner.commit(ticket, actions)
                api.send_action(action_message)
                status["published"] += 1
            status.update(
                status="ready",
                updated_at=time.time(),
                last_inference_s=elapsed,
                observation_age_s=age,
                frame_ages_s=frame_ages,
                frame_counters={k: v[1] for k, v in frame_info.items()},
                image_shapes={k: list(v.shape) for k, v in images.items()},
                last_published=fresh,
                detail="Observation-only deployment; no motor commands",
            )
            save_json("inference_status.json", status)
            save_json("latest_prediction.json", action_message)
            if status["predictions"] <= 3 or status["predictions"] % 100 == 0:
                logging.info("Prediction %s: %.3fs, published=%s", status["predictions"], elapsed, fresh)
    finally:
        api.stop()
        if buffer is not None:
            buffer.close_semaphore()
        status.update(status="stopped", updated_at=time.time())
        save_json("inference_status.json", status)


if __name__ == "__main__":
    main()
