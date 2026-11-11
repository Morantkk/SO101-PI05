#!/usr/bin/env bash
# Launch the original LiOS transport stack and the SO101 observation adapter.
set -euo pipefail
DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/runtime_env.sh"
if [[ -d "$DEPLOY_DIR/.reset_in_progress" ]]; then
    echo 'Reset or initialization is in progress; services were not started.' >&2
    exit 1
fi
REPO_DIR="$(cd -- "$DEPLOY_DIR/../.." && pwd)"
PIXI_BIN="${PIXI_BIN:-$HOME/.pixi/bin/pixi}"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
mkdir -p "$DEPLOY_DIR/logs"
CAMERA_SOURCES="$("$ROBOT_PYTHON" -c '
import json, sys
with open(sys.argv[1]) as stream:
    config = json.load(stream)
fps = config["fps"]
print(",".join(f"{name}={path}@{fps}" for name, path in config["cameras"].items()))
' "$DEPLOY_DIR/config.json")"

start_unit() {
    local component="$1" working_directory="$2"
    shift 2
    if systemctl --user is-active --quiet "lios-deploy-$component"; then
        echo "Already running: lios-deploy-$component"
        return
    fi
    systemd-run --user --collect --unit="lios-deploy-$component" \
        --working-directory="$working_directory" \
        --property="StandardOutput=append:$DEPLOY_DIR/logs/$component.log" \
        --property="StandardError=append:$DEPLOY_DIR/logs/$component.log" \
        --setenv=PYTHONUNBUFFERED=1 --setenv=NO_PROXY=127.0.0.1,localhost \
        --setenv=HF_HUB_OFFLINE=1 --setenv=TOKENIZERS_PARALLELISM=false \
        --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=PYTORCH_ALLOC_CONF=expandable_segments:True \
        --setenv=ROOM=so101-fold --setenv=SIGNAL_URL=ws://127.0.0.1:18080/ws \
        --setenv=STUN= --setenv=TURN= --setenv=VIDEO_SOURCE=v4l2 --setenv=CAPTURE_FORMAT=MJPG \
        --setenv=ENCODER=nv --setenv=DECODER=nv \
        --setenv="CAMERAS=$CAMERA_SOURCES" \
        "$@"
}

start_unit signal "$REPO_DIR/lios-webrtc" \
    "$PIXI_BIN" run --as-is ./signal-server/webrtcssvr serve --addr 127.0.0.1:18080
for attempt in {1..30}; do
    if curl -s --max-time 1 http://127.0.0.1:18080/ >/dev/null; then break; fi
    sleep 1
done
start_unit sender "$REPO_DIR/lios-webrtc" \
    "$PIXI_BIN" run --as-is python examples/two_camera_sender.py
start_unit receiver "$REPO_DIR/lios-webrtc" \
    "$PIXI_BIN" run --as-is python examples/two_camera_receiver_inferbuf.py --streams 3 --host 127.0.0.1 --port 5082
start_unit inference "$REPO_DIR/lios-pi" \
    "$PIXI_BIN" run --as-is -e dev python scripts/deployment/serve_so101.py
if [[ ! -d "$DEPLOY_DIR/.reset_in_progress" ]]; then
    start_unit state-client "$DEPLOY_DIR" \
        "$ROBOT_PYTHON" state_client.py
fi
start_unit panel "$DEPLOY_DIR" \
    "$ROBOT_PYTHON" panel.py
echo "Started LiOS deployment. SO101 client only observes; it cannot command motors."
