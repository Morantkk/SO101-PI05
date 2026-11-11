#!/usr/bin/env bash
# Clear inference runtime state without moving or releasing the robot.
set -euo pipefail
DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/runtime_env.sh"
exec "$ROBOT_PYTHON" "$DEPLOY_DIR/init_inference.py" "$@"
