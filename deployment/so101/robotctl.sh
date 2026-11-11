#!/usr/bin/env bash
# Unified SO101 control entry point using the installed robot environment.
set -euo pipefail
DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/runtime_env.sh"
exec "$ROBOT_PYTHON" "$DEPLOY_DIR/robotctl.py" "$@"
