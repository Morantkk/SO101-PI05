#!/usr/bin/env bash
# Give one bounded trial exclusive access to both motor buses, then resume observation.
set -euo pipefail
DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/runtime_env.sh"
if [[ -d "$DEPLOY_DIR/.reset_in_progress" ]]; then
    echo "Reset is in progress." >&2
    exit 1
fi
if systemctl --user is-active --quiet lios-deploy-joint-trial; then
    echo "A controller trial is already running." >&2
    exit 1
fi
if [[ "$(systemctl --user show lios-deploy-state-client --property=LoadState --value)" != not-found ]]; then
    systemctl --user stop lios-deploy-state-client
fi
stop_trial() {
    touch "$DEPLOY_DIR/STOP_MOTION"
    systemctl --user stop lios-deploy-joint-trial || true
}
trap 'stop_trial; exit 130' INT
trap 'stop_trial; exit 143' TERM
trap 'if [[ ! -f "$DEPLOY_DIR/STOP_MOTION" && ! -d "$DEPLOY_DIR/.reset_in_progress" ]] && systemctl --user is-active --quiet lios-deploy-inference; then bash "$DEPLOY_DIR/start.sh"; fi' EXIT
systemd-run --user --wait --pipe --collect --unit=lios-deploy-joint-trial \
    --working-directory="$DEPLOY_DIR" \
    "$ROBOT_PYTHON" "$DEPLOY_DIR/joint_trial.py" "$@"
