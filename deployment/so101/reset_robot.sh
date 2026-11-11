#!/usr/bin/env bash
# Stop execution, then return home or release torque for manual repositioning.
set -euo pipefail
DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/runtime_env.sh"
MODE="${1:---home}"
case "$MODE" in
    --help|-h)
        echo 'Usage: bash reset_robot.sh [--home|--manual|--stop]'
        echo '--home: stop execution, half-open both grippers, hold 0.5s, then restore the recorded home pose.'
        echo '--manual: stop execution and release both arms and grippers; support them first.'
        echo '--stop: stop execution and keep the current position hold.'
        exit 0 ;;
    --home|--manual|--stop) ;;
    *) echo 'Unknown mode; use --help.' >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo 'Unexpected arguments.' >&2; exit 2; fi
if ! mkdir "$DEPLOY_DIR/.reset_in_progress" 2>/dev/null; then
    echo 'Another reset is in progress; refusing concurrent bus access.' >&2
    exit 1
fi
cleanup() {
    touch "$DEPLOY_DIR/STOP_MOTION"
    rmdir "$DEPLOY_DIR/.reset_in_progress"
}
trap cleanup EXIT
interrupt_reset() {
    touch "$DEPLOY_DIR/STOP_MOTION"
    systemctl --user stop so101-deploy-return-home so101-deploy-manual-release 2>/dev/null || true
}
trap 'interrupt_reset; exit 130' INT
trap 'interrupt_reset; exit 143' TERM
touch "$DEPLOY_DIR/STOP_MOTION"
for component in joint-trial return-home state-client; do
    if [[ "$(systemctl --user show "so101-deploy-$component" --property=LoadState --value)" != not-found ]]; then
        systemctl --user stop "so101-deploy-$component"
    fi
done
case "$MODE" in
    --stop) echo 'Execution stopped; current position hold retained.' ;;
    --manual)
        echo 'Releasing both arms AND grippers. Support the arms and remove any held load first.'
        systemd-run --user --wait --pipe --collect --unit=so101-deploy-manual-release \
            --working-directory="$DEPLOY_DIR" \
            "$ROBOT_PYTHON" "$DEPLOY_DIR/manual_release.py" ;;
    --home)
        if [[ ! -f "$DEPLOY_DIR/home_pose.json" ]]; then echo "Configure your own home_pose.json first." >&2; exit 1; fi
        "$ROBOT_PYTHON" -c \
            'from pathlib import Path; import sys; Path(sys.argv[1]).unlink(missing_ok=True)' \
            "$DEPLOY_DIR/STOP_MOTION"
        systemd-run --user --wait --pipe --collect --unit=so101-deploy-return-home \
            --working-directory="$DEPLOY_DIR" \
            "$ROBOT_PYTHON" \
            "$DEPLOY_DIR/return_home.py" \
            --source "$DEPLOY_DIR/home_pose.json" ;;
esac
