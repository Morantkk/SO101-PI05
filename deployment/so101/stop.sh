#!/usr/bin/env bash
# Stop only the services owned by this deployment, consumers before producers.
set -euo pipefail
DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/runtime_env.sh"
touch "$DEPLOY_DIR/STOP_MOTION"
# Stop inference first so a finishing controller does not relaunch observation.
for component in inference joint-trial panel state-client sender receiver signal; do
    if systemctl --user is-active --quiet "lios-deploy-$component"; then
        systemctl --user stop "lios-deploy-$component"
    fi
done
