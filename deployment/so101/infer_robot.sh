#!/usr/bin/env bash
# Start the configured policy and continuous, protected real-robot control.
set -euo pipefail
DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/runtime_env.sh"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash infer_robot.sh'
    echo 'Starts continuous real-robot inference from the current pose. Does not home first.'
    exit 0
fi
if (( $# )); then echo 'Unexpected arguments; use --help.' >&2; exit 2; fi
if [[ -d "$DEPLOY_DIR/.reset_in_progress" ]]; then
    echo 'Reset is in progress; wait for it to finish.' >&2; exit 1
fi
if systemctl --user is-active --quiet so101-deploy-return-home; then
    echo 'Automatic return is in progress; wait for it to finish.' >&2; exit 1
fi
if systemctl --user is-active --quiet so101-deploy-joint-trial; then
    echo 'Real-robot inference is already running.'; exit 0
fi
RESET_GENERATION="${SO101_EXPECTED_GENERATION:-$(cat "$DEPLOY_DIR/INFERENCE_RESET_GENERATION" 2>/dev/null || true)}"
bash "$DEPLOY_DIR/start.sh"
"$ROBOT_PYTHON" - "$DEPLOY_DIR" "$RESET_GENERATION" <<'PY'
import json, sys, time
from pathlib import Path
root = Path(sys.argv[1])
generation_file = root / 'INFERENCE_RESET_GENERATION'
expected_generation = sys.argv[2]
config = json.loads((root / 'config.json').read_text())
for _ in range(570):
    generation = generation_file.read_text().strip() if generation_file.exists() else ''
    if (root / '.reset_in_progress').exists() or generation != expected_generation:
        raise SystemExit('Reset began during startup; inference cancelled.')
    try:
        status = json.loads((root / 'inference_status.json').read_text())
        if (status.get('model_sha256') == config['model_sha256']
                and status.get('rtc') == config.get('rtc', {})
                and status.get('last_published')
                and time.time() - status.get('updated_at', 0) < 1):
            break
    except (OSError, ValueError):
        pass
    time.sleep(1)
else:
    raise SystemExit('No fresh matching policy output; motors were not started.')
(root / 'STOP_MOTION').unlink(missing_ok=True)
PY
exec bash "$DEPLOY_DIR/run_joint_trial.sh" --execute --full-task --duration 0
