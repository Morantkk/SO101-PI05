#!/usr/bin/env bash
# Run on one H100 with corrected wrist channels. See cloud/so101_h100/README.zh-CN.md.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

: "${DATA_ROOT:?Set DATA_ROOT to the uploaded LeRobot v3 dataset directory}"
PIXI_BIN="${PIXI_BIN:-pixi}"
EXP_NAME="${1:-so101_h100_$(date +%Y%m%d_%H%M%S)}"
WEIGHT_DIR="${WEIGHT_DIR:-$PROJECT_DIR/../weights/pi05_base_pytorch}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$PROJECT_DIR/../checkpoints}"
SOURCE_ASSETS="${SOURCE_ASSETS:-$PROJECT_DIR/assets/pi05_so101_fold/fold100_h30_train90_wrist_fixed_v1}"
TRIM_MANIFEST="${TRIM_MANIFEST:-$SOURCE_ASSETS/original_frame_ranges.json}"
ASSET_ID="$(basename -- "$SOURCE_ASSETS")"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FROZEN="${FROZEN:-false}"
SAVE_STEP_INTERVAL="${SAVE_STEP_INTERVAL:-20000}"
TEST_STEP_INTERVAL="${TEST_STEP_INTERVAL:-10000}"
case "$FROZEN" in
    true) FREEZE_FLAG=--frozen ;;
    false) FREEZE_FLAG=--no-frozen ;;
    *) echo 'FROZEN must be true or false' >&2; exit 1 ;;
esac
RUNTIME_ASSETS_BASE="$CHECKPOINT_BASE_DIR/prepared_assets/$EXP_NAME"
RUNTIME_ASSETS="$RUNTIME_ASSETS_BASE/pi05_so101_fold/$ASSET_ID"

if [[ ! -f "$WEIGHT_DIR/model.safetensors" ]]; then
    echo "Pretrained weights not found: $WEIGHT_DIR/model.safetensors" >&2
    exit 1
fi
if [[ -e "$CHECKPOINT_BASE_DIR/pi05_so101_fold/$EXP_NAME" ]]; then
    echo "Experiment already exists; choose a new experiment name: $EXP_NAME" >&2
    exit 1
fi

"$PIXI_BIN" run -e dev python scripts/train/relocate_so101_assets.py \
    --source-assets "$SOURCE_ASSETS" \
    --trim-manifest "$TRIM_MANIFEST" \
    --data-root "$DATA_ROOT" \
    --output "$RUNTIME_ASSETS"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_MULTIPROCESSING_START_METHOD=spawn
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$PROJECT_DIR/logs"
exec > >(tee -a "$PROJECT_DIR/logs/$EXP_NAME.log") 2>&1

# Set the H100 architecture after Pixi activation overrides the local 4090 setting.
TRAIN_COMMAND=("$PIXI_BIN" run -e dev env TORCH_CUDA_ARCH_LIST=9.0 torchrun \
    --standalone --nnodes=1 --nproc_per_node=1 \
    scripts/train/train_pytorch_fsdp.py pi05_so101_fold \
    --exp-name "$EXP_NAME" \
    --data.repo-id "$DATA_ROOT" \
    --data.asset-id "$ASSET_ID" \
    --data.frame-ranges-manifest "$RUNTIME_ASSETS/frame_ranges.json" \
    --data.split-manifest "$RUNTIME_ASSETS/split.json" \
    --assets-base-dir "$RUNTIME_ASSETS_BASE" \
    --pytorch-weight-path "$WEIGHT_DIR" \
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
    --batch-size "$BATCH_SIZE" --num-workers 8 "$FREEZE_FLAG" \
    --num-epochs "$NUM_EPOCHS" --save-step-interval "$SAVE_STEP_INTERVAL" \
    --test-step-interval "$TEST_STEP_INTERVAL")
if [[ -n "${MAX_STEPS:-}" ]]; then
    TRAIN_COMMAND+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "${MAX_EVAL_BATCHES:-}" ]]; then
    TRAIN_COMMAND+=(--max-eval-batches "$MAX_EVAL_BATCHES")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'Verified relocated data; training command: '
    printf '%q ' "${TRAIN_COMMAND[@]}"
    printf '\n'
    exit 0
fi
exec "${TRAIN_COMMAND[@]}"
