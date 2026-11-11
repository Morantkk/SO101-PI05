#!/usr/bin/env bash
# Environment selection only; no services or motors are started.
if [[ -f "$DEPLOY_DIR/runtime.env" ]]; then source "$DEPLOY_DIR/runtime.env"; fi
: "${PIXI_BIN:=$HOME/.pixi/bin/pixi}"
: "${ROBOT_PYTHON:=$HOME/anaconda3/envs/lerobot/bin/python}"
export PIXI_BIN ROBOT_PYTHON
