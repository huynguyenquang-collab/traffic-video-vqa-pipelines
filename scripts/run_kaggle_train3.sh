#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/kaggle_train3.yaml}"
VENV_DIR="${VENV_DIR:-/kaggle/working/tvqa_venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Kaggle venv at $VENV_DIR. Run: bash scripts/bootstrap_kaggle.sh" >&2
  exit 1
fi

PYTHONPATH=src "$PYTHON_BIN" -m traffic_video_vqa.cli -c "$CONFIG" run-prep
PYTHONPATH=src "$PYTHON_BIN" -m traffic_video_vqa.cli -c "$CONFIG" train
PYTHONPATH=src "$PYTHON_BIN" -m traffic_video_vqa.cli -c "$CONFIG" infer --pipeline micro_hint_rag
