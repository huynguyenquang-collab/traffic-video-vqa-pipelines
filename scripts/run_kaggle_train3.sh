#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/kaggle_train3.yaml}"
VENV_DIR="${VENV_DIR:-/kaggle/working/tvqa_venv}"
ENV_FILE="${ENV_FILE:-/kaggle/working/tvqa_env.sh}"

if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  . "$ENV_FILE"
fi

PYTHON_BIN="${PYTHON_BIN:-${TVQA_PYTHON:-$VENV_DIR/bin/python}}"
EXTRA_PYTHONPATH="${TVQA_EXTRA_PYTHONPATH:-}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Kaggle Python env. Run: bash scripts/bootstrap_kaggle.sh" >&2
  exit 1
fi

PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}src" "$PYTHON_BIN" -m traffic_video_vqa.cli -c "$CONFIG" run-prep
PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}src" "$PYTHON_BIN" -m traffic_video_vqa.cli -c "$CONFIG" train
PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}src" "$PYTHON_BIN" -m traffic_video_vqa.cli -c "$CONFIG" infer --pipeline micro_hint_rag
