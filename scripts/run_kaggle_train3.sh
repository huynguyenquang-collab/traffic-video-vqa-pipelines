#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/kaggle_train3.yaml}"
VENV_DIR="${VENV_DIR:-/kaggle/working/tvqa_venv}"
ENV_FILE="${ENV_FILE:-/kaggle/working/tvqa_env.sh}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Run bootstrap first: bash scripts/bootstrap_kaggle.sh" >&2
  exit 1
fi

# shellcheck disable=SC1090
. "$ENV_FILE"

PYTHON_BIN="${PYTHON_BIN:-${TVQA_PYTHON:-$VENV_DIR/bin/python}}"
EXTRA_PYTHONPATH="${TVQA_EXTRA_PYTHONPATH:-}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Kaggle Python env. Run: bash scripts/bootstrap_kaggle.sh" >&2
  exit 1
fi

PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}src" "$PYTHON_BIN" -m traffic_video_vqa.cli -c "$CONFIG" run-prep
PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}src" "$PYTHON_BIN" -m traffic_video_vqa.cli -c "$CONFIG" train
PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}src" "$PYTHON_BIN" -m traffic_video_vqa.cli -c "$CONFIG" infer --pipeline micro_hint_rag
MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_kaggle_train3.sh [all|yolo|vqa|install]

Modes:
  all      Run the converted notebooks in order: YOLO26 first, then VQA.
  yolo     Run only the converted yolo26.ipynb flow.
  vqa      Run only the converted VQA notebook flow.
  install  Install requirements and run the Kaggle notebook dependency helper.

Environment:
  PYTHON_BIN  Python executable to use. Defaults to python3.
EOF
}

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$MODE" == *.yaml || "$MODE" == *.yml ]]; then
  echo "Config files are no longer used by src/refactor; running mode 'all' instead."
  MODE="all"
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

run_yolo() {
  "$PYTHON_BIN" -m refactor.run_yolo26
}

run_vqa() {
  "$PYTHON_BIN" -m refactor.run_vqa
}

case "$MODE" in
  all)
    run_yolo
    run_vqa
    ;;
  yolo)
    run_yolo
    ;;
  vqa)
    run_vqa
    ;;
  install)
    "$PYTHON_BIN" -m pip install -r requirements.txt
    "$PYTHON_BIN" -m refactor.kaggle_environment
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
