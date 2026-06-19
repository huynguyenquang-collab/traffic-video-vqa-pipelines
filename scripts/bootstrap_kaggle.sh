#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-/kaggle/working/tvqa_venv}"
CONSTRAINTS="${CONSTRAINTS:-$ROOT_DIR/constraints/kaggle-cu124.txt}"

python3 -m venv "$VENV_DIR" --system-site-packages
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

python -m pip install -q --upgrade pip setuptools wheel
python -m pip install -q --no-cache-dir -c "$CONSTRAINTS" -r requirements.txt

python scripts/doctor_kaggle.py

echo
echo "Kaggle environment is ready."
echo "Use: source $VENV_DIR/bin/activate"
