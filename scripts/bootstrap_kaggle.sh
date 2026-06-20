#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-/kaggle/working/tvqa_venv}"
TARGET_DIR="${TARGET_DIR:-/kaggle/working/tvqa_site}"
CONSTRAINTS="${CONSTRAINTS:-$ROOT_DIR/constraints/kaggle-cu124.txt}"
ENV_FILE="${ENV_FILE:-/kaggle/working/tvqa_env.sh}"

rm -f "$ENV_FILE"

# Some Kaggle images load a sitecustomize module that imports wrapt. If a prior
# pip operation removed wrapt, venv/ensurepip can fail before we get a clean env.
python3 -m pip install -q --no-cache-dir wrapt || true

venv_ok=0
rm -rf "$VENV_DIR"
if python3 -m venv "$VENV_DIR" --system-site-packages; then
  if "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
    venv_ok=1
  fi
fi

if [ "$venv_ok" = "1" ]; then
  # shellcheck disable=SC1091
  . "$VENV_DIR/bin/activate"

  python -m pip install -q --upgrade pip setuptools wheel
  python -m pip install -q --no-cache-dir -c "$CONSTRAINTS" -r requirements.txt

  cat > "$ENV_FILE" <<EOF
export TVQA_PYTHON="$VENV_DIR/bin/python"
export TVQA_EXTRA_PYTHONPATH=""
EOF
else
  echo "venv creation failed; falling back to an isolated PYTHONPATH target at $TARGET_DIR" >&2
  rm -rf "$VENV_DIR"
  rm -rf "$TARGET_DIR"
  mkdir -p "$TARGET_DIR"
  python3 -m pip install -q --no-cache-dir --target "$TARGET_DIR" wrapt
  python3 -m pip install -q --no-cache-dir --target "$TARGET_DIR" -c "$CONSTRAINTS" -r requirements.txt

  cat > "$ENV_FILE" <<EOF
export TVQA_PYTHON="$(command -v python3)"
export TVQA_EXTRA_PYTHONPATH="$TARGET_DIR"
EOF
fi

# shellcheck disable=SC1090
. "$ENV_FILE"
PYTHONPATH="${TVQA_EXTRA_PYTHONPATH:+$TVQA_EXTRA_PYTHONPATH:}src" "$TVQA_PYTHON" scripts/doctor_kaggle.py

echo
echo "Kaggle environment is ready."
echo "Use: source $ENV_FILE"
