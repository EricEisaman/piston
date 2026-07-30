#!/usr/bin/env bash
# Prepare Browser Train natural-language datasets (Karpathy / Hugging Face).
# Delegates to scripts/prepare-natural-data.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required" >&2
  exit 1
fi

VENV="${ROOT}/.venv-data"
if [[ ! -d "$VENV" ]]; then
  echo "==> Creating ${VENV}"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -r "$ROOT/scripts/requirements-data.txt"

DATASETS="${DATASETS:-tinyshakespeare,tinystories,fineweb}"
VOCABS="${VOCABS:-char,512,1024,2048,4096,8192}"

echo "==> Preparing natural data (DATASETS=$DATASETS VOCABS=$VOCABS)"
python "$ROOT/scripts/prepare-natural-data.py" --datasets "$DATASETS" --vocabs "$VOCABS"
