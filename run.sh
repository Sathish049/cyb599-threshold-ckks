#!/usr/bin/env bash
# Reproduce everything from a clean checkout:
#   ./run.sh            full evaluation (~10-15 min, needs ~4 GB free RAM)
#   ./run.sh --quick    smoke run: tests + P1/P2, one trial (~1-2 min)
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-}"
VENV=.venv

# 1. Find a Python that OpenFHE ships wheels for (3.12 preferred, 3.10 also works).
if [[ ! -x "$VENV/bin/python" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then PY=python3.12
  elif command -v python3.10 >/dev/null 2>&1; then PY=python3.10
  elif command -v uv >/dev/null 2>&1; then uv venv --python 3.12 "$VENV"; PY=""
  else
    echo "ERROR: need Python 3.12 (or 3.10) on Linux x86_64, or 'uv'. See README." >&2; exit 1
  fi
  if [[ -n "${PY}" ]]; then "$PY" -m venv "$VENV"; fi
fi
PYTHON="$VENV/bin/python"
"$PYTHON" -m ensurepip -q >/dev/null 2>&1 || true
echo "==> Installing dependencies into $VENV"
"$PYTHON" -m pip install -q --upgrade pip
"$PYTHON" -m pip install -q -r requirements.txt

# 2. Correctness tests
echo "==> Running correctness tests"
"$PYTHON" -m pytest -q

# 3. Experiments + figures
echo "==> Running evaluation ${MODE}"
if [[ "$MODE" == "--quick" ]]; then
  "$PYTHON" experiments/run_eval.py --quick
  "$PYTHON" experiments/make_figures.py results/results_quick.json
else
  "$PYTHON" experiments/run_eval.py
  "$PYTHON" experiments/make_figures.py results/results.json
fi
echo "==> Done. See results/summary.md and results/figures/"
