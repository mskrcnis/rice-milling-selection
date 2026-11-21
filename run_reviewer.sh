#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

reviewer_python="${REVIEWER_PYTHON:-.venv/bin/python}"

# Verify the checkout before installing dependencies or executing study code.
if [[ -x "$reviewer_python" ]]; then
  "$reviewer_python" scripts/verify_repository.py
elif [[ -n "${REVIEWER_PYTHON:-}" ]]; then
  printf 'REVIEWER_PYTHON is not an executable file: %s\n' "$reviewer_python" >&2
  exit 1
else
  python3.12 scripts/verify_repository.py
  python3.12 -m venv .venv
  "$reviewer_python" -m pip install -r requirements.lock
fi

PYTHONPATH=study "$reviewer_python" -m pytest -q study/tests \
  --ignore=study/tests/test_arkansas_extraction.py
"$reviewer_python" reproduce.py "$@"
"$reviewer_python" run_diagnostics.py
