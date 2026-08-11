#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/test}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/output/results}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$REPO_ROOT/output/eval}"
MAX_EVAL_STEP="${MAX_EVAL_STEP:-20}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_evaluation.sh [evaluation arguments]

Before running, implement the custom LLM judge in eval/eval_utils.py.

Optional environment variables:
  DATA_DIR         Ground-truth data directory (default: data/test)
  RESULT_DIR       Inference result directory (default: output/results)
  EVAL_OUTPUT_DIR  Evaluation output directory (default: output/eval)
  MAX_EVAL_STEP    Maximum evaluated steps per task (default: 20)
  PYTHON_BIN       Python executable (default: python)

Additional arguments are passed to eval/eval.py. For example:
  bash scripts/run_evaluation.sh --limit 10 --overwrite
EOF
  exit 0
fi

cd "$REPO_ROOT"

"$PYTHON_BIN" eval/eval.py \
  --data_dir "$DATA_DIR" \
  --result_dir "$RESULT_DIR" \
  --output_dir "$EVAL_OUTPUT_DIR" \
  --max_eval_step "$MAX_EVAL_STEP" \
  --use_judge \
  --judge_backend custom \
  "$@"
