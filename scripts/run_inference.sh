#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_PATH="${INPUT_PATH:-$REPO_ROOT/data/test}"
VH_PORT="${VH_PORT:-18188}"
RUN_TURNS="${RUN_TURNS:-20}"
VLM_NAME="${VLM_NAME:-mock}"
MOCK_PLAN="${MOCK_PLAN:-walk,turn_left,report,finish}"
export VH_ROOT="${VH_ROOT:-$REPO_ROOT/third_party/virtualhome}"

if [[ "$VLM_NAME" == "mock" ]]; then
  OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/output/mock_results}"
  IMAGE_PATH="${IMAGE_PATH:-$REPO_ROOT/output/mock_images}"
else
  OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/output/results}"
  IMAGE_PATH="${IMAGE_PATH:-$REPO_ROOT/output/images}"
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_inference.sh [runner arguments]

Optional environment variables:
  VLM_NAME     Registered VLM name (default: mock)
  MOCK_PLAN    Mock tool sequence (default: walk,turn_left,report,finish)
  VH_ROOT      VirtualHome source root (default: third_party/virtualhome)
  VH_PORT      Unity HTTP port (default: 18188)
  INPUT_PATH   Dataset JSON file or directory (default: data/test)
  OUTPUT_PATH  Result directory (default: output/mock_results for mock;
               output/results otherwise)
  IMAGE_PATH   Observation image directory (default: output/mock_images for
               mock; output/images otherwise)
  RUN_TURNS    Maximum turns per task (default: 20)
  PYTHON_BIN   Python executable (default: python)

Additional arguments are passed to exp/runner.py. For example:
  bash scripts/run_inference.sh
  VLM_NAME=my_vlm bash scripts/run_inference.sh --limit 1 --overwrite

The default mock run processes one task and only verifies the runner pipeline.
It does not produce meaningful benchmark results. Implement a VLM in
exp/vlm.py, register it in build_vlm, and add its name to the --vlm choices in
exp/runner.py before running real inference.
EOF
  exit 0
fi

if [[ ! -d "$VH_ROOT/virtualhome/simulation" ]]; then
  echo "VirtualHome source was not found at $VH_ROOT." >&2
  echo "Clone VirtualHome there or set VH_ROOT to its source root." >&2
  exit 2
fi

cd "$REPO_ROOT"

VLM_ARGS=(--vlm "$VLM_NAME")
if [[ "$VLM_NAME" == "mock" ]]; then
  VLM_ARGS+=(--mock_plan "$MOCK_PLAN" --limit 1)
fi

"$PYTHON_BIN" exp/runner.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --image_path "$IMAGE_PATH" \
  "${VLM_ARGS[@]}" \
  --port "$VH_PORT" \
  --run_turns "$RUN_TURNS" \
  --write_each_step \
  "$@"
