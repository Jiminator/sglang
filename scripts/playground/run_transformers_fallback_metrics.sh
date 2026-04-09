#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-$(pwd)}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/playground/run_transformers_fallback_metrics.sh \
    --commit-label COMMIT_LABEL \
    --output-csv OUTPUT_CSV \
    [--trials 20] \
    [--class torchao] \
    [--mode pair|split] \
    [--git-ref GIT_REF]

Modes:
  pair  - run gsm8k then mmlu in one Python process per trial
  split - run gsm8k and mmlu as separate Python processes per trial
EOF
}

COMMIT_LABEL=""
OUTPUT_CSV=""
TRIALS=20
CLASS_NAME="torchao"
MODE="pair"
GIT_REF=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit-label)
      COMMIT_LABEL="$2"
      shift 2
      ;;
    --output-csv)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --trials)
      TRIALS="$2"
      shift 2
      ;;
    --class)
      CLASS_NAME="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --git-ref)
      GIT_REF="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$COMMIT_LABEL" || -z "$OUTPUT_CSV" ]]; then
  echo "--commit-label and --output-csv are required." >&2
  usage >&2
  exit 1
fi

if [[ "$MODE" != "pair" && "$MODE" != "split" ]]; then
  echo "--mode must be 'pair' or 'split'." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"
rm -f "$OUTPUT_CSV"

for TRIAL_ID in $(seq 1 "$TRIALS"); do
  export SGLANG_TF_FALLBACK_METRICS_CSV="$OUTPUT_CSV"
  export SGLANG_TF_FALLBACK_TRIAL_ID="$TRIAL_ID"
  export SGLANG_TF_FALLBACK_COMMIT_LABEL="$COMMIT_LABEL"
  export SGLANG_TF_FALLBACK_GIT_REF="$GIT_REF"

  if [[ "$MODE" == "pair" ]]; then
    python3 "$SCRIPT_DIR/run_transformers_fallback_selected_tests.py" \
      --class "$CLASS_NAME" \
      --tests gsm8k mmlu
  else
    python3 "$SCRIPT_DIR/run_transformers_fallback_selected_tests.py" \
      --class "$CLASS_NAME" \
      --tests gsm8k
    python3 "$SCRIPT_DIR/run_transformers_fallback_selected_tests.py" \
      --class "$CLASS_NAME" \
      --tests mmlu
  fi
done
