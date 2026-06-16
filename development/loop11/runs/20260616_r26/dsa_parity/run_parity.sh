#!/usr/bin/env bash
# DSA-native radix-reuse parity probe (FALLBACK metric: e2e NIAH answer-recall +
# exact greedy-output divergence). Reuses the R24/R25 boot infra (op-point,
# ready_wait/smoke/teardown/gpu_idle_wait) and the proven NIAH heavy-reuse
# construction (L1024 same-seed warmup -> shared ~4288-token prefix).
#
# Op-point: GLM-5.1-FP8 TP=8 fp8_e4m3 page64 mem0.8 max-running64 EAGER
# (--disable-cuda-graph), DSA-NATIVE default (NO --enable-double-sparsity),
# dsa prefill/decode flashmla_kv. NEVER expandable_segments. ONE server at a time.
#
#   fresh : radix-OFF boot (--disable-radix-cache); cached_tokens MUST be 0.
#   reuse : radix-ON boot (default radix); L1024 warmup seeds cached>0 heavy reuse.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../20260616_r25/radix_authorization/r25_env.sh"
cd "$REPO"

OUTDIR="$HERE"; mkdir -p "$OUTDIR/logs"
LOG="$OUTDIR/logs/stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== DSA parity start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
echo "[dsa_parity] expandable_segments=$([[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] && echo UNSET || echo "${PYTORCH_CUDA_ALLOC_CONF}")"

LENGTH=4096
NUM="${1:-144}"
MAXNEW=24
SEED_BASE=1000
WARMUP_LENGTH=1024

run_arm() {  # $1=arm(fresh|reuse)  $2..=extra launch args
  local arm="$1"; shift
  local slog="$OUTDIR/logs/serve_${arm}.log"
  local idx="$OUTDIR/index_${arm}.jsonl"
  teardown
  echo "=== boot DSA-native (arm=$arm) eager $(date -u +%H:%M:%SZ) ==="
  local args=("${DSA_ARGS[@]}" --disable-cuda-graph "$@")
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local rc_wait=0
  ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != "0" ]]; then
    echo "!! boot FAIL ($arm rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
    teardown; gpu_idle_wait; return 10
  fi
  echo ">>> server ready ($arm). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || true
  echo "=== sweep arm=$arm L${LENGTH} num=${NUM} $(date -u +%H:%M:%SZ) ==="
  python "$HERE/dsa_parity_driver.py" \
    --arm "$arm" --length "$LENGTH" --num "$NUM" --max-new-tokens "$MAXNEW" \
    --seed-base "$SEED_BASE" --warmup-length "$WARMUP_LENGTH" \
    --out "$idx" 2>&1 | tee "$OUTDIR/logs/sweep_${arm}.log"
  local sweep_rc=${PIPESTATUS[0]}
  teardown
  echo ">>> sweep ($arm) rc=$sweep_rc ; index=$idx ($(wc -l < "$idx") reqs)"
  gpu_idle_wait
  return $sweep_rc
}

run_arm fresh --disable-radix-cache
FRESH_RC=$?
run_arm reuse
REUSE_RC=$?

echo "=== arms done (fresh=$FRESH_RC reuse=$REUSE_RC) $(date -u +%H:%M:%SZ) ==="

echo "=== analyze $(date -u +%H:%M:%SZ) ==="
python "$HERE/analyze_parity.py" \
  --dir "$OUTDIR" --max-delta-pp 0.5 --min-needles 128 \
  --out "$HERE/dsa_parity_verdict.json" \
  --table-out "$HERE/dsa_parity_per_needle_table.jsonl" 2>&1 | tee "$OUTDIR/logs/analyze.log"
AN_RC=${PIPESTATUS[0]}

echo "=== final nvidia-smi ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "=== DSA parity done (analyze_rc=$AN_RC) $(date -u +%H:%M:%SZ) ==="
exit $(( FRESH_RC != 0 || REUSE_RC != 0 ))
