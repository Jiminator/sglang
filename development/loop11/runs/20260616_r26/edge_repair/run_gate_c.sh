#!/usr/bin/env bash
# R26 GATE C — ROBUST-N edge probe (length 4096, page 64), the original ±0.5pp-clean
# contract. Repairs the R25 thin-sample fail-open (single-prompt n=312).
#
# Reuses the PROVEN cache-engaging harness (R25 l4096_highn + p1_recall infra:
# r25_env op-point + niah oracle path + L1024 same-seed warmup) and records
# per-request cached_tokens so cache engagement is PROVEN from raw data per arm.
#
# Op-point (all boots): GLM-5.1-FP8 TP=8 fp8_e4m3 page64 mem0.8 max-running64 EAGER
# (--disable-cuda-graph), DS table-free recall_oracle=true (scorer_norm off,
# head_agg max, top_k 2048, page_size 64, channel_mask glm51-fp8-channel-mask-s256,
# device_buffer_size 4096, anchor off). NEVER expandable_segments. ONE server at a
# time; teardown to ~0 MiB between arms; kill all at end.
#
#   cold arm  -> radix-OFF boot (--disable-radix-cache), cached MUST be 0
#   on arms   -> radix-ON boot (SGLANG_DS_RADIX_OVERRIDE=1, no --disable-radix-cache):
#                boundary + partial + eviction run in the SAME server
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../20260616_r25/radix_authorization/r25_env.sh"
cd "$REPO"

OUTDIR="$HERE/gate_c"; mkdir -p "$OUTDIR/logs"
LOG="$OUTDIR/logs/stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== R26 GATE C robust-n edge start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
echo "[gate_c] expandable_segments=$([[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] && echo UNSET || echo "${PYTORCH_CUDA_ALLOC_CONF}")"

LENGTH=4096
NUM="${1:-144}"
DECODE_STEPS=4
SEED_BASE=1000
WARMUP_LENGTH=1024
mkdir -p "$REPO/.sglang_ds_oracle"

run_arm() {  # $1=arm (cold|boundary|partial|evict)  $2=boot-tag(off|on)  $3..=extra launch args
  local arm="$1"; local boot="$2"; shift 2
  local slog="$OUTDIR/logs/serve_${arm}.log"
  local sink="$OUTDIR/sink_${arm}.jsonl"
  local idx="$OUTDIR/index_${arm}.jsonl"
  teardown
  : > "$DEFAULT_SINK"
  echo "=== boot DS ($boot, arm=$arm) eager recall_oracle $(date -u +%H:%M:%SZ) ==="
  local args=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_RECALL" --disable-cuda-graph "$@")
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local rc_wait=0
  ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != "0" ]]; then
    echo "!! boot FAIL ($arm rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
    teardown; gpu_idle_wait; return 10
  fi
  echo ">>> server ready ($arm). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || true
  echo "=== edge sweep arm=$arm L${LENGTH} num=${NUM} $(date -u +%H:%M:%SZ) ==="
  DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
  python "$HERE/edge_highn_driver.py" \
    --arm "$arm" --length "$LENGTH" --num "$NUM" --decode-steps "$DECODE_STEPS" \
    --seed-base "$SEED_BASE" --warmup-length "$WARMUP_LENGTH" \
    --out "$idx" 2>&1 | tee "$OUTDIR/logs/sweep_${arm}.log"
  local sweep_rc=${PIPESTATUS[0]}
  teardown
  cp "$DEFAULT_SINK" "$sink"
  echo ">>> sweep ($arm) rc=$sweep_rc ; sink=$sink ($(wc -l < "$sink") records); index=$idx ($(wc -l < "$idx") reqs)"
  gpu_idle_wait
  return $sweep_rc
}

# (1) COLD/FRESH control — radix-OFF boot; cached_tokens MUST be 0.
unset SGLANG_DS_RADIX_OVERRIDE || true
run_arm cold off --disable-radix-cache
COLD_RC=$?

# (2) ON arms — radix-ON (SGLANG_DS_RADIX_OVERRIDE=1, no --disable-radix-cache).
#     boundary + partial + eviction in the SAME server (one boot each, but radix-ON).
export SGLANG_DS_RADIX_OVERRIDE=1
run_arm boundary on
BND_RC=$?
run_arm partial on
PRT_RC=$?
run_arm evict on
EVC_RC=$?
unset SGLANG_DS_RADIX_OVERRIDE || true

echo "=== arms done (cold=$COLD_RC bnd=$BND_RC prt=$PRT_RC evc=$EVC_RC) $(date -u +%H:%M:%SZ) ==="

# (3) analyze -> gate_c_edge_verdict.json
echo "=== analyze $(date -u +%H:%M:%SZ) ==="
python "$HERE/analyze_edge.py" \
  --dir "$OUTDIR" --length "$LENGTH" --page-size 64 --max-delta-pp 0.5 --min-needles 128 \
  --out "$HERE/gate_c_edge_verdict.json" \
  --table-out "$HERE/gate_c_per_needle_table.jsonl" 2>&1 | tee "$OUTDIR/logs/analyze.log"
AN_RC=${PIPESTATUS[0]}

echo "=== final nvidia-smi ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "=== R26 GATE C done (analyze_rc=$AN_RC) $(date -u +%H:%M:%SZ) ==="
exit $(( COLD_RC != 0 || BND_RC != 0 || PRT_RC != 0 || EVC_RC != 0 ))
