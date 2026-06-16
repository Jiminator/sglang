#!/usr/bin/env bash
# GATE A — robust full-sweep recall_equivalence. recall@2048 radix-OFF
# (--disable-radix-cache) vs radix-ON (SGLANG_DS_RADIX_OVERRIDE=1), EAGER,
# recall_oracle=true, lengths {1024,4096,16384}, num>=128 matched-paired needles.
# PROVES per-length cache engagement from raw per-request cached_tokens (ON>0, OFF==0)
# via a shorter same-seed warmup before each measured trial (the proven n=144 L4096
# mechanism, generalized per length). PASS = |mean paired delta| <= 0.5pp OVERALL and
# for EVERY length, applied on the MEAN. Reuses the R25 op-point boot infra.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../radix_authorization/r25_env.sh"
cd "$REPO"

mkdir -p "$HERE/logs"
LOG="$HERE/logs/stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== GATE A start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
echo "[gateA] expandable_segments=$([[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] && echo UNSET || echo "${PYTORCH_CUDA_ALLOC_CONF}")"

LENGTHS="${LENGTHS:-1024 4096 16384}"
NUM="${NUM:-128}"
DECODE_STEPS=4
SEED_BASE=1000

mkdir -p "$REPO/.sglang_ds_oracle"

run_arm() {  # $1=tag (off|on)  $2..=extra launch args
  local tag="$1"; shift
  local slog="$HERE/logs/serve_${tag}.log"
  local sink="$HERE/sink_${tag}.jsonl"
  local idx="$HERE/index_${tag}.jsonl"
  teardown
  : > "$DEFAULT_SINK"
  echo "=== boot DS ($tag) eager recall_oracle $(date -u +%H:%M:%SZ) ==="
  local args=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_RECALL" --disable-cuda-graph "$@")
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local rc_wait=0
  ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != "0" ]]; then
    echo "!! GATE A boot FAIL ($tag rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
    teardown; gpu_idle_wait; return 10
  fi
  echo ">>> server ready ($tag). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || true
  echo "=== sweep ($tag) lengths='${LENGTHS}' num=${NUM} decode-steps=${DECODE_STEPS} seed_base=${SEED_BASE} $(date -u +%H:%M:%SZ) ==="
  DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
  python "$HERE/niah_multilen_driver.py" \
    --lengths $LENGTHS --num "$NUM" --decode-steps "$DECODE_STEPS" --seed-base "$SEED_BASE" \
    --out "$idx" 2>&1 | tee "$HERE/logs/sweep_${tag}.log"
  local sweep_rc=${PIPESTATUS[0]}
  teardown
  cp "$DEFAULT_SINK" "$sink"
  echo ">>> sweep ($tag) rc=$sweep_rc ; sink=$sink ($(wc -l < "$sink") records); index=$idx ($(wc -l < "$idx") reqs)"
  gpu_idle_wait
  return $sweep_rc
}

# (1) radix-OFF control — DS shipped default adds --disable-radix-cache; cached_tokens must be 0.
unset SGLANG_DS_RADIX_OVERRIDE || true
run_arm off --disable-radix-cache
OFF_RC=$?

# (2) radix-ON — SGLANG_DS_RADIX_OVERRIDE=1, drop --disable-radix-cache; prove cached_tokens > 0.
export SGLANG_DS_RADIX_OVERRIDE=1
run_arm on
ON_RC=$?
unset SGLANG_DS_RADIX_OVERRIDE || true

echo "=== analyze off vs on (off_rc=$OFF_RC on_rc=$ON_RC) $(date -u +%H:%M:%SZ) ==="
ANALYZE_RC=99
if [[ $OFF_RC -eq 0 && $ON_RC -eq 0 ]]; then
  python "$HERE/analyze_multilen.py" \
    --sink-off "$HERE/sink_off.jsonl" --sink-on "$HERE/sink_on.jsonl" \
    --index-off "$HERE/index_off.jsonl" --index-on "$HERE/index_on.jsonl" \
    --lengths $LENGTHS --max-delta-pp 0.5 --out "$HERE/gate_a_verdict.json" 2>&1 | tee "$HERE/logs/analyze.log"
  ANALYZE_RC=${PIPESTATUS[0]}
else
  echo "!! sweep(s) failed; not analyzing (off_rc=$OFF_RC on_rc=$ON_RC)"
fi

echo "=== final nvidia-smi ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "=== GATE A done $(date -u +%H:%M:%SZ) (analyze_rc=$ANALYZE_RC) ==="
gpu_idle_wait
exit $(( OFF_RC != 0 || ON_RC != 0 ))
