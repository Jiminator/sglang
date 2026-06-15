#!/usr/bin/env bash
# P2 (CORRECTED) — recall@2048 radix-OFF vs radix-ON on the SAME seed/trial set.
# Boots DS EAGER + recall_oracle twice: (1) radix-OFF (--disable-radix-cache),
# (2) radix-ON (SGLANG_DS_RADIX_OVERRIDE=1, no --disable-radix-cache). Runs the
# IDENTICAL niah_oracle_sweep.py both times, captures each sink, then the
# fail-closed compare helper enforces per-length + overall recall within ±0.5pp.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/r24_env.sh"
REPO=/sgl-workspace/sglang
cd "$REPO"
LOG="$HERE/p2_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== P2 recall off-vs-on start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
MODE="${1:-full}"
OUT="$HERE/recall"; mkdir -p "$OUT"
if [[ "$MODE" == "smoke" ]]; then LENGTHS="1024"; NUM=2; else LENGTHS="1024 4096 16384"; NUM=20; fi

DS_CFG_BASE='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "recall_oracle": true, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'
DEFAULT_SINK="$REPO/.sglang_ds_oracle/sink.jsonl"
mkdir -p "$REPO/.sglang_ds_oracle"

run_sweep() {  # $1=tag (off|on)  $2..=extra launch args ; reads RADIX_OVERRIDE env
  local tag="$1"; shift
  local slog="$OUT/serve_${tag}.log"
  local sink="$OUT/sink_${tag}.jsonl"
  teardown
  : > "$DEFAULT_SINK"
  echo "=== boot DS ($tag) eager recall_oracle $(date -u +%H:%M:%SZ) ==="
  local args=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_BASE" --disable-cuda-graph "$@")
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local rc_wait=0
  ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != "0" ]]; then
    echo "!! P2 boot FAIL ($tag rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
    teardown; gpu_idle_wait; return 10
  fi
  echo ">>> server ready ($tag). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || true
  echo "=== NIAH sweep ($tag) lengths=$LENGTHS num=$NUM $(date -u +%H:%M:%SZ) ==="
  DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
  python "$REPO/development/loop7/niah_oracle_sweep.py" \
    --lengths $LENGTHS --num $NUM --decode-steps 4 \
    --out "$OUT/oracle_trials_${tag}.jsonl" 2>&1 | tee "$OUT/sweep_${tag}.log"
  local sweep_rc=${PIPESTATUS[0]}
  teardown
  cp "$DEFAULT_SINK" "$sink"
  echo ">>> sweep ($tag) rc=$sweep_rc ; sink=$sink ($(wc -l < "$sink") records)"
  gpu_idle_wait
  return $sweep_rc
}

# (1) radix-OFF — shipped default for DS adds --disable-radix-cache.
unset SGLANG_DS_RADIX_OVERRIDE || true
run_sweep off --disable-radix-cache
OFF_RC=$?

# (2) radix-ON — dev override, drop --disable-radix-cache.
export SGLANG_DS_RADIX_OVERRIDE=1
run_sweep on
ON_RC=$?
unset SGLANG_DS_RADIX_OVERRIDE || true

echo "=== compare off vs on $(date -u +%H:%M:%SZ) (off_rc=$OFF_RC on_rc=$ON_RC) ==="
CMP_RC=99
if [[ $OFF_RC -eq 0 && $ON_RC -eq 0 ]]; then
  python "$HERE/p2_recall_compare.py" \
    --sink-off "$OUT/sink_off.jsonl" --sink-on "$OUT/sink_on.jsonl" \
    --max-delta-pp 0.5 --out "$HERE/p2_recall_verdict.json" 2>&1 | tee "$OUT/compare.log"
  CMP_RC=${PIPESTATUS[0]}
else
  echo "!! sweep(s) failed; not running comparator (off_rc=$OFF_RC on_rc=$ON_RC)"
fi

{
  echo "probe=p2_recall_off_vs_on (radix off vs on, SAME seed/trials, eager recall_oracle)"
  echo "off_sweep_rc=$OFF_RC on_sweep_rc=$ON_RC compare_rc=$CMP_RC"
  echo "status=$([[ $CMP_RC -eq 0 ]] && echo PASS || echo FAIL)"
  echo "--- compare summary ---"; tail -20 "$OUT/compare.log" 2>/dev/null || echo "(no compare log)"
} > "$HERE/p2_evidence.txt"
cat "$HERE/p2_evidence.txt"
echo "=== P2 done $(date -u +%H:%M:%SZ) ==="
gpu_idle_wait
exit $CMP_RC
