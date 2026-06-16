#!/usr/bin/env bash
# loop11b GATE A — recall_equivalence (DEC-12). recall@2048 radix-OFF vs radix-ON on the
# SAME NIAH trial set, lengths {1024,4096,16384}, recall_oracle EAGER. PASS = |off-on|
# <= 0.5pp OVERALL and for EVERY length bin. Reuses loop7/niah_oracle_sweep.py + the
# loop-11 R24 fail-closed comparator p2_recall_compare.py (no new comparator code).
# num=20 x decode-steps=4 x 78 layers => 6240 records/length (matches the frozen baseline).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
cd "$REPO"
OUT="$HERE/probes/gate_a_recall"; mkdir -p "$OUT"
LOG="$OUT/stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== loop11b GATE A recall off-vs-on $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) ==="
MODE="${1:-full}"
if [[ "$MODE" == "smoke" ]]; then LENGTHS="1024"; NUM=2; else LENGTHS="1024 4096 16384"; NUM="${NUM_OVERRIDE:-20}"; fi
mkdir -p "$REPO/.sglang_ds_oracle"
CMP="$REPO/development/loop11/runs/20260616_r24/tablefree_radix/p2_recall_compare.py"

run_sweep() {  # $1=tag (off|on)  $2..=extra launch args
  local tag="$1"; shift
  local slog="$OUT/serve_${tag}.log"
  local sink="$OUT/sink_${tag}.jsonl"
  teardown; : > "$DEFAULT_SINK"
  echo "=== boot DS ($tag) eager recall_oracle $(date -u +%H:%M:%SZ) ==="
  local args=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_RECALL" --disable-cuda-graph "$@")
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local rc_wait=0; ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != 0 ]]; then echo "!! boot FAIL ($tag rc=$rc_wait) — tail:"; tail -n 60 "$slog"; teardown; gpu_idle_wait; return 10; fi
  echo ">>> ready ($tag). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || true
  echo "=== NIAH sweep ($tag) lengths=$LENGTHS num=$NUM $(date -u +%H:%M:%SZ) ==="
  DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
  python "$REPO/development/loop7/niah_oracle_sweep.py" \
    --lengths $LENGTHS --num $NUM --decode-steps 4 \
    --out "$OUT/oracle_trials_${tag}.jsonl" 2>&1 | tee "$OUT/sweep_${tag}.log"
  local sweep_rc=${PIPESTATUS[0]}
  teardown; cp "$DEFAULT_SINK" "$sink"
  echo ">>> sweep ($tag) rc=$sweep_rc sink=$sink ($(wc -l < "$sink") records)"
  gpu_idle_wait; return $sweep_rc
}

unset SGLANG_DS_RADIX_OVERRIDE || true
run_sweep off --disable-radix-cache; OFF_RC=$?
export SGLANG_DS_RADIX_OVERRIDE=1; run_sweep on; ON_RC=$?; unset SGLANG_DS_RADIX_OVERRIDE || true

echo "=== compare off vs on (off_rc=$OFF_RC on_rc=$ON_RC) ==="
CMP_RC=99
if [[ $OFF_RC -eq 0 && $ON_RC -eq 0 ]]; then
  python "$CMP" --sink-off "$OUT/sink_off.jsonl" --sink-on "$OUT/sink_on.jsonl" \
    --max-delta-pp 0.5 --out "$HERE/probes/gate_a_verdict.json" 2>&1 | tee "$OUT/compare.log"
  CMP_RC=${PIPESTATUS[0]}
else
  echo "!! sweep(s) failed; not comparing"
fi
{
  echo "probe=gate_a_recall_equivalence (radix off vs on, SAME trial set, eager recall_oracle)"
  echo "lengths=$LENGTHS num=$NUM decode_steps=4"
  echo "off_sweep_rc=$OFF_RC on_sweep_rc=$ON_RC compare_rc=$CMP_RC"
  echo "status=$([[ $CMP_RC -eq 0 ]] && echo PASS || echo FAIL)"
  echo "--- compare summary ---"; tail -25 "$OUT/compare.log" 2>/dev/null || echo "(no compare log)"
} > "$HERE/probes/gate_a_evidence.txt"
cat "$HERE/probes/gate_a_evidence.txt"
echo "=== GATE A done $(date -u +%H:%M:%SZ) ==="; gpu_idle_wait
exit $CMP_RC
