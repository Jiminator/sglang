#!/usr/bin/env bash
# Loop-11 M2 — side-by-side absorbed-latent recall validation (re-targeted to the
# graph-safe selector path). Boots the SAME GLM-5.1-FP8 op-point that produced the
# loop9 M0 recall baseline (recall_baseline.json: overall 64.696%), EAGER with the
# recall oracle, runs the NIAH oracle sweep, then post-processes the sink for BOTH
# the table key and rec["absorbed"]. One TP=8 server at a time. Never sets
# expandable_segments (the env file unsets it).
set -uo pipefail
REPO=/sgl-workspace/sglang
cd "$REPO"
source "$REPO/development/profiling/runs/20260609/_env.sh"

OUT="$REPO/development/loop11/runs/20260614_m2/recall_absorbed"
mkdir -p "$OUT"
MODE="${1:-smoke}"   # smoke | full

DS_CONFIG_ORACLE="${DS_CONFIG%\}}, \"recall_oracle\": true}"
SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7
  --enable-double-sparsity --double-sparsity-config "$DS_CONFIG_ORACLE"
  --disable-cuda-graph)
SERVE_LOG="$OUT/serve.log"

echo "=== boot GLM DS recall-oracle server (eager) $(date -u +%H:%M:%S)Z ==="
teardown
python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
wait_ready || { echo "!! server not ready"; tail -40 "$SERVE_LOG"; teardown; exit 1; }
echo ">>> server ready"

if [[ "$MODE" == "smoke" ]]; then
  LENGTHS="1024"; NUM=2
  SINK="$OUT/sink_smoke.jsonl"
else
  LENGTHS="1024 4096 16384"; NUM=20
  SINK="$OUT/sink.jsonl"
fi

# The server's TP worker writes to the DEFAULT cross-process sink (env does NOT
# reach workers; that is the config-borne-latch design). The driver reads the
# same default. Copy the default sink to the artifact dir afterward. Truncate the
# default first so this run starts clean.
DEFAULT_SINK="$REPO/.sglang_ds_oracle/sink.jsonl"
mkdir -p "$REPO/.sglang_ds_oracle"
: > "$DEFAULT_SINK"

DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
python "$REPO/development/loop7/niah_oracle_sweep.py" \
  --lengths $LENGTHS --num $NUM --decode-steps 4 \
  --out "$OUT/oracle_trials_index_${MODE}.jsonl" \
  2>&1 | tee "$OUT/sweep_${MODE}.log"
SWEEP_RC=${PIPESTATUS[0]}
teardown
cp "$DEFAULT_SINK" "$SINK"
echo ">>> sweep rc=$SWEEP_RC ; sink=$SINK ($(wc -l < "$SINK") records)"
if [[ $SWEEP_RC -ne 0 ]]; then
  echo "!! sweep failed (rc=$SWEEP_RC); skipping gate"; exit $SWEEP_RC
fi

# Post-process: absorbed-vs-frozen-baseline recall@2048 gate (fail-closed ±0.5pp,
# comparability-checked). Exit status IS the gate status — a passing sweep with a
# failing/non-comparable recall is a FAIL.
SUMMARY="$OUT/absorbed_recall_summary_${MODE}.json"
python "$REPO/development/loop11/runs/20260614_m2/absorbed_recall_summary.py" \
  --sink "$SINK" \
  --baseline "$REPO/development/loop9/runs/20260610_m0/recall_baseline.json" \
  --out "$SUMMARY"
GATE_RC=$?
if [[ "$MODE" == "full" ]]; then
  cp "$SUMMARY" "$OUT/absorbed_recall_summary.json"   # canonical tracked artifact
fi
echo ">>> gate rc=$GATE_RC ; summary=$SUMMARY"
exit $GATE_RC
