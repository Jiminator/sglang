#!/usr/bin/env bash
# P2 — recall@2048 under radix-ON. Reuses the R16 absorbed-recall flow
# (niah_oracle_sweep.py + absorbed_recall_summary.py) but boots DS radix-ON
# (SGLANG_DS_RADIX_OVERRIDE=1, no --disable-radix-cache). EAGER + recall_oracle
# (the oracle hook host-syncs; requires --disable-cuda-graph). Gates the absorbed
# overall + per-length recall@2048 within +/-0.5pp of the frozen baseline, with
# the SAME comparability checks (length set + per-length sample counts).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/r23_env.sh"
REPO=/sgl-workspace/sglang
cd "$REPO"
LOG="$HERE/p2_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== P2 recall radix-ON start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
MODE="${1:-full}"
OUT="$HERE/recall_radixon"; mkdir -p "$OUT"

export SGLANG_DS_RADIX_OVERRIDE=1
# DS radix-ON config WITH recall_oracle. EAGER (--disable-cuda-graph).
DS_CONFIG_P2='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "recall_oracle": true, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'
P2_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CONFIG_P2" --disable-cuda-graph)

teardown
slog="$OUT/serve.log"
echo "=== boot DS radix-ON eager recall_oracle $(date -u +%H:%M:%SZ) ==="
echo "DS_CONFIG_P2=$DS_CONFIG_P2"
SGLANG_DS_RADIX_OVERRIDE=1 python -m sglang.launch_server "${P2_ARGS[@]}" > "$slog" 2>&1 &
rc_wait=0
ready_wait "$slog" || rc_wait=$?
if [[ "$rc_wait" != "0" ]]; then
  echo "!! P2 boot FAIL (rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
  { echo "probe=p2_recall_radixon status=BOOT_FAIL rc_wait=$rc_wait";
    grep -anE "Error|ValueError|RuntimeError|OOM|Traceback|radix" "$slog" | tail -40; } > "$HERE/p2_evidence.txt"
  teardown; exit 1
fi
echo ">>> server ready (radix-ON, recall_oracle). smoke=$(smoke)"
grep -aE "disable_radix_cache=False" "$slog" | head -1 || true

if [[ "$MODE" == "smoke" ]]; then LENGTHS="1024"; NUM=2; SINKTAG=smoke; else LENGTHS="1024 4096 16384"; NUM=20; SINKTAG=full; fi
SINK="$OUT/sink_${SINKTAG}.jsonl"
DEFAULT_SINK="$REPO/.sglang_ds_oracle/sink.jsonl"
mkdir -p "$REPO/.sglang_ds_oracle"; : > "$DEFAULT_SINK"

echo "=== NIAH oracle sweep (lengths=$LENGTHS num=$NUM) $(date -u +%H:%M:%SZ) ==="
DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
python "$REPO/development/loop7/niah_oracle_sweep.py" \
  --lengths $LENGTHS --num $NUM --decode-steps 4 \
  --out "$OUT/oracle_trials_index_${SINKTAG}.jsonl" 2>&1 | tee "$OUT/sweep_${SINKTAG}.log"
SWEEP_RC=${PIPESTATUS[0]}
teardown
cp "$DEFAULT_SINK" "$SINK"
echo ">>> sweep rc=$SWEEP_RC ; sink=$SINK ($(wc -l < "$SINK") records)"
if [[ $SWEEP_RC -ne 0 ]]; then
  echo "probe=p2_recall_radixon status=SWEEP_FAIL rc=$SWEEP_RC" > "$HERE/p2_evidence.txt"
  gpu_idle_wait; exit $SWEEP_RC
fi

SUMMARY="$OUT/absorbed_recall_summary_${SINKTAG}.json"
python "$REPO/development/loop11/runs/20260614_m2/absorbed_recall_summary.py" \
  --sink "$SINK" \
  --baseline "$REPO/development/loop9/runs/20260610_m0/recall_baseline.json" \
  --out "$SUMMARY" 2>&1 | tee "$OUT/gate_${SINKTAG}.log"
GATE_RC=${PIPESTATUS[0]}
{
  echo "probe=p2_recall_radixon (radix-ON, recall_oracle, eager) status=$([[ $GATE_RC -eq 0 ]] && echo PASS || echo CHARACTERIZATION/FAIL)"
  echo "mode=$MODE lengths=$LENGTHS num=$NUM"
  echo "--- gate summary tail ---"; tail -20 "$OUT/gate_${SINKTAG}.log"
} > "$HERE/p2_evidence.txt"
cat "$HERE/p2_evidence.txt"
echo "=== P2 done $(date -u +%H:%M:%SZ) ==="
gpu_idle_wait
exit $GATE_RC
