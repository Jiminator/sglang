#!/usr/bin/env bash
# Experiment A — is the radix-on L4096 recall@2048 breach (R24 P2, -0.849pp) noise?
# Reuses the EXACT R24 P2 harness: r24_env.sh op-point, niah_oracle_sweep.py,
# DS recall_oracle config. For each server --random-seed S, boot DS EAGER
# recall_oracle TWICE on the IDENTICAL trial set: radix-OFF then radix-ON; save
# each oracle sink. Then a higher-n run at L4096 num=80 seed 20260607.
# One TP=8 server at a time; teardown to ~0 MiB between every boot.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R24="$HERE/../tablefree_radix"
source "$R24/r24_env.sh"
REPO=/sgl-workspace/sglang
cd "$REPO"
LOG="$HERE/expA_stage.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== ExpA start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
OUT="$HERE/recall"; mkdir -p "$OUT"
DEFAULT_SINK="$REPO/.sglang_ds_oracle/sink.jsonl"
mkdir -p "$REPO/.sglang_ds_oracle"

DS_CFG_BASE='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "recall_oracle": true, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'

# run_one TAG SEED LENGTHS NUM RADIX(off|on)
run_one() {
  local tag="$1"; local seed="$2"; local lengths="$3"; local num="$4"; local radix="$5"
  local slog="$OUT/serve_${tag}.log"
  local sink="$OUT/sink_${tag}.jsonl"
  teardown
  : > "$DEFAULT_SINK"
  echo "=== boot DS ($tag) seed=$seed radix=$radix lengths='$lengths' num=$num $(date -u +%H:%M:%SZ) ==="
  # rebuild COMMON_ARGS with the per-seed --random-seed
  local args=(--model-path "$GLM" --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64
    --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
    --disable-overlap-schedule --disable-piecewise-cuda-graph
    --random-seed "$seed" --trust-remote-code --host 127.0.0.1 --port "$PORT"
    --mem-fraction-static 0.8 --max-running-requests 64 --cuda-graph-max-bs 64
    --enable-double-sparsity --double-sparsity-config "$DS_CFG_BASE" --disable-cuda-graph)
  if [[ "$radix" == "off" ]]; then
    unset SGLANG_DS_RADIX_OVERRIDE || true
    args+=(--disable-radix-cache)
  else
    export SGLANG_DS_RADIX_OVERRIDE=1
  fi
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local rc_wait=0
  ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != "0" ]]; then
    echo "!! ExpA boot FAIL ($tag rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
    teardown; gpu_idle_wait; unset SGLANG_DS_RADIX_OVERRIDE || true; return 10
  fi
  echo ">>> server ready ($tag). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || true
  DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
  python "$REPO/development/loop7/niah_oracle_sweep.py" \
    --lengths $lengths --num "$num" --decode-steps 4 \
    --out "$OUT/trials_${tag}.jsonl" 2>&1 | tee "$OUT/sweep_${tag}.log"
  local sweep_rc=${PIPESTATUS[0]}
  teardown
  cp "$DEFAULT_SINK" "$sink"
  echo ">>> sweep ($tag) rc=$sweep_rc ; sink=$sink ($(wc -l < "$sink") records)"
  unset SGLANG_DS_RADIX_OVERRIDE || true
  gpu_idle_wait
  return $sweep_rc
}

SEEDS=(20260607 20260608 20260609 20260610)
declare -A RC
for S in "${SEEDS[@]}"; do
  run_one "s${S}_off" "$S" "4096 16384" 20 off; RC["s${S}_off"]=$?
  run_one "s${S}_on"  "$S" "4096 16384" 20 on;  RC["s${S}_on"]=$?
done
# higher-n L4096 num=80 seed 20260607
run_one "hi_off" 20260607 "4096" 80 off; RC["hi_off"]=$?
run_one "hi_on"  20260607 "4096" 80 on;  RC["hi_on"]=$?

echo "=== ExpA all boots done $(date -u +%H:%M:%SZ) ==="
for k in "${!RC[@]}"; do echo "  rc[$k]=${RC[$k]}"; done
teardown; gpu_idle_wait
echo "=== ExpA stage complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
