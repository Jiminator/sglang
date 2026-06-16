#!/usr/bin/env bash
# DEC-9 SCORE-PATH parity (R24 scorepath, owner directive). Radix-ON, EAGER, TP=8.
# Two sequential servers on ONE op-point, differing only in score_reduce_dtype:
#   PASS A  bf16 reduce (the default)  -> exp-1 (pre vs post reduce at flips)
#                                          exp-2 bf16 flip count
#                                          exp-3 (radix vs fallback top-k, offline
#                                                 on PASS-A bf16 cold dumps)
#   PASS B  fp32 reduce                -> exp-2 fp32 flip count
# Then combine: bf16 vs fp32 cold/warm SELECTION flip count.
# NEVER expandable_segments. Servers killed after (GPUs -> 0).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
MASK=/models/glm51-fp8-channel-mask-s256.safetensors
PORT=30000
export PORT

# CRITICAL: never expandable_segments (breaks custom-all-reduce-v2 IPC at TP=8).
unset PYTORCH_CUDA_ALLOC_CONF || true

cd /sgl-workspace/sglang
LOG="$HERE/stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== DEC-9 scorepath start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

export SGLANG_DS_RADIX_OVERRIDE=1
export SGLANG_DS_SCORE_CAPTURE_DIR="$HERE/.sglang_ds_scorecap"
export SGLANG_DS_SELECTION_CAPTURE_DIR="$HERE/.sglang_ds_selcap"

COMMON_ARGS=(--model-path "$GLM" --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
  --disable-overlap-schedule --disable-piecewise-cuda-graph
  --random-seed 20260607 --trust-remote-code --host 127.0.0.1 --port "$PORT"
  --mem-fraction-static 0.8 --max-running-requests 64 --cuda-graph-max-bs 64)

# Base DS config (no removed fields) + score_capture + selection_capture. EAGER.
# {SRD} is substituted with the score_reduce_dtype per pass.
DS_TEMPLATE='{"top_k":2048,"page_size":64,"channel_mask_path":"/models/glm51-fp8-channel-mask-s256.safetensors","device_buffer_size":4096,"scorer_norm":"off","head_agg":"max","anchor_mode":"off","anchor_budget":0,"recall_oracle":false,"enable_lifted_budget_decode":false,"lifted_budget_top_k":0,"score_capture":true,"selection_capture":true,"score_reduce_dtype":"{SRD}"}'

ready_wait() {
  local slog="$1"
  for i in $(seq 1 90); do
    if grep -aqE "SIGQUIT received|Scheduler hit an exception|CUDA out of memory|Capture cuda graph failed|Not enough host memory|ValueError|AssertionError|RuntimeError|Traceback" "$slog" 2>/dev/null; then return 2; fi
    if grep -aq "The server is fired up" "$slog" 2>/dev/null; then
      curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "ready ~$((i*10))s"; return 0; }
    fi
    sleep 10
  done
  return 1
}
teardown() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  pkill -f "sglang::scheduler"   2>/dev/null || true
  sleep 20
  rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null || true
}
gpu_idle_wait() {
  for i in $(seq 1 30); do
    local mx
    mx=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    if [[ "${mx:-99999}" -lt 500 ]]; then echo "gpus idle (max=${mx}MiB)"; return 0; fi
    sleep 5
  done
  echo "gpus still busy (max=${mx}MiB)"; return 1
}

run_pass() {
  # $1 = srd (bf16|fp32), $2 = outdir
  local srd="$1" odir="$2"
  mkdir -p "$odir"
  local ds_config="${DS_TEMPLATE/\{SRD\}/$srd}"
  rm -rf "$SGLANG_DS_SCORE_CAPTURE_DIR" "$SGLANG_DS_SELECTION_CAPTURE_DIR" \
         "$odir/cold_score" "$odir/cold_sel" "$odir/warm_score" "$odir/warm_sel" 2>/dev/null || true

  teardown
  echo "=== boot pass srd=$srd $(date -u +%H:%M:%SZ) ==="
  echo "DS_CONFIG=$ds_config"
  local slog="$odir/serve.log"
  python -m sglang.launch_server "${COMMON_ARGS[@]}" \
    --enable-double-sparsity --double-sparsity-config "$ds_config" --disable-cuda-graph \
    > "$slog" 2>&1 &
  local rc=0
  ready_wait "$slog" || rc=$?
  if [[ "$rc" != "0" ]]; then
    echo "!! boot FAIL srd=$srd (rc=$rc) — tail:"; tail -n 60 "$slog"
    teardown; gpu_idle_wait; return 1
  fi
  echo ">>> server ready (srd=$srd); radix authorization:"
  grep -aE "disable_radix_cache|radix|RADIX_OVERRIDE" "$slog" | head -6 || true

  echo "=== run scorepath driver srd=$srd $(date -u +%H:%M:%SZ) ==="
  python "$HERE/scorepath_driver.py" --prefix-tokens 3800 --reduce-dtype "$srd" \
    --outdir "$odir" --rank 0 --spot-rank 4
  local drc=$?
  echo "=== driver rc=$drc srd=$srd ==="
  return $drc
}

# ---------- PASS A: bf16 reduce ----------
PASS_A="$HERE/pass_bf16"
run_pass bf16 "$PASS_A"
A_RC=$?

# exp-3 (radix vs fallback top-k) on PASS-A bf16 cold dumps BEFORE deletion.
if [[ "$A_RC" == "0" && -d "$PASS_A/cold_score" ]]; then
  echo "=== run exp-3 top-k parity on bf16 cold dumps $(date -u +%H:%M:%SZ) ==="
  python "$HERE/exp3_topk_parity.py" --cold-score-dir "$PASS_A/cold_score" \
    --outdir "$HERE" --ranks 0,4 || echo "!! exp-3 failed"
else
  echo "!! skipping exp-3 (pass A rc=$A_RC or no cold_score dir)"
fi

echo "=== teardown after pass A $(date -u +%H:%M:%SZ) ==="
teardown; gpu_idle_wait

# ---------- PASS B: fp32 reduce ----------
PASS_B="$HERE/pass_fp32"
run_pass fp32 "$PASS_B"
B_RC=$?

echo "=== teardown after pass B $(date -u +%H:%M:%SZ) ==="
teardown; gpu_idle_wait

# ---------- combine bf16 vs fp32 flip counts ----------
echo "=== combine bf16 vs fp32 $(date -u +%H:%M:%SZ) ==="
python "$HERE/combine.py" --bf16 "$PASS_A/summary.json" --fp32 "$PASS_B/summary.json" \
  --exp3 "$HERE/exp3_topk_parity.json" --out "$HERE/scorepath_summary.json" \
  || echo "!! combine failed"

echo "=== deleting raw .pt dumps (keep summaries + flip tables) ==="
for d in "$PASS_A" "$PASS_B"; do
  du -sh "$d/cold_score" "$d/cold_sel" "$d/warm_score" "$d/warm_sel" 2>/dev/null || true
  rm -rf "$d/cold_score" "$d/cold_sel" "$d/warm_score" "$d/warm_sel" 2>/dev/null || true
done
rm -rf "$SGLANG_DS_SCORE_CAPTURE_DIR" "$SGLANG_DS_SELECTION_CAPTURE_DIR" 2>/dev/null || true

echo "=== done; A_RC=$A_RC B_RC=$B_RC $(date -u +%H:%M:%SZ) ==="
find "$HERE" -maxdepth 2 -name "*.json" -o -name "*.jsonl.gz" 2>/dev/null | sort
exit 0
