#!/usr/bin/env bash
# DEC-9 Q1 — DETERMINISM CONTROL. Boot DS radix-OFF, EAGER, selection_capture only
# (NO new code). Send the SAME prompt as TWO fresh requests; compare selected
# indices across all ranks + DS layers. One TP=8 server; kill after.
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
echo "=== DEC-9 Q1 start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

export SGLANG_DS_SELECTION_CAPTURE_DIR="$HERE/.sglang_ds_selcap"
rm -rf "$SGLANG_DS_SELECTION_CAPTURE_DIR" "$HERE/fresh1_sel" "$HERE/fresh2_sel" 2>/dev/null || true

# radix OFF (--disable-radix-cache) so the two requests are genuinely fresh.
COMMON_ARGS=(--model-path "$GLM" --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
  --disable-overlap-schedule --disable-piecewise-cuda-graph --disable-radix-cache
  --random-seed 20260607 --trust-remote-code --host 127.0.0.1 --port "$PORT"
  --mem-fraction-static 0.8 --max-running-requests 64 --cuda-graph-max-bs 64)

# Base DS config (no removed fields) + selection_capture ONLY. EAGER.
DS_CONFIG='{"top_k":2048,"page_size":64,"channel_mask_path":"/models/glm51-fp8-channel-mask-s256.safetensors","device_buffer_size":4096,"scorer_norm":"off","head_agg":"max","anchor_mode":"off","anchor_budget":0,"recall_oracle":false,"enable_lifted_budget_decode":false,"lifted_budget_top_k":0,"selection_capture":true}'
ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" --disable-cuda-graph)

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

teardown
echo "=== boot DS radix-OFF eager selection_capture $(date -u +%H:%M:%SZ) ==="
echo "DS_CONFIG=$DS_CONFIG"
slog="$HERE/serve.log"
python -m sglang.launch_server "${ARGS[@]}" > "$slog" 2>&1 &
rc=0
ready_wait "$slog" || rc=$?
if [[ "$rc" != "0" ]]; then
  echo "!! boot FAIL (rc=$rc) — tail:"; tail -n 60 "$slog"
  teardown; gpu_idle_wait; exit 1
fi
echo ">>> server ready; radix authorization:"
grep -aE "disable_radix_cache|radix|RADIX_OVERRIDE" "$slog" | head -8 || true

echo "=== run Q1 determinism-control driver $(date -u +%H:%M:%SZ) ==="
python "$HERE/q1_driver.py" --prefix-tokens 3800 --outdir "$HERE"
DRV_RC=$?

echo "=== driver rc=$DRV_RC; tearing down $(date -u +%H:%M:%SZ) ==="
teardown
gpu_idle_wait

echo "=== deleting raw .pt snapshots (keep summary.json) ==="
du -sh "$HERE/fresh1_sel" "$HERE/fresh2_sel" 2>/dev/null || true
rm -rf "$HERE/fresh1_sel" "$HERE/fresh2_sel" "$SGLANG_DS_SELECTION_CAPTURE_DIR" 2>/dev/null || true
ls -la "$HERE"
exit $DRV_RC
