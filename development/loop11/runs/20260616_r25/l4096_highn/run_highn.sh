#!/usr/bin/env bash
# L4096 HIGH-N characterization — radix OFF vs ON, recall@2048, SAME prompt set.
# CHARACTERIZATION ONLY: no radix authorizing artifact, no no-override boot. Gate
# stays closed. Reuses the PROVEN cache-engaging harness (r25 p1_recall infra:
# r24_env op-point + niah oracle path) and records per-request cached_tokens so
# cache engagement is proven from raw data (owner rule #1).
#
# Op-point (both boots): GLM-5.1-FP8 TP=8 fp8_e4m3 page64 mem0.8 max-running64
# EAGER (--disable-cuda-graph), DS table-free recall_oracle=true (scorer_norm off,
# head_agg max, top_k 2048, page_size 64, channel_mask glm51-fp8-channel-mask-s256,
# device_buffer_size 4096, anchor off). NEVER expandable_segments. ONE server at a
# time; teardown to ~0 MiB between OFF and ON.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Reuse the exact R25 env (sources r24_env.sh: COMMON_ARGS, DS_CFG_RECALL,
# ready_wait/smoke/teardown/gpu_idle_wait, PORT, GLM, REPO).
source "$HERE/../radix_authorization/r25_env.sh"
cd "$REPO"

LOG="$HERE/logs/stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== L4096 high-n start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
echo "[highn] expandable_segments=$([[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] && echo UNSET || echo "${PYTORCH_CUDA_ALLOC_CONF}")"

LENGTH=4096
NUM="${1:-144}"
DECODE_STEPS=4
SEED_BASE=1000
# Pre-seed the radix prefix with the SAME-seed L1024 prompt before each L4096 trial
# (the proven niah_oracle_sweep ran L1024 first; L1024-iN shares ~590 leading words
# with L4096-iN, giving the ~320-640 cached-token prefix reuse seen in p1_recall).
WARMUP_LENGTH=1024

mkdir -p "$REPO/.sglang_ds_oracle" "$HERE/logs"

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
    echo "!! boot FAIL ($tag rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
    teardown; gpu_idle_wait; return 10
  fi
  echo ">>> server ready ($tag). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || true
  echo "=== high-n sweep ($tag) L${LENGTH} num=${NUM} decode-steps=${DECODE_STEPS} seed_base=${SEED_BASE} warmup=${WARMUP_LENGTH} $(date -u +%H:%M:%SZ) ==="
  DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
  python "$HERE/niah_highn_driver.py" \
    --length "$LENGTH" --num "$NUM" --decode-steps "$DECODE_STEPS" --seed-base "$SEED_BASE" \
    --warmup-length "$WARMUP_LENGTH" \
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

echo "=== arms done (off_rc=$OFF_RC on_rc=$ON_RC) $(date -u +%H:%M:%SZ) ==="
echo "=== final nvidia-smi ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "=== L4096 high-n boots complete $(date -u +%H:%M:%SZ) ==="
exit $(( OFF_RC != 0 || ON_RC != 0 ))
