#!/usr/bin/env bash
# Channel-mask calibration runner for GLM-5.1-FP8 (table-free Double Sparsity).
#
# Regenerates the offline channel mask (S_h, w_c) that table-free DS consumes on the
# query side. Table-free scoring removed the per-token TokenLabelTable, NOT the mask.
#
# Usage:
#   bash development/loop11b/run_calibrate.sh dryrun   # FP8 placement preflight (no mask written)
#   bash development/loop11b/run_calibrate.sh full      # full calibration, writes the mask
#
# Corpus: a deterministic committed Pile-val corpus (one doc per line). Reproducible
# from development/loop11b/artifacts/calib_corpus_pileval.txt.
set -euo pipefail

MODE="${1:?usage: run_calibrate.sh dryrun|full}"
MODEL="${MODEL_PATH:-/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db}"
CORPUS="${CORPUS:-development/loop11b/artifacts/calib_corpus_pileval.txt}"
OUT="${OUT:-/cluster-storage/models/glm51-fp8-channel-mask-s256.safetensors}"

# expandable_segments avoids the FP8 mid-load fragmentation OOM during calibration
# (the failed alloc is small while ~GiBs sit reserved-but-unallocated). This is a
# CALIBRATION-ONLY allocator policy; it must NEVER be set for serving (it breaks
# custom-all-reduce IPC at GLM TP=8 — cudaIpcGetMemHandle does not work on VMM
# memory). Calibration and serving are separate processes, so setting it here does
# not affect any server.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GLM-native recipe (loop8 DEC-3, validated 2026-06-07): --dtype fp8_e4m3 and
# --label-dim 32 (NOT the DeepSeek-V3.2 values bfloat16/16 — label_dim 32 is
# proportionate to GLM's qk_nope_head_dim=192 and retains 2x the channels, which
# the recall gate requires). Same corpus/num-samples/block-size/seed as dsv32.
ARGS=( --model "$MODEL" --dtype fp8_e4m3 --kv-cache-dtype fp8_e4m3 --tp 8
  --label-dim 32 --page-size 64 --num-samples 256 --block-size 512 --seed 42
  --dataset "$CORPUS" --output "$OUT" )
if [[ "$MODE" == "dryrun" ]]; then
  ARGS+=( --dry-run-blocks 1 )
elif [[ "$MODE" != "full" ]]; then
  echo "ERROR: mode must be 'dryrun' or 'full', got '$MODE'" >&2; exit 2
fi

echo ">>> calibrate mode=$MODE"
echo "    model  = $MODEL"
echo "    corpus = $CORPUS"
echo "    out    = $OUT"
echo "    alloc  = PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF (calibration-only)"
echo "    cmd    = python3 -m sglang.srt.layers.attention.double_sparsity.calibrate ${ARGS[*]}"
exec python3 -m sglang.srt.layers.attention.double_sparsity.calibrate "${ARGS[@]}"
