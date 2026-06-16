#!/usr/bin/env bash
# loop11b M-B AC-4 — per-step tax guard. DS-vs-DSA one-batch decode latency at bs64 (ratio
# <= 1.10) and bs30, GRAPH mode, both mem 0.8, via sglang.bench_one_batch (fixed batch, times
# prefill + per-step decode). DS runs radix-on via the minted fixture; DSA-native radix-on
# default. The per-step decode cost is radix-independent (radix = prefix caching), but the
# radix state is DECLARED. ilen 4096 (the SLOS workload context).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/loop11b/runs/20260616_ma/mint/env.sh
cd "$REPO"
OUT="$HERE/probes"; mkdir -p "$OUT"
FIX=/sgl-workspace/sglang/development/serve_double_sparsity_radix_fixture.json
ILEN="${ILEN:-4096}"; OLEN="${OLEN:-32}"; BS="${BS:-30 64}"
COMMON=(--model-path "$GLM" --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
  --disable-overlap-schedule --disable-piecewise-cuda-graph
  --mem-fraction-static 0.8 --cuda-graph-max-bs 64 --trust-remote-code)

run_bench() {  # $1=tag $2..=extra args
  local tag="$1"; shift
  teardown
  echo "=== bench_one_batch ($tag) bs=[$BS] ilen=$ILEN olen=$OLEN $(date -u +%H:%M:%SZ) ==="
  python -m sglang.bench_one_batch "${COMMON[@]}" "$@" \
    --batch-size $BS --input-len $ILEN --output-len $OLEN --run-name "$tag" \
    > "$OUT/bench_${tag}.log" 2>&1
  local rc=$?
  echo ">>> $tag rc=$rc ; decode/throughput lines:"
  grep -aiE 'Benchmark ?|Prefill|Decode|median|latency|throughput|batch_size|radix|disable_radix' "$OUT/bench_${tag}.log" | tail -25
  teardown; gpu_idle_wait; return $rc
}

# DS radix-on (production fixture)
run_bench ds --enable-double-sparsity --double-sparsity-config "$DS_CFG_TABLEFREE" \
  --double-sparsity-radix-fixture-artifact "$FIX"
DS_RC=$?
# DSA-native (radix-on default)
run_bench dsa
DSA_RC=$?
echo "=== tax guard runs done (ds_rc=$DS_RC dsa_rc=$DSA_RC) $(date -u +%H:%M:%SZ) ==="
