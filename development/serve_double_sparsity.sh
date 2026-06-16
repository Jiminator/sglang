#!/usr/bin/env bash
# Reference SGLang server invocation for the standalone Double Sparsity path.
#
# Mirrors development/serve_native_nsa.sh but adds --enable-double-sparsity
# and --double-sparsity-config. Targets DeepSeek-V3.2 (FP8) on a single H200
# node, 8-way TP, page=64. Double Sparsity and HiSparse are mutually
# exclusive: enabling both at startup is rejected by the launch validator.
#
# Locked operating point — these dense-prefill / sparse-decode flags MUST
# agree across this script and development/serve_native_nsa.sh so the
# DS-vs-baseline comparison differs only by Double Sparsity enablement and
# the radix-cache gate:
#   --kv-cache-dtype fp8_e4m3
#   --dsa-prefill-backend flashmla_kv
#   --dsa-decode-backend  flashmla_kv
#   --disable-overlap-schedule
#   --disable-piecewise-cuda-graph
#   --page-size 64
#
# DEV-ONLY launcher. The DS startup validator refuses --enable-double-sparsity
# The page-table adapter is now in place; production deployments boot
# without any DS dev-override environment variable. The previous
# adapter-bypass and placeholder-selector overrides were removed;
# production must call bind_runtime_data with a real channel mask +
# page signature table before serving.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db}"
PORT="${PORT:-30000}"
# Bind address. Default 127.0.0.1 (localhost-only, the sglang default). Set
# HOST=0.0.0.0 to make this DS server reachable from another node (symmetry
# with serve_native_nsa.sh; used when a paired DS-vs-baseline quality run
# spans two nodes). The dense-prefill/sparse-decode flags below are unchanged
# by this knob.
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TP_SIZE:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
PAGE_SIZE="${PAGE_SIZE:-64}"
CHANNEL_MASK_PATH="${CHANNEL_MASK_PATH:-/cluster-storage/models/glm51-fp8-channel-mask-s256.safetensors}"
TOP_K="${TOP_K:-2048}"
DEVICE_BUFFER_SIZE="${DEVICE_BUFFER_SIZE:-4096}"
# Double Sparsity selects table-free: absorbed-latent scoring reads the resident
# MLA latent directly, so there is NO per-rank TokenLabelTable and the ~6.8 GB/rank
# it used to consume is freed to the KV pool. That is why mem_fraction runs at 0.8
# with sustained headroom (validated GLM-5.1-FP8: max_total_num_tokens 504,640 /
# bs cap 109). Lower it only to serve a deliberately smaller pool.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
# Right-sized serving envelope: cap running requests and captured CUDA-graph
# batch sizes at the workload concurrency ceiling so the graph-capture set does
# not reserve memory for batches the workload never runs. Part of the 0.8 served
# point. Raise for a larger concurrency envelope.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-64}"
# Fixed server seed so a paired DS-vs-baseline SLO comparison is seed-matched
# (the only intended column differences are DS enablement/config + DS mem fraction).
RANDOM_SEED="${RANDOM_SEED:-20260607}"

LOG_DIR="${LOG_DIR:-$(pwd)/development/logs}"
mkdir -p "${LOG_DIR}"

# Flag-gated non-learned selector variants (config-borne so they reach the TP
# workers). Defaults reproduce the production scorer (byte-identical).
SCORER_NORM="${SCORER_NORM:-off}"                          # off (the only supported mode)
HEAD_AGG="${HEAD_AGG:-max}"                                # max | mean
ANCHOR_MODE="${ANCHOR_MODE:-off}"                          # off | recency | global | strided
ANCHOR_BUDGET="${ANCHOR_BUDGET:-0}"
# Fail-closed NIAH recall-oracle diagnostic (config-borne so it reaches TP
# workers). RECALL_ORACLE=1 emits "recall_oracle": true and forces eager
# (--disable-cuda-graph), which the validator requires.
RECALL_ORACLE="${RECALL_ORACLE:-0}"                        # 0 | 1
if [[ "${RECALL_ORACLE}" == "1" ]]; then RECALL_ORACLE_JSON=true; else RECALL_ORACLE_JSON=false; fi
# Opt-in lifted-budget decode (graph-safe production path). LIFTED_BUDGET=1 emits
# "enable_lifted_budget_decode": true + a wider "lifted_budget_top_k" (must be
# > top_k and a multiple of 128). The path runs UNDER CUDA graph (alloc-free
# dequantize_k_cache_paged_out + fixed-shape compact builder + preallocated
# DSGraphState scratch), so it no longer forces --disable-cuda-graph. Off => the
# config emits the default-off pair (lifted_budget_top_k must be 0 when off, else
# the config fails closed).
LIFTED_BUDGET="${LIFTED_BUDGET:-0}"                        # 0 | 1
if [[ "${LIFTED_BUDGET}" == "1" ]]; then
  LIFTED_BUDGET_JSON=true
  LIFTED_BUDGET_TOP_K="${LIFTED_BUDGET_TOP_K:-4096}"
else
  LIFTED_BUDGET_JSON=false
  LIFTED_BUDGET_TOP_K=0
fi
DS_CONFIG=$(printf '{"top_k": %s, "page_size": %s, "channel_mask_path": "%s", "device_buffer_size": %s, "scorer_norm": "%s", "head_agg": "%s", "anchor_mode": "%s", "anchor_budget": %s, "recall_oracle": %s, "enable_lifted_budget_decode": %s, "lifted_budget_top_k": %s}' \
  "${TOP_K}" "${PAGE_SIZE}" "${CHANNEL_MASK_PATH}" "${DEVICE_BUFFER_SIZE}" "${SCORER_NORM}" "${HEAD_AGG}" "${ANCHOR_MODE}" "${ANCHOR_BUDGET}" "${RECALL_ORACLE_JSON}" "${LIFTED_BUDGET_JSON}" "${LIFTED_BUDGET_TOP_K}")
echo ">>> effective double_sparsity_config = ${DS_CONFIG}"

# Eager is required only for the recall-oracle diagnostic (it host-syncs). As of R9
# all non-learned scorer/anchor variants are graph-safe, and the lifted-budget
# decode path is now graph-safe too (fixed-shape compact builder + alloc-free out=
# dequant + preallocated scratch), so it runs UNDER CUDA graph. Auto-add
# --disable-cuda-graph only for RECALL_ORACLE.
CUDA_GRAPH_ARGS=()
if [[ "${RECALL_ORACLE}" == "1" ]]; then
  CUDA_GRAPH_ARGS=(--disable-cuda-graph)
  echo ">>> recall_oracle diagnostic requires eager: adding --disable-cuda-graph"
fi

# Radix cache: radix-on is the VALIDATED DEFAULT for this op-point under the DEC-12
# PRODUCTION-REPRESENTATIVE-REUSE edge contract. RADIX_FIXTURE_ARTIFACT defaults to the
# committed table-free fixtures-passed state file next to this script (written by
# validator.write_radix_fixture_state once the radix correctness probes passed on this
# exact config). The authorization criterion is value-equivalence, not cold/warm bit-
# identity: recall@2048 radix-on-vs-off within +/-0.5pp overall and per length, cross-rank
# selection identity, no dense fallback, and a clean edge probe evaluated at PRODUCTION
# reuse (boundary page-aligned reuse and the partial-page hit at the production upper bound
# cached ~2752 ~63% — each within +/-0.5pp of a length-matched radix-OFF control — plus a
# clean eviction/recompute). NOTE: NEAR-FULL reuse (cached ~4288, >=~98%) is OUT OF this
# contract — it is a documented value-affecting characterization (+1.57pp, R27 95% CI
# [1.28,1.87]), NOT a gate input; production workloads operate at ~55% prefix hit. The
# launcher passes --double-sparsity-radix-fixture-artifact and drops --disable-radix-cache;
# the validator re-verifies the schema + that the state matches THIS boot's config and only
# then permits radix-on (fail-closed: a config that does not match the artifact fingerprint
# is refused). For a different config, regenerate the artifact; set RADIX_FIXTURE_ARTIFACT=""
# to serve radix-off. (Legacy label-capture / FP8-scale fixtures and the superseded bit-
# identity fixture no longer authorize table-free DS — rejected.)
# SGLANG_DS_RADIX_OVERRIDE=1 is a developer-only escape hatch that enables radix-on WITHOUT
# an artifact (used solely to (re)run the radix fixtures, which need radix reuse).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RADIX_FIXTURE_ARTIFACT="${RADIX_FIXTURE_ARTIFACT-${SCRIPT_DIR}/serve_double_sparsity_radix_fixture.json}"
if [[ -n "${RADIX_FIXTURE_ARTIFACT}" ]]; then
  RADIX_ARGS=(--double-sparsity-radix-fixture-artifact "${RADIX_FIXTURE_ARTIFACT}")
elif [[ "${SGLANG_DS_RADIX_OVERRIDE:-}" == "1" ]]; then
  RADIX_ARGS=()  # developer-only radix-on for fixture runs (see note above)
else
  RADIX_ARGS=(--disable-radix-cache)
fi

# Custom all-reduce: the PROPER GLM op-point keeps it ON (default). The R10 boot
# failure (`custom_all_reduce.cuh: CUDA error: invalid argument` at CUDA-graph
# capture) was NOT the v2 path — it was an inherited
# `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (CUDA VMM, from the calibration OOM):
# `cudaIpcGetMemHandle` does not work on VMM memory, so custom-all-reduce-v2's
# graph-input registration fails. Fix = do NOT set expandable_segments for serving
# (guarded below). DISABLE_CUSTOM_ALL_REDUCE=1 remains ONLY as a degraded NCCL-fallback
# diagnostic (the AC-11 comparator refuses that op-point for publication).
if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
  CUSTOM_AR_ARGS=(--disable-custom-all-reduce)
else
  CUSTOM_AR_ARGS=()
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" != "1" && "${PYTORCH_CUDA_ALLOC_CONF:-}" == *expandable_segments* ]]; then
  echo ">>> WARN: stripping PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}' (expandable_segments breaks custom all-reduce at GLM TP=8); custom all-reduce stays ON."
  unset PYTORCH_CUDA_ALLOC_CONF
fi

LOG_FILE="${LOG_DIR}/server_double_sparsity_$(date +%Y%m%d-%H%M%S).log"
echo ">>> Starting Double Sparsity server (standalone)"
echo "    model            = ${MODEL_PATH}"
echo "    host             = ${HOST}"
echo "    port             = ${PORT}"
echo "    tp_size          = ${TP_SIZE}"
echo "    kv_cache         = ${KV_CACHE_DTYPE}"
echo "    page_size        = ${PAGE_SIZE}"
echo "    channel_mask     = ${CHANNEL_MASK_PATH}"
echo "    top_k            = ${TOP_K}"
echo "    device_buffer    = ${DEVICE_BUFFER_SIZE}"
echo "    mem_fraction     = ${MEM_FRACTION_STATIC}"
echo "    max_running_reqs = ${MAX_RUNNING_REQUESTS}"
echo "    cuda_graph_max_bs= ${CUDA_GRAPH_MAX_BS}"
echo "    recall_oracle    = ${RECALL_ORACLE} (eager: ${#CUDA_GRAPH_ARGS[@]} cuda-graph arg(s))"
echo "    radix_cache      = $([[ -n "${RADIX_FIXTURE_ARTIFACT}" ]] && echo "ENABLED (fixture artifact: ${RADIX_FIXTURE_ARTIFACT})" || echo "disabled (no fixture artifact)")"
echo "    log              = ${LOG_FILE}"
echo ""
echo "    *** RECALL-MODE OPT-IN — NOT A THROUGHPUT DEFAULT ***"
echo "    Double Sparsity is a reversible, DEFAULT-OFF opt-in. On the GLM-5.1 client"
echo "    workload it does NOT meet the throughput SLO (locked sweep: decode-TPS ~23/17/17"
echo "    tok/s << 30 at conc 16/32/64; see development/loop8/task9_gate_results.md). The"
echo "    served default is native DSA. Enable DS only for long-context recall scenarios"
echo "    where the trained indexer underperforms — not to raise standard-workload throughput."

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --page-size "${PAGE_SIZE}" \
  --dsa-prefill-backend flashmla_kv \
  --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule \
  --disable-piecewise-cuda-graph \
  --enable-double-sparsity \
  --double-sparsity-config "${DS_CONFIG}" \
  --random-seed "${RANDOM_SEED}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  "${CUSTOM_AR_ARGS[@]}" \
  `# Radix cache: --disable-radix-cache (default radix-off) or the` \
  `# fixtures-passed artifact path (radix-on), selected above.` \
  "${RADIX_ARGS[@]}" \
  --trust-remote-code \
  ${EXTRA_SERVER_ARGS:-} \
  2>&1 | tee "${LOG_FILE}"
# The DS validator gates radix cache OFF until the TABLE-FREE radix fixture has
# been recorded as passing for this configuration. To enable radix-on:
#   1. Boot DS radix-on (SGLANG_DS_RADIX_OVERRIDE=1, dev-only) and run the radix
#      correctness probes on the served workload (value equivalence, not
#      bit-identity): recall@2048 radix-on-vs-off within the +/-0.5pp bar overall
#      and per length, cross-rank selection identity within each path, an
#      eviction/partial-hit/page-boundary edge probe (page 64, no stale-slot reuse),
#      no dense fallback, and the cold/warm selection flips characterized as
#      value-neutral near-cutoff reshuffling.
#   2. On ALL passing, write a table-free fixtures-passed state file with
#      validator.write_radix_fixture_state(...) and point RADIX_FIXTURE_ARTIFACT at
#      it; the validator re-verifies the schema + that the state matches this config
#      and permits radix-on (without the dev override).
# The legacy label-capture / FP8-scale manual fixtures
# (test_dsv32_radix_label_capture_fixture.py, test_dsv32_fp8_scale_stability.py)
# prove a deleted invariant (the materialized TokenLabelTable), and the superseded
# bit-identity fixture (tablefree_v1) gated the wrong criterion; NEITHER authorizes
# table-free DS.
