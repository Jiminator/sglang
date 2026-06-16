#!/usr/bin/env bash
# R27 GATE C (DEC-12) — PRODUCTION-REPRESENTATIVE-REUSE edge probe (length 4096, page 64).
#
# Re-authors the R26 edge gate to owner DEC-12: the PASS/FAIL contract is evaluated ONLY
# on production-representative reuse (cached <= ~2752, ~63%). Near-full reuse (cached ~4288,
# ~98%) is run as a SEPARATE out-of-contract value-affecting characterization arm.
#
# B1 FAIL-CLOSED FIX (Codex R26): the R26 runner exited $((COLD_RC||BND_RC||PRT_RC||EVC_RC))
# and IGNORED the analyzer rc, so a FAIL analyzer could still exit 0. This R27 runner
# INCLUDES AN_RC in the final exit. (Proof: SELFTEST=1 runs analyze on a fixed FAIL fixture
# and asserts the runner would exit nonzero.)
#
# Reuses the PROVEN cache-engaging harness (R26 edge driver + r25_env op-point + niah oracle
# path + same-seed warmup). Records per-request cached_tokens so cache engagement is PROVEN
# from raw data per arm. Op-point (all boots): GLM-5.1-FP8 TP=8 fp8_e4m3 page64 mem0.8
# max-running64 EAGER (--disable-cuda-graph), DS table-free recall_oracle=true, top_k 2048,
# page_size 64, channel_mask glm51-fp8-channel-mask-s256, device_buffer_size 4096, anchor off.
# NEVER expandable_segments. ONE server at a time; teardown to ~0 MiB between arms; kill all.
#
#   cold arm -> radix-OFF boot (--disable-radix-cache), cached MUST be 0
#   on arms  -> radix-ON boot (SGLANG_DS_RADIX_OVERRIDE=1, no --disable-radix-cache):
#               boundary / partial@production / nearfull / eviction (one boot each, radix-ON)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../20260616_r25/radix_authorization/r25_env.sh"
cd "$REPO"

OUTDIR="$HERE/gate_c"; mkdir -p "$OUTDIR/logs"

LENGTH=4096
NUM="${1:-144}"
DECODE_STEPS=4
SEED_BASE=1000
WARMUP_LENGTH=1024
PARTIAL_PREFIX_TOKENS=2752
mkdir -p "$REPO/.sglang_ds_oracle"

# ---- B1 fail-closed SELFTEST: prove AN_RC propagates to the runner exit ----
# Run the analyzer on a synthetic FAIL fixture; assert it exits nonzero so the
# guard `AN_RC != 0` in the final exit cannot be defeated.
if [[ "${SELFTEST:-0}" == "1" ]]; then
  ST="$OUTDIR/selftest"; rm -rf "$ST"; mkdir -p "$ST"
  python "$HERE/make_selftest_fixture.py" --dir "$ST" --length "$LENGTH"
  python "$HERE/analyze_edge.py" --dir "$ST" --length "$LENGTH" --page-size 64 \
    --max-delta-pp 0.5 --min-needles 128 --production-cached-max "$PARTIAL_PREFIX_TOKENS" \
    --out "$ST/verdict.json" --table-out "$ST/table.jsonl" > "$ST/analyze.log" 2>&1
  ST_AN_RC=$?
  ST_STATUS=$(python3 -c "import json;print(json.load(open('$ST/verdict.json'))['status'])" 2>/dev/null || echo ERR)
  # Emulate the EXACT final exit expression with synthetic zero arm rcs.
  ST_COLD=0; ST_CPRT=0; ST_BND=0; ST_PRT=0; ST_NFL=0; ST_EVC=0
  ST_RUNNER_EXIT=$(( ST_COLD != 0 || ST_CPRT != 0 || ST_BND != 0 || ST_PRT != 0 || ST_NFL != 0 || ST_EVC != 0 || ST_AN_RC != 0 ))
  echo "[selftest] analyzer status=$ST_STATUS analyzer_rc=$ST_AN_RC -> runner_would_exit=$ST_RUNNER_EXIT"
  if [[ "$ST_STATUS" == "FAIL" && "$ST_AN_RC" != "0" && "$ST_RUNNER_EXIT" != "0" ]]; then
    echo "[selftest] PASS: a FAIL analyzer forces the runner to exit nonzero (fail-closed)."
    exit 0
  fi
  echo "[selftest] !! FAIL: fail-closed guard NOT proven (status=$ST_STATUS an_rc=$ST_AN_RC exit=$ST_RUNNER_EXIT)"
  exit 1
fi

echo "[gate_c] expandable_segments=$([[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] && echo UNSET || echo "${PYTORCH_CUDA_ALLOC_CONF}")"

run_arm() {  # $1=arm  $2=boot-tag(off|on)  $3..=extra launch args
  local arm="$1"; local boot="$2"; shift 2
  local slog="$OUTDIR/logs/serve_${arm}.log"
  local sink="$OUTDIR/sink_${arm}.jsonl"
  local idx="$OUTDIR/index_${arm}.jsonl"
  teardown
  : > "$DEFAULT_SINK"
  echo "=== boot DS ($boot, arm=$arm) eager recall_oracle $(date -u +%H:%M:%SZ) ==="
  local args=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_RECALL" --disable-cuda-graph "$@")
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local rc_wait=0
  ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != "0" ]]; then
    echo "!! boot FAIL ($arm rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
    teardown; gpu_idle_wait; return 10
  fi
  echo ">>> server ready ($arm). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || true
  echo "=== edge sweep arm=$arm L${LENGTH} num=${NUM} $(date -u +%H:%M:%SZ) ==="
  DS_TOKENIZER_FILE="$GLM/tokenizer.json" \
  python "$HERE/edge_dec12_driver.py" \
    --arm "$arm" --length "$LENGTH" --num "$NUM" --decode-steps "$DECODE_STEPS" \
    --seed-base "$SEED_BASE" --warmup-length "$WARMUP_LENGTH" \
    --partial-prefix-tokens "$PARTIAL_PREFIX_TOKENS" \
    --out "$idx" 2>&1 | tee "$OUTDIR/logs/sweep_${arm}.log"
  local sweep_rc=${PIPESTATUS[0]}
  teardown
  cp "$DEFAULT_SINK" "$sink"
  echo ">>> sweep ($arm) rc=$sweep_rc ; sink=$sink ($(wc -l < "$sink") records); index=$idx ($(wc -l < "$idx") reqs)"
  gpu_idle_wait
  return $sweep_rc
}

# Single-arm mode: RUN_ARM=<arm> runs exactly ONE arm (one foreground server boot+sweep+
# teardown) and exits with that arm's sweep rc. The orchestrator drives the 5 arms as 5
# sequential foreground calls (each ~340s) so no single Bash call exceeds the tool timeout.
# cold is radix-OFF; all on-arms set SGLANG_DS_RADIX_OVERRIDE=1.
if [[ -n "${RUN_ARM:-}" ]]; then
  alog="$OUTDIR/logs/run_${RUN_ARM}.log"; exec > >(tee "$alog") 2>&1
  echo "=== R27 GATE C DEC-12 single arm=$RUN_ARM start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
  if [[ "$RUN_ARM" == "cold" || "$RUN_ARM" == "cold_partial" ]]; then
    unset SGLANG_DS_RADIX_OVERRIDE || true
    run_arm "$RUN_ARM" off --disable-radix-cache
  else
    export SGLANG_DS_RADIX_OVERRIDE=1
    run_arm "$RUN_ARM" on
  fi
  rc=$?
  echo "=== arm $RUN_ARM done rc=$rc $(date -u +%H:%M:%SZ) ==="
  echo "=== nvidia-smi after $RUN_ARM ==="
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  exit $rc
fi

# Analyze-only mode: ANALYZE_ONLY=1 runs the analyzer over already-collected arms.
if [[ "${ANALYZE_ONLY:-0}" != "1" ]]; then
  LOG="$OUTDIR/logs/stage.log"; exec > >(tee "$LOG") 2>&1
  echo "=== R27 GATE C DEC-12 production-reuse edge start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
  # (1) COLD/FRESH controls — radix-OFF boot; cached_tokens MUST be 0.
  unset SGLANG_DS_RADIX_OVERRIDE || true
  run_arm cold off --disable-radix-cache;         COLD_RC=$?
  run_arm cold_partial off --disable-radix-cache; CPRT_RC=$?
  # (2) ON arms — radix-ON (SGLANG_DS_RADIX_OVERRIDE=1, no --disable-radix-cache).
  export SGLANG_DS_RADIX_OVERRIDE=1
  run_arm boundary on; BND_RC=$?
  run_arm partial on;  PRT_RC=$?
  run_arm nearfull on; NFL_RC=$?
  run_arm evict on;    EVC_RC=$?
  unset SGLANG_DS_RADIX_OVERRIDE || true
  echo "=== arms done (cold=$COLD_RC cprt=$CPRT_RC bnd=$BND_RC prt=$PRT_RC nfl=$NFL_RC evc=$EVC_RC) $(date -u +%H:%M:%SZ) ==="
else
  COLD_RC=0; CPRT_RC=0; BND_RC=0; PRT_RC=0; NFL_RC=0; EVC_RC=0
  LOG="$OUTDIR/logs/analyze_only.log"; exec > >(tee "$LOG") 2>&1
fi

# (3) analyze -> gate_c_edge_verdict.json
echo "=== analyze $(date -u +%H:%M:%SZ) ==="
python "$HERE/analyze_edge.py" \
  --dir "$OUTDIR" --length "$LENGTH" --page-size 64 --max-delta-pp 0.5 --min-needles 128 \
  --production-cached-max "$PARTIAL_PREFIX_TOKENS" \
  --out "$HERE/gate_c_edge_verdict.json" \
  --table-out "$HERE/gate_c_per_needle_table.jsonl" 2>&1 | tee "$OUTDIR/logs/analyze.log"
AN_RC=${PIPESTATUS[0]}

echo "=== final nvidia-smi ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "=== R27 GATE C done (analyze_rc=$AN_RC) $(date -u +%H:%M:%SZ) ==="
# B1 FIX: AN_RC IS INCLUDED — a FAIL analyzer (status!=PASS) forces a nonzero runner exit.
exit $(( COLD_RC != 0 || CPRT_RC != 0 || BND_RC != 0 || PRT_RC != 0 || NFL_RC != 0 || EVC_RC != 0 || AN_RC != 0 ))
