#!/usr/bin/env bash
# Loop 11 task0 (R2): bounded DS selector-width rows — measure the headroom the
# bounded-width feature buys ON TOP of the right-sized envelope, distinguishing
# bounded-right-sized from the R1 unbounded-right-sized rows.
#
# Each bounded probe sets the NEW (committed) DS config fields:
#   selector_width_buckets=[4608], selector_width_overflow_policy="fail_closed"
# so the DS selector-width graph ladder captures ONLY the 4608 width (no full
# 202752-width DS scratch). Compare ready_GB vs the R1 unbounded right-sized
# rows (default config: full_fallback, {5120, full}).
#
# One full_fallback control (buckets=[4608] → {4608, full}) isolates the
# full-width-drop delta. One long-prompt request validates the fail-closed
# guard end-to-end (a >4608-token sequence must error, not silently serve).
# Probe-only env hooks (probe_hacks.patch) cover the tf-mock / indexer-off rows
# and are dev-only; the bounded-width feature itself is committed production code.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/stage_task0_bounded.log"; exec > >(tee "$LOG") 2>&1
LOGDIR="$HERE/probe_logs"; mkdir -p "$LOGDIR"
TSV="$HERE/probes_bounded.tsv"
echo -e "probe\tfraction\tvariant\tindexer\tenvelope\tpolicy\tbuckets\tstatus\tmax_total_num_tokens\tbs_cap_4608\ttable_GB\tready_GB\tgraph_capture\tsmoke\tnote" > "$TSV"
echo "=== TASK0 BOUNDED start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

DS_CONFIG_INT8="${DS_CONFIG/\"signature_dtype\": \"fp16\"/\"signature_dtype\": \"int8\"}"
# Inject bounded-width fields into a DS config JSON (before the closing brace).
inject_bounded() { # $1=base json  $2=policy
  local base="$1" policy="$2"
  echo "${base%\}}, \"selector_width_buckets\": [4608], \"selector_width_overflow_policy\": \"${policy}\"}"
}

smoke() {
  local resp
  resp=$(curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" \
    -H 'Content-Type: application/json' \
    -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 24, "temperature": 0}}' 2>/dev/null)
  python3 - <<'PYEOF' "$resp"
import json, sys
raw = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    d = json.loads(raw); t = (d.get("text") or "").strip().replace("\t"," ").replace("\n"," ")
    print(f"OK:{t[:40]}" if t else "FAIL:empty_text")
except Exception:
    print(f"FAIL:bad_response:{raw[:60]}")
PYEOF
}

run_probe() { # name variant indexer policy frac
  local name="$1" variant="$2" idx="$3" policy="$4" frac="$5"
  echo "=== $name (variant=$variant idx=$idx policy=$policy frac=$frac) $(date -u +%H:%M:%SZ) ==="
  teardown
  local base; case "$variant" in
    fp16) base="$DS_CONFIG" ;;
    int8|tf) base="$DS_CONFIG_INT8"; [[ "$variant" == "tf" ]] && base="$DS_CONFIG" ;;
  esac
  # tf uses fp16 base (mock table caps it anyway); int8 uses int8 base.
  [[ "$variant" == "int8" ]] && base="$DS_CONFIG_INT8" || base="$DS_CONFIG"
  local cfg; cfg="$(inject_bounded "$base" "$policy")"
  local args=("${COMMON_ARGS[@]}" --mem-fraction-static "$frac"
              --max-running-requests 64 --cuda-graph-max-bs 64
              --enable-double-sparsity --double-sparsity-config "$cfg")
  local envs=()
  [[ "$variant" == "tf" ]] && envs+=("SGLANG_DS_PROBE_TABLE_TOKENS=8192")
  [[ "$idx" == "off" ]] && envs+=("SGLANG_DS_PROBE_SKIP_INDEXER=1")
  local slog="$LOGDIR/${name}_serve.log"
  printf '%s ' "${envs[@]}" > "$LOGDIR/${name}_args.txt"
  printf 'DS_CONFIG=%s\n' "$cfg" >> "$LOGDIR/${name}_args.txt"
  env "${envs[@]}" python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local status="OK" cap=0 bs=0 table_gb="-" ready="-" gcap="-" smk="-" note="-"
  if ! wait_ready; then
    status="BOOT_FAIL"
    if   grep -aq "Capture cuda graph failed" "$slog"; then note="graph_capture_oom"
    elif grep -aq "CUDA out of memory"        "$slog"; then note="cuda_oom"
    else note="other:$(grep -aiE 'error|fail' "$slog" | tail -1 | cut -c1-50 | tr '\t' ' ')"; fi
  else
    cap=$(max_batch_from_server); bs=$(( cap / 4608 ))
    [[ $(grep -ac "Capture cuda graph end" "$slog") -ge 1 ]] && gcap="yes" || gcap="NO"
    smk=$(smoke)
    table_gb=$(grep -aoE "token_label_table: [0-9.]+ GB" "$slog" | head -1 | grep -oE "[0-9.]+" | head -1); table_gb="${table_gb:--}"
    ready=$(grep -a "available_gpu_mem=" "$slog" | head -1 | grep -aoE "available_gpu_mem=[0-9.]+ GB" | grep -oE "[0-9.]+"); ready="${ready:--}"
    # confirm the bounded ladder is what the runner built
    grep -a "DS selector-width graph variants" "$slog" | head -1 >> "$LOGDIR/${name}_args.txt" || true
  fi
  echo -e "${name}\t${frac}\t${variant}\t${idx}\trs\t${policy}\t4608\t${status}\t${cap}\t${bs}\t${table_gb}\t${ready}\t${gcap}\t${smk}\t${note}" >> "$TSV"
  echo ">>> $name status=$status cap=$cap bs=$bs table_gb=$table_gb ready=$ready graph=$gcap smoke=$smk note=$note"
}

# --- bounded (fail_closed {4608}) headroom rows at decision-relevant points ---
run_probe bnd_fp16_on_rs_080   fp16 on  fail_closed 0.80
run_probe bnd_fp16_on_rs_085   fp16 on  fail_closed 0.85   # does bounded lift fp16's 0.80 ceiling?
run_probe bnd_int8_off_rs_080  int8 off fail_closed 0.80   # M1 served-config candidate
run_probe bnd_int8_off_rs_085  int8 off fail_closed 0.85
run_probe bnd_tf_off_rs_080    tf   off fail_closed 0.80   # endgame
run_probe bnd_tf_off_rs_085    tf   off fail_closed 0.85
run_probe bnd_tf_off_rs_090    tf   off fail_closed 0.90
# --- matched full_fallback control ({4608, full}) to isolate the full-width-drop delta ---
run_probe ctl_int8_off_rs_080  int8 off full_fallback 0.80

# --- fail-closed end-to-end guard: a >4608-token prompt on a bounded server must ERROR ---
echo "=== fail-closed integration check (bnd_int8_off_rs_080 server, ~6000-token prompt) $(date -u +%H:%M:%SZ) ==="
teardown
CFG_FC="$(inject_bounded "$DS_CONFIG_INT8" fail_closed)"
FCLOG="$LOGDIR/failclosed_check_serve.log"
env SGLANG_DS_PROBE_SKIP_INDEXER=1 python -m sglang.launch_server "${COMMON_ARGS[@]}" \
  --mem-fraction-static 0.80 --max-running-requests 64 --cuda-graph-max-bs 64 \
  --enable-double-sparsity --double-sparsity-config "$CFG_FC" > "$FCLOG" 2>&1 &
if wait_ready; then
  LONG=$(python3 -c "print('the quick brown fox jumps over the lazy dog. ' * 900)")
  RESP=$(curl -s --max-time 180 -X POST "http://127.0.0.1:${PORT}/generate" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys; print(json.dumps({'text': sys.argv[1], 'sampling_params': {'max_new_tokens': 8, 'temperature': 0}}))" "$LONG")" 2>/dev/null)
  echo ">>> long-prompt response (expect error / fail-closed, NOT a clean completion):"
  echo "$RESP" | head -c 400 | tee "$LOGDIR/failclosed_response.txt"
  echo ""
  grep -aiE "fail-closed|fail_closed|selector width|exceeds the largest captured" "$FCLOG" | tail -3 | tee -a "$LOGDIR/failclosed_response.txt" || true
else
  echo "!! fail-closed check server did not boot"; tail -20 "$FCLOG"
fi
teardown

echo "=== TASK0 BOUNDED done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cat "$TSV"
