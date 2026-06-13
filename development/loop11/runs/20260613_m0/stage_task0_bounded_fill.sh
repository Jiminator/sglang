#!/usr/bin/env bash
# Loop 11 task0 (R3): complete the bounded right-sized matrix. Bounded fail_closed
# (selector_width_buckets=[4608], selector_width_overflow_policy=fail_closed) rs sweeps for ALL
# six {fp16,int8,table-free} × {indexer on/off} configs, ascending mem_fraction to the boot
# ceiling (highest-pass + first-fail). Reuses the R2 bnd_* rows (skip-if-present). The committed
# bounded-width FEATURE is real; the table-free/indexer-off MOCKS are dev-only probe hooks
# (probe_hacks.patch), reverted before the round commit.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/stage_task0_bounded_fill.log"; exec > >(tee "$LOG") 2>&1
LOGDIR="$HERE/probe_logs"; mkdir -p "$LOGDIR"
TSV="$HERE/probes_bounded_fill.tsv"
echo -e "probe\tfraction\tvariant\tindexer\tenvelope\tpolicy\tbuckets\tstatus\tmax_total_num_tokens\tbs_cap_4608\ttable_GB\tready_GB\tgraph_capture\tsmoke\tnote" > "$TSV"
echo "=== TASK0 BOUNDED FILL start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

DS_CONFIG_INT8="${DS_CONFIG/\"signature_dtype\": \"fp16\"/\"signature_dtype\": \"int8\"}"
GRID="0.75 0.80 0.85 0.90 0.95"
inject_bounded() { echo "${1%\}}, \"selector_width_buckets\": [4608], \"selector_width_overflow_policy\": \"fail_closed\"}"; }

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

existing_log() { ls "$LOGDIR"/bnd_"${1}"_"${2}"_rs_"${3}"_serve.log 2>/dev/null | head -1; }

run_probe() { # variant indexer frac
  local variant="$1" idx="$2" frac="$3"; local f3="${frac//./}"
  local name="bnd_${variant}_${idx}_rs_${f3}"
  echo "=== $name (frac=$frac) $(date -u +%H:%M:%SZ) ==="
  teardown
  local base; [[ "$variant" == "int8" ]] && base="$DS_CONFIG_INT8" || base="$DS_CONFIG"
  local cfg; cfg="$(inject_bounded "$base")"
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
    grep -a "DS selector-width graph variants" "$slog" | head -1 >> "$LOGDIR/${name}_args.txt" || true
  fi
  echo -e "${name}\t${frac}\t${variant}\t${idx}\trs\tfail_closed\t4608\t${status}\t${cap}\t${bs}\t${table_gb}\t${ready}\t${gcap}\t${smk}\t${note}" >> "$TSV"
  echo ">>> $name status=$status cap=$cap bs=$bs table_gb=$table_gb ready=$ready graph=$gcap smoke=$smk note=$note"
  teardown
  PROBE_STATUS="$status"
}

for variant in fp16 int8 tf; do
  for idx in on off; do
    echo "##### BOUNDED CONFIG ${variant}_${idx}_rs #####"
    for frac in $GRID; do
      f3="${frac//./}"
      prior="$(existing_log "$variant" "$idx" "$f3")"
      if [[ -n "$prior" ]]; then
        # Reuse R2 row. Determine pass/fail to drive early-stop.
        if grep -aq "The server is fired up" "$prior" && grep -aq "Capture cuda graph end" "$prior"; then
          echo ">>> reuse $(basename "$prior") (R2 PASS)"; continue
        else
          echo ">>> reuse $(basename "$prior") (R2 FAIL) — ceiling reached, stop ${variant}_${idx}"; break
        fi
      fi
      run_probe "$variant" "$idx" "$frac"
      [[ "$PROBE_STATUS" != "OK" ]] && { echo ">>> ceiling ${variant}_${idx}_rs: first-fail $frac"; break; }
    done
  done
done

echo "=== TASK0 BOUNDED FILL done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cat "$TSV"
