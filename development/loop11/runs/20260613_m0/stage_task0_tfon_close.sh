#!/usr/bin/env bash
# Loop 11 task0 (R4): close the one open bounded ceiling. Bounded tf/on/rs passed at the R3
# grid-top (0.95) with no first-fail; sweep 0.96/0.97/0.98 (ascending, early-stop at first fail)
# to capture the first-fail row. Appends to probes_bounded_fill.tsv so build_task0_matrix.py picks
# it up. tf table-free mock is the dev-only SGLANG_DS_PROBE_TABLE_TOKENS hook (reverted before commit);
# the bounded-width feature is committed production code.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/stage_task0_tfon_close.log"; exec > >(tee "$LOG") 2>&1
LOGDIR="$HERE/probe_logs"; mkdir -p "$LOGDIR"
TSV="$HERE/probes_bounded_fill.tsv"
echo "=== TASK0 tf/on/rs CEILING CLOSE start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

inject_bounded() { echo "${1%\}}, \"selector_width_buckets\": [4608], \"selector_width_overflow_policy\": \"fail_closed\"}"; }
smoke() {
  local resp
  resp=$(curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
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

for frac in 0.96 0.97 0.98; do
  f3="${frac//./}"; name="bnd_tf_on_rs_${f3}"
  echo "=== $name (frac=$frac) $(date -u +%H:%M:%SZ) ==="
  teardown
  cfg="$(inject_bounded "$DS_CONFIG")"
  args=("${COMMON_ARGS[@]}" --mem-fraction-static "$frac" --max-running-requests 64 --cuda-graph-max-bs 64
        --enable-double-sparsity --double-sparsity-config "$cfg")
  slog="$LOGDIR/${name}_serve.log"
  printf 'SGLANG_DS_PROBE_TABLE_TOKENS=8192 DS_CONFIG=%s\n' "$cfg" > "$LOGDIR/${name}_args.txt"
  env SGLANG_DS_PROBE_TABLE_TOKENS=8192 python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  status="OK" cap=0 bs=0 table_gb="-" ready="-" gcap="-" smk="-" note="-"
  if ! wait_ready; then
    status="BOOT_FAIL"
    if   grep -aq "Capture cuda graph failed" "$slog"; then note="graph_capture_oom"
    elif grep -aq "CUDA out of memory"        "$slog"; then note="cuda_oom"
    elif grep -aiq "Failed to .*alloc"        "$slog"; then note="alloc_fail"
    else note="other:$(grep -aiE 'error|fail' "$slog" | tail -1 | cut -c1-50 | tr '\t' ' ')"; fi
  else
    cap=$(max_batch_from_server); bs=$(( cap / 4608 ))
    [[ $(grep -ac "Capture cuda graph end" "$slog") -ge 1 ]] && gcap="yes" || gcap="NO"
    smk=$(smoke)
    table_gb=$(grep -aoE "token_label_table: [0-9.]+ GB" "$slog" | head -1 | grep -oE "[0-9.]+" | head -1); table_gb="${table_gb:--}"
    ready=$(grep -a "available_gpu_mem=" "$slog" | head -1 | grep -aoE "available_gpu_mem=[0-9.]+ GB" | grep -oE "[0-9.]+"); ready="${ready:--}"
    grep -a "DS selector-width graph variants" "$slog" | head -1 >> "$LOGDIR/${name}_args.txt" || true
  fi
  echo -e "${name}\t${frac}\ttf\ton\trs\tfail_closed\t4608\t${status}\t${cap}\t${bs}\t${table_gb}\t${ready}\t${gcap}\t${smk}\t${note}" >> "$TSV"
  echo ">>> $name status=$status cap=$cap bs=$bs ready=$ready graph=$gcap smoke=$smk note=$note"
  teardown
  [[ "$status" != "OK" ]] && { echo ">>> first-fail at $frac — ceiling closed for bnd_tf_on_rs"; break; }
done
echo "=== TASK0 tf/on/rs CEILING CLOSE done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
