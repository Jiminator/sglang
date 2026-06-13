#!/usr/bin/env bash
# Loop 11 task0: max-stable-fraction / capacity probe matrix.
# Each probe = boot + capacity readout + graph-capture check + short serve smoke.
# Axes: table variant {fp16, int8, tablefree-mock} x indexer {on, off} x envelope
# {default, rs (=--max-running-requests 64 --cuda-graph-max-bs 64), rs16k (rs +
# --context-length 16384)} at selected mem fractions. Probe-only env hooks
# (SGLANG_DS_PROBE_TABLE_TOKENS, SGLANG_DS_PROBE_SKIP_INDEXER) are dev-only
# (see probe_hacks.patch) and are NEVER set on baseline/served runs.
# A probe that fails to boot is a RESULT row (BOOT_FAIL + reason), not an error.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/stage_task0.log"; exec > >(tee "$LOG") 2>&1
TSV="$HERE/probes.tsv"
echo -e "probe\tfraction\tvariant\tindexer\tenvelope\tstatus\tmax_total_num_tokens\tbs_cap_4608\ttable_GB\tgraph_capture\tsmoke\tnote" > "$TSV"
mkdir -p "$HERE/probe_logs"
echo "=== TASK0 probes start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

DS_CONFIG_INT8="${DS_CONFIG/\"signature_dtype\": \"fp16\"/\"signature_dtype\": \"int8\"}"

smoke() { # -> "OK:<text-prefix>" or "FAIL:<reason>"
  local resp
  resp=$(curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" \
    -H 'Content-Type: application/json' \
    -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 24, "temperature": 0}}' 2>/dev/null)
  python3 - <<'PYEOF' "$resp"
import json, sys
raw = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    d = json.loads(raw)
    t = (d.get("text") or "").strip().replace("\t", " ").replace("\n", " ")
    print(f"OK:{t[:40]}" if t else "FAIL:empty_text")
except Exception:
    print(f"FAIL:bad_response:{raw[:60]}")
PYEOF
}

probe() {
  local name="$1" frac="$2" variant="$3" indexer="$4" envelope="$5"
  echo "=== probe $name frac=$frac variant=$variant indexer=$indexer envelope=$envelope $(date -u +%H:%M:%SZ) ==="
  teardown
  local args=("${COMMON_ARGS[@]}" --mem-fraction-static "$frac")
  case "$variant" in
    fp16)      args+=(--enable-double-sparsity --double-sparsity-config "$DS_CONFIG") ;;
    int8)      args+=(--enable-double-sparsity --double-sparsity-config "$DS_CONFIG_INT8") ;;
    tablefree) args+=(--enable-double-sparsity --double-sparsity-config "$DS_CONFIG") ;;
    dsa)       : ;;
  esac
  case "$envelope" in
    default) : ;;
    rs)    args+=(--max-running-requests 64 --cuda-graph-max-bs 64) ;;
    rs16k) args+=(--max-running-requests 64 --cuda-graph-max-bs 64 --context-length 16384) ;;
  esac
  local envs=()
  [[ "$variant" == "tablefree" ]] && envs+=("SGLANG_DS_PROBE_TABLE_TOKENS=8192")
  [[ "$indexer" == "off" ]] && envs+=("SGLANG_DS_PROBE_SKIP_INDEXER=1")
  local slog="$HERE/probe_logs/${name}_serve.log"
  env "${envs[@]}" python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local status="OK" cap=0 bs=0 table_gb="-" gcap="-" smk="-" note="-"
  if ! wait_ready; then
    status="BOOT_FAIL"
    if grep -aq "Capture cuda graph failed" "$slog"; then note="graph_capture_oom"
    elif grep -aq "CUDA out of memory" "$slog"; then note="cuda_oom"
    elif grep -aq "Not enough memory" "$slog"; then note="mem_check_refused"
    else note="timeout_or_other:$(grep -aE 'Error|error|FAIL' "$slog" | tail -1 | cut -c1-60 | tr '\t' ' ')"
    fi
  else
    cap=$(max_batch_from_server); bs=$(( cap / 4608 ))
    gcap=$(grep -ac "Capture cuda graph end" "$slog" || true)
    [[ "$gcap" -ge 1 ]] && gcap="yes" || gcap="NO"
    smk=$(smoke)
    table_gb=$(grep -aoE "token_label_table: [0-9.]+ GB/rank" "$slog" | head -1 | grep -oE "[0-9.]+" | head -1 || true)
    [[ -z "$table_gb" ]] && table_gb="-"
  fi
  # Extract only the proof lines (TP0 + the table line) — NO head cap, so the capture-end and
  # available_gpu_mem proof lines can never be truncated (R0 used `head -50` which, across 8 ranks,
  # cut them; fixed R1 — see build_task0_matrix.py for the canonical durable extractor).
  grep -aE "TP0\] (Load weight end|KV Cache is allocated|Memory pool end|Capture cuda graph (begin|end)|max_total_num_tokens)|token_label_table:" "$slog" > "$HERE/probe_logs/${name}_fields.txt" || true
  echo -e "${name}\t${frac}\t${variant}\t${indexer}\t${envelope}\t${status}\t${cap}\t${bs}\t${table_gb}\t${gcap}\t${smk}\t${note}" >> "$TSV"
  echo ">>> $name => status=$status cap=$cap bs=$bs table_gb=$table_gb graph=$gcap smoke=$smk note=$note"
  teardown
}

# --- probe list (decision-priority order; ~14 boots) ---
probe p01_fp16_on_def_070 0.7  fp16 on  default   # accounting anchor at HEAD
probe p02_fp16_on_def_075 0.75 fp16 on  default   # fp16 ceiling
probe p03_fp16_on_def_080 0.8  fp16 on  default   # expected FAIL (AC-1.1 negative evidence)
probe p04_fp16_on_rs_075  0.75 fp16 on  rs        # envelope effect alone
probe p05_fp16_on_rs16k_080 0.8 fp16 on rs16k     # strongest envelope on fp16
probe p06_int8_on_def_075 0.75 int8 on  default
probe p07_int8_on_def_080 0.8  int8 on  default   # task4 key row
probe p08_int8_on_rs_080  0.8  int8 on  rs
probe p09_int8_on_rs16k_080 0.8 int8 on rs16k     # M1 best-config preview
probe p10_int8_off_rs16k_080 0.8 int8 off rs16k   # M1 + indexer-gate preview
probe p11_tf_on_def_080   0.8  tablefree on default   # M2 core
probe p12_tf_off_rs_080   0.8  tablefree off rs       # M2 + task3 preview
probe p13_tf_off_rs16k_085 0.85 tablefree off rs16k   # endgame ceiling
probe p14_fp16_off_def_070 0.7 fp16 off default   # task3 alone: tokens at same fraction

echo "=== TASK0 probes done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cat "$TSV"
