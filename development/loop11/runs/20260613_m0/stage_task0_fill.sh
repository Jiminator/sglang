#!/usr/bin/env bash
# Loop 11 task0 (R1): complete the explicit 12-config matrix with a per-config mem_fraction
# sweep to the boot/capture/smoke CEILING. Matrix = {fp16,int8,tablefree-mock} × {indexer on/off}
# × {default, right-sized envelope (--max-running-requests 64 --cuda-graph-max-bs 64)}.
# Ascending fraction grid with early-stop at the first boot/capture/smoke FAILURE.
# Reuses R0 serve.logs (skip-if-present); runs only the missing (config, fraction) cells.
# NB: the matrix measures the BOOT ceiling (an upper bound), NOT the sustained-stable served
# fraction (that comes from the task4/M2 ladders under real load).
# Probe-only env hooks (probe_hacks.patch) are dev-only and reverted before the round commit.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/stage_task0_fill.log"; exec > >(tee "$LOG") 2>&1
LOGDIR="$HERE/probe_logs"; mkdir -p "$LOGDIR"
FILL_TSV="$HERE/probes_fill.tsv"
echo -e "probe\tfraction\tvariant\tindexer\tenvelope\tstatus\tmax_total_num_tokens\tbs_cap_4608\ttable_GB\tready_GB\tgraph_capture\tsmoke\tnote" > "$FILL_TSV"
echo "=== TASK0 FILL start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

DS_CONFIG_INT8="${DS_CONFIG/\"signature_dtype\": \"fp16\"/\"signature_dtype\": \"int8\"}"
# Sweep the decision-relevant ceiling region: 0.75 (where the bs>=64 crossing sits) up to 0.95,
# early-stop at first failure. R0 already established that all configs BOOT at 0.70-0.80; those
# lower reference rows are retained from R0 in the merged matrix. Per-config reuse skips re-runs.
GRID="0.75 0.80 0.85 0.90 0.95"

smoke() {
  local resp
  resp=$(curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" \
    -H 'Content-Type: application/json' \
    -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 24, "temperature": 0}}' 2>/dev/null)
  python3 - <<'PYEOF' "$resp"
import json, sys
raw = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    d = json.loads(raw)
    t = (d.get("text") or "").strip().replace("\t"," ").replace("\n"," ")
    print(f"OK:{t[:40]}" if t else "FAIL:empty_text")
except Exception:
    print(f"FAIL:bad_response:{raw[:60]}")
PYEOF
}

# returns 0 (and echoes the canonical name) if a prior serve.log exists for this (config,frac)
existing_log() {
  local variant="$1" idx="$2" env="$3" f3="$4"
  ls "$LOGDIR"/*_"${variant}"_"${idx}"_"${env}"_"${f3}"_serve.log 2>/dev/null | head -1
}

# run one probe; sets PROBE_STATUS to OK / BOOT_FAIL; writes a row + durable extract
run_probe() {
  local variant="$1" idx="$2" env="$3" frac="$4"
  local f3="${frac//./}"   # 0.85 -> 085
  local name="${variant}_${idx}_${env}_${f3}"
  echo "=== $name (frac=$frac) $(date -u +%H:%M:%SZ) ==="
  teardown
  local args=("${COMMON_ARGS[@]}" --mem-fraction-static "$frac")
  case "$variant" in
    fp16)      args+=(--enable-double-sparsity --double-sparsity-config "$DS_CONFIG") ;;
    int8)      args+=(--enable-double-sparsity --double-sparsity-config "$DS_CONFIG_INT8") ;;
    tf)        args+=(--enable-double-sparsity --double-sparsity-config "$DS_CONFIG") ;;
  esac
  [[ "$env" == "rs" ]] && args+=(--max-running-requests 64 --cuda-graph-max-bs 64)
  local envs=()
  [[ "$variant" == "tf" ]] && envs+=("SGLANG_DS_PROBE_TABLE_TOKENS=8192")
  [[ "$idx" == "off" ]] && envs+=("SGLANG_DS_PROBE_SKIP_INDEXER=1")
  local slog="$LOGDIR/${name}_serve.log"
  printf '%s ' "${envs[@]}" > "$LOGDIR/${name}_args.txt"
  printf 'python -m sglang.launch_server %s\n' "${args[*]}" >> "$LOGDIR/${name}_args.txt"
  env "${envs[@]}" python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local status="OK" cap=0 bs=0 table_gb="-" ready="-" gcap="-" smk="-" note="-"
  if ! wait_ready; then
    status="BOOT_FAIL"
    if   grep -aq "Capture cuda graph failed" "$slog"; then note="graph_capture_oom"
    elif grep -aq "CUDA out of memory"        "$slog"; then note="cuda_oom"
    elif grep -aq "Not enough memory"         "$slog"; then note="mem_check_refused"
    else note="timeout_or_other:$(grep -aiE 'error|fail' "$slog" | tail -1 | cut -c1-50 | tr '\t' ' ')"; fi
  else
    cap=$(max_batch_from_server); bs=$(( cap / 4608 ))
    grep -ac "Capture cuda graph end" "$slog" >/dev/null && \
      { [[ $(grep -ac "Capture cuda graph end" "$slog") -ge 1 ]] && gcap="yes" || gcap="NO"; }
    smk=$(smoke)
    table_gb=$(grep -aoE "token_label_table: [0-9.]+ GB" "$slog" | head -1 | grep -oE "[0-9.]+" | head -1); table_gb="${table_gb:--}"
    ready=$(grep -a "available_gpu_mem=" "$slog" | head -1 | grep -aoE "available_gpu_mem=[0-9.]+ GB" | grep -oE "[0-9.]+"); ready="${ready:--}"
  fi
  echo -e "${name}\t${frac}\t${variant}\t${idx}\t${env}\t${status}\t${cap}\t${bs}\t${table_gb}\t${ready}\t${gcap}\t${smk}\t${note}" >> "$FILL_TSV"
  echo ">>> $name status=$status cap=$cap bs=$bs table_gb=$table_gb ready=$ready graph=$gcap smoke=$smk note=$note"
  teardown
  PROBE_STATUS="$status"
}

# 12 configs; ascending sweep with early-stop at first boot/capture/smoke failure.
for variant in fp16 int8 tf; do
  for idx in on off; do
    for env in def rs; do
      echo "##### CONFIG ${variant}_${idx}_${env} #####"
      for frac in $GRID; do
        f3="${frac//./}"
        prior="$(existing_log "$variant" "$idx" "$env" "$f3")"
        if [[ -n "$prior" ]]; then
          echo ">>> reuse $(basename "$prior") for ${variant}_${idx}_${env}_${f3} (R0; passed)"
          continue
        fi
        run_probe "$variant" "$idx" "$env" "$frac"
        if [[ "$PROBE_STATUS" != "OK" ]]; then
          echo ">>> ceiling found for ${variant}_${idx}_${env}: first-fail at $frac; stop ascending."
          break
        fi
      done
    done
  done
done

echo "=== TASK0 FILL done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cat "$FILL_TSV"
