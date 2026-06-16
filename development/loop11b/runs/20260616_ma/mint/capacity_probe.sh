#!/usr/bin/env bash
# loop11b M-A — DS radix-on capacity + "mask serves" probe (GRAPH mode, the production
# op-point). Boots DS radix-on under the dev override with the regenerated mask, captures
# /server_info (token_capacity + the locked key set), a smoke generate, and CUDA-graph
# capture + no-TokenLabelTable evidence from the serve log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
cd "$REPO"
OUT="$HERE/probes/capacity_ds"; mkdir -p "$OUT"
LOG="$OUT/stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== loop11b DS radix-on capacity probe $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) ==="
teardown
slog="$OUT/serve_ds_radixon.log"
export SGLANG_DS_RADIX_OVERRIDE=1
args=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_TABLEFREE")
echo ">>> boot DS radix-on (override) GRAPH mode, mask=$MASK"
python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
rc=0; ready_wait "$slog" || rc=$?
if [[ "$rc" != 0 ]]; then echo "!! boot FAIL rc=$rc — tail:"; tail -n 80 "$slog"; teardown; gpu_idle_wait; exit 10; fi
echo ">>> ready. smoke=$(smoke)"
curl -sf --max-time 20 "http://127.0.0.1:${PORT}/server_info" -o "$OUT/server_info_ds.json" \
  && echo ">>> /server_info captured ($(wc -c < "$OUT/server_info_ds.json") bytes)" \
  || echo "!! /server_info FAILED"
python3 - "$OUT/server_info_ds.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
keys = ["model_path","tp_size","page_size","kv_cache_dtype","enable_double_sparsity",
        "disable_radix_cache","disable_cuda_graph","disable_custom_all_reduce",
        "mem_fraction_static","max_running_requests","cuda_graph_max_bs",
        "double_sparsity_radix_fixture_artifact"]
print("=== locked-key check (top-level server_args) ===")
for k in keys:
    print(f"{k} = {d.get(k)}")
dsc = d.get("double_sparsity_config")
print("double_sparsity_config =", dsc)
ist = d.get("internal_states") or []
print("=== capacity (internal_states[0]) ===")
if ist:
    s0 = ist[0]
    mu = s0.get("memory_usage") or {}
    print("token_capacity =", mu.get("token_capacity"))
    print("effective_max_running_requests_per_dp =", s0.get("effective_max_running_requests_per_dp"))
    tc = mu.get("token_capacity")
    mrr = s0.get("effective_max_running_requests_per_dp")
    # derived decode-batch cap at this op-point (ref only; >=64 is the AC-0.3 floor)
    print("internal_states_count =", len(ist))
else:
    print("NO internal_states in /server_info — top-level keys:", sorted(d.keys())[:40])
PY
echo "--- serve-log: cuda-graph capture / capacity / table evidence ---"
grep -aiE "capture cuda graph|cuda graph|max_total_num_tokens|max_running_requests|TokenLabelTable|token label table|radix" "$slog" | tail -25
echo "--- any TokenLabelTable allocation? (expect NONE) ---"
grep -aiE "TokenLabelTable|token label table" "$slog" && echo "!! TABLE FOUND" || echo "OK: no TokenLabelTable"
unset SGLANG_DS_RADIX_OVERRIDE || true
teardown; gpu_idle_wait
echo "=== capacity probe done $(date -u +%H:%M:%SZ) ==="
