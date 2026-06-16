#!/usr/bin/env bash
# loop11b M-A — AC-7: DSA-native default un-regressed. Boots the shipped DSA-native default
# (NO double sparsity; the DEC-1 validator change is DS-only and cannot touch this path) at
# mem 0.8 (the loop11 Case-2 AC-7 reference op-point), captures /server_info token_capacity,
# and checks it vs the loop11 reference 410560. radix-on default (DSA keeps radix cache ON).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
cd "$REPO"
OUT="$HERE/probes/ac7_dsa"; mkdir -p "$OUT"
LOG="$OUT/stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== loop11b AC-7 DSA-native capacity $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) ==="
teardown
slog="$OUT/serve_dsa.log"
unset SGLANG_DS_RADIX_OVERRIDE || true
echo ">>> boot DSA-native (no DS) GRAPH mode, mem 0.8 (AC-7 ref op-point)"
python -m sglang.launch_server "${DSA_ARGS[@]}" > "$slog" 2>&1 &
rc=0; ready_wait "$slog" || rc=$?
if [[ "$rc" != 0 ]]; then echo "!! DSA boot FAIL rc=$rc — tail:"; tail -n 60 "$slog"; teardown; gpu_idle_wait; exit 10; fi
echo ">>> ready. smoke=$(smoke)"
curl -sf --max-time 20 "http://127.0.0.1:${PORT}/server_info" -o "$OUT/server_info_dsa.json" && echo ">>> /server_info captured"
python3 - "$OUT/server_info_dsa.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("enable_double_sparsity =", d.get("enable_double_sparsity"))
print("disable_radix_cache =", d.get("disable_radix_cache"))
print("mem_fraction_static =", d.get("mem_fraction_static"))
ist = d.get("internal_states") or []
if ist:
    mu = ist[0].get("memory_usage") or {}
    tc = mu.get("token_capacity")
    print("token_capacity =", tc)
    print("effective_max_running_requests_per_dp =", ist[0].get("effective_max_running_requests_per_dp"))
    print("AC7_REF_410560_MATCH =", tc == 410560)
PY
grep -aiE "max_total_num_tokens|KV Cache is allocated" "$slog" | head -2
teardown; gpu_idle_wait
echo "=== AC-7 DSA done $(date -u +%H:%M:%SZ) ==="
