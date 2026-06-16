#!/usr/bin/env bash
# Reproduce the R27 DEC-12 production-reuse edge verdict from COMMITTED evidence ONLY.
#
# The cache-engagement index files and the recall sinks are committed compressed
# (gate_c/index_<arm>.jsonl.gz + gate_c/sink_<arm>.jsonl.gz); analyze_edge.py is
# .gz-aware (falls back to the .gz sibling), so a clean checkout consumes the tracked
# artifacts directly. This proves the DEC-12 edge PASS is independently reproducible:
# it regenerates the verdict and diffs it against the committed gate_c_edge_verdict.json.
#
# Usage:  bash reproduce.sh            # exits 0 iff the reproduced verdict matches
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# Same invocation as run_gate_c.sh's authoritative analyze step (length 4096, page 64,
# +/-0.5pp bar, production-reuse upper bound 2752).
python "$HERE/analyze_edge.py" \
  --dir "$HERE/gate_c" --length 4096 --page-size 64 \
  --max-delta-pp 0.5 --min-needles 128 --production-cached-max 2752 \
  --out "$OUT/repro_verdict.json" --table-out "$OUT/repro_table.jsonl"

echo "=== reproduced verdict status + arm deltas ==="
python3 - "$OUT/repro_verdict.json" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
print("status:", v["status"], "problems:", v["problems"])
for k, c in v["cases"].items():
    print(f"  {k}: {c['paired_delta_case_minus_cold_pp']['mean']}pp (control={c['control_arm']})")
o = v["out_of_contract_value_affecting"]
print(f"  out_of_contract {o['case']}: {o['paired_delta_case_minus_cold_pp']['mean']}pp")
PY

echo "=== diff committed vs reproduced (normalized JSON) ==="
if diff -u \
    <(python3 -m json.tool "$HERE/gate_c_edge_verdict.json") \
    <(python3 -m json.tool "$OUT/repro_verdict.json"); then
  echo "REPRODUCE OK: reproduced verdict == committed gate_c_edge_verdict.json"
else
  echo "REPRODUCE MISMATCH (see diff above)"; exit 1
fi
