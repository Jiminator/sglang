#!/usr/bin/env bash
# Prove DS is genuinely active on the live server: a long-context (>top_k) request
# must report selected < total and dense_fallback == 0.  Run only against a DS server.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
# shellcheck source=_env.sh
source "$HERE/_env.sh" || exit 1

LONG=$(python3 -c 'print("The quick brown fox jumps over the lazy dog near the riverbank at dawn. " * 350)')
RESP=$(curl -sf "http://$HOST:$PORT/generate" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1], "sampling_params": {"max_new_tokens": 16, "temperature": 0}}))' "$LONG Summarize.")")
echo "$RESP" | python3 -c '
import sys, json
ds = json.load(sys.stdin).get("meta_info", {}).get("double_sparsity")
print(json.dumps(ds, indent=2) if ds else "<<< NO double_sparsity (DS off or inactive) >>>")
ok = ds and ds.get("selected_tokens",0) > 0 and ds.get("total_tokens",0) > ds.get("selected_tokens",0) and ds.get("dense_fallback",1) == 0
print("DS_ACTIVE_PASS" if ok else "DS_ACTIVE_FAIL")
sys.exit(0 if ok else 1)'
