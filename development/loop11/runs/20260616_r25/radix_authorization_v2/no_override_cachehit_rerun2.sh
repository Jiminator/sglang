#!/usr/bin/env bash
# Corrected cache-hit confirmation on the authorized no-override product boot. The prior
# rerun's curl body was malformed by shell interpolation (400 Bad Request); here the JSON
# request body is built by python3 into a FILE and POSTed with --data @file (no shell
# interpolation of the prompt). Prompt is >> page_size=64 tokens so committed radix pages
# are reused on the warm request. Product boot: --double-sparsity-radix-fixture-artifact
# <v2>, NO SGLANG_DS_RADIX_OVERRIDE, NO --disable-radix-cache, graph-ON.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../radix_authorization/r25_env.sh"
cd "$REPO"
LOG="$HERE/probes/no_override_cachehit2_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== no-override cache-hit re-run #2 start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

ARTIFACT="${1:?usage: <v2_artifact_path>}"
echo "artifact=$ARTIFACT sha256=$(sha256sum "$ARTIFACT" | awk '{print $1}')"

# Build the request body file (python3, no shell interpolation of the prompt text).
REQ="$HERE/probes/cachehit_req.json"
python3 - "$REQ" <<'PY'
import json, sys
prompt = ("The product mechanism authorizes radix-on via a config-bound fixture "
          "artifact, recorded once the value-equivalence gates pass on hardware. ") * 24
json.dump({"text": prompt.strip(),
           "sampling_params": {"max_new_tokens": 8, "temperature": 0}},
          open(sys.argv[1], "w"))
print("wrote request body chars:", len(open(sys.argv[1]).read()))
PY

teardown
unset SGLANG_DS_RADIX_OVERRIDE || true
slog="$HERE/probes/no_override_cachehit2_serve.log"
POS_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_TABLEFREE"
  --double-sparsity-radix-fixture-artifact "$ARTIFACT")
env | grep -q '^SGLANG_DS_RADIX_OVERRIDE=' && echo "!! ERROR: override still set" || echo "confirmed: SGLANG_DS_RADIX_OVERRIDE unset"
python -m sglang.launch_server "${POS_ARGS[@]}" > "$slog" 2>&1 &
rc_wait=0
ready_wait "$slog" || rc_wait=$?
POS_STATUS=BOOT_FAIL; COLD_CACHED=-1; WARM_CACHED=-1; COLD_PT=-1; WARN_LINE="(none)"; DRC="(none)"
if [[ "$rc_wait" == "0" ]]; then
  POS_STATUS=ACCEPTED
  DRC=$(grep -aoE "disable_radix_cache=(True|False)" "$slog" | head -1)
  echo ">>> server ready. smoke=$(smoke)"
  curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" \
    -H 'Content-Type: application/json' --data @"$REQ" > "$HERE/probes/hit_cold3.json"
  WARM_OUT="$HERE/probes/hit_warm3.json"
  curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" \
    -H 'Content-Type: application/json' --data @"$REQ" > "$WARM_OUT"
  COLD_PT=$(python3 -c "import json;print(json.load(open('$HERE/probes/hit_cold3.json')).get('meta_info',{}).get('prompt_tokens',-1))" 2>/dev/null || echo -1)
  COLD_CACHED=$(python3 -c "import json;print(json.load(open('$HERE/probes/hit_cold3.json')).get('meta_info',{}).get('cached_tokens',-1))" 2>/dev/null || echo -1)
  WARM_CACHED=$(python3 -c "import json;print(json.load(open('$WARM_OUT')).get('meta_info',{}).get('cached_tokens',-1))" 2>/dev/null || echo -1)
  WARN_LINE=$(grep -aE "DS radix-cache fixture recorded as PASSED.*artifact=" "$slog" | head -1 || echo "(none)")
  echo ">>> prompt_tokens=$COLD_PT cold_cached=$COLD_CACHED warm_cached=$WARM_CACHED"
else
  echo "!! boot FAIL (rc_wait=$rc_wait) — tail:"; tail -n 40 "$slog"
fi
teardown; gpu_idle_wait
echo "=== final nvidia-smi ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
{
  echo "no_override_cachehit_rerun2 (corrected POST body)"
  echo "artifact=$ARTIFACT"
  echo "status=$POS_STATUS  $DRC"
  echo "prompt_tokens=$COLD_PT cold_cached_tokens=$COLD_CACHED warm_cached_tokens=$WARM_CACHED (prompt >> page_size=64)"
  echo "authorizing WARNING: $WARN_LINE"
} > "$HERE/probes/no_override_cachehit2_evidence.txt"
cat "$HERE/probes/no_override_cachehit2_evidence.txt"
echo "=== done $(date -u +%H:%M:%SZ) ==="
[[ "$POS_STATUS" == "ACCEPTED" && "$WARM_CACHED" -gt 0 ]] && exit 0 || exit 1
