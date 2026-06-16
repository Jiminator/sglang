#!/usr/bin/env bash
# Re-run ONLY the positive no-override product boot to confirm a REAL radix hit with a
# prompt LONGER than page_size=64 (the prior run used a 40-token prompt < page64, so the
# single partial page yielded cached_tokens=0 — a test-prompt artifact, not a radix
# failure; RadixCache was initialized and GATE A/C already proved real hits up to 2752
# cached tokens). Same authorized product boot: --double-sparsity-radix-fixture-artifact
# <v2>, NO SGLANG_DS_RADIX_OVERRIDE, NO --disable-radix-cache. graph-ON (real serving path).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../radix_authorization/r25_env.sh"
cd "$REPO"
LOG="$HERE/probes/no_override_cachehit_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== no-override cache-hit re-run start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

ARTIFACT="${1:?usage: <v2_artifact_path>}"
echo "artifact=$ARTIFACT  sha256=$(sha256sum "$ARTIFACT" | awk '{print $1}')"

teardown
unset SGLANG_DS_RADIX_OVERRIDE || true
slog="$HERE/probes/no_override_cachehit_serve.log"
POS_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_TABLEFREE"
  --double-sparsity-radix-fixture-artifact "$ARTIFACT")
env | grep -q '^SGLANG_DS_RADIX_OVERRIDE=' && echo "!! ERROR: override still set" || echo "confirmed: SGLANG_DS_RADIX_OVERRIDE unset"
python -m sglang.launch_server "${POS_ARGS[@]}" > "$slog" 2>&1 &
rc_wait=0
ready_wait "$slog" || rc_wait=$?
POS_STATUS=BOOT_FAIL; WARM_CACHED=-1; WARN_LINE="(none)"; DRC="(none)"
if [[ "$rc_wait" == "0" ]]; then
  POS_STATUS=ACCEPTED
  DRC=$(grep -aoE "disable_radix_cache=(True|False)" "$slog" | head -1)
  # A LONG repeated prompt (>> page_size=64 tokens) so committed pages are reused.
  RP=$(python3 -c "print(('The product mechanism authorizes radix-on via a config-bound fixture artifact, recorded once the value-equivalence gates pass on hardware. ' * 24).strip())")
  curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
    -d "{\"text\": $(python3 -c "import json,os;print(json.dumps(os.environ['RP']))" RP="$RP"), \"sampling_params\": {\"max_new_tokens\": 8, \"temperature\": 0}}" > "$HERE/probes/hit_cold2.json" 2>&1
  COLD_CACHED=$(python3 -c "import json;print(json.load(open('$HERE/probes/hit_cold2.json')).get('meta_info',{}).get('cached_tokens',-1))" 2>/dev/null || echo -1)
  COLD_PT=$(python3 -c "import json;print(json.load(open('$HERE/probes/hit_cold2.json')).get('meta_info',{}).get('prompt_tokens',-1))" 2>/dev/null || echo -1)
  WARM_JSON=$(curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
    -d "{\"text\": $(python3 -c "import json,os;print(json.dumps(os.environ['RP']))" RP="$RP"), \"sampling_params\": {\"max_new_tokens\": 8, \"temperature\": 0}}")
  echo "$WARM_JSON" > "$HERE/probes/hit_warm2.json"
  WARM_CACHED=$(echo "$WARM_JSON" | python3 -c "import json,sys;print(json.load(sys.stdin).get('meta_info',{}).get('cached_tokens',-1))" 2>/dev/null || echo -1)
  WARN_LINE=$(grep -aE "DS radix-cache fixture recorded as PASSED.*artifact=" "$slog" | head -1 || echo "(none)")
  echo ">>> prompt_tokens=$COLD_PT cold_cached=$COLD_CACHED warm_cached=$WARM_CACHED"
else
  echo "!! boot FAIL (rc_wait=$rc_wait) — tail:"; tail -n 40 "$slog"
fi
echo ">>> POSITIVE: status=$POS_STATUS disable_radix_cache_line=$DRC warm_cached_tokens=$WARM_CACHED"
echo ">>> authorizing WARNING: $WARN_LINE"
teardown; gpu_idle_wait
echo "=== final nvidia-smi ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
{
  echo "no_override_cachehit_rerun"
  echo "artifact=$ARTIFACT"
  echo "status=$POS_STATUS  disable_radix_cache_line=$DRC"
  echo "warm_cached_tokens=$WARM_CACHED (prompt_tokens=$COLD_PT, prompt >> page_size=64)"
  echo "authorizing WARNING: $WARN_LINE"
} > "$HERE/probes/no_override_cachehit_evidence.txt"
cat "$HERE/probes/no_override_cachehit_evidence.txt"
echo "=== done $(date -u +%H:%M:%SZ) ==="
[[ "$POS_STATUS" == "ACCEPTED" && "$WARM_CACHED" -gt 0 ]] && exit 0 || exit 1
