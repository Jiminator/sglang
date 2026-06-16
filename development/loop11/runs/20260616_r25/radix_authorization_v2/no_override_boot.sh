#!/usr/bin/env bash
# R25 v2 — NO-OVERRIDE BOOT (the product mechanism) + negative control + AC-7.
#
# (1) Boot DS radix-ON authorized by --double-sparsity-radix-fixture-artifact <v2>,
#     with NO SGLANG_DS_RADIX_OVERRIDE (explicitly unset) and WITHOUT
#     --disable-radix-cache. The validator (apply_radix_fixture_artifact ->
#     validate_double_sparsity) MUST ACCEPT. Confirm: boot, smoke gen, a REAL radix
#     hit (cached_tokens>0 on a repeated prompt), and the WARNING log line recording
#     the authorizing artifact path + sha256.
# (2) NEGATIVE CONTROL: same radix-on config, NO artifact, NO override -> validator
#     MUST REFUSE with the table-free-radix-fixture message (boot fails closed).
# (3) AC-7: DSA-native default (NO --enable-double-sparsity): max_total_num_tokens
#     == 410560, all 8 KV ranks, smoke OK.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../radix_authorization/r25_env.sh"
cd "$REPO"
mkdir -p "$HERE/probes"
LOG="$HERE/probes/no_override_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== R25 v2 no-override boot start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

ARTIFACT="${1:?usage: no_override_boot.sh <v2_artifact_path>}"
echo "artifact=$ARTIFACT"
ARTIFACT_SHA=$(sha256sum "$ARTIFACT" | awk '{print $1}')
echo "artifact_sha256=$ARTIFACT_SHA"

# ---- (1) Positive: authorized no-override radix-ON boot ----
teardown
unset SGLANG_DS_RADIX_OVERRIDE || true
slog="$HERE/probes/no_override_serve.log"
echo "=== boot DS radix-ON via fixture artifact, NO env override, NO --disable-radix-cache $(date -u +%H:%M:%SZ) ==="
# graph-on (the real serving path): no recall_oracle/selection_capture in this config.
POS_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_TABLEFREE"
  --double-sparsity-radix-fixture-artifact "$ARTIFACT")
env | grep -q '^SGLANG_DS_RADIX_OVERRIDE=' && echo "!! ERROR: SGLANG_DS_RADIX_OVERRIDE still set" || echo "confirmed: SGLANG_DS_RADIX_OVERRIDE unset"
python -m sglang.launch_server "${POS_ARGS[@]}" > "$slog" 2>&1 &
rc_wait=0
ready_wait "$slog" || rc_wait=$?
POS_STATUS=BOOT_FAIL; SMK="n/a"; WARM_CACHED=-1; WARN_LINE="(none)"; DRC="(n/a)"
if [[ "$rc_wait" == "0" ]]; then
  POS_STATUS=ACCEPTED
  SMK=$(smoke)
  DRC=$(grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || echo "(none)")
  RP='The product mechanism authorizes radix-on via a config-bound fixture artifact. Repeat this exact sentence to populate the radix cache, then request it again to prove a real cache hit occurs without any environment override.'
  curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
    -d "{\"text\": $(python3 -c "import json;print(json.dumps('$RP'))"), \"sampling_params\": {\"max_new_tokens\": 8, \"temperature\": 0}}" > "$HERE/probes/hit_cold.json" 2>&1
  WARM_JSON=$(curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
    -d "{\"text\": $(python3 -c "import json;print(json.dumps('$RP'))"), \"sampling_params\": {\"max_new_tokens\": 8, \"temperature\": 0}}")
  echo "$WARM_JSON" > "$HERE/probes/hit_warm.json"
  WARM_CACHED=$(echo "$WARM_JSON" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('meta_info',{}).get('cached_tokens',-1))" 2>/dev/null || echo -1)
  WARN_LINE=$(grep -aE "DS radix-cache fixture recorded as PASSED.*artifact=" "$slog" | head -1 || echo "(none)")
else
  echo "!! POSITIVE no-override boot FAILED (rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
fi
echo ">>> POSITIVE: status=$POS_STATUS disable_radix_cache_line='$DRC' smoke=$SMK warm_cached_tokens=$WARM_CACHED"
echo ">>> authorizing WARNING: $WARN_LINE"
teardown; gpu_idle_wait

# ---- (2) Negative control: radix-ON, NO artifact, NO override -> MUST REFUSE ----
echo "=== negative control: radix-ON, NO artifact, NO override (must fail closed) $(date -u +%H:%M:%SZ) ==="
unset SGLANG_DS_RADIX_OVERRIDE || true
nslog="$HERE/probes/neg_control_serve.log"
NEG_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_TABLEFREE")
python -m sglang.launch_server "${NEG_ARGS[@]}" > "$nslog" 2>&1 &
neg_rc=0
ready_wait "$nslog" || neg_rc=$?
NEG_REFUSED=no; NEG_MSG="(none)"
if [[ "$neg_rc" != "0" ]]; then
  if grep -aqE "requires --disable-radix-cache until the .*table-free radix fixture" "$nslog"; then
    NEG_REFUSED=yes
    NEG_MSG=$(grep -aE "requires --disable-radix-cache until the .*table-free radix fixture" "$nslog" | head -1)
  else
    NEG_MSG=$(grep -aE "ValueError|radix|fixture" "$nslog" | tail -5 | tr '\n' '|')
  fi
else
  echo "!! NEGATIVE control BOOTED (should have refused) — this is a FAIL"
fi
echo ">>> NEGATIVE: refused=$NEG_REFUSED msg=$NEG_MSG"
teardown; gpu_idle_wait

# ---- (3) AC-7: DSA-native default ----
echo "=== AC-7: DSA-native default (no DS) $(date -u +%H:%M:%SZ) ==="
EXPECTED=410560
teardown
a7log="$HERE/probes/a7_serve.log"
python -m sglang.launch_server "${DSA_ARGS[@]}" > "$a7log" 2>&1 &
a7_rc=0
ready_wait "$a7log" || a7_rc=$?
A7_CAP=0; A7_SMK="n/a"; A7_KV=0
if [[ "$a7_rc" == "0" ]]; then
  A7_CAP=$(curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('max_total_num_tokens') or 0)")
  A7_SMK=$(smoke)
  A7_KV=$(grep -acE "TP[0-7]\] KV Cache is allocated.*#tokens: ${EXPECTED}" "$a7log")
else
  echo "!! AC-7 boot FAIL (rc=$a7_rc)"; tail -n 40 "$a7log"
fi
A7_MATCH=$([[ "$A7_CAP" == "$EXPECTED" ]] && echo MATCH || echo MISMATCH)
echo ">>> AC-7: max_total_num_tokens=$A7_CAP expected=$EXPECTED -> $A7_MATCH kv_ranks=$A7_KV smoke=$A7_SMK"
teardown; gpu_idle_wait

{
  echo "=== R25 v2 no-override boot + negative control + AC-7 ==="
  echo "artifact=$ARTIFACT"
  echo "artifact_sha256=$ARTIFACT_SHA"
  echo "[1] POSITIVE no-override radix-ON boot: status=$POS_STATUS disable_radix_cache_line='$DRC' smoke=$SMK warm_cached_tokens=$WARM_CACHED"
  echo "    authorizing WARNING: $WARN_LINE"
  echo "[2] NEGATIVE control (no artifact, no override): refused=$NEG_REFUSED"
  echo "    refusal msg: $NEG_MSG"
  echo "[3] AC-7 DSA default: max_total_num_tokens=$A7_CAP expected=$EXPECTED -> $A7_MATCH kv_ranks=$A7_KV smoke=$A7_SMK"
} > "$HERE/probes/no_override_evidence.txt"
cat "$HERE/probes/no_override_evidence.txt"
echo "=== final nvidia-smi ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== R25 v2 no-override done $(date -u +%H:%M:%SZ) ==="
[[ "$POS_STATUS" == "ACCEPTED" && "$WARM_CACHED" -gt 0 && "$NEG_REFUSED" == "yes" && "$A7_MATCH" == "MATCH" ]] && exit 0 || exit 1
