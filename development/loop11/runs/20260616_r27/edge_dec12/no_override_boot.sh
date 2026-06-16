#!/usr/bin/env bash
# R27 (DEC-12) — NO-OVERRIDE BOOT (the product mechanism) + negative control + AC-7,
# against the R27 edge-DEC12 v2 artifact. Reuses the R25 boot infra. STAGE-selectable so
# each server boot is ONE self-contained foreground call (boot+probe+teardown).
#
#   STAGE=pos  : authorized no-override radix-ON boot (validator ACCEPTS) + smoke + REAL
#                radix hit (cached_tokens>0 on a repeated >=600-tok prompt) + WARNING line.
#   STAGE=neg  : radix-ON, NO artifact, NO override -> validator MUST REFUSE (fail closed).
#   STAGE=ac7  : DSA-native default (no DS) -> max_total_num_tokens == 410560, 8 KV ranks.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../20260616_r25/radix_authorization/r25_env.sh"
cd "$REPO"
mkdir -p "$HERE/probes"
STAGE="${STAGE:?usage: STAGE=pos|neg|ac7 [ARTIFACT=...] no_override_boot.sh}"
ARTIFACT="${ARTIFACT:-$HERE/ds_radix_fixture_state_tablefree_v2.json}"

if [[ "$STAGE" == "pos" ]]; then
  LOG="$HERE/probes/pos_stage.log"; exec > >(tee "$LOG") 2>&1
  echo "=== R27 no-override POSITIVE boot $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
  echo "artifact=$ARTIFACT"; ARTIFACT_SHA=$(sha256sum "$ARTIFACT" | awk '{print $1}'); echo "artifact_sha256=$ARTIFACT_SHA"
  teardown
  unset SGLANG_DS_RADIX_OVERRIDE || true
  slog="$HERE/probes/no_override_serve.log"
  echo "=== boot DS radix-ON via fixture artifact, NO env override, NO --disable-radix-cache $(date -u +%H:%M:%SZ) ==="
  POS_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_TABLEFREE"
    --double-sparsity-radix-fixture-artifact "$ARTIFACT")
  env | grep -q '^SGLANG_DS_RADIX_OVERRIDE=' && echo "!! ERROR: SGLANG_DS_RADIX_OVERRIDE still set" || echo "confirmed: SGLANG_DS_RADIX_OVERRIDE unset"
  python -m sglang.launch_server "${POS_ARGS[@]}" > "$slog" 2>&1 &
  rc_wait=0; ready_wait "$slog" || rc_wait=$?
  POS_STATUS=BOOT_FAIL; SMK="n/a"; WARM_CACHED=-1; WARM_PROMPT=-1; WARN_LINE="(none)"; DRC="(n/a)"
  if [[ "$rc_wait" == "0" ]]; then
    POS_STATUS=ACCEPTED
    SMK=$(smoke)
    DRC=$(grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || echo "(none)")
    # >=600-token repeated prompt to force a REAL radix hit on the second request.
    RP=$(python3 -c "print('The product mechanism authorizes radix-on via a config-bound fixture artifact. ' + 'Repeat this exact long sentence many times to populate the radix cache then request it again to prove a real cache hit occurs without any environment override. '*40)")
    PAYLOAD=$(python3 -c "import json,sys;print(json.dumps({'text':sys.argv[1],'sampling_params':{'max_new_tokens':8,'temperature':0}}))" "$RP")
    curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' -d "$PAYLOAD" > "$HERE/probes/hit_cold.json" 2>&1
    WARM_JSON=$(curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' -d "$PAYLOAD")
    echo "$WARM_JSON" > "$HERE/probes/hit_warm.json"
    WARM_CACHED=$(echo "$WARM_JSON" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('meta_info',{}).get('cached_tokens',-1))" 2>/dev/null || echo -1)
    WARM_PROMPT=$(echo "$WARM_JSON" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('meta_info',{}).get('prompt_tokens',-1))" 2>/dev/null || echo -1)
    WARN_LINE=$(grep -aE "DS radix-cache fixture recorded as PASSED.*artifact=" "$slog" | head -1 || echo "(none)")
  else
    echo "!! POSITIVE no-override boot FAILED (rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
  fi
  echo ">>> POSITIVE: status=$POS_STATUS disable_radix_cache_line='$DRC' smoke=$SMK warm_prompt_tokens=$WARM_PROMPT warm_cached_tokens=$WARM_CACHED"
  echo ">>> authorizing WARNING: $WARN_LINE"
  {
    echo "[1] POSITIVE no-override radix-ON boot: status=$POS_STATUS disable_radix_cache_line='$DRC' smoke=$SMK warm_prompt_tokens=$WARM_PROMPT warm_cached_tokens=$WARM_CACHED"
    echo "    artifact=$ARTIFACT sha256=$ARTIFACT_SHA"
    echo "    authorizing WARNING: $WARN_LINE"
  } > "$HERE/probes/pos_evidence.txt"
  cat "$HERE/probes/pos_evidence.txt"
  teardown; gpu_idle_wait
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  [[ "$POS_STATUS" == "ACCEPTED" && "$WARM_CACHED" -gt 0 ]] && exit 0 || exit 1

elif [[ "$STAGE" == "neg" ]]; then
  LOG="$HERE/probes/neg_stage.log"; exec > >(tee "$LOG") 2>&1
  echo "=== R27 negative control: radix-ON, NO artifact, NO override (must fail closed) $(date -u +%H:%M:%SZ) ==="
  teardown
  unset SGLANG_DS_RADIX_OVERRIDE || true
  nslog="$HERE/probes/neg_control_serve.log"
  NEG_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_TABLEFREE")
  python -m sglang.launch_server "${NEG_ARGS[@]}" > "$nslog" 2>&1 &
  neg_rc=0; ready_wait "$nslog" || neg_rc=$?
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
  echo "[2] NEGATIVE control (no artifact, no override): refused=$NEG_REFUSED msg=$NEG_MSG" > "$HERE/probes/neg_evidence.txt"
  teardown; gpu_idle_wait
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  [[ "$NEG_REFUSED" == "yes" ]] && exit 0 || exit 1

elif [[ "$STAGE" == "ac7" ]]; then
  LOG="$HERE/probes/ac7_stage.log"; exec > >(tee "$LOG") 2>&1
  echo "=== R27 AC-7: DSA-native default (no DS) $(date -u +%H:%M:%SZ) ==="
  EXPECTED=410560
  teardown
  a7log="$HERE/probes/a7_serve.log"
  python -m sglang.launch_server "${DSA_ARGS[@]}" > "$a7log" 2>&1 &
  a7_rc=0; ready_wait "$a7log" || a7_rc=$?
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
  echo "[3] AC-7 DSA default: max_total_num_tokens=$A7_CAP expected=$EXPECTED -> $A7_MATCH kv_ranks=$A7_KV smoke=$A7_SMK" > "$HERE/probes/ac7_evidence.txt"
  teardown; gpu_idle_wait
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  [[ "$A7_MATCH" == "MATCH" ]] && exit 0 || exit 1
else
  echo "unknown STAGE=$STAGE"; exit 2
fi
