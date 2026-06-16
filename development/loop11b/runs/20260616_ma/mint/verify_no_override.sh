#!/usr/bin/env bash
# loop11b VERIFY NO-OVERRIDE — boot DS radix-ON via the MINTED fixture with NO env override
# (the product mechanism), against the REGENERATED channel mask. Mechanical port of loop-11
# R27 no_override_boot.sh, re-pointed at loop11b env.sh + the minted fixture. STAGE-selectable
# so each server boot is ONE self-contained foreground call (boot+probe+teardown). GRAPH mode
# (NOT eager). ONE server at a time; fail-closed.
#
#   STAGE=pos     : authorized no-override radix-ON boot via the minted fixture + DS config
#                   pointed at the REGENERATED mask path. Capture /server_info, assert
#                   double_sparsity_radix_fixture_artifact set + disable_radix_cache=False,
#                   grep the serve log for the validator fixture-authorization WARNING.
#   STAGE=altpath : DEC-1 PORTABILITY negative control. Copy the mask to a SECOND path and
#                   boot with a DS config variant pointed THERE, but the SAME minted fixture.
#                   Same tensor content, different file path => the content-sha fingerprint
#                   matches and the boot MUST STILL authorize (proves DEC-1 path-independence).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
cd "$REPO"
mkdir -p "$HERE/probes/no_override"
STAGE="${STAGE:?usage: STAGE=pos|altpath [ARTIFACT=...] verify_no_override.sh}"
# Default to the minted fixture written by mint_fixture.py.
ARTIFACT="${ARTIFACT:-/sgl-workspace/sglang/development/serve_double_sparsity_radix_fixture.json}"

# Probe a booted server: capture /server_info, assert the fixture+radix fields, grep the
# serve log for the validator authorization WARNING, and force a REAL radix hit. Sets the
# globals PROBE_STATUS / WARM_CACHED for the caller's exit decision.
probe_authorized_boot() {  # $1=serve-log  $2=server_info.json out  $3=stage-tag
  local slog="$1"; local sinfo="$2"; local tag="$3"
  PROBE_STATUS=BOOT_FAIL; SMK="n/a"; WARM_CACHED=-1; WARM_PROMPT=-1; WARN_LINE="(none)"
  ARTIFACT_SET="(n/a)"; DRC_FIELD="(n/a)"
  local rc_wait=0; ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != "0" ]]; then
    echo "!! no-override boot ($tag) FAILED (rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
    return 1
  fi
  PROBE_STATUS=ACCEPTED
  SMK=$(smoke)
  # /server_info is the current endpoint (returns the full ServerArgs dict).
  curl -s --max-time 15 "http://127.0.0.1:${PORT}/server_info" > "$sinfo" 2>&1 || true
  ARTIFACT_SET=$(python3 -c "import json;d=json.load(open('$sinfo'));v=d.get('double_sparsity_radix_fixture_artifact');print('set:'+str(v) if v else 'UNSET')" 2>/dev/null || echo "(parse-fail)")
  DRC_FIELD=$(python3 -c "import json;d=json.load(open('$sinfo'));print('disable_radix_cache='+str(d.get('disable_radix_cache')))" 2>/dev/null || echo "(parse-fail)")
  WARN_LINE=$(grep -aE "DS radix-cache fixture recorded as PASSED.*artifact=" "$slog" | head -1 || echo "(none)")
  # >=600-token repeated prompt to force a REAL radix hit on the second request.
  local RP PAYLOAD WARM_JSON
  RP=$(python3 -c "print('The product mechanism authorizes radix-on via a config-bound fixture artifact. ' + 'Repeat this exact long sentence many times to populate the radix cache then request it again to prove a real cache hit occurs without any environment override. '*40)")
  PAYLOAD=$(python3 -c "import json,sys;print(json.dumps({'text':sys.argv[1],'sampling_params':{'max_new_tokens':8,'temperature':0}}))" "$RP")
  curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' -d "$PAYLOAD" > "$HERE/probes/no_override/hit_cold_${tag}.json" 2>&1
  WARM_JSON=$(curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' -d "$PAYLOAD")
  echo "$WARM_JSON" > "$HERE/probes/no_override/hit_warm_${tag}.json"
  WARM_CACHED=$(echo "$WARM_JSON" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('meta_info',{}).get('cached_tokens',-1))" 2>/dev/null || echo -1)
  WARM_PROMPT=$(echo "$WARM_JSON" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('meta_info',{}).get('prompt_tokens',-1))" 2>/dev/null || echo -1)
  echo ">>> $tag: status=$PROBE_STATUS smoke=$SMK $DRC_FIELD artifact=$ARTIFACT_SET warm_prompt_tokens=$WARM_PROMPT warm_cached_tokens=$WARM_CACHED"
  echo ">>> $tag: authorizing WARNING: $WARN_LINE"
  return 0
}

if [[ "$STAGE" == "pos" ]]; then
  LOG="$HERE/probes/no_override/pos_stage.log"; exec > >(tee "$LOG") 2>&1
  echo "=== loop11b no-override POSITIVE boot $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
  echo "artifact=$ARTIFACT"; ARTIFACT_SHA=$(sha256sum "$ARTIFACT" | awk '{print $1}'); echo "artifact_sha256=$ARTIFACT_SHA"
  echo "mask=$MASK content_sha256=$MASK_CONTENT_SHA256"
  teardown
  unset SGLANG_DS_RADIX_OVERRIDE || true
  env | grep -q '^SGLANG_DS_RADIX_OVERRIDE=' && echo "!! ERROR: SGLANG_DS_RADIX_OVERRIDE still set" || echo "confirmed: SGLANG_DS_RADIX_OVERRIDE unset"
  slog="$HERE/probes/no_override/no_override_serve.log"
  sinfo="$HERE/probes/no_override/server_info.json"
  echo "=== boot DS radix-ON via minted fixture, GRAPH mode, NO env override, NO --disable-radix-cache $(date -u +%H:%M:%SZ) ==="
  # GRAPH mode: NO --disable-cuda-graph. DS_CFG_TABLEFREE points at the REGENERATED mask.
  POS_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_TABLEFREE"
    --double-sparsity-radix-fixture-artifact "$ARTIFACT")
  python -m sglang.launch_server "${POS_ARGS[@]}" > "$slog" 2>&1 &
  probe_authorized_boot "$slog" "$sinfo" pos || true
  {
    echo "[1] POSITIVE no-override radix-ON boot (GRAPH mode): status=$PROBE_STATUS smoke=$SMK"
    echo "    $DRC_FIELD  fixture_artifact=$ARTIFACT_SET  warm_prompt_tokens=$WARM_PROMPT  warm_cached_tokens=$WARM_CACHED"
    echo "    artifact=$ARTIFACT sha256=$ARTIFACT_SHA"
    echo "    mask=$MASK content_sha256=$MASK_CONTENT_SHA256"
    echo "    authorizing WARNING: $WARN_LINE"
  } > "$HERE/probes/no_override/pos_evidence.txt"
  cat "$HERE/probes/no_override/pos_evidence.txt"
  teardown; gpu_idle_wait
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  # PASS iff authorized boot, fixture set, radix ON (disable_radix_cache=False), real hit.
  [[ "$PROBE_STATUS" == "ACCEPTED" && "$ARTIFACT_SET" == set:* && "$DRC_FIELD" == "disable_radix_cache=False" && "$WARM_CACHED" -gt 0 ]] && exit 0 || exit 1

elif [[ "$STAGE" == "altpath" ]]; then
  LOG="$HERE/probes/no_override/altpath_stage.log"; exec > >(tee "$LOG") 2>&1
  echo "=== loop11b DEC-1 PORTABILITY control: same-content mask at a SECOND path, SAME minted fixture $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
  echo "artifact=$ARTIFACT"; ARTIFACT_SHA=$(sha256sum "$ARTIFACT" | awk '{print $1}'); echo "artifact_sha256=$ARTIFACT_SHA"
  # Copy the REGENERATED mask to a SECOND path (different file path, identical tensor content).
  ALT_MASK="$REPO/.sglang_ds_oracle/mask_altpath.safetensors"
  mkdir -p "$(dirname "$ALT_MASK")"
  cp -f "$MASK" "$ALT_MASK"
  ALT_SHA=$(sha256sum "$ALT_MASK" | awk '{print $1}'); SRC_SHA=$(sha256sum "$MASK" | awk '{print $1}')
  echo "alt_mask=$ALT_MASK (full-file sha=$ALT_SHA, src full-file sha=$SRC_SHA — content-sha is what DEC-1 pins)"
  # DS config variant pointing channel_mask_path at the ALT path (same content, different path).
  DS_CFG_ALTPATH="${DS_CFG_TABLEFREE//$MASK/$ALT_MASK}"
  echo "ds_cfg_altpath channel_mask_path -> $ALT_MASK"
  teardown
  unset SGLANG_DS_RADIX_OVERRIDE || true
  env | grep -q '^SGLANG_DS_RADIX_OVERRIDE=' && echo "!! ERROR: SGLANG_DS_RADIX_OVERRIDE still set" || echo "confirmed: SGLANG_DS_RADIX_OVERRIDE unset"
  slog="$HERE/probes/no_override/altpath_serve.log"
  sinfo="$HERE/probes/no_override/server_info_altpath.json"
  echo "=== boot DS radix-ON via SAME minted fixture but ALT-path mask config, GRAPH mode, NO override $(date -u +%H:%M:%SZ) ==="
  ALT_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CFG_ALTPATH"
    --double-sparsity-radix-fixture-artifact "$ARTIFACT")
  python -m sglang.launch_server "${ALT_ARGS[@]}" > "$slog" 2>&1 &
  probe_authorized_boot "$slog" "$sinfo" altpath || true
  {
    echo "[2] DEC-1 PORTABILITY control (same-content mask, DIFFERENT path, SAME minted fixture):"
    echo "    status=$PROBE_STATUS smoke=$SMK $DRC_FIELD fixture_artifact=$ARTIFACT_SET"
    echo "    warm_prompt_tokens=$WARM_PROMPT warm_cached_tokens=$WARM_CACHED"
    echo "    alt_mask=$ALT_MASK  src_mask=$MASK  content_sha256(pinned)=$MASK_CONTENT_SHA256"
    echo "    authorizing WARNING: $WARN_LINE"
    echo "    expectation: STILL AUTHORIZES (content-sha matches; path is free under DEC-1)"
  } > "$HERE/probes/no_override/altpath_evidence.txt"
  cat "$HERE/probes/no_override/altpath_evidence.txt"
  teardown; gpu_idle_wait
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  # PASS iff the boot STILL authorized despite the different mask file path (DEC-1 portability).
  [[ "$PROBE_STATUS" == "ACCEPTED" && "$ARTIFACT_SET" == set:* && "$DRC_FIELD" == "disable_radix_cache=False" && "$WARM_CACHED" -gt 0 ]] && exit 0 || exit 1

else
  echo "unknown STAGE=$STAGE"; exit 2
fi
