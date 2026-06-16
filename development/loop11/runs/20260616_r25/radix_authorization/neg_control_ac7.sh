#!/usr/bin/env bash
# R25 — NEGATIVE CONTROL + AC-7 (the POSITIVE no-override boot is NOT run: no passing
# v2 artifact was authorized because probes 1/3/5 failed).
#
# (NEG) radix-ON DS, NO artifact, NO SGLANG_DS_RADIX_OVERRIDE -> validator MUST REFUSE
#       with the table-free-radix-fixture message (boot fails closed).
# (AC-7) DSA-native default (NO --enable-double-sparsity, shipped radix default):
#       max_total_num_tokens == 410560, all 8 KV ranks, smoke OK.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/r25_env.sh"
cd "$REPO"
LOG="$HERE/probes/neg_ac7_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== R25 neg-control + AC-7 start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

# ---- NEGATIVE CONTROL: radix-ON, NO artifact, NO override -> MUST REFUSE ----
echo "=== negative control: radix-ON, NO artifact, NO override (must fail closed) $(date -u +%H:%M:%SZ) ==="
teardown
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
  echo "!! NEGATIVE control BOOTED (should have refused) — FAIL"
fi
echo ">>> NEGATIVE: refused=$NEG_REFUSED msg=$NEG_MSG"
teardown; gpu_idle_wait

# ---- AC-7: DSA-native default ----
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
  echo "=== R25 negative control + AC-7 ==="
  echo "NOTE: POSITIVE no-override boot NOT run — no passing v2 artifact (probes 1/3/5 FAILED)."
  echo "[NEG] radix-ON DS, NO artifact, NO override: refused=$NEG_REFUSED"
  echo "      refusal msg: $NEG_MSG"
  echo "[AC-7] DSA default: max_total_num_tokens=$A7_CAP expected=$EXPECTED -> $A7_MATCH kv_ranks=$A7_KV smoke=$A7_SMK"
} > "$HERE/probes/neg_ac7_evidence.txt"
cat "$HERE/probes/neg_ac7_evidence.txt"
echo "=== final nvidia-smi ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== R25 neg+AC7 done $(date -u +%H:%M:%SZ) ==="
[[ "$NEG_REFUSED" == "yes" && "$A7_MATCH" == "MATCH" ]] && exit 0 || exit 1
