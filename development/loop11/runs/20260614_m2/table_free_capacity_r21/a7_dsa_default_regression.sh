#!/usr/bin/env bash
# Loop 11 M2 R21 — A7 DSA-default un-regressed shared-surface check. The shipped
# DSA-native default (NO --enable-double-sparsity) must be un-regressed by the
# TokenLabelTable deletion (which touched shared files dsa_backend.py /
# deepseek_v2.py / pool_configurator.py). Boots DSA default, same model/TP/kv/page,
# mem 0.8; captures max_total_num_tokens; must equal the R20 value 410560.
# One TP=8 server; foreground; teardown the DS server first.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/r21_env.sh"
cd /sgl-workspace/sglang
LOG="$HERE/a7_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== A7 DSA-default regression start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
EXPECTED=410560   # R20 a7_dsa_default_evidence.txt: DSA-native default dsa08 @0.8

teardown
echo "=== boot DSA default (NO --enable-double-sparsity) @ mem0.8 $(date -u +%H:%M:%SZ) ==="
slog="$HERE/a7_dsa08_default_serve.log"
python -m sglang.launch_server "${DSA_ARGS[@]}" > "$slog" 2>&1 &
if ready_wait "$slog"; then
  CAP=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+"); SMK=$(smoke); ST=OK
else ST=BOOT_FAIL; CAP=NA; SMK=-; echo "!! A7 DSA boot FAIL — tail:"; tail -n 60 "$slog"; fi
MATCH=$([[ "$CAP" == "$EXPECTED" ]] && echo "MATCH" || echo "MISMATCH(got $CAP vs expected $EXPECTED)")
echo ">>> A7 dsa08 status=$ST cap=$CAP vs expected=$EXPECTED -> $MATCH smoke=$SMK"
{
  echo "=== A7 DSA-default regression (post-deletion HEAD) ==="
  echo "DSA-default dsa08 @0.8 (NO --enable-double-sparsity): status=$ST max_total_num_tokens=$CAP vs expected=$EXPECTED -> $MATCH smoke=$SMK"
  echo "--- boot field ---"
  grep -aE "TP0\] max_total_num_tokens" "$slog" | head -1
} > "$HERE/a7_dsa_default_evidence.txt"
cat "$HERE/a7_dsa_default_evidence.txt"
teardown
echo "=== A7 done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
