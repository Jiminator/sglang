#!/usr/bin/env bash
# Experiment B v2 — EAGER (the v1 graph-on run made even the radix-OFF fresh-vs-fresh
# control NON-reproducible, so it could not discriminate). EAGER restores the
# deterministic baseline (matches dec9_determinism Q1, which proved fresh==fresh
# bit-identical EAGER). DSA default (NO --enable-double-sparsity). Output/logprob
# probe via standard /generate (return_logprob, greedy temp0 top_k1). No new prod code.
#
# Boot B1 (radix ON, shipped): cold -> warm (identical prompt). Diagnostic: also a
#   2nd warm send to confirm whether DSA-default radix EVER hits for this prompt.
# Boot B2 (radix OFF control): fresh -> fresh. MUST be bit-identical EAGER.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R24="$HERE/../tablefree_radix"
source "$R24/r24_env.sh"
REPO=/sgl-workspace/sglang
cd "$REPO"
LOG="$HERE/expB_v3_stage.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== ExpB v3 (EAGER) start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
OUT="$HERE/probes_v3"; mkdir -p "$OUT"
STEPS=16

boot_dsa() {  # $1=tag  $2=radix(on|off)
  local tag="$1"; local radix="$2"
  local slog="$OUT/serve_${tag}.log"
  teardown
  echo "=== boot DSA-default ($tag) radix=$radix EAGER $(date -u +%H:%M:%SZ) ==="
  local args=(--model-path "$GLM" --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64
    --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
    --disable-overlap-schedule --disable-piecewise-cuda-graph --disable-cuda-graph
    --random-seed 20260607 --trust-remote-code --host 127.0.0.1 --port "$PORT"
    --mem-fraction-static 0.8 --max-running-requests 64 --cuda-graph-max-bs 64)
  if [[ "$radix" == "off" ]]; then
    args+=(--disable-radix-cache)
  fi
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local rc_wait=0
  ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != "0" ]]; then
    echo "!! ExpB v3 boot FAIL ($tag rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
    teardown; gpu_idle_wait; return 10
  fi
  echo ">>> server ready ($tag). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || true
}

# ---- Boot B1: radix ON (shipped default), EAGER ----
boot_dsa b1 on || { echo "B1 boot failed"; exit 1; }
python "$HERE/expB_probe.py" --tag B1_cold  --outdir "$OUT" --steps $STEPS 2>&1 | tee "$OUT/B1_cold.log"
python "$HERE/expB_probe.py" --tag B1_warm  --outdir "$OUT" --steps $STEPS 2>&1 | tee "$OUT/B1_warm.log"
python "$HERE/expB_probe.py" --tag B1_warm2 --outdir "$OUT" --steps $STEPS 2>&1 | tee "$OUT/B1_warm2.log"
teardown; gpu_idle_wait

# ---- Boot B2: radix OFF control, two fresh sends, EAGER ----
boot_dsa b2 off || { echo "B2 boot failed"; exit 1; }
python "$HERE/expB_probe.py" --tag B2_fresh1 --outdir "$OUT" --steps $STEPS 2>&1 | tee "$OUT/B2_fresh1.log"
python "$HERE/expB_probe.py" --tag B2_fresh2 --outdir "$OUT" --steps $STEPS 2>&1 | tee "$OUT/B2_fresh2.log"
teardown; gpu_idle_wait

# ---- Compare ----
python "$HERE/expB_compare.py" --a "$OUT/B1_cold.json" --b "$OUT/B1_warm.json" \
  --label B1_cold_vs_warm --out "$HERE/expB_v3_B1_verdict.json" 2>&1 | tee "$OUT/cmp_B1.log"
python "$HERE/expB_compare.py" --a "$OUT/B2_fresh1.json" --b "$OUT/B2_fresh2.json" \
  --label B2_fresh_vs_fresh --out "$HERE/expB_v3_B2_verdict.json" 2>&1 | tee "$OUT/cmp_B2.log"

echo "=== ExpB v3 done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
teardown; gpu_idle_wait
