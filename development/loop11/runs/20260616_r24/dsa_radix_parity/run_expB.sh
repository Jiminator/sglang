#!/usr/bin/env bash
# Experiment B — does the shipped DSA-native default ride the radix v_h jitter?
# DSA default (NO --enable-double-sparsity). Probe at the OUTPUT/LOGPROB level via
# the standard /generate API (return_logprob, greedy temp0 top_k1). No new prod code.
#
# Boot B1: DSA default, radix-ON (shipped), graph ON allowed.
#   cold (cache empty) -> warm (IDENTICAL prompt, radix hit). Compare logprobs.
# Boot B2: DSA default, radix-OFF (--disable-radix-cache) control.
#   fresh -> fresh(same prompt). Must be bit-identical (proves probe discrimination).
# One TP=8 server at a time; teardown to ~0 MiB between boots.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R24="$HERE/../tablefree_radix"
source "$R24/r24_env.sh"
REPO=/sgl-workspace/sglang
cd "$REPO"
LOG="$HERE/expB_stage.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== ExpB start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
OUT="$HERE/probes"; mkdir -p "$OUT"
STEPS=16

# DSA default args: NO --enable-double-sparsity. radix per-boot.
boot_dsa() {  # $1=tag  $2=radix(on|off)
  local tag="$1"; local radix="$2"
  local slog="$OUT/serve_${tag}.log"
  teardown
  echo "=== boot DSA-default ($tag) radix=$radix $(date -u +%H:%M:%SZ) ==="
  local args=(--model-path "$GLM" --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64
    --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
    --disable-overlap-schedule --disable-piecewise-cuda-graph
    --random-seed 20260607 --trust-remote-code --host 127.0.0.1 --port "$PORT"
    --mem-fraction-static 0.8 --max-running-requests 64 --cuda-graph-max-bs 64)
  if [[ "$radix" == "off" ]]; then
    args+=(--disable-radix-cache)
  fi
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local rc_wait=0
  ready_wait "$slog" || rc_wait=$?
  if [[ "$rc_wait" != "0" ]]; then
    echo "!! ExpB boot FAIL ($tag rc_wait=$rc_wait) — tail:"; tail -n 60 "$slog"
    teardown; gpu_idle_wait; return 10
  fi
  echo ">>> server ready ($tag). smoke=$(smoke)"
  grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1 || true
}

# ---- Boot B1: radix ON (shipped default), graph on (no --disable-cuda-graph) ----
boot_dsa b1 on || { echo "B1 boot failed"; exit 1; }
python "$HERE/expB_probe.py" --tag B1_cold --outdir "$OUT" --steps $STEPS 2>&1 | tee "$OUT/B1_cold.log"
python "$HERE/expB_probe.py" --tag B1_warm --outdir "$OUT" --steps $STEPS 2>&1 | tee "$OUT/B1_warm.log"
teardown; gpu_idle_wait

# ---- Boot B2: radix OFF control, two fresh sends ----
boot_dsa b2 off || { echo "B2 boot failed"; exit 1; }
python "$HERE/expB_probe.py" --tag B2_fresh1 --outdir "$OUT" --steps $STEPS 2>&1 | tee "$OUT/B2_fresh1.log"
python "$HERE/expB_probe.py" --tag B2_fresh2 --outdir "$OUT" --steps $STEPS 2>&1 | tee "$OUT/B2_fresh2.log"
teardown; gpu_idle_wait

# ---- Compare ----
python "$HERE/expB_compare.py" --a "$OUT/B1_cold.json" --b "$OUT/B1_warm.json" \
  --label B1_cold_vs_warm --out "$HERE/expB_B1_verdict.json" 2>&1 | tee "$OUT/cmp_B1.log"
python "$HERE/expB_compare.py" --a "$OUT/B2_fresh1.json" --b "$OUT/B2_fresh2.json" \
  --label B2_fresh_vs_fresh --out "$HERE/expB_B2_verdict.json" 2>&1 | tee "$OUT/cmp_B2.log"

echo "=== ExpB done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
teardown; gpu_idle_wait
