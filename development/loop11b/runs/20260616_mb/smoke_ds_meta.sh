#!/usr/bin/env bash
# R2-B1 smoke: prove the GLM/dsa_backend DS per-request summary now populates
# meta_info["double_sparsity"] -> bench_serving emits non-null dense_fallback_total /
# selected_tokens_mean / total_tokens_mean and trial_evidence.py exits 0. GRAPH mode
# (the real verdict condition), radix-on, mem 0.8. Short bench.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/loop11b/runs/20260616_ma/mint/env.sh
cd "$REPO"
V2="$HERE/results_v2"; mkdir -p "$V2"/smoke
export HOST=127.0.0.1 PORT=30000
LOG="$V2/smoke/smoke.log"; exec > >(tee "$LOG") 2>&1
echo "=== R2-B1 smoke start $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) ==="

teardown 2>/dev/null || true
SLOG="$V2/smoke/serve_ds.log"
echo "=== boot ds (radix-on, mem 0.8, GRAPH) $(date -u +%H:%M:%SZ) ==="
MODEL_PATH="$GLM" CHANNEL_MASK_PATH="$MASK" bash "$REPO/development/serve_double_sparsity.sh" > "$SLOG" 2>&1 &
ready_wait "$SLOG" || { echo "!! boot FAIL"; tail -60 "$SLOG"; teardown; exit 10; }
echo ">>> ready. smoke=$(smoke)"

OUT="$V2/smoke/ds_smoke.jsonl"
echo "=== short bench --output-details (conc 16, 64 prompts, gen 128) $(date -u +%H:%M:%SZ) ==="
python3 -m sglang.bench_serving --backend sglang --host "$HOST" --port "$PORT" --seed 7 \
  --dataset-name generated-shared-prefix --gsp-num-groups 1 --gsp-prompts-per-group 64 \
  --gsp-system-prompt-len 2253 --gsp-question-len 1843 --gsp-output-len 128 --gsp-range-ratio 1.0 \
  --num-prompts 64 --max-concurrency 16 --warmup-requests 4 \
  --output-file "$OUT" --output-details > "$V2/smoke/bench.log" 2>&1
echo ">>> bench done. aggregate DS fields:"
python3 -c "
import json; d=json.load(open('$OUT'))
for k in ('dense_fallback_total','selected_tokens_mean','total_tokens_mean','cached_fraction'):
    print(f'  {k} = {d.get(k)!r}')
sr=d.get('sparsity_rate') or []
nn=[x for x in sr if x is not None]
print(f'  per-request sparsity_rate: {len(nn)}/{len(sr)} non-null', ('e.g. %.4f'%nn[0]) if nn else '')
"
echo "=== trial_evidence.py (must exit 0) ==="
python3 "$HERE/trial_evidence.py" "$OUT"; echo ">>> trial_evidence rc=$?"
teardown; gpu_idle_wait
echo "=== smoke done $(date -u +%H:%M:%SZ) ==="
