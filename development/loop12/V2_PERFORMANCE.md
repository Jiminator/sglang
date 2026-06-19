# Double Sparsity v2 — current performance

Performance of the clean table-free Double Sparsity (DS) shipping branch
`double-sparsity-v2` (on `Jiminator/sglang`, cut from `origin/main`
`<BASE>=105e095e005d02a178fb6c5a23bd22ba644c90e4`). Measured 2026-06-18/19 on
**GLM-5.1-FP8, 8×H200, TP=8**.

## Headline (concurrency 64)

| Metric | loop-11b reference | Parity band | **DS v2 (this branch)** |
|--------|--------------------|-------------|--------------------------|
| p50 decode TPS | 26.9 | ≥ 24.2 (−10%) | **35.05** ✅ |
| P99 TTFT | 25.1 s | ≤ 30.1 s (+20%) | **22.90 s** ✅ |

**Verdict: PASS** — DS v2 is within the loop-11b regression band on both metrics
(and ahead of the reference). This is a *regression gate*, not the 30-TPS SLO
floor (which neither DS nor native DSA meets at conc 64 — see caveats).

## Workload (exactly mirrors the loop-11b conc-64 candidate)

- dataset `generated-shared-prefix`, **1 prefix group**, all requests sharing the
  one system prompt (`--gsp-num-groups 1 --gsp-prompts-per-group 256`)
- `--gsp-system-prompt-len 2253 --gsp-question-len 1843` → ISL ≈ 4096 (system
  prompt ≈ 55% of each input), `--gsp-output-len 512`, `--gsp-range-ratio 1.0`
- `--max-concurrency 64`, `--num-prompts 256`, one trial, seed 42, `--backend sglang`
- **256 / 256 requests completed** (`request_shape_ok = true`, 0 errors)
- metric: p50 decode TPS = median over requests of
  `(output_tokens − 1) / Σ(inter-token latencies)`; P99 TTFT from per-request TTFTs

Run via the shipped wrapper over stock `bench_serving`:
`benchmarks/bench_double_sparsity.py --model <GLM-5.1-FP8> --num-prompts 256 --seed 42`.

## DS is genuinely active (not a silent dense fallback)

A long-context decode response carried
`meta_info["double_sparsity"] = {sparsity_rate: 0.635, selected_tokens: 2048,
total_tokens: 5608, dense_fallback: 0}` — i.e. selection prunes to top-2048 of
5608 KV tokens, never falling back to dense. Per-layer startup bind logs present
(`double_sparsity bind shape check passed`, `bind_runtime_data completed`,
label_dim=32, page_size=64). Mask content SHA-256
`35155ac46ad79fa82e531138434ff35708e2d8c2932889323a21a455342a9b00`.

## Serving configuration

GLM-5.1-FP8, TP=8, `dsa` backend (`glm4_moe` path), FP8 KV cache, page size 64,
**CUDA graphs ON**, **radix cache ON (no fixture artifact, no override)**, custom
all-reduce ON, `mem-fraction-static 0.8`, `--disable-overlap-schedule
--disable-piecewise-cuda-graph`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` is
**never** set for serving (it breaks custom all-reduce IPC at TP=8;
calibration-only).

## Why decode is fast: the selector-width graph ladder

DS decode performance depends on the CUDA-graph **selector-width ladder**
(`selector_width_buckets`, default `[5120]`). The captured decode graph scores
only the covering width (5120) instead of the full `req_to_token` width
(~202,756 here), so the per-step selection is ~40× cheaper. Without it, conc-64
DS decode collapses to **~18.8 TPS / ~39 s TTFT** (the slow selection backs up
the queue). The runner keys decode graphs by `(batch size, selector width)`.

## Same-base context: native DSA (NOT a corrected-shape baseline)

For reference, native DSA (DS off) on the **same latest-main base** measured
**26.06 p50 decode TPS / 46.50 s P99 TTFT**. ⚠️ This was a separate, earlier run
taken *before* the perf wrapper pinned the GSP grouping, so it is **same-base
context only — not a corrected single-group measurement and not a pass/fail
baseline**. Its 46.50 s TTFT is shown only to make the point below.

## Caveats (base-environment drift from branching off latest main)

The absolute numbers are not directly comparable to loop-11b's dev base:
- **Latest `origin/main` requires `sglang-kernel >= 0.4.4`** (its flash-attention
  path uses the `only_qv` kernel); the dev base used 0.4.2.post2/0.4.3. The env
  was upgraded to 0.4.4 to serve this branch. This is a base requirement, not a
  DS dependency.
- **High conc-64 TTFT on this base is not DS-specific**: native DSA also shows
  ~46 s TTFT here, driven by a missing triton-3.6.0 MoE tuning config (falls back
  to the 3.5.1-tuned config → "Performance might be sub-optimal" on every MoE
  layer). DS, at 22.90 s, is comfortably inside the band regardless.

## Evidence

`development/loop12/perf_evidence/verdict.json` (DS, accepted: 256/256,
p50_decode_tps 35.053, p99_ttft_s 22.901, request_shape_ok true, parity true),
`development/loop12/perf_evidence/bench_result.json` (per-request detail),
`development/loop12/dsa_evidence/verdict.json` (native-DSA same-base context),
`development/loop12/m6m8_eval_ladder.out` / `m6m8_eval_r1.out` (run logs). The
shipped provenance doc is `benchmarks/DOUBLE_SPARSITY.md` on the branch.
