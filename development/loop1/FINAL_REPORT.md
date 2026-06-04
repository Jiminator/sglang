# GLM-5.1-FP8 Flags-Only Hill-Climb — Final Report

**Goal:** drive `zai-org/GLM-5.1-FP8` on one node of 8× H200 (TP8) as close as possible to the
client SLOs using **only** SGLang CLI flags + `SGLANG_*` env vars (no source / benchmark /
workload / SLO edits), measured with the fixed `development/benchmark.sh`.

**Workload (fixed):** generated-shared-prefix, 4096 ISL (2253 shared system prompt + 1843
question) / 512 OSL, 320 prompts, max-concurrency 64, ~55% prefix-cache hit, greedy, fixed seed 31234.

## SLOs and how they are measured (read this first)

The client specified **"30 tok/s per user"** and **"P99 TTFT < 22 s"**. "Per-user speed" is
reported as **sustained decode rate = 1000 / median_TPOT** (the faithful number), because
under EAGLE speculation `1000/ITL` is inflated by 3-token bursts (near-zero intra-burst
gaps deflate median ITL ~2.3×). We therefore:
- **Headline metric:** `1000 / median_TPOT` (typical user). Target = 30 tok/s ⇔ median_TPOT ≤ 33.3 ms.
- **Worst-case guarantee:** `1000 / p99_TPOT` (matches the client's P99-style framing for TTFT).
- `1000/median_ITL` is shown only as the *literal-but-speculation-inflated* figure, never as the headline.

> `mean_ITL ≈ mean_TPOT` always (the per-request itls sum to `latency − ttft`); they only
> diverge at the **median**, which is exactly where bursty emission skews the ITL distribution.

## Result summary (per-target, directional)

| target | metric | best safe (combo) | best achievable (combo+IndexCache) | met? |
|---|---|---|---|---|
| **30 tok/s/user** | 1000/median_TPOT | **25.3 tok/s** (39.5 ms) | **27.4 tok/s** (36.5 ms) | ✗ (gap 4.7 / **2.6**) |
| (worst-case) | 1000/p99_TPOT | ~14.3 tok/s (70 ms) | ~15.7 tok/s (64 ms) | ✗ |
| **P99 TTFT < 22 s** | p99_ttft_ms | **12.1 s** | **11.4 s** | ✅ (45–48 % margin) |
| literal formula | 1000/median_ITL | 55–60 tok/s | 56–60 tok/s | ✅ (but inflated) |

**Verdict:** P99 TTFT is met comfortably. The 30 tok/s/user target is **not** reached
flags-only; the honest remaining gap at the median is **~2.6 tok/s (≈9 %)** with the
accuracy-risk config and **~4.7 tok/s (≈16 %)** without it. "Target met" (both crossed) = **NO**.

## The two recommended configs

### A) Safe winner — no accuracy risk (recommended default)
```
SGLANG_ENABLE_SPEC_V2=1 sglang serve \
  --model-path zai-org/GLM-5.1-FP8 --tp 8 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.85 --max-running-requests 64 \
  --chunked-prefill-size 4096 --schedule-policy lpm
```
Confirmed (3 fresh runs): median_TPOT 40.31 / 39.37 / 38.73 ms → **24.8–25.8 tok/s**;
mean_TPOT ~41.2 ms; p99_TTFT ~12.1 s; median_ITL ~17 ms; accept ~3.1; 320/320, 0 errors.
bf16 KV, page 64, DSA prefill=flashmla_sparse / decode=fa3, max_total_num_tokens=300,352.

### B) Best-achievable — **ACCURACY-RISK (IndexCache, flagged)**
Same as (A) plus:
```
  --json-model-override-args '{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'
```
Confirmed (3 fresh runs): median_TPOT 36.30 / 36.65 / 36.66 ms → **27.3–27.5 tok/s**;
mean_TPOT ~37.7 ms; p99_TTFT ~11.4 s; accept ~3.1 (unchanged); 320/320, 0 errors.
**⚠ Accuracy-risk:** IndexCache reuses the DSA indexer result across layers; SGLang docs
state "negligible accuracy loss," but this latency benchmark **cannot verify quality** —
an accuracy eval (e.g. the model's reasoning/eval suite) must gate any production use.

## Why we can't reach 30 tok/s flags-only (binding bottleneck)

At concurrency 64 the model is **compute-bound on the 256-expert MoE decode forward**, not
on attention/KV. Evidence:
- **Speculation is a median-ITL/TPOT trap.** spec-OFF sustained TPOT (37.7 ms) is *better*
  than spec-ON (42.2 ms) — at conc 64 the 4-draft-token verify costs more than accept≈3.1
  buys — yet spec-OFF fails both gates (median_ITL 35.6 ms, p99_TTFT 29.6 s). Spec stays ON
  because it's the only config passing both gates (it frees request slots in ~165 verify
  iters vs ~512 plain-decode iters → lower TTFT).
- **DP attention regressed everything** at conc 64 (TPOT 48.6, throughput 1057 vs 1266 tok/s,
  p99_ITL 444): per-rank batch collapses to 64/8=8, and DP-attn + TP-MoE forces all-gather/
  reduce-scatter around the MoE every layer with nothing to amortize. It is a *high-concurrency
  / KV-bound* optimization; this workload is neither. (Stay TP8 — also satisfies the exclusions.)
- **FP8 KV (accuracy-risk r1) regressed TPOT** (45.6 ms) despite flipping DSA→flashmla_kv and
  doubling capacity — confirming the KV/attention path is **not** the bottleneck.
- **IndexCache (accuracy-risk r2) is the only knob that moved the binding metric** (−3.6 ms
  median TPOT) because it cuts *decode-path* MoE-adjacent indexer compute, not KV bandwidth.

The remaining gap is MoE-decode FLOPs, which no flag addresses without expert parallelism
(EP / a2a) — explicitly out of scope for this fixed TP8 path.

## Sweep table (14 distinct candidates + 6 confirmation reruns; fresh server each)

See `development/loop1/sweep_table.md` for the full machine-generated table (per-candidate
changed knob, median/mean/p99 ITL, mean/median/p99 TPOT, p99 TTFT, accept_length, observed
concurrency, max_total_num_tokens, completed/errors). Highlights:

| candidate | knob vs prev best | median_TPOT (tok/s) | mean_TPOT | p99_TTFT | note |
|---|---|---|---|---|---|
| baseline | cookbook + mrr 64 | 40.10 (24.9) | 42.16 | 13.3 s | gate-passing reference |
| nospec | spec OFF | 37.74 → 26.5* | 37.74 | 29.6 s | *fails both gates |
| dp_attn | +DP dp8 | 48.6 ✗ | 48.63 | 18.7 s | regressed all axes |
| spec_decmode | +attn-mode decode | ~42 | 42.15 | 12.3 s | neutral |
| eagle_light | steps2/draft3 | ~43 ✗ | 43.65 | 11.4 s | worse (accept 2.59) |
| lpm | +schedule lpm | 41.9 | 41.92 | 11.4 s | TTFT↓ |
| chunk4096 | +chunked-prefill 4096 | 41.2 | 41.19 | 12.0 s | mild win |
| dsa_pf_fa3 | DSA prefill fa3 | 41.6 | 41.60 | 11.5 s | neutral (prefill) |
| page32 | --page-size 32 | 41.2 | 41.21 | 11.3 s | **overridden→64** |
| fp8kv ⚠ | +FP8 KV | 45.6 ✗ | 45.62 | 13.7 s | acc-risk r1, regressed |
| **combo** | chunk4096 + lpm | **39.5 (25.3)** | 41.4 | 12.1 s | **safe winner** ×3 |
| **indexcache ⚠** | combo + IndexCache | **36.5 (27.4)** | 37.7 | 11.4 s | **best**, acc-risk r2 ×3 |

## Page-size (flexibility check)
`--page-size 32` launched + benchmarked successfully but the server resolved **page_size=64**:
on CUDA, DSA unconditionally sets 64 (`python/sglang/srt/server_args.py:1918-1920`; FlashMLA
"only supports a page_size of 64", `:2852`). GLM-5.1 DSA on H200 therefore supports exactly
**one** effective page size (64) — the winner uses 64 by hard architectural constraint, not
by preference.

## Out-of-scope axes — confirmed ABSENT in both winners (hard constraint)
Neither config uses EP / MoE a2a (`--moe-a2a-backend`, deepep), alternate MoE runners,
`--enable-torch-compile`, NGRAM speculative, or pd-multiplexing. Parallelism is **TP8**
(`tp_size=8, ep_size=1, dp_size=1, moe_a2a_backend=none`, verified in server_args). No sweep
budget was spent crash-probing excluded axes.

## Accuracy-risk ladder (hard constraint) — how it was respected
Non-accuracy-risk knobs were exhausted first (scheduler capacity, DSA backends under bf16,
speculative params, DP-vs-TP, page size, schedule policy). Then, in strict order:
- **r1 FP8 KV** (`--kv-cache-dtype fp8_e4m3`): tested → regressed TPOT → rejected. *Flagged.*
- **r2 IndexCache** (`index_topk_pattern`): tested → improved TPOT → best-achievable. *Flagged.*
- **r3 raised `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`:** **not pursued** — it affects
  dense *prefill* attention (TTFT, which has ~10 s slack), not the binding decode-TPOT metric.
The capacity check did **not** force early FP8 (AC-7.1): bf16 max_total_num_tokens=300,352 ≥
64×4608=294,912, so FP8 KV was introduced only as a (failed) speed experiment, never for capacity.

## Reproducibility metadata
- **SGLang:** version `0.0.0.dev1+g64e2b54a8` (built from upstream `64e2b54a8`); branch
  `perf/sglang-hillclimb-c64`; harness commit recorded in git log. **AC-3:** session commits
  touched only `development/` artifacts — `git diff --name-only e6e411889..HEAD -- python sgl-kernel test` is empty.
- **Model:** `zai-org/GLM-5.1-FP8`, snapshot `f396cf805182f4ca10fa675e1a99815b3ca384db`
  (142 safetensors, FP8 e4m3), served from `/cluster-storage/models/` (756 GB does not fit local disk).
- **Container/OS:** NGC container, Ubuntu 24.04.3. **torch** 2.11.0+cu130, **CUDA** 13.0,
  **driver** 580.105.08.
- **GPU:** 8× NVIDIA H200 (141 GB), SM90; persistence ON; SM clock 1980 MHz (max 1980).
- **Harness:** `development/loop1/run_candidate.sh` (fresh server per candidate: teardown →
  launch → readiness wait → unchanged `development/benchmark.sh` → parse → GPU-mem drain).
  No server state or prefix cache reused across candidates.
- **Result files:** `development/loop1/results/<tag>_isl4096_osl512_c64.jsonl` (+ `.log`);
  per-candidate server-log facts in `development/loop1/logs/facts_<tag>.txt`;
  full sweep table `development/loop1/sweep_table.md`; analysis `development/loop1/analysis_notes.md`.
- **Resolved capacity after final combination** (combo / IndexCache, bf16, page 64,
  mem-fraction 0.85, mrr 64): `max_total_num_tokens=300,352`, KV dtype bf16, DSA
  prefill=flashmla_sparse / decode=fa3, cuda_graph_max_bs=512.

## Bottom line
- **Deploy A (combo)** for a no-accuracy-risk config at **~25 tok/s/user**, P99 TTFT ~12 s.
- **Deploy B (combo+IndexCache)** to reach **~27 tok/s/user** (closest to 30), *after* an
  accuracy eval validates the IndexCache pattern for your quality bar.
- 30 tok/s sustained per user is **not achievable flags-only** on GLM-5.1-FP8 at concurrency
  64 on 8× H200 in this build — the binding cost is MoE-decode compute, which needs expert
  parallelism (out of scope here) or a smaller/faster model.
