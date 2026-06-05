# GLM-5.1-FP8 Flags-Only Hill-Climb — Final Report

**Goal:** drive `zai-org/GLM-5.1-FP8` on one node of 8× H200 (TP8) as close as possible to the
client SLOs using **only** SGLang CLI flags + `SGLANG_*` env vars (no source / benchmark /
workload / SLO edits), measured with the fixed `development/benchmark.sh`.

**Workload (fixed):** generated-shared-prefix, 4096 ISL (2253 shared system prompt + 1843
question) / 512 OSL, 320 prompts, max-concurrency 64, ~55% prefix-cache hit, greedy, fixed seed 31234.

## SLOs and how they are measured (read this first)

The client specified **"30 TPS"** and **"P99 TTFT < 22 s"**. This report presents the per-user
speed verdict under **two metrics** (a deliberate reconciliation — see Plan Evolution PE-1):

1. **Plan-designated scalar** (`development/loop1/plan.md` DEC-1, implemented by `parse_result.py`):
   **`median_itl_ms ≤ 33.3`** (client's stated `1000/ITL` form), `p99_ttft_ms < 22000`. Kept for
   plan traceability.
2. **Client ground-truth TPS** (verbatim from the requirements owner, who clarified the draft's
   "(or 1000/ITL)" was an in-house gloss, since retracted): *"(total latency − TTFT) / total
   tokens"* → **TPS = Σtokens / Σ(decode_time) ≈ 1000 / mean_tpot_ms**. Treated as authoritative.

Why they differ: under EAGLE speculation `1000/median_ITL` is inflated ~2.3× by 3-token bursts
that deflate median ITL, so it reads "met" while the sustained decode rate (TPS / TPOT) does not.
Both are reported; the ground-truth TPS is the substantive engineering verdict.

## Result summary

| metric | safe (combo) | best achievable (combo+IndexCache) ⚠ | target | met? |
|---|---|---|---|---|
| **Plan scalar** `median_itl_ms` | ~17 ms (~57/s) | ~17 ms (~58/s) | ≤ 33.3 ms | **✅ MET** (both) |
| **Client ground-truth TPS** `Σtok/Σdecode` | **24.3** | **26.5** | ≥ 30 | ✗ (gap 5.7 / **3.5**) |
| **P99 TTFT** | 12.1 s | 11.4 s | < 22 s | **✅ MET** (45–48 % margin) |

**Verdict:**
- **Under the plan-designated scalar** (`median_itl ≤ 33.3` AND `p99_ttft < 22 s`): **target MET**
  by both `combo` and `combo+IndexCache`.
- **Under the client's ground-truth TPS formula:** **30 TPS NOT met flags-only** — best achievable
  **26.5 TPS** (combo+IndexCache, accuracy-risk), gap ~3.5 (≈12 %); safe `combo` 24.3 TPS, gap
  ~5.7 (≈19 %). P99 TTFT met either way.

The divergence is the speculative-burst artifact: the plan scalar is satisfied, but true sustained
per-user generation (the client's actual formula) falls ~3.5 TPS short. TPOT figures below are
**sustained-decode-rate analysis**, equivalent to the client TPS metric, not a separate SLO.

> Aggregation note: averaging the *per-request* rate (mean of tokenᵢ/decodeᵢ) reads higher (~29.8
> for IndexCache) because it over-weights fast short-decode requests; the client's literal
> "total ÷ total" does not, so 26.5 TPS is the honest figure.

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
Confirmed (3 fresh runs): **client TPS (Σtokens/Σdecode) = 24.2 / 24.3 / 24.5 → ~24.3 TPS**;
mean_TPOT ~41.2 ms; p99_TTFT ~12.1 s; accept ~3.1; 320/320, 0 errors.
bf16 KV, page 64, DSA prefill=flashmla_sparse / decode=fa3, max_total_num_tokens=300,352.

### B) Best-achievable — **ACCURACY-RISK (IndexCache, flagged)**
Same as (A) plus:
```
  --json-model-override-args '{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'
```
Confirmed (3 fresh runs): **client TPS (Σtokens/Σdecode) = 26.6 / 26.5 / 26.6 → ~26.5 TPS**;
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
- **FP8 KV regressed in all 3 configs tested** (baseline / combo / combo+IndexCache → 21.9 /
  21.7 / 22.9 TPS vs their bf16 counterparts 23.8 / 24.3 / 26.5). FP8 KV is **fully permitted**
  here (user confirmed), so it is rejected on **merit**: it forces DSA→`flashmla_kv` for
  prefill+decode (slower than bf16 `fa3`) and doubles capacity we don't need — confirming the
  KV/attention path is **not** the bottleneck.
- **IndexCache is the only knob that moved the binding metric** (+2.2 TPS, 24.3→26.5) because
  it cuts *decode-path* indexer compute (reused across layers), not KV bandwidth.

The remaining gap is MoE-decode FLOPs, which no flag addresses without expert parallelism
(EP / a2a) — explicitly out of scope for this fixed TP8 path.

## Sweep table (20 distinct candidates + 4 confirmation reruns = 24 fresh-server runs)

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
| fp8kv | baseline +FP8 KV | 45.6 ✗ | 45.62 | 13.7 s | FP8 regresses (21.9 TPS) |
| **combo** | chunk4096 + lpm | 39.5 | 41.4 | 12.1 s | **safe winner ×3 → 24.3 TPS** |
| combo+FP8 | combo +FP8 KV | 46.3 ✗ | 46.25 | 14.5 s | FP8 regresses (21.7 TPS) |
| **indexcache** | combo + IndexCache | 36.5 | 37.7 | 11.4 s | **best ×3 → 26.5 TPS** (acc-risk) |
| index+FP8 | combo+IndexCache +FP8 | 43.7 ✗ | 43.69 | 13.8 s | FP8 regresses (22.9 TPS) |

(TPOT columns are diagnostic; the official metric is client TPS = Σtokens/Σdecode, shown in the note column.)

## Page-size (AC-4 — requirement WAIVED by owner, PE-2)
The requirements owner stated **"page size 64 is no longer a requirement at all"**, so the AC-4
page-size-flexibility / no-preference-for-64 requirement is **waived** and no winner-level
page-size runs were spent. For completeness, the earlier finding stands: `--page-size 32`
launched + benchmarked but the server resolved **page_size=64** — on CUDA, DSA unconditionally
sets 64 (`python/sglang/srt/server_args.py:1918-1920`; FlashMLA "only supports a page_size of
64", `:2852`). So even absent the waiver, GLM-5.1 DSA on H200 supports exactly **one** effective
page size (64); the CLI accepts alternate `--page-size` flags but the effective size cannot vary
without source changes.

## Out-of-scope axes — confirmed ABSENT in both winners (hard constraint)
Neither config uses EP / MoE a2a (`--moe-a2a-backend`, deepep), alternate MoE runners,
`--enable-torch-compile`, NGRAM speculative, or pd-multiplexing. Parallelism is **TP8**
(`tp_size=8, ep_size=1, dp_size=1, moe_a2a_backend=none`, verified in server_args). No sweep
budget was spent crash-probing excluded axes.

## Lower-risk ladder — exhaustion evidence (AC-7)
Before any accuracy-risk knob is treated as best-achievable, the non-accuracy-risk ladder was
swept on the safe incumbent base (`combo` = EAGLE steps3/topk1/draft4, mem0.85, mrr64,
chunked-prefill 4096, lpm), fresh server each, all 320/0 err. Client TPS (Σtok/Σdecode):

| candidate | knob vs combo | client TPS | accept | result |
|---|---|---|---|---|
| combo (incumbent) | — | 24.2 | 3.09 | safe reference |
| combo_mrr80 | max-running-requests 80 | 24.4 | 3.15 | ≈combo (noise; conc capped at 64) |
| combo_mrr96 | max-running-requests 96 | 24.1 | 3.08 | ≈combo (conc capped at 64) |
| combo_mem90_cg64 | mem-fraction 0.90 + cuda-graph-max-bs 64 | 23.8 | 2.98 | no gain (not capacity-bound) |
| eagle_xlight | spec steps1/draft2 | 22.3 | **1.88** | worse (accept collapses) |
| dsa_decode_sparse | bf16 decode=flashmla_sparse | 23.9 | 3.09 | neutral (≈fa3) |
| dsa_pf_auto | bf16 prefill=flashmla_auto | 24.0 | 3.09 | neutral |

**Conclusion:** every lower-risk knob lands at ~24 TPS — none beats the incumbent. Admission
knobs (mrr80/96) can't help because the workload caps concurrency at 64; capacity knobs
(mem0.9) can't help because bf16 KV is not the constraint; lighter speculation hurts; DSA bf16
backend swaps are neutral (decode is pinned to `fa3`-class cost). Lower-risk is **exhausted**;
only the accuracy-risk IndexCache (26.5) moves the metric — satisfying AC-7's ordering.

## Accuracy-risk ladder (hard constraint) — how it was respected
Non-accuracy-risk knobs were exhausted first (scheduler capacity, DSA backends under bf16,
speculative params, DP-vs-TP, page size, schedule policy — see table above). Then, in strict order:
- **FP8 KV** (`--kv-cache-dtype fp8_e4m3`): **fully permitted** (user-confirmed, not gated).
  Tested in 3 configs (baseline, combo, combo+IndexCache) → **regressed every time** (forces
  slower `flashmla_kv` decode; not capacity-bound) → rejected on merit.
- **IndexCache** (`index_topk_pattern`): tested → improved TPS → best-achievable. *Accuracy-risk flagged.*
- **r3 raised `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`:** **not pursued** — it affects
  dense *prefill* attention (TTFT, which has ~10 s slack), not the binding decode metric.
The capacity check did **not** force early FP8 (AC-7.1): bf16 max_total_num_tokens=300,352 ≥
64×4608=294,912, so FP8 KV was introduced only as a (failed) speed experiment, never for capacity.

## Reproducibility metadata
- **SGLang:** version `0.0.0.dev1+g64e2b54a8` (built from upstream `64e2b54a8`); branch
  `perf/sglang-hillclimb-c64`; harness commit recorded in git log.
- **AC-3 (flags-only), complete diff since plan-setup `50c72c79a`:**
  `git diff --name-only 50c72c79a..HEAD -- python sgl-kernel test development/benchmark.sh`
  is **EMPTY** — no SGLang source, kernel, test, or benchmark-harness edits; the winning configs
  are reproducible from flags + env alone. The **only** non-artifact change is
  `development/CLIENT_SLOS.md` (one SLO line: `30 TPS per user (or 1000/ITL)` →
  `30 TPS (Total Latency − TTFT / total tokens)`), an **owner-authorized target-definition
  correction** (the client never specified `1000/ITL`; see PE-1). It changes no workload,
  dataset, or measurement input — every benchmark number comes from the unchanged
  `benchmark.sh` — so it is not the performance-affecting SLO tampering AC-3 prohibits.
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
- **Plan-designated scalar** (`median_itl ≤ 33.3` AND `p99_ttft < 22 s`): **MET** by both `combo`
  and `combo+IndexCache` — so against the plan's literal acceptance gates, the target passes.
- **Client ground-truth TPS** (`Σtokens/(Σlatency−ΣTTFT)`): **not met flags-only** —
  - **Deploy A (`combo`)** = no-accuracy-risk, **24.3 TPS**, P99 TTFT ~12 s.
  - **Deploy B (`combo+IndexCache`)** = **26.5 TPS** (closest to 30, gap ~3.5), *after* an accuracy
    eval validates the IndexCache pattern for your quality bar (accuracy-risk flagged).
- **Lower-risk ladder exhausted** (mrr80/96, mem0.9+cuda-graph, lighter EAGLE, bf16 DSA backends
  all ≈24 TPS); **FP8 KV regresses** in all 3 configs (fully permitted, rejected on merit).
- **30 TPS sustained is not achievable flags-only** on GLM-5.1-FP8 at concurrency 64 on 8× H200 in
  this build — the binding cost is MoE-decode compute, which needs expert parallelism (out of
  scope) or a smaller/faster model. P99 TTFT is met with wide margin throughout.
