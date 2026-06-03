# Ask Codex Input

## Question

CONFIRMATION convergence round. The candidate plan v2 below applied ALL of your round-1 REQUIRED_CHANGES. Verify convergence: respond with whether any REQUIRED_CHANGES remain. Use the same exact output headers (AGREE / DISAGREE / REQUIRED_CHANGES / OPTIONAL_IMPROVEMENTS / UNRESOLVED).

## Context recap (verified facts)
SGLang; make Double Sparsity (DS) pass absolute client SLO on DeepSeek-V3.2 FP8: 4096 ISL / 512 OSL / conc 16/32/64 / ~55% cache, fixed Option B (TP=8, fp8 KV, page 64, flashmla_kv, overlap+piecewise-graph OFF, radix ON). SLO = strict P99 TTFT < 22 s AND >= 30 TPS/req. Verified: serve_double_sparsity.sh defaults DS radix-OFF (`--disable-radix-cache`) unless RADIX_FIXTURE_ARTIFACT is supplied; benchmark_compare.py uses `ttft_p99_s <= 22.0` (boundary false-pass vs strict <22); DoubleSparsityConfig rejects unknown fields; FlashMLA decode hard-asserts indices.shape[-1]==dsa_index_topk(=2048).

## CANDIDATE PLAN v2 (round-1 changes applied)

ACCEPTANCE CRITERIA (every AC copies its acceptance artifact into runs/<date>_dsv32_loop6/, even when the underlying tool writes to development/results):

- AC-1 (strategic gate, analyze): ds_on_v32_decision.md records pursue-Tier-2-or-not + index_topk/shared-kernel/selector rationale + explicit Tier-2 consequence. Neg: any Tier-2 work before the doc.

- AC-2.0 (footprint FEASIBILITY budget, analyze — PRE-CODING, gates DEC-4): compute required freed HBM, scale-storage overhead, target max_total_num_tokens, and expected achieved concurrency at conc 64, for each candidate lever; pick the minimum lever PREDICTED to restore nominal admission with headroom. Pos: a budget artifact with the numbers and the chosen lever's predicted achieved-conc@64. Neg: starting footprint coding before this budget exists.

- AC-2.1 (TokenLabelTable footprint reduction, coding): per-rank bytes reduced; compact path flag-gated (new allowed config field or `extra`), fp16 default until hardware-validated; CUDA-graph-safe (preallocated static-shaped scales, no host sync / Python dtype dispatch in captured path). Validation SPLIT BY LEVER CLASS:
  - Quantization class (same label_dim, e.g. int8-symmetric + per-(layer/slot/head) scales applied at scoring): Pos = unit test shows compact selected-token set matches fp16 baseline within explicit tolerance (top-k overlap@2048 / selected-token recall / score-error distribution) + measured per-rank byte drop.
  - Structural class (narrower label_dim / page-level / two-stage selector): does NOT "preserve numerics"; Pos = regenerated/saved mask artifact + NIAH quality non-regression vs Loop-5 DS baseline + measured byte drop.
  - Config/unit test: compact-flag path parses; a DSA-default boot allocates NO DS table and does not alter decode (DSA non-regression).
  - Neg: selection divergence beyond tolerance (quant class) or NIAH regression (structural class); compact path default before hardware validation; OR any Tier-1 change touching/bypassing the FlashMLA indices.shape[-1]==dsa_index_topk assert (Tier-1 ABI is locked: top_k == dsa_index_topk == 2048).

- AC-3 (mem-fraction lift + no-OOM, coding/hwrun): Pos = sweep log 0.6->target with max_total_num_tokens rising; full HBM budget logged INCLUDING NVML/torch reserved+allocated residual (not only named tensors): weights + KV pool + table + scales + written + score scratch + FlashMLA metadata + CUDA-graph pool + headroom; sustained long /generate completes with no generation-time OOM and no monotonic memory growth over the run; /get_server_info recorded. Neg: generation OOM at target, or hidden memory growth.

- AC-4 (client-SLO validation — DONE-CRITERION, coding/hwrun): benchmark.sh NUM_PROMPTS=320 at ALL conc 16/32/64, full 4096 ISL / 512 OSL / ~55% cache, radix-on PROVEN from server args / .meta.json sidecars (RADIX_FIXTURE_ARTIFACT present). Trial aggregation rule DEFINED BEFORE running: all trials pass, OR median pass with the worst trial disclosed (no failed trial hidden behind a summary). Pos = DS strict P99 TTFT < 22.0 s AND per-request output TPS p50 >= 30 at every conc 16/32/64; client_slo_report.md with absolute numbers vs SLO + valid sidecars + MEASURED admission-wait vs prefill-compute attribution (queue/admission wait vs prefill compute) — OR an explicit statement that attribution is unavailable, in which case the report makes NO root-cause claim. The strict `<22.0` is asserted in the report even though benchmark_compare.py's gate uses `<=22.0`. Neg: any conc with P99 TTFT >= 22.0 or TPS < 30 fails the MVP claim (recorded as a follow-up with the breakdown).

- AC-5 (AC-11 directional re-sweep, coding/hwrun): 3-trial DS+DSA at lifted point, radix-on both sides (proven), per-side mem_fraction consistency enforced. Pos = DS achieved concurrency tracks nominal (~100%); refreshed ac11_analysis.md. Neg: a sweep hiding queue-dominated admission (achieved << nominal undisclosed).

- AC-6 (64K servability, coding/hwrun): Pos = ~70K /generate returns 200 at lifted mem with served max_total_num_tokens recorded and no OOM/instability; OR a documented/characterized new admission ceiling (a 70K failure is a CHARACTERIZED ceiling, not a pass). Neg: silently re-recording the Loop-5 HTTP 400 without the lifted-mem retry.

- AC-7 (within-budget from real token counts, coding): harness records usage.prompt_tokens per NIAH prompt and asserts within_budget from it, failing CLOSED if usage missing/inconsistent; rename length_tokens->length_words or add input_tokens; re-run gate still PASSES (DS-fair definition UNCHANGED) + diff vs word-count proxy; emits artifact into runs/<date>_dsv32_loop6/ (harness natively writes development/results). Neg: any change to the DS-fair gate thresholds/definition.

- AC-8 (Tier-2 recall R&D — GATED, coding/hwrun, ONLY if AC-1 opened it): the ONLY place permitted to relax the FlashMLA top_k>index_topk assert and/or change the selector; NIAH 4K/16K/64K recall delta vs DS 75/5/0 + TPS/TTFT cost. Neg: starting before AC-1, or letting it block the Tier-1 spine.

PATH BOUNDARIES:
- Upper: all Tier-1 ACs pass on hardware with full HBM accounting + measured TTFT attribution; Tier-2 explored if DEC-1 opens it.
- Lower: AC-1 + AC-2.0/2.1 + AC-3 + AC-4 + AC-7 pass; Tier-2 deferred; AC-5/AC-6 may be CHARACTERIZED (documented ceiling/root-cause) if hardware reveals a deeper bottleneck (e.g. prefill-bound TTFT at conc 64).
- Allowed: minimum compaction lever (int8-symmetric + per-slot/head scales / narrower label_dim / page-level / two-stage), constrained by AC-2.0; reuse Loop-5 mask + scripts + comparator; fp16 default behind a flag. CANNOT: change the DECIDED DS-fair AC-12 gate; rebuild scaffolding; put plan markers (AC-/DEC-/Tier/Option B/Round N) in code; start Tier-2 or touch the FlashMLA assert before DEC-1.

PENDING USER DECISIONS (discussion mode; DEC-6 REMOVED — Option B is the fixed target, production-flag variation is post-MVP characterize-later only): DEC-1 pursue Tier-2?; DEC-2 "shippable" = DS meets SLO itself vs DS opt-in while DSA default (must not weaken MVP success def); DEC-3 confirm SLO is absolute strict P99<22s at NUM_PROMPTS=320; DEC-4 footprint lever + target mem_fraction + HBM-headroom bar + mask-regeneration-allowed?, CONSTRAINED by AC-2.0 feasibility budget; DEC-5 deployment topology single-node TP=8 vs multi-node (SLO claim still uses fixed TP=8 Option B).

TASK ROUTING: only `coding`/`analyze` tags. hwrun maps to `coding` (Claude drives cluster scripts); gate/feasibility analysis = `analyze`.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 900s
- Timestamp: 2026-05-30_02-40-35
- Tool: codex
