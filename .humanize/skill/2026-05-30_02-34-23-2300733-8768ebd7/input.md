# Ask Codex Input

## Question

You are doing a SECOND-PASS reasonability review of a CANDIDATE implementation plan. The plan already incorporated your first-pass critique. Challenge what remains. Be specific and adversarial where warranted, but acknowledge what is now sound.

## Project (verified, recap)
SGLang serving framework. Make **Double Sparsity (DS)** pass the absolute client SLO on **DeepSeek-V3.2 FP8**: 4096 ISL / 512 OSL / conc 16-64 / ~55% cache, at fixed "Option B" operating point (TP=8, fp8 KV, page 64, flashmla_kv prefill+decode, overlap-schedule + piecewise-cuda-graph DISABLED, radix on). SLO = **P99 TTFT < 22 s AND >= 30 TPS/req** (absolute, not DS-vs-DSA ratio).

Loop-5 state: DS gen TPS 34/33.9/33.9 (PASSES). P99 TTFT 57.7/132.9/292.0 s (FAILS). Cause: ~8 GB/rank fp16 TokenLabelTable forces `mem_fraction_static=0.6` (DSA uses 0.85); small KV pool admits 14.5/24.6/35.7 of nominal 16/32/64 → queueing → TTFT blows up; raising mem past 0.6 OOMs DS in generation. NIAH recall DS 75/5/0 vs DSA 100. FlashMLA decode hard-asserts `indices.shape[-1] == dsa_index_topk(=2048)`. Config surface `DoubleSparsityConfig` REJECTS unknown fields (flag must be a new allowed field or go in `extra`). `benchmark_compare.py` ALREADY has `SLO_TTFT_P99_S=22.0` and computes `ttft_p99_s` + `per_request_output_tps_p50>=30`. The within-budget proxy's `length_tokens` is actually a WORD count.

## CANDIDATE PLAN v1 (review this)

GOAL: Shrink the per-rank TokenLabelTable so DS boots at higher admission headroom with no generation OOM, restoring concurrency so DS passes the absolute client SLO on real H200 hardware; plus Tier-1 hardening; plus a strategic gate (DEC-1) deciding whether Tier-2 long-context recall R&D happens at all. KEY REFRAME (from your v1): the target is NOT "mem_fraction=0.8" per se — it is "enough admitted KV-pool slots / concurrency with HBM headroom to drop P99 TTFT < 22 s," and the SLO claim must separate admission-wait from prefill-compute because TTFT may be prefill-bound at conc 64.

ACCEPTANCE CRITERIA (each drops an artifact under runs/<date>_dsv32_loop6/):
- AC-1 (strategic gate, analyze): decision doc `ds_on_v32_decision.md` records pursue-Tier-2-or-not with index_topk/shared-kernel/selector rationale + explicit Tier-2 consequence. Neg: any Tier-2 work before the doc exists is out of order.
- AC-2 (TokenLabelTable footprint, coding): per-rank table bytes reduced by target factor; DS selection numerics preserved within an explicit tolerance; compact path flag-gated with fp16 the default until hardware-validated; CUDA-graph-safe (preallocated static-shaped scales, no host sync / Python dtype dispatch in captured path); mask-compatibility addressed (whether narrowing label_dim requires mask regeneration or slicing the Loop-5 mask is valid). Pos: unit test shows compact selected-token set matches fp16 baseline within tolerance (top-k overlap@2048 / selected-token recall / score-error distribution) + measured per-rank byte drop. Neg: selection divergence beyond tolerance, OR compact path default before hardware validation, OR DS flags allocating tables / altering decode when DSA is selected (DSA non-regression).
- AC-3 (mem-fraction lift + no-OOM, coding/hwrun): DS boots at the lifted point and survives sustained long /generate. Pos: a mem-fraction sweep log (0.6 -> target) shows max_total_num_tokens rising; full HBM budget logged (weights + KV pool + table + scales + written + score scratch + FlashMLA metadata + CUDA-graph pool + headroom); no generation-time OOM; no monotonic memory growth over the full run; /get_server_info recorded. Neg: generation OOM at target, or hidden memory growth.
- AC-4 (client-SLO validation — DONE-CRITERION, coding/hwrun): full workload benchmark.sh NUM_PROMPTS=320 conc 16/32/64. Pos: DS absolute P99 TTFT < 22 s AND >= 30 TPS/req at conc 16 and 64; `client_slo_report.md` with absolute numbers vs SLO + valid .meta.json sidecars + an admission-wait vs prefill-compute attribution (or documented attribution method if server lacks the split). Neg: any conc with P99 TTFT >= 22 s or < 30 TPS/req fails the MVP claim (recorded as follow-up with the admission/compute breakdown).
- AC-5 (AC-11 directional re-sweep, coding/hwrun): 3-trial DS+DSA sweep at lifted point, radix-on both sides, per-side mem_fraction consistency enforced. Pos: DS achieved concurrency tracks nominal (~100%); refreshed ac11_analysis.md. Neg: a sweep hiding queue-dominated admission (achieved << nominal undisclosed).
- AC-6 (64K servability, coding/hwrun): Pos: ~70K-token /generate returns 200 at lifted mem with served max_total_num_tokens recorded AND no OOM/instability; OR a documented new admission ceiling. Neg: silently re-recording the Loop-5 HTTP 400 without the lifted-mem retry.
- AC-7 (within-budget from real token counts, coding): harness records usage.prompt_tokens per NIAH prompt, asserts within_budget from it, fails CLOSED if usage missing/inconsistent; rename length_tokens->length_words or add input_tokens; re-run gate still PASSES (DS-fair definition UNCHANGED) + diff vs word-count proxy. Neg: any change to the DS-fair gate thresholds/definition.
- AC-8 (Tier-2 recall R&D — GATED, coding/hwrun, ONLY if AC-1 opened it): selector or top_k>index_topk kernel variant with NIAH 4K/16K/64K recall delta vs DS 75/5/0 + TPS/TTFT cost. Neg: starting before AC-1 or letting it block the Tier-1 spine.

PATH BOUNDARIES:
- Upper: all Tier-1 ACs pass on hardware with full HBM accounting + TTFT attribution; Tier-2 explored if DEC-1 opens it.
- Lower: footprint reduction (AC-2) + mem lift no-OOM (AC-3) + client-SLO validation (AC-4) + strategic gate decided (AC-1) + token-count gate (AC-7); Tier-2 deferred; 64K (AC-6) and AC-11 re-sweep (AC-5) may be CHARACTERIZED (documented ceiling/root-cause) rather than fully passing if hardware reveals a deeper bottleneck (e.g. prefill-bound TTFT).
- Allowed: pick the MINIMUM compaction lever that works — int8-symmetric signatures with per-layer/slot/head scales applied at scoring, narrower label_dim, tighter/page-level/two-stage slot model. Reuse Loop-5 mask + serve/bench scripts + comparator. fp16 default behind a flag. CANNOT: change the DECIDED DS-fair AC-12 gate definition; rebuild scaffolding; put plan-process markers (AC-/DEC-/Tier/Option B/Round N) in code; start Tier-2 before DEC-1.

PENDING USER DECISIONS (discussion mode resolves with the user): DEC-1 pursue Tier-2?; DEC-2 "shippable" = DS meets SLO itself vs DS opt-in while DSA default; DEC-3 confirm absolute P99 TTFT<22s at NUM_PROMPTS=320; DEC-4 footprint lever + target mem_fraction (0.7/0.8) + OOM-safety/HBM-headroom bar + mask-regeneration allowed?; DEC-5 deployment topology single-node TP=8 vs multi-node; DEC-6 (NEW, from your v1) may production serve flags differ from the fixed Loop-5 scripts (esp. CUDA graphs / overlap scheduling), or is the measured shipping mode exactly Option B?

TASK ROUTING: gen-plan template allows only `coding` or `analyze` tags. Hardware-run ("hwrun") tasks map to `coding` (Claude drives the cluster serve/bench scripts; Codex never touches hardware). Strategic-gate / analysis tasks are `analyze`.

## Required output format (exact headers)
AGREE: points accepted as reasonable.
DISAGREE: points considered unreasonable and why.
REQUIRED_CHANGES: must-fix items before convergence.
OPTIONAL_IMPROVEMENTS: non-blocking improvements.
UNRESOLVED: opposite opinions needing user decisions.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 900s
- Timestamp: 2026-05-30_02-34-23
- Tool: codex
