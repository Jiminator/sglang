# Loop 9 Ledger — DS-on Decode Kernel Optimization

Authoritative ledger, rewritten after Round 1 (the per-idea history is preserved below,
explicitly labeled as the measured progression; THIS section is the current state).

## Current state (final landed, after Round 1)

Case-1 (DS-on, frozen recipe from development/profiling/plan.md, one trial per run):

| Metric | frozen baseline (20260609) | final landed (20260611, runs/20260611_r1/) |
|---|---|---|
| decode GPU-kernel µs / 10-step window | 632,239 | **480,989** (−23.9%) |
| ratio vs frozen Case-2 DSA floor (342,857) | 1.84× | **1.403×** |
| aggregate decode tok/s | 459 | **654.28** (+43%) |

**All four AC-1 per-bucket gates are MET:**

- **AC-1.1 (score-reduce): MET** — `AllReduce_Sum_f32_RING` eliminated (124,873 → 0); the
  named custom-AR kernel `all_reduce_two_shot_kernel<bf16,8u>` serves the DS reduce
  (backend recorded: `custom_ar_v2`, bf16 two-shot pull at decode buckets; logged NCCL-bf16
  fallback for >16 MB capture buckets). Honest attribution: the win is the bf16 byte halving;
  custom-AR ≈ NCCL at equal bytes.
- **AC-1.2 (top-k): MET with margin** — DS-attributed selection 138.6k → ≈36.3k µs (gate
  ≤80k); torch mbtopk/radixSort/sbtopk/gatherTopK lines at zero; the shipped kernel is the
  deterministic sequence-aware Triton radix suite (exact, tie-deterministic, zero-alloc
  replay). The 20.5k "topk/sort" residual in the category rollup is shared non-DS sorting
  (present identically in Case 2).
- **AC-1.3 (logical-score): MET** — 63,107 → **36,908 µs** (gate ≤40,000), closed in Round 1
  by the persistent-worker kernel (static (bs, ≤128) grid, device-side loops over live
  blocks) plus the radix-path dead-store skip.
- **AC-1.4 (total, secondary): strong marker MET** — 480,989 ≤ 516,000 (minimum 560,000).

**AC-2 held for every landed change**: recall@2048 deltas ≤ +0.010pp (bound 0.5pp); cross-rank
selection bit-identity PASS on every gate run (hard); the top-k/logical-score changes are
selection-bit-identical to the M1 served state (selcap diffs 0/2496). **AC-3 intact** (channel
mask → signatures → scoring → top-k → sparse MLA decode; no dense/DSA fallback). **AC-4
untouched** (no shared kernel modified; the DS-specific AOT op is a new file/op; DS-off
behavior unchanged). **AC-5**: one trial per run, Case 1 only re-profiled, frozen Case-2/3
references reused throughout; deviations recorded in the goal tracker's Plan Evolution Log
and under Notes below.

**The shipped path** (all on the unmodified prebuilt sgl-kernel wheel):
bf16 score-reduce transport through custom-AR v2 (`reduce_token_scores`, fp32 escape hatch via
`score_reduce_dtype`) + the deterministic seq-aware Triton radix top-k (`topk_kernel.py`) +
the persistent-worker logical-score kernel with radix-path dead-store skip.

**M2 two-candidate contract (DEC-4): complete.** All candidates built and measured — exact
fast_topk_v2 wrapper (1530.1 µs/call: correctness around the racy kernel costs back its win),
raw fast_topk_v2 (17.7 µs floor, disqualified: tie-nondeterministic), B-Triton (52.6 µs
captured op-point, shipped), B-AOT one-block-per-row op (true single launch; 43.2 µs op-point
captured but 71.0 at 16k / 629.3 all-live → not integrated on the all-shapes decision). The
AOT op is source-complete in the sgl-kernel tree with registered tests, and the full wheel
build succeeded on this box (op symbol + schema verified in `sm90/common_ops.abi3.so`; wheel
deliberately not installed — frozen-reference protection). Full record:
m2_benchmark_off_final.md; build log: runs/20260611_r1/sgl_kernel_build.log (committed).

**Penalty B**: the audit (reviews/task13_m4_memory_audit.md) found the recoverable memory in
the over-captured decode graph ladder, not signatures; the measured re-tune (mem 0.77 +
cuda-graph-max-bs 64 — a NEW characterized op point, the frozen recipe untouched) lifts the
admission cap **bs 29 → 64 decoding under CUDA graph** (pool 142,208 → 330,048 tokens, capture
pool 17.68 → 0.88 GB). Artifacts: runs/20260611_m4/.

## Measured progression (historical, per landed idea — no current verdicts here)

Per-bucket µs / 10-step decode window, torch TP-0, one trial each:

| Bucket | frozen (20260609) | M0 dry-run | M1 score-reduce | M2+M3 (combined run) | R1 final |
|---|---|---|---|---|---|
| `AllReduce_Sum_f32_RING` (DS reduce) | 124,873 | 124,949 | 0 | 0 | 0 |
| named custom-AR bf16 kernel | 0 | 1,269† | 67,343 | 95,225‡ | 93,480 |
| torch top-k/sort lines (DS-attributed) | 138,602 | ≈same | ≈134,714 | 0 | 0 |
| new radix selection kernels | — | — | — | ≈36,290 | ≈36.3k |
| shared non-DS topk/sort residual | 20,564 | ≈same | ≈same | 20,470 | 20,524 |
| `_logical_score_kernel` | 63,107 | 63,211 | 63,161 | 43,180 | **36,908** |
| **Total** | **632,239** | **631,381** | **585,158** | **512,687** | **480,989** |
| decode tok/s | 459 | 459.4 | 500.75 | 646.79 | 654.28 |
| recall gate | (baseline) | — | PASS +0.010pp | PASS (=M1) | PASS (=M2) |
| cross-rank bit-identity | PASS (M0) | — | PASS | PASS | PASS |

† small pre-existing non-DS usage of that kernel in the baseline trace.
‡ the pull kernel absorbs cross-rank arrival skew in-kernel once the long serializing top-k is
gone (wait, not work); net total still fell 72k that step.

Boot-to-boot noise context: the M0 dry-run reproduced the frozen baseline per-bucket within
~600 µs; shared-kernel variance (trtllm fusion, fp8-quant, MoE) of up to ~27k µs appears
between boots — per-bucket attribution is primary, totals secondary (per DEC-1).

## Frozen M0 baselines (AC-2 references) — captured 2026-06-10

- **Production selection oracle** (CUDA-graph mode, served op-point, fixed 4-prompt
  deterministic workload: 546/2878/6121/12531 prompt tokens × 8 decode steps, 2 passes):
  runs/20260610_m0/selcap_baseline_digest.json — 64 steps × 78 layers × 8 ranks bit-identical,
  contract clean, same-boot AND cross-boot deterministic (fresh-boot digest identical:
  runs/20260610_m0_xboot/). Raw dumps on disk (gitignored; regenerable at the recorded
  commits).
- **NIAH oracle recall@2048 baseline** (eager, config-borne oracle, fixed gated workload:
  1024/4096/16384 words × N=20 × 4 decode steps = 18,720 samples):
  runs/20260610_m0/recall_baseline.json — overall **64.696%** (100.0 / 58.045 / 36.042 per
  length); recall@2048 == selected_contains_needle at every length; all 60 trials
  offline-token == server-token. Gate resolution: 0.5pp ≈ 31 samples per length bucket.
- **Tie-semantics check**: the pre-loop production pipeline's torch.topk tie behavior matched
  the documented (score desc, pos asc) contract on this build (probed at widths 4/4096/202752
  incl. boundary plateaus); the shipped radix kernel enforces that contract by construction.

## Historical context notes (kept for attribution; no current verdicts)

- **M1 premise correction**: the plan assumed a ~534 KB reduce; measured reality is
  [bs, context_len=202752] fp32 ≈ 23.5 MB at bs 29 (matches the frozen 160 µs/call). Custom-AR
  caps: v1 8 MB, v2 16 MB pull / 160 KB one-shot (TP=8 H200); `_ATTN_TP is _TP` under plain
  TP=8. Full evidence: m1_spike_findings.md + runs/20260610_m0/m1_spike.json.
- **M3 measure-first chain**: the dead-grid floor dominated the logical-score kernel
  (TOKEN_BLOCK 64→256 first, then Round 1's persistent-worker grid landed the gate; the
  dead-store-only hypothesis measured at just −1.1 µs/call before the grid restructure).
- The recall-oracle runs measure the eager server (the oracle is host-syncing); pre/post
  comparisons hold the mode constant. The binding production-selection evidence is the
  graph-mode selcap captures.

## Structural headroom after this loop (follow-ons; recorded, not deferred work)

1. **Width-bucketed DS selector graphs + compact per-bucket score buffers** — attacks the
   remaining static-width dead tax in the reduce (the dominant residual: ~93k µs bf16 reduce
   on a ≤4608-token live window); projected total ~1.10–1.15× the DSA floor. A real
   cuda-graph-runner integration change. **Needs-user-decision** for a future loop
   (reviews/task15_m5_wildcard_proposal.md).
2. **Fused multi-block AOT top-k redesign** — several blocks per row with cross-block
   coordination, targeting the 17.7 µs/call floor across ALL context lengths (the landed
   one-block-per-row AOT op wins only at the op point). Ideally folded into (1).
3. **Rebuilt-wheel adoption** — gated op-point change + mandatory DSA regression (DS-off
   smoke + Case-2 re-validation) whenever a wheel containing the AOT op is to be installed.
4. **bs-64 re-tuned op point** (mem 0.77 + graph-max-bs 64) — needs its own SLO/profiling
   characterization loop before becoming a served default.
5. Boot-log `scales=fp16` wording in fp16 mode (no scales sidecar exists) — cosmetic.

## Notes / process deviations (recorded for AC-5)

- M2 and M3 shared one Case-1 gate/profile run (sequencing slip; both selection-bit-identical,
  disjoint buckets — per-bucket attribution exact; recorded in the Plan Evolution Log).
- The original task8 review ran before the final candidates were built; the FINAL
  benchmark-off analyze review over the built artifacts is
  reviews/task8_m2_benchmark_off_final_review.md (Round 2).
- The wheel-build evidence log is force-committed past the repo's *.log ignore because the
  round summaries cite that exact path.
