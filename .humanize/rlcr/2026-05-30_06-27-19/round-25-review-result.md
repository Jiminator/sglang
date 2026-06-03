# Round 25 Review Result

Mainline Progress Verdict: STALLED

Goal Alignment Summary: ACs: 9/10 addressed | Forgotten items: 0 | Unjustified deferrals: 2

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and
taste-review guidance, `round-25-prompt.md`, `goal-tracker.md`, R22-R24 summaries/reviews, commits
`27fca1102` and `5d65eed25`, the R25 handoff docs, the AC-5 full-context verifier, and the current DS
selection/decode hot spots.

## Implementation Review

R25 correctly fixes the one R24 evidence-hygiene issue. `topk_design_microbench.json` and its generator now
state that B is the only measured design reaching conc-16 `>=30` and that C/C′ are worse than monolithic. The
timings are unchanged, and this aligns with `ac5_topk_design_finding.md`.

The roadmap, Loop-7 draft, and as-built architecture doc are useful handoff documentation for a lower-bound
close. They do not complete the original plan. The original done criterion reserves "done" / "shippable" for
strict `P99 TTFT < 22.0s` and `>=30 TPS/req` at every conc 16/32/64
(`development/loop6/refined_plan_v1.md:7`, `:66-70`, `:399-403`). The current fail-closed verifier still
recomputes:

```text
c16: P99 TTFT 13.13s, TPS 24.9
c32: P99 TTFT 25.33s, TPS 19.5
c64: P99 TTFT 77.90s, TPS 17.3
```

So AC-5 is directional-complete only. AC-10 is also not implemented: the plan requires a custom
adjustable-`top_k` sparse-matmul kernel and/or learned selector plus NIAH 4K/16K/64K recall-delta artifact
(`development/loop6/refined_plan_v1.md:104-108`). R25 created `development/loop7.md/draft.md`, but that is a
draft, not the AC-10 implementation or hardware evidence.

## Mainline Gaps

1. **AC-5 strict client SLO remains unmet.**

   Evidence: the committed AC-5 verifier still fails TPS at all conc and TTFT at c32/c64. The R25 docs
   acknowledge this as downstream/deferred (`development/roadmap.md:171-176`,
   `development/past_implementations/study/08-current-system-architecture.md:183-188`). Under this review's
   rules, that is incomplete work, not closure.

2. **AC-10 is deferred, not completed.**

   Evidence: R25 explicitly moves Tier-2 recall R&D to Loop 7 (`development/roadmap.md:186-192`,
   `development/loop7.md/draft.md:59-79`). The live code still preserves the Tier-1 ABI lock:
   `retrieve_topk_graph_safe` selects `max_top_k` with full-width `torch.topk`
   (`python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:932-999`), and FlashMLA decode
   still asserts `indices.shape[-1] == self.dsa_index_topk`
   (`python/sglang/srt/layers/attention/dsa_backend.py:2148-2151`). No adjustable-`top_k` DS decode path or
   learned selector recall artifact landed.

## Blocking Side Issues

None separate from the two mainline gaps. The blockers are the unfinished ACs themselves.

## Queued Side Issues

1. Cross-node benchmark wrapper smoke remains queued for future remote-host artifacts.
2. DSA-default c64 TPS around 29.4 remains a product/client-SLO tension. It becomes directly relevant if the
   strict DS all-concurrency SLO work proves DS is bounded by the same decode path.

## Goal Alignment Check

| AC | Status | Evidence / issue |
|----|--------|------------------|
| AC-1 | MET | Strategic gate doc verified earlier. |
| AC-2 | MET | Footprint budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, scale-aware consumers, launcher, real-mask NIAH, and decode microbench verified earlier. |
| AC-4 | MET | DS int8/mem0.7 point, full HBM budget, no-OOM long generate, and NVML plateau verified earlier. |
| AC-5 | PARTIAL / DIRECTIONAL | Full strict SLO is not met: c16 TPS <30; c32/c64 TTFT and TPS miss. |
| AC-6 | MET | Opt-in DS and DSA-default/no-table product proof verified under the R12 owner-approved non-regression semantics. |
| AC-7 | MET / CHARACTERIZED | Verified in R15 as characterized/soft-met. |
| AC-8 | MET | 64K servability at lifted DS int8/mem0.7 point verified in R16. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in R10. |
| AC-10 | NOT MET | Deferred to Loop 7 draft only; no kernel/selector implementation or recall-delta artifact. |

Forgotten items: none. The strict SLO miss and AC-10 are represented, but R25 tried to leave them deferred.

Deferred items: AC-10 deferral is acceptable only for the Loop-6 Lower Bound. It is unjustified for this
full-completion review because the prompt explicitly says deferrals are incomplete and must not be treated as
done. AC-5 strict all-concurrency SLO was also downstreamed, so it must drive the next implementation work.

Plan evolution: R25's owner-directed handoff is valid context for a lower-bound close. I reject the requested
terminal/no-active-task tracker state for the all-AC completion gate.

## Required Implementation Plan

1. **Reopen AC-5 strict as the next mainline.** The next contract must target strict all-concurrency DS SLO,
   not documentation or Loop-7 handoff. Keep the R24 microbench result: do not spend another round on the
   full-context blocked-topk design family that already measured worse than monolithic.

2. **Produce a strict-SLO gap artifact before changing code.** Under
   `runs/20260530_dsv32_loop6/ac5_strict_slo/`, capture DS and DSA steady-state profiles at the locked
   DS int8/mem0.7/radix-on/TP=8 point for conc 16/32/64. The artifact must split queue, prefill, decode,
   DS selection, FlashMLA decode, and scheduler/admission time, and must name the single code-owned bottleneck
   that will be changed.

3. **Implement that bottleneck in the real hot path and prove it before a full rerun.** If the bottleneck is
   DS selection, change `retrieve_topk_graph_safe` / `DSGraphState` and cover graph replay. If it is shared
   decode, change the DSA/FlashMLA decode dispatch path without weakening the default DSA ABI. If it is
   scheduling, change the scheduler path that mixes prefill/decode or controls active decode batch. Publish a
   closed-batch proof that c16 reaches `>=30 TPS/req` before spending a full AC-5 rerun.

4. **Rerun AC-5 after the code fix.** Run conc 16/32/64, 4096 ISL / 512 OSL, radix-on, single-node TP=8, with
   sidecars, exact arrays, admission/prefill/decode attribution, and a fail-closed verifier. The strict pass
   requires every conc to meet `P99 TTFT < 22.0s` and `>=30 TPS/req`; directional language is not sufficient.

5. **Then implement AC-10.** Add a DS-only adjustable-`top_k` decode path rather than weakening the default
   DSA assert. The implementation should introduce an explicit config surface for the wider DS recall mode,
   allocate DS buffers using `self.ds_max_top_k`, route DS decode with `top_k > dsa_index_topk` through a
   kernel that supports the wider index shape, preserve the default `flashmla_kv` assert for DSA, and publish
   NIAH 4K/16K/64K recall deltas vs 75/5/0 plus TPS/TTFT cost.

## Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 25 Review.
- Added a `25-review` plan-evolution row.
- Reopened strict AC-5 and AC-10 as Active Tasks.
- Removed AC-10 from Explicitly Deferred for this full-completion checkpoint.
- Kept the R25 JSON-note hygiene issue marked resolved.

## Validation Performed

- `python3 runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py --verify` -> PASS, with strict
  misses visible: c16 13.13s/24.9 TPS, c32 25.33s/19.5, c64 77.90s/17.3.
- Inspected `topk_design_microbench.py/json`; R25 note now matches measured rows.
- `git diff --check` -> clean.

NOT COMPLETE
