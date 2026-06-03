# Round 0 Review Result

Mainline Progress Verdict: ADVANCED

ACs: 2/10 addressed | Forgotten items: 0 | Unjustified deferrals: 1

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline, `round-0-prompt.md`, `round-0-contract.md`, `round-0-summary.md`, `goal-tracker.md`, the Round 0 commit, both new artifacts, and the relevant DS table/write/selection/config/backend code.

## Mainline Gaps

1. **AC-2 is not yet acceptable as a binding downstream decision because it makes AC-3 conditional on a pre-AC-3 fp16 hardware sweep.**

   Evidence:
   - The plan says AC-2 must pick a binding lever for AC-3, with int8 implemented when predicted sufficient; AC-4 is defined as validating the lifted mem fraction **with the compact table** (`development/loop6/refined_plan_v1.md:20`, `:36-40`, `:45-58`, `:115-119`, `:180-184`).
   - The AC-2 artifact says int8 is the compaction path, but then says to sweep fp16 first and build int8 only if that sweep fails (`runs/20260530_dsv32_loop6/footprint_feasibility.md:113-127`).
   - The tracker repeats that drift by making AC-3 int8 conditional on the fp16 sweep (`goal-tracker.md:108` before this review update).
   - The round contract itself says the next round must move to AC-3 footprint code, then AC-4 hardware (`round-0-contract.md:25-28`).

   Impact: Claude can now skip the selected footprint work and run a no-code hardware round, which cannot satisfy AC-3 and cannot satisfy AC-4 as written. It also leaves the "binding lever" ambiguous: fp16 is called the true minimum deployment lever while int8 is called the selected compaction lever.

   Required correction: keep the useful budget math, but revise the AC-2 conclusion/next contract so the binding path is unambiguous: **implement int8 same-`label_dim` for AC-3 next**. A low-f fp16 sweep may be logged as a sanity datapoint, but it must not gate, replace, or precede the AC-3 compact-table implementation.

2. **The Loop remains incomplete.**

   Round 0 correctly avoided source/test/hardware changes, and AC-1 passes. But AC-3 through AC-10 are still pending; under this review prompt, those deferrals mean the work is not complete and the final output must not be `COMPLETE`.

## Blocking Side Issues

1. **AC-2/tracker sequencing drift blocks safe AC-3 execution.**

   I updated the mutable tracker section to record this blocker, marked AC-1 verified, and left AC-2 unverified pending revision. The blocker is not a separate code defect; it is the mainline decision ambiguity above.

## Queued Side Issues

1. **Clean the Anchor B label while revising AC-2.**

   `footprint_feasibility.md:31` still labels Anchor B as `mem_fraction_static≈0.77-0.8`, while the verification note says to treat the same 396K-token anchor as `≈0.70` (`:162`). This is not the primary blocker, but leaving both values in the same artifact invites the next hardware round to target the wrong window.

## Goal Alignment Check

AC-1 is verified: the decision doc states pursue recall R&D only after the engineering spine, names the custom adjustable-`top_k` kernel direction, gives the index-topk/shared-kernel/offline-selector rationale, and preserves DSA default / DS opt-in sequencing.

AC-2 is addressed but not verified: the artifact contains scale overhead, larger-pool feedback, admission math, and the `top-k overlap@2048 >= 0.99` metric, but its conclusion conflicts with the plan’s AC-3/AC-4 dependency order.

AC-3 through AC-10 are tracked in Active Tasks and are not forgotten. AC-10 remains gated by the plan; do not start it before the full Tier-1 spine lands.

## Directive Implementation Plan

1. Revise `runs/20260530_dsv32_loop6/footprint_feasibility.md` and the next contract so the binding decision is: **AC-3 implements int8 same-`label_dim`; page-level/two-stage is only for int8 insufficiency after hardware evidence; fp16 lower-f sweep is only optional instrumentation, not a deployment gate.**

2. Implement AC-3 in the existing DS modules:
   - Add an explicit config field in `DoubleSparsityConfig`, defaulting to fp16 and accepting one compact value such as `int8_symmetric`.
   - Extend `TokenLabelTable` so fp16 remains unchanged, while compact mode allocates `signatures` as `torch.int8` plus static fp16 per-(layer, slot, head) scales. Include signatures + scales in byte accounting.
   - In `token_label_write.py`, gather labels once; fp16 writes follow today’s path, int8 writes symmetric per-vector quantized labels plus scale.
   - In `selection_kernel.py` and graph-safe scoring, multiply loaded int8 labels by the stored scale on device before the dot product. Keep dtype/mode decisions outside captured paths; no host sync, allocation, or Python dtype dispatch inside capture.
   - Wire allocation from `deepseek_v2.finalize_double_sparsity_bind()` using the parsed config; DSA-default boots must still allocate no DS table.

3. Implement AC-3/AC-6 tests before hardware:
   - Config parsing/default tests for fp16 default and compact opt-in.
   - Byte-count tests proving the expected int8+scale reduction.
   - Synthetic selection-equivalence test enforcing `top-k overlap@2048 >= 0.99` vs fp16.
   - DSA-default non-regression test: no DS flag means no `TokenLabelTable`, no DS decode path.
   - CUDA graph-safety test or allocation audit covering the compact graph-safe path.

4. Run AC-3 hardware evidence:
   - Real-mask NIAH non-regression against the Loop-5 fp16 DS baseline.
   - Compact-vs-fp16 decode-scoring microbench tied to the 33.9 -> 30 TPS/req margin.

5. Then run AC-4 with the compact path: sweep `mem_fraction_static` from 0.6 toward 0.8, log full HBM accounting including NVML and `torch.cuda.memory_reserved/allocated`, record `/get_server_info`, and run sustained long `/generate` with no OOM and no monotonic memory growth.

6. Continue the remaining tracked ACs in order: AC-5 client SLO benchmark with attribution, AC-6 hardware opt-in/DSA-default proof, AC-9 real-token within-budget harness edit/re-run, then AC-7/AC-8 hardening. AC-10 starts only after AC-3 through AC-9 land.

## Tracker Update

Updated `.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md` mutable section:
- AC-1 moved to verified.
- AC-2 changed to needs revision.
- Added a blocking issue for the fp16-first / AC-3-conditional sequencing drift.
- Rejected the plan-evolution entry’s claimed impact as a valid refinement.
