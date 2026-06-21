# Code Review - Round 0

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop12/plan.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-0-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Loop 12 — Round 0 Summary

## Objective
Extract the minimal correct table-free Double Sparsity (DS) runtime from the dev branch onto a fresh
branch off latest `origin/main` in the v2 shipping clone, add one perf eval, and prove parity. All
shipping work is in `/sgl-workspace/double-sparisty-v2/sglang` (branch `double-sparsity-v2`, off
`<BASE>=105e095e0`); loop machinery stays in this dev clone.

## Outcome: COMPLETE — all 10 ACs pass, branch pushed to the fork

| AC | Result |
|----|--------|
| AC-1 branch hygiene | 42-file clean diff vs `<BASE>`; zero `.pensieve`/`.humanize`/`development`/`SLOS.md`; pushed to `Jiminator/sglang` |
| AC-2 exclusions | 0 references to any of the 6 dropped modules in the shipping diff (precise per-module sweep) |
| AC-3 import & prune closure | `import sglang` + `dsa_backend` + `double_sparsity` pkg clean; no shipped file imports a dropped module |
| AC-4 validator ships, radix gate removed | `validate_double_sparsity` ships + gates; all radix-fixture machinery stripped (0 refs) |
| AC-5 calibration | fresh GLM-5.1-FP8 mask via shipped `calibrate.py`, loader-accepted (ChannelMask [78,64,32] page64 fp8_e4m3); calibrate smoke tests pass |
| AC-6 DS active | long-context decode `meta_info["double_sparsity"]`: selected=2048, total=5608, dense_fallback=0; per-layer bind logs present |
| AC-7 abort path | 3/3 unit test — `_maybe_abort_on_ds_error` drives `set_finish_with_abort`+`update_finish_state` same-step |
| AC-8 perf parity | **DS conc-64: 29.34 p50 decode TPS / 23.29s P99 TTFT** — within the loop-11b band (≥24.2 / ≤30.1) and ahead of native DSA on the same base (26.06 / 46.50s) |
| AC-9 dependency closure | no NEW build dep from DS (triton + flash_mla_sparse_fwd resolve; Codex-confirmed) |
| AC-10 no dead code | `metrics.record_selection` + `ds_recall_oracle_enabled` + selection-capture mirror path removed |

114 unit tests pass (slim runtime + calibrate 91, lifted-budget 20, abort 3).

## What was implemented (v2 clone, 6 commits)
- **Port mechanism**: additive minimal closure. Copied the 15 keep-list `double_sparsity/` modules;
  re-applied DS hunks onto current main by triage (11 files applied clean incl. the 895-line
  `deepseek_v2.py`; 10 hand-reconciled for drift).
- **5 modified-upstream files the plan missed, now shipped**: `output_streamer` / `detokenizer_manager`
  / `multi_tokenizer_mixin` (the `per_request_summary` transport that carries `meta_info`),
  `custom_all_reduce_v2` (`override_algo` deterministic AR), `dsa/dequant_k_cache` (lifted-budget).
- **Drift landmines fixed**: gated v2's new `DSATokenToKVPool.move_kv_cache` (would crash under
  DS+radix on the None index-k buffer); DS bind reduced to the module iteration (the pool refs were
  vestigial; `self.token_to_kv_pool` doesn't exist in v2); `forward_mla` `_select_topk_indices`
  dispatch onto the `is_nextn` gate; abort finisher renamed (`_handle_finish_state_updated_req`);
  radix-fixture CLI arg + `apply_radix_fixture_artifact` call removed from `server_args`.
- **Diagnostic prune (entangled)**: removed `_maybe_record_recall_oracle` + score/selection/latent
  capture guards + the 4 diagnostic config fields + the dead selection-capture mirror plumbing
  (dsa_backend flags + cuda_graph buffers) + the radix-fixture recorders from `validator.py`.
- **CUDA-graph selector-width ladder (the perf-critical fix)**: retargeted onto the refactored
  `DecodeCudaGraphRunner` (the old `cuda_graph_runner.py` was split into `runner/`). Captures one
  decode graph per selector width, stamping `_ds_graph_variant_key=(bs,width)` around capture/replay;
  width-encoded `variant_label` keys each graph (no `ShapeKey` change). Without it the captured graph
  scored the full `req_to_token` width (~202k) every step → DS decode 18.8 TPS; with it → 29.3 TPS.
- **One perf eval**: `benchmarks/bench_double_sparsity.py` (thin wrapper over stock `bench_serving`,
  derives p50 decode TPS) + `benchmarks/DOUBLE_SPARSITY.md` provenance doc.
- **Tests**: slim runtime+calibrate extracted from the 70-class dev harness; lifted-budget kept;
  new `test_ds_abort_path.py`.

## Files changed (shipping branch, 42 in diff vs `<BASE>`)
- New: `python/sglang/srt/layers/attention/double_sparsity/` (15 modules); `benchmarks/bench_double_sparsity.py`; `benchmarks/DOUBLE_SPARSITY.md`; `test/registered/unit/managers/test_ds_abort_path.py`; extracted `test_double_sparsity_unit.py` + kept `test_lifted_budget_decode.py`.
- Modified-upstream (DS hunks): `dsa_backend.py`, `dsa/dequant_k_cache.py`, `server_args.py`, `deepseek_v2.py`, `forward_mla.py`, `forward_mha.py`, `model_runner.py`, `model_runner_kv_cache_mixin.py`, `pool_configurator.py`, `runner/decode_cuda_graph_runner.py`, `memory_pool.py`, `memory_pool_host.py`, `logits_processor.py`, `custom_all_reduce_v2.py`, `managers/{schedule_batch,scheduler,io_struct,tokenizer_manager,detokenizer_manager,multi_tokenizer_mixin}.py`, `managers/scheduler_components/{batch_result_processor,output_streamer}.py`.

## Validation
- Cheap gates (CPU): `import sglang`/dsa_backend/double_sparsity OK; AC-2 module sweeps 0; AC-4 radix machinery 0; 114 unit tests pass.
- GPU (8×H200, GLM-5.1-FP8 TP8): mask calibrated + loader-accepted; DS-active boot (radix ON, graphs ON, no fixture/override, no expandable_segments); abort same-step; conc-64 perf 29.34 TPS / 23.29s (evidence in `development/loop12/perf_evidence/`), native-DSA baseline 26.06 / 46.50s (`development/loop12/dsa_evidence/`).
- Branch pushed: `Jiminator/sglang` `double-sparsity-v2` (6 commits).

## Key decisions / deviations (see Goal Tracker Plan Evolution Log)
- **DEC-8 / AC-8 base drift**: branching off LATEST main raised real floors — `sgl-kernel` 0.4.4
  (its flash-attn `only_qv` path; env had 0.4.3 → upgraded via prebuilt wheel) and a triton-3.6.0
  MoE-config fallback that slows BOTH DS and native DSA (TTFT ~46s base-bound). A native-DSA baseline
  on the same base isolated decode (a port gap → the ladder) from TTFT (base drift). The user chose
  to reframe AC-8 as DS-vs-DSA on the same base; once the ladder was restored, DS meets the *original*
  loop-11b band outright AND beats DSA, so the reframe is satisfied a fortiori.
- task5 was initially deferred as a "no-op"; the perf gate proved it required — corrected.

## Remaining items
None blocking. The branch is pushed; a PR can be opened at
`https://github.com/Jiminator/sglang/pull/new/double-sparsity-v2`. Optional follow-ups (not loop-12
scope): bundle triton-3.6.0-tuned MoE configs to recover base TTFT; a CPU unit test for the ladder
width selection.

## BitLesson Delta
Action: add
Lesson ID(s): BL-20260619-ds-selector-width-ladder, BL-20260619-latest-main-base-drift
Notes: The selector-width graph ladder is decode-critical for DS (full-width selection collapses
decode throughput; the runner must key decode graphs by (bs, width)). Branching off latest main drags
newer kernel/triton floors that confound an absolute perf-parity reference — always capture a
same-base native baseline before calling a perf delta a port regression.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
(base commit unavailable, showing recent branch commits)
be71d4fc3 [loop11b] R0: task10 production UX pass (Cat-A/B docs + runbook; no ABI)
c6e3e943e [loop11b] R0: task10 DONE (UX pass); checkpoint — awaiting DSA matched re-run
d672d962f [loop11b] R0: matched-op-point verdict (task7/8/9 done) — DS meets SLO to conc32
425cdbcef [loop11b] R0: close-out — regenerate results.md (M-A+M-B+M-C complete)
65997cb4c [loop11b] R0: close-out evidence preflight — residual probe evidence + ignore .pt dumps
5df030348 [loop11b] R0: task11 close-out complete — all 11 tasks done; queue finalized
9af9d7835 [loop11b] R1: bench_serving emits prefix-reuse + DS no-op evidence (AC-5/AC-9)
8cde27faa [loop11b] R1: clean M-B re-run orchestrator (both op-points, tax probe, evidence)
73338e539 [loop11b] R1: fix mb_v2 tax_probe local-var bug; task10 serve-script de-plan
4ceba0ead [loop11b] R1: queue checkpoint — bench evidence + cleanup done, mb_v2 clean re-run running
86ddf6faf [loop11b] R1: fix stale a4be98c4 capacity claim (Codex gap 5) — note ld32 504640 reconfirm
1a29be00d [loop11b] R1: fix DS error-abort crash — check_finished was renamed upstream (#25725)
99ac584ac [loop11b] R1: document DS crash finding + selector reuse-edge; mb_v2 emits selector-error count
8fbe848ed [loop11b] R1: M-B verdict re-established clean — comparators ACCEPT both op-points
811c40420 [loop11b] R1: AC-5 no-op proof (dense_fallback=0 + structural sparsity) + GLM meta_info gap doc
9d2c4253d [loop11b] R1: headline M-B verdict + AC-4 dedicated per-step tax (both PASS)
f1b90c797 [loop11b] R1: AC-8 close-out — results.md + queue.md regenerated to the R1 publishable state
44310f230 [loop11b] R1: complete evidence package — DSA server_info + crash-probe txt + crash-log hashes
c16c0d202 [loop11b] R2: wire GLM/dsa-backend DS per-request summary (AC-5) — host-side, graph-robust
b5c4d72be [loop11b] R2: verdict re-established + AC-5 PASS + raw evidence committed (lossless)
8062039d8 [loop11b] R2: AC-8 ledgers regenerated to final state + push status; de-AC the new backend comment
df18a93d0 [loop11b] R3: fix total_tokens metric semantics (AC-5) — explicit field, not rate-inverse
96202e4c4 [loop11b] R3: corrected verdict evidence (results_r3) + supersede results_r2
2ce2adf4e [loop11b] R3: ledgers to one current state (AC-8) — results_r3, mask=regenerated, close-out ACTIVE-until-push
e0935e5a9 [loop11b] R3: AC-8 close-out COMPLETE — pushed to owner fork Jiminator/sglang
da12616a5 [loop11b] R3 review fix [P3]: build_corpus.py creates the output dir before writing
101926d76 [loop11b] R3 review fixes [P2 x2]: report verdict vs exit consistency + fail-closed partial DS evidence
9ab62e6ad [loop11b] R3 review fixes: DS abort test rename + comparator report verdict/labels + green test suite
bec9348c7 [loop12] add gen-plan output + draft; record pensieve doctor state
879532819 [loop12] R0 evidence: calibrate/boot/perf scripts + verdicts + BASE
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
(first round, no prior history)

Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Implementation Review

- Your task is to conduct a deep critical review, focusing on finding implementation issues and identifying gaps between "plan-design" and actual implementation.
- Relevant top-level guidance documents, phased implementation plans, and other important documentation and implementation references are located under @docs.
- If Claude planned to defer any tasks to future phases in its summary, DO NOT follow its lead. Instead, you should force Claude to complete ALL tasks as planned.
  - Such deferred tasks are considered incomplete work and should be flagged in your review comments, requiring Claude to address them.
  - If Claude planned to defer any tasks, please explore the codebase in-depth and draft a detailed implementation plan. This plan should be included in your review comments for Claude to follow.
  - Your review should be meticulous and skeptical. Look for any discrepancies, missing features, incomplete implementations.
- If Claude does not plan to defer any tasks, but honestly admits that some tasks are still pending (not yet completed), you should also include those pending tasks in your review.
  - Your review should elaborate on those unfinished tasks, explore the codebase, and draft an implementation plan.
  - A good engineering implementation plan should be **singular, directive, and definitive**, rather than discussing multiple possible implementation options.
  - The implementation plan should be **unambiguous**, internally consistent, and coherent from beginning to end, so that **Claude can execute the work accurately and without error**.

## Part 2: Goal Alignment Check (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/goal-tracker.md and verify:

1. **Acceptance Criteria Progress**: For each AC, is progress being made? Are any ACs being ignored?
2. **Forgotten Items**: Are there tasks from the original plan that are not tracked in Active/Completed/Deferred?
3. **Deferred Items**: Are deferrals justified? Do they block any ACs?
4. **Plan Evolution**: If Claude modified the plan, is the justification valid?

Include a brief Goal Alignment Summary in your review:
```
ACs: X/Y addressed | Forgotten items: N | Unjustified deferrals: N
```

## Part 3: Required Finding Classification

You MUST classify your findings into these lanes:
- **Mainline Gaps**: plan-derived work or AC progress that is missing, incomplete, or regressing
- **Blocking Side Issues**: bugs or implementation issues that block the current mainline objective from succeeding safely
- **Queued Side Issues**: valid non-blocking follow-up issues that should be documented but must NOT take over the next round

Also include a one-line verdict:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
```

This verdict line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

If Claude mostly worked on queued side issues and failed to advance the mainline, say so explicitly.

## Part 4: ## Goal Tracker Update Requests (YOUR RESPONSIBILITY)

Claude should normally keep the **mutable section** of `goal-tracker.md` up to date directly. If Claude's summary contains a "Goal Tracker Update Request" section, or if you detect tracker drift during review, YOU must:

1. **Evaluate the tracker state**: Is the mutable section still aligned with the Ultimate Goal and current AC progress?
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/goal-tracker.md yourself with the requested changes:
   - Move tasks between Active/Completed/Deferred sections as appropriate
   - Add entries to "Plan Evolution Log" with round number and justification
   - Add new issues to "Blocking Side Issues" or "Queued Side Issues" as appropriate
   - **NEVER modify the IMMUTABLE SECTION** (Ultimate Goal and Acceptance Criteria)
3. **If you reject a requested tracker change**: Include in your review why it was rejected

Common update requests you should handle:
- Task completion: Move from "Active Tasks" to "Completed and Verified"
- New blocking issues: Add to "Blocking Side Issues"
- New queued issues: Add to "Queued Side Issues"
- Plan changes: Add to "Plan Evolution Log" with your assessment
- Deferrals: Only allow with strong justification; add to "Explicitly Deferred"

## Part 5: Output Requirements

- In short, your review comments can include: problems/findings/blockers; claims that don't match reality; implementation plans for deferred work (to be implemented now); implementation plans for unfinished work; goal alignment issues.
- Your output should be structured so Claude can tell which items are mainline gaps, blocking side issues, and queued side issues.
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-18_13-17-09/round-0-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
