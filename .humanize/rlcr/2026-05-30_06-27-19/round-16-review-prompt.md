# Code Review - Round 16

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop6/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-16-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 16 Summary — AC-8 64K servability PASS at the lifted DS operating point

## Mainline objective (round contract)
Codex R15-review Required-Plan **step 1**: complete AC-8 — at the lifted DS int8 / `mem_fraction_static=0.7`
/ radix-on operating point, demonstrate that a ~70K-token `/generate` is now **ADMITTED (HTTP 200)** rather
than the Loop-5 mem-0.6 `HTTP 400 "Input length (69970) exceeds the maximum allowed (53050)"`, recording the
served `max_total_num_tokens`, real `prompt_tokens`, and no OOM/instability — or a characterized ceiling.
AC-5 strict-SLO remediation (Required-Plan step 2) and gated AC-10 were explicitly out of scope this round.

## What landed (hardware round; commit below)
**AC-8 PASS (servability).** Single-node TP=8, node-0 localhost:
1. **Lifted operating point booted + proven** (identical to AC-4/AC-5/AC-7): DS int8 @ mem 0.7, radix-on via
   the config-bound fixture `ds_radix_fixture_state_int8.json` (sha `f3b67943`, both M3-B fixtures PASSED),
   `disable_radix_cache=False`, int8 `token_label_table` **6.48 GB/rank on all 8 ranks**
   (`dtype=torch.int8 scales=float16`), `max_total_num_tokens=396096`, `context_len=163840`,
   `chunked_prefill_size=8192`. Proven from `get_server_info_{before,after}.json` + boot log.
2. **Named AC-8 deliverable** `development/loop6/probe_64k.json` (the plan's one scaffolding exemption):
   deterministic varied prose (seed 20260531) + a one-line question, raw `/generate`, `max_new_tokens=16`,
   `temperature=0`, `text_sha256=652e4f51…`. Local tokenizer estimate **70759 tokens** == the server-reported
   `prompt_tokens=70759` (exact provenance). `70759 > 53056` (Loop-5 pool), `>= 69970` (Loop-5 64K reference),
   `< 396096` (lifted pool) — so it exercises the same admission length-check that 400'd at mem-0.6.
3. **Probe result** (`ac8_probe_response.json`): **HTTP 200**, served `max_total_num_tokens=396096`, generated
   16 tokens (`finish_reason=length`), latency 11.95 s, **server alive before AND after**. Server log:
   chunk-prefill 8×8192+5248 = 70759 (matches `prompt_tokens`), `#queue-req:0` (admitted immediately),
   token usage 0.02→0.18 of the pool, **0 OOM / CUDA-error lines** in the whole boot+serve log.

## Result
AC-8 PASS — the Loop-5 64K **HTTP-400 admission ceiling is removed** at the lifted operating point: a
~70K-token `/generate` now serves cleanly (HTTP 200, no OOM, server stable) with `max_total_num_tokens=396096`.
No characterized ceiling is needed (the prompt fits with large margin — 70759 of a 396096 pool, 18% token
usage). This is a **servability/admission** result; 64K **recall** accuracy is bounded by the kernel-locked
`top_k=2048` and remains a Tier-2/AC-10 concern, unchanged here. The raw-`/generate` output is a degenerate
continuation (no chat template) — irrelevant to admission, noted explicitly.

## Files Changed
- `development/loop6/probe_64k.json` — named ~70K-token AC-8 probe payload (NEW; the one allowed exemption).
- `runs/20260530_dsv32_loop6/ac8_servability/` (NEW): `ac8_64k_servability.md` (report + Loop-5 contrast),
  `ac8_probe.py` (reproducible driver — reads payload, asserts sha, captures before/after server-info, sends
  raw `/generate` catching rejections as recordable results), `ac8_probe_response.json`,
  `get_server_info_{before,after}.json`, `server_log_excerpt.txt` (chunked-prefill window + 0-OOM scan).
- `.humanize/bitlesson.md` — updated `BL-20260528-dsv32-hf-calibration-load` with the tokenizer-only corollary.
- goal-tracker (R16 Plan Evolution row; task9/AC-8 → done-pending-verification), round-16 contract/summary
  (gitignored loop state).

## Validation
- Boot config + `get_server_info`: `mem_fraction_static=0.7`, `signature_dtype=int8`, `disable_radix_cache=False`,
  `enable_double_sparsity=True`, `max_total_num_tokens=396096`, int8 table 6.48 GB/rank ×8.
- Probe HTTP 200; `prompt_tokens=70759` == local tokenizer estimate; served pool 396096; 16 tokens generated;
  server `/get_server_info` 200 before and after; 0 OOM lines (`grep -icE "out of memory|OOM|CUDA error|..."` = 0).
- Loop-5 contrast embedded: mem-0.6 pool 53056 → HTTP 400 for 69970 tokens vs mem-0.7 pool 396096 → HTTP 200
  for 70759 tokens (lifted-mem retry, not a silent re-record — AC-8 negative test satisfied).
- GPUs freed at round end (all 8 at 0 MiB, no live `launch_server`). `git diff --check` run before commit.

## Remaining Items
- **Open mainline blocker:** AC-5 DS strict client SLO (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc) —
  Codex Required-Plan **step 2**, the next mainline after AC-8 is review-clean: the smallest scheduling/decode/
  operating-point change to restore both `<22 s` and `≥30 TPS/req`, then a full re-run with exact arrays + a
  fail-closed verifier.
- **Gated AC-10** (Tier-2 adjustable-`top_k` kernel) — only after AC-3..AC-9 are all verified.
- **Cross-node wrapper smoke** — future-gated (this round was single-node localhost; no cross-node artifact).
- **DSA-default conc-64 TPS ~29.4** — queued pre-existing DSA limit (R12 user decision).
- No FlashMLA decode-assert changes; DS-fair AC-12 gate unchanged.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260528-dsv32-hf-calibration-load
Notes: Added a tokenizer-only corollary (clause e + Source Rounds += loop6 R16): to size a long-context probe to
an exact token count offline, `AutoTokenizer.from_pretrained` fails the SAME way as AutoConfig on the unregistered
`deepseek_v32` model type, so load `PreTrainedTokenizerFast(tokenizer_file=.../tokenizer.json)` directly (no config),
and cross-check the local count against the live server's authoritative `meta_info.prompt_tokens` (R16: 70759 == 70759).
Applied existing lessons without new failure modes: BL-20260528-dsv32-ds-serving-boot-chain + BL-20260529-ds-radix-flip-config-bound-artifact
(int8/mem-0.7/radix-on boot), BL-20260529-ds-longcontext-needle-recall-vs-topk (framed AC-8 as servability, not recall),
BL-20260529-gate-record-artifact-before-raise (probe driver catches HTTP rejections as recordable results),
BL-20260530-remote-server-launch (background boot + `ps | grep "[s]glang.launch_server"` + `pkill || true`; foreground
`sleep` blocked), BL-20260530-durable-tracked-acceptance-evidence (tracked `.json`/`.txt`/`.md`, exact sha provenance).
No new standalone lesson — clean execution of a well-understood hardware probe.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
1aa24cfc1 [Sparsity] Loop-6: refined plan v1 + QA ledger + DEC-5 roadmap deferral
88c6498e5 [Sparsity] Loop-6 R0: strategic recall-R&D gate + footprint feasibility budget
84d3410b9 [Sparsity] Loop-6 R1: int8-symmetric compact TokenLabelTable (flag-gated, fp16 default, CUDA-graph-safe)
e85cd2564 [Sparsity] Loop-6 R2: scale-aware proof/sanity consumers + AC-3.1/AC-6 evidence
5d8e47fb3 [Sparsity] Loop-6 R3: serve_double_sparsity.sh exposes SIGNATURE_DTYPE (compact-table selection)
8a05b1688 [Sparsity] Loop-6 R3: real-mask NIAH non-regression PASS (int8 DS vs fp16 Loop-5 baseline, TP=8)
75e68053f [Sparsity] Loop-6 R4: AC-4 mem-fraction sweep PASS (int8 lifts no-OOM ceiling 0.6->0.7, TP=8)
91e9c20a3 [Sparsity] Loop-6 R5: AC-4 evidence addendum (full HBM budget + durable no-OOM proof)
8883848e9 [Sparsity] Loop-6 R6: AC-5 client-SLO directional result (int8 @ 0.7 radix-on, TP=8) + attribution
51dd009b8 [Sparsity] Loop-6 R7: durable AC-5 evidence + corrected per-conc attribution
bd09d1ca7 [Sparsity] Loop-6 R8: exact-recomputable AC-5 evidence + reconciled attribution
57f86b66f [Sparsity] Loop-6 R9: exact ITL source + fail-closed AC-5 verifier
d6e884aa9 [Sparsity] Loop-6 R10: AC-9 within_budget from real usage.prompt_tokens
daad92923 [Sparsity] Loop-6 R10: AC-9 within-budget gate re-run on hardware (real tokens)
2fd2c6937 [Sparsity] Loop-6 R10: AC-6 opt-in / DSA-default product proof on hardware
0e1ce974d [Sparsity] Loop-6 R11: AC-6 redo — proper-methodology DSA SLO + radix-on toggle
d0cc9fdc9 [Sparsity] Loop-6 R12: benchmark scripts pass --host to bench_serving
f9bc51b13 [Sparsity] Loop-6 R12: recomputable DSA SLO evidence + honest AC-6 verdict
5e6d3afb5 [Sparsity] Loop-6 R13: AC-7 — 3-trial DS+DSA re-sweep at the lifted point (characterized)
147b6d05f [Sparsity] Loop-6 R14: AC-7 exact-recomputable metrics + fail-closed verifier
99e51ad00 [Sparsity] Loop-6 R14: AC-7 profiling discharged at AC-7 methodology
40ccc4b63 [Sparsity] Loop-6 R15: AC-7 evidence review-clean (verifier precision + provenance + reconciliation)
9915630ca [Sparsity] Loop-6 R16: AC-8 64K servability PASS at lifted DS int8/mem-0.7
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-30_06-27-19/round-15-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-15-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-14-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-14-review-result.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-13-summary.md
- @.humanize/rlcr/2026-05-30_06-27-19/round-13-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-16-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
