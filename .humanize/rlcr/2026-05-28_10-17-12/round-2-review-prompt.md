# Code Review - Round 2

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-2-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 2 Summary — First DS boot on V3.2 FP8 (M2 Phase A)

## Round objective (round-2-contract.md)
First DS boot smoke on DeepSeek-V3.2 FP8 with the calibrated mask, and capture the
evidence the boot yields (AC-1 boot, AC-0 hardware capture probe, AC-1.1 sparsity,
AC-6 CUDA-graph status).

## Outcome
The DS server now BOOTS and SERVES on V3.2 FP8 (TP=8) — the DS serving path had
never run on hardware, and this round fixed the chain of latent blockers that the
first boot surfaced. AC-1 boot + AC-6 are met; AC-0-hardware, AC-1.1, and AC-Q are
blocked by two newly-discovered DS-runtime issues (decode degeneration + DS
meta_info not surfacing).

## Code fixes (commit `34b243b07`; regressions in `44a12d5d1` evidence)
Four latent blockers, each found by a successive boot attempt and fixed:
1. **validator.py** — `is_deepseek_nsa` → `is_deepseek_dsa` (model_config renamed it).
   Stale name → ImportError at server startup. +regression (capability symbol guard).
2. **deepseek_v2.py:1516** — DS-enablement branch `self.use_nsa` → `self.use_dsa`
   (set from `is_deepseek_dsa(config)`). Stale attr → AttributeError at model
   construction. +regression (source guard).
3. **deepseek_v2.py:1979** — move bound `_ds_channel_selection` to the label-table
   device. It loaded on CPU; the KV-write hook gathers GPU-resident K_nope with it
   as the index → `RuntimeError: index on cpu, tensors on cuda` in the MHA_ONE_SHOT
   write path during warmup.
4. **serve_double_sparsity.sh** — add `--mem-fraction-static` (default 0.6). Stock
   0.897 OOMs at boot (no room for the ~31 GiB DS TokenLabelTable); 0.7 boots but
   OOMs during generation; 0.6 boots + serves stably (~38 GB runtime headroom).

## Evidence (`runs/20260528_dsv32_mvp/`)
- `ds_boot_*.log` — successive boots; the final (mem_fraction 0.6) reaches
  "The server is fired up and ready to roll!".
- `ds_boot_knobs_AC1.json` — AC-1 knobs from server_args: model_path = cluster path
  (not HF-id), tp_size=8, kv_cache_dtype=fp8_e4m3, page_size=64,
  enable_double_sparsity=True, disable_radix_cache=True, disable_cuda_graph=False,
  disable_piecewise_cuda_graph=True, disable_overlap_schedule=True,
  attention_backend=dsa. `/get_model_info` independently confirmed the cluster path.
- `ds_generate_probe.json` — `/generate` "The capital of France is" → `" Paris."`
  (correct prefill) then a repeated token (decode degeneration).

## AC status this round
- **AC-1 (boot smoke): MET (core).** Boots single-node TP=8 with the cluster
  MODEL_PATH (DEC-6) + the mask; validator accepts (radix-off); all 8 GPUs; KV cache
  + DS TokenLabelTable allocated; DS `bind_runtime_data completed` on all ranks;
  `/generate` returns non-empty text; knobs confirmed. (`/get_server_info` endpoint
  is flaky in this build — empty response — so knobs are from server_args +
  `/get_model_info`.)
- **AC-6 (regular CUDA-graph status): MET.** Regular CUDA-graph capture completed all
  52 batch sizes at boot (capture success), distinct from `--disable-piecewise-cuda-graph`
  (piecewise disabled, logged). Recorded.
- **AC-0 hardware capture probe (task2): BLOCKED** — `meta_info` has no
  `double_sparsity_radix_capture` (see Blocking #B).
- **AC-1.1 (genuine sparsity, task6): BLOCKED** — no `double_sparsity` meta to read,
  and decode output is degenerate (see Blocking #A, #B).

## UPDATE (same round, continued) — #16 decode degeneration FIXED; new #C/#18 found
After the localization below, #16 was fully fixed and validated on hardware, and a
distinct deeper bug (#C/#18) was discovered:
- **#16 FIXED (commits `2af5f4e65` + `8375b76a5`).** Two decode-path bugs: (1)
  `req_to_token` None during decode → wrong selection domain + skipped
  `logical_to_physical`; fixed via ForwardContext resolution. (2) decode tokens never
  label-written — `kv_b_proj` not on `attn_mqa` AND `head_width` derived from
  `attn_mqa.v_head_dim` (512 vs real 128); fixed by attaching `kv_b_proj` to `attn_mqa`
  and deriving `head_width` from the projection output. Validated: short-prompt decode
  coherent ("Paris. The capital of the United States is Washington, D.C. ..."),
  `selected_tokens` grows with seq (was frozen at prompt_len), `dense_fallback=0`.
  253 DS unit tests pass.
- **#C/#18 NEW (OPEN, task #18).** With #16 fixed, DS serves coherently for
  seq<top_k=2048 but ANY request with seq>top_k crashes with `cudaErrorIllegalAddress`
  in `_select_topk_indices` (the genuine-sparse path). Bisected: 1376/1933 tok OK,
  2316/~3500 tok crash. Blocks AC-1.1 (needs seq>top_k) + real-shape benchmarks.
  Next round: compute-sanitizer to localize the OOB kernel, then fix. Evidence:
  `runs/20260528_dsv32_mvp/sparse_path_oob_finding.md`.

## New blocking issues — investigated + localized this round (fix is round 3)
- **#A: DS decode degenerates (DS-specific selection over-count).** Decisively
  localized with three on-hardware experiments:
  1. **DSA baseline control** (`serve_native_nsa.sh`, same model + `dsa` backend +
     fp8 KV + flashmla_kv, NO Double Sparsity) → coherent output (" Paris. 法国的首都
     是巴黎。 The capital of Italy is Rome. ..."). So the V3.2 dsa serving stack is
     correct; the bug is **DS-specific**.
  2. **DS eager** (`--disable-cuda-graph`) degenerates identically → **not
     CUDA-graph-related**; the bug is the core DS selection.
  3. The surfaced `double_sparsity` meta shows `valid_lengths` EXCEEDING `seq_len`
     (negative `sparsity_rate`; `selected_tokens=19` when seq≈12, and `=5` when
     seq≈28) — DS over-/mis-selects. **Single-step decode over a clean prefill is
     correct (" Rome"); multi-step corrupts.** Localized to
     `retrieve_topk_via_labels` logical selection returning more valid tokens than
     the sequence length → wrong/duplicate physical slots → garbage decode
     attention. Round-3 fix plan in `decode_degeneration_diagnosis.md`. Task #16.
- **#B: DS per_request_summary not surfacing in `/generate` meta_info (graph mode).**
  Root cause found: `_publish_ds_request_summary` is intentionally skipped during
  CUDA-graph capture/replay (deepseek_v2.py:2326-2329, host `.item()`/`.tolist()`
  are illegal under capture), and decode runs under CUDA graph. In **eager** mode the
  `double_sparsity` summary DOES surface (used above to diagnose #A). The AC-0
  hardware capture probe (graph mode) needs a capture-safe publish path or an
  eager-mode probe. Task #17.

## Commits (round 2)
- `34b243b07` — the four DS serving-path fixes (validator rename, use_dsa, channel
  device, mem-fraction) + rename regressions.
- `44a12d5d1` — AC-1 boot knobs + DS `/generate` probe evidence.
- `610b65c15` — `EXTRA_SERVER_ARGS` launcher passthrough + decode-degeneration
  control/diagnosis artifacts (DSA baseline + DS eager probes).

## Operational notes
- DS server booted on **port 30010** (port 30000 held by a pre-existing orphaned
  `sglang::router` the agent is not authorized to kill; downstream tasks take
  configurable URLs).
- All servers shut down after probing; GPUs free.

## Tests
- `python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`
  → **253 passed** (was 251; +2 rename guards).

## Remaining / next round
Resolve #A (decode degeneration) and #B (meta surfacing) — without them AC-1.1, the
AC-0 hardware probe, and AC-Q cannot be demonstrated. #A is the priority (it questions
DS output correctness on V3.2). Then proceed to M2 Phase B/C (smoke benchmarks,
comparator, paired quality smoke).

## Goal Tracker Update Request
### Requested Changes
- Mark **AC-1 (boot smoke)** met (core) and **AC-6** met, with the evidence above.
- Add **Blocking Side Issues #A (DS decode degeneration)** and **#B (DS meta_info not
  surfacing)**; mark **AC-0 hardware probe (task2)** and **AC-1.1 (task6)** blocked on them.
- Record the four serving-path fixes + the mem_fraction default (0.6) in the Plan
  Evolution Log (round 2). No immutable-section or AC-definition changes.
### Justification
The boot (M2 Phase A core) is achieved and committed; the two findings are genuine
DS-runtime correctness/transport bugs that block the remaining AC-1 evidence and must
be resolved before benchmarks/quality. Documenting them as blocking issues keeps the
mainline honest and sets the next round's focus.

## FINAL STATE (end of round 2) — DS-on V3.2 works end-to-end (TIER-1 smoke core)
After the model cutover, the round continued to a working DS-on MVP:
- **#16 decode degeneration FIXED** (commits `2af5f4e65`, `8375b76a5`): coherent decode,
  decode tokens labelled, selection tracks the sequence.
- **#18 long-prompt OOB FIXED** (commit `eba4c640e`) via the user's domain insight that
  classic DS is **dense-prefill / sparse-decode**: keep dense MHA prefill for DS, run DS
  selection only at decode. A 2272-token (>top_k) prompt now serves; **AC-1.1 satisfied**
  (sparsity_rate=0.105, selected_tokens=2048, dense_fallback=0; `ac1_1_genuine_sparsity.json`).
- **Radix-capture CUDA-graph crash FIXED** (commit `8e9138af6`): capture+graph boots cleanly.
- **Critical review of loops 1-5** delivered (`REVIEW_loop4_loop5_precutover.md`): all
  defects were loop4-integration / loop5-calibrate "never ran on hardware" bugs; the
  loops 1-3 foundations are sound and now hardware-validated.

TIER-1 smoke status: AC-4 (mask), AC-1 (boot), AC-1.1 (genuine sparsity), AC-6 (CUDA-graph
status), AC-0 (producer code+regression) — DONE. Remaining TIER-1: AC-8/9 smoke benchmarks,
AC-8/9 comparator, AC-Q paired quality smoke. Remaining TIER-2: AC-10 radix flip (+ the
radix-capture /generate surfacing, now a queued TIER-2 item), AC-1b, AC-11, AC-12.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260528-dsv32-ds-serving-boot-chain, BL-20260528-dsv32-ds-decode-degeneration, BL-20260528-dsv32-ds-dense-prefill-sparse-decode, BL-20260528-ds-radix-capture-cuda-graph-safe
- Notes: Added a lesson capturing the V3.2 DS-serving boot-blocker chain (NSA→DSA
  rename stragglers in validator.py + deepseek_v2.py; the channel_selection CPU/CUDA
  device-placement bug in the KV-write hook; and the DS mem_fraction_static headroom
  for the TokenLabelTable + CUDA graphs — 0.897 OOMs at boot, 0.7 OOMs in generation,
  0.6 serves). `bitlesson-selector` for the boot returned
  {BL-20260527-ds-metadata-via-forward-context, BL-20260527-shell-json-into-python-source};
  both were applied (server JSON read via `json.load(sys.stdin)`, not source-spliced;
  the metadata-via-ForwardContext lesson informs the #B meta-surfacing investigation).
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
eb914678e [Sparsity] Loop-5: refined plan v1 + QA ledger
8979848ab [Sparsity] Loop-5: untrack active RLCR plan file
4f4c620df [Sparsity] Thread forward_batch into _write_token_labels (radix capture producer fix)
7cbbce088 [Sparsity] Calibration: native-FP8 sharded load + one-block dry-run mode
c99ed3644 [Sparsity] Calibration: load DeepSeek-V3.2 via deepseek_v3 remap + fail-closed dry-run
610f364c9 [Sparsity] Loop-5: V3.2 channel-mask calibration evidence (AC-4 complete)
df8d7c6c6 [Sparsity] Untrack .humanize/bitlesson.md (loop state, per .gitignore)
34b243b07 [Sparsity] Fix the DS serving path so DeepSeek-V3.2 boots on hardware
44a12d5d1 [Sparsity] Loop-5: round-2 DS boot evidence (AC-1 knobs + /generate probe)
610b65c15 [Sparsity] Loop-5: localize DS decode degeneration (DS-specific, selection over-count)
05a25f197 [Sparsity] Loop-5: refine decode diagnosis (eager scorer masks seq_len; instrument inputs in round 3)
2af5f4e65 [Sparsity] Fix DS decode selecting wrong domain: resolve req_to_token via ForwardContext
d9ad3066f [Sparsity] Loop-5: decode-degeneration is two bugs (req_to_token fixed; decode label-write open)
6429cf539 [Sparsity] Loop-5: complete bug #2 root cause (decode passes pre-projected k_nope, not latent)
8375b76a5 [Sparsity] Fix DS decode degeneration: label decode tokens (attn_mqa kv_b_proj + robust head_width)
b231942fa [Sparsity] Loop-5: DS genuine-sparse path OOB when seq_len>top_k (#18 finding)
da1ff651e [Sparsity] Loop-5: #18 deeper root cause — DS prefill selection bad req_pool_indices (long-prompt OOB)
802b51b84 [Sparsity] Loop-5: confirm #18 mechanism — DS selection uses decode batch shape, breaks on prefill per-token batch
ffe6c2b97 [Sparsity] Loop-5: critical review of loop4 DS scaffolding + pre-cutover loop5 fixes
eba4c640e [Sparsity] DS dense-prefill / sparse-decode: fix long-prompt OOB + unblock AC-1.1
590b0dc05 [Sparsity] Loop-5: extend code review to loops 1-3 foundational DS modules
3f9478128 [Sparsity] Loop-5: mark #18 resolved in review doc (dense-prefill fix)
8e9138af6 [Sparsity] Make radix fixture capture CUDA-graph-safe (no host copies during capture)
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-1-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-1-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-0-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-0-review-result.md


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

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md and verify:

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
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md yourself with the requested changes:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-2-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
