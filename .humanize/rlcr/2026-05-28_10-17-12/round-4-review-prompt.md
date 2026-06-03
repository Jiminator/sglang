# FULL GOAL ALIGNMENT CHECK - Round 4

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop5/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 4 Summary

## Work Completed
Produced the TIER-1 smoke benchmark pair + comparator (AC-8/AC-9) on 8x H200, fixing the
two Codex-flagged launcher blockers plus two more gaps that surfaced under real bench load.

- **#D — launchers default to the cluster weights.** Both `serve_double_sparsity.sh` and
  `serve_native_nsa.sh` now default `MODEL_PATH` to
  `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2` (env override preserved). Verified on
  both live servers.
- **#E — DSA radix-off smoke knob.** Added `DISABLE_RADIX_CACHE=1` to `serve_native_nsa.sh`
  (appends `--disable-radix-cache`); default 0 keeps radix-on for the later AC-11 sweep.
- **Two more gaps found on hardware (committed as smoke-enablement):**
  - The DSA baseline OOMed at the stock `mem_fraction_static=0.897` in
    `flash_mla_with_kvcache` (~84 GB/rank weights leave no kernel-workspace headroom at the
    4096-ISL shape). Added a `MEM_FRACTION_STATIC` knob (default 0.85).
  - The "shortened window" did not shorten the run: bench_serving runs FULL epochs of
    `NUM_PROMPTS` before re-checking the window, and one epoch at conc 16 / ISL 4096 is
    ~900 s. Made `NUM_PROMPTS` env-overridable (default unchanged 320) so a smoke can use a
    small per-epoch count.
- **Ran the pair (DSA refs first, then DS, single node sequential):** DSA (mem 0.85) and DS
  (mem 0.6, `--disable-radix-cache`) booted radix-off from the cluster weights; smoke shape
  `conc 16/32/64`, `TRIALS=1`, `NUM_PROMPTS=64`, `WARMUP_SECONDS=0`, `MEASUREMENT_WINDOW_S=30`,
  GSP ISL≈4096 / OSL 512.
- **Comparator:** `benchmark_compare.py --baseline --ds` exited 0 for all three
  concurrencies (radix parity held — no `disable_radix_cache` mismatch refusal). Assembled
  `mvp_compare.md` with the smoke context, the three tables, the radix-parity proof, and the
  directional reading + KV caveat.

## Files Changed
- `development/serve_double_sparsity.sh` — `MODEL_PATH` default → cluster weights.
- `development/serve_native_nsa.sh` — `MODEL_PATH` default → cluster weights;
  `DISABLE_RADIX_CACHE` knob; `MEM_FRACTION_STATIC` knob (default 0.85).
- `development/benchmark.sh`, `development/benchmark_baseline.sh` — `NUM_PROMPTS`
  env-overridable (default unchanged).
- `runs/20260528_dsv32_mvp/` — `mvp_compare.md`, `mvp_compare_c{16,32,64}.{md,json}`,
  `dsa_smoke_server_info.json`, `ds_smoke_server_info.json`,
  `smoke_results/*.meta.json` (6 sidecars). Raw `*.jsonl` are gitignored (repo policy
  `.gitignore:179`); metrics are preserved in the sidecars + comparator JSON.
- Commits: `6acdfb94f` (#D/#E launcher parity), `f2bc1eb6a` (mem-fraction + NUM_PROMPTS
  smoke enablement), `2220a793f` (smoke artifacts + comparator). Pushed to remote.

## Validation
- All 6 bench JSONLs have observed `duration` 168–533 s ≥ the 30 s window → the hard
  duration guard passed; smoke explicitly labeled non-AC-11.
- 6 `.meta.json` sidecars confirm `disable_radix_cache=true` on BOTH sides, `tp_size=8`,
  `mem_fraction_static` 0.85 (DSA) / 0.6 (DS), DS on/off correctly, cluster `model_path`.
- `benchmark_compare.py` exit 0 × 3 concurrencies.
- Directional read (smoke only): DS per-token decode competitive-to-better (TPOT P50
  25.7–27.1 ms flat vs DSA up to 193 ms P99 at conc 32/64; DS per-request tok/s on par or
  higher at conc 32/64). DS TTFT worse (P99 120/244/502 s vs DSA 34/70/155 s) — a
  single-node KV-pool/effective-concurrency artifact at mem 0.6 (see #F), not an
  algorithmic regression.

## Remaining Items
- **#F (queued, blocks honest AC-11 TTFT, not the smoke):** DS at mem 0.6 admits only ~2
  concurrent 4096-tok requests (TokenLabelTable + ~84 GB/rank weights → small KV pool), so
  TTFT is queuing-dominated. Resolve before the radix-on AC-11 sweep (scale the workload,
  raise the DS KV budget post-radix-flip if headroom allows — capped, or report effective
  vs nominal concurrency).
- **task9 / AC-Q** paired quality smoke — next (M2 Phase C quality half); needs both
  servers sequentially.
- **TIER-2:** task11 AC-10 radix flip, task12 AC-1b, task13 AC-11, task14 AC-12, task15
  bundle.
- Queued cleanup: stale `calibrate.py` operator recipe docstring.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-dsv32-bench-smoke-sizing
- Notes: Added `BL-20260529-dsv32-bench-smoke-sizing` — (1) the DSA baseline launcher needs
  a mem-fraction knob (stock 0.897 OOMs `flash_mla_with_kvcache` on V3.2 FP8 TP=8 under
  bench load; use 0.85); (2) bench_serving's time-window runs FULL epochs of `NUM_PROMPTS`
  before re-checking elapsed time, so a small window does not shorten the run — make
  `NUM_PROMPTS` small (≥ max concurrency) and `WARMUP_SECONDS=0` for a quick smoke while the
  duration guard still passes. Keep large NUM_PROMPTS + 120s/600s floors for AC-11. The DS
  KV-pool/concurrency constraint is tracked as project state #F in the goal tracker rather
  than a BitLesson (it is config/deployment-specific, not a reusable code pattern).
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
6f95a9711 [Sparsity] AC-0: radix-capture publish resolves req_to_token via backend/ForwardContext; dtype-safe SHA
bc534da7c [Sparsity] Fix /get_server_info crash (DS stashes tensors on server_args) + AC-0/AC-1 evidence
76eef9c80 [Sparsity] AC-1 negative test: invalid channel-mask path -> fail-closed validator rejection
6acdfb94f [Sparsity] Launcher parity: default MODEL_PATH to cluster weights; add DSA radix-off smoke knob
f2bc1eb6a [Sparsity] Make the TIER-1 smoke benchmark actually runnable on V3.2 FP8
2220a793f [Sparsity] TIER-1 smoke benchmark pair + comparator (AC-8/AC-9), radix-off both sides
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-28_10-17-12/round-3-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-3-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-2-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-2-review-result.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-1-summary.md
- @.humanize/rlcr/2026-05-28_10-17-12/round-1-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Goal Tracker Audit (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/goal-tracker.md and verify:

### 1.1 Acceptance Criteria Status
For EACH Acceptance Criterion in the IMMUTABLE SECTION:
| AC | Status | Evidence (if MET) | Blocker (if NOT MET) | Justification (if DEFERRED) |
|----|--------|-------------------|---------------------|----------------------------|
| AC-1 | MET / PARTIAL / NOT MET / DEFERRED | ... | ... | ... |
| ... | ... | ... | ... | ... |

### 1.2 Forgotten Items Detection
Compare the original plan (@development/loop5/refined_plan_v1.md) with the current goal-tracker:
- Are there tasks that are neither in "Active", "Completed", nor "Deferred"?
- Are there tasks marked "complete" in summaries but not verified?
- List any forgotten items found.

### 1.3 Deferred Items Audit
For each item in "Explicitly Deferred":
- Is the deferral justification still valid?
- Should it be un-deferred based on current progress?
- Does it contradict the Ultimate Goal?

### 1.4 Goal Completion Summary
```
Acceptance Criteria: X/Y met (Z deferred)
Active Tasks: N remaining
Estimated remaining rounds: ?
Critical blockers: [list if any]
```

## Part 2: Mainline Drift Audit (MANDATORY)

Determine whether the recent rounds are still serving the original plan:
- Is the current round's mainline objective clear and singular?
- Has Claude been advancing mainline ACs, or mostly clearing side issues?
- Which findings are true **blocking side issues** versus merely **queued side issues**?

Include a short drift summary:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
Blocking Side Issues: N
Queued Side Issues: N
```

The `Mainline Progress Verdict` line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

## Part 3: Implementation Review

- Conduct a deep critical review of the implementation
- Verify Claude's claims match reality
- Identify any gaps, bugs, or incomplete work
- Reference @docs for design documents

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

## Part 5: Progress Stagnation Check (MANDATORY for Full Alignment Rounds)

To implement the original plan at @development/loop5/refined_plan_v1.md, we have completed **5 iterations** (Round 0 to Round 4).

The project's `.humanize/rlcr/2026-05-28_10-17-12/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-05-28_10-17-12/round-3-review-result.md` (previous round)
- `@.humanize/rlcr/2026-05-28_10-17-12/round-2-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-05-28_10-17-12/round-3-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-4-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-28_10-17-12/round-4-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
