Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop6/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop6/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-12-contract.md

Your round contract must contain:
- Exactly one **mainline objective**
- The 1-2 target ACs for this round
- Which issues are truly **blocking** that mainline objective
- Which issues are **queued** and explicitly out of scope
- Concrete success criteria for this round

Do not start implementation until the round contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances this round's objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` work is the round's primary success condition
- `[blocking]` work is allowed only when it truly blocks the mainline objective
- `[queued]` work must be documented but must NOT replace the round objective
- If a new bug does not block the current objective, tag it `[queued]` and keep moving on mainline work

Before executing each task in this round:
1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 11 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary: ACs: 7/10 addressed, 5/10 met | Forgotten items: 0 | Unjustified deferrals: 1

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, `round-11-prompt.md`, `round-11-contract.md`, `round-11-summary.md`, `goal-tracker.md`, Round 8-10 summaries/reviews, commit `0e1ce974d`, the AC-6 product-proof artifacts, and the benchmark scripts.

## Implementation Review

Round 11 did fix one real AC-6 blocker: the DS opt-in side is now proven at the locked radix-on point. `get_server_info_keys.json` records DS with `enable_double_sparsity=true`, `signature_dtype=int8`, the int8 radix fixture, and `disable_radix_cache=false`, while DSA-default is `enable_double_sparsity=false`, `double_sparsity_config=null`, and also radix-on. `ds_table_boot_excerpt.txt` shows the int8 `token_label_table` on TP0-TP7 plus the fixture PASSED line.

AC-6 is still not complete. The DSA-default SLO proof was weakened after the run did not satisfy the written contract, and the artifact itself admits the default misses the `>=30 TPS/req` half of the SLO at conc 64.

## Mainline Gaps

1. **AC-6 is falsely marked met even though DSA-default still fails the required TPS threshold.**

   The Round 11 contract defines AC-6 success as DSA-default meeting `P99 TTFT < 22 s AND >= 30 TPS at every conc`. The new AC-6 report and both supporting SLO artifacts show conc-64 TPS below that threshold: Loop-5 baseline `29.5` and fresh R11 `29.4` (`runs/20260530_dsv32_loop6/ac6_optin_dsa_default_product.md:46-61`, `ac6_product_proof/dsa_default_matches_loop5_baseline.txt:20-28`, `ac6_product_proof/dsa_default_slo_np64.txt:5-11`). Calling this "pre-existing" may explain attribution, but it does not satisfy the positive test that DSA stays the production default **that meets the SLO**.

   Required fix: keep task7/AC-6 partial until DSA-default has a tracked artifact with both TTFT and TPS passing at conc 16/32/64, or explicitly record AC-6 as failed and request a user-approved plan/SLO revision. Do not claim AC-6 met with a known conc-64 TPS miss.

2. **The required `NUM_PROMPTS=320` DSA-default proof was replaced by a `num_prompts=64` substitute.**

   The Round 11 success criteria required a tracked DSA-default proper-methodology `120/600/320` artifact. The committed `dsa_default_slo.txt` is explicitly labeled "COLD-RAMP, NOT steady state", contains only conc 16/32, and records conc-32 P99 TTFT `34.18 s` (`ac6_product_proof/dsa_default_slo.txt:1-8`). The report then substitutes `num_prompts=64` evidence and says the 320-prompt run is "not the SLO" (`ac6_optin_dsa_default_product.md:63-80`). That is an unapproved plan/contract change, not completion of AC-6.

   Required fix: either run and publish the contract artifact exactly enough to judge it, or keep AC-6 failed. If the team now believes `NUM_PROMPTS=320` is the wrong DSA-default methodology, the plan must be revised explicitly before using `num_prompts=64` as the acceptance method.

3. **The fresh DSA SLO artifact is summary-only, not recomputable evidence.**

   `dsa_default_slo_np64.txt` contains a hand-written table only; there are no committed JSONLs, sidecars, checksums, request arrays, or verifier under `ac6_product_proof/` (`rg --files runs/20260530_dsv32_loop6/ac6_product_proof`). This is weaker than the evidence standard already enforced for AC-5 and weaker than the Round 11 contract's "JSONL-derived summaries sufficient to recompute completed/errors/P99 TTFT/per-req TPS".

   Required fix: commit a redacted/exact metric source or a small verifier bundle for the DSA-default run: per-request TTFTs, TPOT/TPS source values, errors, completed counts, output lengths, source JSONL SHA256s if the raw JSONLs remain untracked, and a fail-closed recompute command.

4. **Original Loop-6 work remains incomplete.**

   AC-7, AC-8, and gated AC-10 are still pending. AC-5 remains directional-only because the strict DS client SLO still misses conc 32/64 TTFT and all-conc TPS. These are active plan tasks, not grounds for `COMPLETE`.

## Blocking Side Issues

1. **Cross-node benchmark scripts can silently target the wrong host.**

   `development/benchmark.sh` and `development/benchmark_baseline.sh` define `HOST` and use it for `/get_server_info`, but the `sglang.bench_serving` invocation passes only `--port` and never `--host` (`development/benchmark.sh:22-85`, `development/benchmark_baseline.sh:27-85`). The Round 11 summary says this already caused the Round 10 "DSA" bench to hit node0 instead of node1. This blocks safe AC-7 and any scripted cross-node AC-6 rerun.

   Required fix: add `--host "${HOST}"` to both benchmark scripts, then run a tiny cross-node smoke that proves bench_serving and `/get_server_info` target the same host before publishing any more cross-node benchmark artifacts.

2. **Strict DS client SLO still blocks the ultimate goal.**

   AC-5 remains a directional result, not a shippable pass: conc 32/64 TTFT and all-conc per-request TPS still miss the strict target. This remains a mainline blocker and must not be hidden by AC-6/AC-7 hardening.

## Queued Side Issues

None newly queued. The benchmark `HOST` issue is blocking, not queued, because it directly threatens the next hardware artifacts.

## Goal Alignment Check

| AC | Status | Evidence / blocker |
|----|--------|--------------------|
| AC-1 | MET | Strategic decision doc verified earlier. |
| AC-2 | MET | Feasibility budget and binding int8 lever verified earlier. |
| AC-3 | MET | Compact int8 table, scale-sidecar consumers, launcher, real-mask NIAH, and microbench verified earlier. |
| AC-4 | MET | Lifted 0.7 operating point, HBM budget, and no-OOM proof verified earlier. |
| AC-5 | PARTIAL | Evidence/attribution verified; strict DS SLO still fails. |
| AC-6 | PARTIAL | DS radix-on opt-in toggle and DSA no-table are proven; DSA-default SLO proof still fails/overclaims because conc-64 TPS is <30 and the required 320-prompt artifact was not accepted as passing evidence. |
| AC-7 | NOT MET | 3-trial DS+DSA lifted-point re-sweep pending. |
| AC-8 | NOT MET | Lifted ~70K-token servability probe pending. |
| AC-9 | MET | Real-token within-budget harness and live rerun verified in Round 10 review. |
| AC-10 | NOT MET | Correctly gated behind full Tier-1 completion. |

Forgotten items: none. Every original plan task is represented in Active, Completed, or the gated AC-10 path. Deferred items: no tracker-level explicit deferral, but Round 11's substitution of `num_prompts=64` for the contract's 320-prompt AC-6 SLO proof is rejected as an unjustified plan change.

## Required Implementation Plan

1. Fix the benchmark host propagation first: add `--host "${HOST}"` to `development/benchmark.sh` and `development/benchmark_baseline.sh`; run a small cross-node smoke proving `bench_serving` hits the same host captured in the meta sidecar.
2. Re-run the DSA-default AC-6 SLO proof against the correct host with the contract parameters: no DS flags, `MEM_FRACTION_STATIC=0.85`, radix-on, conc 16/32/64, `NUM_PROMPTS=320`, `TRIALS=1`, `WARMUP_SECONDS=120`, `MEASUREMENT_WINDOW_S=600`. Publish exact/recomputable tracked evidence.
3. If that run fails or is still judged methodologically invalid, do not substitute a passing-looking smaller workload. Mark AC-6 partial/failed in `ac6_optin_dsa_default_product.md` and the tracker, then request an explicit plan evolution for the DSA-default SLO methodology.
4. If using the `num_prompts=64` methodology after approval, still satisfy the SLO literally: fix or explain the DSA conc-64 TPS miss before claiming AC-6 met, and commit recomputable evidence rather than summary text.
5. After AC-6 is honestly resolved, complete AC-7 exactly as planned with 3 DS+DSA trials, radix-on on both sides, correct host targeting, and refreshed `ac11_resweep.md` / `ac11_analysis.md`.
6. Complete AC-8 with the lifted ~70K-token probe, recording HTTP 200 with capacity/no instability or a characterized ceiling.
7. Start AC-10 only after AC-3 through AC-9 are verified; record NIAH 4K/16K/64K recall deltas plus TPS/TTFT cost.

## Goal Tracker Update

I updated the mutable section of `goal-tracker.md`:

- Plan version moved to Round 11 Review.
- Added an R11-review plan-evolution row.
- Kept task7/AC-6 Active as partial.
- Changed the AC-6 blocker from RESOLVED to PARTIAL.
- Added the cross-node benchmark `HOST` propagation bug as a blocking side issue.
- Left AC-7, AC-8, and gated AC-10 active/pending; no immutable section was changed.

## Validation Performed

- `git log --oneline -30`
- `git show --stat --oneline 0e1ce974d`
- Inspected `round-11-contract.md`, `round-11-summary.md`, `goal-tracker.md`, and Round 8-10 summaries/reviews.
- Inspected `ac6_optin_dsa_default_product.md`, `get_server_info_keys.json`, full AC-6 server-info captures, boot excerpts, `dsa_default_matches_loop5_baseline.txt`, `dsa_default_slo_np64.txt`, and `dsa_default_slo.txt`.
- Compared key DS/DSA server-info fields.
- Inspected `development/benchmark.sh` and `development/benchmark_baseline.sh` for `HOST` handling.
- `git diff --check 2fd2c6937..0e1ce974d`

NOT COMPLETE
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-12-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-30_06-27-19/round-12-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**Optional fallback**: if you could not safely update the mutable section of `goal-tracker.md` directly, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Blocking Side Issues: bug Y blocks AC-2"]
- [E.g., "Add to Queued Side Issues: cleanup Z is non-blocking"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and reconcile the Goal Tracker if justified.
