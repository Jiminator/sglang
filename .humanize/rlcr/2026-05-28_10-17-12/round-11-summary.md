# Round 11 Summary

## Mainline objective (met)
Closed TIER 2 by **executing the AC-12 hard quality gate (task14) on real hardware** and
**assembling the task15 evidence bundle**. Both plan tasks ran. AC-12 is a **recorded HARD
FAILURE** — the honest, plan-anticipated outcome of DS's `top_k`-bounded long-context recall.

## Work completed

### AC-12 full quality gate (task14) — executed, HARD FAIL recorded
Ran the paired DS-vs-DSA gate on **two H200 nodes** (two TP=8 V3.2 servers can't co-reside on one
8-GPU node): DS radix-on via fixture artifact (mem 0.6, node 0), DSA radix-on (mem 0.85, node 1),
both at the locked Option B point. `/get_server_info` captured for both.

| Gate | DSA | DS | Δ (DSA−DS) | Threshold | Verdict |
|------|-----|-----|-----------|-----------|---------|
| MMLU 5-shot (200) | 89.00% | 89.00% | 0.00 pp | ≤ 1.0 pp | **PASS** |
| NIAH 4K (20) | 100% (20/20) | 75% (15/20) | 25.0 pp | ≤ 5 pp | **FAIL** |
| NIAH 16K (20) | 100% (20/20) | 5% (1/20) | 95.0 pp | ≤ 5 pp | **FAIL** |
| NIAH 64K (20) | served 20/20 | HTTP 400 (unservable) | — | ≤ 5 pp | **FAIL** |

Two real, non-bug mechanisms (`ac12_analysis.md`): (1) DS sparse decode is `top_k=2048`-bounded →
needle recall degrades monotonically with context (75% → 5%); (2) DS at mem 0.6 has
`max_total_num_tokens=53,056` < the 69,970-token 64K prompt → cannot admit 64K (DSA pool 910,784).
MMLU passes because short prompts (seq ≤ top_k) use effectively-dense selection → DS short-context
quality is identical to DSA. DSA (native long-context sparse attention) recalls 100% throughout,
validating the harness. **AC-12 is hard pass/fail (DEC-7 directional handling is AC-11-only) →
recorded as a hard failure, not reclassified.** Therefore the **loop4-compatible MVP is NOT
complete**: TIER-1 smoke complete, TIER-2 incomplete (AC-10/11/6/1b done, AC-12 MMLU passes,
AC-12 NIAH hard-fails).

### Enabling work
- **HOST knob (blocking #B1):** added a `HOST` env knob (default 127.0.0.1) to both Option-B
  launchers, passed through as `--host`, so the DSA baseline binds `0.0.0.0` for cross-node reach.
  Locked Option-B flags + default localhost behavior unchanged; +1 lock regression.
- **Node-1 sync:** node 1 was on a stale commit (`cb6004a36`, pre the NSA→DSA boot-chain fix and
  pre the locked Option-B launchers) — DSA crashed with `'…' object has no attribute 'use_nsa'`.
  Fast-forwarded node 1 to HEAD (`7478c27a0`); DSA then booted cleanly cross-node.
- **Harness transport fix:** `_generate` now uses `/v1/chat/completions` for NIAH (raw `/generate`
  returns an immediate-EOS empty string for these instruction prompts → would falsely pass the
  paired gate 0/0 on both servers) and KEEPS raw `/generate` for MMLU 5-shot (a few-shot
  *completion* benchmark the chat template breaks: verified raw 10/10 vs chat 0/10). Thresholds
  and prompt fixtures unchanged.

### Evidence bundle (task15)
`runs/20260528_dsv32_mvp/evidence_bundle.md` — AC-by-AC index with artifact paths, mask
provenance/SHA, server args/server_info, CUDA-graph + chunked-prefill status, radix fixture, and
an explicit **AC-10 label-capture provenance note** (resolving the Round-8 queued item). States
AC-11 as "executed; directional TTFT/TPS target missed; #F admission caveat + follow-up filed" and
AC-12 as a hard failure → loop4 MVP incomplete.

## Files changed
- `development/serve_native_nsa.sh`, `development/serve_double_sparsity.sh` — `HOST` knob.
- `test/registered/unit/development/test_option_b_scripts.py` — HOST-knob lock regression.
- `test/manual/test_double_sparsity_v32.py` — `_generate` task-specific transport (NIAH chat /
  MMLU raw); `_run_niah` uses chat.
- `runs/20260528_dsv32_mvp/` — `ac12_analysis.md`, `evidence_bundle.md`,
  `ac12_{ds,dsa}_server_info.json`, `ac12_results/` (MMLU+NIAH JSON, pytest summary, DS boot
  excerpt).
- Commits `7478c27a0` (HOST knob), `1a1293f01` (AC-12 + bundle). Both pushed.

## Validation
- **407 CPU tests pass** (`test_ac11_comparator` + `test_double_sparsity_unit` +
  `test_dsv32_quality_smoke_sequential` + `test_option_b_scripts` + `test_ac12_helpers`); up from
  359 (+47 ac12_helpers, +1 HOST-knob lock).
- Hardware: both servers booted at the locked Option B point (server_info captured); AC-12 gate
  ran to completion (`3 failed, 1 passed, 2 skipped`); all gate artifacts written and copied to
  `ac12_results/`. Both servers shut down afterward; both nodes' GPUs freed (the pre-existing
  port-30000 router was already down at round start and was not touched).

## Remaining Items
- **No mainline work remains** — all plan tasks (task1–task15) have been executed. AC-12 is a
  recorded hard failure; per the Ultimate Goal this makes the deliverable a TIER-1 smoke milestone
  with TIER-2 loop4 quality not met (recorded, not a build-break to fix).
- **Queued (out of scope, documented):** (a) comparator per-side `mem_fraction_static` validation
  hole — tighten when the comparator is next touched (AC-12 didn't touch it); (b) AC-11 directional
  performance follow-up (TokenLabelTable / KV-budget) — same lever now also bounds AC-12 64K
  admission; (c) stale `calibrate.py` `--tp 1` recipe docstring (doc-accuracy only; mask+SHA are
  authoritative).

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260529-ds-longcontext-needle-recall-vs-topk, BL-20260529-dsv32-quality-smoke-needs-chat-template
- Notes: Updated the long-context-recall lesson with the R11 AC-12 **quantified** result (DS NIAH
  recall 75%→5%→unservable at 4K/16K/64K vs DSA 100%) and a SECOND, independent limit it now
  documents: at mem 0.6 the DS KV pool is only ~53K tokens, so a >pool prompt is rejected (HTTP
  400) as an admission failure distinct from a recall miss — distinguish them from the server log.
  Updated the chat-template lesson with the R11 **counterpoint**: transport is task-specific — NIAH
  instruction prompts need chat completions, but MMLU few-shot *completion* needs RAW /generate
  (chat broke MMLU 0/10 vs raw 10/10). No new lesson added (both findings extend existing entries).

## Goal Tracker Update Request

### Requested Changes:
- Move **task14 / AC-12** to Completed and Verified as **EXECUTED with a recorded HARD FAILURE**
  (gate run on hardware, honest pass/fail recorded; MMLU PASS, NIAH 4K/16K/64K FAIL).
- Move **task15 / evidence bundle** to Completed and Verified (`evidence_bundle.md` assembled).
- Confirm the **AC-10 label-capture provenance** queued item RESOLVED (explicit provenance note
  added to `evidence_bundle.md`).
- Record the **comparator per-side `mem_fraction_static` hole** as a queued (non-blocking) item to
  fix when the comparator is next touched.

### Justification:
Both remaining plan tasks were executed this round. AC-12 is a hard pass/fail gate (DEC-7
directional handling is explicitly AC-11-only), so its failure is recorded as a hard failure, not
reclassified as directional — exactly per Codex's Round-10 directive ("If NIAH 64K fails … publish
that as a hard AC-12 failure with evidence. Do not reclassify AC-12 as directional. … If AC-12
fails, the bundle must say the Loop4-compatible MVP is not complete"). The Ultimate Goal itself
anticipates this terminal state: with AC-12 full quality not met, the deliverable is a TIER-1 smoke
milestone, not the loop4 MVP. The failure modes (top_k-bounded recall; mem-0.6 KV-budget admission
limit) are inherent DS design / operating-point tradeoffs, not bugs with a code fix that would turn
AC-12 green. No immutable AC or threshold was changed; the only harness change fixes the NIAH
measurement transport so the model actually answers (raw /generate returned empty), which makes the
gate meaningful rather than vacuous.
