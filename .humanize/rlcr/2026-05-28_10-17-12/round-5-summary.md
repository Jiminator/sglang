# Round 5 Summary

## Work Completed
Built the single-node sequential AC-Q quality smoke (resolving #G) and ran it end-to-end
on 8x H200. AC-Q is **evidenced but NOT met** — 3/4 gates pass; the ROUGE-L gate misses on
benign temperature-0 decode drift (not a correctness regression). Reported honestly with a
Goal Tracker Update Request rather than altering the immutable threshold.

- **#G resolved — sequential harness.** Two TP=8 servers cannot co-reside on one 8-GPU
  node, but the smoke required both `DS_BASE_URL` and `DSA_BASE_URL` up at once and
  interleaved DSA/DS per prompt. Split into:
  - `test/manual/_dsv32_quality_smoke_lib.py` — shared prompt fixtures, generation,
    pure-Python ROUGE-L / first-n overlap, the load-bearing `compute_gates()`, plus
    `capture_reference_outputs()` and `evaluate_against_references()`.
  - `test/manual/test_dsv32_quality_smoke.py` — kept the legacy simultaneous unittest, added
    a `capture`/`compare` CLI (capture writes the 20+5 DSA refs with only DSA up; compare
    loads them with only DS up, scores the gates, exits non-zero on any miss).
  - `test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py` — 8-test CPU
    regression proving `compute_gates` verdicts + the capture→compare round-trip with no
    live servers (`generate` monkeypatched).
- **Generation switched to `/v1/chat/completions`.** The raw `/generate` path returned
  dataset/JSON scaffolding for the instruction prompts and **empty** outputs for the long
  NIAH prompts (base-model continuation with no chat template). Chat completions apply the
  template server-side, so the model actually answers (Hamlet→"William Shakespeare",
  NIAH→"ZEBRA-7"); both DS and DSA use the identical path.
- **Ran it on hardware (sequential):** booted DSA (radix-off, cluster path) → captured 20+5
  coherent references (NIAH 5/5) → shut DSA down → booted DS (radix-off, cluster path) →
  compared.

## Files Changed
- `test/manual/_dsv32_quality_smoke_lib.py` (new) — shared library; generation via chat
  completions.
- `test/manual/test_dsv32_quality_smoke.py` — refactored to use the lib; `capture`/`compare`
  CLI; legacy simultaneous unittest retained.
- `test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py` (new) — CPU regression.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — repointed
  `TestDSv32SmokeHelpers` at the shared lib (helpers moved there).
- `runs/20260528_dsv32_mvp/` — `dsa_quality_refs.json`, `dsv32_quality_smoke.json`,
  `ac_q_analysis.md` (logs gitignored).
- Commits: `99ac93691` (harness + regression), `d8fce372a` (AC-Q evidence + analysis).
  Pushed to remote.

## Validation
- `pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py -q` → **262 passed**
  (254 DS unit + 8 new sequential regression).
- Hardware AC-Q gate verdict (`dsv32_quality_smoke.json`):
  - prefix_match_rate = 0.80 (≥ 0.80) — **PASS**
  - mean_rouge_l = 0.726 (≥ 0.85) — **FAIL**
  - niah_mini_recall = 5/5 (≥ 4/5) — **PASS**
  - first_8_tokens_divergence = 0 (== 0) — **PASS**
  - **AC-Q overall: FAIL** (hard gate).
- Analysis (`ac_q_analysis.md`): ROUGE-L median = 1.000; all short factual answers match
  DSA verbatim, NIAH recall perfect. The mean is dragged down by 7 open-ended explanatory
  prompts where DS and DSA agree on the answer + first tokens then diverge in wording/length
  under greedy decoding with different attention numerics — benign drift, not a regression.

## Remaining Items
- **AC-Q not met** as literally defined — see the Goal Tracker Update Request below.
- **TIER-2:** task11 AC-10 radix flip, task12 AC-1b chunked-prefill probe, task13 AC-11
  (gated on #F), task14 AC-12, task15 evidence bundle.
- **#F (queued):** DS KV-pool/effective-concurrency limit at mem 0.6 — resolve before the
  AC-11 TTFT comparison (does not affect AC-Q, which is sequential single-prompt).
- Queued cleanup: stale `calibrate.py` operator recipe docstring.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-dsv32-quality-smoke-needs-chat-template
- Notes: Added `BL-20260529-dsv32-quality-smoke-needs-chat-template` — instruction/QA eval
  prompts must go through `/v1/chat/completions` (chat template applied), not raw
  `/generate`, or the served model returns degenerate continuations / empty NIAH outputs;
  also records that ROUGE-L over long temp-0 free-form generations is a noisy DS-vs-DSA
  signal (greedy divergence after the shared prefix).

## Goal Tracker Update Request

### Requested Changes:
- Record AC-Q as **executed with evidence but the gate is NOT met** (mean_rouge_l 0.726 <
  0.85); keep task9's harness work (#G) as resolved/verified.
- Decide the reconciliation for the ROUGE-L gate. Options (I did NOT change the immutable
  AC or threshold myself):
  1. Treat the ROUGE-L gate like AC-11's directional targets (DEC-7) — a documented miss +
     follow-up rather than a hard TIER-1 blocker — given the evidence shows DS quality is
     substantively intact (median ROUGE-L 1.0, all short answers exact, NIAH 5/5, prefix
     0.80, zero first-8 divergence).
  2. Refine the AC-Q measurement to reduce benign long-generation noise (e.g. compare a
     bounded answer span, or lower `max_tokens` for the open-ended prompts) — a plan change
     requiring your approval.
  3. Treat it as a genuine DS gap and require investigation before TIER-1 is "complete".

### Justification:
The Ultimate Goal's TIER-1 narrative is "quality smoke passes on 20 paired prompts." The
run shows DS reproduces DSA's *answers* faithfully; the only gate miss is mean ROUGE-L over
free-form 256-token generations, which is inherently sensitive to temperature-0 greedy
divergence between two attention implementations (median ROUGE-L is a perfect 1.0). Lowering
the threshold unilaterally would be gaming the gate, so I'm surfacing the evidence + analysis
(`ac_q_analysis.md`) for you to reconcile against the immutable AC definition.
