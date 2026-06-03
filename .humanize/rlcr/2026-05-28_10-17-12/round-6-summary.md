# Round 6 Summary

## Work Completed
Diagnosed the AC-Q failure (#H) to a definitive root cause on 8x H200 and resolved #I. The
mainline outcome is the contract's "not-a-DS-bug → measurement-change proposal" branch:
**AC-Q remains NOT MET**, but the failure is now proven to be temperature-0 greedy decode
degeneration, not a DS correctness or CUDA-graph bug, and a measurement-change proposal is
filed for approval (no threshold/prompt/decoding default changed unilaterally).

- **#H diagnosis (controls on hardware, `ac_q_diagnosis_round6.md`):**
  - **Eager == CUDA-graph.** DS booted with `--disable-cuda-graph` produces the *identical*
    `17 * 23` repetition loop (and the same dropped `17`). → not a CUDA-graph bug.
  - **DS knows the answers.** Asked concisely: `17 * 23`→`391`, primes→`53, 59, 61`, SI
    unit→ampere. → not a DS correctness/knowledge bug.
  - **DS escapes under sampling.** Temp 0.5 on the same prompt → reaches `391`. → the loop
    is a greedy (temperature-0) decode artifact.
  - **Trajectory divergence is early.** Offline first-N-token ROUGE: N=8 → 0.894, N=16 →
    0.815, N=32 → 0.790, full → 0.726. DS and DSA diverge in trajectory/verbosity within
    ~16 tokens on the 7 open-ended prompts, so a bounded-token comparison does not rescue
    the gate.
  - Root cause: DS decode attention differs numerically from DSA's, so temperature-0 greedy
    decoding follows different (both valid) trajectories on long-CoT prompts; on `17*23` DS
    falls into a greedy repetition loop. There is no DS code fix for temp-0 greedy
    degeneration. The ROUGE-L gate measures DS-vs-DSA lexical trajectory identity, which two
    different attention mechanisms cannot satisfy on open-ended generation; the other three
    gates (answer-agreement) pass.
- **#I resolved.** Hardened `_validate_reference_artifact` to enforce the exact 20 smoke
  prompts + 5 NIAH prompts/needles position-by-position; added 3 regressions
  (truncated / reordered / wrong-needle). A truncated/reordered reference can no longer pass
  a future compare on a subset.

## Files Changed
- `test/manual/_dsv32_quality_smoke_lib.py` — `_validate_reference_artifact` enforces the
  exact committed fixture (#I).
- `test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py` — +3 rejection
  regressions (now 11 tests).
- `runs/20260528_dsv32_mvp/` — `ac_q_diagnosis_round6.md`, `ds_diag_graph_chat_1723.json`,
  `ds_diag_eager_chat_1723.json`.
- Commit `70bb52a15` (diagnosis + #I). Pushed to remote.

## Validation
- `pytest test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py
  test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` → **265 passed**
  (254 DS unit + 11 sequential regression).
- Hardware controls recorded as artifacts (eager-vs-graph, concise-answer, sampling-escape,
  offline first-N ROUGE table).

## Remaining Items
- **AC-Q not met** pending the measurement-change decision (Goal Tracker Update Request below).
- **TIER-2:** task11 AC-10 radix flip, task12 AC-1b, task13 AC-11 (gated on #F), task14
  AC-12, task15 evidence bundle.
- **#F (queued):** DS KV-pool/effective-concurrency at mem 0.6 — resolve before AC-11 TTFT.
- Queued cleanup: stale `calibrate.py` operator recipe docstring.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-ds-greedy-decode-degeneration-vs-dsa
- Notes: Added the lesson that a DS-vs-DSA temp-0 quality miss must first be triaged with
  eager-vs-graph + concise-prompt + sampling controls; here they prove the `17*23` loop is
  greedy decode degeneration (DS answers are correct), not a DS bug, and that lexical ROUGE
  over open-ended temp-0 generations measures trajectory identity two different attention
  mechanisms cannot meet. (Complements `BL-20260529-dsv32-quality-smoke-needs-chat-template`.)

## Goal Tracker Update Request

### Requested Changes:
- Confirm #H as **diagnosed: temperature-0 greedy decode degeneration, not a DS correctness
  or CUDA-graph bug** (evidence: `ac_q_diagnosis_round6.md`, eager==graph, concise→correct,
  sampling→escape). Mark #I resolved.
- **Decide the AC-Q measurement** (I did NOT change the immutable AC, the threshold, the
  prompt fixture, or the decoding default). Candidate resolutions, in my recommended order:
  1. **Change the AC-Q decoding config** to one that is not greedy-degenerate, applied
     identically to DS and DSA with a fixed seed (e.g. a small repetition penalty, or a low
     fixed temperature) — keeps determinism, removes the greedy-loop confound, and tests DS
     quality on the same footing as DSA.
  2. **Make AC-Q an absolute DS-quality gate** on the known-answer prompts (DS output
     contains the expected answer / is non-degenerate) instead of DS-vs-DSA lexical ROUGE —
     but note `17*23` still fails absolutely under temp-0/256-token greedy, so this needs (1)
     too.
  3. **Constrain the open-ended prompts** to request concise answers (where DS == DSA).
  The harness already supports running these via the `capture`/`compare` CLI once a measure
  is chosen; the exact-fixture validator (#I) guards a legitimate future pass.
- If instead you judge this a DS decode regression that must be fixed in code, please say so
  — but the controls show DS produces correct answers and only greedy long-CoT degenerates,
  for which no DS-attention code fix exists (DS cannot be bit-identical to DSA).

### Justification:
The Ultimate Goal's TIER-1 narrative is "the quality smoke passes." The evidence shows DS's
*answers* are correct (concise arithmetic, primes, NIAH 5/5, prefix 0.80, zero first-8
divergence); the only failing gate, mean ROUGE-L, is confounded with temperature-0 greedy
trajectory chaos that DS is more susceptible to than DSA on long CoT. Lowering the threshold
or editing prompts unilaterally would game the gate, so I'm surfacing the controlled
evidence and concrete options for you to reconcile against the immutable AC. Once a
measurement is approved, the next round reruns the sequential `capture`→`compare` workflow to
a genuine all-four-pass.
