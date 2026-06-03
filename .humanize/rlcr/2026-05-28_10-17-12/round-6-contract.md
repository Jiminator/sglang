# Round 6 Contract

## Mainline Objective
**Resolve #H — the DS AC-Q decode failure — to a definitive root cause, then either fix
the DS behavior and rerun the sequential AC-Q smoke until all four gates pass, or (only if
the controls prove it is not a DS bug) produce the root-cause evidence plus a specific
proposed resolution for Codex/user approval.** No silent threshold or prompt-fixture change.

Codex verified DS produces degenerate output where DSA is clean: for `Compute 17 * 23`, DS
drops the leading `17` and loops `17 × 23 = 17 × 23 …` to the 256-token cap, never emitting
`391`; for `List three primes 50–80`, DS is verbose and truncates before listing three. The
Pythagorean prompt (which passed) is fine. These are short prompts (seq ≪ top_k=2048), where
DS sparse-decode should select all tokens and behave dense-equivalently — so DS diverging
from DSA this early is the thing to explain.

## Target ACs (≤ 2)
- **AC-Q** — the four-gate paired quality smoke must pass (`prefix_match_rate ≥ 0.80`,
  `mean_rouge_l ≥ 0.85`, `niah_mini_recall ≥ 4/5`, `first_8_tokens_divergence == 0`). This
  round's success is a verified root cause + a fix that makes AC-Q pass, or a definitive
  not-a-DS-bug diagnosis + an approved measurement-change proposal.

## Blocking issues in scope
- **#H — DS AC-Q decode failure, not proven benign.** Investigate on 8x H200 via controls
  that distinguish a DS decode/selection bug from greedy-numerics divergence:
  1. Reproduce `17 * 23` and `List three primes 50–80` on DS via `/v1/chat/completions`,
     temp 0, and capture `meta_info["double_sparsity"]` (`sparsity_rate`, `selected_tokens`,
     `dense_fallback`) for these short sequences. If DS selects < all tokens when seq ≤
     top_k (sparsity_rate < 1 / dense_fallback wrong), that is a context-dropping bug.
  2. Eager-vs-graph control: re-run the same prompts on DS booted with `--disable-cuda-graph`.
     If the loop disappears in eager, it is a regular-CUDA-graph decode bug; if it persists,
     it is selection/kernel numerics.
  3. Confirm DSA reaches `391` at the same knobs (reference already shows it does).
  - Then fix the DS-side defect the controls point at, or — only if they show DS attention
    is selecting all tokens and is dense-equivalent yet still diverges (inherent greedy
    numerics) — write the evidence and a proposed AC-Q measurement change for approval.

## Queued / explicitly out of scope this round
- **#I — harden `_validate_reference_artifact`** to enforce the exact 20 smoke prompts + 5
  NIAH needles (counts/order/needles) + a truncated/reordered regression. Cheap CPU work
  that gates *accepting* a future AC-Q pass; do it this round if the mainline fix lands with
  time to spare, else keep queued. Must NOT displace the #H investigation.
- **#F — DS KV-pool/effective-concurrency at mem 0.6** (blocks AC-11 TTFT only).
- **TIER-2:** task11 AC-10, task12 AC-1b, task13 AC-11, task14 AC-12, task15 bundle.
- Stale `calibrate.py` operator recipe docstring.

## Round success criteria
1. A definitive, evidence-backed root cause for the DS `17 * 23` repetition loop:
   DS `meta_info["double_sparsity"]` on the short prompt recorded; eager-vs-graph control
   recorded; DSA reference confirmed clean at the same knobs. Artifacts saved under
   `runs/20260528_dsv32_mvp/`.
2. EITHER: the DS defect is fixed (with a unit/CPU regression where the bug is unit-testable),
   the sequential `capture`→`compare` AC-Q workflow is rerun on hardware, and all four gates
   pass (`all_pass=true`); OR: controls prove it is not a DS bug, and a precise measurement-
   change proposal is filed as a Goal Tracker Update Request (threshold/prompts unchanged in
   code until approved).
3. DS unit suite stays green; any new regression passes.
4. Commit + push each step. Goal tracker updated (task9, #H); `round-6-summary.md` with a
   BitLesson Delta. No immutable-section changes.

## Known risks / notes
- The loop may be inherent temperature-0 greedy divergence between the DS and DSA decode
  kernels (not a "bug"); if so the honest outcome is a measurement-change proposal, not a
  forced code change. Decide based on the controls, not convenience.
- Operational: don't kill the pre-existing port-30000 router; use a free port; verify
  `nvidia-smi` clear before each boot; standalone `pkill`/`commit`/`push`.
