# Loop 10 Ledger — DS-on Decode Dead-Width Tax

Rewrite-over-append: this file holds ONE authoritative current-state section, rewritten whenever
state changes. History lives in git. Plan: `development/loop10/plan.md`. Queue (single source of
truth for task state): `development/loop10/queue.md`.

## Current state (round 2 — M1 BANKED)

### The number

| Reference | µs / 10-step decode window | ratio | status |
|---|---|---|---|
| Loop-9 final landed = loop-10 baseline (`loop9/runs/20260611_r1/`) | 480,989 | 1.403× | frozen starting point |
| **Loop-10 current landed (M1: compact W=5120 selector variants)** | **375,892** | **1.096×** | `runs/20260611_task6r2_gates/` |
| Case-2 DSA floor (frozen, never re-run) | 342,857 | 1.0× | target reference |
| AC-1 hard bar ≤420,000 | — | ~1.23× | **MET** |
| AC-1 stretch ≤395,000 | — | ~1.15× | **MET** |

### Per-bucket state vs bars (named kernels, `cmp_vs_loop9r1.txt`)

| Bucket | R1 µs | now µs | hard bar | stretch | status |
|---|---|---|---|---|---|
| DS transport: `all_reduce_two_shot_kernel<bf16,8u>` 24,420 + casts (`direct_copy` 4,602 + `bfloat16_copy` 1,397) | ~108–111k | **30,419** | ≤60k | ≤45k | **hard + stretch MET** |
| `_logical_score_kernel` | 36,908 | **23,080** | ≤20k | ≤15k | **OPEN — misses hard by 3.1k** → queue task11 (M4 contingency) |
| DS radix top-k (`_radix_hist` 18,394 + `_block_count` 1,338 + `_emit` 2,718 + `_block_prefix` 894) | ≈36,300 | **≈23,344** | ≤28k | ≤24k | **hard + stretch MET** → task9 condition currently FALSE |
| shared non-DS topk/sort | 20,524 | 20,554 | n/a | n/a | control, flat |
| TOTAL | 480,989 | **375,892** | ≤420k | ≤395k | **both MET** |

### What is landed (loop 10)

1. **Gate tooling (task1, R0)**: bucket-identity-tagged selection-capture dumps; loop-10 capture
   tool (bs-1 + op-point harness, diff/diff-digest hard gates, identity declarations); gate
   script with two selcap phases; frozen pre-change digests (bs-1 = loop-9 R1 fingerprint
   bit-exact; op-point raw_bs=29/padded 32/replay proven).
2. **task4 (R1, KEEP)**: `(bs, selector_width)` graph-variant keying gated to DS-on decode;
   config-borne width buckets; DSA metadata keyed by variant; all five gates passed with zero
   index diffs (sole declared identity change: graph_key int→tuple).
3. **task5+task6 (R2, KEEP — M1)**: compact W=5120 selector variants across the whole bs ladder
   with guaranteed full-width fallback; real-row width dispatch; DS score reduce pinned per-call
   `override_algo=TWO_SHOT_PULL` via `PinnedDSScoreReduceCA` (weak-contiguity REFUSAL; per-bucket
   transport evidence logging); per-request-valid masks prefix-sliced; **DSGraphState shared per
   width** (amendment below); fail-closed width-bucket config parsing. CPU: 383 DS unit tests.

### Gate baselines (diff targets for task7+ exact changes)

- **bs-1**: `runs/20260611_task6r2_gates/bs1_digest.json` (embeds the DEC-L10-1 declared
  transport state: compact buckets custom-AR two-shot).
- **op-point**: `runs/20260611_task6r2_gates/op_digest.json` (compact (32,5120) live,
  bit-identical to the pre-compact baseline — zero index diffs, 0/27,144 rows).
- recall: `loop9/runs/20260610_m0/recall_baseline.json` (frozen, reused; current 64.706%).
- Historical (superseded as diff targets, retained as evidence): `runs/20260611_m0_freeze/`,
  `runs/20260611_task4_gates/`.

### DEC-L10-1 (R2): compact-bucket transport flip declared value-affecting (plan-sanctioned)

The first M1 gate run FIRED the bs-1 hard diff gate (16 steps) — diagnosis:

1. **Pre-existing alignment accident**: bs-1 FULL-WIDTH reduces were never custom-AR eligible
   (1×202,756×2 B = 405,512 B fails the 16-byte divisibility in `should_custom_ar`) — every
   frozen bs-1 fingerprint embeds a silent NCCL reduce, while the op point embeds custom-AR
   two-shot: the pre-change transport was already shape-dependent. Compact buffers
   (bs×5120×2 B, always 16-aligned) made those buckets eligible → NCCL→pinned-two-shot flip →
   summation-order change → boundary churn ONLY where selection is score-sensitive
   (seq > top_k): measured median ~107/2,048 positions (5.2%), mean 130, max 695. Dense steps
   and full-width steps are bit-identical.
2. **Correctness probe** (`ar_algo_correctness_probe.py`, 8 ranks,
   `runs/20260611_task6_gates/ar_algo_probe.json`): every algorithm (two-shot, one-shot pull/
   push, size-based, NCCL) produces EXACT sums on exactly-representable inputs at all probed
   shapes incl. 10 KiB — order-only effect, no corruption. Side finding: ONE_SHOT_PUSH
   hard-errors >160 KB, reinforcing the pin.
3. Re-classification per the plan's sanctioned path; value-affecting teeth: cross-rank
   bit-identity PASS, run-to-run determinism PASS, recall 64.706% (+0.01pp) PASS, declaration
   here + queue. Replicating the alignment accident (forcing NCCL on compact) was rejected as
   enshrining a bug. **The op point is exact (zero index diffs) — the declared component only
   covers buckets whose full-width counterpart was NCCL-by-misalignment (bs-1-class).**

### Capture-memory amendment (R2, measured): DSGraphState shared per width

Per-variant ownership OOM'd capture at 41/104 (mem 0.7; `runs/20260611_task6_gates/`):
width-proportional scratch + selcap mirrors multiplied by the 52-entry ladder. Amended (dossier
§1 deviation, Plan-Evolution-logged): ONE shared DSGraphState per selector width at the global
max capture bs; same-width variants' DSAMetadata reference it (aliasing safe: one replay at a
time; every replay fully rewrites the rows it reads; dump padded_bs derives from the graph key).
Result: all 104 dual-width captures fit with **26.7 GB still free** (`capture_budget.txt`) —
this REALIZES the loop-9 M4 audit's recoverable headroom. Boot capture window ≈ 2 min.

### task11 BANKED (round 3, KEEP) — and the AC-1.2 roofline finding

- Landed (commit `302aed47b`): width-conditional logical-score `TOKEN_BLOCK` (512 compact / 256
  full), bitwise-invariant by construction (per-position label-dim reductions are
  self-contained), pinned by a CUDA-gated regression + the bench's cross-variant equality.
- Gates (`runs/20260611_task11/`): bs-1 AND op-point selcap bit-exact vs task6r2 under the full
  hard gate (zero identity changes); recall 64.706%; Case-1 total 385,276 µs (hard + stretch
  still met; the two-shot reduce shows boot-to-boot skew variance 24.4k↔35.4k, both within
  bars — per-bucket attribution primary).
- **AC-1.2 measured-infeasibility finding**: the bucket landed at **22,869 µs** (−211; R1 was
  36,908). The kernel is DRAM-roofline-bound: 29 rows × 4,608 positions × 512 B of signature
  gathers ≈ 68.5 MB/call → ~21 µs/call isolated floor; captured-replay sweep
  (`task11_logical_score_bench.py` + `runs/20260611_task11/task11_bench.json`) measures 23.3
  µs/call isolated at the tb=512 optimum (24.6 at tb=256; fewer-worker and larger-block variants
  worse) vs the bar's 25.6 µs/call with ~6 µs/call of real-context interference on top. The
  remaining levers are barred: int8 signatures (halves bytes) violates the frozen recipe's
  `signature_dtype: fp16`; approximate/hierarchical scoring violates the no-added-lossiness
  contract. **AC-1.2 hard (≤20k) is NOT MET and is assessed infeasible in the exact regime at
  the frozen op point**; stretch (≤15k) likewise. Disposition: documented for close-out
  adjudication — an immutable-AC re-scope requires explicit owner authorization (it is NOT
  silently relaxed here).

### task8 DROPPED (round 3, measured cause — Codex analyze `reviews/task8_transport_verdict.md`)

Incumbent pinned two-shot at binding 31.3 µs/call (320 KiB) beats the best alternative evidence
(eager NCCL 38.5 µs/call at a smaller 267 KiB; coordinator CA 51.9; full-width NCCL 105.9 ≈ CA
104.1); ONE_SHOT_PUSH hard-errors >160 KiB (a declared push would crash at the op point);
declared ONE_SHOT_PULL has no measured win. No candidate justifies value-affecting churn
(declaration + third re-freeze + recall-blind risk) on a bucket 2× under its stretch bar.

### Open items / next

- task7 (bf16-authoritative radix input, copy-back eliminated — commit `fac0b0cfa`): gate run in
  flight vs the task11 baselines; expected exact (zero diffs) with the `direct_copy` copy-back
  kernel (~4.6k/window) removed from the transport bucket.
- task9 (top-k redesign): condition read after task7's profile; currently ≈23.3k ≤ 28k —
  expected drop.
- task10 close-out after this round's reconciliation; AC-1.2 disposition escalated as above.
