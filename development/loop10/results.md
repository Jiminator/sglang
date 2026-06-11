# Loop 10 Ledger — DS-on Decode Dead-Width Tax

Rewrite-over-append: this file holds ONE authoritative current-state section, rewritten whenever
state changes. History lives in git. Plan: `development/loop10/plan.md`. Queue (single source of
truth for task state): `development/loop10/queue.md`.

## Current state (round 6 — LOOP COMPLETE: AC-1.2 owner-re-scoped to characterized on the completed frontier; all other bars met at hard AND stretch)

### The number

| Reference | µs / 10-step decode window | ratio | status |
|---|---|---|---|
| Loop-9 final landed = loop-10 baseline (`loop9/runs/20260611_r1/`) | 480,989 | 1.403× | frozen starting point |
| **Loop-10 current landed (M1 compact variants + task11 tb-512 + task7 bf16-authoritative top-k)** | **361,824** | **1.055×** | `runs/20260611_task7/` |
| Case-2 DSA floor (frozen, never re-run) | 342,857 | 1.0× | target reference |
| AC-1 hard bar ≤420,000 | — | ~1.23× | **MET** |
| AC-1 stretch ≤395,000 | — | ~1.15× | **MET** (beats the M5 projection band 1.10–1.15×) |

### Per-bucket state vs bars (named kernels, `runs/20260611_task7/cmp_vs_loop9r1.txt`)

| Bucket | R1 µs | now µs | hard bar | stretch | status |
|---|---|---|---|---|---|
| DS transport: `all_reduce_two_shot_kernel<bf16,8u>` 14,137 + `direct_copy` 2,206 + `bfloat16_copy` 1,280 | ~108–111k | **17,623** | ≤60k | ≤45k | **hard + stretch MET** (AR kernel boot-variance 14.1k↔35.4k across gate boots — skew absorption; within bars at every observed boot) |
| `_logical_score_kernel` | 36,908 | **22,887** | ≤20k | ≤15k | **RE-SCOPED to characterized (owner, DEC-L10-3)** — the bar sits below the landed kernel's measured cold-cache floor; completed frontier in DEC-L10-2/3 |
| DS radix top-k (`_radix_hist` 18,501 + `_block_count` 1,338 + `_emit` 2,546 + `_block_prefix` 886) | ≈36,300 | **≈23,271** | ≤28k | ≤24k | **hard + stretch MET** → task9 DROPPED condition-false |
| shared non-DS topk/sort | 20,524 | ~20.5k | n/a | n/a | control, flat |
| TOTAL | 480,989 | **361,824** | ≤420k | ≤395k | **both MET** |

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
- AC-1.2 remained above the bar after this landing (22,869 → 22,887 across boots). The
  authoritative AC-1.2 record — including the round-5 retraction of the round-4 "lower bound"
  framing and the corrected one-process measured frontier — is DEC-L10-2 below.

### task8 DROPPED (round 3, measured cause — Codex analyze `reviews/task8_transport_verdict.md`)

Incumbent pinned two-shot at binding 31.3 µs/call (320 KiB) beats the best alternative evidence
(eager NCCL 38.5 µs/call at a smaller 267 KiB; coordinator CA 51.9; full-width NCCL 105.9 ≈ CA
104.1); ONE_SHOT_PUSH hard-errors >160 KiB (a declared push would crash at the op point);
declared ONE_SHOT_PULL has no measured win. No candidate justifies value-affecting churn
(declaration + third re-freeze + recall-blind risk) on a bucket 2× under its stretch bar.

### task7 BANKED (round 3, KEEP — exact, zero diffs)

- Landed (commit `fac0b0cfa`, gates `d6c511b20`, evidence `runs/20260611_task7/`): the radix
  suite's score loads upcast in-register (identity for fp32, exact for bf16 — required before
  the `_key_of` fp32 bitcast); with the radix active + bf16 reduce, the reduced bf16 buffer is
  the authoritative top-k input and `reduce_token_scores(copy_back=False)` skips the bf16→fp32
  copy-back; the per-request-valid mask applies to the authoritative buffer; oracle/anchor/
  legacy consumers keep fp32. CUDA-gated regression: bf16 vs exact-fp32-upcast selection
  identity under tie plateaus + non-finite contract.
- Gates: bs-1 AND op-point selcap bit-exact (zero diffs, zero identity changes — the exact
  claim PROVEN, no re-classification needed); recall 64.706%; Case-1 **361,824 µs**.

### task9 DROPPED (round 3, condition false — the plan's own trigger)

Post-M2 profile reads the radix suite at ≈23,271 µs ≤ 28k hard (and ≤ 24k stretch). Recorded
per queue protocol with the measured cause; no redesign performed.

### Gate baselines for task10+ exact changes

`runs/20260611_task7/` digests (bs-1 + op-point). Chain: m0_freeze → task4 → task6r2 → task11
→ task7, every hop either zero-diff-proven or declared (DEC-L10-1).

### DEC-L10-2 (rounds 4–5): AC-1.2 owner adjudication and the corrected measured-frontier record

The owner ruled (AskUserQuestion, round 4): **"Keep the bar; keep trying exact only; once
that's exhausted, authorize int8 signatures."**

**Round-5 correction (Codex round-4 review)**: the round-4 "stripped lower bound" proof was
INVALID — its 25.47 µs/call exceeded the production kernel's own isolated 23.58 (cross-process
clock states; a "bound" slower than a real implementation is no bound). It is retracted. The
corrected, one-process, interleaved-replay evidence (`task11_exact_floor_harness.py`,
artifacts `runs/20260611_task11/exact_floor_random.json` / `exact_floor_page64.json`;
`task11_headsplit_prototype.py`, artifact `headsplit_proto.json`):

1. **The landed kernel IS the measured optimum of its family.** Production tb=512 measures
   20.74 (random) / 19.94 (page64) µs/call isolated — at the 19.64 µs/call isolated budget.
   Every measured structural variant is SLOWER: tb=256 (+5%), tb=1024 (+70%), fewer workers
   (+70%), one-block-per-program grid (+24%), stripped-same-structure with the math DELETED
   (+75% — the compute-memory interleave is a local optimum, so no stripped-gather bound
   exists), head-split hs2/hs4 via bitwise-exact atomic_max (+17%/+57% — duplicated per-split
   indirection loads outweigh parallelism gains).
2. **The real-profile miss is cold-cache gather bandwidth, not kernel structure**: binding
   bucket 22,887 µs = 29.34 µs/call vs the bar's 25.64; the kernel achieves ~2.33 TB/s
   effective on 68.5 MB/call of scattered signature gathers in-context, where the bar implies
   ≥2.67 TB/s; isolated replay overstates achievable bandwidth via cross-replay L2 reuse.
3. **The authorized int8 fallback measured non-viable** (round 4, `roofline_probe.json` —
   retained as a NOTES artifact; its per-layout numbers are reproducible via
   `task11_roofline_probe.py --layout {random,page64}`): the gather is transaction-limited,
   not byte-limited (int8 floors 26.14/23.68 ≈ fp16's); best-case projected real bucket
   ~21.5–22k vs the 20k bar. int8 NOT landed — decline-with-evidence of the conditional
   authorization, flagged for override.

**AC-1.2 disposition: NOT MET.** The bar is not formally "proven impossible" — the corrected
record is a measured frontier, not a mathematical bound — but every implementable exact lever
in the measured family is exhausted, the landed kernel sits at the isolated budget, and the
residual is cold-cache bandwidth physics. The bucket closed 36,908 → 22,887 (−38%).

### task8 measurement contract SATISFIED (round 4)

Same-shape 8-rank matrix at [32, 5120] bf16 (`task8_transport_matrix.py`,
`runs/20260611_task8/task8_matrix.json`): captured-replay two-shot **14.85** vs one-shot-pull
20.42 vs NCCL 69.73 µs/call — the incumbent wins the binding mode by 27% / 4.7×. The eager
column INVERTS the ranking (NCCL 33.7 "wins" eager), retroactively explaining the loop-9
spike-bench conclusion as an eager artifact (BitLesson addendum recorded). DROP confirmed on
direct evidence.

### DEC-L10-3 (round 6): AC-1.2 owner-re-scoped to CHARACTERIZED on the completed frontier

The round-5 review mandated the data-layout lever as the remaining exact direction. It was
prototyped and measured (`task11_layout_prototype.py`,
`runs/20260611_task11/layout_proto.json`): head/dim-major `[H, D, T]` signatures are
bitwise-DIFFERENT from production (last-bit reduction-order change — the lever would have been
value-affecting, not exact) and 3–10× SLOWER in warm AND L2-flushed cold modes on both slot
layouts. The same harness delivered the causal closure: an L2-flushed cold-cache replay of the
LANDED kernel costs 27.84 (page64) / 29.38 (random) µs/call — reproducing the binding
in-context bucket (29.34) almost exactly. The AC-1.2 residual is the landed kernel's
cold-cache floor, and that floor exceeds the bar (25.64 µs/call).

With the frontier complete (8 measured levers: tb sweep, workers, block-grid, stripped,
bitwise-exact head-split, the layout transposition, plus the int8 fallback — every one worse
than the landed kernel), the owner ruled (AskUserQuestion, round 6): **AC-1.2 is RE-SCOPED to
a characterized finding.** The bucket's landed state: 36,908 → 22,887 µs (−38%), at the
measured cold-cache bandwidth ceiling of the table's access pattern. No threshold was silently
relaxed at any point: the bar was kept through two owner rulings and re-scoped only after the
reviewer-named final lever was measured.

### CLOSE-OUT (round 4) — task10 complete; the loop's task table is fully terminal

`reviews/task10_closeout.md` (Codex analyze): final AC tally — AC-1 total / AC-1.1 / AC-1.3
MET at hard AND stretch; **AC-1.2 NOT MET (owner-adjudicated DEC-L10-2, not re-scoped)**;
AC-2 MET under the declared regimes (DEC-L10-1 the sole exception, op-point zero-diff);
AC-2.1–2.4, AC-3, AC-4.1/4.2, AC-5 all MET with binding artifacts. Evidence pre-flight: no
cited-but-untracked artifacts; raw traces/.pt dirs explicitly local-forensic. Queue
reconciled: task0–task11 terminal, task8/task9 dropped with measured causes, cand1/cand2
dispositioned. Protocol: frozen references never re-run/replaced; baseline chain
m0_freeze → task4 → task6r2 → task11 → task7 integral (every hop zero-diff or declared).

**Loop headline: 480,989 → 361,824 µs — 1.403× → 1.055× vs the frozen DSA floor.**
