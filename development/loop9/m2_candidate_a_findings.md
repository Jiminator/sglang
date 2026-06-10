# M2 Candidate A — sgl-kernel fast_topk_v2 adaptation: measured evaluation

Date: 2026-06-10. Probes/benches on H200 (cuda:0), shapes = served Case-1
([29, 202752] fp32 scores, live window 4608, K=2048). Today's torch pipeline
(two full-width `torch.topk` passes + searchsorted) = 203.7 µs/call ≈ 155-159k
µs / 10-step window (the AC-1.2 bucket; gate ≤ 80k µs/window ≈ 103 µs/call).

## What the kernel gives

- `fast_topk_v2(score [B,L], lengths [B], topk=2048)` is SEQ-AWARE: it bounds
  its scan by `lengths[i]` per row. At the op point it costs **17.7 µs/call**
  (22.5 µs at length 16384) — 11.5× under today's pipeline. This is the
  structural fix for the full-width over-scan (the same dead-width tax that
  dominates the score-reduce and logical-score buckets).
- Dense rows (length ≤ 2048): emits identity ascending indices + -1 padding —
  but INCLUDES -inf-scored positions (probe: pos 5 with score=-inf returned),
  so unwritten/masked slots would be selected. The wrapper must gather scores
  at the returned indices, mask -inf entries, re-sort, and recompute
  valid_lengths.

## Why the naive adaptation is disqualified

- **The radix boundary bin admits ties by `atomicAdd` race** (topk.cu:233).
  Probe: 10 runs on an identical tie plateau straddling the K boundary
  returned DIFFERENT selections (run-to-run nondeterministic), and the picks
  are not lowest-position (e.g. head [1500, 1501, 1502, 1503, 1536]).
- Post-M1 this is not a corner case: the bf16 score transport quantizes scores
  to bf16 values, so exact fp32 ties near the selection boundary are common at
  width 202752. Per-rank racy tie picks ⇒ ranks select different tokens ⇒ the
  AC-2 cross-rank bit-identity HARD gate fails, and the frozen-oracle
  reproducibility (M0) breaks.

## Why the exact repair erases the win

A deterministic repair must find ALL positions tied at the threshold and admit
the lowest positions. Measured cost of the torch repair components
(full-width eq-compare + masked-position int64 where + full-width
`topk(largest=False)`): **338.6 µs/call — more than today's entire pipeline.**
The repair is full-width because tied positions can sit anywhere in the live
window and the compaction back to ascending order is itself a top-k-class
operation; a live-window-bounded Triton repair still needs an ordered prefix
over tied positions (two more passes) plus the 2048-candidate re-sort,
landing the assembled pipeline at ~80 µs/call — at which point it is a
bolt-on around a racy kernel rather than a clean design.

A low-bit position-packing variant (bf16-valued scores leave 15 zero mantissa
bits in fp32 that could carry a position tie-break) was considered and
REJECTED: it couples selection correctness to the bf16 transport being active
(fp32 escape hatch would silently break it), cannot fit the 18 position bits
the width needs, and inverts under negative scores — fragility for a
correctness-critical path.

## Verdict for the benchmark-off (task8 input)

Candidate A as specified (adapt fast_topk_v2 + output-contract wrapper +
boundary handling) is measured-out: the kernel core is excellent (17.7 µs) but
its tie semantics cannot meet the cross-rank gate, and every exact repair
considered costs back the win or worse. The candidate-B shape — a
DS-specific kernel that is seq-aware AND breaks ties deterministically by
(score desc, pos asc) inside the selection itself (composite-key radix
select with ordered boundary admission) — is the design that keeps the 10×
headroom. fast_topk_v2's 17.7 µs stands as the cost floor a candidate-B
kernel should approach (it does strictly more work: deterministic tie
admission + sequence-ordered emission).

Note on the AOT vehicle: sgl-kernel on this box is a prebuilt wheel with no
build tree; adding an AOT kernel means a from-scratch source build. Candidate
B is therefore built as a Triton JIT prototype for the head-to-head (DEC-4
explicitly permits Triton prototypes for benchmarking); AOT promotion is
recorded as a follow-on item per the close-out's notes.

Probe artifacts: this document's numbers are reproducible via the commands in
the session log; the determinism probe and component bench are inline scripts
over `sgl_kernel.top_k.fast_topk_v2` (no repo source modified — no DSA
regression trigger).
