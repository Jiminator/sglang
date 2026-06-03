# Round 24 Contract

## Mainline Objective (exactly one)
**Empirically determine the graph-safe blocked-top-k design that actually achieves conc-16 ≥30 at full
context — microbench the candidate block-widths against the monolithic baseline — and drive the kernel
implementation from that evidence (not a guessed design).** Codex R23 prescribed `block_width=512,
partial_k=512`; but with `partial_k == block_width` each block emits ALL its candidates, so under CUDA-graph
FIXED shapes the Stage-2 merge runs over `num_blocks × partial_k == max_seq_len` (the full 163840 width) —
which would NOT reduce the residual `torch.topk` over-scan (the actual conc-16 TPS lever). This round
measures, on GPU (61 layers, bs=16, seq=4096): (1) monolithic topk over 163840 (current), (2) the
skip-ideal cost (topk over only the live region — the theoretical win if dead blocks past `seq_len` are
skipped), (3) blocked `bw=8192/partial_k=2048` (within-block top-2048 + merge over 40960, no context cap),
so the kernel is built against the design that wins. The exact `blocked_topk_sequence_order` (R23) is the
correctness oracle for whichever design is chosen.

## Target AC(s)
- **AC-5 (task6)** — the done-criterion (owner-approved np64 methodology; owner-chosen full-context kernel
  path). `coding` (GPU microbench + analysis; the Triton kernel implementation follows the winning design).

## Truly Blocking This Objective
- **Choosing a graph-safe blocked-top-k design that actually wins.** Building a research-grade Triton kernel
  against a design that doesn't reduce the merge over-scan (bw=512/partial_k=512) would burn rounds for no
  conc-16 gain. The microbench is the prerequisite that picks the winning design (skip-kernel and/or
  within-block top-k at bw>2048) or proves the win needs a context cap (which the owner declined → escalate).

## Queued / Explicitly Out of Scope This Round
- **The post-kernel AC-5 rerun + closed-batch proof** — after the kernel lands on the winning design.
- **AC-10** — gated. **Cross-node smoke** — future-gated. **DSA conc-64 TPS ~29.4** — queued. conc-32/64
  ≥30 remains structurally unattainable (DS ≤ DSA) regardless of the kernel.

## Concrete Success Criteria
1. A GPU microbench (`runs/.../ac5_topk_design/`) timing, at 61 layers / bs=16 / seq=4096 / max_seq_len=163840:
   monolithic topk-163840; skip-ideal (topk over the live region only); blocked bw=8192/partial_k=2048
   (within-block top-2048 + merge over 40960). Report per-step ms for each + the implied conc-16 step/TPS, so
   the winning design (and whether it needs a context cap) is evidence-based.
2. A short analysis artifact stating which design achieves conc-16 ≥30 at full context (no context cap), and
   the concrete kernel spec to implement (block_width, partial_k, skip semantics, merge width).
3. If the evidence shows the only no-context-cap win requires a genuinely research-grade within-block top-k
   kernel (multi-round) for a borderline conc-16 ~30 while conc-32/64 stay unattainable, surface that to the
   owner with the hard numbers (the R22 kernel choice was made without this perf evidence) — as an informed
   escalation, NOT in place of the microbench work.
4. GPUs freed; commit + push to `jimmy`; goal-tracker + round-24-summary + BitLesson Delta updated.

## Applicable BitLessons (confirm per-task via bitlesson-selector)
- `BL-20260531-ds-selection-fullwidth-overscan` (the topk over-scan is capture-width-bound; the design space).
- `BL-20260531-topk-deterministic-tiebreak` (the oracle/tie contract the chosen design must honor).
- `BL-20260530-durable-tracked-acceptance-evidence` (the microbench is durable/recomputable evidence).
- `BL-20260527-torch-topk-aliasing-corrupts-input` (no aliasing in the timed topk/argsort).

## Out-of-bounds reminders
No ABI-lock / FlashMLA-assert / `top_k` change. No context cap as the AC-5 pass (owner declined). No
plan-process tokens in code/comments. Do not change the DS-fair AC-12 gate. Must not exit by lying / editing
loop state / cancel.
