# M1 Spike — DS score-reduce custom-AR feasibility (evidence + recommendation)

Date: 2026-06-10. Bench: `m1_spike_allreduce_bench.py` (torchrun, 8×H200, sglang
parallel_state, median of 50 iters, all-rank barrier per region). Raw report:
`runs/20260610_m0/m1_spike.json`. Production facts from the served Case-1 boot log
(`runs/20260610_m0/selcap_serve.log`).

## Premise correction (vs the plan's Feasibility Hints)

The plan assumed the reduce tensor is ~[29, 4608] fp32 ≈ 534 KB. Measured reality: the
graph-safe DS reduce operates on `scratch_scores[:bs, :max_seq_len]` with
`max_seq_len = req_to_token.shape[1] = context_len = 202752` → **[29, 202752] fp32 ≈ 23.5 MB
per call**, once per layer per decode step (78 × 10 = 780 calls per profiled window). The
microbench reproduces the frozen per-call cost: 167 µs ring vs 160 µs in the frozen profile
(124,873 µs / 780).

## Spike questions → answers

1. **Group aliasing** — YES: live `get_attn_tp_group() is get_tp_group()` under plain TP=8
   (parallel_state.py:1906-1907). The DS bind site's `.device_group` is the TP coordinator's
   raw group; the coordinator itself is custom-AR-capable.
2. **Which custom-AR** — `CustomAllReduceV2` active (disabled=False), `max_size` 16 MB
   (pull), one/two-shot thresholds 160 KB at TP=8 H200.
3. **Eligibility at the real shape** — fp32 [29, 202752] = 23.52 MB: `should_custom_ar`
   **False** (> 16 MB cap) → the coordinator falls through to NCCL ring (170.5 µs ≈ raw ring
   167.1 µs). bf16 [29, 202752] = 11.76 MB: **True** → v2 two-shot pull.
4. **Coordinator capture context** — a `torch.cuda.graph` capture of
   `tp_group.all_reduce` inside `graph_capture()` works in the standalone 8-rank harness:
   replay correct after `copy_` input mutation, **zero replay allocations**, and the replay
   kernel is the NAMED custom-AR kernel
   `all_reduce_two_shot_kernel<float, 8u, true>` (the AC-1.1 named-kernel evidence form).
5. **Measured cost curve** (median µs/call, eager, 8 ranks):

| shape | MB | should_custom_ar | NCCL ring | coordinator |
|---|---|---|---|---|
| [29, 202752] fp32 (production) | 23.5 | no | **167.1** | 170.5 (→ring) |
| [29, 202752] bf16 | 11.8 | yes | 105.9 | **104.1** (v2 two-shot) |
| [64, 202752] fp32 | 51.9 | no | 292.0 | 298.4 |
| [1, 202752] fp32 | 0.8 | yes | 40.9 | 47.4 |
| [29, 65536] fp32 | 7.6 | yes | 87.3 | 76.9 |
| [29, 16384] fp32 | 1.9 | yes | 50.8 | 45.7 |
| [29, 4608] fp32 (plan's assumed shape) | 0.53 | yes | 37.4 | 45.9 |
| [29, 4608] bf16 | 0.27 | yes | 38.5 | 51.9 |

   (The 32 MB-pull wide-cap v2 probe could not init in the harness — v2 asserts a non-NCCL
   group; moot, since at bandwidth-bound sizes v2 ≈ ring, so fp32-23.5 MB-via-v2 would still
   cost ~ring.)

## Reading

- **Custom-AR per se is NOT a perf lever here.** At every measured size the coordinator
  dispatch ties or slightly loses to raw NCCL ring; both are bandwidth-bound at MBs. The
  ring line's cost is BYTES, and the bytes are dominated by the dead width (202752 static
  width vs ≤4608 live tokens at the Case-1 op point: 167 → ~37-50 µs/call if width were
  live-sized — but the graph-static shape pins the width to context_len; live-width
  reduction is a selection-path redesign question, not a reduce-dispatch one).
- **The viable in-scope lever is dtype: a bf16 score reduce.** −61 µs/call (167.1 → 105.9)
  ≈ **−48k µs / 10-step window** on the +124,873 µs bucket. At 11.76 MB the coordinator
  selects custom-AR v2 (two-shot pull), so the `AllReduce_Sum_f32_RING` line is ELIMINATED
  and replaced by the named v2 kernel — satisfying AC-1.1's positive form literally — while
  the honest attribution is the halved bytes (custom-AR ≈ NCCL at equal bytes; ledger must
  say so). bf16 reduce is explicitly allowed by the plan ("bf16/lower-precision ... gated by
  the AC-2 recall bound") and DEC-3's bounded recall-delta.
- Implementation shape (task5): cleanest is bf16 score storage for the reduce — either the
  Triton score kernel writes bf16 scores natively (no cast kernels; top-k consumes bf16) or
  fp32 scores are cast → bf16-reduce → cast back (≈ +10-15 µs/call D2D, still ≈ −45 µs net).
  Native-bf16 changes tie granularity at the score level (coarser mantissa ⇒ more ties);
  the deterministic tie-break + AC-2 gates (cross-rank hard, recall ≤0.5pp, fixtures) decide.
  Out-of-place coordinator result needs the captured copy-back into `scratch_scores` (or
  feed the returned tensor into the topk pipeline directly within the captured region).
- bs-64 future note: [64, 202752] bf16 = 25.9 MB > 16 MB → would fall back to NCCL-bf16
  ring (still the dtype win). Case-1 is bs 29; the M4 audit owns any bs change.

## Recommendation: GO (bf16-reduce-through-coordinator), with the honest caveat

GO for M1 Phase B implemented as: bind the GroupCoordinator at the DS bind site (replacing
the raw `.device_group`), one reducible abstraction over both call sites, score reduce in
bf16 via `coordinator.all_reduce` (v2 two-shot pull at bs 29), captured copy-back, AC-2
gates. Expected bucket: 124,873 → ~82k µs (window), ring line eliminated, named custom-AR
kernel present, backend ledger entry `custom_ar_v2`.

Caveat for the ledger/review: the speedup attribution is the bf16 byte halving, not the AR
implementation swap; an fp32 custom-AR at production width is ineligible (and would not be
faster). The residual ~82k µs is the dead-width tax (202752 static vs ≤4608 live) — a
structural remainder this loop can only address via a selection-path redesign (M5 wildcard
/ follow-on), not via reduce dispatch.
