# Round 17 Summary — AC-5 decode-throughput remediation (DS selection over-scan fix)

## Mainline objective (round contract)
Codex R16-review Required-Plan steps 2-3: AC-5 strict remediation as a **decode-throughput-first**
problem — profile the DS conc-16 decode hot path at the lifted DS int8 / mem-0.7 / radix-on point,
then make the smallest decode-path code change to move conc-16 per-req TPS toward ≥ 30, preserving
the ABI lock (`indices.shape[-1] == dsa_index_topk == 2048`). AC-10 and more AC-7/AC-8 evidence were
out of scope.

## What landed (commit `ece26eb52`)
1. **Profiled + localized the bottleneck (Codex step 3).** A CLOSED-batch pure-decode measurement (N
   parallel `/generate`, `ignore_eos`, no new arrivals → clean decode batch, `#queue-req:0`) gave
   **17.4 TPS/req at batch 16** (step 57.6 ms) — ≈ the AC-5 cold-flood 17.6, so conc-16 is **genuinely
   decode-bound, NOT a WARMUP=0 artifact**. A selection-width microbench showed the graph-safe DS
   selection scores + top-k over `max_seq_len = req_to_token.shape[1] = context_len = 163840` every
   layer (×61) every step — a **~35× over-scan** for a ~4096-token request (~32 ms of the 57.6 ms step
   is selection; ~23.5 ms is pure over-scan; the score kernel did the per-head loads/dots for the whole
   context and only masked the result).
2. **The fix — a numerically-identical, CUDA-graph-safe score-kernel early-exit.** `_logical_score_kernel`
   now skips token-blocks entirely past each request's `seq_len` (store -inf + return before the per-head
   loop). **No flag** (bit-identical output), **ABI lock untouched**, **AC-8 context preserved** (each
   program still scans its own seq, no context cap). Verified: selection **identical at width 4608 vs
   163840** (layers 0/7/30/60); selection @163840 **32.08 → 12.50 ms/step**; **281 DS unit tests pass**.
3. **End-to-end re-measure (patched, same operating point).** Closed-batch pure decode:
   conc-1 39.7→40.9, conc-8 24.6→**32.6 (now ≥30)**, conc-16 **17.4→27.1 TPS/req (+56%)**; step
   57.6→36.9 ms (−20.7 ms == the profiled over-scan savings). Coherence smoke unchanged.

## Result
The AC-5 decode bottleneck is **localized and the dominant component fixed**: conc-16 pure-decode
**17.4 → 27.1 TPS/req (+56%)**, conc-8 now passes ≥ 30 — from a bit-identical, graph-safe, no-flag,
AC-8-preserving kernel change. conc-16 strict (≥30) is **not yet reached** (step 36.9 vs the 33.3 ms
target, ~3.6 ms over): the residual is the first `torch.topk` over-scan (runs over the full captured
163840-wide score row). Shrinking it is capture-width-bound (needs a seq-aware blocked/partial top-k or
bucketed width, **not** a context cap) — the next round's lever. **AC-5 strict SLO stays a live mainline
blocker (DEC-3).**

## Files Changed
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — score-kernel early-exit
  (the only production code change; ~16 lines, numerically identical).
- `runs/20260530_dsv32_loop6/ac5_decode_profile/` (NEW): `ac5_decode_remediation.md` (the profile +
  fix + before/after), `closed_batch_decode.py` (pure-decode profiler), `ds_closed_batch_decode.txt` /
  `ds_closed_batch_decode_patched.txt` (before/after curves), `selection_width_microbench.py`+`.json`
  (over-scan attribution), `verify_early_exit.py` (bit-identical equivalence + timing),
  `get_server_info_ds{,_patched}.json` (operating-point sidecars), `closed_batch_b{1,8,16}.json`.
- `.humanize/bitlesson.md` — new lesson `BL-20260531-ds-selection-fullwidth-overscan`; goal-tracker
  (R17 Plan Evolution row + AC-5/task6 note); round-17 contract/summary (gitignored loop state).

## Validation
- `verify_early_exit.py`: selection bit-identical at width 4608 vs 163840 (layers 0/7/30/60); selection
  @163840 32.08 → 12.50 ms/step.
- `pytest test/.../test_double_sparsity_unit.py`: **281 passed**.
- Closed-batch end-to-end (patched, same sidecar: mem 0.7 / int8 / radix-on / max_total 396096):
  conc-16 27.1 TPS/req, conc-8 32.6, conc-1 40.9; coherence "The capital of France is" → " Paris. The
  capital of the United States" (no degeneration). `git diff --check` clean; commit `ece26eb52` pushed
  to `jimmy`. GPUs freed at round end (all 8 at 0 MiB, no live `launch_server`).

## Remaining Items
- **Open mainline blocker (AC-5 strict):** conc-16 per-req TPS 27.1 < 30 — residual ~3.6 ms `torch.topk`
  over-scan (capture-width-bound). Next round: a seq-aware blocked/partial top-k (the
  `DSGraphState.scratch_partial_*` buffers exist) or bucketed selection width — without a context cap;
  then conc-32/64 TTFT tuning (the over-scan fix should help proportionally more there); then the full
  AC-5 client re-run (NUM_PROMPTS=320, conc 16/32/64) with exact arrays + a fail-closed verifier.
- **Gated AC-10** — only after AC-5 strict is verified. **Cross-node wrapper smoke** — future-gated.
  **DSA-default conc-64 TPS ~29.4** — queued pre-existing limit. No ABI-lock / FlashMLA-assert changes;
  DS-fair AC-12 gate unchanged.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260531-ds-selection-fullwidth-overscan
Notes: New lesson — the DS graph-safe decode selection over-scanned the full KV-index buffer width
(`ds_graph_state.max_seq_len = req_to_token.shape[1] = context_len = 163840`) every layer every step,
scoring the entire context for a ~4096-token request (~32 of the 57.6 ms decode step). Captures the
profile-first method (CLOSED-batch pure decode to isolate decode from prefill-interleave + a
scan-width microbench to attribute), the numerically-identical CUDA-graph-safe early-exit fix (skip
token-blocks past seq_len; no flag, no context cap, ABI lock intact), and the residual topk over-scan
caveat (capture-width-bound). Applied existing lessons: BL-20260530-cold-flood-not-steady-state-slo
(the closed-batch rejected the cold-flood hypothesis), BL-20260530-admission-restore-tps-tradeoff
(per-req TPS = 1/decode_step_time), BL-20260527-torch-topk-aliasing-corrupts-input (kept the topk
out=/scratch contract intact), BL-20260530-durable-tracked-acceptance-evidence (tracked profiler +
microbench + bit-identical verifier), BL-20260530-remote-server-launch (background boot + ps-grep +
pkill||true; no foreground sleep).
