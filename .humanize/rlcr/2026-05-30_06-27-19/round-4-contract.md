# Round 4 Contract

## Mainline Objective (exactly one)
**AC-4 — mem-fraction lift + no-OOM validation with the compact int8 table.** Boot DS with `SIGNATURE_DTYPE=int8`, sweep `MEM_FRACTION_STATIC=0.6 → 0.7 → 0.8`, record `max_total_num_tokens` rising and the full HBM budget (NVML + torch reserved/allocated residual, not only named tensors) at each point, and survive a sustained long `/generate` at the lifted fraction with **no generation-time OOM and no monotonic memory growth**. This is the spine's payoff: the footprint reduction exists to let DS boot at a higher mem fraction without the generation OOM fp16 hit at 0.7.

## Target ACs (1–2)
- **AC-4** (`coding`, hardware-run) — primary.

## Blocking Side Issues in Scope
None known. (The launcher `SIGNATURE_DTYPE` surface — last round's blocker — is fixed and verified; AC-4 uses it.)

## Queued / Out of Scope
- **AC-5 client-SLO benchmark** (NUM_PROMPTS=320, conc 16/32/64, strict `<22.0` TTFT + attribution) — the *next* round, gated on AC-4. I will not run the full client-SLO benchmark this round.
- AC-6 hardware proof, AC-7/AC-8/AC-9, gated AC-10 — later. Do not touch the FlashMLA `indices.shape[-1]==dsa_index_topk` assert (AC-3.3).
- Per AC-2's binding decision: int8 is the compaction lever; the fp16 lower-`f` window is at most an optional comparison datapoint, **not** a substitute for the int8 compact-table sweep.

## Round Success Criteria
1. A mem-fraction sweep artifact (`runs/20260530_dsv32_loop6/memfraction_sweep_int8.md`) records, for `f ∈ {0.6, 0.7, 0.8}` with the int8 table: `max_total_num_tokens` (must **rise** with `f`), the `token_label_table` GB/rank (int8), KV-pool size, post-pool + post-cuda-graph headroom, and an **NVML (nvidia-smi) + torch (avail/reserved/allocated) HBM snapshot** — the full budget, not just named tensors.
2. At the lifted fraction (target 0.8; 0.7 acceptable as the conservative step if 0.8 OOMs): a **sustained long `/generate`** (long-context prompt + a decode burst) completes with **no generation-time OOM**, and an NVML time series over the run shows **no monotonic growth**. `/get_server_info` (or `/get_model_info` + log) recorded.
3. If 0.8 OOMs during generation (the AC-2 negative outcome), that is recorded honestly with the verbatim OOM + the achieved fraction, and the result is the highest no-OOM fraction (the AC-2 ladder is re-examined) — a genuine miss is a recorded finding, not a hidden failure.
4. Servers killed cleanly (stale-`sglang::router` + explicit-PID kill); commit + push to `jimmy`; `round-4-summary.md` with BitLesson Delta; tracker updated (task5/AC-4).

## Out-of-Scope Guards
- fp16 stays the launcher default. Reuse `serve_double_sparsity.sh` (no new serve scaffolding). No FlashMLA decode-assert changes.
- This round validates admission *capacity* (max_total rising, no OOM). The TTFT/SLO claim + admission-vs-prefill attribution is AC-5 (next round) — I will not conflate AC-4's no-OOM with the SLO.
