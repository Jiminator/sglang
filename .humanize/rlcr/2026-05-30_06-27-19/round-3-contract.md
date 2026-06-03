# Round 3 Contract

## Mainline Objective (exactly one)
**Run the real-mask NIAH non-regression on TP=8 hardware** — boot DS with the compact int8 table on the Loop-5 mask, measure needle recall via `test_double_sparsity_v32.py`, and prove int8 does **not** regress DS recall vs the fp16 Loop-5 DS baseline. This is the one outstanding AC-3.1 evidence item and it gates AC-4 (task4 → task5).

Correction to the prior round's premise: the RLCR box is **node 0 with 8× H200** (the R1 `nvidia-smi | head -2` undercounted), so V3.2 TP=8 serving is feasible here; the "2-GPU dev box" deferral was wrong. Codex is right to reject it.

## Target ACs (1–2)
- **AC-3** — the real-mask NIAH non-regression completes AC-3.1's recall gate (`coding`, hardware-run).

## Blocking Side Issues in Scope (Codex R2 review)
**`serve_double_sparsity.sh` cannot select the compact table.** It builds `DS_CONFIG` from `top_k/page_size/channel_mask_path/device_buffer_size` only — never `signature_dtype` — and the config default is fp16. So `MEM_FRACTION_STATIC=… bash serve_double_sparsity.sh` silently boots **fp16** DS, which would invalidate the NIAH (and the later AC-4 mem-sweep). Fix first:
- add `SIGNATURE_DTYPE="${SIGNATURE_DTYPE:-fp16}"`, include `"signature_dtype": "${SIGNATURE_DTYPE}"` in `DS_CONFIG`, echo it in the launch log;
- add a static test that `SIGNATURE_DTYPE=int8` yields a config with `"signature_dtype": "int8"` and the default stays `"fp16"`.

## Queued / Out of Scope
- **AC-4 mem-fraction sweep + no-OOM long generate** — the *next* round, gated on this NIAH passing (task5 depends on task4). I will not start the 0.6→0.8 sweep this round.
- AC-5/AC-7/AC-8/AC-9 and gated AC-10 — later. Do not touch the FlashMLA `indices.shape[-1]==dsa_index_topk` assert (AC-3.3).

## Round Success Criteria
1. `serve_double_sparsity.sh` exposes `SIGNATURE_DTYPE` (config + log); static test green; default fp16 preserved.
2. DS boots TP=8 with `signature_dtype=int8`, mem 0.6, Loop-5 mask; the server log / config proves `signature_dtype=int8` (and the `token_label_table:` line shows `dtype=int8`).
3. The NIAH harness runs (DS=node 0, DSA=node 1 cross-node) at the within-budget + 4K/16K lengths; `ac12_niah_*.json` copied into `runs/20260530_dsv32_loop6/real_mask_niah_int8/`.
4. `runs/20260530_dsv32_loop6/real_mask_niah_nonregression.md` records int8 DS recall vs the fp16 Loop-5 DS baseline (`runs/20260528_dsv32_mvp/ac12_results/`: 1024=100, 1536=100, 4K=75, 16K=5, 64K=0). **PASS iff** int8 ds_recall ≥ fp16 ds_recall at every comparable length **and** no new DS unservable error where the fp16 baseline served.
5. Servers killed cleanly (`pkill -f 'sglang::router'` gotcha respected). Commit + push to `jimmy`; `round-3-summary.md` with BitLesson Delta; tracker reconciled (real-mask NIAH out of Explicitly Deferred).

## Out-of-Scope Guards
- fp16 stays the launcher default. No new scaffolding (reuse the NIAH harness + serve scripts). No FlashMLA decode-assert changes.
- A genuine recall regression (int8 < fp16 at some length) is a real finding to record honestly, not to hide — but the expectation (per the synthetic top-k overlap@2048 ≥ 0.99 and the dense=100% property) is non-regression.
