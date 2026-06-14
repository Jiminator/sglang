# Loop 11 Results — Authoritative Current State

> Maintained rewrite-over-append: this document always reflects the loop's current state.
> Last regenerated: Round 17, 2026-06-14. HEAD at round start: `4b0be6936` (R16).

## 1. Current state summary

- **M0 COMPLETE (Rounds 0–4); M1 (task3+task4) COMPLETE+VERIFIED (R9). M2 task5 COMPLETE+VERIFIED (R16); task6 IN PROGRESS (R17 slice 1).**
- **R17 — task6 slice 1: table-free latent selector ABI (config-gated, default OFF).** New `table_free`
  config: when on, the selector binds the absorbed state (no `TokenLabelTable`), `retrieve_topk_graph_safe`
  returns the production top-k from the absorbed-latent score (the R16-validated scorer), pool-sizing
  skips the table reservation, and validation rejects cosine/hybrid (absorbed identity is `scorer_norm
  ="off"` only). Default OFF ⇒ shipped DSA default + the DS table path are byte-identical (every gate
  reduces to the original; independently re-verified). +13 table_free unit tests (selection==table,
  validation, pool-sizing skip); full DS suite 442 pass. Also fixed the AC-8 results.md header. NEXT
  task6 slices: DELETE `token_label_table.py`/`token_label_write.py` (DEC-2), serving capacity payoff
  @0.8, same-round AC-7. (§10)
- **R16 — task5 served recall@2048 gate PASS on GLM-5.1 (Codex R15 gap + 2 helper fixes).** Added the
  comparability checks to `absorbed_recall_summary.py` (length-set + per-length sample-count equality —
  non-comparable populations now FAIL) and made `run_absorbed.sh` post-process + exit with the gate
  status. **`run_absorbed.sh full` at the matched GLM-5.1/fp16/mem-0.7 baseline op-point: PASS ±0.5pp**
  — absorbed recall@2048 overall 64.781% vs baseline 64.696% (+0.085pp); per-length Δ ≤0.352pp;
  18,720 records, 100% carry `rec["absorbed"]`, comparable population, no failures. The `recall_oracle`
  bind fires on GLM (`GlmMoeDsaForCausalLM` ⊂ `DeepseekV2AttentionMLA`). task5's last gate is closed.
  (§10)
- **R15 — task5 served recall wiring + live validation (Codex R14 gap) + AC-8 ledger.** Wired the
  absorbed scorer side-by-side into the GRAPH-SAFE serving recall-oracle path (gated by `recall_oracle`,
  additive, selection unchanged); first attempt targeted the eager path — re-targeted after a live
  boot+sweep proved the oracle rides graph-safe; fixed a bind device bug. **Live (DeepSeek-V3.2, eager,
  recall_oracle): 244/244 records carry `payload["absorbed"]`; absorbed↔table agreement 99.88%,
  recall@2048 Δ vs table ~0.02pp** → the fp8-latent scorer reproduces the table's selection on real
  NIAH traffic. The absorbed-vs-FROZEN-baseline numbers are CONFOUNDED (ran DSv3.2+int8; baseline is
  GLM-5.1+fp16 — table sits equally low), so the matched-config run (`run_absorbed.sh full`, GLM/fp16)
  is R16. 443 DS+oracle tests pass. (§10)
- **R14 — task5 GPU kernel budget optimization (Codex R13 gap 1 + wrapper bug).** Rewrote the kernel
  to block-scale reassociation with a `tl.dot` tensor-core MMA (no `[TOKEN_BLOCK,512]` spill); fixed
  the wrapper default (token_block 64→128, exposed; harness uses the public path); tuned `num_warps=2,
  num_stages=2` (Triton's default 4 was ~2× too many). **Captured-replay budget 64.0k → 15.1k
  µs/window — PASS** (under 20k target + 23.1k landed logical-score; 4.2× faster). tf32 parity ~1e-3,
  selection exact (oracle recall=1.0); NEW production-byte test (`quantize_k_cache_separate`). Owner
  ruled: keep tf32, no fp8-`v_h`. 5 GPU tests + full DS suite (424) pass. NIAH recall@2048 = R15 slice
  (server-required; reclassification rejected by Codex). (§10)
- **R13 — task5 GPU paged kernel + budget + validity (Codex R12 gap 1) + AC-8 ledger repair (gap 2).**
  New `absorbed_latent_kernel.py`: a Triton paged fp8-latent score kernel mirroring `_logical_score_kernel`,
  reading the resident `[512 fp8 | 4 fp32 scales]` latent with in-register dequant. CPU-vs-GPU equivalence
  vs the R12 reference (the oracle): finite-score parity (~7.6e-6), identical `-inf` mask, **oracle recall
  = 1.0**, unwritten hole masked, pool-byte-layout consumption (`TestAbsorbedLatentPagedGPU`, 4 CUDA tests).
  **Captured-replay budget measured: best 82 µs/call ≈ 64.0k µs/window — OVER_BUDGET (~2.8× the 23.1k
  landed / 20k target), an HONEST measured miss** (plan's flagged H×512-vs-H×32 compute risk; keeps task5
  active, no move to task6). Written-slot validity invariant documented (§10). Ledger metadata/status
  corrected (header / task4 VERIFIED review R9). Served NIAH recall@2048 is the remaining **task5**
  gate (R13 review REJECTED folding it into task6) — wired + run in R15. full DS suite + 4 GPU tests
  pass (§10).
- **R12 — task5 CPU proof hardened to contract (Codex R11 repairs).** `build_absorbed_projection` now
  returns the bind-time **selected** rows `[H, label_dim, kv_lora_rank]` and the score helpers consume
  them. Logical equivalence strengthened from top-k to **EXACT score-tensor parity**
  (`torch.testing.assert_close`) vs the LIVE `_compute_logical_token_scores`, with an **in-range
  unwritten hole** proven masked to `-inf` and excluded. The fp8 oracle now reads the **real pool byte
  layout** `[512 fp8 | 4 fp32 scales]` (not a hand-rolled dequant). 7 absorbed tests + full DS suite
  pass (§10).
- **R11 — task5 logical-domain + real-`kv_b_proj` + fp8 oracle.** Added `build_absorbed_projection`
  (bind-time W_UK from the real block-fp8 `kv_b_proj`) and `absorbed_latent_score_logical` (paged
  over `req_to_token` with seq-len/unwritten masking, mirroring the production logical scorer).
  Equivalence now proven in the **production logical domain**: absorbed top-k EXACTLY matches the
  LIVE `retrieve_topk_via_labels` logical mode on a multi-request fixture (non-contiguous slots,
  holes, unequal seq_lens); plus an fp8-latent overlap oracle (≥0.9). Remaining: GPU Triton kernel,
  serving recall@2048, written-slot validity (§10).
- **R10 — M2 task5 absorbed-latent score-only prototype (in progress).** Proved the load-bearing
  identity (signature = `channel_select(W_UK·c_kv)`, weights query-side) and landed a CPU reference
  `absorbed_latent.py` (`absorbed_latent_score`) computing `score = max_h v_h·c_kv` directly from the
  latent — no table. Equivalence GATE passes: exact fp32 match vs the hand-derived label score AND
  ≥0.99 top-k overlap vs the **live** `retrieve_topk_via_labels` (scorer_norm="off"); 3 unit tests
  (§10). This proves the TokenLabelTable can be eliminated exactly. Remaining for task5: the GPU
  paged score kernel (over `req_to_token`, in-kernel fp8 dequant), the serving NIAH recall@2048 gate
  (fp8-latent value-affecting), and the written-slot validity analysis.
- task0/task1/task2 done with durable evidence. **task3 (indexer-cache gate) COMPLETE R6** (§8);
  **task4 (int8 served config + table-aware pool sizing)** landed R7, R7/R8-review gaps closed R8/R9
  (§9) — cap lifted bs30→**bs74 ≥ 64** at mem 0.8; decode batch reaches 63. **Owner ratified
  (R9) the admission-based AC-1.1 conc-64 gate** (decode `#running-req` peak ≥ 61 AND DS agg ≥ DSA at
  matched config — DS meets both); the Little's-law "achieved concurrency" field is a descriptor, not
  the gate.
- **R9 — task4 verification repair.** Put the DS admission proof in the TRACKED evidence
  (`r8_durable_int8_evidence.txt`: `max_decode_running_req=63`, `ac1_1_conc64_gate=PASS`); folded the
  `written` bitmap into `TokenLabelTable.bytes_per_rank()`/`estimate_hbm_bytes()` + the boot log
  (now `6.82 GB/rank`, full table) + a unit test; fixed the Little's-law arithmetic. Owner ratified
  the AC-1.1 gate. M1 complete pending R9 verification.
- **R8 — task4 R7-review repair.** (1) the DURABLE launcher `serve_double_sparsity.sh` now defaults
  to the served config (int8 / mem 0.8 / right-sized envelope, explicit flags); stale fp16/mem0.6/
  Loop-7 comments removed. (2) AC-1.1 conc-64 **admission isolated**: DS decode `#running-req` reaches
  **63** (≈ nominal 64; the bs30 cap is gone), and at conc-64 radix-OFF DS **outperforms** the
  apples-to-apples DSA radix-OFF on every real metric (777.7 vs 665.6 tok/s, 320 vs 256 completed,
  31.5 vs 49.2 s mean latency, 10.6 vs 24.9 s mean TTFT). The benchmark "achieved concurrency" field
  (DS 47.91 vs DSA 63.98) is `request_throughput × mean_e2e_latency` (Little's law, verified exactly)
  — it reads LOWER for the FASTER DS and is **not** an admission signal. (3) AC-7 completed across
  the required surfaces (Case-2 dsa07 = 142,208 unchanged; DSA @0.8 radix-ON = 410,560; DSA @0.8
  radix-off = 410,560). (4) the `written` bitmap is now folded into the reservation. Evidence:
  `runs/20260613_m0/{stage_r8_task4_repair.sh, r8_*_evidence.txt}`.
- **R7 — task4 int8 + table-aware pool sizing (M1 closer).** The DS pool sizing now reserves the
  per-token `TokenLabelTable` bytes deliberately in `_compute_cell_size` (DS-only), so the signature
  table is no longer carved from accidental post-capture headroom (the fp16-0.8 instability cause).
  **AC-1.1:** DS int8 @0.8 right-sized = `max_total_num_tokens` **342,784 → bs_cap 74 ≥ 64**, graph
  capture OK, coherent smoke; int8 `token_label_table` 6.77 GB/rank reserved; **22.32 GB available
  post-capture** (sustainable). Reserved per-token bytes match the allocation exactly
  (L78·H8·(D32+2)=21,216 B/tok = 6.77 GB/342,848 tok). **AC-5 (int8 quality):** existing int8-vs-fp16
  selection overlap gate ≥ 0.99 passes. **AC-7:** DSA-native @0.8 = 410,560 unchanged (sizing is
  DS-only). Directional ladder (short-window, **radix-OFF**) conc 16/32/64 = decTPS p50
  34.9/29.9/26.0, p99 TTFT 3.1/6.2/26.3 s — the conc-64 tail is the radix-OFF full-prefill cost
  (prefix reuse = task7); steady-state conc-64 + TTFT/throughput verdict is the task9 AC-11 sweep.
  Evidence: `runs/20260613_m0/r7_*_evidence.txt`, `stage_r7_task4.sh`.
- **R6 — task3 state-path closure (HiCache/hierarchical-radix sidecar gap).** R5's in-class gate was
  incomplete: `HiRadixCache`/`UnifiedRadixCache` build a DSA indexer **host** sidecar
  (`DSAIndexerPoolHost`) for any `DSATokenToKVPool`, and its `init_kv_buffer` iterates
  `device_pool.index_k_with_scale_buffer` — `None` under the gate → crash at host-pool construction.
  Closed two ways: (1) `validate_double_sparsity` rejects `--enable-double-sparsity` +
  `--enable-hierarchical-cache` with a clear `ValueError` (HiCache is not a served config; DS
  cached-prefix label semantics are task7-gated); (2) a defensive guard in `DSAIndexerPoolHost`
  raises clearly when built against a gated pool. **AC-7:** DS+HiCache → clean validator error, NO
  `NoneType` crash; DSA+HiCache boots, the indexer host sidecar still builds (8.45 GB/rank), capacity
  410,560 unchanged, coherent smoke (guard skipped for the non-gated DSA pool). 4 new unit tests.
  Evidence: `runs/20260613_m0/r6_*_evidence.txt`.
- **R5 — task3 indexer-cache gate (production code, DS-only, default byte-compatible).** The DSA
  indexer index-k sidecar (~10.3 KB/token) is gated off under DS: `DSATokenToKVPool` skips the
  allocation, the configurator drops the matching cell-size term, the prefill indexer-store is
  skipped, and data accessors fail loudly if any DS path touches index-k. **AC-1.1 capacity
  payoff:** DS @0.7 gated = 174,848 tokens vs ungated 142,208 (+23%, matching task0's indexer-off
  probe), graph capture OK, coherent smoke. **AC-7:** DSA @0.8 = 410,560 unchanged + radix-ON DSA
  coherent — DSA-native byte-untouched (gate is DS-only). The fail-loud guard caught a real prefill
  index-k store the static audit missed; the gate now skips that dead store under DS too. Details
  + the ds@0.8 graph_capture_oom (expected fp16/default-envelope ceiling — task4 closes it):
  `runs/20260613_m0/task3_indexer_gate_validation.md`.
- **R4 closed the last open bounded ceiling** (Codex R3): bounded `tf/on/rs` passed the R3 grid-top
  0.95, so R4 probed 0.96 → graph_capture_oom — first-fail captured, ceiling closed. All six
  bounded configs now have a real highest-pass + first-fail (§5.1).
- **R3 completed the bounded right-sized matrix** Codex R2 flagged as a subset: all six
  `{fp16,int8,tf} × {indexer on/off}` rs configs swept `fail_closed [4608]`, with a canonical
  bounded-ceiling table next to the unbounded control grid (§5.1). Finding: bounded reclaims ~0.3
  GB per config (subsumed by the envelope) and lifts one ceiling (tf/on/rs 0.90→0.95).
- **R2 landed the loop's first production code:** the bounded selector-width feature
  (`selector_width_overflow_policy`), DS-gated, default byte-compatible; matrix/extracts carry
  `graph_capture` + `smoke` + first-fail `note`. **AC-7 verified** (§7).
- **Production code this round (DS-gated, default byte-compatible):** `selector_width_overflow_policy`
  in the DS config (`full_fallback` default = today's `{compact, full}` ladder; opt-in
  `fail_closed` captures only compact widths and raises a clear error on overflow). Files:
  `double_sparsity/config.py`, `model_executor/cuda_graph_runner.py` (pure helpers
  `compute_ds_selector_widths` / `ds_covering_width`), `dsa_backend.py`. Unit tests in
  `test_double_sparsity_unit.py` (TestDSSelectorWidthLadder + config cases). **AC-7 verified**
  (§7): DSA-native @0.8 unchanged + DS-default @0.7 reproduces the frozen anchor exactly.
- The table-free / indexer-off MOCKS remain dev-only probe hooks
  (`runs/20260613_m0/probe_hacks.patch`), reverted before every commit — the *designed* indexer
  gate is task3/M1. The bounded-width feature is NOT a mock; it is committed.
- **Scope notes (Plan Evolution R1+R2):** (a) the boot-probe matrix measures the
  **boot/capture/smoke ceiling** per config — an *upper bound* on the servable fraction, not the
  sustained-stable served fraction (task4/M2 ladders confirm under real load); (b) envelope axis
  = `{default, right-sized=(--max-running-requests 64 --cuda-graph-max-bs 64)}`; `rs16k`
  (`--context-length 16384`) is a labeled supplementary set; (c) bounded selector-width is now
  measured (§5.1) — at the right-sized envelope it reclaims only ~0.3 GB (its headroom value is
  largely subsumed by the envelope's bs64 cap; its real value is the fail-closed served-width
  contract).

## 2. FROZEN: radix-ON DSA @0.8 directional baseline (task1) — the loop's comparison column

Recipe = `development/profiling/runs/20260612/` stage2 with exactly one change: radix cache ON
(production default). 1 trial, 60 s warmup / 180 s window, NUM_PROMPTS=64, gsp 4096-ISL/512-OSL
~55% prefix, seeds {16:213, 32:431, 64:31234}, server seed 20260607, HEAD `6714a5663`.
Artifacts: `runs/20260613_m0/serving/` (SUMMARY.txt + per-conc meta sidecars), boot fields in
`runs/20260613_m0/dsa08_radixon_boot_fields.txt`. **FROZEN — never re-run.**

| conc | decTPS_p50 | agg tok/s | ach conc | TTFT mean/med/p99 (s) | TPOT p99 (ms) | done |
|---:|---:|---:|---:|---|---:|---:|
| 16 | 38.95 | 597.6 | 16.00 | 0.59 / 0.56 / 0.87 | 26.22 | 256 |
| 32 | 32.40 | 981.5 | 31.99 | 0.81 / 0.84 / 1.04 | 32.11 | 384 |
| 64 | 25.23 | 1530.1 | 63.96 | 1.06 / 1.12 / 1.27 | 40.78 | 576 |

What the radix-ON bar means for this loop (vs the radix-off 20260612 reference: agg 404/541/676,
p99 TTFT 7.2/14.2/28.1 s):

- The ~55% prefix reuse turns DSA's TTFT tail into ~1 s and >2×'s its aggregate. DS radix-on
  enablement (task7) is load-bearing for AC-2/AC-3, not a chore: a radix-off DS cannot approach
  agg ≥ 0.95× × 1530 ≈ 1454 tok/s at conc 64.
- **Measured AC-3 tension (flagged for owner ruling, logged in the goal tracker):** the frozen
  baseline itself has per-request decode-TPS p50 = 25.23 < 30 at conc 64 — batched decode at
  bs≈64 costs per-request speed (the known admission-restore tradeoff). AC-3's unconditional
  "DS p50 ≥ 30 at conc 16/32/64" clause and "DS aggregate ≥ 0.95× DSA at conc 64" cannot both
  hold if DS reaches DSA-like batches, unless the ≥30 clause is read like AC-2's absolute bar
  ("wherever DSA meets it" — DSA does not at conc 64). No reinterpretation applied; raised, not
  decided.

## 3. task2: DS-Offload — REJECTED (memo final)

`development/loop11/ds_offload_rejection_memo.md` (analyze-routed via Codex; mechanism claims
verified against `development/past_implementations/DoubleSparse/offloading/model.py`).
Measured PCIe (this node, pinned 1 GiB, `runs/20260613_m0/pcie_bandwidth.txt`): **H2D 55.5 /
D2H 55.1 GB/s**. Gather = 2048 × 78 × 576 B ≈ 92 MB/req/step → bs30 ≈ 2.76 GB/step ≈ 50 ms,
bs64 ≈ 5.89 GB/step ≈ 107 ms vs the ~33 ms/step budget — optimistic bound (sequential BW, zero
gather overhead/contention). The table itself cannot be offloaded: dense read every step ⇒
~159 GB/s/rank sustained. Revisit only as a genuinely new design (context-horizon hybrid, new
lossiness discussion).

## 4. task0: componentized per-rank memory accounting (frozen 20260612 boot logs, HEAD-consistent)

Per-rank budget at DS@0.7 (frozen 20260612 case1, TP0; all GB as logged). Source `serve.log`s are
gitignored (repo policy); the cited lines are durably tracked in
`runs/20260613_m0/frozen_20260612_boot_fields.txt`:

| stage | avail after | delta | component |
|---|---:|---:|---|
| pre-weights | 138.01 | — | H200 140.5 GB total, ~2.5 driver/torch reserve |
| weights loaded | 48.76 | 89.24 | GLM-5.1-FP8 weights + dist state |
| memory pool end | 38.99 | 9.77 | KV pool **8.14** (142,208 tokens, kv_buffer + DSA indexer sidecar) + ReqToTokenPool **~1.66** (2049 × 202752 × 4 B — default envelope `max_running_requests=2048`, `context_len=202752`) |
| DS bind done (capture begin) | 31.31 | 7.68 | TokenLabelTable **5.29** + DS bind state **~2.39** (channel mask, selector buffers) |
| capture end | 26.63 | 4.68 | DS CUDA graphs **4.68** (DSA-only: 2.73 → DS graph delta +1.95) |
| server ready | 26.12 | 0.51 | misc/allocator |

DSA@0.8 (case3): pool end 23.60 (KV 23.51 GB, 410,560 tokens) → capture 2.73 → ready 18.50 GB
free. Measured @0.8 headroom (§5 matrix, supersedes the pre-probe priors): table-free DS = 16.26
GB (p11); int8 DS = 8.17 GB (p07, table 8.11 GB — the int8 table at the 0.8 pool, not the
draft's 2.97 GB which was the 0.7/142k pool); fp16 DS = 1.25 GB (p03, not sustainable).

**Two clean derived constants (close the cell-size question):**
- **Indexer sidecar = 18.6–18.7% of the per-token KV cell** — measured two ways at fixed
  fraction (p01→p14 @0.7: 142,208→174,848 tokens; p09→p10 @0.8: 410,560→504,640), i.e. **+23%
  tokens** when the sidecar is gated. (Plan estimate was ~17%.) This is the task3 justification
  number, and it confirms the cell-size/configurator accounting converts freed sidecar bytes
  into admitted tokens — gating the buffer alone, without the accounting update, would free
  nothing.
- **TokenLabelTable = 39.0 KB/token (fp16), 20.7 KB/token (int8 = 0.531×)**, sized from the
  physical KV-slot count (`pool.size + page`). **The table is NOT a fixed 5.29 GB — it scales
  with the pool you are trying to grow.** 5.29 GB is only the 0.7/142k-token pool; at the 0.8
  operating point the fp16 table would be **15.27 GB**. So absorbed-latent elimination (task6)
  frees 15.27 GB at the 0.8 op-point, not 5.29 — a materially stronger payoff than the draft's
  fixed-5.29-GB framing implied.

## 5. task0: capacity matrix + boot-ceiling sweep — COMPLETE (full 12-config grid)

The full `{fp16, int8, table-free mock} × {indexer on/off} × {default, right-sized}` cross-product,
each swept in mem_fraction (0.75→0.95) to its **boot/capture/smoke ceiling** (highest pass +
first fail). Drivers: `runs/20260613_m0/stage_task0_probes.sh` (R0 anchor rows) +
`stage_task0_fill.sh` (R1 sweep). Unified rows: `task0_matrix.tsv` (51 probes); per-config
summary: `task0_ceilings.md`; **durable per-probe evidence**: `probe_logs/<name>_evidence.txt`
(server args + KV alloc + table bytes + capture begin/end + `max_total_num_tokens` +
`available_gpu_mem` + smoke — the R0 `head -50` truncation is fixed; the gitignored `.log`s are
no longer the only proof). Probe-only env hooks (`SGLANG_DS_PROBE_TABLE_TOKENS=8192` → ~0.30 GB
mock table, kept visible; `SGLANG_DS_PROBE_SKIP_INDEXER=1` sidecar-gate preview) per
`probe_hacks.patch` — **reverted before commit; zero production code changed.**

`ready_GB` = `available_gpu_mem` at server-ready (post weights + KV pool + table + DS graph
capture). **It — not the token readout — is the real discriminator**, but at the ceiling it is
boot-only headroom (see caveat). Boot ceiling per config (table-free `*` = 0.30 GB mock, true
table-free frees 0.30 more):

| variant | idx | env | highest PASS (frac / bs_cap / ready GB) | first FAIL | bs≥64 cleared at |
|---|---|---|---|---|---|
| fp16 | on  | default | 0.80 / bs89  / **1.25** | 0.85 | 0.80 |
| fp16 | on  | rs      | 0.80 / bs89  / 6.71 | 0.85 | 0.80 |
| fp16 | off | default | 0.75 / bs73  / 11.21 | 0.80 | 0.75 |
| fp16 | off | rs      | 0.80 / bs109 / 2.75 | 0.85 | 0.75 |
| int8 | on  | default | 0.80 / bs89  / 8.17 | 0.85 | 0.80 |
| int8 | on  | rs      | 0.85 / bs118 / 3.38 | 0.90 | 0.80 |
| int8 | off | default | 0.80 / bs109 / 5.84 | 0.85 | 0.75 |
| int8 | off | rs      | 0.85 / bs145 / 1.05 | 0.90 | 0.75 |
| tf*  | on  | default | 0.90 / bs147 / 0.87 | 0.95 | 0.80 |
| tf*  | on  | rs      | 0.90 / bs147 / 6.32 | 0.95 | 0.80 |
| tf*  | off | default | 0.85 / bs145 / 7.87 | 0.90 | 0.75 |
| tf*  | off | rs      | 0.90 / bs181 / 5.41 | 0.95 | 0.75 |

DSA reference (frozen case3): @0.8 = 410,560 tokens / bs89 / **18.50 GB** ready.
Supplementary (separate axis, not in the grid): the R0 `rs16k` rows (`--context-length 16384`) in
`task0_matrix.tsv` — context-length sensitivity, not bounded selector width.

**What the matrix establishes:**

1. **Token capacity / bs_cap is a function of (mem_fraction, indexer-gate) ONLY — table dtype does
   not move it.** At any fraction, fp16/int8/table-free read identical tokens; only the
   indexer-gate shifts it (+23%). The pool is sized from `available_bytes // cell_size` *before*
   the table is taken from leftover, so the table dtype changes the *leftover* (headroom), not the
   pool. **Every config clears AC-1.1's bs≥64 floor** — indexer-off configs already at 0.75
   (bs73), indexer-on at 0.80 (bs89). AC-1.2's ≥390k-tokens @0.8 readout is likewise mechanical.
   The binding question is sustainability (headroom), not the token count.
2. **Headroom is what the table / indexer-gate / envelope levers buy, and it pins the draft's "DS
   stuck at 0.7" exactly:** at the same 0.80/bs89, fp16/on/def has **1.25 GB** ready (boots but
   gen-OOMs under load — the established finding); int8 lifts it to 8.17; table-free to 16.26; the
   right-sized envelope adds ~5–6 GB more. The boot *ceiling* climbs with the levers: fp16 0.80 →
   int8 0.80–0.85 → table-free 0.90.
3. **task3 indexer-gate works end to end** (+23% tokens at fixed fraction, two independent
   measurements) — but a subtlety the sweep exposed: gating the indexer at a *high* fraction can
   OOM (fp16/off/def fails at 0.80 where fp16/on/def boots), because the freed bytes feed a bigger
   pool *and* a pool-sized table. The gate's real benefit is **more tokens at a sustainable
   fraction** (every indexer-off config clears bs≥64 at 0.75) — which is exactly why task3 must
   land with task4's **table-aware sizing** (deduct the table before sizing the pool).
4. **Endgame — table-free + indexer-gate strictly dominates DSA's memory op-point:** tf/off/rs
   reaches **bs181 @0.90 / 5.41 GB ready**, and at 0.80 = bs109 / **21.25 GB ready** vs DSA@0.8's
   bs89 / 18.50 GB — more tokens *and* deeper headroom. The table stays 0.30 GB (mock) across the
   whole sweep: with the table gone, the pool grows unpenalized. This is the absorbed-latent
   (task6) + indexer-gate (task3) target, measured.

**Honesty caveat (binding on every PASS above).** These are **boot + graph-capture + a single
24-token smoke**, NOT sustained-load proofs — the boot ceiling is an *upper bound* on the
servable fraction. The only GLM-5.1 config with a *sustained* 4096-ISL anchor is fp16 @0.7 =
26.63 GB ready (the 20260612 served config); fp16's boot ceiling is 0.80 at 1.25 GB, so the
sustained-stable fraction sits a full step below the boot ceiling (it needs ~15–25 GB of decode-
activation headroom the smoke never exercises). The sustained served fraction per config is
established only on the milestone ladders — **task4** (int8) and the **M2 gate** (table-free).

**Sequencing read-out for M1 (DEC-3 = full M1 first):** the M1 served config is int8 + indexer-gate
+ right-sized envelope (= the int8/off/rs row): boot ceiling 0.85/bs145, with **0.80 = bs109 /
11.49 GB ready** as the comfortable rung (the 0.85 rung's 1.05 GB is boot-only). That ~11.5 GB at
3.5× the fp16@0.7 pool is the task4-ladder candidate; task4 confirms the sustained fraction.
Because indexer-off clears bs≥64 at 0.75 already, M1 has fraction headroom to trade for stability.
The endgame (table-free, M2) serves the DSA op-point (0.80/bs89) with 21 GB to spare.

## 5.1. task0: bounded selector-width — feature (R2) + FULL right-sized ceiling matrix (R3)

The bounded-selector-width axis the plan names in the task0 right-sized envelope is a real DS
feature (R2): `selector_width_overflow_policy` (`full_fallback` default = byte-compatible
`{compact, full}` ladder; `fail_closed` captures ONLY the compact buckets, no full 202752-width DS
scratch, and raises on overflow). R3 completed the **full bounded right-sized ceiling matrix** — all
six `{fp16,int8,tf} × {indexer on/off}` rs configs swept `fail_closed [4608]` to their boot
ceilings (highest-pass + first-fail). Drivers `stage_task0_bounded.sh` (R2) + `stage_task0_bounded_fill.sh`
(R3); canonical table in `task0_ceilings.md` (§"bounded right-sized ceilings"); comparison
`task0_bounded_compare.md`; durable per-probe evidence `probe_logs/bnd_*` + `ctl_*`.

**Bounded right-sized ceilings (`fail_closed [4608]`, rs envelope):**

| variant | idx | highest PASS (frac/bs/ready GB) | first FAIL | vs unbounded rs ceiling |
|---|---|---|---|---|
| fp16 | on  | 0.80 / bs89 / 7.02 | 0.85 | same ceiling, +0.31 GB |
| fp16 | off | 0.80 / bs109 / 3.06 | 0.85 | same ceiling, +0.31 GB |
| int8 | on  | 0.85 / bs118 / 3.72 | 0.90 | same ceiling, +0.34 GB |
| int8 | off | 0.85 / bs145 / 1.39 | 0.90 | same ceiling, +0.34 GB |
| tf | off | 0.90 / bs181 / 5.72 | 0.95 | same ceiling, +0.31 GB |
| tf | on | **0.95 / bs176 / 1.14** | 0.96 (graph_capture_oom) | **+1 step (unbounded 0.90→0.95)** |

All six bounded configs are closed with a measured highest-pass AND first-fail (R3 swept five;
R4 closed `tf/on/rs` — it passed the R3 grid-top 0.95, so R4 probed 0.96 → graph_capture_oom).

**Two findings:** (1) Bounded reclaims a uniform **~0.3 GB** vs the unbounded rs row at every
matched point — clean attribution from the matched control `ctl_int8_off_rs_080` (`full_fallback`
`{4608, full}` = 11.49 GB = unbounded exactly; bounded `{4608}` = 11.83 → the +0.34 GB is precisely
the dropped full-width DS scratch). (2) For **5 of 6** configs the ~0.3 GB does not cross a
fraction step, so the boot ceiling is unchanged; for **tf/on/rs** it tips a ragged 0.90→0.95 (bs147
→ bs176, 1.14 GB ready; 0.96 then fails). The small delta is because `cuda_graph_max_bs=64` already
shrinks the full-width score plane (`[64, 202752]` ≈ 80–240 MB, not GB; the measured ~1.95 GB DS
graph overhead was a bs512-default cost). **So bounded selector-width is largely subsumed by the
right-sized envelope as a headroom lever; its durable value is the fail-closed served-width
contract** (§7), plus an occasional one-step ceiling lift where headroom is on the edge.

task0 is now complete to the plan contract: the unbounded `{default, right-sized}` grid (§5) AND
the bounded right-sized ceilings (this section), each of the six bounded configs with a measured
highest-pass + first-fail, all backed by durable extracts with graph_capture/smoke/note.

## 6. Queue state

See `development/loop11/queue.md`: task0–task9 mainline (task0–2 + the q2 bounded-width feature
DONE), q1–q8 kickoff candidates (q2 now landed+measured; q7 parked on owner AC-6 ruling; q8
conditional), r1–r5 recorded rejections.

## 7. AC-7: shared-surface regression for the bounded-width feature (R2)

The feature touches the shared `cuda_graph_runner.py`, so the DSA-native default and the DS default
were re-validated on the feature-only tree (probe hacks reverted). Driver
`runs/20260613_m0/stage_r2_regression.sh`; evidence `r2_dsa_off_080_evidence.txt`,
`r2_ds_default_070_evidence.txt`.

- **AC-7a DSA-native @0.8 (DS off):** boots, captures (8 ranks), `max_total_num_tokens=410560`
  (matches frozen case3), coherent smoke → DSA-native byte-unchanged (the feature is DS-gated via
  `use_ds_selector_width_keys`, so DSA never reaches it).
- **AC-7b DS-on default (`full_fallback`) @0.7:** `max_total_num_tokens=142208` / bs30 / table
  5.29 GB / coherent smoke — **reproduces the frozen p01 anchor exactly**, proving the default
  policy is byte-compatible at runtime, not just in the unit test.
- **Fail-closed guard, end-to-end:** a 9002-token prompt on a `fail_closed` `[4608]` server raised
  the clear `RuntimeError: DS selector width fail-closed: live sequence length 9002 exceeds the
  largest captured selector width 4608 …` (`probe_logs/failclosed_response.txt`) — a too-long
  sequence is rejected, never silently routed to full-width or eager.

## 8. M1 task3: DS-mode indexer-cache gate — COMPLETE (R5 in-class gate + R6 state-path closure)

First M1 capacity lever, the designed replacement for the R0 `SGLANG_DS_PROBE_SKIP_INDEXER`
preview. Production code (DS-only, default byte-compatible):
- `DSATokenToKVPool(gate_index_k_cache=…)`: under DS, skips the index-k sidecar allocation
  (`index_k_with_scale_buffer = None`); the five data accessors fail loudly; clear/offload/state/
  size methods are None-safe.
- `pool_configurator._compute_cell_size`: drops the indexer term (132 B/token/layer) iff
  `enable_double_sparsity and not enable_hisparse` — in lockstep with the pool, so freed bytes
  become admitted tokens.
- `forward_mha.py`: skips the dead prefill `self.indexer(...)` index-k store under DS (caught by
  the fail-loud guard — the static audit had missed this store path).
- Wired in `model_runner_kv_cache_mixin` (non-hisparse DS path only). Unit tests:
  `TestDSIndexerCacheGate` (6).

**Evidence** (`runs/20260613_m0/task3_indexer_gate_validation.md`, `r5v2_*_evidence.txt`):
AC-1.1 DS @0.7 gated = 174,848 tokens (+23% vs ungated 142,208; matches task0), capture OK,
coherent smoke. AC-7 DSA @0.8 = 410,560 unchanged + radix-ON coherent. The gate is a contributing
lever; DS @0.8 fp16/default-envelope still graph_capture_oom's (task0 ceiling 0.75) — the
sustainable served 0.8 config is task4 (int8 + table-aware sizing + right-sized envelope, where
task0 measured int8/off/rs = bs109 / 11.5 GB ready).

**R6 — state-path closure (the "offload/disagg/radix state-path audit" task3 owes).** R5's in-class
gate did not cover the hierarchical-cache host sidecar: `HiRadixCache`/`UnifiedRadixCache` build a
DSA indexer host pool (`DSAIndexerPoolHost`) for any `DSATokenToKVPool`, and its `init_kv_buffer`
iterates `device_pool.index_k_with_scale_buffer` — `None` under the gate → crash before any
fail-loud accessor fires. Closed fail-closed (HiCache is not a served config; DS radix/cached-prefix
label semantics are task7-gated, not proven):
- `double_sparsity/validator.py::validate_double_sparsity`: rejects `--enable-double-sparsity` +
  `--enable-hierarchical-cache` with a clear `ValueError` (DS-only early-return → cannot affect DSA).
- `DSAIndexerPoolHost.__init__`: defensive guard raises a clear `RuntimeError` when the device pool
  is gated, before `init_kv_buffer` — defense-in-depth so programmatic construction can't crash.
- Unit coverage (4 new): validator rejects DS+HiCache and allows DSA+HiCache; the host guard fires
  for a gated pool and is skipped for a non-gated (DSA) pool.

**R6 AC-7** (`runs/20260613_m0/r6_*_evidence.txt`, `stage_r6_task3_hicache.sh`): DS+HiCache launch →
clean fail-closed `ValueError`, no `NoneType` crash. DSA+HiCache boot → indexer host sidecar still
builds (8.45 GB/rank, `layer_first`), `max_total_num_tokens=410,560` unchanged, capture OK, coherent
smoke ("Paris…") — the guard is correctly skipped for the non-gated DSA pool. The served DSA default
(no HiCache) never constructs `DSAIndexerPoolHost` and is untouched by this round's diff.

## 9. M1 task4: int8 served config + table-aware pool sizing — VERIFIED COMPLETE (R7 land + R8/R9 repair; owner-ratified AC-1.1 gate; review R9 verified)

The root-cause fix for the fp16-0.8 instability: the per-token `TokenLabelTable` footprint is now
**reserved deliberately** in the DS pool-sizing equation instead of being carved from accidental
post-capture headroom. Production code (DS-only, default byte-compatible):
- `pool_configurator._compute_cell_size`: adds, for `enable_double_sparsity and not enable_hisparse`,
  a per-token term mirroring `allocate_token_label_table`'s per-slot footprint —
  `num_hidden_layers · num_local_heads · (label_dim·2 fp16 | label_dim+2 int8)` signatures + scales,
  **plus the `written` bitmap (`num_hidden_layers` bytes/token, R8)** = the FULL per-token table.
  Dimensions read from the validator-set `_double_sparsity_parsed_config.signature_dtype` +
  `_double_sparsity_channel_mask.label_dim` + `model_config.{num_hidden_layers,
  get_num_attention_heads(tp)}`. `max_total_num_tokens = available_bytes // cell_size` then reserves
  the table proportionally. DSA-native + HiSparse get no term (byte-unchanged). +4 unit tests.
- **Durable served config (R8):** `development/serve_double_sparsity.sh` now DEFAULTS to the served
  config — `SIGNATURE_DTYPE=int8`, `MEM_FRACTION_STATIC=0.8`, `MAX_RUNNING_REQUESTS=64`,
  `CUDA_GRAPH_MAX_BS=64` (explicit flags); stale fp16/mem0.6/Loop-7 comments removed. Env overrides
  preserve fp16 diagnostics. radix-OFF (radix-on = task7).

**AC-1.1 capacity** (R9 durable-launcher re-run, `r8_durable_int8_evidence.txt`): DS int8 @0.8 rs →
`max_total_num_tokens` **344,064 → bs_cap 74 ≥ 64**, graph capture OK, coherent smoke; int8
`token_label_table` **6.82 GB/rank** (R9 `bytes_per_rank` now reports the FULL table — signatures +
int8 scales + `written` bitmap), **22.29 GB available post-capture** (sustainable). The cell-size
reservation includes `written` (R8) and the helper/log now agree (R9).

**AC-1.1 conc-64 admission — isolated, tracked, OWNER-RATIFIED (R9).** The bs30 cap is gone: the
tracked evidence now records `max_decode_running_req=63` and `ac1_1_conc64_gate=PASS`. At conc-64
radix-OFF, DS vs the apples-to-apples DSA radix-OFF (`r8_dsa08_radixoff_conc64_evidence.txt`):
| conc-64 radix-OFF | DS int8 | DSA radix-off |
|---|---|---|
| completed | **320** | 256 |
| output tok/s | **777.7** | 665.6 |
| mean e2e latency | **31.5 s** | 49.2 s |
| mean TTFT | **10.6 s** | 24.9 s |
| decode `#running-req` peak | 63 | 64 |
| "achieved concurrency" (Little's law) | 47.91 | 63.98 |

DS **outperforms** DSA radix-off on every real metric. The benchmark "achieved concurrency" field is
`request_throughput × mean_e2e_latency` (Little's law — verified to the decimal: DS 1.519 × 31.54 s =
47.91; DSA 1.300 × 49.21 s = 63.98), so it reads LOWER for the FASTER DS and is **not** an admission
signal. **Owner ruling (R9):** AC-1.1 conc-64 is gated on decode `#running-req` peak ≥ 61 AND DS agg ≥
DSA at the matched config (DS meets both: 63; 777.7 ≥ 665.6) — the Little's-law concurrency field is a
descriptor, not the gate. Admission/capacity is met.

**AC-5 int8 quality:** int8-vs-fp16 top-2048 selection overlap ≥ 0.99 (unit gate passes).

**AC-7 (full, R8, new artifacts — frozen refs not mutated):** Case-2 DSA-native @0.7 (dsa07) =
142,208 unchanged; DSA @0.8 radix-ON = 410,560 (radix confirmed on) coherent; DSA @0.8 radix-off =
410,560 coherent. The DS-only sizing change perturbs no DSA surface.

M1 is VERIFIED COMPLETE (R9 review). M2 task5 STARTED (§10).

## 10. M2 task5: absorbed-latent score-only prototype — VERIFIED COMPLETE (R10–R16; review R16)

The structural fix (DEC-2): replace the materialized signature table with scores read directly from
the resident fp8 KV latent. **Load-bearing identity (proven, code-cited):** today's
`signature = channel_select(W_UK · c_kv)` — the per-head K_nope (`k_nope[h] = W_UK[h]·c_kv`, the
`kv_b_proj`/`w_kc` up-projection) sliced to the offline-mask channels, with channel weights `w_c`
applied on the query side (`dsa_backend.py:1696-1733` + `token_label_write.py:80` +
`selection_kernel.project_query_onto_channels`). Substituting collapses the score to
`max_h (v_h · c_kv)` with `v_h = Σ_{c∈S_h} w_c·q_c·W_UK[h][c,:]` — the table is unnecessary.

**Landed R10–R11 (`double_sparsity/absorbed_latent.py`, fp32, `scorer_norm="off"`):**
- R10: `absorbed_latent_v` / `absorbed_latent_score` + `TestAbsorbedLatentScore` (3): exact fp32
  match vs the hand-derived label score; ≥0.99 top-k overlap vs the LIVE `retrieve_topk_via_labels`;
  head_agg mean parity.
- R11: `build_absorbed_projection` (bind-time W_UK from the REAL `kv_b_proj` — `block_quant_dequant`
  or `.float()`, reshape, slice K-noPE rows) + `absorbed_latent_score_logical` (LOGICAL-domain paged
  reference mirroring `_compute_logical_token_scores`: req_to_token gather, unwritten-then-seq_len
  masking). First logical-domain equivalence (top-k overlap) + a first fp8-latent overlap oracle.
- R12 (hardened to contract, Codex R11 repairs 1–5): `build_absorbed_projection` now takes
  `channel_selection` and returns the bind-time **selected** rows `[H, label_dim, kv_lora_rank]`
  (dequant → reshape → K-noPE slice → gather mask channels); `absorbed_latent_v`/`_score`/
  `_score_logical` consume the pre-gathered rows (query side still gathers `channel_selection`).
  `TestAbsorbedLatentLogical` now proves **EXACT logical score parity** (the score TENSOR via
  `torch.testing.assert_close`, not only top-k) against the LIVE `_compute_logical_token_scores`,
  plus top-k equivalence vs `retrieve_topk_via_labels`; an **in-range UNWRITTEN hole** (logical pos
  mapped through `req_to_token` to a slot with a deliberately 10× latent norm) is asserted masked to
  `-inf` and excluded from top-k. The fp8 oracle now packs/reads the **real pool byte layout**
  `[512 fp8 bytes | 4 fp32 per-128-block scales]` (`packed[:, :512].view(float8_e4m3fn)`,
  `packed[:, 512:528].view(float32)`), dequants block-by-block, and records ≥0.9 top-k overlap vs
  the full-precision latent (value-affecting CPU oracle). 7 absorbed-latent tests + full DS suite pass.

**Landed R13 (`double_sparsity/absorbed_latent_kernel.py`, GPU; Codex R12 gap 1):**
- **GPU paged Triton kernel** `_absorbed_score_kernel` + wrappers `absorbed_score_paged_fp8` /
  `absorbed_latent_score_logical_paged`. Mirrors the production `_logical_score_kernel`
  persistent-worker topology (static `(bs, WORKERS)` grid, per-worker live-block stride, loop bound =
  device-computed live block count, written-then-`seq_len` masking in production order). It reads the
  **resident fp8 latent** `[max_tokens, 512]` + per-128-block fp32 scales `[max_tokens, 4]`, dequants
  **in-register per element** (`fp8.to(f32) · scale_b`), and reduces `max_h Σ_l v_h[b,h,l]·latent_deq`
  — `v_h` precomputed host-side via `absorbed_latent_v`. `quantize_latent_fp8`/`dequantize_latent_fp8`
  helpers reproduce the pool's per-128-block fp8 scheme.
- **CPU-vs-GPU equivalence** (`TestAbsorbedLatentPagedGPU`, 4, CUDA-guarded): the GPU scores match
  `absorbed_latent_score_logical` fed the dequantized latent value-for-value — finite-score
  `torch.testing.assert_close` (max abs diff measured ~7.6e-6, pure fp32 summation reassociation),
  identical `-inf` mask, **oracle recall = 1.0** (top-k selection overlap), the in-range UNWRITTEN
  hole masked to `-inf` and excluded, a pool-byte-layout (`[512 fp8 | 4 fp32 scales]` pack→view→feed)
  consumption test, and head_agg mean parity. The CPU reference is the kernel's oracle (BL-20260614).
- **Budget (captured-replay, the binding number — BL-20260602/BL-20260611):**
  `runs/20260614_m2/absorbed_kernel_budget.{py,json}` at the logical-score op point (bs=29, H=8,
  width=5120, seq=4608, lora=512), CUDA-graph replay median × 780 calls/window. **Best tb=32:
  82.0 µs/call ≈ 64.0k µs/window — OVER_BUDGET vs the ~23.1k landed logical-score bucket / 20k
  target (~2.8×).** This is the plan's flagged measured risk realized: the absorbed read is H×512
  MACs/token vs the label kernel's H×32 (16×), done as per-head vector dot-reductions, not a
  tensor-core matmul; tb≥64 spills the `[TOKEN_BLOCK,512]` tile (1.4–2.0 ms/call). The miss keeps
  task5 active and does NOT advance to task6 (per the contract / Codex R12). Optimization path (plan
  task5 design space, next slice): reformulate the per-block reduction as a `tl.dot` tensor-core
  matmul `v[H,512] @ latent_deqᵀ[512,TB] → [H,TB]` then head-max; block-scale reassociation (apply the
  4 scales to per-block partial dots instead of dequantizing 512 elements/token); optional fp8 `v_h`
  for fp8 tensor-core dots (additionally value-affecting, gate separately).

**Written-slot validity invariant (R13 analysis, Codex R12 gap-1 item 4):** the table path masks
unwritten label slots via the `written` bitmap; the latent path must score **exactly the slots the KV
cache has written**. The invariant the kernel enforces, mirroring `_compute_logical_token_scores`: a
logical position `pos` of request `r` is scored iff (a) `pos < seq_lens[r]` AND (b)
`written[ req_to_token[req_pool_indices[r], pos] ]` — else `-inf`, in that order. Ordering rationale
on the served path: during decode the scheduler allocates `out_cache_loc` and writes the KV/latent
for the new token BEFORE the next step's selection reads it, so at selection time every `pos <
seq_lens[r]` maps through `req_to_token` to a physical slot whose latent is already resident — the
`written` bit is the belt-and-suspenders guard for the transient where a slot index exists in
`req_to_token` but the latent write has not landed (and for radix-reuse / partial-hit slots that are
allocated but not yet written this sequence). Unlike the table path, the latent has no separate
write-hook to fall out of sync: the latent IS what attention reads, so "scored == written" reduces to
"scored == KV-resident", which is structurally true under the decode allocate→write→select order. The
fp8 quantization is applied at KV-write time (`set_mla_kv_buffer`), so a written slot always carries
valid fp8+scales. (The serving-time confirmation of this ordering on real traffic is a task6 item,
once the latent scorer is in the loop.)

**Landed R14 (kernel budget optimization — Codex R13 gap 1 + blocking issue 1):** the kernel inner
loop was rewritten to **block-scale reassociation with a `tl.dot` tensor-core MMA** — per 128-wide
latent block, `partial[tok,h] = tl.dot(fp8_latent_blk, v_blk)` (tf32), weighted by that token's fp32
block scale and fp32-accumulated per head, then head-max — so it **never materializes the
`[TOKEN_BLOCK,512]` dequant tile** that spilled the R13 vector loop. The wrapper-default bug is fixed
(`token_block` 64→128 default, exposed through `absorbed_latent_score_logical_paged`, harness calls
the PUBLIC wrapper). A `num_warps`/`num_stages` sweep found Triton's default `num_warps=4`
over-subscribed this shape ~2×; **`num_warps=2, num_stages=2, token_block=128`** is the measured best.
**Captured-replay budget: 19.4 µs/call ≈ 15.1k µs/window — PASS (under the 20k target AND the ~23.1k
landed logical-score bucket)**, down from R13's 64.0k (4.2× faster). Equivalence preserved: the tf32
MMA gives ~1e-3 score parity (vs R13's 7.6e-6 exact) but **selection stays exact (oracle recall =
1.0)**; a NEW `test_paged_gpu_production_quantizer_bytes` feeds the kernel the real
`quantize_k_cache_separate` `[512 fp8 | 4 fp32 scales]` bytes (not the helper — Codex showed they are
not byte-identical) and confirms parity + selection. 5 GPU tests; full DS suite 424 pass. Owner ruled
(R14): keep the less-lossy tf32 path, tune without fp8 `v_h` — the fp8-`v_h` tensor-core variant stays
documented but unused (additionally value-affecting). Budget slice CLOSED at budget.

**Landed R15 (served NIAH recall@2048 — side-by-side absorbed diagnostic wiring + live validation):**
the absorbed scorer is wired SIDE-BY-SIDE into the **graph-safe** serving path
(`retrieve_topk_graph_safe` → `_maybe_record_recall_oracle`), gated by `recall_oracle`, additive
(shipped/DS-non-oracle path byte-identical), selection-output unchanged (table still drives decode).
The first R15 attempt wired the EAGER `retrieve_topk_via_labels`, but a live boot+sweep proved the
oracle rides the graph-safe path even under `--disable-cuda-graph` — re-targeted (and a real bind
device bug fixed: `build_absorbed_projection` gathered a CPU index against a CUDA weight). Bind builds
`absorbed_w_sel` per layer (`channel_selection[self.layer_id]`); the per-step graph-safe call reads
this layer's resident fp8 nope latent + per-128-block scales off the MLA pool and scores
`absorbed_latent_score_logical_paged` → all-reduce → top-k, emitting `payload["absorbed"]`.
**Live validation (DeepSeek-V3.2, TP=8, fp8, eager, recall_oracle): 244/244 records carried
`payload["absorbed"]`; absorbed↔table selection agreement 99.88% (14,623/14,640), recall@2048 Δ vs the
same run's table ~0.02pp** — the fp8-latent value-affecting change reproduces the table's selection on
real NIAH traffic (AC-5 value-affecting gate evidence). `+TestSideBySideAbsorbedOracleRecord`; 443 DS
+ oracle tests pass. `runs/20260614_m2/recall_absorbed/` (FINDINGS.md, absorbed_recall_summary.json),
`run_absorbed.sh`.

**CONFOUND (honest):** that validation run used `serve_double_sparsity.sh` (DeepSeek-V3.2 + int8 + mem
0.8), but the frozen baseline (64.696%) is the **GLM-5.1 + fp16 + mem 0.7** op-point
(`profiling/runs/20260609/_env.sh`). Both table (55.7%) AND absorbed (55.7%) sit ~13pp below the
baseline at long contexts — the gap is the model+dtype mismatch (int8 vs fp16 signature precision),
NOT the absorbed scorer (99.88% table agreement). The absorbed-vs-frozen-baseline numbers are
therefore NOT a valid AC-5 verdict.

**Landed R16 (matched-config served NIAH recall@2048 gate — PASS; Codex R15 gap + 2 helper fixes):**
the absorbed-vs-frozen-baseline gate now runs on the BASELINE op-point. The gate helper
`absorbed_recall_summary.py` gained the comparability checks it was missing (length-set equality +
per-length sample-count equality vs the baseline, mirroring `oracle_recall_summary.py` — a
non-comparable population now FAILs instead of silently passing); `run_absorbed.sh` now post-processes
the sink through the gate and exits with the gate status (canonical `absorbed_recall_summary.json` on
`full`). **`run_absorbed.sh full` on GLM-5.1-FP8 / fp16 / mem 0.7 (the baseline op-point): PASS ±0.5pp.**
Smoke: GLM-5.1 boots, 624/624 records carry `rec["absorbed"]` (the `recall_oracle` bind fires on
`GlmMoeDsaForCausalLM` via its `DeepseekV2AttentionMLA` inheritance), 0 errors. Full (18,720 records,
6,240/length, 100% carry `rec["absorbed"]`, comparable population, `problems[]` empty):

| length | absorbed recall@2048 | baseline | Δ (pp) |
|--------|----------------------|----------|--------|
| 1024   | 100.000% | 100.0  | +0.000 |
| 4096   | 58.397%  | 58.045 | +0.352 |
| 16384  | 35.946%  | 36.042 | −0.096 |
| overall| 64.781%  | 64.696 | +0.085 |

The fp8-latent absorbed scorer reproduces the frozen DSA/label recall baseline on the served GLM-5.1
op-point (overall +0.085pp, ≤0.352pp per length) — the declared value-affecting change is within bar.
`runs/20260614_m2/recall_absorbed/{absorbed_recall_summary.json,FINDINGS.md}`. Config comment fixed
(recall_oracle disables graph CAPTURE but the selector still runs the graph-safe path eagerly).

**task5 COMPLETE (pending R16 verification):** CPU equivalence (R10–12) + GPU paged kernel & budget PASS
(R13–14) + written-slot validity (R13) + served recall@2048 PASS (R15 wiring + R16 GLM gate). **task6
now UNBLOCKED:** swap the selector ABI to the latent binding, graph scratch for `v_h`, reject
cosine/hybrid, DELETE `token_label_table.py`/`token_label_write.py` (DEC-2), capacity payoff @0.8,
same-round AC-7.

## 11. M2 task6: table-free latent-selector integration — IN PROGRESS (R17 slice 1)

The structural payoff (DEC-2): make DS selection score the resident fp8 latent directly so the
`TokenLabelTable` (5.29 GB/rank fp16) is never allocated → the KV pool grows → the bs cap lifts. task5
proved the absorbed scorer is correct (CPU+GPU equivalence, served recall@2048 PASS); task6 integrates
it as the production selection path and removes the table.

**Landed R17 (slice 1 — the table-free ABI, config-gated `table_free`, default OFF):**
- `config.py`: `table_free: bool = False` (config-borne → reaches TP workers). `validator.py`:
  `table_free` requires `scorer_norm="off"` (the absorbed identity holds only there — cosine/hybrid
  rejected with a clear error).
- `selector.py`: `bind_runtime_data` accepts a table-free bind (`_bind_table_free`) — binds the
  channel mask + the pre-installed `absorbed_w_sel`, NO table; `retrieve_topk` routes to the absorbed
  selection when `table_free`.
- `selection_kernel.py`: `absorbed_topk_select` (score the resident latent → reduce → mask → top-k);
  `retrieve_topk_graph_safe` gains a `table_free` early branch that writes the RETURNED
  `out_indices`/`out_lengths` from the absorbed score and skips all table scoring/labels-gather. Score
  buffer shape `[max_bs, max_seq_len]` unchanged; rope dims excluded by construction (no-PE queries +
  K-noPE W_UK rows).
- `deepseek_v2.py`: builds `selector.absorbed_w_sel` at bind when `table_free OR recall_oracle`; skips
  the `TokenLabelTable` allocation under `table_free`; per-step reads the resident fp8 latent +
  per-128-block scales and threads `table_free` into the selector call. The non-None-table precondition
  accepts the table-free bind.
- `pool_configurator.py`: `_compute_cell_size` omits the `TokenLabelTable` reservation under
  `table_free` (the freed bytes become tokens — the capacity payoff, measured in a serving slice).

**Default-OFF byte-identicality (the AC-7 safety property, independently re-verified):** every changed
site reduces to the original when `table_free=False` (and `recall_oracle=False`) — `table = None if
table_free else <original>`; the alloc/precondition guards become `not table_free`; the
`retrieve_topk_graph_safe` table-free block is a guarded early return; pool cell-size reserves the
table exactly as before. The 427 pre-existing tests (table path) pass unchanged + 13 new `table_free`
tests (selection == table path; validation rejects cosine/hybrid; pool-sizing omits the table term
under table_free and is unchanged off). Full DS suite 442 pass. No shared-default behavior changed →
no AC-7 boot this slice.

**Remaining for task6** (next slices): DELETE `token_label_table.py` + `token_label_write.py` +
the prefill label-write hook (DEC-2) once `table_free` is the validated default; the **serving capacity
payoff @ mem 0.8** (boot readout: table-free pool grows → bs cap ≥64, AC-1.2); CUDA-graph capture-safe
`v_h` scratch for the captured table-free path; **same-round AC-7 serving regression** (DS-off smoke +
Case-2 + radix-ON DSA) when table_free changes the served default. Then task7 (radix-on), task8 (bs64
tax), task9 (locked AC-11 sweep).
