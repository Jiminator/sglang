# Loop 11 Results — Authoritative Current State

> Maintained rewrite-over-append: this document always reflects the loop's current state.
> Last regenerated: Round 2, 2026-06-13. HEAD at round start: `aae4a3f90`.

## 1. Current state summary

- **M0 COMPLETE (Rounds 0–2).** task0 (memory accounting + the full 12-config capacity matrix +
  the bounded-selector-width axis), task1 (frozen radix-ON DSA @0.8 baseline), task2 (DS-Offload
  rejection memo) all done with durable, tracked evidence. No AC *verdicts* — M0 is ground truth.
- **R2 closed the R1 review gaps + landed the loop's first production code.** The bounded
  selector-width axis Codex required for task0 is now a real, committed DS feature
  (`selector_width_overflow_policy`), measured, with the matrix distinguishing bounded from
  unbounded right-sized (§5.1). The canonical matrix/extracts now carry `graph_capture` + `smoke`
  + first-fail `note` (the R1 AC-8 evidence gap).
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

## 5.1. task0: bounded selector-width (R2 feature + measurement)

The bounded-selector-width axis the plan names in the task0 right-sized envelope is a real DS
feature, landed this round: `selector_width_overflow_policy` (`full_fallback` default =
byte-compatible `{compact, full}` ladder; `fail_closed` captures ONLY the compact buckets, no
full 202752-width DS scratch, and raises on overflow). Driver `runs/20260613_m0/stage_task0_bounded.sh`;
comparison `task0_bounded_compare.md`; per-probe evidence `probe_logs/bnd_*` + `ctl_*`.

Bounded (`fail_closed`, `selector_width_buckets=[4608]`) vs the R1 unbounded right-sized row at the
same point — `ready_GB` delta = the reclaimed full-width DS graph scratch:

| config @frac | unbounded ready GB | bounded ready GB | delta | bs_cap |
|---|---:|---:|---:|---:|
| fp16/on/rs @0.80 | 6.71 | 7.02 | +0.31 | 89 |
| int8/off/rs @0.80 | 11.49 | 11.83 | +0.34 | 109 |
| int8/off/rs @0.85 | 1.05 | 1.39 | +0.34 | 145 |
| tf/off/rs @0.80 | 21.25 | 21.56 | +0.31 | 109 |
| tf/off/rs @0.85 | 13.33 | 13.64 | +0.31 | 145 |
| tf/off/rs @0.90 | 5.41 | 5.72 | +0.31 | 181 |

Clean attribution: the matched control `ctl_int8_off_rs_080` (`full_fallback`, buckets=[4608] →
`{4608, full}`) reads **11.49 GB = the unbounded row exactly**; bounded (`{4608}`) reads 11.83 →
the **+0.34 GB is precisely the dropped full-width DS scratch**.

**Finding:** at the right-sized envelope the bounded lever reclaims only a uniform **~0.3 GB** and
lifts no boot ceiling — because `cuda_graph_max_bs=64` already shrinks the full-width score plane
(`[64, 202752]` ≈ 80–240 MB, not GB; the measured ~1.95 GB DS graph overhead was a bs512-default-
envelope cost). So bounded selector-width is **largely subsumed by the envelope** as a headroom
lever. Its durable value is the **fail-closed served-width contract**, not capacity — see §7.

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
