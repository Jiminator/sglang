# Loop 11 Results — Authoritative Current State

> Maintained rewrite-over-append: this document always reflects the loop's current state.
> Last regenerated: Round 0, 2026-06-13. HEAD at round start: `6714a5663`.

## 1. Current state summary

- **M0 in flight (Round 0).** task1 (frozen radix-ON DSA baseline) DONE; task2 (DS-Offload
  rejection memo) DONE; task0 (memory accounting + probe matrix) RUNNING — table below updates
  when the matrix completes.
- No production code changes landed yet. Probe-only env hooks (dev-only, reverted before the
  round-0 commit) are recorded in `runs/20260613_m0/probe_hacks.patch`.

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

## 5. task0: max-stable-fraction / capacity probe matrix — COMPLETE

Driver `runs/20260613_m0/stage_task0_probes.sh`; rows `runs/20260613_m0/probes.tsv`; per-probe
boot fields `runs/20260613_m0/probe_logs/`. Each probe = boot + capacity readout + graph-capture
success + short serve smoke. Probe-only env hooks (`SGLANG_DS_PROBE_TABLE_TOKENS=8192` table-free
mock → ~0.30 GB residual mock table, stays visible in the table_GB column so the rows are honest;
`SGLANG_DS_PROBE_SKIP_INDEXER=1` sidecar-gate preview) per `runs/20260613_m0/probe_hacks.patch` —
**reverted before this round's commit; zero production code changed this round.**

`ready_GB` = `available_gpu_mem` at server-ready (post weights + KV pool + table + DS graph
capture), the headroom that sustains concurrent 4096-ISL decode. **It — not the token readout —
is the real discriminator.**

| probe | frac | variant | idx | env | tokens | bs_cap | table GB | ready GB | capture/smoke |
|---|---:|---|---|---|---:|---:|---:|---:|---|
| p01 | 0.70 | fp16 | on  | default | 142,208 | 30  | 5.29 | **26.63** | yes / OK |
| p14 | 0.70 | fp16 | off | default | 174,848 | 37  | 6.51 | 25.27 | yes / OK |
| p02 | 0.75 | fp16 | on  | default | 276,416 | 59  | 10.28 | 13.87 | yes / OK |
| p04 | 0.75 | fp16 | on  | rs      | 276,416 | 59  | 10.28 | 19.33 | yes / OK |
| p06 | 0.75 | int8 | on  | default | 276,416 | 59  | 5.46 | 18.45 | yes / OK |
| p03 | 0.80 | fp16 | on  | default | 410,560 | 89  | 15.27 | **1.25** | yes / OK (boot-only) |
| p05 | 0.80 | fp16 | on  | rs16k   | 410,560 | 89  | 15.27 | 6.94 | yes / OK |
| p07 | 0.80 | int8 | on  | default | 410,560 | 89  | 8.11 | 8.17 | yes / OK |
| p08 | 0.80 | int8 | on  | rs      | 410,560 | 89  | 8.11 | 13.81 | yes / OK |
| p09 | 0.80 | int8 | on  | rs16k   | 410,560 | 89  | 8.11 | 14.04 | yes / OK |
| p10 | 0.80 | int8 | off | rs16k   | 504,640 | 109 | 9.97 | 11.71 | yes / OK |
| p11 | 0.80 | tablefree | on  | default | 410,560 | 89  | 0.30* | 16.26 | yes / OK |
| p12 | 0.80 | tablefree | off | rs      | 504,640 | 109 | 0.30* | 21.25 | yes / OK |
| p13 | 0.85 | tablefree | off | rs16k   | 669,568 | 145 | 0.30* | 13.55 | yes / OK |

`*` mock residual (true table-free = 0; these rows are ~0.30 GB pessimistic).
DSA reference (frozen case3): @0.8 = 410,560 tokens / bs89 / **18.50 GB** ready.

**What the matrix establishes (all four are robust to the smoke-vs-sustained caveat below):**

1. **Token capacity / bs_cap is a function of (mem_fraction, indexer-gate) ONLY — table dtype
   does not move it.** Every @0.8 config reads ~410,560 tokens / bs89 (indexer-on) or ~504,640 /
   bs109 (indexer-off), regardless of fp16/int8/table-free. The pool is sized from
   `available_bytes // cell_size` *before* the table is allocated from leftover; the table dtype
   changes the leftover (headroom), not the pool. So **AC-1.1's bs≥64 floor and AC-1.2's
   ≥390k-tokens @0.8 readout are mechanically cleared by the mem-fraction alone** — the binding
   question is sustainability (headroom), not the token number.
2. **Headroom is what the table/indexer-gate/envelope levers actually buy, and it explains the
   draft's "DS stuck at 0.7" exactly:** fp16 @0.8 (p03) has **1.25 GB** ready — it boots,
   captures graphs, and answers a 1-request smoke, but cannot sustain the 4096-ISL workload
   (the established gen-OOM). Each lever adds headroom at the same 0.8/bs89 capacity:
   fp16 1.25 → int8 8.17 (p07) → table-free 16.26 (p11); the envelope adds ~5–6 GB on top
   (int8+rs 13.81, p08).
3. **task3 indexer-gate works end to end** (+23% tokens, two independent measurements) and adds
   headroom — verified in the live boot path, not just on paper.
4. **Endgame preview — table-free + indexer-gate strictly dominates DSA's memory op-point:**
   p12 (table-free / indexer-off / rs @0.8) = **504,640 tokens / bs109 / 21.25 GB ready** vs
   DSA@0.8's 410,560 / bs89 / 18.50 GB. More tokens, deeper headroom. p13 pushes to bs145 @0.85
   with 13.55 GB still in hand. This is the absorbed-latent (task6) + indexer-gate (task3) target
   laid bare.

**Honesty caveat (binding on every "OK" above).** These are **boot + graph-capture + a single
24-token smoke**, NOT sustained-load proofs. The only GLM-5.1 config with a *sustained* 4096-ISL
anchor is fp16 @0.7 = 26.63 GB ready (the 20260612 served config). The `ready_GB` column
*predicts* sustainability (and predicts fp16@0.8's 1.25 GB cannot serve), but the served fraction
per config is confirmed only on the milestone ladders — **task4** for the int8 served config,
the **M2 gate** for table-free. No probe row is presented as a serving verdict.

**Sequencing read-out for M1 (DEC-3 = full M1 first):** the int8 served-fallback config (task4)
plus the indexer-gate (task3) plus the right-sized envelope land in the 11–14 GB-ready band at
0.8 / bs89–109 (p08/p09/p10) — roughly half the fp16@0.7 anchor's headroom but at ~3× the pool.
To clear AC-1.1's bs≥64 floor, int8 needs **0.8** (bs89): int8 @0.75 reads bs59 (p06), under the
floor, so the floor on int8 comes from the fraction or the indexer-gate (int8+gate would lift a
0.75 pool to ~bs73), not from int8 alone. The richest-headroom config that still clears bs≥64 is
int8 @0.8 + envelope (p08/p09, 13.8–14.0 GB ready, bs89). task4's sustained ladder picks the
served fraction; the matrix supplies the fallback rungs.

## 6. Queue state

See `development/loop11/queue.md` (kickoff-populated Round 0): task0–task9 mainline, q1–q8
kickoff candidates (q7 parked on owner AC-6 ruling; q8 conditional), r1–r5 recorded rejections.
