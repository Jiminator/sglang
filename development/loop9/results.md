# Loop 9 Ledger — DS-on Decode Kernel Optimization (running results)

One column per landed idea; kept ideas stack and the running Case-1 number becomes the next
idea's baseline. Frozen references (development/profiling/runs/20260609/, do NOT re-run except
under the AC-4 mandatory-regression rule): Case 2 (DSA, mem 0.7, bs 29) = **342,857 µs**/10-step
decode window, Case 3 (DSA, mem 0.8, bs 64) = 422,236 µs. One trial per profiling run; per-bucket
gates primary (shared-kernel boot-to-boot noise ~27k µs on the total).

Per-bucket gates: score-reduce — NCCL ring line eliminated + named custom-AR kernel;
top-k stack ≤ 80,000 µs; logical-score ≤ 40,000 µs. Total (secondary trend): ≤ 560k min / ≤ 516k
strong.

## Per-idea kernel-bucket ledger (Case-1 re-profiles, torch TP-0, µs / 10-step decode window)

| Bucket | frozen baseline (20260609) | M0 dry-run (20260610) | M1 score-reduce (20260610) | M2 top-k | M3 logical-score |
|---|---|---|---|---|---|
| NCCL ring score all-reduce (`AllReduce_Sum_f32_RING`) | 124,873* | 124,949 | **0 (eliminated)** | | |
| named custom-AR kernel (`all_reduce_two_shot_kernel<bf16,8u>`) | 0 | 1,269† | 67,343 | | |
| score-reduce cast overhead (fp32↔bf16, in elementwise) | — | — | ≈ +18,156 | | |
| top-k/sort stack (mbtopk/radixSort/sbtopk/scan/searchsorted) | 159,166 | 159,162 | 155,184 | | |
| `_logical_score_kernel` | 63,107 | 63,211 | 63,161 | | |
| all-reduce category total (incl. shared trtllm-fusion) | 163,790 | 163,177 | 102,653 | | |
| **Total decode GPU-kernel µs** | **632,239** | **631,381** | **585,158** | | |
| ratio vs frozen Case-2 (342,857) | 1.84× | 1.84× | 1.71× | | |
| aggregate decode tok/s | 459 | 459.4 | **500.75** | | |
| recall gate (Δ recall@2048 vs frozen baseline, ≤0.5pp) | — (baseline) | — (no code change) | **PASS** (+0.010pp overall; max per-length +0.24pp) | | |
| cross-rank bit-identity (hard) | PASS (M0 selcap, 8 ranks) | — | **PASS** (selcap 8 ranks + 8-rank torchrun) | | |
| reduce backend at the DS reduce site | torch_dist (NCCL ring) | torch_dist | **custom_ar_v2** (bf16 two-shot pull) at decode buckets; NCCL-bf16 logged fallback for >16 MB capture buckets (e.g. bs 512 prefill bucket) | | |

† small pre-existing non-DS usage of the kernel in the baseline trace.

M1 verdict (AC-1.1 + AC-2): the f32 ring line is eliminated at the DS reduce site and replaced by
the NAMED custom-AR v2 bf16 kernel; CUDA graph capture succeeded in the production runner (74.6 s,
all buckets) and the 8-rank determinism test proved zero replay allocations + cross-rank
bit-identity at the real shape. Honest attribution per the spike: the win is the bf16 byte
halving (custom-AR ≈ NCCL at equal bytes); net score-reduce path 124,949 → ≈85.5k
(67,343 kernel + ≈18.2k casts) = −39.4k µs, total −46.2k µs. Selected-index diff vs the frozen
oracle: 74.84% of (layer,row) selections moved (the expected bf16 boundary reshuffle —
recorded for attribution; recall gate proves the swaps quality-neutral). The residual ~67k µs
reduce cost is the dead-width tax (static 202752 width vs ≤4608 live tokens) — structural
remainder for M5/follow-on, per the spike findings. Artifacts: development/loop9/runs/20260610_m1/.

M0 dry-run verdict (protocol check, AC-5): run_case.sh + summarize_torch.py + compare_decode.py all
work end-to-end; dry-run vs frozen Case-1 same-config reboot agrees per-bucket within ~600 µs
(total Δ = −858 µs), far inside the planned ~27k µs noise allowance — per-bucket gates have full
sensitivity. Dry-run vs frozen Case 2: 1.84×, deltas reproduce (+123,362 all-reduce / +138,598
topk-stack / +63,211 logical-score). Artifacts: development/loop9/runs/m0_dryrun/.

\* the ring line is the DS-attributed share measured as Case1−Case2 NCCL f32 delta; the category
row above it is the full all-reduce category including the shared trtllm fusion all-reduce.

## Frozen M0 baselines (AC-2 references) — captured 2026-06-10

- **Production selection oracle** (CUDA-graph mode, served Case-1 op-point at cuda-graph-max-bs 4,
  fixed 4-prompt deterministic workload: 546/2878/6121/12531 prompt tokens × 8 decode steps,
  2 passes): `development/loop9/runs/20260610_m0/selcap_baseline_digest.json` — **PASS**:
  64 steps × 78 layers × 8 ranks bit-identical (cross-rank hard gate), output contract clean,
  pass0 == pass1 (same-boot run-to-run deterministic). Raw per-(layer,step) dumps (315 MB) on
  disk at `runs/20260610_m0/selcap_baseline/` (untracked; regenerable deterministically at this
  commit) — the `diff` input for per-change attribution.
- **NIAH oracle recall@2048 baseline** (eager mode, recall_oracle config-borne, fixed gated
  workload: lengths 1024/4096/16384 words × N=20 × 4 decode steps = 18,720 samples):
  `development/loop9/runs/20260610_m0/recall_baseline.json` — overall **64.696%**
  (1024w: 100.0% — dense, decode sound; 4096w: 58.045%; 16384w: 36.042%); zero failure markers;
  recall@2048 == selected_contains_needle at every length (score-rank rule matches decode
  selection); all 60 trials offline-token == server-token (needle span mapping exact).
  Gate resolution: 0.5pp ≈ 31 samples per length bucket.
- **Tie-semantics check**: production graph-safe pipeline (raw torch.topk) tie behavior matches
  the documented (score desc, pos asc) eager contract on this torch build — probed at widths
  4/4096/163840 incl. boundary plateaus; pinned by `TestGraphSafePipelineAdversarial` fixtures.
  No pre-existing tie-semantics defect.

## M1 premise correction (spike evidence, feeds task3/task4)

The plan's Feasibility Hints assumed the DS score reduce is ~[29, 4608] fp32 ≈ 534 KB. Measured
reality: the graph-safe reduce operates on `scratch_scores[:bs, :max_seq_len]` with
`max_seq_len = req_to_token.shape[1] = context_len = 202752` (served boot log) — **[29, 202752]
fp32 ≈ 23.5 MB per call**, consistent with the frozen 160 µs/call (124,873 µs / 780 calls).
Custom-AR caps: v1 `_MAX_CAR_SIZE` = 8 MB; v2 max pull 16 MB (default), one/two-shot thresholds
160 KB at TP=8 on H200. ⇒ the production-width fp32 reduce is custom-AR-ineligible as-is; the
spike benches NCCL ring vs coordinator dispatch vs a 32 MB-pull v2 across widths/dtypes
(m1_spike_allreduce_bench.py) to evidence the viable levers (bf16 reduce ≈ 11.7 MB ≤ 16 MB pull;
width-vs-cost curve; wide-cap v2 at 23.5 MB). Group fact verified: under plain TP=8,
`_ATTN_TP is _TP` (parallel_state.py:1906-1907) — the attention-TP group IS the custom-AR-capable
TP GroupCoordinator.

## Notes / deviations

- (none yet)
