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

| Bucket | frozen baseline (20260609) | M0 dry-run | M1 score-reduce | M2 top-k | M3 logical-score |
|---|---|---|---|---|---|
| NCCL ring score all-reduce (`AllReduce_Sum_f32_RING`) | 124,873* | TBD | | | |
| top-k/sort stack (mbtopk/radixSort/sbtopk/scan/searchsorted) | 159,166 | TBD | | | |
| `_logical_score_kernel` | 63,107 | TBD | | | |
| all-reduce category total (incl. shared trtllm-fusion) | 163,790 | TBD | | | |
| **Total decode GPU-kernel µs** | **632,239** | TBD | | | |
| aggregate decode tok/s | 459 | TBD | | | |
| recall gate (Δ recall@2048 vs frozen baseline, ≤0.5pp) | — (baseline) | — | | | |
| cross-rank bit-identity (hard) | TBD (M0) | — | | | |
| reduce backend at the DS reduce site | torch_dist (NCCL ring) | torch_dist | | | |

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
