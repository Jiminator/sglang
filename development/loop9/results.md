# Loop 9 Ledger — DS-on Decode Kernel Optimization (running results)

## Final gap statement (close-out)

Frozen baseline → after this loop (Case-1, frozen recipe, one trial each):
**632,239 → 512,687 µs** per 10-step decode window (−18.9%; the strong AC-1.4 marker ≤516k is
met); **1.84× → 1.495×** vs the frozen DSA floor (342,857); decode throughput
**459 → 646.79 tok/s (+41%)**. Per-bucket: score-reduce f32 ring eliminated (124.9k → 0; bf16
custom-AR v2 + casts ≈ 113k incl. cross-rank wait absorption), DS top-k 138.6k → ≈36.3k,
logical-score 63.1k → 43.2k (its 40k gate near-missed by 8% — DEC-1 trend documentation). Every
landed change passed the recall gate (overall Δ ≤ +0.01pp, bound 0.5pp) and the hard cross-rank
bit-identity check; M2+M3 are selection-bit-identical to M1 served (0/2496 selcap rows). The
DS index/scoring tax vs DSA's fused indexer remains structurally dominated by the static-width
(202752) dead tax — see the wildcard proposal below. Separately, the Penalty-B admission cap is
lifted at a re-tuned op point (bs 29 → 64, see the memory audit section).

## Follow-on notes (close-out)

1. **Width-bucketed DS selector graphs (+ compact score buffers)** — the structural fix for the
   dead-width tax in all three buckets; projected ~1.10–1.15× DSA floor; needs-user-decision
   (cuda-graph-runner width bucketing is a real integration change).
2. **Persistent/bounded-grid logical-score kernel** — the small slice that closes the 43.2k →
   ≤40k gate on its own (~10–15k expected); de-risking alternative to (1).
3. **AOT promotion of the radix top-k** — the Triton suite lands at ≈36.3k/window; a fused
   single-kernel CUDA version (fast_topk_v2-class, ~17.7 µs/call floor) needs an sgl-kernel
   source build (prebuilt wheel here); worth folding into (1) if pursued.
4. **Graph-safe support for non-default DS variants** (DEC-5 follow-on: cosine/hybrid/mean/
   anchors/int8/lifted keep riding the existing paths unchanged this loop).
5. **Re-tuned serving op point** (mem 0.77 + cuda-graph-max-bs 64, bs-64 admission) — if it is
   to become a served default, it needs its own SLO/profiling characterization loop; the
   loop-9 profiling recipe stays at the frozen mem-0.7 op point.
6. Boot-log wording: `token_label_table ... scales=fp16` is misleading in fp16 mode (no scales
   sidecar is allocated) — cosmetic fix candidate.

One column per landed idea; kept ideas stack and the running Case-1 number becomes the next
idea's baseline. Frozen references (development/profiling/runs/20260609/, do NOT re-run except
under the AC-4 mandatory-regression rule): Case 2 (DSA, mem 0.7, bs 29) = **342,857 µs**/10-step
decode window, Case 3 (DSA, mem 0.8, bs 64) = 422,236 µs. One trial per profiling run; per-bucket
gates primary (shared-kernel boot-to-boot noise ~27k µs on the total).

Per-bucket gates: score-reduce — NCCL ring line eliminated + named custom-AR kernel;
top-k stack ≤ 80,000 µs; logical-score ≤ 40,000 µs. Total (secondary trend): ≤ 560k min / ≤ 516k
strong.

## Per-idea kernel-bucket ledger (Case-1 re-profiles, torch TP-0, µs / 10-step decode window)

| Bucket | frozen baseline (20260609) | M0 dry-run (20260610) | M1 score-reduce (20260610) | M2+M3 top-k + logical-score (20260611, combined run) |
|---|---|---|---|---|
| NCCL ring score all-reduce (`AllReduce_Sum_f32_RING`) | 124,873* | 124,949 | **0 (eliminated)** | 0 |
| named custom-AR kernel (`all_reduce_two_shot_kernel<bf16,8u>`) | 0 | 1,269† | 67,343 | 95,225‡ |
| score-reduce cast overhead (fp32↔bf16, in elementwise) | — | — | ≈ +18,156 | ≈ +18k (unchanged) |
| torch top-k/sort lines (mbtopk/radixSort/sbtopk/gatherTopK) | 138,602 DS-attr | ≈ same | ≈ 134,714 DS-attr | **0 (eliminated)** |
| new radix selection kernels (hist/scan/count/prefix/emit + fill) | — | — | — | **≈ 36,290** (hist 19,422 + scan 5,690 + count 3,616 + emit 3,569 + fill 3,993) |
| shared non-DS topk/sort residual (present in Case 2 at 20,564) | 20,564 | ≈ same | ≈ same | 20,470 |
| `_logical_score_kernel` | 63,107 | 63,211 | 63,161 | **43,180** |
| **Total decode GPU-kernel µs** | **632,239** | **631,381** | **585,158** | **512,687** |
| ratio vs frozen Case-2 (342,857) | 1.84× | 1.84× | 1.71× | **1.495×** |
| aggregate decode tok/s | 459 | 459.4 | **500.75** | **646.79** |
| recall gate (Δ recall@2048 vs frozen baseline, ≤0.5pp) | — (baseline) | — (no code change) | **PASS** (+0.010pp overall; max per-length +0.24pp) | **PASS** (64.706 — identical to M1: both changes selection-bit-identical) |
| cross-rank bit-identity (hard) | PASS (M0 selcap, 8 ranks) | — | **PASS** (selcap 8 ranks + 8-rank torchrun) | **PASS**; selcap diff vs M1 served baseline: **0/2496 rows** |
| reduce backend at the DS reduce site | torch_dist (NCCL ring) | torch_dist | **custom_ar_v2** (bf16 two-shot pull) at decode buckets; NCCL-bf16 logged fallback for >16 MB capture buckets (e.g. bs 512 prefill bucket) | custom_ar_v2 (unchanged) |

‡ the bf16 two-shot pull kernel's attributed time grew +27,882 µs after M2 removed the long
serializing top-k: the pull kernel absorbs cross-rank arrival skew in-kernel (wait, not work).
Net total still −72,471 µs vs M1. The structural fix for the whole reduce bucket remains
live-width reduction (follow-on).

### Round-1 column — logical-score gate closed (20260611, runs/20260611_r1/)

| Bucket | M2+M3 (prev) | R1: persistent-worker logical score |
|---|---|---|
| `_logical_score_kernel` | 43,180 | **36,908 — AC-1.3 GATE MET (≤40,000)** |
| new radix selection kernels (in "other") | ≈36.3k | ≈36.3k (unchanged; hist 19,527) |
| torch top-k/sort lines | 0 | 0 (residual 20,524 = shared non-DS sorts) |
| NCCL ring score reduce | 0 | 0; custom-AR bf16 two-shot 93,480 |
| **Total decode GPU-kernel µs** | 512,687 | **480,989** |
| ratio vs frozen Case-2 (342,857) | 1.495× | **1.403×** |
| aggregate decode tok/s | 646.79 | **654.28** |
| recall gate (≤0.5pp) | PASS 64.706 | **PASS 64.706** (identical — change is selection-bit-identical) |
| cross-rank bit-identity (hard) | PASS | **PASS**; selcap diff vs M2 served: **0/2496 rows** |

The change: `_logical_score_kernel` restructured to a persistent-worker grid (static
(bs, ≤128) programs; each strides device-side over its LIVE blocks) + dead `-inf` stores
skipped on the radix path (the seq-bounded selector never reads past seq_len; the legacy
torch fallback and the recall-oracle/anchor paths keep them — regression-pinned). The −31.7k
total also carries shared-kernel boot variance (trtllm fusion −16.4k, fp8-quant −6.5k); the
attributable per-bucket win is the logical-score −6,272. All four per-bucket gates now MET.

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

## Per-bucket gate verdicts after M1–M3 (AC-1)

- **Score-reduce (AC-1.1): MET literally** — ring line eliminated, named custom-AR v2 bf16
  kernel at the DS reduce site, backend recorded, zero replay allocations. Honest attribution:
  the win is the bf16 byte halving; custom-AR ≈ NCCL at equal bytes.
- **Top-k (AC-1.2): MET with margin** — DS-attributed selection cost 138.6k → ≈36.3k µs
  (gate ≤80k); torch top-k/sort lines at zero; deterministic seq-aware radix kernel, selection
  bit-identical, tie-deterministic across ranks. (compare_decode's frozen classifier does not
  know the new kernel names — they appear under "other"; the ledger rows above give the
  per-kernel-line truth.)
- **Logical-score (AC-1.3): trend, gate near-missed (DEC-1 documentation)** — 63,107 →
  43,180 µs vs the 40,000 gate (−32%, miss by 3.2k/8%). The microbench predicted ~34k at fixed
  seq 4608; the served window (seq 4097→4608 growing + real cache state) lands at ~55 µs/call.
  The remaining cost is the dead-grid launch floor over the static 202752 width; the earnest
  next lever is the persistent/bounded-live-grid kernel redesign (task11's candidate D) —
  recorded as the follow-on, not bundled per the surgical-change doctrine.
- **Total (AC-1.4, secondary): STRONG marker met** — 512,687 ≤ 516,000 (minimum 560,000),
  attributable per-bucket as above; 1.84× → 1.495× vs the frozen DSA floor; decode throughput
  459 → 646.79 tok/s (+41%).

## Memory audit + admission re-tune (Penalty B)

Audit verdict (Codex over the measured per-rank budget): **re-tune** — the recoverable memory is
NOT the signatures (fp16 mode allocates no scales sidecar; the boot log's `scales=fp16` wording
is misleading) and NOT table padding (~2.4 MiB), but the **over-captured decode graph ladder**:
the default capture set goes to bs 512 while the KV pool caps admission at ~30, and the DS graph
state (scratch_scores fp32 + bf16 reduce scratch per decode bucket, each [bs_i, 202752]) made the
capture pool 17.68 GB.

Measured re-tune (new characterized op point — the frozen Case-1 recipe is unchanged):
`--cuda-graph-max-bs 64 --mem-fraction-static 0.77` →
- max_total_num_tokens 142,208 → **330,048** (KV 18.9 GB); token_label_table 5.29 → 12.28 GB/rank
  (grows proportionally with the pool); graph capture pool 17.68 → **0.88 GB**; 12.6 GB steady
  headroom; boot + capture clean (29.6 s).
- bs-64 bench batch (4096 ISL): **64 requests decoding concurrently under CUDA graph**
  (server log `#running-req: 64, cuda graph: True`, token usage 0.81), output throughput
  1023.44 tok/s on a 64-OSL probe. The DS bs-29 admission cap is lifted; artifacts:
  development/loop9/runs/20260611_m4/.

## Wildcard proposal (logical-score gate shortfall; next-loop user decision)

The one unmet per-bucket gate after M1–M3 is logical-score (43.2k vs 40k). The reviewed redesign
ranking (Codex, full analysis in the round artifacts): (1) **width-bucketed DS selector graphs
with compact per-bucket score buffers** — kills the dead-width tax in ALL THREE buckets incl.
the reduce; projected total ~377–395k µs ≈ 1.10–1.15× the DSA floor; requires a real
cuda-graph-runner integration (width bucketing alongside bs bucketing); (2) persistent/
bounded-grid logical-score kernel — smallest reliable patch, ~10–15k for that bucket alone,
total ~1.35×. Disposition: **needs-user-decision** for the next loop.

## Notes / deviations

- M2 and M3 share one Case-1 re-profile/gate run (the M3 one-line change landed while the M2
  gate sequence was booting; all three phases measured the combined state). Both changes are
  selection-bit-identical by proof and touch disjoint buckets, so per-idea attribution stays
  exact per-bucket; the recall gate covers the combined landed state. Recorded in the goal
  tracker's Plan Evolution Log.
- Candidate A of the top-k milestone was delivered as a measured disqualification
  (m2_candidate_a_findings.md) rather than a full wrapper build — its radix tie races fail the
  cross-rank hard gate and the exact repair costs more than today's pipeline; disposition
  blessed by the benchmark-off review. The new kernel is Triton JIT; AOT promotion is a
  follow-on (sgl-kernel here is a prebuilt wheel).
