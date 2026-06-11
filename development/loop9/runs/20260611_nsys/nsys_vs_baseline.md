# Post-loop-9 nsys capture vs frozen Case-1 baseline

One nsys run of the loop-9 final landed state (commit 0a04a964d) with the frozen Case-1
recipe verbatim (`run_case.sh case1 nsys 29`, DELAY=210, OSL=512, one trial). This is a NEW
capture for timeline analysis — it does not replace any frozen reference, and the binding
per-bucket decode numbers remain the torch ledger (runs/20260611_r1/).

- rep: `case1_ds/nsys/trace.nsys-rep` (940M, on disk, gitignored)
- kernel CSV / rollup: `case1_ds/nsys/kern_sum.csv`, `case1_ds/nsys/decode_summary.txt`
- baseline: `development/profiling/runs/20260609/case1_ds/nsys/` (old DS path, 2026-06-09)

## Whole-capture GPU-kernel time, all 8 ranks (prefill + 512 decode steps)

| bucket | baseline s | now s | delta | calls (base → now) |
|---|---|---|---|---|
| `AllReduce_Sum_f32_RING` (DS score reduce) | 52.71 | 0.00 | −52.71 | 328,224 → 0 |
| `all_reduce_two_shot_kernel<bf16,8u>` | 0.52 | 35.73 | +35.20 | 8,416 → 336,640 |
| torch topk/sort stack (mbtopk/radixSort/scan_by_key/gatherTopK<long>) | 58.55 | 0.00 | −58.55 | 7,220,928 → 0 |
| new radix suite (_radix_hist/_radix_scan/_block_count/_block_prefix/_emit) | 0.00 | 14.21 | +14.21 | 0 → 3,610,464 |
| `_logical_score_kernel` | 26.53 | 15.57 | −10.96 | 328,224 (same) |
| **TOTAL** | **333.72** | **268.82** | **−64.90 (−19.4%)** | |

Net by AC-1 bucket: score reduce −17.5 s (−33%, 161 → 106 µs/call incl. cross-rank skew
absorbed in the pull kernel), top-k −44.3 s (−76%), logical score −11.0 s (−41%). The
decode-only torch window shows −23.9%; the whole-capture −19.4% is diluted by the shared
prefill and idle tail.

Run-to-run controls (validate apples-to-apples): `fused_moe_kernel` 45.76 → 45.69 s,
`deep_gemm` 19.22 → 19.21 s, sparse MLA decode 13.36 → 13.62 s, non-DS
`AllReduce_Sum_bf16_RING_LL` 31.69 → 31.69 s with identical call counts (23,864) — confirming
that kernel is pre-existing non-DS usage, not a DS fallback.

## Caveats

- `decode_summary.txt`'s category rollup classifies the new radix-suite kernels under
  "other" (the frozen `summarize_nsys.py` classifier predates them); the table above uses
  explicit kernel-name matching. The frozen script is left untouched.
- The loop-9 NVTX ranges (`ds_logical_score`, `ds_score_allreduce`, `ds_topk_select`) do NOT
  appear in this capture: decode executes as CUDA-graph replay, so host-side range
  annotations never fire inside the traced region (nvtx_sum shows only NCCL/cuBLAS/CCCL
  library ranges). They are visible in eager-mode/torch-profiler runs only. Kernel-name
  attribution is unaffected.
- The dominant DS residual on the timeline is the bf16 score reduce (35.7 s, 13.3% of all
  GPU-kernel time) — the static-width dead tax the width-bucketed selector-graph follow-on
  targets (results.md, follow-on 1).
