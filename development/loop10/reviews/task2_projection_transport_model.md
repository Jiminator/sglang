## 1. TRANSPORT MODEL

Basis: loop-9 R1 Case-1 total is **480,989 us / 10-step decode window**. The DS score-reduce path is **780 calls/window** by the loop model: 78 layers x 10 decode steps. R1 transport is **93,480 us** of named `all_reduce_two_shot_kernel<bf16,8u>` plus the current fp32->bf16 and bf16->fp32 copy/cast tax, recorded in the plan as **~15-18k us/window**, so the AC transport bucket is **~108-111k us/window** today.

The real Case-1 torch-profile server captured this ladder, logged in `development/loop9/runs/20260611_r1/case1_ds/torch/serve.log` and derived by `get_batch_sizes_to_capture` from `server_args.cuda_graph_bs`:

`[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]`

For W=5120 bf16 compact buffers, message size is:

`padded_graph_bs * 5120 * 2 bytes = padded_graph_bs * 10 KiB`

The 8-rank H200 custom-AR v2 threshold is **160 KiB**. Because `_determine_algo` checks `one_shot_push_threshold` first, the unpinned default is effectively `ONE_SHOT_PUSH` for `<=160 KiB`, then `TWO_SHOT_PULL` above that. “With pin” below means a strict two-shot pin, preferably `override_algo = AllReduceAlgo.TWO_SHOT_PULL`; `override_shot(2)` alone does not zero the push threshold in the current code.

| padded graph bs | compact size KiB | threshold side | default without pin | strict two-shot pin |
|---:|---:|---|---|---|
| 1 | 10 | <=160 | ONE_SHOT_PUSH | TWO_SHOT_PULL |
| 2 | 20 | <=160 | ONE_SHOT_PUSH | TWO_SHOT_PULL |
| 4 | 40 | <=160 | ONE_SHOT_PUSH | TWO_SHOT_PULL |
| 8 | 80 | <=160 | ONE_SHOT_PUSH | TWO_SHOT_PULL |
| 12 | 120 | <=160 | ONE_SHOT_PUSH | TWO_SHOT_PULL |
| 16 | 160 | <=160 | ONE_SHOT_PUSH | TWO_SHOT_PULL |
| 24 | 240 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 32 | 320 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 40 | 400 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 48 | 480 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 56 | 560 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 64 | 640 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 72 | 720 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 80 | 800 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 88 | 880 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 96 | 960 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 104 | 1040 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 112 | 1120 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 120 | 1200 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 128 | 1280 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 136 | 1360 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 144 | 1440 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 152 | 1520 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 160 | 1600 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 168 | 1680 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 176 | 1760 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 184 | 1840 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 192 | 1920 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 200 | 2000 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 208 | 2080 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 216 | 2160 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 224 | 2240 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 232 | 2320 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 240 | 2400 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 248 | 2480 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 256 | 2560 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 272 | 2720 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 288 | 2880 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 304 | 3040 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 320 | 3200 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 336 | 3360 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 352 | 3520 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 368 | 3680 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 384 | 3840 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 400 | 4000 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 416 | 4160 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 432 | 4320 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 448 | 4480 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 464 | 4640 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 480 | 4800 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 496 | 4960 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |
| 512 | 5120 | >160 | TWO_SHOT_PULL | TWO_SHOT_PULL |

Op point: raw bs=29 pads to graph bs=32, so compact W=5120 bf16 is **327,680 B = 320 KiB**, above the one-shot boundary. The closest direct spike-bench measurement is `[29,4608]` bf16, **267,264 B**, where coordinator custom-AR measured **51.9 us/call** and NCCL measured **38.5 us/call**. I use the direct compact-class measurement as the base estimate, while marking it lower confidence because it is an eager CUDA-event microbench. The nsys replay number, **~106 us/call at 11.76 MB**, includes cross-rank arrival skew absorbed as in-kernel wait and should not be scaled linearly by bytes.

Projection:

- Pinned two-shot compact reduce only: **~50-60 us/call** modeled from 51.9 us/call plus graph/skew uncertainty -> **~39-47k us/window**.
- Pinned two-shot compact reduce plus today’s cast/copy path: add a launch-floor-limited **~8-14k us/window**, not byte-scaled to near zero -> **~49-58k us/window**.
- Therefore **<=60k hard is reachable with M1 pinned two-shot alone**, but with limited margin and only if compact replay resembles the spike-bench rather than the full-width nsys wait profile.
- **<=45k stretch is not reliably reachable with pinned two-shot plus casts**. It becomes plausible with task7 cast elimination, and robust only if task8 can choose a faster declared transport such as NCCL or a measured one-shot variant under the value-affecting regime.

## 2. THRESHOLD-FLIP MAP

For W=5120 bf16, the compact threshold crossing is simple: every bucket is `10 KiB * bs`.

| bucket group | buckets | default compact algorithm | exact-regime consequence |
|---|---|---|---|
| at/below boundary | 1, 2, 4, 8, 12, 16 | ONE_SHOT_PUSH | Would silently change summation order versus R1’s full-width bf16 two-shot path. Must pin two-shot or declare value-affecting. |
| above boundary | 24, 32, 40, ..., 512 | TWO_SHOT_PULL | No threshold-induced flip; op point bs=29 -> padded bs=32 is here. |

Quantifying the small-bs value of one-shot is necessarily an estimate because loop-9 did not measure one-shot at `[bs,5120]` buckets. The useful bracket is:

- Measured compact-class two-shot custom-AR: **51.9 us/call** at `[29,4608]` bf16.
- Measured compact-class NCCL: **38.5 us/call** at the same shape.
- A declared one-shot small-bucket path would need to beat pinned two-shot by roughly **7-17 us/call** to matter, i.e. **~5.5-13.3k us/window** if an entire 780-call window ran in those small buckets.
- This saving is irrelevant to the frozen bs=29 op point unless task8 explicitly tests a forced one-shot setup above 160 KiB, which may require a larger push buffer and is value-affecting.

Implementation caveat: in current `custom_all_reduce_v2.py`, `override_shot(2)` only changes `one_shot_pull_threshold`; it leaves `one_shot_push_threshold` intact. For exact M1 pinning, use `override_algo = TWO_SHOT_PULL` or equivalent logic that actually disables one-shot push for `<=160 KiB`.

## 3. CAST-TAX ACCOUNTING

Current `reduce_token_scores` structure in `selection_kernel.py`:

1. `bf16_view.copy_(token_scores)` casts/copies fp32 scores into preallocated bf16 scratch.
2. `reduce_ca.custom_all_reduce(bf16_view)` performs the out-of-place custom-AR reduce when eligible.
3. `token_scores.copy_(reduced)` copies/casts the bf16 reduced output back into fp32 scores for the existing top-k consumer.
4. NCCL fallback has the same final `token_scores.copy_(bf16_view)` copy-back.

The R1 torch summary shows the named reduce at **93,480 us**. The loop-10 plan attributes the associated fp32<->bf16 copy/cast kernels at **~15-18k us/window**. The visible kernel lines are consistent with that order: `bfloat16_copy_kernel_cuda` is **7,481 us / 780 calls ~= 9.6 us/call**, while copy-back is mixed into direct-copy elementwise lines, so the plan’s combined **~19-23 us/call** cast-pair estimate is the right accounting unit.

Width scaling:

- W=5120 versus current 202752 width is **5120 / 202752 = 2.525%**, i.e. **39.6x fewer elements**.
- The “~25x” shrink applies to an 8192 bucket: **202752 / 8192 ~= 24.75x**. For the selected W=5120 bucket, the byte shrink is stronger.
- Pure byte scaling would make the 15-18k cast tax only **~0.4-0.5k us/window**, but that is not credible because the path still launches two copy/cast kernels per score-reduce call.
- Until task7 removes/fuses the casts, compact casts should be modeled as launch-floor-limited: **~8-14k us/window**. Treat any lower value as a measurement result, not a projection premise.

## 4. LOGICAL-SCORE AND TOP-K RE-RATE

`_logical_score_kernel` already uses `seq_lens` to bound live work. In the persistent-worker path, full width mainly inflates the fixed worker grid: at 202752 width and token block 256, `num_token_blocks ~= 792`, so workers cap at 128. With W=5120, `num_token_blocks = 20`, close to the served live window of <=4608 tokens, so the grid drops from `(bs,128)` to `(bs,20)` while useful scoring math stays roughly the same.

Logical-score projection:

- R1: **36,908 us/window**, **~47 us/call**.
- M1 compact-width estimate: **~16-22k us/window**.
- The hard bar **<=20k** is realistic but not guaranteed; stretch **<=15k** is lower confidence.
- Failure mode: live query-signature math, memory loads through `req_to_token`, or graph replay launch floors dominate more than the dead worker programs.

Top-k is also sequence-aware. The radix kernels load only up to `min(seq_len,width)` and early-exit dead blocks, but full width still inflates fixed grids and scratch: W=202752 gives `nblocks ~= 198` at block 1024; W=5120 gives `nblocks = 5`. The suite still has multiple launches: four histogram rounds, four scan rounds, block count, block prefix, and emit.

Top-k projection:

- R1 DS radix top-k: **~36.3k us/window**, excluding the shared non-DS topk/sort residual **20,524 us**.
- M1 compact-width estimate: **~24-32k us/window**. The hard bar **<=28k** is plausible but not assured.
- M2 bf16-authoritative / 2-round bf16 top-k path could bring this to **~20-28k us/window** if it lands without selection diffs or is accepted under the value-affecting gates.
- Failure mode: the launch count, histogram zeroing, and per-call scratch maintenance dominate, so shrinking `nblocks` does not buy enough.

## 5. SPIKE-BENCH READ

Loop-9 spike bench measured the following relevant cases, eager CUDA-event median over 50 iterations with all-rank barriers:

| shape | size | NCCL ring | coordinator custom-AR | read |
|---|---:|---:|---:|---|
| `[29,202752]` fp32 | 23.52 MB | 167.1 us | 170.5 us fallback/ring | Current original fp32 shape is over v2 cap. |
| `[29,202752]` bf16 | 11.76 MB | 105.9 us | 104.1 us two-shot | R1 shipped path: dtype win, not custom-AR win. |
| `[29,4608]` fp32 | 0.53 MB | 37.4 us | 45.9 us custom-AR | Compact-class custom-AR loses. |
| `[29,4608]` bf16 | 0.27 MB | 38.5 us | 51.9 us custom-AR | Closest evidence for W=5120 compact bf16; custom-AR loses by ~13.4 us/call. |

Implications for task8:

- Do not assume pinned two-shot custom-AR is the fastest transport at compact sizes.
- The matrix must include pinned two-shot, declared one-shot where legal/constructible, and NCCL.
- Any switch from pinned two-shot to one-shot or NCCL is value-affecting because summation order changes.
- Build the loser before verdicts: the spike bench says NCCL may be the “boring” winner at compact-class sizes, but captured replay is binding and can differ because custom-AR absorbs arrival skew in-kernel.

## 6. PER-BUCKET PROJECTION TABLE

I keep non-target work constant at the R1 residual, using the midpoint transport accounting:

`non-target residual ~= 480,989 - 109,980 - 36,908 - 36,300 = 297,801 us`

This includes shared kernels and the shared non-DS topk/sort residual; per loop-9 DEC-1, totals can move by ~27k across boots, so bucket attribution is primary.

| bucket | R1 us | projected after M1: compact, pinned two-shot | projected after M2: cast elimination + transport choice | hard bar | stretch bar | confidence |
|---|---:|---:|---:|---:|---:|---|
| DS transport: reduce + casts | 108-111k | 49-58k | 35-45k | <=60k | <=45k | Medium; compact-class bench is eager, nsys skew may not shrink fully. |
| `_logical_score_kernel` | 36,908 | 16-22k | 15-20k | <=20k | <=15k | Medium; grid shrink is real, but useful live math remains. |
| DS radix top-k | ~36.3k | 24-32k | 20-28k | <=28k | <=24k | Medium-low; launch count remains high. |
| shared / non-target residual | ~297.8k | ~297.8k plus boot noise | ~297.8k plus boot noise | n/a | n/a | High as accounting, but total noise is high. |
| TOTAL | 480,989 | ~387-410k | ~368-391k, wider caution band ~367-403k | <=420k | <=395k | Medium. |

Readout:

- M1 alone should clear the total hard bar if the compact two-shot transport bucket lands below 60k.
- M1 alone may or may not clear stretch; it depends on transport landing near the low end and top-k/logical both hitting their bars.
- M2 is the credible stretch path: task7 removes the cast floor, and task8 can choose NCCL or a measured one-shot path if it beats pinned two-shot under the declared value-affecting regime.
- This re-rates the old M5 projection: the earlier **~377-395k** estimate assumed one-shot at compact sizes and used the older **512.7k** baseline. Against R1 **480,989**, the informative landing band remains compatible with the plan’s **~367-403k** estimate, but the transport stretch depends on follow-ups.

## 7. RISKS / WHERE THE MODEL COULD BE WRONG

1. **Skew absorption does not shrink with bytes.**  
   Falsifier: compact pinned two-shot remains near the nsys **~106 us/call** full-width replay number instead of the spike-bench **~52 us/call** compact-class number. Transport would exceed 60k even before casts.

2. **The two-shot pin is incomplete.**  
   Falsifier: bs<=16 compact buckets log or profile as one-shot despite the exact-regime M1 claim. Current `override_shot(2)` is suspicious; use `override_algo=TWO_SHOT_PULL` or prove equivalent behavior.

3. **Cast launch floors dominate.**  
   Falsifier: after compaction, fp32->bf16 and bf16->fp32 copy/cast kernels remain near **15-18k us/window** instead of dropping to **~8-14k**. M1 transport could miss the hard bar.

4. **NCCL beats pinned two-shot enough to matter.**  
   Falsifier: task8 captured replay reproduces the spike-bench relation, e.g. NCCL compact around **38-40 us/call** while custom-AR two-shot is **50+ us/call**. Exact M1 is then deliberately paying ~8-12k/window for fixed summation order.

5. **Logical-score gains are overestimated.**  
   Falsifier: `_logical_score_kernel` stays above **22k us/window** after W=5120. That means live scoring math or memory indirection, not dead workers, is the current limiter.

6. **Top-k remains launch-bound.**  
   Falsifier: radix suite stays above **32k us/window** after compact width. That means shrinking `nblocks` is insufficient, and the conditional multi-block/single-launch redesign becomes necessary.

7. **Full-width fallback is hit unexpectedly.**  
   Falsifier: op-point bucket identity logs show real rows with `max(seq_len) > 5120`, routing material decode calls to the full-width graph. The transport projection assumes the frozen 4096/512 window stays within W=5120.

8. **Compact buffers are not truly custom-AR eligible.**  
   Falsifier: `should_custom_ar` is false due to weak-contiguity, byte alignment, or accidental strided views. That would silently change the transport path unless the planned contiguity assertion and per-bucket backend logs catch it.
