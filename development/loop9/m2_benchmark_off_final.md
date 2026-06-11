# M2 benchmark-off — final record (all candidates built, per DEC-4)

Round-1 completion of the two-candidate contract. All numbers: H200, captured-shape Case-1
tensors ([29, 202752] fp32 scores, bf16-quantized values, live window 4608, K=2048), median of
50–100 iters. "Captured" = CUDA-graph replay (the production decode mode; eager numbers carry
host JIT/dispatch latency that replay does not).

| Candidate | op point (seq 4608) | seq 16384 | all-live (202752) | exact? | tie-deterministic? | graph-safe? |
|---|---|---|---|---|---|---|
| baseline: torch two-pass topk pipeline | 203.7 µs | ≈204 (width-bound) | ≈204 | yes | empirically on this build | yes (shipped pre-loop) |
| **A: exact fast_topk_v2 wrapper** (`fast_topk_candidate_a.py`) | **1530.1 µs eager** | — | — | yes (all fixtures) | yes (10-run) | no (allocating full-width repair) |
| A-raw: fast_topk_v2 unwrapped | 17.7 µs | 22.5 | — | yes on distinct scores | **NO** (atomicAdd boundary race; 10 identical runs → different selections) | n/a (disqualified by the cross-rank hard gate) |
| **B-Triton: deterministic seq-aware radix suite** (`topk_kernel.py`, SHIPPED) | **52.6 µs captured** / 194.3 eager | 36.1* | 440.9 | yes (all fixtures) | yes | yes (zero-alloc replay, mutation-tracking) |
| **B-AOT: one-block-per-row CUDA op** (`sgl-kernel/csrc/elementwise/ds_topk.cu`) | **44.4 µs captured** / 50.6 eager | 72.8 | 631.0 | yes (all fixtures) | yes | yes (zero-alloc replay, mutation-tracking, no scratch) |

\* B-Triton's 16384 number predates the round-1 dead-store change and is the topk suite only;
the suite's passes were already seq-bounded.

## Winner: B-Triton stays integrated

- At the served op point B-AOT is 16% faster (44.4 vs 52.6 ≈ −6.4k µs/window), but it is
  **2× slower at 16k contexts and 1.4× slower all-live**: one block per row caps parallelism at
  bs=29 blocks on a 132-SM part, while the Triton multi-launch suite spreads each pass across
  the GPU. Production serves long contexts; trading a 6.4k µs op-point win for a 2× mid/long-
  context regression fails the benchmark-off's own risk rule ("benchmark all captured shapes,
  not one median" — task8 review).
- Candidate A as specified is built, exact, and deterministic — and 29× slower than B-Triton
  (the full-width repair is the price of exactness around the racy kernel). Its raw kernel
  remains the measured cost floor (17.7 µs) that a future fused multi-block AOT design should
  target.

## AOT op status (DEC-4's "AOT additions are allowed")

The operator is source-complete in the sgl-kernel tree: `csrc/elementwise/ds_topk.cu`
(single launch, shared-memory histograms + ordered block scans, no global scratch, any K),
declared in `include/sgl_kernel_ops.h`, registered in `csrc/common_extension.cc`, listed in
`CMakeLists.txt`, exposed as `sgl_kernel.top_k.ds_topk_sequence_order`, with registered tests
(`TestDsTopkAOT`: exact-vs-reference fixtures, tie determinism, graph-replay mutation +
zero-alloc) that activate when an op-bearing wheel is installed (env-gated JIT build of the
in-tree source for dev boxes). Compile-verified and benchmarked on this box via
`torch.utils.cpp_extension` over the in-tree source (30.7 s compile, sm_90).

The full wheel build **succeeded** on this box after an escalating fix chain (nvcc not on
PATH → stale subproject caches → CUDA 13's CCCL relocated to `<toolkit>/include/cccl/` which
third-party deps don't expect → a vendored cu12-CCCL version conflict; final working
configuration: clean build dir + `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`
`-DCMAKE_CUDA_ARCHITECTURES=90` `-DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include/cccl`,
wheel-only via `uv build`). Result: `sglang_kernel-0.4.2.post2-cp310-abi3-linux_x86_64.whl`
(323 MB, sm_90; sha256 20ec104f6f2b7d7e9c60ae2ea8839804…), with `ds_topk_sequence_order`
registered in `sgl_kernel/sm90/common_ops.abi3.so` (symbol + torch schema verified by
inspection). Attempt log: `runs/20260611_r1/sgl_kernel_build.log`.

The wheel is deliberately NOT installed: force-reinstalling a rebuilt wheel would replace the
prebuilt binary that every frozen reference and baseline in this loop ran on, invalidating the
frozen-reference premise for all subsequent profiling (the AC-4 rule's spirit). Adopting it is
a separate, gated op-point change for a future loop, paired with the mandatory DSA regression
(DS-off smoke + Case-2 re-validation).

## Follow-on (unchanged)

A fused multi-block AOT design (several blocks per row + cross-block coordination) targeting
the 17.7 µs floor across all context lengths remains the headroom item, ideally folded into
the width-bucketed selector-graph redesign (M5 proposal).
