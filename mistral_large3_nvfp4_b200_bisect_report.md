# CI Regression Bisection Report — `test_mistral_large3_all_variants` (NVFP4 variant) on B200

**Investigator:** Claude (sglang-bisect-ci-regression skill)
**Date:** 2026-05-15
**Repo:** `sgl-project/sglang`
**Reporting run:** https://github.com/sgl-project/sglang/actions/runs/25835354140/job/75909128362
**Related report:** see [`mistral_large3_tp8_mtp_b200_bisect_report.md`](mistral_large3_tp8_mtp_b200_bisect_report.md) for the **TP8+MTP** variant regression in the same CI partition — *different* root cause (PR #24436, `_resolve_speculative_algorithm_alias`), *different* fix, **not addressed by PR #25407**.
**Status:** Reproduced; root cause isolated and confirmed by an A/B/C/D experiment matrix. PR #25407 ("Fix Mistral Large 3 nightly test") implements the proper call-site fix and is verified to make this variant pass — see "PR #25407 verification" below.

> **Correction note:** An earlier draft of this report (commit `b0591ab`) blamed the flashinfer dependency bump `d5f3254` (PR #24452). That conclusion was wrong. `d5f3254` *aggravates* the failure but is **not** the introducing commit, and the revert in **PR #25310 (`22dfcda`) does not fix the Mistral NVFP4 test** — verified by running the same test locally at the post-revert SHA `0fde6153` and getting the identical traceback. The actual culprit is `1d80a1a` (PR #23745, "Use Cute-DSL NVFP4 quantization kernels"), which wraps `flashinfer.fp4_quantize` and forces `backend="cute-dsl"` on SM100/B200. The cute-dsl backend has a hard `reshape(1)` on `global_scale` that is incompatible with the per-expert `[num_experts]` scale the MoE call site passes; the same kernel ships in both flashinfer 0.6.8.post1 *and* 0.6.11+, so the flashinfer version is irrelevant to this failure. The corrected analysis is below.

---

## Failure Signature

- **Test:** `test/registered/8-gpu-models/test_mistral_large3.py::TestMistralLarge3::test_mistral_large3_all_variants`
- **Workflow / job:** `Nightly Test (Nvidia)` (run `25835354140`) → `nightly-test-general-8-gpu-b200 (3)` (job `75909128362`), suite `nightly-8-gpu-common`, partition 3/4, runner `b200-novita-1` (8× B200, drv 580.126.09, CUDA 13).
- **Head SHA at time of failure:** `34c0029f0aff4c3d1c714e7d55b2a522bbc0ff69` ("[diffusion] [AMD] feat: support online MXFP4 and fp8 quantization (#21431)", 2026-05-14).
- **Step that failed:** "Run common 8-GPU model tests" — exit code 255.
- **Variant that fails:** **NVFP4** (`mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4`, 128 experts, TP=8, `--attention-backend=trtllm_mla --moe-runner-backend=flashinfer_trtllm`). TP8 (FP8 basic) is unaffected.

Failure trace (reproduced locally at `34c0029`, and verified identical at `0fde6153` post-revert):

```
File ".../layers/moe/fused_moe_triton/layer.py", line 1093, in run_moe_core
    return self.quant_method.apply(...)
File ".../quantization/compressed_tensors/compressed_tensors.py", line 1030, in apply
    return scheme.apply_weights(layer, dispatch_output)
File ".../compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py", line 315, in apply_weights
    hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
File ".../sglang/srt/layers/quantization/fp4_utils.py", line 36, in _flashinfer_fp4_quantize_impl
    return _flashinfer_fp4_quantize(... backend="cute-dsl")
File ".../flashinfer/quantization/fp4_quantization.py", line 703/924, in fp4_quantize -> _fp4_quantize_cute_dsl
    return nvfp4_quantize_cute_dsl(...)
File ".../flashinfer/quantization/kernels/nvfp4_quantize.py", line 1270, in nvfp4_quantize_cute_dsl
    global_scale.float().reshape(1).contiguous().to(input.device)
RuntimeError: shape '[1]' is invalid for input of size 128
```

The `128` matches the model's number of experts; `layer.w13_input_scale_quant` has shape `[num_experts]`.

---

## Historical Failure Window (from CI metrics artifacts)

Using `gh run download <run_id> -n metrics-8gpu-b200-partition-3` for each scheduled `nightly-test-nvidia.yml` run:

| Run | SHA | Date (UTC) | flashinfer | Mistral NVFP4 perf rows present? | Status |
|---|---|---|---|---|---|
| 607 | `aa7a9af1` | 2026-05-11 01:00 | 0.6.8.post1 | **YES (4 batch sizes)** | **PASS** |
| 608 | `74d70af0` | 2026-05-12 00:54 | 0.6.8.post1 | NO | FAIL |
| 609 | `4fb40bf` | 2026-05-13 00:58 | 0.6.11.post1 | NO | FAIL |
| 610 | `34c0029` | 2026-05-14 01:00 | 0.6.11.post1 | NO | FAIL (the run in the question) |
| 613 | `0fde6153` | 2026-05-15 00:58 | **0.6.8.post1 (post-revert)** | NO | **STILL FAIL** |

Boundary: PASS at `aa7a9af1` (run 607) → FAIL at `74d70af0` (run 608). Window = 51 commits. The notable change in that window is `1d80a1a` ("Use Cute-DSL NVFP4 quantization kernels", PR #23745), which touches the `fp4_quantize` import in `compressed_tensors_w4a4_nvfp4_moe.py:309` and adds the `backend="cute-dsl" if is_sm100_supported() else "cuda"` wrapper in `python/sglang/srt/layers/quantization/fp4_utils.py:22`.

`d5f3254` (flashinfer 0.6.8.post1 → 0.6.11) and `51a9403` (0.6.11 → 0.6.11.post1) both land **after** the regression boundary (May 12+); they cannot have introduced the failure because run 608 (still on flashinfer 0.6.8.post1) already exhibits it. PR #25310 (`22dfcda`) reverts those two flashinfer bumps but leaves `1d80a1a` in place — which is why run 613 (post-revert) is still failing on the same NVFP4 partition.

---

## A/B/C/D Local Experiments

All on 8× B200 (drv 580.126.09, CUDA 13). `SGLANG_IS_IN_CI=true`, `SGLANG_ENABLE_JIT_DEEPGEMM=0`, `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`. NVFP4-only variant.

| # | sglang SHA | `fp4_utils.py:22` backend | flashinfer | sgl-kernel | torch | Outcome | Total wall time |
|---|---|---|---|---|---|---|---|
| A | `13afe8a` (pre-`1d80a1a`) | n/a (no wrapper) | 0.6.8.post1 | 0.4.1.post1+cu130 | 2.9.1+cu130 | **PASS** — gsm8k 0.951 | 8m 55s |
| B | `34c0029` (CI failure SHA, has wrapper) | `"cute-dsl"` | 0.6.11.post1 | 0.4.2.post1+cu130 | 2.11.0+cu130 | **FAIL** — `reshape '[1]' invalid for input of size 128` (cute-dsl) | crashed at `fp4_gemm` autotune |
| C | `34c0029` | `"cuda"` (patch) | 0.6.11.post1 | 0.4.2.post1+cu130 | 2.11.0+cu130 | **FAIL** — `globalScale should have shape [1] or [num_tokens]` (`fp4Quantize.cpp:64`, *new strict cuda-side check in flashinfer 0.6.11+*) | crashed at autotune |
| D | `0fde6153` (post-revert HEAD) | `"cute-dsl"` (unchanged) | 0.6.8.post1 | 0.4.2.post1+cu130 | 2.11.0+cu130 | **FAIL** — identical traceback to B (`nvfp4_quantize_cute_dsl:1270` `reshape '[1]' invalid`) | crashed before bench |
| E | `0fde6153` | **`"cuda"` (patch)** | 0.6.8.post1 | 0.4.2.post1+cu130 | 2.11.0+cu130 | **PASS** — gsm8k 0.949 ≥ 0.85 | 23m 43s |
| **F** | **`ea217a2` (parent of `1d80a1a`, the falsification test)** | **n/a (no wrapper)** | **0.6.8.post1** | **0.4.2.post1+cu130** | **2.11.0+cu130** | **PASS — gsm8k 0.956 ≥ 0.85** | **9m 33s** |

Interpretation:

- A → D shows the regression is **independent of flashinfer version** (both 0.6.8.post1; A passes, D fails). The only relevant sglang-side change between A and D is `1d80a1a` adding the wrapper that routes B200/SM100 through `backend="cute-dsl"`.
- D → E (one-line patch of `fp4_utils.py:22` from `"cute-dsl"` to `"cuda"`) restores the pre-`1d80a1a` routing and the test passes — confirming the cute-dsl backend is the proximate failure, not the per-expert scale itself.
- B → C shows that flashinfer 0.6.11 *additionally* added the same strict shape check to the cuda backend (`fp4Quantize.cpp:64`), so once `1d80a1a` is in *and* flashinfer is bumped, even the cuda fallback fails. PR #25310's revert removes that strict-check side, but the cute-dsl-side failure (D) remains because `1d80a1a` is still in.
- **F is the direct falsification test:** check out the literal git parent of `1d80a1a` (`ea217a2`, "ci: remove Execute Notebooks workflow") with all other dependencies identical to D, and the test passes. Combined with D failing on the same dependency stack, this isolates the regression to exactly the diff introduced by `1d80a1a`.

---

## Candidate Commits / Independent Review

| Rank | SHA | PR | Title | Verdict |
|---|---|---|---|---|
| **C1 (root cause)** | **`1d80a1a`** | **#23745** | **Use Cute-DSL NVFP4 quantization kernels** | **Confirmed.** Touches `python/sglang/srt/layers/quantization/fp4_utils.py` to wrap `flashinfer.fp4_quantize` with `_flashinfer_fp4_quantize_backend = "cute-dsl" if is_sm100_supported() else "cuda"`, and changes `compressed_tensors_w4a4_nvfp4_moe.py:309` to import that wrapper. After this change, every B200 NVFP4 MoE forward routes to flashinfer's cute-dsl kernel, which assumes scalar `global_scale` and fails on the per-expert `layer.w13_input_scale_quant`. Date: 2026-05-10 17:40 PDT — first nightly to ship it is run 608 (May 12), which is exactly where the regression starts in CI metrics. |
| C2 | `d5f3254` | #24452 | [Dependency] Flashinfer 0.6.8post1 → 0.6.11 | **Aggravates, but does not cause.** Tightened the *cuda* backend to also reject `numel != 1 && numel != num_tokens`, eliminating the fallback that would otherwise work if `_flashinfer_fp4_quantize_backend` were forced to `"cuda"`. Without `1d80a1a` in place, this PR alone wouldn't have broken Mistral NVFP4 (sglang's call site went via the looser pre-0.6.11 `fp4_quantize`). |
| C3 | `51a9403` | #25129 | Update flashinfer to 0.6.11.post1 | Not relevant. One-line version-string updates. The breaking change is between 0.6.10 and 0.6.11. |
| C4 | `28758d3` | #24816 | Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4 | Not relevant. SM90+MXFP4 path; never enters this test. |
| C5 | `22dfcda` | #25310 | revert flashinfer 0.6.11 bumps | Reverts C2+C3; **does not fix the regression** because it doesn't touch `1d80a1a`. Verified by experiment D above. |

### Why the previous session's hypothesis was wrong (and why my earlier draft was too)

> "Likely 51a9403104 (NVFP4/flashinfer); possibly 28758d37dd (FlashInfer SM90 cutlass MXFP4 MoE)" — prior session
> "d5f3254 is the root cause" — earlier draft of this report (commit `b0591ab`)

The prior session pattern-matched on "recent flashinfer-touching commit," which mis-ranked `51a9403` and `28758d3` (a patch-version bump and an SM90-MXFP4 path that never executes on this test). My earlier draft moved one step earlier in the same direction (`d5f3254`, the bigger flashinfer bump), which had circumstantial evidence (a flashinfer-side stricter check) but was still wrong because:

1. The CI metrics show NVFP4 already failed in run 608 (2026-05-12), when flashinfer was still 0.6.8.post1.
2. The revert in #25310 returned main to flashinfer 0.6.8.post1 yet run 613 (post-revert) still fails.
3. Locally, `0fde6153` (post-revert, flashinfer 0.6.8.post1) reproduces the identical `reshape '[1]'` cute-dsl traceback; only patching `_flashinfer_fp4_quantize_backend` to `"cuda"` makes the test pass.

The lesson: the dependency bump narrative was downstream/confounded. The real change is the *sglang-side* routing decision in `fp4_utils.py:22`.

---

## Root Cause Classification

**Code regression — `1d80a1a` (#23745) is the introducing commit.**

The wrapper in `python/sglang/srt/layers/quantization/fp4_utils.py` unconditionally forces `backend="cute-dsl"` for any SM100 device. The cute-dsl path in flashinfer (`flashinfer/quantization/kernels/nvfp4_quantize.py:nvfp4_quantize_cute_dsl`) does:

```python
global_scale.float().reshape(1).contiguous().to(input.device)
```

which is only valid for `global_scale.numel() == 1`. The MoE call site in `compressed_tensors_w4a4_nvfp4_moe.py:315` passes `layer.w13_input_scale_quant`, which is shape `[num_experts]` (= 128 for Mistral-Large-3). Before `1d80a1a` the call went through `flashinfer.fp4_quantize` without a backend kwarg, hitting flashinfer's then-permissive cuda kernel that accepted per-expert scales.

---

## Recommended Fix

### Quick (1-line, sglang-side, immediately unblocks B200 NVFP4 MoE on current main)

`python/sglang/srt/layers/quantization/fp4_utils.py:22`

```diff
-    _flashinfer_fp4_quantize_backend = "cute-dsl" if is_sm100_supported() else "cuda"
+    _flashinfer_fp4_quantize_backend = "cuda"
```

Confirmed PASS by experiment E (gsm8k 0.949) on the current post-revert main. The cute-dsl backend was a performance optimization; falling back to `"cuda"` matches pre-`1d80a1a` behavior. Note: this only stays safe while main is pinned to flashinfer 0.6.8.post1 (the cuda backend in 0.6.11+ added the same strict shape check — see experiment C). If main re-bumps flashinfer to 0.6.11+ in the future, the proper fix (below) is required.

### Proper fix (sglang-side)

Stop passing a per-expert tensor as `global_scale` to `flashinfer.fp4_quantize` in the MoE input-quantization path. Two options:

1. **Per-token expansion:** at `compressed_tensors_w4a4_nvfp4_moe.py:315`, expand `layer.w13_input_scale_quant` to a `[num_tokens]` tensor via the topk routing (each token's experts → scale → reduce to per-token). This matches what flashinfer's cute-dsl and post-0.6.10 cuda backends both want.
2. **Different flashinfer API:** flashinfer's `trtllm_fp4_block_scale_moe` already accepts per-expert state and may have an `input_quantize` helper that doesn't need the per-token contract. Worth checking with the flashinfer team.

### Flashinfer-side request

The error message at `flashinfer/data/csrc/nv_internal/tensorrt_llm/thop/fp4Quantize.cpp:64` is good ("`globalScale should have shape [1] or [num_tokens]`"), but the cute-dsl branch in `flashinfer/quantization/kernels/nvfp4_quantize.py:1270` silently `reshape(1)`s the input — that's the bug that surfaces as the cryptic "shape '[1]' is invalid for input of size 128". A simple shape assertion at the top of `nvfp4_quantize_cute_dsl` with the same message would make this debuggable in 30 seconds instead of via dual-backend bisection.

---

## Files of Interest

- `python/sglang/srt/layers/quantization/fp4_utils.py:22` — the wrapper that picks `"cute-dsl"` on SM100 (introduced by `1d80a1a`). **One-line patch site for the quick fix.**
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py:315` — the call site passing per-expert `layer.w13_input_scale_quant` as `global_scale`.
- `flashinfer/quantization/kernels/nvfp4_quantize.py:1270` — cute-dsl kernel that crashes on per-expert `global_scale` (both 0.6.8.post1 and 0.6.11+).
- `flashinfer/data/csrc/nv_internal/tensorrt_llm/thop/fp4Quantize.cpp:64` — strict shape check in flashinfer 0.6.11+'s cuda backend (added by the same family of MoE refactors).

---

## Reproduction Recipe

```bash
# Experiment A — pre-1d80a1a; should PASS
git checkout 13afe8a
# venv already had flashinfer 0.6.8.post1 + sglang-kernel 0.4.1.post1+cu130 + torch 2.9.1+cu130
SGLANG_IS_IN_CI=true SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  python -m unittest test.registered.8-gpu-models.test_mistral_large3.TestMistralLarge3
# → ✓ Performance, Accuracy 0.951 ≥ 0.85

# Experiment D — post-revert HEAD; should FAIL identically to the CI run
git checkout 0fde6153
# need flashinfer 0.6.8.post1, sglang-kernel 0.4.2.post1+cu130, torch 2.11.0+cu130
# (The PyPI sglang-kernel 0.4.2.post1 is built against torch 2.11.x; if your torch is 2.9.x
# you'll see "undefined symbol _ZN3c104cuda29c10_cuda_check_implementation…jb" — upgrade torch first.)
uv pip install "torch==2.11.0" "torchaudio==2.11.0" "torchvision" --index-url https://download.pytorch.org/whl/cu130
curl -L -o /tmp/sgl0.4.2.post1.whl \
  'https://github.com/sgl-project/whl/releases/download/v0.4.2.post1/sglang_kernel-0.4.2.post1+cu130-cp310-abi3-manylinux2014_x86_64.whl'
uv pip install --reinstall --no-deps /tmp/sgl0.4.2.post1.whl
# flashinfer is already at 0.6.8.post1 on this SHA's pyproject
SGLANG_IS_IN_CI=true SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  python -m unittest test.registered.8-gpu-models.test_mistral_large3.TestMistralLarge3
# → RuntimeError: shape '[1]' is invalid for input of size 128

# Experiment E — same as D but with the one-line patch; should PASS
git checkout 0fde6153
sed -i 's|_flashinfer_fp4_quantize_backend = "cute-dsl" if is_sm100_supported() else "cuda"|_flashinfer_fp4_quantize_backend = "cuda"|' \
  python/sglang/srt/layers/quantization/fp4_utils.py
SGLANG_IS_IN_CI=true SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  python -m unittest test.registered.8-gpu-models.test_mistral_large3.TestMistralLarge3
# → ✓ Performance, Accuracy 0.949 ≥ 0.85
```

---

## Open Items / Not Investigated

- **TP8+MTP variant of the same test.** TP8+MTP is *also* red on the same partition for the same time window, but for a completely unrelated reason (PR #24436's `_resolve_speculative_algorithm_alias` crashes on Mistral-native-format drafts). It is documented separately in [`mistral_large3_tp8_mtp_b200_bisect_report.md`](mistral_large3_tp8_mtp_b200_bisect_report.md) and is **not fixed by PR #25407**.
- **Whether the `"cuda"` quick-fix performance is acceptable.** `1d80a1a` made the cute-dsl routing the default specifically because it was faster on SM100. The proper fix (PR #25407's per-token/scalar collapse of `global_scale`) keeps the cute-dsl path *and* compatibility; the one-line `cute-dsl → cuda` patch was correct only on flashinfer 0.6.8.post1 and traded performance for correctness.
- **Why CI didn't catch this in PR #23745's own pre-merge CI.** PR #23745 may not run on B200, or its NVFP4 path was exercised against a model with `num_experts == num_tokens` such that the shape mismatch was hidden. Worth a separate pass.

---

## TL;DR

- CI partition `nightly-test-general-8-gpu-b200 (3)` fails on the NVFP4 variant of `test_mistral_large3` because **`1d80a1a` (PR #23745, "Use Cute-DSL NVFP4 quantization kernels") routes B200's `fp4_quantize` through flashinfer's cute-dsl kernel, which assumes scalar `global_scale` and crashes on the MoE call site's per-expert `[num_experts]` tensor**.
- **The previous session's pointers (`51a9403`, `28758d3`) are wrong.** My earlier `d5f3254` hypothesis is also wrong.
- **PR #25310's revert of the flashinfer bump does NOT fix this** — confirmed by reproducing the identical failure at the post-revert HEAD (`0fde6153`).
- **PR #25407 ("Fix Mistral Large 3 nightly test") is the correct fix** — verified to pass gsm8k 0.957 on the full 3-variant test on B200 with flashinfer 0.6.11.post1. The diff slices `layer.w13_input_scale_quant[:1]` at the call site so the per-expert tensor becomes a length-1 tensor that satisfies the strict `globalScale.numel() == 1` contract on both cute-dsl and cuda backends.
- The **TP8+MTP** variant in the same partition fails for an unrelated reason (PR #24436); see [`mistral_large3_tp8_mtp_b200_bisect_report.md`](mistral_large3_tp8_mtp_b200_bisect_report.md). PR #25407 does not address it.

---

## Follow-up note (2026-05-15, later in the same day)

After the corrected NVFP4 analysis above, two additional questions were investigated by request.

### 1. Does the one-line `cute-dsl → cuda` fix still hold on the *latest* `main`?

**No** — `main` has since moved past PR #25310's revert. PR #25335 ("Fix gpt oss triton kernels and upgrade flashinfer back to 0.6.11.post1", merged `0c19540`) **re-bumped flashinfer to 0.6.11.post1**, which carries a stricter cuda-side check (the `globalScale.numel() == 1 || numel == num_tokens` assertion at `fp4Quantize.cpp:64`, same one observed in experiment C). Reproduced locally:

| # | sglang SHA | `fp4_utils.py:22` backend | flashinfer | NVFP4 Outcome |
|---|---|---|---|---|
| **G** | `0c19540` (latest `main` at time of test) | `"cuda"` (one-line patch) | **0.6.11.post1** | **FAIL** — `globalScale should have shape [1] or [num_tokens]` (identical to experiment C) |

So the one-line patch only restores the green state while `main` is pinned to flashinfer `0.6.8.post1`; once flashinfer is `0.6.11.post1` again (which it is today), a proper call-site fix is required: collapse `layer.w13_input_scale_quant` (per-expert, shape `[num_experts]`) to either scalar `[1]` or per-token `[num_tokens]` in `compressed_tensors_w4a4_nvfp4_moe.py:315` before passing as `global_scale` — which is exactly what PR #25407 does (see next section).

### 2. PR #25407 verification

PR **#25407** ("Fix Mistral Large 3 nightly test", head `e3fb4ee`, open at time of writing) is the proper call-site fix. The diff is one hunk:

```diff
-            # Quantize input hidden states using fp4_quantize
+            # global_scale must be shape [1] (strict in cute-dsl backend).
             hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
                 x,
-                layer.w13_input_scale_quant,
+                layer.w13_input_scale_quant[:1],
                 self.group_size,  # sf_vec_size
                 False,  # use_ue8m0
                 False,  # is_sf_swizzled_layout
```

i.e. it slices the per-expert tensor to a length-1 tensor before passing it as `global_scale`, satisfying flashinfer's `numel() == 1` shape contract on every backend (cute-dsl *and* post-0.6.10 cuda).

Verified locally on 8× B200 with `flashinfer==0.6.11.post1`, `sglang-kernel==0.4.2.post2+cu130`, `torch==2.11.0+cu130` against the full 3-variant test:

| Variant on PR #25407 (`e3fb4ee`) | Outcome |
|---|---|
| TP8 | ✓ PASS — gsm8k 0.953 |
| TP8+MTP | ✗ STILL FAIL — *unrelated*, see [`mistral_large3_tp8_mtp_b200_bisect_report.md`](mistral_large3_tp8_mtp_b200_bisect_report.md) |
| **NVFP4** | **✓ PASS — gsm8k 0.957** (this regression's fix) |

Total wall time 1574s (≈ 26 min). PR #25407 lands the green light for the NVFP4 variant of this test.
