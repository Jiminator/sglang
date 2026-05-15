# CI Regression Bisection Report — `test_mistral_large3_all_variants` (NVFP4 variant) on B200

**Investigator:** Claude (sglang-bisect-ci-regression skill)
**Date:** 2026-05-15
**Repo:** `sgl-project/sglang`
**Reporting run:** https://github.com/sgl-project/sglang/actions/runs/25835354140/job/75909128362

---

## Status

**Reproduced locally.** Root cause identified. Bisection-introducing commit isolated by code review + an A/B reproduction (13afe8a → PASS, 34c0029 → FAIL; backend swap in `fp4_utils.py` also FAILs with a more revealing flashinfer-side assertion).

The previous session's hypothesis ("Likely 51a9403104; possibly 28758d37dd") is **not the root cause**. The real introducer is `d5f3254` (#24452).

## Failure Signature

- **Test:** `test/registered/8-gpu-models/test_mistral_large3.py::TestMistralLarge3::test_mistral_large3_all_variants`
- **Workflow / job:** `Nightly Test (Nvidia)` (run `25835354140`) → `nightly-test-general-8-gpu-b200 (3)` (job `75909128362`), suite `nightly-8-gpu-common`, partition 3/4, runner `b200-novita-1` (8× B200, drv 580.126.09, CUDA 13).
- **Head SHA at time of failure:** `34c0029f0aff4c3d1c714e7d55b2a522bbc0ff69` ("[diffusion] [AMD] feat: support online MXFP4 and fp8 quantization (#21431)", 2026-05-14).
- **Step that failed:** "Run common 8-GPU model tests" — exit code 255.
- **Variant that fails:** **NVFP4** (`mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4`, 128 experts, TP=8, `--attention-backend=trtllm_mla --moe-runner-backend=flashinfer_trtllm`). The TP8 (FP8) and TP8+MTP (FP8 + EAGLE) variants do not exercise this NVFP4 path.

GitHub's raw-job-log endpoint (`/repos/.../actions/jobs/75909128362/logs`) returned 403 — the exact assertion text was reconstructed from local reproduction (below).

### Failure trace (from local reproduction at `34c0029` + flashinfer 0.6.11.post1)

```
File ".../layers/moe/fused_moe_triton/layer.py", line 1093, in run_moe_core
    return self.quant_method.apply(...)
File ".../quantization/compressed_tensors/compressed_tensors.py", line 1030, in apply
    return scheme.apply_weights(layer, dispatch_output)
File ".../compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py", line 315, in apply_weights
    hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
File ".../sglang/srt/layers/quantization/fp4_utils.py", line 36, in _flashinfer_fp4_quantize_impl
    return _flashinfer_fp4_quantize(... backend=_flashinfer_fp4_quantize_backend)
File ".../flashinfer/quantization/fp4_quantization.py", line 924, in _fp4_quantize_cute_dsl
    return nvfp4_quantize_cute_dsl(...)
File ".../flashinfer/quantization/kernels/nvfp4_quantize.py", line 1270, in nvfp4_quantize_cute_dsl
    global_scale.float().reshape(1).contiguous().to(input.device)
RuntimeError: shape '[1]' is invalid for input of size 128
```

`128` is the model's number of experts; the global-scale tensor passed in is `layer.w13_input_scale_quant`, shape `[num_experts]`.

When the cute-dsl backend is patched out (set `_flashinfer_fp4_quantize_backend = "cuda"` in `python/sglang/srt/layers/quantization/fp4_utils.py:22`), the same call falls through to flashinfer's *cuda* kernel and fails with the **more informative** error from `flashinfer/data/csrc/nv_internal/tensorrt_llm/thop/fp4Quantize.cpp:64`:

```
TVM_FFI_ICHECK(globalScale.value().numel() == 1 || globalScale.value().numel() == m)
    << "globalScale should have shape [1] or [num_tokens]";
RuntimeError: Check failed ... is false: globalScale should have shape [1] or [num_tokens]
```

i.e. `flashinfer.fp4_quantize` in 0.6.11.post1 explicitly **rejects per-expert global-scale tensors**; only `[1]` (scalar) or `[num_tokens]` (per-token) is accepted.

## Boundary (verified locally)

| Status | SHA | flashinfer | torch | sgl-kernel | Hardware | NVFP4 + gsm8k |
|---|---|---|---|---|---|---|
| **Last pass** | `13afe8a` (2026-04-29 HEAD-of-main) | `0.6.8.post1` | `2.9.1+cu130` | `0.4.1.post1+cu130` | 8× B200 (drv 580.126.09) | **PASS** — perf bench succeeded; gsm8k **score = 0.951** ≥ baseline 0.85; total runtime 8m55s |
| **First fail** | `34c0029` (CI failure SHA) | `0.6.11.post1` (pinned by the SHA's `pyproject.toml`) | `2.11.0+cu130` (pinned by `pyproject.toml`) | `0.4.2.post1+cu130` (pinned) | same machine | **FAIL** — `RuntimeError: shape '[1]' is invalid for input of size 128` during the first MoE forward (during `fp4_gemm` autotune) |
| Sub-experiment | `34c0029` + patch `fp4_utils.py` to `backend="cuda"` | same | same | same | same | **FAIL** — same call site, cleaner error: `globalScale should have shape [1] or [num_tokens]` |

Reproduction used the user-modified `test_mistral_large3.py` reduced to the NVFP4 variant (TP8+MTP needs `mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle` weights — these were downloaded successfully (11 GB), but TP8+MTP was deferred to keep the bisection cycle fast).

Two environment notes:
- `SGLANG_IS_IN_CI=true` is required for the test to reach the same code path as CI: the per-server context-length check in `model_config.py:_derive_context_length` short-circuits with a `ValueError` outside CI. (This is unrelated to the regression; just an env-parity gotcha.)
- The `nightly-test-nvidia.yml` workflow pins this env at the workflow level. With it set, reproduction is byte-for-byte the CI path.

## Candidate Commits / Independent Review

Filtered set of commits in `13afe8a..34c0029` (587 commits) that touch NVFP4 / FP4 / flashinfer / `trtllm_mla_backend` / `compressed_tensors_w4a4_nvfp4_moe`:

| Rank | SHA | PR | Title | Verdict |
|---|---|---|---|---|
| **C1 (root cause)** | **`d5f3254`** | **#24452** | `[Dependency] Flashinfer 0.6.8post1 -> 0.6.11` | **Confirmed.** The 0.6.8.post1 → 0.6.11 bump pulls in a flashinfer kernel that *explicitly checks* `globalScale.numel() == 1 || globalScale.numel() == num_tokens` (`fp4Quantize.cpp:64`). The 13afe8a call site `compressed_tensors_w4a4_nvfp4_moe.py:315` passes a per-expert tensor (numel == 128). Pre-bump (0.6.8.post1), the kernel accepted the per-expert tensor (or treated it as a broadcast/no-op), so the test was green on 13afe8a. Post-bump, every NVFP4 + flashinfer_trtllm MoE forward fails. |
| C2 | `1d80a1a` | #23745 | `Use Cute-DSL NVFP4 quantization kernels` | **Aggravates, but does not cause.** This wraps `flashinfer.fp4_quantize` in a sglang custom op and routes through `backend="cute-dsl"` on SM100 (B200). When the cute-dsl backend hits the same per-expert tensor, the failure surfaces as `reshape '[1]' is invalid for input of size 128` in `nvfp4_quantize.py:1270` (still rooted in flashinfer ≥ 0.6.11's MoE refactor). Reverting to `backend="cuda"` reveals the underlying flashinfer assertion (above) — the regression persists. |
| C3 | `51a9403` | #25129 | `Update flashinfer to 0.6.11.post1` | **Not the root cause.** Diff is two version-string updates and a `pyproject.toml` patch-bump. The flashinfer changelog 0.6.11 → 0.6.11.post1 covers SM120 W4A16 MoE, sccache, JIT `-DNDEBUG` fixes, a typo in `trtllm_fused_moe_runner.cu` — none of these revert the fp4_quantize shape contract. |
| C4 | `28758d3` | #24816 | `Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4` | **Not the root cause.** Gated on SM90 + MXFP4 + GPT-OSS/DeepSeek-V4. The B200 (SM100) NVFP4 + Mistral-Large-3 path never enters this code. PR #25329 later disabled this PR's own tests, but the disablement is unrelated to the Mistral test. |
| C5 | `7618ad7` | #24925 | `[attn backend] Integrate tokenspeed_mla prefill/decode kernels` | Refactors `trtllm_mla_backend.py` (+223/-92), but the new code is a no-op until `--attention-backend tokenspeed_mla`. The Mistral test uses `trtllm_mla`. Not the root cause. |
| C6 | `73e93be` | #21954 | `[1/4] NVFP4 KV cache: quantization strategy abstraction and kernel` | Doesn't touch the MoE fp4_quantize call site. |

## Root Cause Classification

**Code regression — `d5f3254` (#24452) is the introducing commit.**

The bug is at the sglang↔flashinfer interface:
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py:315` passes `layer.w13_input_scale_quant` (per-expert, shape `[num_experts]`) as `global_scale` to `flashinfer.fp4_quantize`.
- Flashinfer ≥ 0.6.11 now enforces `numel == 1 || numel == num_tokens`, so per-expert tensors are rejected.

`1d80a1a` (cute-dsl wrapper) is a co-conspirator: it picks a different flashinfer backend on B200, which converts the friendly cuda-side assertion into a raw `RuntimeError: shape '[1]' is invalid` from `nvfp4_quantize_cute_dsl:1270`. The fix has to be the same in both cases.

## Independent Reading of the Prior Session's Hypothesis

> Likely 51a9403104 (NVFP4/flashinfer); possibly 28758d37dd (FlashInfer SM90 cutlass MXFP4 MoE)

- **51a9403 (#25129)** — wrong. Diff is one-line version-string updates and a patch-version bump in `pyproject.toml`. The flashinfer 0.6.11.post1 changelog (vs 0.6.11) does not touch `globalScale`'s shape contract.
- **28758d3 (#24816)** — wrong. SM90 + MXFP4 path only. B200 is SM100, Mistral-Large-3 NVFP4 uses NVFP4 + `flashinfer_trtllm`, which routes through `compressed_tensors_w4a4_nvfp4_moe.apply_weights → flashinfer.fp4_quantize`, not through the new SM90 MXFP4 cutlass backend.

The actual chain is `d5f3254` (major flashinfer bump) → exposes a long-standing per-expert global_scale mismatch in `compressed_tensors_w4a4_nvfp4_moe.py:315`. The prior session pattern-matched on "any recent NVFP4/flashinfer commit"; the substantive code/dependency change is one step earlier.

## Recommended Fix

Two complementary directions; pick (1) for an immediate unblock, (2) for the correct long-term fix.

1. **(sglang-side, fastest)** Update `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py:315` to pass a shape-conformant `global_scale`. The hidden states `x` is shape `[num_tokens, hidden]`, so flashinfer wants either `globalScale.shape == [1]` or `globalScale.shape == [num_tokens]`. Concretely, replace the second argument with a per-token tensor derived from `layer.w13_input_scale_quant` — for example expand to per-token using the topk routing, or pass a scalar reduction (`layer.w13_input_scale_quant.amax()` or `.amin()`) if the kernel only needs an upper-bound. The exact reduction matches whatever the pre-0.6.11 flashinfer kernel was doing internally with the per-expert tensor.
2. **(flashinfer-side)** Restore the previous behavior in `flashinfer/data/csrc/nv_internal/tensorrt_llm/thop/fp4Quantize.cpp:64` so per-expert global_scale (numel == `num_experts`) is accepted again, or add a documented separate code path / overload for the MoE case. Filing an issue against flashinfer-ai/flashinfer that quotes PR #24452's bump-the-pin-only behavior would be appropriate.

While the proper fix lands, a one-line revert in `python/sglang/srt/layers/quantization/fp4_utils.py:22` to force `backend="cuda"` will *not* unblock the test (the cuda backend also rejects per-expert global_scale — confirmed above). The sgl-side fix in (1) is unavoidable.

## Files of Interest

- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py:315` — the broken call site (per-expert global_scale).
- `python/sglang/srt/layers/quantization/fp4_utils.py:22-45` — sglang wrapper that picks `cute-dsl` vs `cuda` backend on SM100 (introduced in 1d80a1a); both backends now reject the call.
- `flashinfer/data/csrc/nv_internal/tensorrt_llm/thop/fp4Quantize.cpp:64` — the new strict shape assertion (post-0.6.10).
- `flashinfer/quantization/kernels/nvfp4_quantize.py:1270` — cute-dsl variant; `reshape(1)` on a 128-element tensor is the raw failure surfaced by the wrapper.
- `python/pyproject.toml` (between 13afe8a and 34c0029) — `flashinfer_python==0.6.8.post1` → `0.6.11.post1` (via PRs #24452 then #25129).

## Reproduction Recipe

```bash
# at 13afe8a (PASS)
git checkout 13afe8a
# venv already had flashinfer 0.6.8.post1, sglang-kernel 0.4.1.post1+cu130, torch 2.9.1+cu130
SGLANG_IS_IN_CI=true SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  python -m unittest test.registered.8-gpu-models.test_mistral_large3.TestMistralLarge3
# → ✓ Performance, Accuracy 0.951 ≥ 0.85

# at 34c0029 (FAIL) — upgrade env to match the SHA's pyproject pins
git checkout 34c0029
uv pip install "flashinfer_python==0.6.11.post1" "flashinfer_cubin==0.6.11.post1" \
               "torch==2.11.0" "torchaudio==2.11.0" "torchvision" \
               --index-url https://download.pytorch.org/whl/cu130
curl -L -o '/tmp/sgl0.4.2.post1.whl' \
  'https://github.com/sgl-project/whl/releases/download/v0.4.2.post1/sglang_kernel-0.4.2.post1+cu130-cp310-abi3-manylinux2014_x86_64.whl'
uv pip install --reinstall --no-deps /tmp/sgl0.4.2.post1.whl
SGLANG_IS_IN_CI=true SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  python -m unittest test.registered.8-gpu-models.test_mistral_large3.TestMistralLarge3
# → RuntimeError: shape '[1]' is invalid for input of size 128
```

Note: the local `pyproject.toml` at 34c0029 dropped the `[[tool.uv.index]] sglang-kernel-cu130` index because PyPI now hosts sglang-kernel — but the PyPI wheel is built against torch 2.11.x (uses `c10::cuda::c10_cuda_check_implementation` with the `unsigned int` mangled signature), so torch must be upgraded to 2.11.0+cu130 before `sglang-kernel==0.4.2.post1` will load on a Blackwell host. The Dockerfile for `CUDA 13.0.1` does this implicitly; a user-level `uv sync` from the lockfile-less 34c0029 needs the manual install order above.

## Timeline (compute summary)

- 02:35 UTC — kicked off baseline run at 13afe8a; hit `--speculative-draft-model-path` context-length ValueError on TP8+MTP because `SGLANG_IS_IN_CI` was not set.
- 02:39 UTC — restart with `SGLANG_IS_IN_CI=true`; NVFP4 cache had 8 incomplete blob shards; serial CI download was watchdog-slow.
- 02:57 UTC — parallel `snapshot_download(max_workers=8)` finished the missing NVFP4 shards (3:37 wall-clock).
- 02:57–03:06 UTC — 13afe8a NVFP4 PASS (perf + gsm8k 0.951, 8m55s).
- 03:13 UTC — `git checkout 34c0029`; `uv pip install` for flashinfer/sgl-kernel/torch upgrades; sgl-kernel PyPI wheel did **not** load against torch 2.9.1 (`undefined symbol _ZN3c104cuda29c10_cuda_check_implementation…jb`), so torch was upgraded to 2.11.0+cu130.
- 03:25 UTC — first FAIL on 34c0029 with cute-dsl backend (`reshape '[1]' is invalid for input of size 128`).
- 03:35 UTC — confirmed the same failure with `backend="cuda"` patch — flashinfer's cuda kernel rejects per-expert globalScale directly. Both paths converge to the same root cause in `compressed_tensors_w4a4_nvfp4_moe.py:315`.

## Open Items / Not Investigated

- **TP8 (basic) and TP8+MTP variants.** The user's modified test file was further reduced to NVFP4-only to keep the bisection short. The TP8 basic variant has its own performance-results JSON from an earlier session on this machine and appears to be a separate signal; TP8+MTP needs `SGLANG_IS_IN_CI=true` + EAGLE draft weights and was not re-executed at 34c0029. If the CI job fails because of TP8 or TP8+MTP rather than NVFP4, the failure signature in those variants needs a separate look — but the NVFP4 variant alone is enough to make the partition fail with `exit code 255` (matching CI).
- **Whether per-token expansion of `layer.w13_input_scale_quant` matches the kernel's pre-0.6.11 semantics.** The pre-0.6.11 flashinfer kernel must have either ignored or broadcast the per-expert tensor; identifying the exact original behavior is a flashinfer-history task and the right person to make the call is whoever shepherded PR #24452.

## TL;DR

- CI partition `nightly-test-general-8-gpu-b200 (3)` fails on the NVFP4 variant of `test_mistral_large3` because `compressed_tensors_w4a4_nvfp4_moe.apply_weights` passes a per-expert `globalScale` to `flashinfer.fp4_quantize`, which flashinfer ≥ 0.6.11 rejects.
- The introducing commit is `d5f3254` (PR #24452, "Flashinfer 0.6.8post1 → 0.6.11"). The previous session's pointers (`51a9403`, `28758d3`) are *not* the root cause.
- Fix: pass a scalar or per-token tensor at `compressed_tensors_w4a4_nvfp4_moe.py:315` (and/or open a flashinfer issue for the silently-broken globalScale contract).
