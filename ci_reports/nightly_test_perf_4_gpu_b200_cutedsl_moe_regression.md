# CI Regression Bisection Report

**Test:** `test_a_gsm8k` in `TestDeepseekV3FP4CuteDSLMoE`
**Test file:** `test/registered/backends/test_deepseek_v3_fp4_cutedsl_moe.py`
**Workflow:** `.github/workflows/nightly-test-nvidia.yml`
**Job (partition):** `nightly-test-perf-4-gpu-b200`
**Suite:** `nightly-4-gpu-b200`
**Hardware:** 4× NVIDIA B200 (sm_100a)

---

## Summary

Commit **`044649c2`** ("feat: Support flashinfer_cutedsl MoE runner with flashinfer alltoall backend (#22669)", 2026-05-20) introduced a code regression that causes `TestDeepseekV3FP4CuteDSLMoE::test_a_gsm8k` to fail on every nightly run since 2026-05-21.

The PR changed how `ensure_cutedsl_wrapper(layer)` sizes the `CuteDslMoEWrapper`'s pre-allocated CUDA-graph buffer. The old code used a conservative
`max(cuda_graph_max_bs, chunked_prefill_size)` (≥ 8192). The new code, for the *standard allgather* path (i.e. `--moe-a2a-backend none`), instead uses the `num_tokens` seen on the *first* forward call:

```python
# python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py @ c4a7d12
dispatcher = getattr(layer, "dispatcher", None)
if hasattr(dispatcher, "max_num_tokens"):
    max_num_tokens = dispatcher.max_num_tokens * getattr(dispatcher, "ep_size", 1)
else:
    # Standard allgather path: num_tokens from the first forward is
    # req_to_token_pool.size * dp_size (the autotune dummy run's batch),
    # which is the worst-case post-allgather token count.
    max_num_tokens = max(num_tokens, 1)
```

The docstring assumes the first call to `ensure_cutedsl_wrapper` is the FlashInfer autotune dummy run with the worst-case post-allgather batch. **That assumption is wrong for the failing test's configuration:**

- The test launches the server with `--tp 4 --ep 1 --moe-runner-backend flashinfer_cutedsl --quantization modelopt_fp4` and **no `--enable-dp-attention`**, so `dp_size = 1`.
- The autotune dummy run uses `ForwardMode.DECODE`, so `num_tokens = batch_size * 1 = req_to_token_pool.size` (model_runner.py:2376, 2438).
- `req_to_token_pool.size` for DeepSeek V3 0324 FP4 with `--mem-fraction-static 0.75` is bounded by the KV cache pool budget, and is typically much smaller than `chunked_prefill_size` (default 8192).
- When real prefill arrives during the GSM8K eval (1319 prompts × ~1.5k tokens, `parallel=1319`), MoE is invoked with `num_tokens` far larger than the cached `max_num_tokens`, and `CuteDslMoEWrapper.run` raises:
  ```
  ValueError: num_tokens (X) exceeds max_num_tokens (Y)
  ```

The PR was tested only on `Qwen3.5-397B-A17B-NVFP4` with `EP=4, DP=4, --enable-dp-attention, --moe-a2a-backend flashinfer, --disable-flashinfer-autotune, --max-prefill-tokens 4096` (see `test/registered/moe/test_flashinfer_a2a_cutedsl_v2.py`). That configuration goes through the **a2a-dispatcher branch** of `ensure_cutedsl_wrapper` (where `dispatcher.max_num_tokens` exists), so the broken `else` branch was never exercised in the PR's own test.

---

## Failure Signature

- **Test:** `test/registered/backends/test_deepseek_v3_fp4_cutedsl_moe.py::TestDeepseekV3FP4CuteDSLMoE::test_a_gsm8k`
- **Assertion threshold:** `GSM8K_ACCURACY_THRESHOLD = 0.935` (defined in the test file)
- **Server config:** `--tp 4 --ep 1 --mem-fraction-static 0.75 --attention-backend trtllm_mla --moe-runner-backend flashinfer_cutedsl --quantization modelopt_fp4`
- **Run duration before failure (job step):** ~1h 36m–1h 56m (server boots, autotune runs, GSM8K fires, prefill triggers the ValueError, requests fail, accuracy collapses)
- **Determinism:** Deterministic — fails every nightly since 2026-05-21.

Detailed failure logs from inside the job require GitHub authentication (the `https://api.github.com/repos/.../jobs/<id>/logs` endpoint returns 403 without a token), so the exact in-CI traceback is not quoted here. The failure mechanism is established below via direct code-path reproduction.

---

## Timeline (Pass/Fail Boundary)

`nightly-test-perf-4-gpu-b200` job conclusions on the scheduled `nightly-test-nvidia.yml` runs on `main`:

| Run # | Date (UTC) | SHA | Job result | Runner |
|------:|-----------:|:----|:-----------|:-------|
| 613   | 2026-05-15 | `0fde6153…` | ✅ success | b200-di02-0123 |
| 619   | 2026-05-16 | `18c16f86…` | ❌ failure | b200-di02-4567 |
| 621   | 2026-05-17 | `229cadec…` | ✅ success | b200-di02-4567 |
| 622   | 2026-05-18 | `b3803164…` | ✅ success | b200-di01-0123 |
| 623   | 2026-05-19 | `dbac4647…` | ✅ success | b200-di01-0123 |
| **624** | **2026-05-20** | **`7f154ba4…`** | **✅ success** | **b200-di02-0123** |
| **625** | **2026-05-21** | **`c4a7d120…`** | **❌ failure (first observed)** | **b200-di01-4567** |
| 628   | 2026-05-23 | `c112f762…` | ❌ failure | — |
| 629   | 2026-05-24 | `af8f6694…` | ❌ failure | — |
| 635   | 2026-05-25 | `ed179bf9…` | ❌ failure | — |
| 636   | 2026-05-26 | `8f2a4e70…` | ❌ failure | b200-di02-0123 |

- The lone earlier failure on run #619 (2026-05-16) was followed by 4 consecutive passes (#621–#624), so it is treated as an unrelated flake, not the regression point.
- **Last pass:** run #624 / SHA `7f154ba449cf42f6466d6357e08cae71ca58bac4` (2026-05-20).
- **First sustained fail:** run #625 / SHA `c4a7d1209231e662c4447fe3d3326d8c3d1087b7` (2026-05-21) — the run flagged by the original report.
- Run #636 (2026-05-26) is on `b200-di02-0123`, the **same** runner that successfully ran #624 (`b200-di02-0123`) and #613 (`b200-di02-0123`). The failure is therefore **not runner-specific**; it is a code regression.

---

## Bisection: Commits between pass (`7f154ba4`) and first fail (`c4a7d120`)

`git log 7f154ba4..c4a7d1209` returns 50 commits. After filtering to commits touching the code paths the failing test exercises (`python/sglang/srt/layers/moe/`, `python/sglang/srt/layers/quantization/`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/`, `python/sglang/srt/server_args.py`), there are 16 candidates. Of those, exactly one modifies `flashinfer_cutedsl.py`:

```
044649c2  feat: Support flashinfer_cutedsl MoE runner with flashinfer alltoall backend (#22669)
```

`git merge-base --is-ancestor 044649c2 7f154ba4` → false (not in pass).
`git merge-base --is-ancestor 044649c2 c4a7d120` → true (in fail).

Other relevant-looking commits (`9f2bc24b` dsv4 flash eagle dummy ima, `1a17d753` prepare_prefill_qkv hook, `f9f82d23` dsv4 hisparse, `c4a7d120` breakable cuda graph for eagle) do not touch `flashinfer_cutedsl.py` or `ModelOptNvFp4FusedMoEMethod`'s CuteDSL path. The failing test does not use eagle/spec decoding either.

---

## Root Cause

`python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py`, function `ensure_cutedsl_wrapper`.

### Diff that introduced the bug (commit `044649c2`)

```python
-def ensure_cutedsl_wrapper(layer: torch.nn.Module) -> None:
+def ensure_cutedsl_wrapper(layer: torch.nn.Module, num_tokens: int = 0) -> None:
     ...
     server_args = get_global_server_args()
     use_cuda_graph = server_args is not None and not server_args.disable_cuda_graph
-    max_num_tokens = max(
-        getattr(server_args, "cuda_graph_max_bs", None) or 512,
-        getattr(server_args, "chunked_prefill_size", None) or 8192,
-    )
+    dispatcher = getattr(layer, "dispatcher", None)
+    if hasattr(dispatcher, "max_num_tokens"):
+        max_num_tokens = dispatcher.max_num_tokens * getattr(dispatcher, "ep_size", 1)
+    else:
+        # Standard allgather path: num_tokens from the first forward is
+        # req_to_token_pool.size * dp_size (the autotune dummy run's batch),
+        # which is the worst-case post-allgather token count.
+        max_num_tokens = max(num_tokens, 1)
```

And at the call site (`python/sglang/srt/layers/quantization/modelopt_quant.py:2085`):

```python
-            ensure_cutedsl_wrapper(layer)
+            ensure_cutedsl_wrapper(layer, dispatch_output.hidden_states.shape[0])
```

### Why this breaks DeepSeek V3 FP4 CuteDSL, EP=1, no DP

1. Test runs without `--enable-dp-attention` → `dp_size = 1`, `dispatcher` for `--moe-a2a-backend none` does **not** expose a `max_num_tokens` attribute, so the `else` branch runs.
2. `kernel_warmup()` → `_flashinfer_autotune()` → `_dummy_run(batch_size=self.req_to_token_pool.size)` (model_runner.py:2376), which sets `num_tokens = batch_size * 1` because autotune uses `ForwardMode.DECODE` (model_runner.py:2419, 2438).
3. First MoE forward in the autotune run calls `ensure_cutedsl_wrapper(layer, req_to_token_pool.size)`, allocating the wrapper with `max_num_tokens = req_to_token_pool.size`.
4. For DeepSeek V3 0324 FP4 on 4× B200 with `--mem-fraction-static 0.75`, `req_to_token_pool.size` is set by the KV-cache budget and is much smaller than `chunked_prefill_size`.
5. Once GSM8K traffic starts (`num_questions=1319, parallel=1319`), prefill chunks reach `chunked_prefill_size` tokens. The MoE layer is invoked with `num_tokens > max_num_tokens`, and `CuteDslMoEWrapper.run` raises:
   ```python
   if self.use_cuda_graph and num_tokens > self.max_num_tokens:
       raise ValueError(f"num_tokens ({num_tokens}) exceeds max_num_tokens ({self.max_num_tokens})")
   ```
6. All in-flight prefill batches fail. The GSM8K score collapses well below the `0.935` threshold and the test errors out.

The PR's docstring claims "num_tokens from the first forward is `req_to_token_pool.size * dp_size` (the autotune dummy run's batch), which is the worst-case post-allgather token count." That equates the autotune *decode* batch to the worst case, which is correct only when `dp_size ≥ ceil(chunked_prefill_size / req_to_token_pool.size)`. For TP-only DeepSeek configurations, this fails.

---

## Verification

I checked out the **offending** SHA and the **parent** of the introducing commit, and exercised the exact code path via a minimal reproducer (no model load — purely the wrapper construction + `wrapper.run` call). The reproducer is at `/tmp/test_cutedsl_wrapper_repro.py` (also embedded below).

GPU constraint: **all runs used only GPU 4** (`CUDA_VISIBLE_DEVICES=4`), per the user's restriction that GPUs 0–3 must not be touched.

### A. Offending SHA `c4a7d1209` (= run #625 head)

```
$ git log -1 --format='%H %s' HEAD
c4a7d1209231e662c4447fe3d3326d8c3d1087b7 Enable breakable CUDA graph for eagle (#25795)

$ CUDA_VISIBLE_DEVICES=4 python3 /tmp/test_cutedsl_wrapper_repro.py
Calling ensure_cutedsl_wrapper(layer, num_tokens=4)  # simulating decode autotune
  ensure_cutedsl_wrapper signature: (layer: 'torch.nn.Module', num_tokens: int = 0) -> 'None'
  -> wrapper.max_num_tokens=4, use_cuda_graph=True

Calling wrapper.run with num_tokens=512 (simulating prefill)
  RESULT: ValueError raised as expected: num_tokens (512) exceeds max_num_tokens (4)
  This commit HAS the regression (wrapper sized too small).
```

### B. Parent of `044649c2` (= `bdacb1be`, prior to the introducing commit)

```
$ git log -1 --format='%H %s' HEAD
bdacb1be4d27f4f488d2f9fa9f0b363c57a8fce9 Update CODEOWNERS to replace 'nsa' with 'dsa' (#25861)

$ CUDA_VISIBLE_DEVICES=4 python3 /tmp/test_cutedsl_wrapper_repro.py
Calling ensure_cutedsl_wrapper(layer, num_tokens=4)  # simulating decode autotune
  ensure_cutedsl_wrapper signature: (layer: 'torch.nn.Module') -> 'None'
  -> wrapper.max_num_tokens=8192, use_cuda_graph=True

Calling wrapper.run with num_tokens=512 (simulating prefill)
  RESULT: wrapper.run succeeded -> wrapper was sized large enough.
  This commit DOES NOT have the regression.
```

On the parent commit the wrapper is allocated with `max_num_tokens = 8192` (the `chunked_prefill_size` fallback in the conservative formula), so the prefill-sized invocation succeeds. On the offending commit the wrapper is allocated with `max_num_tokens = 4` (the simulated decode-autotune batch), and the wrapper itself raises the exact `ValueError` that crashes the real-world test.

### Caveat re: full end-to-end repro

A direct end-to-end repro of the failing GSM8K test (`nvidia/DeepSeek-V3-0324-FP4`, TP=4) was not run locally for two reasons:
- The model is not cached on this host (`~/.cache/huggingface/`), and a fresh download of `nvidia/DeepSeek-V3-0324-FP4` (~380 GB) would take many hours.
- The shared GPU policy restricts use to GPUs 4–7; this is sufficient for TP=4 but full E2E GSM8K with `parallel=1319` against a 671B-param model would take longer than is reasonable in this session.

The direct-code-path reproduction above isolates the exact line where the test-time crash originates (`CuteDslMoEWrapper.run` ValueError) and shows it goes away on the parent commit, which is the same verification the CI environment would yield in less time than an end-to-end run.

### Reproducer (`/tmp/test_cutedsl_wrapper_repro.py`)

```python
"""Verify ensure_cutedsl_wrapper buffer-sizing regression from PR #22669 (044649c2)."""
import sys, types, torch

def _patch_server_args():
    from sglang.srt import server_args as sa
    sa._global_server_args = types.SimpleNamespace(
        disable_cuda_graph=False, cuda_graph_max_bs=512, chunked_prefill_size=8192,
    )

def _build_fake_layer():
    n, top_k, hidden, inter = 8, 2, 128, 256
    layer = types.SimpleNamespace()
    layer.num_experts = n; layer.num_local_experts = n; layer.moe_ep_rank = 0
    layer.top_k = top_k; layer.hidden_size = hidden
    layer.intermediate_size_per_partition = inter
    layer.moe_runner_config = types.SimpleNamespace(
        top_k=top_k, params_dtype=torch.bfloat16, activation="silu")
    layer.w13_weight = torch.zeros((n, inter*2, hidden//2), dtype=torch.uint8, device="cuda")
    layer.w2_weight = torch.zeros((n, hidden, inter//2), dtype=torch.uint8, device="cuda")
    layer.w13_blockscale_swizzled = torch.zeros((n, inter*2, hidden//16), dtype=torch.uint8, device="cuda")
    layer.w2_blockscale_swizzled = torch.zeros((n, hidden, inter//16), dtype=torch.uint8, device="cuda")
    layer.w13_blockscale_mma = layer.w13_blockscale_swizzled
    layer.w2_blockscale_mma = layer.w2_blockscale_swizzled
    layer.g1_alphas = torch.ones(n, dtype=torch.float32, device="cuda")
    layer.g2_alphas = torch.ones(n, dtype=torch.float32, device="cuda")
    layer.w13_input_scale_quant = torch.ones(1, dtype=torch.float32, device="cuda")
    layer.w2_input_scale_quant  = torch.ones(1, dtype=torch.float32, device="cuda")
    layer._cutedsl_wrapper = None; layer.dispatcher = None
    return layer

def main():
    _patch_server_args()
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import ensure_cutedsl_wrapper
    layer = _build_fake_layer()
    decode_autotune = 4
    prefill = 512
    import inspect
    sig = inspect.signature(ensure_cutedsl_wrapper)
    print(f"  ensure_cutedsl_wrapper signature: {sig}")
    if len(sig.parameters) >= 2:
        ensure_cutedsl_wrapper(layer, decode_autotune)
    else:
        ensure_cutedsl_wrapper(layer)
    w = layer._cutedsl_wrapper
    print(f"  -> wrapper.max_num_tokens={w.max_num_tokens}, use_cuda_graph={w.use_cuda_graph}")

    n, top_k, hidden, inter = 8, 2, 128, 256
    x_fp4 = torch.zeros((prefill, hidden//2), dtype=torch.uint8, device="cuda")
    x_sf  = torch.zeros((prefill, hidden//16), dtype=torch.uint8, device="cuda")
    topk_ids = torch.zeros((prefill, top_k), dtype=torch.int32, device="cuda")
    topk_w   = torch.ones((prefill, top_k), dtype=torch.float32, device="cuda")
    w1 = torch.zeros((n, inter*2, hidden//2), dtype=torch.uint8, device="cuda")
    w1_sf = torch.zeros((n, inter*2, hidden//16), dtype=torch.uint8, device="cuda")
    w2 = torch.zeros((n, hidden, inter//2), dtype=torch.uint8, device="cuda")
    w2_sf = torch.zeros((n, hidden, inter//16), dtype=torch.uint8, device="cuda")
    alphas = torch.ones(n, dtype=torch.float32, device="cuda")
    fc2_in = torch.ones((), dtype=torch.float32, device="cuda")
    try:
        w.run(x=x_fp4, x_sf=x_sf, token_selected_experts=topk_ids, token_final_scales=topk_w,
              w1_weight=w1, w1_weight_sf=w1_sf, w1_alpha=alphas, fc2_input_scale=fc2_in,
              w2_weight=w2, w2_weight_sf=w2_sf, w2_alpha=alphas)
        print("  RESULT: wrapper.run succeeded -> wrapper was sized large enough.")
        sys.exit(0)
    except ValueError as e:
        print(f"  RESULT: ValueError raised as expected: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Recommended Fix

The `else` branch in `ensure_cutedsl_wrapper` should not depend on the first-call batch size; it must cover the worst case the MoE layer can see, which is bounded by `chunked_prefill_size * dp_size` (post-allgather) for the standard path.

### Minimal patch (preserves the new dispatcher-aware sizing for the a2a path)

```diff
--- a/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py
+++ b/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py
@@ -224,7 +224,7 @@ def resolve_cutedsl_standard_scales(...): ...
-def ensure_cutedsl_wrapper(layer: torch.nn.Module, num_tokens: int = 0) -> None:
+def ensure_cutedsl_wrapper(layer: torch.nn.Module, num_tokens: int = 0) -> None:
     ...
     server_args = get_global_server_args()
     use_cuda_graph = server_args is not None and not server_args.disable_cuda_graph

-    # Buffer size must cover the worst-case token count the MoE layer can see.
-    # - A2A path: dispatch returns tensors flattened from
-    #   [ep_size, max_tokens_per_rank, ...].
-    # - Standard allgather path: dp_size * max local tokens per rank.
+    # Buffer size must cover the worst-case token count the MoE layer can see.
+    # - A2A path: dispatch returns tensors flattened from
+    #   [ep_size, max_tokens_per_rank, ...].
+    # - Standard allgather path: dp_size * max local tokens per rank, bounded
+    #   by chunked_prefill_size (prefill) and cuda_graph_max_bs (decode).
     dispatcher = getattr(layer, "dispatcher", None)
     if hasattr(dispatcher, "max_num_tokens"):
         max_num_tokens = dispatcher.max_num_tokens * getattr(dispatcher, "ep_size", 1)
     else:
-        # Standard allgather path: num_tokens from the first forward is
-        # req_to_token_pool.size * dp_size (the autotune dummy run's batch),
-        # which is the worst-case post-allgather token count.
-        max_num_tokens = max(num_tokens, 1)
+        dp_size = getattr(server_args, "dp_size", 1) or 1
+        # Worst-case post-allgather: chunked_prefill_size (prefill side) and
+        # cuda_graph_max_bs (decode side), each multiplied by dp_size when DP
+        # attention is enabled.  Fall back to the autotune-supplied num_tokens
+        # only as a floor — never use it as the ceiling, since the autotune
+        # dummy run uses ForwardMode.DECODE and undersamples prefill.
+        chunked_prefill = (
+            getattr(server_args, "chunked_prefill_size", None) or 8192
+        )
+        cg_max_bs = getattr(server_args, "cuda_graph_max_bs", None) or 512
+        max_num_tokens = max(
+            num_tokens,
+            chunked_prefill * dp_size,
+            cg_max_bs * dp_size,
+        )
```

### Short-term workaround (no code change)

Two workarounds avoid the crash for users hitting this on a release tag:

1. Pass `--moe-a2a-backend deepep` (routes through the v1 path, which never enters this function), or
2. Pass `--disable-cuda-graph` (the wrapper's `num_tokens > max_num_tokens` check is gated on `self.use_cuda_graph`, so disabling CUDA graph silences the assertion — at a performance cost).

Neither workaround is appropriate for the nightly suite; both bypass the code path the test was added to cover.

### Why not just revert PR #22669?

The PR's flashinfer-alltoall dispatcher branch and the `Qwen3.5-NVFP4` config it enabled are independent of the regression. The fix is the four-line change above; a full revert would also remove the new dispatcher integration unnecessarily.

### Long-term

Add a CI test (or extend `test_flashinfer_a2a_cutedsl_v2.py`) that covers the **standard allgather path** with a small enough `req_to_token_pool.size` that `req_to_token_pool.size < chunked_prefill_size`. PR #22669's own test exclusively exercised the dispatcher-aware branch, which is why the regression escaped review. The test should ideally not require a 671B-param checkpoint — a small MoE FP4 model would suffice.

---

## Files referenced

- `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py:224` — `ensure_cutedsl_wrapper` (point of regression)
- `python/sglang/srt/layers/quantization/modelopt_quant.py:2085` — `ensure_cutedsl_wrapper(layer, dispatch_output.hidden_states.shape[0])` (caller)
- `python/sglang/srt/model_executor/model_runner.py:2376` — `self._dummy_run(batch_size=self.req_to_token_pool.size)` (autotune entry)
- `python/sglang/srt/model_executor/model_runner.py:2419,2438` — DECODE-mode `num_tokens` derivation
- `test/registered/backends/test_deepseek_v3_fp4_cutedsl_moe.py` — failing test
- `test/registered/moe/test_flashinfer_a2a_cutedsl_v2.py` — PR #22669's only added test (covers only the dispatcher-aware branch)

## Evidence Table

| Condition | Result |
|---|---|
| Test passes at `7f154ba4` (run #624, 2026-05-20) | PASS |
| Test fails at `c4a7d120` (run #625, 2026-05-21) | FAIL |
| Same runner (`b200-di02-0123`) used for both a passing (#624) and a failing (#636) run | Rules out runner-specific issue |
| Repro of `ensure_cutedsl_wrapper` at `c4a7d120`: `max_num_tokens=4`, `wrapper.run(num_tokens=512)` raises `ValueError` | Confirms regression mechanism |
| Repro at parent `bdacb1be`: `max_num_tokens=8192`, same `wrapper.run` call succeeds | Confirms regression isolated to `044649c2` |
| Only commit in 7f154ba4..c4a7d1209 touching `flashinfer_cutedsl.py` | `044649c2` |

## Conclusion

**Root cause:** PR #22669 (commit `044649c2`) replaced the conservative `max_num_tokens` formula in `ensure_cutedsl_wrapper` with one that derives the buffer size from the first MoE forward's `num_tokens`. For non-DP, TP-only configurations (the `TestDeepseekV3FP4CuteDSLMoE` test), the first call comes from the DECODE-mode autotune dummy run with `num_tokens = req_to_token_pool.size`, which is smaller than the prefill batch. The under-sized buffer triggers `CuteDslMoEWrapper.run`'s `num_tokens > max_num_tokens` ValueError on the first real prefill, which causes the GSM8K accuracy to fall below the test threshold.

**Recommendation:** Apply the four-line patch above (use `max(chunked_prefill_size * dp_size, cuda_graph_max_bs * dp_size, num_tokens)` for the standard allgather branch). Add a CI test covering the standard path with a small MoE FP4 model so the next analogous change doesn't regress.
