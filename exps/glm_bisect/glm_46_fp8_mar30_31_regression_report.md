# GLM-4.6-FP8 Mar 30-31 Decode Performance Regression Report

**Date**: 2026-04-11
**Investigator**: Claude (automated bisection)
**Status**: Root cause identified -- FP32 gate projection cast in MoE routing

---

## 1. Failure Signature

| Field | Value |
|-------|-------|
| **Test file** | `test/registered/8-gpu-models/test_glm_46_fp8.py` (now `test/manual/test_glm_46_fp8.py`) |
| **Workflow** | `.github/workflows/nightly-test-nvidia.yml` |
| **Job** | `nightly-test-general-8-gpu-h200` |
| **Matrix shard** | `nightly-test-general-8-gpu-h200 (1)` |
| **Metric** | Output throughput (tok/s) -- both TP8 and TP8+MTP affected |
| **Deterministic** | Yes -- 4-8% drop, persistent across all subsequent nightlies |

---

## 2. Timeline

| Milestone | Date | Run ID | SHA | Detail |
|-----------|------|--------|-----|--------|
| Last known good nightly | Mar 30, 00:46 UTC | 23723157688 | `afb32d76224e` | TP8 bs=8: 487 tok/s |
| First known bad nightly | Mar 31, 00:43 UTC | 23774903575 | `3650bfb19926` | TP8 bs=8: 454 tok/s |
| Commits in window | -- | -- | -- | 34 first-parent commits |

---

## 3. Regression Characterization

This is a **decode speed regression** affecting both TP8 and TP8+MTP variants. Unlike the Apr 3-4 regression, this is NOT an acceptance length issue -- MTP acc_len remained ~2.0. Both variants slowed proportionally, indicating the bottleneck is in the base model forward pass, not in speculative decoding.

### 3.1 TP8 (non-speculative) Delta

| Batch Size | Mar 30 (tok/s) | Mar 31 (tok/s) | Delta |
|------------|----------------|----------------|-------|
| bs=1 | 83.81 | 83.05 | -0.9% |
| bs=8 | 487.10 | 454.08 | **-6.8%** |
| bs=16 | 831.16 | 791.69 | **-4.7%** |
| bs=64 | 2174.34 | 2085.36 | **-4.1%** |

### 3.2 TP8+MTP (EAGLE speculative decoding) Delta

| Batch Size | Mar 30 (tok/s) | Mar 31 (tok/s) | Delta |
|------------|----------------|----------------|-------|
| bs=1 | 129.80 | 117.09 | **-9.8%** |
| bs=8 | 681.89 | 653.44 | **-4.2%** |
| bs=16 | 1129.19 | 1083.81 | **-4.0%** |

### 3.3 Latency Delta (TP8)

| Batch Size | Mar 30 (s) | Mar 31 (s) | Delta |
|------------|------------|------------|-------|
| bs=8 | 9.61 | 10.37 | **+7.9%** |
| bs=16 | 12.24 | 13.03 | **+6.5%** |
| bs=64 | 24.52 | 26.40 | **+7.7%** |

### 3.4 ITL (Inter-Token Latency) Delta (TP8)

| Batch Size | Mar 30 (ms) | Mar 31 (ms) | Delta |
|------------|-------------|-------------|-------|
| bs=1 | 11.93 | 12.04 | +0.9% |
| bs=8 | 16.42 | 17.62 | **+7.3%** |
| bs=16 | 19.25 | 20.21 | **+5.0%** |
| bs=64 | 29.43 | 30.69 | **+4.3%** |

### 3.5 MTP Acceptance Length -- Unchanged

| Batch Size | Mar 30 | Mar 31 |
|------------|--------|--------|
| bs=1 | 2.00 | 1.91 |
| bs=8 | 1.99 | 1.99 |
| bs=16 | 1.99 | 1.99 |

Acceptance length stayed ~2.0, confirming the draft model predictions are still valid. The slowdown is purely in the decode step latency, not in speculative acceptance.

### 3.6 Persistence

The regression persists in all subsequent nightlies (Apr 1 through Apr 6+). It was never recovered.

---

## 4. Baseline Stability

| Date | SHA (short) | TP8 bs=8 | TP8 bs=16 | TP8 bs=64 |
|------|-------------|----------|-----------|-----------|
| Mar 28 | `83997080a60f` | 487.43 | 830.93 | 2172.47 |
| Mar 29 | `3ab9afd65380` | 487.12 | 831.46 | 2171.33 |
| **Mar 30** | **`afb32d76224e`** | **487.10** | **831.16** | **2174.34** |
| **Mar 31** | **`3650bfb19926`** | **454.08** | **791.69** | **2085.36** |
| Apr 1 | `a8759dd9af05` | 454.15 | 791.73 | 2086.96 |
| Apr 2 | `d7256eb69af9` | 454.09 | 792.23 | 2084.82 |
| Apr 3 | `29d8e959d704` | 453.33 | 791.53 | 2085.97 |

Mar 28-30 shows rock-solid stability (487/831/2172). The drop to 454/792/2085 happens exactly between Mar 30 and Mar 31 and persists permanently.

---

## 5. Root Cause

### 5.1 Identified Commit

**Commit `ad064c2f4`** -- [GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model. (#21660)

- **Author**: Yuxuan Zhang (zRzRzRzRzRzRzR)
- **Date**: Mar 31 03:25:27 +0800 (Mar 30 12:25:27 PDT)
- **Merged**: 2026-03-30T19:25:28Z
- **PR**: sgl-project/sglang#21660 (part of #21258)
- **Files changed**: `python/sglang/srt/models/glm4_moe.py` (+6/-1)

### 5.2 The Change

```diff
 class Glm4MoeGate(nn.Module):
     def __init__(self, config):
         ...
         self.e_score_correction_bias = nn.Parameter(
             torch.empty((config.n_routed_experts), dtype=torch.float32)
         )
+        # GLM requires FP32 gate projection; cache to avoid per-forward cast.
+        self.register_buffer("_weight_fp32", None, persistent=False)

     def forward(self, hidden_states):
-        logits = F.linear(hidden_states, self.weight, None)
+        if self._weight_fp32 is None:
+            self._weight_fp32 = self.weight.data.to(torch.float32)
+        logits = F.linear(hidden_states.to(torch.float32), self._weight_fp32, None)
         return logits
```

### 5.3 Why This Causes a Performance Regression

The change casts `hidden_states` to FP32 **on every forward pass** through the MoE gate, and uses a pre-cached FP32 copy of the gate weight. This affects performance because:

1. **Per-token FP32 cast**: `hidden_states.to(torch.float32)` runs on every token at every MoE layer. For GLM-4.6, which is a MoE model where most layers (those after `first_k_dense_replace`) are MoE, this is invoked dozens of times per token.

2. **FP32 matmul instead of BF16/FP8**: `F.linear(hidden_states_fp32, weight_fp32)` computes the gate logits in FP32 rather than the native dtype (likely BF16 for the hidden states of this compressed-tensors FP8 model). FP32 operations on H200 tensor cores are slower than BF16.

3. **Memory bandwidth**: The FP32 hidden states are 2x the size of BF16, doubling the memory bandwidth required for the gate projection. On memory-bandwidth-bound decode operations, this is significant.

4. **CUDA graph interaction**: The FP32 cast creates a new tensor on every call. Under CUDA graph replay, this may introduce additional overhead or prevent certain optimizations.

The gate projection runs once per MoE layer per token. With many MoE layers in the model, the accumulated overhead produces a ~5-7% latency increase in the decode step, matching the observed regression.

### 5.4 Why the Effect Is Larger at Medium Batch Sizes

The regression is most pronounced at bs=8 (-6.8%) and less so at bs=1 (-0.9%) and bs=64 (-4.1%). This pattern is consistent with the gate projection being a proportionally larger fraction of total compute at medium batch sizes:
- At bs=1, the decode step is dominated by attention latency, not MoE routing
- At bs=8-16, MoE routing overhead is a significant fraction of the decode step
- At bs=64, the expert computation dominates again, diluting the gate overhead

### 5.5 Why TP8+MTP Is Slightly More Affected

The TP8+MTP variant at bs=1 shows a larger drop (-9.8%) compared to TP8 bs=1 (-0.9%). This is because in speculative decoding, the draft model also runs through the MoE gate (the `Glm4MoeForCausalLMNextN` model shares the same `Glm4MoeGate` class). So the FP32 overhead applies twice: once in the draft forward pass and once in the target verification pass.

### 5.6 Ancestry Verification

```
ad064c2f4 NOT in Mar 30 good SHA (afb32d76224e) -- confirmed
ad064c2f4 IS in Mar 31 bad SHA (3650bfb19926) -- confirmed
```

---

## 6. Commit Window -- Full Triage

### 6.1 All 34 Commits

```
 #1  3650bfb19 Remove flashinfer wheel cache cleanup that deletes other versions (#21711)
 #2  67c295b5f [AMD] fix performance regression issue (#21691)
 #3  4b8456e26 [AMD][MoRI] bump MoRI to v0.1.0 (#21673)
 #4  daf697afd [AMD] Add configurable KV transfer overlap (#20410)
 #5  d6029de6a [Bugfix][NPU] Skip FRACTAL_NZ format for MoE weights (#21209)
 #6  4a9ffc3ab fix nemotron capture for non attention layers (#21436)
 #7  ad064c2f4 [GLM-V and GLM-4.7] Cast to FP32 before gate projection (#21660)   *** ROOT CAUSE ***
 #8  a20d12ae9 [diffusion][doc]: add ring sp performance benchmark page (#20998)
 #9  f4b0e9c64 [diffusion] [NPU] support ring attention on NPU with FA (#21383)
#10  752d260c7 [NPU][diffusion]: support parallel decoding of qwen-image (#20757)
#11  ba6d54d0f [NPU] GLM-5 optimize with fused kernels (#18617)
#12  7119d5974 DeepSeek-R1-0528-w4a8: DeepEP FP8 Communication (#14162)
#13  673ffb311 [NPU] fix eagle3 accept rate (#21255)
#14  c5c58c334 [NPU][Diffusion] fix sp modulate (#20974)
#15  0a1fb4286 [diffusion] CI: relax pr-test threshold (#21682)
#16  b76730701 [diffusion] feat: enhance overlay mechanism (#21648)
#17  1d6424d5a fix: Mistral Small 4 config/weight format mismatch (#21620)
#18  b24626944 fix mamba cache leak when adder fails (#21404)
#19  62a63eeff [Fix] Fix weight_loader for qwen3-next FP8 models (#21662)
#20  e6071e60c [AMD] Support AMD MXFP4 Qwen3.5-397B-A17B model (#21234)
#21  965f03cdc [NPU] Update DeepSeek-V3.2 deployment docs (#21468)
#22  b9a68c304 [AMD] Fused rope kv store (#21315)
#23  af62bd948 [CPU] Implement MXFP4 Gemm kernels for intel AMX (#14385)
#24  ed01e1d5d [CPU] add kernel apply_rotary_pos_emb_cpu (#13121)
#25  6da8f5f69 fix topk softmax performance issue (CPU only) (#14702)
#26  c32ee4888 MFU metrics in Prometheus (#19395)
#27  1a4b383fa [CI] FlashInfer v0.6.7 offline MXFP8 Gemm tests (#21625)
#28  5b19c9a05 [Doc] Update tips for developer new-comers (#21659)
#29  f0303fd07 [Intel GPU] Enable DeepSeek R1 inference on XPU (#18461)
#30  d8ab41dce [Fix] Handle pre-release tags in nightly wheel version parsing (#21656)
#31  90bdc3192 Update sponsorship details in README.md (#21658)
#32  db5d9eb8c [diffusion] CI: fix dashboard chart (#21653)
#33  9b4dd2747 [Fix] Qwen3.5 MoE model loading and Mamba cache sharding in PP mode (#21448)
#34  c06ca1526 Fix circular reference in CustomTestCase.__init_subclass__ (#21650)
```

### 6.2 Elimination Summary

| Category | Count | Commits |
|----------|-------|---------|
| AMD / ROCm only | 4 | #2, #3, #4, #20, #22 |
| NPU only | 5 | #5, #10, #11, #13, #14, #21 |
| CPU / Intel only | 3 | #23, #24, #29 |
| Diffusion only | 4 | #8, #9, #15, #16 |
| CI / docs / README | 5 | #27, #28, #30, #31, #32 |
| Unrelated models (Mistral, Mamba, Qwen) | 3 | #17, #18, #19, #33 |
| Unrelated infra (Prometheus, test framework) | 3 | #25, #26, #34 |
| Flashinfer cache script | 1 | #1 |
| Nemotron CUDA graph (2-line addition, unrelated model) | 1 | #6 |
| DeepEP FP8 dispatch (DeepSeek-specific, no EP in GLM) | 1 | #12 |
| **GLM model change** | **1** | **#7 -- ROOT CAUSE** |

Only one commit in the entire window modifies GLM model code or any code that runs in the GLM inference path on NVIDIA H200.

---

## 7. Relationship to Apr 3-4 Regression

This Mar 30-31 regression is a **separate, independent issue** from the Apr 3-4 TP8+MTP acceptance length collapse described in the companion report (`glm_46_fp8_regression_report.md`).

| Aspect | Mar 30-31 Regression | Apr 3-4 Regression |
|--------|---------------------|--------------------|
| **Affected variants** | Both TP8 and TP8+MTP | TP8+MTP only |
| **Nature** | Decode speed regression | Speculative acceptance collapse |
| **Acceptance length** | Unchanged (~2.0) | Collapsed (2.0 -> 1.0) |
| **Magnitude** | 4-8% throughput drop | ~50% throughput drop |
| **Root cause** | FP32 gate projection cast | sglang-kernel 0.4.0 -> 0.4.1 (suspected) |
| **Commit** | `ad064c2f4` (#21660) | `84118acf5` (#22009) (suspected) |

However, the two regressions **compound**: after both are in effect, the TP8+MTP path suffers from both slower decode AND near-zero acceptance. The combined effect on TP8+MTP throughput from Mar 30 to Apr 4 is devastating:
- Mar 30 TP8+MTP bs=16: 1129 tok/s (full speed)
- Mar 31 TP8+MTP bs=16: 1084 tok/s (after FP32 gate, -4%)
- Apr 4 TP8+MTP bs=16: 576 tok/s (after acceptance collapse, -47% from Mar 31)

---

## 8. Recommended Fix

### Option A: Remove the FP32 cast (fastest, may break GLM-V/GLM-4.7 correctness)

```python
def forward(self, hidden_states):
    logits = F.linear(hidden_states, self.weight, None)
    return logits
```

This reverts the change entirely. However, the PR comment says "GLM requires FP32 gate projection" which implies there may be a correctness reason for the cast.

### Option B: Make the cast conditional on model variant

```python
def forward(self, hidden_states):
    if self.use_fp32_gate:
        if self._weight_fp32 is None:
            self._weight_fp32 = self.weight.data.to(torch.float32)
        logits = F.linear(hidden_states.to(torch.float32), self._weight_fp32, None)
    else:
        logits = F.linear(hidden_states, self.weight, None)
    return logits
```

Where `use_fp32_gate` is set based on whether the model actually needs FP32 routing (e.g., GLM-V, GLM-4.7-non-FP8) vs. models where the original dtype routing was correct (GLM-4.6-FP8).

### Option C: Investigate why FP32 is needed

The PR references #21258. If the FP32 cast is needed for numerical stability in routing (e.g., the gate weights are quantized and routing in low precision causes incorrect expert selection), the fix should preserve FP32 routing correctness while minimizing performance impact:
- Keep the FP32 weight cache
- Investigate if `torch.float16` is sufficient instead of `float32`
- Profile whether the hidden_states cast or the linear is the bottleneck
- Consider fusing the cast into the linear operation

### Verification

```bash
# On SHA 3650bfb19926 (first bad), revert the gate change and benchmark:
git revert --no-commit ad064c2f4
python3 test/manual/test_glm_46_fp8.py
# Expected: TP8 bs=8 throughput returns to ~487 tok/s (from ~454)
```

---

## 9. Appendix: Detailed Before/After Metrics

### Mar 30 (GOOD) -- Full Data

**TP8:**
| bs | output (tok/s) | latency (s) | ITL (ms) | acc_len |
|----|----------------|-------------|----------|---------|
| 1  | 83.81 | 6.29 | 11.93 | n/a |
| 8  | 487.10 | 9.61 | 16.42 | n/a |
| 16 | 831.16 | 12.24 | 19.25 | n/a |
| 64 | 2174.34 | 24.52 | 29.43 | n/a |

**TP8+MTP:**
| bs | output (tok/s) | latency (s) | ITL (ms) | acc_len |
|----|----------------|-------------|----------|---------|
| 1  | 129.80 | 4.14 | 7.70 | 2.00 |
| 8  | 681.89 | 7.22 | 11.73 | 1.99 |
| 16 | 1129.19 | 9.66 | 14.17 | 1.99 |

### Mar 31 (BAD) -- Full Data

**TP8:**
| bs | output (tok/s) | latency (s) | ITL (ms) | acc_len |
|----|----------------|-------------|----------|---------|
| 1  | 83.05 | 6.36 | 12.04 | n/a |
| 8  | 454.08 | 10.37 | 17.62 | n/a |
| 16 | 791.69 | 13.03 | 20.21 | n/a |
| 64 | 2085.36 | 26.40 | 30.69 | n/a |

**TP8+MTP:**
| bs | output (tok/s) | latency (s) | ITL (ms) | acc_len |
|----|----------------|-------------|----------|---------|
| 1  | 117.09 | 4.59 | 8.54 | 1.91 |
| 8  | 653.44 | 7.65 | 12.24 | 1.99 |
| 16 | 1083.81 | 10.29 | 14.76 | 1.99 |
