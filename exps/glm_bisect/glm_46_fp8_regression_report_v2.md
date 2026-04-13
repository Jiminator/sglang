# GLM-4.6-FP8 TP8+MTP Nightly CI Performance Regression Report (v2)

**Date**: 2026-04-11
**Investigator**: Claude (automated bisection) + user local experiments
**Status**: v1 kernel-bump hypothesis **disproven** via local experiments. Root cause narrowed to a 43-commit window. New primary suspect identified.
**Supersedes**: `glm_46_fp8_regression_report.md` (v1)

---

## 0. Changelog From v1

| Change | Rationale |
|--------|-----------|
| **v1 primary suspect `84118acf5` (sglang-kernel 0.4.0 -> 0.4.1 bump) ruled out** | User's local experiments show identical TP8+MTP throughput at `84118acf5` (559.54 tok/s) vs its parent `84118acf5~1 = eb407b80f` (559.44 tok/s). Both are equally bad. The kernel version change is not the cause. |
| **Window narrowed from 51 commits to 43 commits** | User's `84118acf5~1` experiment shows `eb407b80f` is already BAD. So commits #1-#8 of the original list (including the kernel bump) are cleared. |
| **New primary suspect: `ad0516d9c` [NPU] optimize glm4.7 (#19246)** | Directly modifies `glm4_moe_nextn.py` -- the GLM EAGLE draft model. Previously flagged but dismissed; this re-evaluation is prompted by the fact that the sglang source code must contain the regression (not the kernel binary). Also happens to be the **midpoint** of the new 43-commit window. |
| **Evidence section expanded with local reproduction data** | User collected 10-trial means at 4 checkpoints (`ad064c2f4~1`, `ad064c2f4`, `84118acf5~1`, `84118acf5`), matching the nightly observations with high precision. |
| **New bisection plan added** | Concrete commands for the user to narrow the window further with `git bisect`-style tests. |

---

## 1. Failure Signature (Unchanged)

| Field | Value |
|-------|-------|
| **Test file** | `test/registered/8-gpu-models/test_glm_46_fp8.py` (now moved to `test/manual/test_glm_46_fp8.py`) |
| **Workflow** | `.github/workflows/nightly-test-nvidia.yml` |
| **Job** | `nightly-test-general-8-gpu-h200` |
| **Matrix shard** | `nightly-test-general-8-gpu-h200 (1)` |
| **Table headers** | `### zai-org/GLM-4.6-FP8 (TP8) [8-gpu-h200] (TP8)` and `### zai-org/GLM-4.6-FP8 (TP8+MTP) [8-gpu-h200] (TP8+MTP)` |
| **Metric** | Output throughput (tok/s) and speculative acceptance length |
| **Deterministic** | Yes -- consistent and dramatic (~50% throughput drop) |

---

## 2. Test Configuration (Unchanged)

### TP8 (baseline, no speculative decoding)
```
sglang serve --model-path zai-org/GLM-4.6-FP8 --tp=8 --trust-remote-code
```

### TP8+MTP (EAGLE speculative decoding v2)
```
sglang serve --model-path zai-org/GLM-4.6-FP8 --tp=8 --trust-remote-code \
  --speculative-algorithm=EAGLE \
  --speculative-num-steps=3 \
  --speculative-eagle-topk=1 \
  --speculative-num-draft-tokens=4
```
With environment: `SGLANG_ENABLE_SPEC_V2=1`

Benchmarked with `bench_one_batch_server` at batch sizes 1, 8, 16, 64 with input_len=4096, output_len=512.

---

## 3. Updated Timeline

| Milestone | Identifier | Source | Status |
|-----------|-----------|--------|--------|
| Baseline before Mar 30-31 regression | `ad064c2f4~1` | Local 10-trial | GOOD (TP8 bs=8 = 476.02, TP8+MTP bs=16 = 1053.98) |
| After GLM FP32 gate cast (Mar 30-31 regression) | `ad064c2f4` | Local 10-trial | GOOD (TP8 bs=8 = 442.49, TP8+MTP bs=16 = 1010.72) |
| Last known good nightly | `29d8e959d704` (Apr 3) | Nightly run 23928679336 | GOOD (TP8+MTP bs=16 = 1022.93) |
| **Regression introduced somewhere in this gap** | | | |
| 84118acf5~1 = eb407b80f | `eb407b80f` (#1 of new window) | Local 10-trial | **BAD** (TP8+MTP bs=16 = 559.44) |
| After sglang-kernel 0.4.0 -> 0.4.1 bump | `84118acf5` | Local 10-trial | BAD (TP8+MTP bs=16 = 559.54) -- same as parent, kernel bump is innocent |
| First known bad nightly | `95cdbce34fa9` (Apr 4) | Nightly run 23967509528 | BAD (TP8+MTP bs=16 = 576.19) |

**Narrowed window**: 43 first-parent commits between `29d8e959d704` (Apr 3 good nightly, exclusive) and `eb407b80f` (tested BAD, inclusive).

---

## 4. Evidence

### 4.1 Local Reproduction Matrix (10 trials per checkpoint)

| Checkpoint | TP8 bs=1 | TP8 bs=8 | TP8 bs=16 | TP8 bs=64 | TP8+MTP bs=1 | TP8+MTP bs=8 | **TP8+MTP bs=16** | State |
|------------|----------|----------|-----------|-----------|--------------|--------------|-------------------|-------|
| `ad064c2f4~1` | 80.75 | 476.02 | 810.88 | 2139.89 | 114.70 | 645.00 | **1053.98** | GOOD (pre-FP32-gate, pre-MTP-cliff) |
| `ad064c2f4` | 80.56 | 442.49 | 768.66 | 2044.48 | 108.42 | 607.31 | **1010.72** | GOOD (FP32-gate present -> TP8 drop, MTP still fine) |
| `eb407b80f` (=`84118acf5~1`) | 80.75 | 442.75 | 766.41 | 2048.66 | 60.37 | 339.36 | **559.44** | **BAD** (MTP cliff present) |
| `84118acf5` (kernel bump) | 80.79 | 442.78 | 766.81 | 2048.80 | 60.62 | 339.90 | **559.54** | **BAD** (identical to parent; kernel bump has no effect) |

### 4.2 What This Rules Out

**Kernel bump `84118acf5` (v1 primary suspect)**: DISPROVEN.
- `eb407b80f` (pre-kernel-bump) and `84118acf5` (post-kernel-bump) show statistically identical throughput: 559.44 vs 559.54 tok/s at TP8+MTP bs=16 (<0.02% difference, within noise).
- The diff between these two commits is ONLY the `sglang-kernel==0.4.0` -> `sglang-kernel==0.4.1` line in `python/pyproject.toml`.
- Furthermore, the two experiments actually installed different kernel versions:
  - `eb407b80f` pip-installed `sglang-kernel==0.4.0` (from its pyproject.toml)
  - `84118acf5` pip-installed `sglang-kernel==0.4.1` (from its pyproject.toml)
  - Both produced the same bad throughput -- the kernel binary itself is not the cause.
- **Conclusion**: The regression is in the sglang source code, not in the kernel binary.

**Mar 30-31 TP8 regression (`ad064c2f4`)**: CONFIRMED but ORTHOGONAL to the Apr 3-4 cliff.
- `ad064c2f4~1` vs `ad064c2f4` shows TP8 bs=8 dropping from 476 -> 442 (~7%), but MTP bs=16 stays at ~1050-1010 tok/s. This matches the observation in `glm_46_fp8_mar30_31_regression_report.md`.
- The FP32 gate cast is real, but it does NOT explain the MTP acceptance cliff.

### 4.3 Local Data Matches Nightly Data

| Source | SHA | TP8+MTP bs=16 (tok/s) |
|--------|-----|------------------------|
| Nightly Mar 31 | `3650bfb19926` (with `ad064c2f4`) | 1083.81 |
| Local (this study) | `ad064c2f4` | 1010.72 |
| Nightly Apr 3 | `29d8e959d704` (with `ad064c2f4`, before MTP regression) | 1022.93 |
| Nightly Apr 4 | `95cdbce34fa9` (with MTP regression) | 576.19 |
| Local (this study) | `eb407b80f` | 559.44 |
| Local (this study) | `84118acf5` | 559.54 |

The local reproduction matches the nightly observations within typical run-to-run variance, confirming the experimental methodology and confirming that `eb407b80f` is past the regression boundary.

### 4.4 Historical Nightly History (Unchanged from v1, Included for Completeness)

#### TP8 (non-speculative) -- stable through entire Apr 3-4 window

| Date | SHA (short) | Run ID | bs=1 | bs=8 | bs=16 | bs=64 |
|------|-------------|--------|------|------|-------|-------|
| Mar 29 | `3ab9afd65380` | 23697967641 | 83.88 | 487.12 | 831.46 | 2171.33 |
| Mar 30 | `afb32d76224e` | 23723157688 | 83.81 | 487.10 | 831.16 | 2174.34 |
| Mar 31 | `3650bfb19926` | 23774903575 | 83.05 | 454.08 | 791.69 | 2085.36 |
| Apr 1  | `a8759dd9af05` | 23826444093 | 83.12 | 454.15 | 791.73 | 2086.96 |
| Apr 2  | `d7256eb69af9` | 23877953915 | 83.16 | 454.09 | 792.23 | 2084.82 |
| Apr 3  | `29d8e959d704` | 23928679336 | 83.29 | 453.33 | 791.53 | 2085.97 |
| **Apr 4** | **`95cdbce34fa9`** | **23967509528** | **83.12** | **453.32** | **790.94** | **2087.12** |
| Apr 5  | `70658bfeb52a` | 23991022988 | 83.10 | 453.30 | 791.14 | 2086.84 |
| Apr 6  | `93109cc89be3` | 24014261122 | 83.06 | 453.24 | 790.40 | 2086.24 |

#### TP8+MTP (EAGLE) -- cliff between Apr 3 and Apr 4

| Date | SHA (short) | bs=1 | bs=8 | **bs=16** | acc_len (bs=16) | PASS/FAIL |
|------|-------------|------|------|-----------|-----------------|-----------|
| Mar 29 | `3ab9afd65380` | 124.78 | 671.71 | 1114.15 | ~2.0 | PASS |
| Mar 30 | `afb32d76224e` | 129.80 | 681.89 | 1129.19 | ~2.0 | PASS |
| Mar 31 | `3650bfb19926` | 117.09 | 653.44 | 1083.81 | ~2.0 | PASS |
| Apr 1  | `a8759dd9af05` | 117.55 | 645.06 | 1068.21 | ~2.0 | PASS |
| Apr 2  | `d7256eb69af9` | 115.50 | 647.93 | 1071.46 | ~2.0 | PASS |
| Apr 3  | `29d8e959d704` | 121.61 | 633.78 | 1022.93 | 1.96 | PASS |
| **Apr 4** | **`95cdbce34fa9`** | **62.25** | **350.15** | **576.19** | **1.01** | **FAIL** |
| Apr 5  | `70658bfeb52a` | 62.28 | 354.32 | 576.30 | ~1.0 | FAIL |
| Apr 6  | `93109cc89be3` | 65.77 | 355.05 | 575.35 | ~1.0 | FAIL |

**Key signal**: Acceptance length collapses from 1.96 to 1.01. This indicates the draft model's outputs no longer match the target model's outputs. The system effectively falls back to sequential generation.

---

## 5. New Narrowed Window -- 43 Commits

Commits between `29d8e959d704` (exclusive, Apr 3 good nightly) and `eb407b80f` (inclusive, tested BAD). Numbered newest-to-oldest in this window (i.e., #1 is closest to BAD, #43 is closest to GOOD):

```
 #1  eb407b80f [Kernel] Make FA3/FA4 imports lazy in FlashAttentionBackend (#22028)   [known BAD -- boundary]
 #2  6aafe756b Revert "[Feature] NVFP4 Marlin fallback for non-Blackwell GPUs"
 #3  0c9dc098e Fix DP attention worker port binding for IPv6 support (#21917)
 #4  ed3435e37 [HiSparse]: Optimize server args checking (#22065)
 #5  151f72716 [diffusion] fix: fix gated repo failing the generate cmd (#22040)
 #6  896ea7582 Remove reverted test (#22058)
 #7  47f4fd275 [CI] Fix test suite names and add suite validation (#21937)
 #8  44e5d3570 [Feature][JIT Kernel] JIT activation and update skills (by codex) (#21766)
 #9  030fb1c4b refactor: replace mm_inputs dict with MultimodalProcessorOutput (#21738)
#10  9f409d074 [CI] Adjust CI server launch timeout (#22045)
#11  ee9d922f5 Revert "[Kernel] Fuse temperature + softmax in sampling" (#22046)
#12  97adf8a29 [misc] Add hint for kernel release trigger (#22036)
#13  98ac40192 [Workflow] Fix kernel release build failures (#22018)
#14  838f815e9 [diffusion] CI: temporarily disable accuracy ci (#22031)
#15  56ac9c993 [Fix] Add _MOE_TP to graph_capture for MoE models with ep>1 (#21907)
#16  ac593fed9 [AMD][Dockerfile] Support build-arg AITER_COMMIT (#21949)
#17  cd75d54fc [Bugfix] Fix CUDA graph replay issues in trtllm_mla draft_extend (#21987)
#18  4f84ce580 [CI] ci: add test_http_server_auth.py to CI (#21866)
#19  658a2813d [NPU] Update CI Dependency (#21578)
#20  d07d0a15c [AMD] Add MiniMax-M2.5 nightly perf benchmarks (#21524)
#21  7431db739 [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA (#21511)
#22  ad0516d9c [NPU] optimize glm4.7 (#19246)                            *** NEW PRIMARY SUSPECT / WINDOW MIDPOINT ***
#23  d82097a0d [PD] Tiny register info field cleanup for mooncake backend (#22016)
#24  24f52e66d fix: remove duplicate words in comments (#22007)
#25  4cc970290 [CI] Fix duplicate job names that bypass branch protection (#22001)
#26  6b876a771 [ROCM][RL] Shuffle Weight In-Place to Preserve Parameter Attributes (#21825)
#27  75de47968 [Misc] Update CI permission (#22014)
#28  4d097047f [PD]: HiSparse support for cache transfer (#21591)
#29  5c082c307 [Workflow] Fix kernel release jobs skipped on push events (#22011)
#30  0a709cfe0 [Workflow] Avoid triggering nightly tests in kernel bump (#22010)
#31  2c4fb8892 chore: bump sgl-kernel version to 0.4.1 (source version bump only) (#21447)
#32  5bcbc9757 [AMD] Resolve performance degression with aiter-allreduce-fusion (#21947)
#33  d1b7c3907 [Parallel State Refactor 2/n] Unify AMD deterministic all reduce (#20871)
#34  81efcc353 [NPU] Optimized the wording in the npu docs (#21998)
#35  efa7b2d5d Revert "[MUSA] Add FA3 attention backend support" (#22002)
#36  5f0df1e2a [Bugfix] Fix incorrect dp-attention parallel info in bench_one_batch (#21519)
#37  69e89a1fc [VLM] Enable per-image MM splitting by default and remove MULTI_IMAGES (#21899)
#38  8897ac58f [PP] qwen3 vl skip layer id for pp (#19135)
#39  991f3aa5b [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (#19652)
#40  2b5aed94f Remove maxItems=1 restriction when tool_choice is specified (#20208)
#41  0539c62bc [Diffusion][NPU] Add support for MOVA (#21633)
#42  1f97714f9 [CI] Add timeouts to Slack upload urlopen and WebClient (#21903)
#43  89affff29 Skip broken AutoModel mapping entries for Llava submodules (#21892)
```

---

## 6. Updated Suspect Ranking

### 6.1 HIGH PRIORITY

#### Suspect A: `ad0516d9c` [NPU] optimize glm4.7 (#19246) -- #22 in window (midpoint)

**Why this is now the top suspect**:
- Directly modifies `python/sglang/srt/models/glm4_moe_nextn.py` -- this is THE GLM-4.6 EAGLE draft model
- Also modifies `python/sglang/srt/models/glm4_moe.py` -- the base GLM model (though NVIDIA-path changes are `_is_npu`-gated)
- Happens to be the **exact midpoint** of the 43-commit window -- testing it gives both a direct-hypothesis check AND an efficient bisection step
- In v1 I dismissed this too quickly by assuming `self.quantization` would be `"fp8"` (making `needs_quant_draft` truthy); the server args logs actually show `quantization=None` for GLM-4.6-FP8, making `needs_quant_draft = None` (falsy) -- so the draft model is loaded with `quant_config = None`

**The suspicious change (in `glm4_moe_nextn.py`)**:
```python
# BEFORE:
self.quant_config = quant_config  # <CompressedTensorsConfig> for GLM-4.6-FP8
self.model = Glm4MoeModelNextN(config, quant_config, prefix=add_prefix("model", prefix))

# AFTER:
self.needs_quant_draft = (
    get_global_server_args().speculative_draft_model_quantization
)
quant_config = quant_config if self.needs_quant_draft else None
self.model = Glm4MoeModelNextN(config, quant_config, prefix=add_prefix("model", prefix))
```

And in `forward()`:
```python
# BEFORE:
hidden_states = self.model(input_ids, positions, forward_batch)

# AFTER:
if self.needs_quant_draft:
    cxt = contextlib.nullcontext()
else:
    unquant_patch = {
        "SGLANG_DEEPEP_BF16_DISPATCH": "1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "0",
    }
    cxt = temp_set_env(allow_sglang=True, **unquant_patch)
with cxt:
    hidden_states = self.model(input_ids, positions, forward_batch)
```

**Hypothesized failure mechanism for GLM-4.6-FP8**:
1. Server args: `quantization=None` (not explicitly set; FP8 is auto-detected from model config file as compressed-tensors)
2. Server args: `speculative_draft_model_quantization=None` (user didn't pass this flag)
3. After `_handle_speculative_decoding_settings()`: `speculative_draft_model_quantization = self.quantization = None` (stays None)
4. In `Glm4MoeForCausalLMNextN.__init__`: `self.needs_quant_draft = None` (falsy)
5. `quant_config = quant_config if self.needs_quant_draft else None` -> `quant_config = None`
6. **The draft model is constructed without a quant_config**, even though the checkpoint uses compressed-tensors FP8
7. Draft model weights load in an inconsistent state relative to the target model
8. Draft predictions diverge from target predictions -> acceptance length collapses 2.0 -> 1.0

**Consistency check with observed data**:
- Target (base) model unaffected -> TP8 throughput stable ✓
- Draft model broken -> TP8+MTP acceptance drops to ~1.0 ✓
- Persistent after commit -> all subsequent nightlies bad ✓
- Matches MTP-only symptom ✓

**Open question**: Does `Glm4MoeModelNextN(config, quant_config=None)` silently produce a mis-quantized draft (e.g., loading FP8 tensors as raw bytes interpreted as BF16), or does it fail loudly? The fact that runs complete without error suggests silent mis-loading.

#### Suspect B: `eb407b80f` [Kernel] Make FA3/FA4 imports lazy in FlashAttentionBackend (#22028) -- #1 in window

**Why this is still a suspect**:
- Tested BAD by the user's local experiment, confirming the regression is present at this commit (but does not prove it was introduced HERE)
- 490-line refactor in `flashattention_backend.py`, restructuring FA3/FA4 import paths and some `metadata.extend_with_prefix` handling
- In v1 I concluded the diff was functionally equivalent on NVIDIA, but subtle changes to the cascade-attention path used by spec-decode `target_verify` and `draft_extend` could have been missed

**Why it might NOT be the cause**:
- Line-by-line analysis of the extend/decode path shows the `elif (not _is_musa or ...): flash_attn_with_kvcache(...)` -> `else: flash_attn_with_kvcache(...)` transformation is equivalent on NVIDIA (`_is_musa=False` makes the condition trivially true)
- FA3/FA4 selection logic is identical in effect (based on `self.fa_impl_ver`)
- If this were the sole cause, reverting this commit at the Apr 4 nightly SHA would fix it -- worth testing

### 6.2 MEDIUM PRIORITY

#### Suspect C: `030fb1c4b` refactor: replace mm_inputs dict with MultimodalProcessorOutput (#21738) -- #9 in window
- Touches `scheduler.py` (14 lines) and `schedule_batch.py` (62 lines)
- Most changes are multimodal-specific dataclass refactor, but scheduler changes could subtly affect speculative batch construction
- Low prior probability but worth ruling out if Suspect A is negated

#### Suspect D: `ee9d922f5` Revert "Fused temperature + softmax in sampling" (#22046) -- #11 in window
- In v1 dismissed because the fused kernel only activates at `batch_size >= 128`
- Benchmark uses bs=1,8,16,64 -- all below threshold
- The `_warmup_fused_sampling` call in `model_runner.py` was also removed -- unlikely to affect correctness but worth considering
- Low probability

#### Suspect E: `56ac9c993` [Fix] Add _MOE_TP to graph_capture for MoE models with ep>1 (#21907) -- #15 in window
- GLM-4.6 is MoE, but the test runs with TP=8 (no EP), so the `ep>1` condition should not trigger
- Low prior probability

### 6.3 LOW PRIORITY (Plausible but weaker signal)

| # | SHA | Title | Why low |
|---|-----|-------|---------|
| #2 | `6aafe756b` | Revert NVFP4 Marlin fallback for non-Blackwell GPUs | GLM-4.6 uses compressed-tensors FP8, not NVFP4 |
| #3 | `0c9dc098e` | Fix DP attention worker port binding for IPv6 | Network-level; test runs without DP attention |
| #4 | `ed3435e37` | HiSparse: Optimize server args checking | GLM doesn't use HiSparse |
| #17 | `cd75d54fc` | Fix CUDA graph replay issues in trtllm_mla draft_extend | MLA-specific; GLM-4.6 does not use MLA |
| #31 | `2c4fb8892` | chore: bump sgl-kernel version to 0.4.1 (source version) | Source-only bump; CI installs from PyPI so this is only a file edit. Effect validated by user's experiment. |
| #35 | `efa7b2d5d` | Revert MUSA FA3 attention backend | Removes 269 lines from `flashattention_backend.py` (mostly MUSA), worth spot-checking if NVIDIA-path code was collateral damage |
| #36 | `5f0df1e2a` | Fix dp-attention parallel info in bench_one_batch | GLM TP8 doesn't use dp-attention |

### 6.4 RULED OUT

| # | SHA | Reason |
|---|-----|--------|
| #8 | `44e5d3570` JIT activation | Reverted by `ac1e437f6` in the Apr 3-4 window (which is #5 of the 51-commit list, OUTSIDE the new window). If it were the cause, the Apr 4 nightly would already have been fixed by the revert, but it's still BAD. |
| #31 | `2c4fb8892` sgl-kernel 0.4.1 source version bump | Only edits version strings; doesn't change kernel behavior |
| Various | AMD/NPU/CPU/Intel/MUSA/diffusion/docs/CI | Platform-scoped changes that do not run on NVIDIA H200 |

---

## 7. New Bisection Plan

### 7.1 Step 1 -- Test the primary suspect (`ad0516d9c`) and its parent

This is both a hypothesis test AND the optimal bisection midpoint (position 22/43).

```bash
# Test 1a: ad0516d9c itself (should be BAD if this is the cause)
git checkout ad0516d9c
bash scripts/ci/cuda/ci_install_dependency.sh
mkdir -p exps/glm_bisect/ad0516d9c
cd exps/glm_bisect/ad0516d9c
{ for i in $(seq 1 5); do
    echo "===== Trial $i =====";
    pytest /sgl-workspace/sglang/test/registered/8-gpu-models/test_glm_46_fp8.py;
    echo "Exit code: $?";
    echo;
  done; } > ad0516d9c.txt 2>&1
# Copy results_*.json files here from wherever the test writes them

# Test 1b: parent of ad0516d9c (should be GOOD if ad0516d9c is the cause)
git checkout ad0516d9c^
bash scripts/ci/cuda/ci_install_dependency.sh
mkdir -p exps/glm_bisect/ad0516d9c~1
# ... same trial loop
```

**Expected outcomes and next actions**:

| Result | Interpretation | Next step |
|--------|----------------|-----------|
| `ad0516d9c` BAD, `ad0516d9c^` GOOD | **Root cause confirmed** | File bug/revert for PR #19246 on GLM-4.6-FP8 path |
| `ad0516d9c` GOOD, `ad0516d9c^` GOOD | Regression is in commits #1-#21 of the window (newer than `ad0516d9c`) | Bisect between `ad0516d9c` and `eb407b80f` (midpoint ≈ #11 `ee9d922f5`) |
| `ad0516d9c` BAD, `ad0516d9c^` BAD | Regression is in commits #23-#43 of the window (older than `ad0516d9c`) | Bisect between `29d8e959d704` and `ad0516d9c^` (midpoint ≈ #33 `d1b7c3907`) |
| `ad0516d9c` BAD, `ad0516d9c^` mixed | Intermittent / multi-commit cause | Run more trials and consider non-deterministic sources |

### 7.2 Step 2 -- Additional targeted tests (regardless of Step 1 outcome)

To independently verify hypotheses:

```bash
# Test 2: Does forcing --speculative-draft-model-quantization=fp8 on a BAD SHA fix it?
# (Tests the ad0516d9c hypothesis at the source)
git checkout 95cdbce34fa9  # Apr 4 bad nightly
# Modify test_glm_46_fp8.py to add --speculative-draft-model-quantization=fp8 to mtp_args
# If acc_len returns to 2.0, confirms the quant_config=None is the mechanism
```

```bash
# Test 3: On a BAD SHA, revert ONLY ad0516d9c and rerun
git checkout 95cdbce34fa9
git revert --no-commit ad0516d9c
bash scripts/ci/cuda/ci_install_dependency.sh
# Run trials
# If acc_len returns to 2.0, confirms ad0516d9c is the root cause
```

### 7.3 Step 3 -- Verification on Apr 4 nightly SHA

After identifying the culprit, verify the fix on `95cdbce34fa9` (the first BAD nightly) to confirm nothing else in the 43-commit window contributes.

---

## 8. Data Used in This Report

All data supporting this v2 report is in `/sgl-workspace/sglang/exps/glm_bisect/`:

```
exps/glm_bisect/
├── glm_46_fp8_regression_report.md          # v1 (kernel-bump hypothesis, now disproven)
├── glm_46_fp8_regression_report_v2.md       # THIS DOCUMENT
├── glm_46_fp8_mar30_31_regression_report.md # Companion: smaller Mar 30-31 TP8 regression
├── ad064c2f4~1/    # Local experiment: before GLM FP32 gate cast (GOOD)
│   └── results.jsonl
├── ad064c2f4/      # Local experiment: after GLM FP32 gate cast (GOOD for MTP)
│   └── result.jsonl
├── 84118acf5~1/    # Local experiment at eb407b80f (BAD)
│   └── result.jsonl
└── 84118acf5/      # Local experiment after sglang-kernel 0.4.1 bump (BAD, identical to parent)
    └── result.jsonl
```

Each experiment directory contains `results_*.json` files (one per (variant, trial) pair) and a summary `result.jsonl` or `results.jsonl`.

---

## 9. Summary

1. **v1 hypothesis disproven**: The sglang-kernel 0.4.0 -> 0.4.1 PyPI bump is NOT the cause. User's local experiment shows `84118acf5~1` and `84118acf5` produce identical TP8+MTP throughput (559.44 vs 559.54 tok/s).

2. **Regression is in sglang source code, not in the kernel binary**. The user's experiment at `eb407b80f` (which pip-installs `sglang-kernel==0.4.0`) already shows the BAD throughput.

3. **New narrowed window**: 43 first-parent commits between `29d8e959d704` (Apr 3 good) and `eb407b80f` (tested BAD).

4. **New primary suspect**: `ad0516d9c` [NPU] optimize glm4.7 (#19246), which modifies the GLM EAGLE draft model. The change sets `quant_config = None` for the draft model when `speculative_draft_model_quantization` is not explicitly set -- which is exactly the case for GLM-4.6-FP8. This would cause the draft model to load compressed-tensors FP8 weights without the corresponding quant config, producing predictions that diverge from the target model.

5. **This also happens to be the midpoint of the 43-commit window**, so testing it is both a hypothesis check and an efficient bisection step.

6. **Recommended next action**: Run the test at `ad0516d9c` and `ad0516d9c^` to confirm or refute. Commands provided in Section 7.1.
