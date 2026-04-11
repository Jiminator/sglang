# GLM-4.6-FP8 TP8+MTP Nightly CI Performance Regression Report

**Date**: 2026-04-11
**Investigator**: Claude (automated bisection)
**Status**: Root cause narrowed to 51-commit window; top suspect identified

---

## 1. Failure Signature

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

## 2. Test Configuration

The GLM-4.6-FP8 nightly test runs two variants:

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

Both variants are benchmarked with `bench_one_batch_server` at batch sizes 1, 8, 16, 64 with input_len=4096, output_len=512.

---

## 3. Timeline

| Milestone | Date | Run ID | SHA | Detail |
|-----------|------|--------|-----|--------|
| Last known good nightly | Apr 3, 00:44 UTC | 23928679336 | `29d8e959d704` | TP8+MTP acc_len=1.96 |
| First known bad nightly | Apr 4, 00:40 UTC | 23967509528 | `95cdbce34fa9` | TP8+MTP acc_len=1.01 |
| Commits in window | -- | -- | -- | 51 first-parent commits |

---

## 4. Historical Throughput Data

### 4.1 Full Nightly History (Mar 29 -- Apr 6)

All throughput values are **output throughput in tok/s**.

#### TP8 (non-speculative) -- stable across entire window

| Date | SHA (short) | Run ID | Job ID | bs=1 | bs=8 | bs=16 | bs=64 |
|------|-------------|--------|--------|------|------|-------|-------|
| Mar 29 | `3ab9afd65380` | 23697967641 | 69036426954 | 83.88 | 487.12 | 831.46 | 2171.33 |
| Mar 30 | `afb32d76224e` | 23723157688 | 69101654442 | 83.81 | 487.10 | 831.16 | 2174.34 |
| Mar 31 | `3650bfb19926` | 23774903575 | 69274478043 | 83.05 | 454.08 | 791.69 | 2085.36 |
| Apr 1  | `a8759dd9af05` | 23826444093 | 69450407169 | 83.12 | 454.15 | 791.73 | 2086.96 |
| Apr 2  | `d7256eb69af9` | 23877953915 | 69624981226 | 83.16 | 454.09 | 792.23 | 2084.82 |
| Apr 3  | `29d8e959d704` | 23928679336 | 69790979057 | 83.29 | 453.33 | 791.53 | 2085.97 |
| **Apr 4** | **`95cdbce34fa9`** | **23967509528** | **69910340256** | **83.12** | **453.32** | **790.94** | **2087.12** |
| Apr 5  | `70658bfeb52a` | 23991022988 | 69970665978 | 83.10 | 453.30 | 791.14 | 2086.84 |
| Apr 6  | `93109cc89be3` | 24014261122 | 70030854934 | 83.06 | 453.24 | 790.40 | 2086.24 |

**Observation**: TP8 throughput is completely unaffected by the regression. There is a minor ~7% drop from Mar 30 to Mar 31 (at bs=8 and bs=16), but this is a separate, smaller issue.

#### TP8+MTP (EAGLE speculative decoding) -- cliff between Apr 3 and Apr 4

| Date | SHA (short) | Run ID | Job ID | bs=1 | bs=8 | bs=16 | acc_len (bs=16) | PASS/FAIL |
|------|-------------|--------|--------|------|------|-------|-----------------|-----------|
| Mar 29 | `3ab9afd65380` | 23697967641 | 69036426954 | 124.78 | 671.71 | 1114.15 | ~2.0 | PASS |
| Mar 30 | `afb32d76224e` | 23723157688 | 69101654442 | 129.80 | 681.89 | 1129.19 | ~2.0 | PASS |
| Mar 31 | `3650bfb19926` | 23774903575 | 69274478043 | 117.09 | 653.44 | 1083.81 | ~2.0 | PASS |
| Apr 1  | `a8759dd9af05` | 23826444093 | 69450407169 | 117.55 | 645.06 | 1068.21 | ~2.0 | PASS |
| Apr 2  | `d7256eb69af9` | 23877953915 | 69624981226 | 115.50 | 647.93 | 1071.46 | ~2.0 | PASS |
| Apr 3  | `29d8e959d704` | 23928679336 | 69790979057 | 121.61 | 633.78 | 1022.93 | 1.96 | PASS |
| **Apr 4** | **`95cdbce34fa9`** | **23967509528** | **69910340256** | **62.25** | **350.15** | **576.19** | **1.01** | **FAIL** |
| Apr 5  | `70658bfeb52a` | 23991022988 | 69970665978 | 62.28 | 354.32 | 576.30 | ~1.0 | FAIL |
| Apr 6  | `93109cc89be3` | 24014261122 | 70030854934 | 65.77 | 355.05 | 575.35 | ~1.0 | FAIL |

### 4.2 Detailed Comparison: Last Good vs First Bad

#### Apr 3 (GOOD) -- TP8+MTP

| Metric | bs=1 | bs=8 | bs=16 |
|--------|------|------|-------|
| Output throughput (tok/s) | 121.61 | 633.78 | 1022.93 |
| Latency (s) | 4.42 | 7.84 | 10.75 |
| Acceptance length | 2.0 | 1.99 | 1.98 |
| ITL (ms) | 8.22 | 12.62 | 15.64 |

Runtime decode logs: `accept len: 1.96, accept rate: 0.49, gen throughput: 2294 tok/s`

#### Apr 4 (BAD) -- TP8+MTP

| Metric | bs=1 | bs=8 | bs=16 |
|--------|------|------|-------|
| Output throughput (tok/s) | 62.25 | 350.15 | 576.19 |
| Latency (s) | 8.48 | 13.08 | 16.95 |
| Acceptance length | 1.01 | 1.12 | 1.12 |
| ITL (ms) | 16.06 | 22.85 | 27.77 |

Runtime decode logs: `accept len: 1.01, accept rate: 0.25, gen throughput: 1184 tok/s`

### 4.3 Regression Characterization

- **Throughput**: ~50% drop across all batch sizes for TP8+MTP
- **Acceptance length**: Collapsed from ~2.0 to ~1.0 (effectively zero draft token acceptance)
- **Accept rate**: Dropped from 0.49 to 0.25 (consistent with acc_len going from ~2/4 to ~1/4)
- **TP8 (non-spec)**: Completely unaffected -- confirms the base model is healthy
- **Interpretation**: The EAGLE draft model produces predictions that are almost entirely rejected by the target model. The system degrades to sequential (non-speculative) generation.

---

## 5. Commit Window Analysis

### 5.1 All 51 Commits in the Window

Commits between `29d8e959d704` (Apr 3 good) and `95cdbce34fa9` (Apr 4 bad), numbered from newest to oldest:

```
 #1  95cdbce34 [Test] Extract common PD server setup into base fixture (#22080)
 #2  9593d434c fix: pause_generation should not populate running_batch on prefill nodes (#20273)
 #3  5118295f7 [CI] Support CPU stage and auto-batch same-stage files in `/rerun-test` (#22081)
 #4  90e86800f [Score API] Implement EngineScoreMixin for scoring functionality (#21342)
 #5  ac1e437f6 Revert "[Feature] JIT activation and update skills (by codex)" (#22078)
 #6  8cb337c8e [Bugfix] Temporarily skip TRTLLM attention on (G)B300 (SM103) (#21906)
 #7  1d7a53dd0 [Fix] XGrammarGrammarBackend reset to clear inherited cache (#22054)
 #8  84118acf5 chore: bump sglang-kernel version to 0.4.1 (#22009)          *** SUSPECT ***
 #9  eb407b80f [Kernel] Make FA3/FA4 imports lazy in FlashAttentionBackend (#22028)
#10  6aafe756b Revert "[Feature] NVFP4 Marlin fallback for non-Blackwell GPUs" (#22047)
#11  0c9dc098e Fix DP attention worker port binding for IPv6 support (#21917)
#12  ed3435e37 [HiSparse]: Optimize server args checking (#22065)
#13  151f72716 [diffusion] fix: fix gated repo failing the generate cmd (#22040)
#14  896ea7582 Remove reverted test (#22058)
#15  47f4fd275 [CI] Fix test suite names and add suite validation (#21937)
#16  44e5d3570 [Feature][JIT Kernel] JIT activation and update skills (#21766)
#17  030fb1c4b refactor: replace mm_inputs dict with MultimodalProcessorOutput (#21738)
#18  9f409d074 [CI] Adjust CI server launch timeout (#22045)
#19  ee9d922f5 Revert "[Kernel] Fuse temperature + softmax in sampling" (#22046)
#20  97adf8a29 [misc] Add hint for kernel release trigger (#22036)
#21  98ac40192 [Workflow] Fix kernel release build failures (#22018)
#22  838f815e9 [diffusion] CI: temporarily disable accuracy ci (#22031)
#23  56ac9c993 [Fix] Add _MOE_TP to graph_capture for MoE models with ep>1 (#21907)
#24  ac593fed9 [AMD][Dockerfile] Support build-arg AITER_COMMIT (#21949)
#25  cd75d54fc [Bugfix] Fix CUDA graph replay issues in trtllm_mla draft_extend (#21987)
#26  4f84ce580 [CI] ci: add test_http_server_auth.py to CI (#21866)
#27  658a2813d [NPU] Update CI Dependency (#21578)
#28  d07d0a15c [AMD] Add MiniMax-M2.5 nightly perf benchmarks (#21524)
#29  7431db739 [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA (#21511)
#30  ad0516d9c [NPU] optimize glm4.7 (#19246)                               *** SUSPECT ***
#31  d82097a0d [PD] Tiny register info field cleanup for mooncake backend (#22016)
#32  24f52e66d fix: remove duplicate words in comments (#22007)
#33  4cc970290 [CI] Fix duplicate job names that bypass branch protection (#22001)
#34  6b876a771 [ROCM][RL] Shuffle Weight In-Place (#21825)
#35  75de47968 [Misc] Update CI permission (#22014)
#36  4d097047f [PD]: HiSparse support for cache transfer (#21591)
#37  5c082c307 [Workflow] Fix kernel release jobs skipped on push events (#22011)
#38  0a709cfe0 [Workflow] Avoid triggering nightly tests in kernel bump (#22010)
#39  2c4fb8892 chore: bump sgl-kernel version to 0.4.1 (#21447)
#40  5bcbc9757 [AMD] Resolve performance degression with aiter-allreduce-fusion (#21947)
#41  d1b7c3907 [Parallel State Refactor 2/n] Unify AMD deterministic all reduce (#20871)
#42  81efcc353 [NPU] Optimized the wording in the npu docs (#21998)
#43  efa7b2d5d Revert "[MUSA] Add FA3 attention backend support" (#22002)
#44  5f0df1e2a [Bugfix] Fix incorrect dp-attention parallel info (#21519)
#45  69e89a1fc [VLM] Enable per-image MM splitting by default (#21899)
#46  8897ac58f [PP] qwen3 vl skip layer id for pp (#19135)
#47  991f3aa5b [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (#19652)
#48  2b5aed94f Remove maxItems=1 restriction when tool_choice is specified (#20208)
#49  0539c62bc [Diffusion][NPU] Add support for MOVA (#21633)
#50  1f97714f9 [CI] Add timeouts to Slack upload urlopen and WebClient (#21903)
#51  89affff29 Skip broken AutoModel mapping entries for Llava submodules (#21892)
```

### 5.2 Triage Categories

Most commits were eliminated as irrelevant based on platform, scope, or affected code paths:

| Category | Count | Commits |
|----------|-------|---------|
| CI / workflow / docs only | 15 | #1,#3,#14,#15,#18,#20,#21,#22,#26,#27,#33,#35,#37,#38,#42 |
| AMD / ROCm only | 5 | #24,#28,#29,#34,#40 |
| NPU only (except #30) | 2 | #42,#49 |
| MUSA only | 1 | #43 |
| Diffusion only | 2 | #13,#22 |
| VLM / multimodal only | 3 | #17,#45,#46 |
| PD / disaggregation only | 3 | #1,#31,#36 |
| Unrelated features | 6 | #4,#7,#11,#12,#32,#48 |
| **Potentially relevant** | **~10** | See Section 5.3 |

### 5.3 Suspects Investigated

#### Suspect 1 (PRIMARY): `84118acf5` -- sglang-kernel version bump to 0.4.1 (#22009)

**What changed**: `python/pyproject.toml` dependency `sglang-kernel==0.4.0` -> `sglang-kernel==0.4.1`

**Why it matters**:
- The nightly CI installs `sglang-kernel` from PyPI as a pre-built binary wheel
- Apr 3 (good) installed `sglang-kernel==0.4.0`; Apr 4 (bad) installed `sglang-kernel==0.4.1`
- `sglang-kernel` provides compiled FA3 attention kernels (`sgl_kernel.flash_attn.flash_attn_with_kvcache`)
- Both TP8 and TP8+MTP use FA3, but spec decode uses additional code paths (cascade attention, draft_extend, target_verify)

**Kernel source changes between 0.4.0 and 0.4.1** (released from `sgl-kernel/` directory):

| Commit | Description | Relevance |
|--------|-------------|-----------|
| `a12fea21e` | Expose `get_scheduler_metadata` for FA3 decode optimization | **HIGH** -- modifies FA3 decode path |
| `c7d03a621` | Revert "Rollback flashmla to older version" | Low -- GLM doesn't use MLA |
| `dbe871efd` | Rollback flashmla to older version (later reverted) | Low -- net no-change |
| `ca3286d2d` | FA3 attention backend on MUSA | None on NVIDIA |
| `6a9b09847` | CUTLASS NVFP4 GEMM SM120 | None on H200/SM90 |
| `cdd7d6a22` | Remove obsolete sgl-kernel legacy paths | Possibly relevant |
| `6da8f5f69` | Fix topk softmax performance (CPU only) | None on NVIDIA |
| `8a56a7b04` | Migrate cast (downcast_fp8) from AOT to JIT | Possibly relevant |

**Verification needed**: Run the bad SHA (`95cdbce34fa9`) with `sglang-kernel==0.4.0` force-installed. If acceptance length returns to ~2.0, the kernel is confirmed as root cause.

**CI log evidence**:
```
# Apr 3 good run install:
+ uv pip install sglang-kernel==0.4.0 --force-reinstall
 ~ sglang-kernel==0.4.0

# Apr 4 bad run install:
+ uv pip install sglang-kernel==0.4.1 --force-reinstall
 ~ sglang-kernel==0.4.1
```

#### Suspect 2: `ad0516d9c` -- [NPU] optimize glm4.7 (#19246)

**What changed**: `python/sglang/srt/models/glm4_moe_nextn.py` (the GLM EAGLE draft model) and `python/sglang/srt/models/glm4_moe.py`

**Changes to `glm4_moe_nextn.py` (draft model)**:

1. **`__init__`**: Added quantization gating logic:
   ```python
   # BEFORE:
   self.quant_config = quant_config

   # AFTER:
   self.needs_quant_draft = get_global_server_args().speculative_draft_model_quantization
   quant_config = quant_config if self.needs_quant_draft else None
   ```

2. **`forward()`**: Added env var patching:
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

**Analysis for GLM-4.6-FP8**:
- Server args show `quantization=None, speculative_draft_model_quantization=None`
- After `server_args._handle_speculative_decoding_settings()`, `speculative_draft_model_quantization` is set to `self.quantization` (which is `None`) -- so it stays `None`
- Therefore `needs_quant_draft = None` (falsy)
- `quant_config = None` (but it was already `None` since `quantization=None`)
- The forward path sets `SGLANG_DEEPEP_BF16_DISPATCH=1` and `DEEP_NORMAL_MODE_USE_INT8_QUANT=0` during every draft model forward call
- These env vars are checked in `sgl-kernel` DeepEP dispatch code and NPU MoE code
- On NVIDIA H200 without EP, these should be no-ops

**Risk assessment**: Low on its own, but the interaction of setting `SGLANG_DEEPEP_BF16_DISPATCH=1` during every draft forward call could have unexpected effects if any code path reads this env var at runtime on NVIDIA.

**Changes to `glm4_moe.py` (base model)**:
- All changes gated behind `_is_npu` checks
- No effect on NVIDIA GPUs

#### Suspect 3: `ee9d922f5` -- Revert fused temperature+softmax in sampling (#22046)

**What changed**: Removed fused Triton kernel for temperature+softmax in `sampler.py`

**Analysis**: The fused kernel only activated at `batch_size >= 128`. The benchmark uses batch sizes 1, 8, 16, 64 -- all below the threshold. Therefore, the fused path was **never active** for these benchmarks. The revert is functionally a no-op for the affected test.

**Verdict**: Eliminated as cause.

#### Suspect 4: `eb407b80f` -- Make FA3/FA4 imports lazy in FlashAttentionBackend (#22028)

**What changed**: 490 lines refactored in `flashattention_backend.py`. FA3/FA4 imports moved from module-level to `__init__`. The `_is_musa` conditionals were removed.

**Analysis**: Line-by-line comparison of the old and new code for the extend/decode paths used by speculative decoding confirmed functional equivalence on NVIDIA:
- Old: `elif (not _is_musa or ...): flash_attn_with_kvcache(...)` -- always enters on NVIDIA
- New: `else: flash_attn_with_kvcache(...)` -- always enters
- FA3/FA4 selection logic is equivalent (based on `fa_impl_ver` in both)

**Verdict**: Eliminated as cause (on NVIDIA).

#### Suspect 5: `9593d434c` -- pause_generation should not populate running_batch on prefill nodes (#20273)

**What changed**: Added `and self.disaggregation_mode != DisaggregationMode.PREFILL` guard and `and not self.running_batch.is_empty()` guard in scheduler retract path.

**Analysis**: The disaggregation guard only applies in PD disaggregation mode. The `running_batch.is_empty()` guard on the retract path affects `v1_spec_info_filtered` which is spec v1 only. GLM uses spec v2.

**Verdict**: Eliminated as cause.

---

## 6. CI Infrastructure Notes

### How the nightly runs were identified

1. Listed runs of `nightly-test-nvidia.yml` on `main` via GitHub API
2. The April 2 and April 3 scheduled runs were `cancelled` overall, but the `nightly-test-general-8-gpu-h200 (1)` job completed successfully within those cancelled runs
3. Job logs were downloaded via `gh api repos/sgl-project/sglang/actions/jobs/{id}/logs`
4. Throughput data was extracted from the markdown tables written to `GITHUB_STEP_SUMMARY` by `NightlyBenchmarkRunner.write_final_report()`

### Dispatch limitation

Workflow dispatch on `sgl-project/sglang` requires admin rights (`HTTP 403: Must have admin rights to Repository`). Binary bisection via dispatched runs was not possible. The analysis relied on code review of the 51-commit window instead.

### Artifact status

| Run | `metrics-8gpu-h200-partition-1` | `consolidated-metrics` |
|-----|--------------------------------|------------------------|
| Mar 29-Apr 2 | Expired | Available |
| Apr 3 | N/A (no consolidation job) | N/A |
| Apr 4+ | Expired (partition) / Available (consolidated) | Available |

---

## 7. Recommended Next Steps

### Immediate verification (pick one)

1. **Test with forced kernel downgrade** (fastest):
   ```bash
   # On the bad SHA (95cdbce34fa9), force install old kernel:
   pip install sglang-kernel==0.4.0 --force-reinstall
   # Then run the GLM MTP benchmark
   python3 test/manual/test_glm_46_fp8.py
   ```
   If acceptance length returns to ~2.0, the kernel version is confirmed as root cause.

2. **Revert ad0516d9c on bad SHA** (rules out GLM NextN change):
   ```bash
   git revert --no-commit ad0516d9c
   # Run the GLM MTP benchmark
   ```

3. **Dispatch bisection runs** (requires admin):
   - Test SHA `84118acf5` (right after kernel bump) -- if BAD, kernel confirmed
   - Test a SHA between `ad0516d9c` and `84118acf5` to isolate further

### Kernel investigation

If the kernel version is confirmed as root cause:
1. Diff the FA3 kernel source between the 0.4.0 and 0.4.1 release tags
2. Focus on `a12fea21e` (expose `get_scheduler_metadata` for FA3 decode optimization) -- this directly modifies the FA3 decode path that spec decode relies on
3. Check if `get_scheduler_metadata` changes the `num_splits` heuristic or scheduling strategy in a way that affects numerical output consistency between draft and target model forward passes

### Test hardening

- Add an explicit acceptance length threshold assertion to the GLM TP8+MTP test (e.g., `acc_len >= 1.5`)
- Consider pinning `sglang-kernel` version in nightly tests or adding a kernel smoke test before performance benchmarks

---

## 8. Appendix: Raw Runtime Logs

### Apr 3 (GOOD) -- TP8+MTP decode logs (bs=64 batch)

```
accept len: 1.95, accept rate: 0.49, cuda graph: True, gen throughput (token/s): 36.91
accept len: 1.95, accept rate: 0.49, cuda graph: True, gen throughput (token/s): 2294.53
accept len: 1.96, accept rate: 0.49, cuda graph: True, gen throughput (token/s): 2299.39
accept len: 1.96, accept rate: 0.49, cuda graph: True, gen throughput (token/s): 2304.63
accept len: 1.96, accept rate: 0.49, cuda graph: True, gen throughput (token/s): 2275.97
accept len: 1.96, accept rate: 0.49, cuda graph: True, gen throughput (token/s): 2273.53
accept len: 1.97, accept rate: 0.49, cuda graph: True, gen throughput (token/s): 2289.60
accept len: 1.97, accept rate: 0.49, cuda graph: True, gen throughput (token/s): 2278.59
accept len: 1.96, accept rate: 0.49, cuda graph: True, gen throughput (token/s): 2255.30
```

### Apr 4 (BAD) -- TP8+MTP decode logs (bs=64 batch)

```
accept len: 1.01, accept rate: 0.25, cuda graph: True, gen throughput (token/s): 17.34
accept len: 1.02, accept rate: 0.25, cuda graph: True, gen throughput (token/s): 1184.26
accept len: 1.02, accept rate: 0.25, cuda graph: True, gen throughput (token/s): 1176.37
accept len: 1.03, accept rate: 0.26, cuda graph: True, gen throughput (token/s): 1184.60
accept len: 1.01, accept rate: 0.25, cuda graph: True, gen throughput (token/s): 1168.68
accept len: 1.02, accept rate: 0.25, cuda graph: True, gen throughput (token/s): 1174.19
accept len: 1.01, accept rate: 0.25, cuda graph: True, gen throughput (token/s): 1165.01
accept len: 1.02, accept rate: 0.25, cuda graph: True, gen throughput (token/s): 1171.89
accept len: 1.01, accept rate: 0.25, cuda graph: True, gen throughput (token/s): 1154.80
accept len: 1.01, accept rate: 0.25, cuda graph: True, gen throughput (token/s): 1156.23
accept len: 1.01, accept rate: 0.25, cuda graph: True, gen throughput (token/s): 1157.53
```

### Server args comparison (both identical)

```
command=sglang serve --model-path zai-org/GLM-4.6-FP8 --tp=8 --trust-remote-code \
  --speculative-algorithm=EAGLE --speculative-num-steps=3 \
  --speculative-eagle-topk=1 --speculative-num-draft-tokens=4 \
  --tp 8 --enable-multimodal --device cuda --host 127.0.0.1 --port 11000

# Key server_args fields (same in both):
quantization=None
speculative_algorithm='EAGLE'
speculative_draft_model_quantization=None
speculative_num_steps=3
speculative_eagle_topk=1
```

### Kernel installation logs

```
# Apr 3 (GOOD):
+ uv pip install sglang-kernel==0.4.0 --force-reinstall --index-strategy unsafe-best-match --prerelease allow
 ~ sglang-kernel==0.4.0

# Apr 4 (BAD):
+ uv pip install sglang-kernel==0.4.1 --force-reinstall --index-strategy unsafe-best-match --prerelease allow
 ~ sglang-kernel==0.4.1
```
