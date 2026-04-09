# CI Regression Bisection Report: `TestTransformersFallbackTorchAO.test_mmlu`

**Date**: 2026-04-08
**Investigated by**: Claude Code
**Status**: Resolved (PR #22210 merged 2026-04-06)

---

## 1. Test Location & Partition

| Field | Value |
|-------|-------|
| **File** | `test/registered/models/test_transformers_models.py` |
| **Test class** | `TestTransformersFallbackTorchAO` |
| **Test method** | `test_mmlu` |
| **Suite** | `stage-b-test-1-gpu-small` |
| **Runner label** | `1-gpu-5090` (RTX 5090, SM120) |
| **Regression window (earlier)** | `cffc95edf45547c22c3d1493400ec8c3fdf0757d` |
| **Regression window (later)** | `990c7590b835549c17cf089422f0e5c3f520ad8b` |

### Partition Shift During the Window

The test's partition assignment **changed mid-window** due to commit `875a61599` ("fix(ci): update est_time for 57 tests based on runtime analysis (#21896)"), which bumped `est_time` from 245 to 450. Combined with new tests being added to the suite, the LPT partitioner reassigned the test:

| Period | Partition | est_time | Total Tests |
|--------|-----------|----------|-------------|
| Before `875a61599` (runs 1-5) | **7** | 245 | 81 |
| After `875a61599` (runs 6-18) | **5** | 450 | 81-89 |

**Critical note**: Any analysis that checks only a single partition number across all runs will produce incorrect results. The correct partition must be determined per-SHA.

### Test Behavior

`TestTransformersFallbackTorchAO` inherits from `TestTransformersFallbackEndpoint` and:
1. Launches `meta-llama/Llama-3.1-8B-Instruct` with `--model-impl transformers --torchao-config int4wo-128`
2. Runs a 64-example MMLU eval via `sglang.test.run_eval`
3. Asserts `score >= 0.65` (the `mmlu_lower_bound`)

A score of 0.640625 = 41/64 correct. Passing requires 42/64 = 0.65625. **One additional wrong answer** causes failure.

---

## 2. Failure Signature

### Confirmed Assertion Error

From PR #22210 and CI run 24042766142/job 70131079937:

```
FAIL: test_mmlu (test_transformers_models.TestTransformersFallbackTorchAO.test_mmlu)
----------------------------------------------------------------------
AssertionError: 0.640625 not greater than or equal to 0.65
```

### Job Metadata (3 failing scheduled runs)

| Run | Run ID | SHA | Date | Runner | Failed Step | Exit Code |
|-----|--------|-----|------|--------|-------------|-----------|
| 11 | 23945858055 | 97adf8a2 | Apr 03 12:16 | 5090-a-runner-2 | "Run test" (step 7) | 255 |
| 14 | 23978641020 | abc29752 | Apr 04 12:12 | 5090-b-runner-4 | "Run test" (step 7) | 255 |
| 15 | 23984641093 | efee62ef | Apr 04 18:11 | 5090-a-runner-1 | "Run test" (step 7) | 255 |

Note: Full log content (exact scores per run, nvidia-smi output) could not be retrieved — the GitHub Actions logs API requires authentication not available in this environment.

---

## 3. Corrected Scheduled Run Analysis

The correct partition was determined per-SHA using `git checkout <SHA>` and running the LPT partitioner. Each run was then checked against the **correct partition's job**.

### Runs 1-5: Partition 7 (before est_time bump)

| # | Run ID | SHA | Date | Partition 7 | Runner |
|---|--------|-----|------|-------------|--------|
| 1 | 23826038158 | b6fe0cca | Apr 01 00:34 | **PASS** | 5090-a-runner-3 |
| 2 | 23835508455 | a188208e | Apr 01 06:37 | **PASS** | 5090-b-runner-6 |
| 3 | 23848321552 | e67b95d6 | Apr 01 12:22 | **PASS** | 5090-b-runner-5 |
| 4 | 23864035900 | a1c725bd | Apr 01 18:21 | **PASS** | 5090-b-runner-2 |
| 5 | 23877591623 | d7256eb6 | Apr 02 00:28 | **PASS** | 5090-b-runner-4 |

### Runs 6-18: Partition 5 (after est_time bump)

| # | Run ID | SHA | Date | Partition 5 | Runner |
|---|--------|-----|------|-------------|--------|
| 6 | 23887287765 | d24ea24e | Apr 02 06:29 | **PASS** | 5090-a-runner-1 |
| 7 | 23900098396 | 083304ca | Apr 02 12:21 | **PASS** | 5090-a-runner-0 |
| 8 | 23915364384 | 8732b2e9 | Apr 02 18:20 | **PASS** | 5090-b-runner-3 |
| 9 | 23928333410 | 29d8e959 | Apr 03 00:30 | **PASS** | 5090-a-runner-1 |
| 10 | 23936636279 | 4d097047 | Apr 03 06:29 | **PASS** | 5090-b-runner-1 |
| 11 | 23945858055 | 97adf8a2 | Apr 03 12:16 | **FAIL** | 5090-a-runner-2 |
| 12 | 23967260370 | 95cdbce3 | Apr 04 00:28 | **PASS** | 5090-b-runner-6 |
| 13 | 23973150935 | 005e582d | Apr 04 06:24 | **PASS** | 5090-b-runner-5 |
| 14 | 23978641020 | abc29752 | Apr 04 12:12 | **FAIL** | 5090-b-runner-4 |
| 15 | 23984641093 | efee62ef | Apr 04 18:11 | **FAIL** | 5090-a-runner-1 |
| 16 | 24001263672 | df9c831a | Apr 05 12:12 | **PASS** | 5090-b-runner-7 |
| 17 | 24007438409 | 596c34ee | Apr 05 18:11 | **PASS** | 5090-a-runner-4 |
| 18 | 24097433475 | 0c204fbd | Apr 07 18:21 | **PASS** | 5090-a-runner-0 |

### Summary

- **Runs 1-8**: 0/8 failures (0%)
- **Runs 9-18**: 3/10 failures (30%) — runs 11, 14, 15
- Failures occur on **different runners** across both pools (5090-a-runner-2, 5090-b-runner-4, 5090-a-runner-1), ruling out a runner-specific issue.
- The flake rate is ~30% after onset, not 100%, confirming this is **flaky, not deterministic**.

---

## 4. Suspect Commit Inclusion Matrix

Using `git merge-base --is-ancestor`, I verified exactly which key commits are included in each scheduled run SHA:

| Commit | Description | First included in run | Effect |
|--------|-------------|----------------------|--------|
| `875a61599` | est_time 245→450 (partition shift) | Run 6 (d24ea24e) | Moves test from P7 to P5 (no behavioral change) |
| `cb0c2cbfd` | Enable multi-thread weight loading by default (#20289) | Run 6 (d24ea24e) | Changes model loading concurrency |
| `7a59e05dd` | Fuse temperature+softmax in sampling (#20501) | Run 6 (d24ea24e) | Modifies sampling numerics |
| `34ddf135f` | **Stronger transformers modeling backend** (#19163) | **Run 9** (29d8e959) | **1641-line rewrite of transformers.py** |
| `ee9d922f5` | Revert kernel fusion (#22046) | Run 12 (95cdbce3) | Undoes 7a59e05dd |

### Failure Rate by Suspect Commit Window

| Window | Runs | Failures | Rate | Key commits present |
|--------|------|----------|------|-------------------|
| Before all suspects (P7) | 1-5 | 0/5 | 0% | None |
| After cb0c2cbfd + 7a59e05dd, before 34ddf135f (P5) | 6-8 | 0/3 | 0% | weight loading + kernel fusion |
| After 34ddf135f, with kernel fusion active | 9-11 | 1/3 | 33% | **transformers rework** + kernel fusion |
| After 34ddf135f, kernel fusion reverted | 12-18 | 2/7 | 29% | **transformers rework** only |

**Key observation**: The failure rate is ~0% before `34ddf135f` and ~30% after, regardless of whether the kernel fusion is active or reverted. This strongly implicates `34ddf135f` as the primary cause.

---

## 5. Suspect Commits Ranked

### PRIMARY SUSPECT: `34ddf135f` — Stronger transformers modeling backend (#19163)

| Field | Value |
|-------|-------|
| **Author** | Adarsh Shirawalmath (@adarshxs) |
| **Date** | 2026-04-02 16:02:33 -0700 |
| **Insertions** | +2169 |
| **Deletions** | -184 |
| **Files** | `transformers.py` (1641 lines), `model_loader/utils.py` (156), `model_runner.py` (10), `scheduler.py` (57), `model_config.py` (15), + new `transformers_auto.py` (215), `test_transformers_backend_eval.py` (43) |

This commit is a massive rework of the `--model-impl transformers` code path — the exact path exercised by `TestTransformersFallbackTorchAO`. Changes to model loading, weight handling, or execution flow could subtly shift the accuracy distribution of `int4wo-128` quantized inference, pushing it from reliably above 0.65 to occasionally below.

**Evidence**: 0/8 failures before inclusion → 3/10 (30%) after inclusion.

### SECONDARY SUSPECT: `7a59e05dd` — Fuse temperature + softmax in sampling (#20501)

| Field | Value |
|-------|-------|
| **Files** | `sampler.py` (+29/-5), `model_runner.py` (+16), `fused_sampling.py` (+371) |
| **Reverted at** | `ee9d922f5` (included from run 12) |

Fuses temperature scaling and softmax in the sampling decode path, which changes numerical computation order and could produce slightly different token probabilities. However, the revert didn't reduce the failure rate (33% with fusion → 29% after revert), so this is likely not the cause.

### LOW PRIORITY: `cb0c2cbfd` — Enable multi-thread weight loading by default (#20289)

Changes model loading to use multiple threads. Included from run 6, but runs 6-8 all passed. Unlikely to affect inference accuracy.

---

## 6. Runner Comparison

Failures occurred on three different runners across two pools:

| Result | Runner | Pool |
|--------|--------|------|
| **FAIL** (run 11) | 5090-a-runner-2 | pool-a |
| **FAIL** (run 14) | 5090-b-runner-4 | pool-b |
| **FAIL** (run 15) | 5090-a-runner-1 | pool-a |
| PASS (run 6) | 5090-a-runner-1 | pool-a |
| PASS (run 9) | 5090-a-runner-1 | pool-a |
| PASS (run 12) | 5090-b-runner-6 | pool-b |
| PASS (run 16) | 5090-b-runner-7 | pool-b |
| PASS (run 17) | 5090-a-runner-4 | pool-a |

Notably, **5090-a-runner-1** both passed (runs 6, 9) and failed (run 15), proving the failure is **not runner-specific** — it's a stochastic accuracy issue.

---

## 7. Local Reproduction Commands

### Prerequisites
- A machine with at least 1 GPU (16+ GB VRAM for int4wo-128 quantized Llama-3.1-8B)
- Access to `meta-llama/Llama-3.1-8B-Instruct` on HuggingFace
- CUDA toolkit, apt packages: `python3 python3-pip python3-venv python3-dev git libnuma-dev libssl-dev pkg-config libibverbs-dev ffmpeg`

### CI-Matching Setup

The CI uses `scripts/ci/cuda/ci_install_dependency.sh`, **not** `pip install -e "python[all]"`. The script installs `python[dev]` (not `[all]`), plus a pinned `sglang-kernel` version from PyPI, flashinfer artifacts, `nvidia-cudnn-cu12==9.16.0.29`, `nvidia-nvshmem-cu12==3.4.5`, `mooncake-transfer-engine`, `scipy`, `lmms-eval`, and more. To replicate CI as closely as possible:

```bash
cd /path/to/sglang
git checkout <SHA>

# CI install (matches what the runner does).
# On a non-CI machine, /etc/profile.d/sglang-ci.sh won't exist — skip sourcing it
# and manually export any needed env vars (e.g., CUDA_VISIBLE_DEVICES).
CUSTOM_BUILD_SGL_KERNEL=false bash scripts/ci/cuda/ci_install_dependency.sh
```

**Key differences** between `pip install -e "python[all]"` and the CI script:
| | `pip install -e "python[all]"` | CI script |
|---|---|---|
| Extras | `[all]` (includes every optional dep) | `[dev]` (dev/test deps only) |
| sglang-kernel | From source or whatever is cached | Pinned version from PyPI (`--force-reinstall`) |
| flashinfer | Whatever pip resolves | Specific cubin + JIT cache downloads |
| nvidia-cudnn | Whatever torch pulls | Pinned to `9.16.0.29` |
| torchaudio/torchvision | Whatever pip resolves | Reinstalled to match torch's CUDA version |
| lmms-eval, human-eval | Not installed | Installed |
| Stale packages | Not cleaned | Cleaned (`sgl-kernel`, `flash-attn`, etc. uninstalled first) |

If the CI script fails on your machine (e.g., missing `/etc/profile.d/sglang-ci.sh`, no flashinfer JIT cache URL), a reasonable fallback is:

```bash
pip install -e "python[dev]" --extra-index-url https://download.pytorch.org/whl/cu129
pip install sglang-kernel==$(grep -Po -m1 '(?<=sglang-kernel==)[0-9A-Za-z.\-]+' python/pyproject.toml) --force-reinstall
pip install scipy pytest mooncake-transfer-engine
```

### Run the specific test once

```bash
python -m pytest test/registered/models/test_transformers_models.py::TestTransformersFallbackTorchAO::test_mmlu -xvs
```

### Run it N times to detect flakiness (recommended: 10 iterations)

```bash
PASS=0; FAIL=0; for i in $(seq 1 10); do
  echo "=== Iteration $i ==="
  python -m pytest test/registered/models/test_transformers_models.py::TestTransformersFallbackTorchAO::test_mmlu -xvs 2>&1 | tail -5
  if [ $? -eq 0 ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
done
echo "Results: $PASS passed, $FAIL failed out of $((PASS+FAIL))"
```

### Bisection: test on suspect commit vs. its parent

For each SHA below, run the full setup + 10 test iterations:

```bash
# Helper function: checkout, install, run N iterations
bisect_test() {
  local sha=$1 label=$2 n=${3:-10}
  echo "============================================"
  echo "Testing: $label ($sha)"
  echo "============================================"
  git checkout "$sha"
  CUSTOM_BUILD_SGL_KERNEL=false bash scripts/ci/cuda/ci_install_dependency.sh
  local pass=0 fail=0
  for i in $(seq 1 "$n"); do
    echo "--- Iteration $i/$n ---"
    if python -m pytest test/registered/models/test_transformers_models.py::TestTransformersFallbackTorchAO::test_mmlu -xvs 2>&1 | tail -5; then
      pass=$((pass+1))
    else
      fail=$((fail+1))
    fi
  done
  echo ">>> $label: $pass passed, $fail failed out of $((pass+fail))"
}

# 1. Baseline: commit BEFORE the transformers rework (expect ~0% flake rate)
bisect_test 34ddf135f~1 "before-transformers-rework"

# 2. Primary suspect: the transformers rework itself (expect ~30% flake rate)
bisect_test 34ddf135f "transformers-rework"

# 3. After kernel fusion revert, transformers rework still present
bisect_test 95cdbce34fa9 "after-kernel-fusion-revert"

# 4. Current main (after PR #22210 fix, threshold 0.65→0.64)
bisect_test main "main-with-threshold-fix"
```

### Additional control points

```bash
# Before kernel fusion (baseline in regression window, before 34ddf135f)
bisect_test 8732b2e9c6f1 "before-34ddf135f-and-kernel-fusion-revert"

# With kernel fusion active + transformers rework (run 11 SHA, first failure)
bisect_test 97adf8a2909d "first-failure-sha"
```

### Comparing exact scores

To see the actual MMLU score (not just pass/fail), add this wrapper:

```bash
python -c "
from types import SimpleNamespace
from sglang.test.run_eval import run_eval
import subprocess, time, signal, os

# Launch server
proc = subprocess.Popen([
    'python', '-m', 'sglang.launch_server',
    '--model-path', 'meta-llama/Llama-3.1-8B-Instruct',
    '--model-impl', 'transformers',
    '--torchao-config', 'int4wo-128',
    '--port', '30000',
    '--host', '0.0.0.0',
])
time.sleep(120)  # wait for server startup

args = SimpleNamespace(
    base_url='http://127.0.0.1:30000',
    model='meta-llama/Llama-3.1-8B-Instruct',
    eval_name='mmlu',
    num_examples=64,
    num_threads=32,
)
metrics = run_eval(args)
print(f'MMLU score: {metrics[\"score\"]} ({int(metrics[\"score\"]*64)}/64)')
print(f'Threshold: 0.65 -> {\"PASS\" if metrics[\"score\"] >= 0.65 else \"FAIL\"} (old)')
print(f'Threshold: 0.64 -> {\"PASS\" if metrics[\"score\"] >= 0.64 else \"FAIL\"} (new)')
os.kill(proc.pid, signal.SIGTERM)
"
```

---

## 8. Diagnosis

### Classification: **Flaky regression introduced by `34ddf135f`**

This is a code change that marginally shifted MMLU accuracy for the `--model-impl transformers --torchao-config int4wo-128` path, pushing it from reliably above 0.65 to occasionally below.

### Evidence

| Condition | Result |
|-----------|--------|
| Before `34ddf135f` (runs 1-8, any partition) | 0/8 failures (0%) |
| After `34ddf135f` (runs 9-18, partition 5) | 3/10 failures (30%) |
| Kernel fusion active + transformers rework (runs 9-11) | 1/3 failures (33%) |
| Kernel fusion reverted + transformers rework (runs 12-18) | 2/7 failures (29%) |
| Same runner passing and failing (5090-a-runner-1) | Runs 6, 9: PASS; Run 15: FAIL |
| PR run on unrelated branch (24042766142, partition 5) | FAIL (score 0.640625) |

### Root Cause

Commit `34ddf135f` ("[Feature] Stronger transformers modeling backend with TP, PP, MoE, VLMs, and torch compile (#19163)") rewrote 1641 lines in `python/sglang/srt/models/transformers.py` plus changes to `model_loader/utils.py`, `model_runner.py`, and `scheduler.py`. This is the exact code path exercised by `TestTransformersFallbackTorchAO` (which uses `--model-impl transformers`).

The rework likely introduced a subtle change in model loading, weight handling, or execution flow that shifts the accuracy distribution of `int4wo-128` quantized Llama-3.1-8B from comfortably above 0.65 to straddling the boundary (~0.64-0.66). With only 64 MMLU examples, the difference between pass and fail is a single question.

### Timeline

| Date | Event |
|------|-------|
| Apr 01 00:34 | Run 1: PASS (partition 7, before all suspects) |
| Apr 01 20:16 | Commit `875a61599` merges — est_time 245→450, test moves to partition 5 |
| Apr 01 21:27 | Commit `cb0c2cbfd` merges — multi-thread weight loading enabled |
| Apr 01 21:46 | Commit `7a59e05dd` merges — kernel fusion in sampling |
| Apr 02 06:29 | Run 6: PASS (partition 5, first run with kernel fusion + weight loading) |
| **Apr 02 16:02** | **Commit `34ddf135f` merges — transformers backend rework (PRIMARY SUSPECT)** |
| Apr 03 00:30 | Run 9: PASS (first run including 34ddf135f) |
| **Apr 03 12:16** | **Run 11: FAIL (first observed failure, partition 5, 5090-a-runner-2)** |
| Apr 03 21:32 | Commit `ee9d922f5` merges — kernel fusion reverted |
| Apr 04 00:28 | Run 12: PASS (kernel fusion reverted, transformers rework remains) |
| **Apr 04 12:12** | **Run 14: FAIL (partition 5, 5090-b-runner-4)** |
| **Apr 04 18:11** | **Run 15: FAIL (partition 5, 5090-a-runner-1)** |
| Apr 05 12:12 | Run 16: PASS |
| Apr 06 19:08 | PR run 24042766142 job 70131079937: FAIL (score 0.640625, 5090-b-runner-6) |
| **Apr 06 22:32** | **PR #22210 merges — threshold relaxed 0.65 → 0.64** |
| Apr 07 18:21 | Run 18: PASS (post-fix) |

---

## 9. Fix Already Applied

### PR #22210: "[CI] Relax transformers MMLU threshold from 0.65 to 0.64"

| Field | Value |
|-------|-------|
| **Author** | alisonshao |
| **Merged** | 2026-04-06T22:32:09Z |
| **Merge commit** | `6f1412f4f58db045acb80d9477251075bf4b52e0` |
| **Change** | `mmlu_lower_bound` 0.65 → 0.64 in both test classes |

```diff
@@ -36,7 +36,7 @@ def setUpClass(cls):
-        cls.mmlu_lower_bound = 0.65
+        cls.mmlu_lower_bound = 0.64

@@ -86,7 +86,7 @@ def setUpClass(cls):
-        cls.mmlu_lower_bound = 0.65
+        cls.mmlu_lower_bound = 0.64
```

---

## 10. Recommendations

1. **Confirm `34ddf135f` as the cause** by running the local reproduction commands (Section 7) on `34ddf135f~1` vs `34ddf135f` with 10+ iterations each. If the parent commit shows 0% flake rate and `34ddf135f` shows ~30%, the regression is confirmed.

2. **The threshold fix (PR #22210) addresses the symptom but not the root cause.** If the transformers backend rework degraded accuracy, it may be worth investigating whether `34ddf135f` introduced an unintended behavioral change in the `int4wo-128` quantization path.

3. **Monitor the relaxed threshold.** If 0.64 also proves flaky, consider:
   - Lowering further to 0.62
   - Increasing `num_examples` from 64 to 128 (reduces standard error by ~30%)
   - Using a fixed random seed for deterministic MMLU question sampling

4. **The kernel fusion** (`7a59e05dd`) was already reverted (`ee9d922f5`). The revert did not fix the flakiness, confirming it is not the cause.

---

## Appendix A: Commits Between Last Pass (Run 10) and First Fail (Run 11)

These 16 commits landed between SHA `4d097047` (run 10, PASS) and `97adf8a2` (run 11, FAIL). None touch the transformers model path — the regression was already present from `34ddf135f` (included since run 9) and manifested stochastically at run 11:

```
97adf8a29 [misc] Add hint for kernel release trigger (#22036)
98ac40192 [Workflow] Fix kernel release build failures for aarch64 and wheel renaming (#22018)
838f815e9 [diffusion] CI: temporarily disable accuracy ci (#22031)
56ac9c993 [Fix] Add _MOE_TP to graph_capture for MoE models with ep>1 (#21907)
ac593fed9 [AMD][Dockerfile] Support build-arg AITER_COMMIT for rocm.Dockerfile (#21949)
cd75d54fc [Bugfix] Fix CUDA graph replay issues in trtllm_mla draft_extend (#21987)
4f84ce580 [CI] ci: add test_http_server_auth.py to CI (#21866)
658a2813d [NPU] Update CI Dependency (#21578)
d07d0a15c [AMD] Add MiniMax-M2.5 nightly perf benchmarks for MI30x and MI35x (#21524)
7431db739 [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 (#21511)
ad0516d9c [NPU] optimize glm4.7 (#19246)
d82097a0d [PD] Tiny register info field cleanup for mooncake backend (#22016)
24f52e66d fix: remove duplicate words in comments (#22007)
4cc970290 [CI] Fix duplicate job names that bypass branch protection (#22001)
6b876a771 [ROCM][RL] Shuffle Weight In-Place to Preserve Parameter Attributes (#21825)
75de47968 [Misc] Update CI permission (#22014)
```

## Appendix B: Files Changed by Primary Suspect `34ddf135f`

```
 python/sglang/srt/configs/model_config.py               |   15 +-
 python/sglang/srt/disaggregation/encode_receiver.py     |    5 +
 python/sglang/srt/managers/io_struct.py                  |    2 +
 python/sglang/srt/managers/multimodal_processor.py       |   32 +-
 python/sglang/srt/managers/scheduler.py                  |   57 +-
 python/sglang/srt/managers/tokenizer_manager.py          |   11 +-
 python/sglang/srt/model_executor/model_runner.py         |   10 +
 python/sglang/srt/model_loader/utils.py                  |  156 +-
 python/sglang/srt/models/qwen2.py                        |    1 +
 python/sglang/srt/models/transformers.py                 | 1641 ++++++++++++++++++--
 python/sglang/srt/models/utils.py                        |  164 ++
 python/sglang/srt/multimodal/processors/qwen_vl.py      |    1 +
 python/sglang/srt/multimodal/processors/transformers_auto.py | 215 +++
 test/registered/models/test_transformers_backend_eval.py  |   43 +
 14 files changed, 2169 insertions(+), 184 deletions(-)
```
