# Debugging `stage-b-test-1-gpu-small` CI Locally

Guide for reproducing the `stage-b-test-1-gpu-small` CI stage locally, with focus on debugging `TestTransformersFallbackTorchAO.test_mmlu`.

---

## How `stage-b-test-1-gpu-small` Works

### Overview

- **Runner**: RTX 5090 (1 GPU, SM120, 32GB)
- **Matrix**: 8 partitions (load-balanced by estimated time)
- **Workflow file**: `.github/workflows/pr-test.yml` (line 661)
- **Suite runner**: `test/run_suite.py`
- **Test execution**: `python/sglang/test/ci/ci_utils.py`
- **Test registration**: `python/sglang/test/ci/ci_register.py`

### Steps Executed (in order)

| # | Step | What it does |
|---|------|-------------|
| 1 | **Checkout** | `actions/checkout@v4` — checks out PR commit |
| 2 | **check-stage-health** | Queries GitHub API for prior failed jobs; fast-fails if any found |
| 3 | **check-maintenance** | Checks GitHub issue #21065 for maintenance mode |
| 4 | **Download artifacts** | Downloads pre-built `sgl-kernel` wheels (only if kernel code changed) |
| 5 | **Install dependencies** | Runs `scripts/ci/cuda/ci_install_dependency.sh` (~20 min timeout) |
| 6 | **Run test** | `python3 run_suite.py --hw cuda --suite stage-b-test-1-gpu-small --auto-partition-id $PARTITION --auto-partition-size 8` |
| 7 | **Upload coredumps** | Only on failure |

### Environment Variables Set by CI

```bash
SGLANG_IS_IN_CI=true
SGLANG_CUDA_COREDUMP=1
SGLANG_JIT_DEEPGEMM_FAST_WARMUP=true
```

---

## What the Install Script Does

**Script**: `scripts/ci/cuda/ci_install_dependency.sh`

Full installation sequence:

1. **Config**: `CU_VERSION=cu129`, `NVIDIA_CUDNN_VERSION=9.16.0.29`, `NVIDIA_NVSHMEM_VERSION=3.4.5`
2. **Detect GPU**: auto-detect compute capability via `nvidia-smi --query-gpu=compute_cap`. 5090 = SM120, so `IS_BLACKWELL=1` (compute_cap >= 10.0). Uses **pip** (not uv) on Blackwell.
3. **Kill stale processes**: `python3 python/sglang/cli/killall.py`
4. **APT packages**:
   ```
   python3 python3-pip python3-venv python3-dev git libnuma-dev libssl-dev pkg-config
   libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils
   ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
   ```
5. **Site hygiene**: clear torchinductor cache, remove broken dist-info dirs, install protoc
6. **Pip setup**: upgrade pip, uninstall stale `sgl-kernel`/`sglang`/`flash-attn`
7. **Flashinfer version check**: keep installed versions if they match `python/pyproject.toml`, uninstall if mismatch
8. **Install sglang**:
   ```bash
   pip install -e "python[dev,runai,tracing]" --extra-index-url https://download.pytorch.org/whl/cu129 --break-system-packages
   ```
9. **Install sglang-kernel**: from PyPI (`sglang-kernel==$VERSION`) or from wheel artifact if kernel code changed
10. **Install sglang-router**: `pip install sglang-router`
11. **Download flashinfer**: jit-cache (`ci_download_flashinfer_jit_cache.sh`) and cubin (`ci_download_flashinfer_cubin.sh`)
12. **Extra deps**:
    ```bash
    pip install mooncake-transfer-engine==0.3.10.post1 nvidia-cuda-nvrtc-cu12 py-spy scipy huggingface_hub[hf_xet] pytest
    ```
    - On Blackwell (5090): **skips lmms-eval install**
    - Uninstalls xformers
13. **Fix deps**:
    - Reinstall torchaudio/torchvision if torch CUDA version != `cu129`
    - Pin `nvidia-nvshmem-cu12==3.4.5`
    - Pin `nvidia-cudnn-cu12==9.16.0.29`
    - Force-reinstall `nvidia-cutlass-dsl>=4.4.1`
    - Download kernels community: `kernels download python && kernels lock python`
    - Install human-eval from source
14. **Prepare runner**: cleanup HF cache (`cleanup_hf_cache.py`), prevalidate cached models (`prevalidate_cached_models.py`)
15. **Verify imports**: `pip list`, verify torch CUDA version, verify cutlass imports

---

## Test Partitioning

`test_transformers_models.py` (est_time=450s) lands in **partition 5**.

Other tests in partition 5:

| File | est_time |
|------|----------|
| `test/registered/models/test_transformers_models.py` | 450 |
| `test/registered/openai_server/function_call/test_tool_choice.py` | 250 |
| `test/registered/openai_server/features/test_openai_server_hidden_states.py` | 186 |
| `test/registered/model_loading/test_runai_model_loader.py` | 120 |
| `test/registered/models/test_reward_models.py` | 103 |
| `test/registered/sampling/test_penalty.py` | 82 |
| `test/registered/core/test_request_queue_validation.py` | 47 |
| `test/registered/layers/mamba/test_causal_conv1d.py` | 25 |
| `test/registered/quant/test_triton_scaled_mm.py` | 8 |
| `test/registered/quant/test_bnb.py` | 5 |

**Note**: Partition assignment is deterministic (LPT heuristic sorted by est_time DESC, filename ASC) but will shift if tests are added/removed from this suite.

---

## How Tests Are Executed

`run_suite.py` calls `run_unittest_files()` which runs each test file as:

```bash
python3 /absolute/path/to/test_file.py -f
```

- `-f` = unittest failfast mode (stop at first failure within the file)
- **Timeout per file**: 1800s (base 1200s + 600s retry increase)
- **Retry**: enabled by default in CI, max 2 attempts, 60s wait between retries
- **Retriable failures**: `AssertionError` with comparison patterns, `accuracy`, `score`, `latency`, `throughput`, `timeout`
- **Non-retriable failures**: `SyntaxError`, `ImportError`, `RuntimeError`, `CUDA out of memory`, `Segmentation fault`, etc.

---

## Reproducing Locally

### Step 1: Set environment variables

```bash
export SGLANG_IS_IN_CI=true
export SGLANG_CUDA_COREDUMP=1
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=true
```

### Step 2: Install dependencies (same script CI uses)

```bash
# From repo root. Needs root for apt-get. Will kill existing GPU processes.
CUSTOM_BUILD_SGL_KERNEL=false bash scripts/ci/cuda/ci_install_dependency.sh
```

`CUSTOM_BUILD_SGL_KERNEL=false` installs sgl-kernel from PyPI (the normal path when kernel code hasn't changed).

If `/etc/profile.d/sglang-ci.sh` doesn't exist on your machine, create an empty file:
```bash
sudo touch /etc/profile.d/sglang-ci.sh
```

### Step 3: Run the test

**Option A** — Full partition 5 (exactly what CI runs):

```bash
cd test/
python3 run_suite.py --hw cuda --suite stage-b-test-1-gpu-small \
    --auto-partition-id 5 --auto-partition-size 8
```

**Option B** — Full partition 5 with CI retry behavior:

```bash
cd test/
python3 run_suite.py --hw cuda --suite stage-b-test-1-gpu-small \
    --auto-partition-id 5 --auto-partition-size 8 \
    --enable-retry --max-attempts 2 --retry-wait-seconds 60
```

**Option C** — Just the flaky test file (same invocation as CI):

```bash
python3 test/registered/models/test_transformers_models.py -f
```

**Option D** — Just the specific flaky test class+method:

```bash
python3 -m pytest test/registered/models/test_transformers_models.py::TestTransformersFallbackTorchAO::test_mmlu -v
```

Or with unittest:
```bash
python3 -m unittest test.registered.models.test_transformers_models.TestTransformersFallbackTorchAO.test_mmlu
```

---

## About the Flaky Test

`TestTransformersFallbackTorchAO.test_mmlu` (in `test/registered/models/test_transformers_models.py`):

- Launches a server with `--model-impl transformers --torchao-config int4wo-128`
- Runs 64 MMLU examples with 32 threads
- Asserts score >= 0.63
- Skipped on AMD GPUs (`@unittest.skipIf(is_hip(), ...)`)

**Why it's flaky**: int4 weight-only quantization introduces variance, and a 64-example MMLU subset is small enough that the score can fluctuate around the 0.63 threshold. CI classifies this as "retriable" because the failure output matches the `score` and `AssertionError` retriable patterns.

---

## Key Differences: CI vs Local

| Aspect | CI (5090 runner) | Local |
|--------|-----------------|-------|
| GPU | RTX 5090 (SM120, 32GB) | Check with `nvidia-smi` |
| `/etc/profile.d/sglang-ci.sh` | Exists (runner-specific) | Likely missing — create empty file |
| Docker container | Yes (pre-built image) | No (bare metal) |
| `IS_BLACKWELL` | `1` (auto-detected, SM120 >= 10.0) | Depends on your GPU |
| Retry | Enabled (2 attempts, 60s wait) | Not enabled by default with options C/D |
| `continue-on-error` | Off for PRs, on for scheduled | Off by default locally |

---

## Relevant Source Files

| File | Purpose |
|------|---------|
| `.github/workflows/pr-test.yml:661-717` | Job definition |
| `scripts/ci/cuda/ci_install_dependency.sh` | Full dependency install script |
| `test/run_suite.py` | Suite runner (discovery, partition, dispatch) |
| `python/sglang/test/ci/ci_register.py` | Test registration and LPT partitioning |
| `python/sglang/test/ci/ci_utils.py` | Test execution, retry logic, failure classification |
| `test/registered/models/test_transformers_models.py` | The test file itself |
