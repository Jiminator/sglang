# DeepSeek-V4-Pro Accuracy Eval Bring-up — Handoff

Runbook + findings for serving **DeepSeek-V4-Pro** (FP4 checkpoint) on an
8× B200 node and running accuracy evals (GPQA-Diamond, AIME25) against it.
Written to unblock the nightly accuracy-eval migration.

## TL;DR

- V4-Pro is a **reasoning model**; thinking is enabled **server-side via env
  vars**, not a chat template or client flag. With thinking on it scores
  ~90% GPQA-Diamond; without it (Non-think) it scores ~70%.
- The reference accuracy comes from running with a **large token budget** so
  the long reasoning chains don't truncate.
- The host image is built for an older `main`; the env must be re-aligned to
  the current `main` deps before V4-Pro will launch.

## 1. Environment

The default `dev-cu13` image ships an older-main stack (torch 2.9.1,
sglang-kernel 0.4.1.post1). Current `main` needs **torch 2.11.0 +
sglang-kernel 0.4.3** (V4-Pro hard-asserts kernel ≥ 0.4.3). Align with the
repo's own installer:

```bash
UV_BREAK_SYSTEM_PACKAGES=1 bash scripts/ci/cuda/ci_install_dependency.sh
```

(`uv pip --system` otherwise fails on the externally-managed `/usr`
interpreter.) `nvcc` lives at `/usr/local/cuda/bin` (not on PATH by default)
and is needed for the runtime JIT kernels.

## 2. Weights

V4-Pro/Flash weights are pre-staged on `/cluster-storage/models` as a HF
cache. The HF **snapshot symlinks get cache-evicted periodically** (blobs
stay, 806 GB), which makes the path fail to load. Re-hydrate the snapshot
from the existing blobs (fast, no re-download):

```bash
hf download deepseek-ai/DeepSeek-V4-Pro --cache-dir /cluster-storage/models
# -> .../models--deepseek-ai--DeepSeek-V4-Pro/snapshots/<commit>/
```

`/cluster-storage` is a **virtiofs** mount: 8 concurrent rank readers contend
badly, so a full 806 GB load takes ~40–60 min and is uneven across ranks.
`--model-loader-extra-config '{"enable_multithread_load": true}'` helps.

## 3. Serve recipe (8× B200, FP4 checkpoint)

```bash
SGLANG_DEFAULT_THINKING=1 SGLANG_REASONING_EFFORT=max \
PATH=/usr/local/cuda/bin:$PATH \
python3 -m sglang.launch_server \
  --model-path /cluster-storage/models/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/<commit> \
  --tp 8 --trust-remote-code \
  --moe-runner-backend flashinfer_mxfp4 \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --context-length 409600 \
  --dist-timeout 3600 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --host 0.0.0.0 --port 30000
```

Required, learned the hard way:

- **`SGLANG_DEFAULT_THINKING=1 SGLANG_REASONING_EFFORT=max`** — turns on
  Think-Max server-side; every chat response then includes reasoning. Without
  these, V4-Pro serves Non-think (much lower accuracy). (`SGLANG_REASONING_EFFORT`
  is being renamed to `SGLANG_DSV4_REASONING_EFFORT`; both work for now.)
- **`--moe-runner-backend flashinfer_mxfp4`** — for the FP4 checkpoint. With
  the default `auto` backend the triton MoE path crashes during CUDA-graph
  capture with `AssertionError: Hidden size mismatch`.
- **`--dist-timeout 3600`** — slow/uneven virtiofs loading otherwise trips the
  NCCL collective timeout (~10 min) → "Rank N scheduler died (exit -15)".
- **`UNBALANCED_MODEL_LOADING_TIMEOUT_S`** (model_runner.py, default 480 s) is a
  separate post-load straggler `monitored_barrier`. On slow virtiofs the
  fastest rank finishes >8 min before the slowest → barrier fails with
  "TP rank 0 could finish ... other ranks didn't finish loading". Raise it
  (e.g. 2400) for very large models on slow storage.
- Cap `--context-length` to fit the eval (model default is 1M). Memory is
  **context-flat**: the KV pool is memory-bound (~2.07M tokens, ~25 GB/GPU
  free after capture at 131K / 256K / **400K** alike), so a large context does
  not OOM — it only lowers concurrency.

## 4. Accuracy eval

`sgl-eval` is the recommended tool (`pip install
git+https://github.com/sgl-project/sgl-eval`); plain
`python3 -m sglang.test.run_eval --eval-name gpqa ...` works too since
thinking is server-side.

```bash
# GPQA-Diamond (reference ~90% on Pro)
sgl-eval run gpqa --model deepseek-ai/DeepSeek-V4-Pro \
  --n-repeats 16 --max-tokens 400000 --temperature 1.0 --top-p 1.0 --thinking \
  --base-url http://localhost:30000/v1 --out-dir /sgl-workspace/logs

# AIME25 (reference ~97.5% on Pro)
sgl-eval run aime25 --model deepseek-ai/DeepSeek-V4-Pro \
  --n-repeats 16 --max-tokens 400000 --temperature 1.0 --top-p 1.0 --thinking \
  --base-url http://localhost:30000/v1 --out-dir /sgl-workspace/logs
```

Recommended sampling per the model card: `temperature=1.0, top_p=1.0`. Use a
large `--max-tokens` (Pro: 400000) and `--n-repeats 16` to average out the
high variance of Think-Max + the small AIME set (30 problems).

## 5. Observed results

| Config | GPQA-Diamond |
|---|---|
| Non-think (no env vars, max_tokens 2048) | **0.707** |
| Think-Max, 120K context cap (single pass) | 0.75 aggregate, but **0.96 on non-truncated answers** (22% of chains truncated at 120K) |
| Think-Max, target recipe (400K, 16 repeats) | reference **~0.90** |

Takeaway: V4-Pro's GPQA capability is genuinely ~90%. A low aggregate is
almost always a serving issue — Non-think mode, too-small `max_tokens`/context
truncating the reasoning, or the FP8 KV cache (auto-enabled; logs
"Using FP8 KV cache ... may be less accurate"). The Think-Max reasoning
chains are large: mean ~47K tokens at a 120K cap, with a long tail that needs
the 400K budget to finish and emit the final `Answer:`.

## 6. Throughput / timing notes

- Weight load: ~40–60 min from virtiofs (varies; multithread load helps).
- Post-load: KV alloc + first-run DeepGEMM/dsv4 JIT compile + CUDA-graph
  capture, then ready.
- Decode at 8 concurrent, long context: ~540–650 tok/s aggregate. A full
  198-question GPQA single pass at 400K is ~5–7 h of generation
  (throughput-bound; ×16 repeats for the reference number).

## 7. Gotchas / failure modes

- **Do not mutate the editable checkout (`git checkout`/`reset`) while a server
  is running.** The server JIT-compiles DeepGEMM/jit_kernel code from on-disk
  source at runtime (new GEMM shapes mid-generation); changing the source
  under it crashes a rank → `gloo: Connection closed by peer` → SIGQUIT
  cascade. Do git work in a separate `git worktree`.
- **Shared GPU box:** once you free the GPUs, another tenant's job can claim
  them within seconds (memory grows, no PID visible via `nvidia-smi` =
  different namespace). Plan relaunches accordingly.
- Snapshot eviction (see §2) recurs — re-hydrate right before launch.

## 8. Open items

- Capture the clean full-recipe aggregate (198 GPQA × 16 repeats, 400K) and
  AIME25 once an 8× B200 node is free; compare to the ~90% / ~97.5%
  references.
- Consider a `compile_deep_gemm` pre-pass to move JIT compile out of the hot
  path (logs recommend it).
