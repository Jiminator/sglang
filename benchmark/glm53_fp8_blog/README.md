# GLM-5.3 FP8 on SGLang — figure reproduction

Everything needed to reproduce the GLM-5.3 FP8 GB300/B300 performance figures,
using **SGLang from the `lmsysorg/sglang:latest` docker image** and
**[evalscope](https://github.com/modelscope/evalscope)** (pinned by commit) as
the benchmark client. Nothing is checked out or patched at run time — the
serving code is exactly what the image ships.

Workload: OpenHands multi-turn agentic replay — mean input ≈ 80k tokens/request,
220 output tokens/turn, 13 turns/conversation, ~92% aggregate prefix-cache hit
rate, real EAGLE speculative acceptance (nothing simulated).

## Quick start — one command per machine

Start the container (adjust the HF cache mount to where your models live):

```bash
docker run --gpus all --network host --ipc host --shm-size 600g -it \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    lmsysorg/sglang:latest bash
```

Inside it:

```bash
git clone --depth 1 -b glm53-fp8-blog-repro https://github.com/Jiminator/sglang.git
cd sglang/benchmark/glm53_fp8_blog
b300/run_all.sh     # on an 8xB300 node  -> figure_b300.png  (TP=8 + TEP=8)
gb300/run_all.sh    # on a 4xGB300 node  -> figure_gb300.png (TP=4 + TEP=4)
```

On a bare machine each master handles everything itself: it installs the pinned
evalscope on first use, builds the dataset, runs the two sweeps (TP first,
then TEP), and renders the single-panel figure with both curves. Expect
roughly 1–2 hours end to end (dominated by the model download if the HF cache
is cold). Compare against `expected_figure_b300.png`.

## Script hierarchy — go as granular as you like

Every level is safe to run on its own: the per-sweep scripts skip themselves
when their results already exist, so the masters simply resume after an
interruption.

```
b300/run_all.sh              both sweeps + the figure
├── b300/run_glm53_tp8.sh    one sweep: server up -> client -> server down
└── b300/run_glm53_tep8.sh
```

`gb300/` mirrors this at TP4/TEP4. Only the leaf scripts touch the server and
the client (`run_client.sh`); the masters just sequence and plot.

- Server flags live in the leaf scripts; the two panels differ only by
  `--ep-size` (expert parallelism). Both run with `SGLANG_OPT_USE_TOPK_V2=1`
  and real speculative acceptance (MTP/EAGLE k=5).
- `run_client.sh` builds the per-model dataset on first use (~10–20 min,
  cached under `datasets/`) and runs concurrency 1→8 in a single evalscope
  invocation (offset rotation keeps every step on fresh conversations).
- Replot any time with `python3 plot_figure.py <b300|gb300|all>`. The x-axis
  defaults to p90 interactivity (1000 / p90 per-request TPOT, where
  TPOT = (latency − TTFT)/(output_tokens − 1)); `--x-tpot avg|p50|p90|p99`
  selects a different aggregation.

## Requirements and expectations

- ~760 GB of HF cache for `zai-org/GLM-5.3` (FP8). The scripts auto-download
  it via `--model-path` on first boot, so pre-stage it to avoid a long mid-run
  download. If your HF cache is on slow/network storage, copy the snapshot to
  fast local disk and point `HF_HUB_CACHE` (or `HF_HOME`) at it.
- The client and `datasets>=4.0` are installed automatically on first run (the
  OpenHands dataset needs the `List` feature type that older `datasets` lacks).
- On a fresh machine the very first sweep pays one-time JIT kernel compilation;
  its c=1 point can read a few percent low. Compiled-kernel caches persist on
  disk, so a re-run of just that sweep (delete its `results/...` directory) is
  clean.
- Run-to-run variance is ~±2–4% per point (widest at c=1): match the expected
  figure in shape and ordering, not to the pixel. The ~92% cache-hit and ~5.0
  acceptance-length invariants in each `benchmark_summary.json` / server log
  confirm a faithful replay.
