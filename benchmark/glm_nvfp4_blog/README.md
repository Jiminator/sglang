# GLM NVFP4 on SGLang — blog figure reproduction

Everything needed to reproduce the figures from the GLM NVFP4 GB300/B300 blog
post, using only **SGLang** (this branch, based on `release/v0.5.15`) and
**[evalscope](https://github.com/modelscope/evalscope)** (pinned by commit) as
the benchmark client.

Workload: OpenHands multi-turn agentic replay — mean input ≈ 80k tokens/request,
220 output tokens/turn, 13 turns/conversation, ~92% aggregate prefix-cache hit
rate, real EAGLE speculative acceptance (nothing simulated).

## Quick start — one command per machine

```bash
gb300/run_all.sh    # on a 4xGB300 node  -> figure_gb300.png (TP=4 + TEP=4, all 3 curves)
b300/run_all.sh     # on an 8xB300 node  -> figure_b300.png  (TP=8 + TEP=8, all 3 curves)
```

On a bare machine each master handles everything itself: it installs the pinned
evalscope on first use, creates the day-0 SGLang checkout, builds the datasets,
runs the six sweeps (day-0 first, then the v0.5.15 models), and renders the
figure. Expect roughly 2–3 hours end to end. Compare against
`expected_figure_gb300.png` / `expected_figure_b300.png`.

## Script hierarchy — go as granular as you like

Every level is safe to run on its own and in any order: the per-sweep scripts
skip themselves when their results already exist, so the masters simply resume
after an interruption.

```
gb300/run_all.sh                 all three curves + the full figure
├── gb300/run_day0.sh            one curve (both panels) + a single-curve figure
│   ├── gb300/run_day0_tp4.sh    one sweep: server up -> client -> server down
│   └── gb300/run_day0_tep4.sh
├── gb300/run_glm52_v0515.sh
│   ├── gb300/run_glm52_v0515_tp4.sh
│   └── gb300/run_glm52_v0515_tep4.sh
└── gb300/run_glm51_v0515.sh
    ├── gb300/run_glm51_v0515_tp4.sh
    └── gb300/run_glm51_v0515_tep4.sh
```

`b300/` mirrors this at TP8/TEP8. Only the leaf scripts touch the server and
the client (`run_client.sh`); everything above them just sequences and plots —
that is what makes the levels composable.

- Server flags live in the leaf scripts. The v0.5.15 sweeps run with
  `SGLANG_OPT_USE_TOPK_V2=1` and `SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1` (the
  blog configuration). The day-0 sweeps launch the launch-day SGLang tree
  (commit `22dce5720`, fetched from the public `glm-opt` branch into a git
  worktree) with launch-day flags.
- `run_client.sh` builds the per-model dataset on first use (~10–20 min,
  cached under `datasets/`) and runs concurrency 1→8 in a single evalscope
  invocation (offset rotation keeps every step on fresh conversations).
- Plot any subset yourself with
  `python3 plot_figure.py <gb300|b300|all> [--curves day0,glm52_v0515,glm51_v0515]`.
  `all` renders the combined 2×2 figure once both machines' `results/` are
  merged into one checkout.

## ISL ablation (GB300 only)

```bash
cd isl_ablation
./run_isl_client.sh v0515        # 80K -> 1M context ladder at c=1, ~2-3 h
./run_isl_client.sh day0         # same ladder on the day-0 tree
python3 plot_isl_figure.py       # -> isl_ablation.png  (compare: expected_isl_ablation.png)
```

The ladder driver boots its own server per rung (context length is a boot-time
setting), scaling only the dataset first-turn budget and `--context-length`.

## Requirements and expectations

- ~470 GB of HF cache per model. If your cache is on slow network storage,
  copy the snapshot to local disk and point `--model-path` at it (keep
  `--served-model-name nvidia/GLM-5.2-NVFP4`).
- Day-0 sweeps are verified against `sgl-kernel 0.4.4`, `flashinfer 0.6.12`,
  `torch 2.11` (the day-0 tree predates newer kernel wheels).
- On a fresh machine the very first sweep pays one-time JIT kernel compilation;
  its c=1 point can read a few percent low. Compiled-kernel caches persist on
  disk, so a re-run of just that sweep (delete its `results/...` directory) is
  clean.
- Run-to-run variance is ~±2–4% per point (widest at c=1): match the expected
  figures in shape and ordering, not to the pixel. The ~92% cache-hit and ~5.0
  acceptance-length invariants in each `benchmark_summary.json` / server log
  confirm a faithful replay.
