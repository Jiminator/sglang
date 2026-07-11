<!--
Saved draft body for the blog-reproduction PR.
The original draft PR (#30851) was closed to avoid attention pre-publication;
recreate with:
  gh pr create --repo sgl-project/sglang --base release/v0.5.15 \
      --head Jiminator:glm-nvfp4-blog-repro --draft \
      --title "[DO NOT MERGE] Reproduction scripts for the GLM NVFP4 GB300/B300 blog figures" \
      --body-file <this file, without this comment block>
Branch: Jiminator/sglang @ glm-nvfp4-blog-repro (single commit 2de921acd on release/v0.5.15 @ f63458b5b).
-->

**Do not merge** — this is a companion branch for the GLM NVFP4 GB300/B300 blog post so readers can reproduce its figures. Everything lives under `benchmark/glm_nvfp4_blog/`; no SGLang source is touched. The only dependencies are SGLang (this branch, based on `release/v0.5.15`) and [evalscope](https://github.com/modelscope/evalscope) pinned by commit as the benchmark client.

### One command per machine

```bash
gb300/run_all.sh    # 4xGB300 node  -> figure_gb300.png (TP=4 + TEP=4, all three curves)
b300/run_all.sh     # 8xB300 node   -> figure_b300.png  (TP=8 + TEP=8, all three curves)
```

On a bare machine (SGLang from this branch installed) the master script handles everything: installs the pinned evalscope on first use, creates the day-0 SGLang checkout (commit `22dce5720` from the public `glm-opt` branch, in a git worktree), builds the datasets, runs the six sweeps — day-0 first, then the v0.5.15 models — and renders the figure. Roughly 2–3 hours end to end.

### Expected output

| GB300 | B300 |
|---|---|
| ![gb300](https://raw.githubusercontent.com/Jiminator/sglang/glm-nvfp4-blog-repro/benchmark/glm_nvfp4_blog/expected_figure_gb300.png) | ![b300](https://raw.githubusercontent.com/Jiminator/sglang/glm-nvfp4-blog-repro/benchmark/glm_nvfp4_blog/expected_figure_b300.png) |

**Workload**: OpenHands multi-turn agentic replay — ~80k mean input tokens/request, 220 output tokens/turn, 13 turns/conversation, ~92% aggregate prefix-cache hit, real EAGLE (5/1/6) speculative acceptance. Datasets are built deterministically from public HF datasets (`nebius/SWE-rebench-openhands-trajectories` padded with `nvidia/OpenScienceReasoning-2`).

### Going granular

The masters are thin wrappers — every level runs standalone and resumes safely (each sweep skips itself when its results exist):

```
gb300/run_all.sh                    all three curves + full figure
├── gb300/run_day0.sh               one curve (both panels) + single-curve figure
│   ├── gb300/run_day0_tp4.sh       one sweep: server up -> client -> server down
│   └── gb300/run_day0_tep4.sh
├── gb300/run_glm52_v0515.sh  (…_tp4.sh, …_tep4.sh)
└── gb300/run_glm51_v0515.sh  (…_tp4.sh, …_tep4.sh)
```

`b300/` mirrors this at TP8/TEP8. Server flags are spelled out in the leaf scripts: v0.5.15 sweeps run with `SGLANG_OPT_USE_TOPK_V2=1` + `SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1` (the blog configuration); day-0 sweeps use launch-day flags (no `--bf16-gemm-backend`, `--cuda-graph-max-bs` spelling, no fused-top-k/deferred-finalize env vars) — that's the point of the curve. Plot any subset with `python3 plot_figure.py <gb300|b300|all> [--curves …]`.

### ISL ablation (GB300)

```bash
cd isl_ablation
./run_isl_client.sh v0515       # 80K -> 1M context ladder at c=1, ~2-3 h
./run_isl_client.sh day0
python3 plot_isl_figure.py      # -> isl_ablation.png
```

![isl](https://raw.githubusercontent.com/Jiminator/sglang/glm-nvfp4-blog-repro/benchmark/glm_nvfp4_blog/isl_ablation/expected_isl_ablation.png)

### Notes

- Expect ~±2–4% run-to-run variance per point (widest at c=1): match the expected figures in shape and ordering, not to the pixel. The ~92% cache-hit and ~5.0 acceptance-length invariants in `benchmark_summary.json` confirm a faithful replay.
- On a fresh machine the very first sweep pays one-time JIT kernel compilation; delete that sweep's `results/...` directory and re-run it for a clean point (compiled-kernel caches persist on disk).
- Day-0 sweeps are verified against `sgl-kernel 0.4.4` / `flashinfer 0.6.12` / `torch 2.11`.
