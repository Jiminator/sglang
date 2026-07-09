# OpenHands Agentic Sweep (evalscope client)

Multi-turn agentic serving benchmark for SGLang, driven end-to-end by the
[evalscope](https://github.com/modelscope/evalscope) perf client
(`--dataset swe_smith --multi-turn`). Replays OpenHands coding-agent
conversations shaped to a fixed recipe against a locally-launched SGLang
server, sweeping concurrency 1 → 16.

Dependencies: **sglang** (this repo, for `sglang.launch_server` and the
tokenizer helper) and **evalscope** (pinned by commit in `sweep.sh`).

## Quickstart

```bash
cd benchmark/agentic_evalscope
./sweep.sh
```

One invocation does everything: installs the pinned evalscope, builds (or
reuses) the dataset, then for each config launches the server, runs the
concurrency sweep, tears the server down, and writes the report:

```
outputs/<ts>/
├── DATASET.openhands            # workload recipe marker
├── <config>/parallel_N_number_M/benchmark_{summary,percentile,args}.json
├── server_logs/<config>.log     # server log (accept-length source)
├── server_logs/<config>.startup # launch->ready seconds
├── metrics.txt                  # bench_report.py tables
└── pareto.png                   # throughput/latency Pareto chart
```

Environment overrides:

| Variable | Default | Effect |
|---|---|---|
| `PORT` | `8002` | server + client port |
| `PAD_SOURCE` | `openscience` | dataset pad filler (`openscience` \| `random`); changing it rebuilds the dataset |
| `REBUILD_DATASET` | `0` | `1` forces a dataset rebuild |
| `PIP_NO_DEPS` | unset | `1` installs evalscope without its dep tree (deps must already be present) |

## Workload recipe

Target ISL ≈ 80k, OSL = 220/turn, expected KV-cache hit ≈ 92%:

| Parameter | Value |
|---|---|
| first turn (system pad + real first user msg) | 74,160 bare tokens |
| each subsequent turn (real + synthetic pad) | 753 bare tokens |
| turns per conversation | 13 |
| unique conversations built | 128 |
| concurrency steps (`--parallel` / `--number`) | 1/4, 2/8, 4/8, 8/16, 16/32 |

`build_openhands_padded_dataset.py` pads every OpenHands trajectory
(`nebius/SWE-rebench-openhands-trajectories`) to exactly this shape with a
unique-per-conversation synthetic system prompt + per-turn padding, sized by
exact bare-token targets (encode → slice → decode). evalscope's offset
rotation then keeps each sweep step on fresh conversations — no
cycling-induced cache-hit inflation. See the builder's docstring for the
padding rationale and pad sources.

Server configs live in `configs/` (GLM-5.2-NVFP4, TP4, FP8 KV cache,
EAGLE 5/1/6; `attn_tp4_moe_ep4` adds `--ep-size 4`). hicache must stay OFF
and no profiler flags may be set — both change the numbers.

## Metric caveats

- **Accept length**: evalscope's client-side `Decoded Tok/Iter` is
  `(completion_tokens-1)/(n_chunks-1)`; under concurrency SSE chunks coalesce,
  so it drifts and can exceed the physical ceiling of `num_spec+1`.
  `bench_report.py` therefore sources accept length from the server log's own
  spec-decode counters (`accept len:` in the `Decode batch` lines,
  `#running-req`-weighted per step) and only falls back to the evalscope value
  (with a warning) when no server log is present.
- **TTFT** is reported as the p50 — the first request of each step pays the
  cold start. TPOT / ITL / output throughput are steady-state means.
- **Cross-client comparisons** (against a different benchmark client): compare
  throughput, e2e latency, accept length, and KV-hit rate. TTFT/ITL/TPOT-mean
  are stamped differently per client and are not comparable.
- Under real EAGLE acceptance, per-step throughput tracks the measured accept
  length, which is content-dependent and differs by conversation slice —
  compare accept-normalized throughput across runs, not raw.
