# Agentic Multi-Turn Serving Benchmark

Replays multi-turn agentic conversations (OpenHands SWE trajectories by
default) against a live server: turn 1 is sent, the **real** server reply is
appended, then the turn-2 delta, and so on. Reports TTFT/TPOT/ITL/throughput,
KV-cache hit rate, and speculative-decoding accept length.

## Quick start

```bash
# Server: --enable-cache-report enables cache-hit reporting
python -m sglang.launch_server --model-path <model> --enable-cache-report

# Concurrency sweep with fresh conversations per step
./sweep.sh
```

`sweep.sh` runs the `(num_prompts, concurrency)` pairs
`(4,1) (8,2) (8,4) (16,8) (32,16)`, advancing `--agentic-offset` by the
conversations consumed so far — replaying a conversation whose prefix is
already cached would inflate the measured cache-hit rate. `--flush-cache`
clears prior-step cache pollution; `--warmup-requests 0` avoids pre-warming
conversation #1's prefix. A single point:

```bash
python3 -m sglang.benchmark.serving --backend sglang-oai-chat \
  --dataset-name agentic --num-prompts 8 --max-concurrency 2 \
  --cache-report --flush-cache --warmup-requests 0
```

## Workload shape

Turn 1 is a unique synthetic system prompt plus the real first user message;
each later turn is one user message sized exactly to
`--agentic-subsequent-turn-len` bare tokens. The defaults (74160 / 753 / 13
turns / OSL 220) give ISL ≈ 80k per conversation and ~92% expected KV-cache
hit rate at full depth. Pads come from `nvidia/OpenScienceReasoning-2`
reasoning traces (control markup stripped) or seeded random text; both are
sized to identical shapes. Sources, caching, and the full flag list are
documented in `docs_new/docs/developer_guide/bench_serving.mdx`.

## Reading the report

Reported "requests" are turns; the report also prints conversation counts.
Accept length comes from `/server_info` (`accept_length` in the result JSON,
null without speculative decoding). Compare latency medians/percentiles
rather than means, and treat TTFT/ITL as tool-specific:

- TTFT is stamped on the first **visible** token. When a reasoning parser
  strips a turn's first stream flush down to markup only, that turn's TTFT
  moves to the next flush and the inter-flush gap drops out of its ITL
  samples. Clients that stamp TTFT on the first SSE event report lower TTFT
  and higher TPOT for the same run — total latency is identical.
- With speculative decoding, reported ITL spreads each stream gap over the
  re-tokenized visible text of the chunk. Under `SGLANG_SIMULATE_ACC_LEN`
  the forced-acceptance output re-tokenizes to far fewer tokens than were
  generated, inflating per-token ITL; trust TPOT instead.
- Safe to compare across benchmark clients: TPOT, end-to-end latency,
  throughput, cache-hit rate, `usage` token counts, and `/server_info`
  accept length.

## Notes

- `ValueError: Feature type 'List' not found` while streaming a source
  dataset means the `datasets` cache under `~/.cache/huggingface/datasets`
  was written by a newer `datasets` version; point `HF_DATASETS_CACHE` at a
  fresh directory and rerun.
- The HF datasets are referenced by name and downloaded at runtime under
  their own licenses; nothing is redistributed with SGLang.
