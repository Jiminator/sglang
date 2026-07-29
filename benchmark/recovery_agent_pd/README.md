# Recovery-Agent PD Serving Recipes (sglang v0.5.16)

Server-side recipes for running the `recovery-agent` benchmark workload
(`python -m sglang.benchmark.serving --dataset-name recovery-agent`) against a
prefill/decode-disaggregated sglang v0.5.16 deployment fronted by the
sgl-router. The reference deployment translated here originally ran on a
Dynamo frontend with `dynamo.sglang` workers; this directory adapts that
configuration to plain `sglang.launch_server` + `sglang_router.launch_router`.

Layout:

- `launch_small_2p2d.sh` — 2 prefill + 2 decode single-GPU workers of a small
  pinned chat model plus a consistent-hashing PD router; validates routing,
  session affinity, and KV transfer with minimal footprint.
- `launch_glm_tp4_unified.sh` — one TP4 GLM-5.2-NVFP4 server (no PD) on one
  4-GPU node; validates the model path under the workload.
- `launch_glm_pd_worker.sh` — one GLM-5.2-NVFP4 TP4 PD worker (prefill or
  decode role, staged feature levels) for the 2 prefill + 2 decode, 16-GPU
  deployment.
- `launch_router.sh` — the PD router with consistent hashing on both pools.

## Flag parity table (reference config → sglang v0.5.16)

Every flag of the reference deployment appears exactly once, either carried
(with its role placement) or as an explicit deviation.

### Prefill workers

| Reference flag | v0.5.16 placement | Status |
|---|---|---|
| `tensor-parallel-size: 4` | `--tp 4` | carried |
| `data-parallel-size: 4` + `enable-dp-attention: true` | — | deviation: the validated v0.5.16 recipe runs plain TP4 without DP-attention; the reference's per-tray DP-attention layout is not enabled by the launch scripts |
| `ep-size: 4` | — | deviation: expert parallelism follows the plain-TP4 layout above (v0.5.16 default `ep-size 1`); not emitted by the launch scripts |
| `quantization: modelopt_fp4` | same | carried |
| `kv-cache-dtype: fp8_e4m3` | same | carried |
| `context-length: 262144` | `--context-length 262144` | carried |
| `mem-fraction-static: 0.93` | same | carried |
| `attention-backend: dsa` | `--attention-backend dsa` | carried (stage 2; stage 1 pins `--attention-backend triton` so the dense baseline is genuinely distinct — v0.5.16 auto-selects DSA for this model) |
| `dsa-prefill-backend: trtllm` (worker CLI) | same flag in v0.5.16 | carried (stage 2) |
| `moe-runner-backend: flashinfer_trtllm_routed` | — | deviation: the PD worker script leaves the MoE runner at the v0.5.16 default, which resolves to `flashinfer_trtllm` (the verified GLM NVFP4 value) on this hardware; the unified shakeout script's `doc-full` stage does exercise the reference value |
| `chunked-prefill-size: 16384` | same | carried |
| `max-prefill-tokens: 32768` | same | carried |
| `max-running-requests: 32` | same | carried |
| `disaggregation-mode: prefill` | same | carried |
| `disaggregation-transfer-backend: nixl` | same, **pinned explicitly** | carried (v0.5.16 default is `mooncake`) |
| `speculative-algorithm: EAGLE`, `speculative-num-steps: 3`, `speculative-eagle-topk: 1`, `speculative-num-draft-tokens: 4` | same | carried (stage 3) |
| `kv-events-config: {publisher: zmq, ...}` | — | deviation: publishes KV events for the Dynamo KV router; the sgl-router consistent-hashing policy does not consume them |
| `disable-overlap-schedule: true` | same | carried |
| `enable-hierarchical-cache: true` | same | carried (stage 4, prefill only) |
| `hicache-size: 160` | `--hicache-size 32` | deviation (stage 4): 160 GB is per DP rank; on a shared validation host start at 32 GB/rank |
| `hicache-io-backend: direct` | same | carried (stage 4) |
| `hicache-mem-layout: page_first_direct` | same | carried (stage 4) |
| `hicache-write-policy: write_back` | same | carried (stage 4; `write_through` stalls prefill during cache building) |
| `cuda-graph-config: '{"prefill": {"backend": "disabled"}, "decode": {"backend": "disabled"}}'` | same | carried |
| — | `--reasoning-parser glm45` | addition: required so `reasoning_content` is separated on the wire and the benchmark's content-only replay matches live agent traffic |

### Decode workers

| Reference flag | v0.5.16 placement | Status |
|---|---|---|
| `tensor-parallel-size: 4` | `--tp 4` | carried |
| `data-parallel-size: 4` + `enable-dp-attention: true` | — | deviation: plain TP4 on the decode workers as well; DP-attention is not enabled by the launch scripts |
| `moe-a2a-backend: none` | same | carried |
| `fp4-gemm-backend: flashinfer_cutlass` | same | carried |
| `quantization: modelopt_fp4`, `kv-cache-dtype: fp8_e4m3` | same | carried |
| `disaggregation-mode: decode`, `disaggregation-transfer-backend: nixl` | same | carried |
| `speculative-algorithm: EAGLE`, `speculative-num-steps: 3`, `speculative-eagle-topk: 1`, `speculative-num-draft-tokens: 4`, `speculative-attention-mode: decode` | same | carried (stage 3; 5-1-6 measured slower at load in the reference) |
| `max-running-requests: 128` | same | carried (256 measured slower in the reference) |
| `mem-fraction-static: 0.87` | same | carried |
| `num-continuous-decode-steps: 3` | same | carried |
| `enable-flashinfer-allreduce-fusion: true` | `--flashinfer-allreduce-fusion-backend auto` | deviation: the boolean flag is a deprecated alias in v0.5.16 |
| `enable-symm-mem: true` | — | deviation: NCCL symmetric-memory registration proved unstable in this pod environment during EAGLE cuda-graph init (one wedge, one `NCCL symmetric memory registration failed` crash); omitted from the validated recipe |
| `cuda-graph-config: '{"decode": {"backend": "full", "max_bs": 128}}'` | same | carried |
| `attention-backend: dsa` | same | carried (stage 2; stage 1 pins `--attention-backend triton`; the decode DSA backend is left at the v0.5.16 default and the resolved backend is recorded from server logs) |
| — | `--reasoning-parser glm45` | addition: same wire-format requirement as the prefill workers — the decode workers produce the streamed output whose `reasoning_content` must be separated |
| — | decode radix cache / HiCache restore | intentionally absent: v0.5.16 rejects `--disaggregation-decode-enable-radix-cache` together with speculative decoding |

### Frontend / router

| Reference flag | v0.5.16 placement | Status |
|---|---|---|
| Dynamo frontend `--kv-cache-block-size 64` | — | deviation: Dynamo KV-router block size; no sgl-router equivalent, not translated into engine page-size flags |
| `--router-session-affinity-ttl-secs 3600` | — | deviation: Dynamo session-affinity TTL; sgl-router consistent hashing keys on `X-SMG-Routing-Key` and needs no TTL |
| `--router-temperature 0` | — | deviation: Dynamo KV-router sampling knob; not applicable |
| session affinity via `x-session-id` | `--prefill-policy consistent_hashing --decode-policy consistent_hashing` + per-session `X-SMG-Routing-Key` (sent by the benchmark client) | carried, mechanism translated |
| `--trust-remote-code` | worker-side `--trust-remote-code` | carried |

### Worker environment

| Reference setting | This deployment | Status |
|---|---|---|
| `SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX` | same | carried |
| `UCX_TLS=cuda_ipc,cuda_copy,rc` | same | carried |
| `UCX_NET_DEVICES=mlx5_0:1,...` | discovered per host (`ibv_devices`) | deviation: HCA names are fleet-specific; forcing absent devices prevents startup |
| 100 Gi `/dev/shm` | pod-provided (verify ≥ 64 Gi) | deviation: sized by the environment; too small ⇒ NCCL "unhandled system error" |

Notes:

- PD+DSA transfer is supported in v0.5.16; the NIXL restriction applies only
  to the DSA cache-layer-split feature, which these recipes do not use.
- Outside the reference parity scope, the launch scripts also pass harness
  and observability flags: `--model-path`, `--revision`, `--host`, `--port`,
  `--disaggregation-bootstrap-port`, `--enable-metrics`,
  `--enable-cache-report`, and the router topology flags
  (`--pd-disaggregation`, `--policy`, `--prefill`, `--decode`), plus the
  small-model script's reduced `--mem-fraction-static`, the unified script's
  `--bf16-gemm-backend`/`--cuda-graph-max-bs`/`--max-prefill-tokens`/
  `--chunked-prefill-size`/`--max-running-requests` shakeout settings, and
  the staged speculative flags (`--speculative-algorithm`,
  `--speculative-num-steps`, `--speculative-eagle-topk`,
  `--speculative-num-draft-tokens`, `--speculative-attention-mode`) whose
  reference values appear in the tables above.
- Stages for the PD deployment (fault attribution): stage 1 = dense baseline
  (`--attention-backend triton` pinned explicitly, no speculation), stage 2 =
  +DSA, stage 3 = +EAGLE 3-1-4, stage 4 = +HiCache on prefill.
