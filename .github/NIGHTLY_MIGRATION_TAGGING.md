# Tag-gated nightly migration — tagging report

This document is the design report for moving 59 long-running tests off the per-commit PR pipeline (`.github/workflows/pr-test.yml`) and into the nightly pipeline (`.github/workflows/nightly-test-nvidia.yml`), while keeping each test reachable from a PR run that has a relevant label.

It does **not** modify any code or workflow yet — it captures the tagging decisions so the implementation can follow.

---

## Mechanism

A nightly test carries a `tags` tuple. On a PR commit:

1. The labeler (`.github/labeler.yml`) auto-applies labels to the PR based on the file globs the PR touches; users can also add labels manually.
2. The PR pipeline reads `github.event.pull_request.labels.*.name` and forwards them as a comma-separated list (e.g. `--include-tags=lora,moe`) to the suite runner.
3. The suite runner additionally pulls in any nightly test whose `tags ∩ pr_labels ≠ ∅`, restricted to nightly suites that share the **same hardware class** as the per-commit suite being run (so a 4-GPU-B200 nightly test never lands on a 1-GPU-H100 runner).
4. Nightly tests with `tags=()` only ever run at nightly.

A test can carry multiple tags — any matching label pulls it in.

---

## Tag inventory

### Reuse from `.github/labeler.yml` (already auto-applied by file globs)

| Tag | Existing labeler.yml glob (no change required) |
|---|---|
| `Multi-modal` | `**/*multimodal*`, `**/*vision*`, `**/*vlm*` |
| `lora` | `**/*lora*` |
| `quant` | `**/*quant*`, `**/*quantization*` |
| `speculative-decoding` | `**/*speculative*` |
| `deepseek` | `**/*deepseek*` |
| `hicache` | `**/*hicache*` |
| `blackwell` | `**/*nvfp4*`, `attention/cutlass_sm100_mla/*`, `trtllm_m{l,h}a_backend.py` |
| `piecewise-cuda-graph` | `python/sglang/srt/compilation/**/*` |

### New tags to add

| Tag | Why a new tag is needed | Suggested labeler.yml glob |
|---|---|---|
| `perf` | Bench / one-batch / serving throughput tests don't share a single source path; many of them are also tagged with feature labels (e.g. `quant`, `Multi-modal`) so they ride along on those, but `perf` lets a perf-only PR opt in. | (no auto-glob — apply manually or via slash command); optionally `python/sglang/bench_*` |
| `attention-backend` | FA3-hybrid, torch-native, Triton sliding-window, and FP8-KV-on-Triton tests all live under `python/sglang/srt/layers/attention/`. No existing tag covers it. | `python/sglang/srt/layers/attention/**/*` |
| `moe` | MoE / EP / DeepEP / CuteDSL-MoE / routed-experts. `quant` doesn't capture MoE-only changes. | `python/sglang/srt/layers/moe/**/*` |
| `rl` | Online weight-sync (`update_weights_from_*`, `load_weights_from_remote_instance`) and `release_memory_occupation` form a tight group of training-loop integration tests. | `python/sglang/srt/weight_sync/**/*`, `python/sglang/srt/entrypoints/engine.py` |
| `scoring` | `/v1/score`, `/v1/embeddings`, pooled hidden states, multi-item scoring — distinct surface from generation. | `python/sglang/srt/scoring/**/*`, `python/sglang/srt/entrypoints/openai/serving_embedding*`, `python/sglang/srt/entrypoints/openai/serving_score*` |
| `session` | Streaming session / control / session-latency. | `python/sglang/srt/managers/session_*`, `python/sglang/srt/server/streaming_*` |

**Decisions deliberately not made:**
- No `sm120` / `5090` tag. Only two tests are SM12.0-specific (`test_fp8_gemm_sm120.py`, `test_gpt_oss_sm120.py`); both are quant kernels and `quant` already covers anyone touching the relevant code paths.
- No "model-coverage" tag. Three tests (`test_generation_models.py`, `test_priority_scheduling.py`, `test_tracing.py`) don't fit a clear feature group and would only dilute existing tags. They stay untagged → nightly-only.

---

## Test → tag assignments

The `Reasoning` column captures *why* each tag was chosen. Subset markers (e.g. *"only TestEpDeepGEMM"*) reproduce your spec — only the named subset is moved.

### 1-GPU large (H100-class)

| Test | Tags | Reasoning |
|---|---|---|
| `test_bench_serving_1gpu_part1.py` | `perf` | Pure throughput benchmark; no feature-specific surface. |
| `test_bench_serving_1gpu_part2.py` | `perf`, `Multi-modal`, `scoring` | Bench file but exercises VLM and scoring code paths inside the same run, so a `Multi-modal` or `scoring` PR should re-bench it. |
| `test_bench_serving_1gpu_large.py` | `perf`, `quant`, `speculative-decoding` | Bench against a quantized model with EAGLE; perf change in any of those three pulls it in. |
| `test_streaming_session.py` *(excl. `TestStreamingSessionEagleV2RetractLargePage`, `TestStreamingSessionAbortLeakRepro`)* | `session` | Retract / EAGLE-variant session classes test session lifecycle on heavier configs. |
| `test_session_latency.py` | `session`, `perf` | Measures inter-turn latency — both feature (`session`) and perf-sensitive. |
| `test_session_control.py` | `session` | Branching / abort / backtrack on streaming-session API. |
| `test_hybrid_attn_backend.py` | `attention-backend` | FA3 prefill + FlashInfer decode — a backend-mix test. |
| `test_torch_native_attention_backend.py` | `attention-backend` | Pure backend coverage. |
| `test_triton_sliding_window.py` | `attention-backend` | Triton SWA backend behavior. |
| `test_torch_compile.py` | `piecewise-cuda-graph` | `torch.compile` interacts directly with the piecewise-CUDA-graph path. |
| `test_eagle_infer_a.py` | `speculative-decoding` | EAGLE inference. |
| `test_standalone_speculative_decoding.py` *(non-V2 classes only)* | `speculative-decoding` | Legacy non-V2 path; V2 stays in PR per-commit. |
| `test_ngram_speculative_decoding.py` *(Triton + Flashinfer classes only)* | `speculative-decoding` | Backend-specific NGRAM coverage; FA3 path stays in PR. |
| `test_w8a8_quantization.py` | `quant` | W8A8 INT8. |
| `test_gptqmodel_dynamic.py` | `quant` | GPTQ. |
| `test_fp8kv_triton.py` | `quant`, `attention-backend` | FP8 KV-cache *and* Triton attention — both surfaces matter. |
| `test_compressed_tensors_models.py` | `quant` | CompressedTensors FP8. |
| `test_vlm_models.py` | `Multi-modal` | VLM accuracy via MMMU. |
| `test_generation_models.py` | *(untagged)* | Generic text-generation model coverage; doesn't tie to any feature group. Nightly-only. |
| `test_lora_load_from_tensor.py` | `lora`, `rl` | Online tensor-based LoRA load — used in RL pipelines. |
| `test_lora_qwen3_5_4b_logprob_diff.py` | `lora` | LoRA logprob accuracy. |
| `test_lora_qwen3_8b_logprob_diff.py` | `lora` | LoRA logprob accuracy. |

### 1-GPU small (5090 / SM12.0)

| Test | Tags | Reasoning |
|---|---|---|
| `test_gpt_oss_sm120.py` | `quant` | MxFP4 on 5090; gated by `quant` since SM12.0 has only two relevant tests. |
| `test_fp8_gemm_sm120.py` | `quant` | FP8 GEMM kernel for SM12.0. |
| `test_lora_eviction.py` | `lora` | Adapter-eviction policy. |
| `test_lora_drainer.py` | `lora` | Adapter-drainer heuristic (unit-style). |
| `test_multi_item_scoring.py` | `scoring` | MIS delimiter optimization on `/v1/score`. |
| `test_pooled_hidden_states.py` | `scoring` | Pooled-representation API. |
| `test_score_engine.py` | `scoring` | Engine-level `/v1/score`. |
| `test_score_api.py` | `scoring` | HTTP `/v1/score`. |
| `test_openai_embedding.py` | `scoring` | OpenAI-compatible `/v1/embeddings`. |
| `test_update_weights_from_tensor.py` | `rl` | Online weight sync from a host tensor. |
| `test_priority_scheduling.py` | *(untagged)* | Single-purpose scheduler edge case; no clear feature group. Nightly-only. |
| `test_tracing.py` | *(untagged)* | OTLP/observability is its own surface; one test alone doesn't justify a tag. Nightly-only. |

### 2-GPU

| Test | Tags | Reasoning |
|---|---|---|
| `test_bench_serving_2gpu.py` | `perf`, `moe` | Multi-GPU MoE throughput. |
| `test_bench_one_batch_2gpu.py` | `perf`, `moe`, `piecewise-cuda-graph` | MoE + `torch.compile` perf. |
| `test_pcg_with_speculative_decoding.py` *(STANDALONE + NGRAM + MTP only; EAGLE3 stays)* | `piecewise-cuda-graph`, `speculative-decoding` | PCG × spec-decode interaction; EAGLE3 already in PR. |
| `test_moe_ep.py` *(only `TestEpDeepGEMM`)* | `moe`, `quant` | DeepGEMM FP8 path on top of EP — touches both tags. The plain `TestEp` class stays in PR. |
| `test_ministral4_models.py` | `Multi-modal` | Mistral-Small-4 with MMMU eval. |
| `test_lora_moe_tp_logprob_diff.py` | `lora`, `moe` | MoE + LoRA — pulled in by either label. |
| `test_load_weights_from_remote_instance.py` | `rl` | Remote-instance weight sync. |
| `test_update_weights_from_distributed.py` | `rl` | Distributed weight sync. |

### 4-GPU B200 (Blackwell)

| Test | Tags | Reasoning |
|---|---|---|
| `test_deepseek_v3_fp4_4gpu.py` *(only `TestDeepseekV3FP4CutlassMoE`)* | `deepseek`, `moe`, `quant`, `blackwell` | The Cutlass-MoE class spans all four; the other two FP4 classes stay in PR. |
| `test_nvidia_nemotron_3_super_nvfp4.py` | `quant`, `blackwell` | NvFP4 quantization on Blackwell. |
| `test_fp8_blockwise_gemm.py` | `quant`, `blackwell` | FP8 blockwise GEMM, B200-only kernel. |
| `test_cutedsl_moe.py` | `moe`, `quant`, `blackwell` | CuteDSL MoE + NvFP4. |
| `test_update_weights_from_disk_blackwell.py` | `rl`, `blackwell` | Disk-based RL weight sync, Blackwell-specific. |
| `test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` | `lora`, `moe`, `blackwell` | LoRA on a MoE model on Blackwell — three orthogonal triggers. |
| `test_lora_gpt_oss_20b_logprob_diff.py` | `lora`, `moe`, `blackwell` | Same pattern. |
| `test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py` | `lora`, `moe`, `blackwell` | Same pattern. |

### 4-GPU H100

| Test | Tags | Reasoning |
|---|---|---|
| `test_return_routed_experts.py` | `moe`, `deepseek` | DeepEP state capture for DeepSeek MoE. |
| `test_eagle_dp_attention.py` | `speculative-decoding` | EAGLE3 + DP-attention. |
| `test_multi_instance_release_memory_occupation.py` | `rl` | Multi-instance memory-occupation tracking — RL-control-plane test. |

### 8-GPU H200

| Test | Tags | Reasoning |
|---|---|---|
| `test_step3p5_flash_chain_mtp.py` | `speculative-decoding`, `deepseek`, `moe` | Multi-layer EAGLE/MTP on a DeepSeek MoE model. |
| `test_return_indexer_topk.py` | `deepseek` | DSv3.2 indexer-topk capture. |
| `test_deepseek_v32_indexcache.py` | `deepseek`, `hicache` | Index-cache pattern (DSv3.2 + hicache). |
| `test_nvidia_nemotron_3_super_bf16.py` | `speculative-decoding` | TP8 + EAGLE. |
| `test_deepseek_v32_cp_single_node.py` | `deepseek` | Context-parallel for DSv3.2. |
| `test_deepep_large.py` | `moe`, `deepseek` | DeepEP large-scale on DeepSeek. |

---

## Tag-frequency summary

| Tag | Tests | Source |
|---|---:|---|
| `quant` | 13 | reused |
| `moe` | 12 | new |
| `lora` | 11 | reused |
| `blackwell` | 9 | reused |
| `speculative-decoding` | 9 | reused |
| `deepseek` | 8 | reused |
| `rl` | 7 | new |
| `perf` | 6 | new |
| `scoring` | 6 | new |
| `attention-backend` | 4 | new |
| `Multi-modal` | 4 | reused |
| `piecewise-cuda-graph` | 3 | reused |
| `session` | 3 | new |
| `hicache` | 1 | reused |
| *(untagged — nightly-only)* | 3 | — |

Total: 59 tests / 14 tags + untagged.

---

## Edge cases worth noting before implementation

- **Hardware compatibility.** The runner needs a per-commit-suite → compatible-nightly-suite map so that, e.g., a `lora` label on a PR running `stage-b-test-1-gpu-large` only pulls `nightly-1-gpu` LoRA tests, not the four-GPU B200 LoRA tests. Without that, a label-pull will land a test on a runner that can't host it.
- **5090 / SM12.0 nightly slot.** The two `*_sm120` tests need a runner with SM12.0; today's `nightly-test-nvidia.yml` only has `1-gpu-h100`, where they would silently skip. They need a new `nightly-1-gpu-5090` job (or to remain on the per-commit `1-gpu-5090` runner via a different mechanism).
- **2-GPU nightly slot.** `nightly-2-gpu` is a registered suite name but no nightly job runs it today. A new `nightly-test-general-2-gpu-h100` job is required.
- **Subset moves.** Six files (`test_streaming_session.py`, `test_standalone_speculative_decoding.py`, `test_ngram_speculative_decoding.py`, `test_pcg_with_speculative_decoding.py`, `test_moe_ep.py`, `test_deepseek_v3_fp4_4gpu.py`) require a class-level split because only some classes move to nightly. The existing pattern in `test_streaming_session_swa.py` (sibling `sys.path.insert` import) is the precedent to follow.
- **AMD registrations.** Each test's `register_amd_ci(...)` is independent of `register_cuda_ci(...)` and should remain untouched — the AMD per-commit pipeline keeps these tests where they are.
