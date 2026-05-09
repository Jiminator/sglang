# Tag-gated nightly migration — tagging report

This document is the design report for moving 62 long-running tests off the per-commit PR pipeline (`.github/workflows/pr-test.yml`) and into the nightly pipeline (`.github/workflows/nightly-test-nvidia.yml`), while keeping each test reachable from a PR run that has a relevant label.

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

### Existing tags to reuse or tighten

These tags already exist as label names, so the migrated tests can reuse them.
Some existing glob patterns still need tightening so directory-level changes
actually apply the intended label. In particular, a single `*` does not cross
path separators; directory surfaces should use `/**/*`.

| Tag | Existing / recommended labeler.yml globs |
|---|---|
| `Multi-modal` | Existing `**/*multimodal*`, `**/*vision*`, `**/*vlm*` are sufficient for the migrated VLM tests. |
| `lora` | Keep `**/*lora*`; also add `python/sglang/srt/lora/**/*` so files under the LoRA package whose basename does not contain `lora` still auto-label. |
| `quant` | Keep `**/*quant*`, `**/*quantization*`; also add `python/sglang/srt/layers/quantization/**/*`, `python/sglang/srt/hardware_backend/*/quantization/**/*`, and FP8/FP4 kernel globs such as `**/*fp8*`, `**/*nvfp4*`, `**/*mxfp4*`. |
| `speculative-decoding` | Keep `**/*speculative*`; also add `python/sglang/srt/speculative/**/*`, `**/*eagle*`, `**/*mtp*`, and `**/*ngram*` so EAGLE/MTP/NGRAM-specific files auto-label. |
| `deepseek` | Keep `**/*deepseek*`; also add `python/sglang/srt/models/deepseek_common/**/*` because nested files under that directory are DeepSeek-specific even when their basenames do not contain `deepseek`. |
| `hicache` | Keep `**/*hicache*`; also add `python/sglang/srt/mem_cache/hiradix_cache.py`, `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py`, `python/sglang/srt/mem_cache/storage/**/*`, `python/sglang/srt/managers/cache_controller.py`, `python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`, `python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py`, and `python/sglang/srt/mem_cache/unified_cache_components/swa_component.py` so the core HiCache / HiCache-Mamba / storage-backend paths auto-label even when the basename does not contain `hicache`. |
| `blackwell` | Existing `**/*nvfp4*`, `sgl-kernel/csrc/attention/cutlass_sm100_mla/**/*`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mha_backend.py`. |
| `piecewise-cuda-graph` | Existing `python/sglang/srt/compilation/**/*` is sufficient for the migrated PCG tests. |

### New tags to add

| Tag | Why a new tag is needed | Suggested labeler.yml glob |
|---|---|---|
| `perf` | Bench / one-batch / serving throughput tests; auto-fire on changes to bench scripts and the executor / forward-info hot path so perf-sensitive PRs always re-run them. | `python/sglang/bench_*.py`, `python/sglang/srt/model_executor/**/*` |
| `attention-backend` | FA3-hybrid, torch-native, Triton sliding-window, FP8-KV-on-Triton, and hybrid/Mamba attention tests all live on the attention backend surface. No existing tag covers it. | `python/sglang/srt/layers/attention/**/*`, `python/sglang/srt/models/qwen3_next*.py`, `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/configs/qwen3_5.py` |
| `moe` | MoE / EP / DeepEP / CuteDSL-MoE / routed-experts. `quant` doesn't capture MoE-only changes. | `python/sglang/srt/layers/moe/**/*`, `python/sglang/srt/eplb/**/*`, `python/sglang/srt/models/**/*moe*.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/nemotron_h*.py`, `python/sglang/srt/models/step3p5*.py`, `python/sglang/srt/models/qwen3_next*.py` |
| `disaggregation` | PD disaggregation, KV-transfer / KV-event plumbing, and disaggregated prefill/decode tests are distinct from normal serving and cache-only coverage. | `python/sglang/srt/disaggregation/**/*`, `test/server_fixtures/disaggregation_fixture.py`, `test/**/*disaggregation*.py` |
| `rl` | Online weight-sync (`update_weights_from_*`, `load_weights_from_remote_instance`) and `release_memory_occupation` form a tight group of training-loop integration tests. | `python/sglang/srt/weight_sync/**/*`, `python/sglang/srt/managers/scheduler_update_weights_mixin.py`, `python/sglang/srt/managers/tokenizer_control_mixin.py` |
| `scoring` | `/v1/score`, `/v1/embeddings`, pooled hidden states, multi-item scoring — distinct surface from generation. | `python/sglang/srt/managers/tokenizer_manager_score_mixin.py`, `python/sglang/srt/entrypoints/engine_score_mixin.py`, `python/sglang/srt/entrypoints/openai/serving_embedding*`, `python/sglang/srt/entrypoints/openai/serving_score*` |
| `session` | Streaming session / control / session-latency. | `python/sglang/srt/session/**/*`, `python/sglang/srt/entrypoints/http_server*`, `python/sglang/srt/entrypoints/openai/serving_chat*` |
| `scheduler` | Closes the previously-untagged `test_priority_scheduling.py` slot. The scheduler / batch-scheduling code is its own tight surface. | `python/sglang/srt/managers/scheduler*`, `python/sglang/srt/managers/schedule_batch*`, `python/sglang/srt/managers/schedule_policy.py` |
| `model-coverage` | Closes the previously-untagged `test_generation_models.py` slot. Any change under `python/sglang/srt/models/**` should re-run the model-coverage tests. | `python/sglang/srt/models/**/*` |

**Decisions deliberately not made:**
- No `sm120` / `5090` tag. Only two tests are SM12.0-specific (`test_fp8_gemm_sm120.py`, `test_gpt_oss_sm120.py`); both are quant kernels and `quant` already covers anyone touching the relevant code paths.
- The `rl` glob deliberately *does not* include broad shared files such as `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/entrypoints/http_server.py`, or `python/sglang/srt/model_executor/model_runner.py`. Those files are touched by many unrelated server-side changes. If a PR edits the RL/update-weight methods inside one of those broad files, the PR should receive the `rl` label manually.
- Similarly, scoring/session/disaggregation/HiCache-specific edits in broad files such as `python/sglang/srt/managers/io_struct.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/mem_cache/unified_radix_cache.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/managers/scheduler.py`, or `python/sglang/srt/managers/tp_worker.py` may need a manual corresponding label unless the implementation chooses to accept broader auto-labeling.
- `test_tracing.py` (OTLP/observability) stays untagged. One test alone doesn't justify a dedicated tag, and no existing label fits the surface area.

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
| `test_hicache_spec_file_storage.py` | `hicache`, `speculative-decoding` | HiCache file-storage loadback with EAGLE3 speculative decoding; either cache storage or EAGLE changes should pull it into PR CI. |
| `test_standalone_speculative_decoding.py` *(non-V2 classes only)* | `speculative-decoding` | Legacy non-V2 path; V2 stays in PR per-commit. |
| `test_ngram_speculative_decoding.py` *(Triton + Flashinfer classes only)* | `speculative-decoding` | Backend-specific NGRAM coverage; FA3 path stays in PR. |
| `test_w8a8_quantization.py` | `quant` | W8A8 INT8. |
| `test_gptqmodel_dynamic.py` | `quant` | GPTQ. |
| `test_fp8kv_triton.py` | `quant`, `attention-backend` | FP8 KV-cache *and* Triton attention — both surfaces matter. |
| `test_compressed_tensors_models.py` | `quant` | CompressedTensors FP8. |
| `test_vlm_models.py` | `Multi-modal` | VLM accuracy via MMMU. |
| `test_generation_models.py` | `model-coverage` | Generic text-generation model coverage; rides on the new `model-coverage` glob. |
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
| `test_priority_scheduling.py` | `scheduler` | Priority-queue scheduling edge case; closed by the new `scheduler` glob. |
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
| `test_nvidia_nemotron_3_super_nvfp4.py` | `quant`, `blackwell`, `moe`, `speculative-decoding` | NvFP4 quantization on Blackwell for a Nemotron MoE model; the file also contains an EAGLE/MTP variant. |
| `test_fp8_blockwise_gemm.py` | `quant`, `blackwell` | FP8 blockwise GEMM, B200-only kernel. |
| `test_cutedsl_moe.py` | `moe`, `quant`, `blackwell` | CuteDSL MoE + NvFP4. |
| `test_update_weights_from_disk_blackwell.py` | `rl`, `blackwell` | Disk-based RL weight sync, Blackwell-specific. |
| `test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` | `lora`, `moe`, `blackwell` | LoRA on a MoE model on Blackwell — three orthogonal triggers. |
| `test_lora_gpt_oss_20b_logprob_diff.py` | `lora`, `moe`, `blackwell` | Same pattern. |
| `test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py` | `lora`, `moe`, `blackwell` | Same pattern. |

### 4-GPU H100

| Test | Tags | Reasoning |
|---|---|---|
| `test_qwen35_hicache.py` | `hicache`, `attention-backend`, `disaggregation` | Qwen3.5 HiCache accuracy and KV-event smoke test; it exercises HiCache-Mamba / hybrid-attention cache behavior plus the shared disaggregation KV-event types. |
| `test_return_routed_experts.py` | `moe` | DeepEP routed-expert state capture on Qwen3 MoE; this is not DeepSeek-specific. |
| `test_eagle_dp_attention.py` | `speculative-decoding` | EAGLE3 + DP-attention. |
| `test_multi_instance_release_memory_occupation.py` | `rl` | Multi-instance memory-occupation tracking — RL-control-plane test. |

### 8-GPU H200

| Test | Tags | Reasoning |
|---|---|---|
| `test_step3p5_flash_chain_mtp.py` | `speculative-decoding`, `moe` | Multi-layer EAGLE/MTP on the Step-3.5 MoE model; this is not DeepSeek-specific. |
| `test_disaggregation_hybrid_attention.py` | `disaggregation`, `attention-backend`, `moe` | PD disaggregation on Qwen3-Next with hybrid/Mamba attention and a sparse-MoE model, including DP-attention decode coverage. |
| `test_return_indexer_topk.py` | `deepseek` | DSv3.2 indexer-topk capture. |
| `test_deepseek_v32_indexcache.py` | `deepseek`, `hicache` | Index-cache pattern (DSv3.2 + hicache). |
| `test_nvidia_nemotron_3_super_bf16.py` | `speculative-decoding`, `moe` | TP8 Nemotron MoE with EAGLE/MTP. |
| `test_deepseek_v32_cp_single_node.py` | `deepseek` | Context-parallel for DSv3.2. |
| `test_deepep_large.py` | `moe`, `deepseek` | DeepEP large-scale on DeepSeek. |

---

## Related PR changes not tag-gated

| Test | Disposition | Reasoning |
|---|---|---|
| `test_vision_openai_server_a.py` | Keep in `stage-b-test-1-gpu-large`; no nightly tag assignment. | The PR trims this per-commit OpenAI vision/audio/omni/OCR smoke file from `780s` to `420s` by removing model families that already have nightly VLM accuracy coverage (`test_vlms_mmmu_eval.py` / `nightly-eval-vlm-2-gpu`, plus Gemma in `test_vlm_models.py`). It does not move this file, or the removed classes, into the tag-gated nightly migration. The remaining per-commit file still covers OpenAI-server behavior for the model families not duplicated by those nightly accuracy tests plus Qwen2-VL context-length validation. |

---

## Tests grouped by tag

Each row lists the test plus its other tags (so you can see what else gets pulled along). When only a subset of classes in a file moves, the qualifier is reproduced here.

### Reused tags

#### `Multi-modal` (3)
- `test_bench_serving_1gpu_part2.py` — also `perf`, `scoring`
- `test_vlm_models.py`
- `test_ministral4_models.py`

#### `blackwell` (8)
- `test_deepseek_v3_fp4_4gpu.py` *(only `TestDeepseekV3FP4CutlassMoE`)* — also `deepseek`, `moe`, `quant`
- `test_nvidia_nemotron_3_super_nvfp4.py` — also `quant`, `moe`, `speculative-decoding`
- `test_fp8_blockwise_gemm.py` — also `quant`
- `test_cutedsl_moe.py` — also `moe`, `quant`
- `test_update_weights_from_disk_blackwell.py` — also `rl`
- `test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` — also `lora`, `moe`
- `test_lora_gpt_oss_20b_logprob_diff.py` — also `lora`, `moe`
- `test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py` — also `lora`, `moe`

#### `deepseek` (5)
- `test_deepseek_v3_fp4_4gpu.py` *(only `TestDeepseekV3FP4CutlassMoE`)* — also `moe`, `quant`, `blackwell`
- `test_return_indexer_topk.py`
- `test_deepseek_v32_indexcache.py` — also `hicache`
- `test_deepseek_v32_cp_single_node.py`
- `test_deepep_large.py` — also `moe`

#### `hicache` (3)
- `test_hicache_spec_file_storage.py` — also `speculative-decoding`
- `test_qwen35_hicache.py` — also `attention-backend`, `disaggregation`
- `test_deepseek_v32_indexcache.py` — also `deepseek`

#### `lora` (9)
- `test_lora_load_from_tensor.py` — also `rl`
- `test_lora_qwen3_5_4b_logprob_diff.py`
- `test_lora_qwen3_8b_logprob_diff.py`
- `test_lora_eviction.py`
- `test_lora_drainer.py`
- `test_lora_moe_tp_logprob_diff.py` — also `moe`
- `test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` — also `moe`, `blackwell`
- `test_lora_gpt_oss_20b_logprob_diff.py` — also `moe`, `blackwell`
- `test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py` — also `moe`, `blackwell`

#### `piecewise-cuda-graph` (3)
- `test_torch_compile.py`
- `test_bench_one_batch_2gpu.py` — also `perf`, `moe`
- `test_pcg_with_speculative_decoding.py` *(STANDALONE + NGRAM + MTP only; EAGLE3 stays in PR)* — also `speculative-decoding`

#### `quant` (12)
- `test_bench_serving_1gpu_large.py` — also `perf`, `speculative-decoding`
- `test_w8a8_quantization.py`
- `test_gptqmodel_dynamic.py`
- `test_fp8kv_triton.py` — also `attention-backend`
- `test_compressed_tensors_models.py`
- `test_gpt_oss_sm120.py`
- `test_fp8_gemm_sm120.py`
- `test_moe_ep.py` *(only `TestEpDeepGEMM`)* — also `moe`
- `test_deepseek_v3_fp4_4gpu.py` *(only `TestDeepseekV3FP4CutlassMoE`)* — also `deepseek`, `moe`, `blackwell`
- `test_nvidia_nemotron_3_super_nvfp4.py` — also `blackwell`, `moe`, `speculative-decoding`
- `test_fp8_blockwise_gemm.py` — also `blackwell`
- `test_cutedsl_moe.py` — also `moe`, `blackwell`

#### `speculative-decoding` (10)
- `test_bench_serving_1gpu_large.py` — also `perf`, `quant`
- `test_eagle_infer_a.py`
- `test_hicache_spec_file_storage.py` — also `hicache`
- `test_standalone_speculative_decoding.py` *(non-V2 classes only; V2 stays in PR)*
- `test_ngram_speculative_decoding.py` *(Triton + Flashinfer classes only; FA3 path stays in PR)*
- `test_pcg_with_speculative_decoding.py` *(STANDALONE + NGRAM + MTP only)* — also `piecewise-cuda-graph`
- `test_eagle_dp_attention.py`
- `test_step3p5_flash_chain_mtp.py` — also `moe`
- `test_nvidia_nemotron_3_super_nvfp4.py` — also `quant`, `blackwell`, `moe`
- `test_nvidia_nemotron_3_super_bf16.py` — also `moe`

### New tags

#### `attention-backend` (6)
- `test_hybrid_attn_backend.py`
- `test_torch_native_attention_backend.py`
- `test_triton_sliding_window.py`
- `test_fp8kv_triton.py` — also `quant`
- `test_qwen35_hicache.py` — also `hicache`, `disaggregation`
- `test_disaggregation_hybrid_attention.py` — also `disaggregation`, `moe`

#### `disaggregation` (2)
- `test_qwen35_hicache.py` — also `hicache`, `attention-backend`
- `test_disaggregation_hybrid_attention.py` — also `attention-backend`, `moe`

#### `model-coverage` (1)
- `test_generation_models.py`

#### `moe` (15)
- `test_bench_serving_2gpu.py` — also `perf`
- `test_bench_one_batch_2gpu.py` — also `perf`, `piecewise-cuda-graph`
- `test_moe_ep.py` *(only `TestEpDeepGEMM`)* — also `quant`
- `test_lora_moe_tp_logprob_diff.py` — also `lora`
- `test_deepseek_v3_fp4_4gpu.py` *(only `TestDeepseekV3FP4CutlassMoE`)* — also `deepseek`, `quant`, `blackwell`
- `test_nvidia_nemotron_3_super_nvfp4.py` — also `quant`, `blackwell`, `speculative-decoding`
- `test_cutedsl_moe.py` — also `quant`, `blackwell`
- `test_lora_nemotron_3_super_120b_a12b_logprob_diff.py` — also `lora`, `blackwell`
- `test_lora_gpt_oss_20b_logprob_diff.py` — also `lora`, `blackwell`
- `test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py` — also `lora`, `blackwell`
- `test_return_routed_experts.py`
- `test_step3p5_flash_chain_mtp.py` — also `speculative-decoding`
- `test_nvidia_nemotron_3_super_bf16.py` — also `speculative-decoding`
- `test_disaggregation_hybrid_attention.py` — also `disaggregation`, `attention-backend`
- `test_deepep_large.py` — also `deepseek`

#### `perf` (6)
- `test_bench_serving_1gpu_part1.py`
- `test_bench_serving_1gpu_part2.py` — also `Multi-modal`, `scoring`
- `test_bench_serving_1gpu_large.py` — also `quant`, `speculative-decoding`
- `test_session_latency.py` — also `session`
- `test_bench_serving_2gpu.py` — also `moe`
- `test_bench_one_batch_2gpu.py` — also `moe`, `piecewise-cuda-graph`

#### `rl` (6)
- `test_lora_load_from_tensor.py` — also `lora`
- `test_update_weights_from_tensor.py`
- `test_load_weights_from_remote_instance.py`
- `test_update_weights_from_distributed.py`
- `test_update_weights_from_disk_blackwell.py` — also `blackwell`
- `test_multi_instance_release_memory_occupation.py`

#### `scheduler` (1)
- `test_priority_scheduling.py`

#### `scoring` (6)
- `test_bench_serving_1gpu_part2.py` — also `perf`, `Multi-modal`
- `test_multi_item_scoring.py`
- `test_pooled_hidden_states.py`
- `test_score_engine.py`
- `test_score_api.py`
- `test_openai_embedding.py`

#### `session` (3)
- `test_streaming_session.py` *(excl. `TestStreamingSessionEagleV2RetractLargePage`, `TestStreamingSessionAbortLeakRepro`)*
- `test_session_latency.py` — also `perf`
- `test_session_control.py`

### Untagged (nightly-only)

#### *(untagged)* (1)
- `test_tracing.py`

---

## Tag-frequency summary

| Tag | Tests | Source |
|---|---:|---|
| `moe` | 15 | new |
| `quant` | 12 | reused |
| `speculative-decoding` | 10 | reused |
| `lora` | 9 | reused |
| `blackwell` | 8 | reused |
| `attention-backend` | 6 | new |
| `perf` | 6 | new |
| `rl` | 6 | new |
| `scoring` | 6 | new |
| `deepseek` | 5 | reused |
| `Multi-modal` | 3 | reused |
| `hicache` | 3 | reused |
| `piecewise-cuda-graph` | 3 | reused |
| `session` | 3 | new |
| `disaggregation` | 2 | new |
| `model-coverage` | 1 | new |
| `scheduler` | 1 | new |
| *(untagged — nightly-only)* | 1 | — |

Total: 62 tests / 17 tags + untagged.

---

## Edge cases worth noting before implementation

- **Hardware compatibility.** The runner needs a per-commit-suite → compatible-nightly-suite map so that, e.g., a `lora` label on a PR running `stage-b-test-1-gpu-large` only pulls `nightly-1-gpu` LoRA tests, not the four-GPU B200 LoRA tests. Without that, a label-pull will land a test on a runner that can't host it.
- **5090 / SM12.0 nightly slot.** The two `*_sm120` tests need a runner with SM12.0; today's `nightly-test-nvidia.yml` only has `1-gpu-h100`, where they would silently skip. They need a new `nightly-1-gpu-5090` job (or to remain on the per-commit `1-gpu-5090` runner via a different mechanism).
- **2-GPU nightly slot.** `nightly-2-gpu` is a registered suite name but no nightly job runs it today. A new `nightly-test-general-2-gpu-h100` job is required.
- **Subset moves.** Six files (`test_streaming_session.py`, `test_standalone_speculative_decoding.py`, `test_ngram_speculative_decoding.py`, `test_pcg_with_speculative_decoding.py`, `test_moe_ep.py`, `test_deepseek_v3_fp4_4gpu.py`) require a class-level split because only some classes move to nightly. The existing pattern in `test_streaming_session_swa.py` (sibling `sys.path.insert` import) is the precedent to follow.
- **AMD registrations.** Each test's `register_amd_ci(...)` is independent of `register_cuda_ci(...)` and should remain untouched — the AMD per-commit pipeline keeps these tests where they are.
