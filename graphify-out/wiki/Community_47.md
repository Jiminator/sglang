# Community 47

> 132 nodes

## Key Concepts

- **RuntimeError** (313 connections) — `python/sglang/srt/entrypoints/engine.py`
- **common.py** (20 connections) — `python/sglang/srt/mem_cache/common.py`
- **BasePrefixCache** (15 connections) — `python/sglang/srt/mem_cache/common.py`
- **Tensor** (14 connections) — `python/sglang/srt/mem_cache/common.py`
- **alloc_token_slots()** (12 connections) — `python/sglang/srt/mem_cache/common.py`
- **alloc_paged_token_slots_extend()** (12 connections) — `python/sglang/srt/mem_cache/common.py`
- **Tensor** (12 connections) — `python/sglang/srt/speculative/ngram_info.py`
- **ScheduleBatch** (12 connections) — `python/sglang/srt/speculative/ngram_info.py`
- **cache_locs.py** (12 connections) — `python/sglang/srt/speculative/triton_ops/cache_locs.py`
- **get_last_loc()** (11 connections) — `python/sglang/srt/mem_cache/common.py`
- **kv_cache_scales_loader()** (11 connections) — `python/sglang/srt/model_loader/weight_utils.py`
- **.verify()** (10 connections) — `python/sglang/srt/speculative/ngram_info.py`
- **alloc_for_extend()** (9 connections) — `python/sglang/srt/mem_cache/common.py`
- **.prepare_for_decode()** (9 connections) — `python/sglang/srt/speculative/eagle_info_v2.py`
- **LogitsProcessorOutput** (9 connections) — `python/sglang/srt/speculative/ngram_info.py`
- **constexpr** (9 connections) — `python/sglang/srt/speculative/triton_ops/cache_locs.py`
- **assign_req_to_token_pool_func()** (9 connections) — `python/sglang/srt/speculative/triton_ops/cache_locs.py`
- **alloc_paged_token_slots_decode()** (8 connections) — `python/sglang/srt/mem_cache/common.py`
- **remote_instance_weight_loader_utils.py** (8 connections) — `python/sglang/srt/model_loader/remote_instance_weight_loader_utils.py`
- **.prepare_for_verify()** (8 connections) — `python/sglang/srt/speculative/eagle_info.py`
- **evict_from_tree_cache()** (7 connections) — `python/sglang/srt/mem_cache/common.py`
- **alloc_req_slots()** (7 connections) — `python/sglang/srt/mem_cache/common.py`
- **alloc_for_decode()** (7 connections) — `python/sglang/srt/mem_cache/common.py`
- **._draft_preprocess_decode()** (7 connections) — `python/sglang/srt/speculative/eagle_worker.py`
- **mindspore_runner.py** (6 connections) — `python/sglang/srt/model_executor/mindspore_runner.py`
- *... and 107 more nodes in this community*

## Relationships

- [[CLI Arg Parsing & Deprecation]] (25 shared connections)
- [[Hybrid Attention Backend]] (24 shared connections)
- [[Context-Parallel Attention]] (24 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (22 shared connections)
- [[Grammar Manager & HiCache Clear]] (18 shared connections)
- [[Disaggregation Bootstrap & Decode]] (17 shared connections)
- [[HiCache Controller & Radix Tree]] (14 shared connections)
- [[Community 33]] (12 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (12 shared connections)
- [[Community 42]] (11 shared connections)
- [[Breakable CUDA Graph (TBO)]] (9 shared connections)
- [[Community 35]] (9 shared connections)

## Source Files

- `python/sglang/srt/configs/device_config.py`
- `python/sglang/srt/disaggregation/nixl/conn.py`
- `python/sglang/srt/entrypoints/engine.py`
- `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`
- `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py`
- `python/sglang/srt/kv_canary/req_to_expected_token_ids_manager.py`
- `python/sglang/srt/kv_canary/single_forward_manager/manager.py`
- `python/sglang/srt/layers/attention/mamba/ops/layernorm_gated.py`
- `python/sglang/srt/layers/attention/vision_utils.py`
- `python/sglang/srt/layers/moe/hash_topk.py`
- `python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py`
- `python/sglang/srt/layers/vocab_parallel_embedding.py`
- `python/sglang/srt/lora/lora_manager.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/mem_cache/mamba_radix_cache.py`
- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/model_executor/mindspore_runner.py`
- `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
- `python/sglang/srt/model_executor/pool_configurator.py`
- `python/sglang/srt/model_loader/remote_instance_weight_loader_utils.py`

## Audit Trail

- EXTRACTED: 362 (43%)
- INFERRED: 486 (57%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*