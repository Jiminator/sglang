# Community 42

> 145 nodes

## Key Concepts

- **common.py** (193 connections) — `python/sglang/srt/utils/common.py`
- **is_flashinfer_available()** (15 connections) — `python/sglang/srt/utils/common.py`
- **get_device_memory_capacity()** (11 connections) — `python/sglang/srt/utils/common.py`
- **is_musa()** (9 connections) — `python/sglang/srt/utils/common.py`
- **cpu_has_amx_support()** (9 connections) — `python/sglang/srt/utils/common.py`
- **._handle_attention_backend_compatibility()** (8 connections) — `python/sglang/srt/server_args.py`
- **get_device()** (8 connections) — `python/sglang/srt/utils/common.py`
- **direct_register_custom_op()** (8 connections) — `python/sglang/srt/utils/common.py`
- **is_cuda()** (7 connections) — `python/sglang/srt/utils/common.py`
- **get_device_sm()** (7 connections) — `python/sglang/srt/utils/common.py`
- **._get_default_attn_backend()** (6 connections) — `python/sglang/srt/server_args.py`
- **get_device_capability()** (6 connections) — `python/sglang/srt/utils/common.py`
- **_process_weight_after_loading()** (6 connections) — `python/sglang/srt/utils/common.py`
- **is_cuda_alike()** (5 connections) — `python/sglang/srt/utils/common.py`
- **is_xpu()** (5 connections) — `python/sglang/srt/utils/common.py`
- **is_cpu()** (5 connections) — `python/sglang/srt/utils/common.py`
- **device_context()** (5 connections) — `python/sglang/srt/utils/common.py`
- **empty_device_cache()** (5 connections) — `python/sglang/srt/utils/common.py`
- **get_dispatch_device_backend()** (5 connections) — `python/sglang/srt/utils/common.py`
- **get_device_module()** (5 connections) — `python/sglang/srt/utils/common.py`
- **_wait_for_reap_or_raise()** (5 connections) — `python/sglang/srt/utils/common.py`
- **is_shm_available()** (5 connections) — `python/sglang/srt/utils/common.py`
- **is_npu()** (4 connections) — `python/sglang/srt/utils/common.py`
- **is_host_cpu_arm64()** (4 connections) — `python/sglang/srt/utils/common.py`
- **is_jpeg_with_cuda()** (4 connections) — `python/sglang/srt/utils/common.py`
- *... and 120 more nodes in this community*

## Relationships

- [[Pipeline Parallel & Custom Allreduce]] (34 shared connections)
- [[CLI Arg Parsing & Deprecation]] (17 shared connections)
- [[Community 47]] (11 shared connections)
- [[Hybrid Attention Backend]] (10 shared connections)
- [[Community 279]] (7 shared connections)
- [[Community 45]] (7 shared connections)
- [[Breakable CUDA Graph (TBO)]] (7 shared connections)
- [[Disaggregation Bootstrap & Decode]] (6 shared connections)
- [[Community 39]] (5 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (5 shared connections)
- [[Community 33]] (4 shared connections)
- [[Community 148]] (4 shared connections)

## Source Files

- `python/sglang/srt/distributed/device_communicators/hpu_communicator.py`
- `python/sglang/srt/layers/deep_gemm_wrapper/configurer.py`
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- `python/sglang/srt/layers/moe/utils.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 509 (87%)
- INFERRED: 73 (13%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*