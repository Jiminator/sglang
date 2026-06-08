# Community 330

> 20 nodes

## Key Concepts

- **utils.py** (19 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **set_default_server_args()** (4 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **process_shared_expert()** (4 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **FusedMoEMode** (3 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **init_zbal()** (3 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **get_share_stream()** (3 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **get_routed_stream()** (3 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **process_routed_expert()** (3 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **._handle_npu_backends()** (3 connections) — `python/sglang/srt/server_args.py`
- **get_npu_memory_capacity()** (3 connections) — `python/sglang/srt/utils/common.py`
- **init_npu_backend()** (2 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **get_indexer_weight_stream()** (2 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **set_share_stream()** (2 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **set_routed_stream()** (2 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **wait_share_stream()** (2 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **wait_routed_stream()** (2 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **_call_once()** (1 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **Set default server arguments for NPU backend.** (1 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **Initialize NPU backend. This function should be called only once.** (1 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`
- **init zbal, if is mix alloc mode, only register for sma & comm** (1 connections) — `python/sglang/srt/hardware_backend/npu/utils.py`

## Relationships

- [[Community 51]] (3 shared connections)
- [[CLI Arg Parsing & Deprecation]] (3 shared connections)
- [[Community 42]] (2 shared connections)
- [[Community 107]] (1 shared connections)
- [[Hybrid Attention Backend]] (1 shared connections)
- [[Community 419]] (1 shared connections)
- [[Community 49]] (1 shared connections)
- [[Community 32]] (1 shared connections)
- [[DeepSeek MLA Attention & MoE]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/npu/utils.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 56 (88%)
- INFERRED: 8 (12%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*