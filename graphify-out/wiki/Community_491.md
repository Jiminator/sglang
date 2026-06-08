# Community 491

> 11 nodes

## Key Concepts

- **cmo.py** (8 connections) — `python/sglang/srt/hardware_backend/npu/cmo.py`
- **get_cmo_stream()** (5 connections) — `python/sglang/srt/hardware_backend/npu/cmo.py`
- **prepare_weight_cache()** (5 connections) — `python/sglang/srt/hardware_backend/npu/cmo.py`
- **shared_expert_on_independent_stream()** (4 connections) — `python/sglang/srt/hardware_backend/npu/cmo.py`
- **wait_cmo_stream()** (3 connections) — `python/sglang/srt/hardware_backend/npu/cmo.py`
- **get_share_stream()** (3 connections) — `python/sglang/srt/hardware_backend/npu/cmo.py`
- **set_cmo_stream()** (2 connections) — `python/sglang/srt/hardware_backend/npu/cmo.py`
- **set_share_stream()** (2 connections) — `python/sglang/srt/hardware_backend/npu/cmo.py`
- **wait_share_stream()** (2 connections) — `python/sglang/srt/hardware_backend/npu/cmo.py`
- **Cache Management Operation(CMO).     Launch a new stream to prefetch the weight** (1 connections) — `python/sglang/srt/hardware_backend/npu/cmo.py`
- **PREFETCH_MAX_SIZE: maximum size (bytes) for each prefetch operation.     This af** (1 connections) — `python/sglang/srt/hardware_backend/npu/cmo.py`

## Relationships

- [[Model Configs & Pooler]] (2 shared connections)
- [[NCCL Symmetric Memory]] (1 shared connections)
- [[Qwen3 / Kimi Model Configs]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/npu/cmo.py`

## Audit Trail

- EXTRACTED: 32 (89%)
- INFERRED: 4 (11%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*