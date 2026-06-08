# Community 121

> 54 nodes

## Key Concepts

- **ssd_combined.py** (14 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_combined.py`
- **ssd_chunk_state.py** (9 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py`
- **_mamba_chunk_scan_combined_fwd()** (9 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_combined.py`
- **ssu_dispatch.py** (7 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py`
- **MambaSSUBackend** (7 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py`
- **TritonSSUBackend** (7 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py`
- **FlashInferSSUBackend** (7 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py`
- **__init__.py** (6 connections) — `python/sglang/srt/layers/attention/mamba/ops/__init__.py`
- **mamba_ssm.py** (5 connections) — `python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py`
- **mamba_chunk_scan_combined()** (5 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_combined.py`
- **Tensor** (5 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py`
- **initialize_mamba_selective_state_update_backend()** (5 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py`
- **softplus()** (4 connections) — `python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py`
- **_bmm_chunk_fwd()** (4 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_bmm.py`
- **selective_state_update()** (4 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py`
- **_selective_scan_update_kernel()** (3 connections) — `python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py`
- **ssd_bmm.py** (3 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_bmm.py`
- **ssd_chunk_scan.py** (3 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_scan.py`
- **_chunk_scan_fwd()** (3 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_scan.py`
- **_chunk_cumsum_fwd_kernel()** (3 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py`
- **constexpr** (3 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py`
- **_chunk_cumsum_fwd()** (3 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py`
- **_chunk_state_fwd()** (3 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py`
- **chunk_state_varlen()** (3 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py`
- **ssd_state_passing.py** (3 connections) — `python/sglang/srt/layers/attention/mamba/ops/ssd_state_passing.py`
- *... and 29 more nodes in this community*

## Relationships

- [[CLI Arg Parsing & Deprecation]] (5 shared connections)
- [[Aibrix KV Cache Storage]] (2 shared connections)
- [[Vision-Language Model Configs]] (1 shared connections)
- [[Community 32]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/mamba/ops/__init__.py`
- `python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py`
- `python/sglang/srt/layers/attention/mamba/ops/ssd_bmm.py`
- `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_scan.py`
- `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py`
- `python/sglang/srt/layers/attention/mamba/ops/ssd_combined.py`
- `python/sglang/srt/layers/attention/mamba/ops/ssd_state_passing.py`
- `python/sglang/srt/layers/attention/mamba/ops/ssu_dispatch.py`

## Audit Trail

- EXTRACTED: 164 (96%)
- INFERRED: 7 (4%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*