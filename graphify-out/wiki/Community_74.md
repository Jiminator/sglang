# Community 74

> 84 nodes

## Key Concepts

- **NaiveDistributed** (25 connections) — `python/sglang/srt/distributed/naive_distributed.py`
- **offloader.py** (23 connections) — `python/sglang/srt/utils/offloader.py`
- **HostSharedMemoryManager** (22 connections) — `python/sglang/srt/utils/host_shared_memory.py`
- **_BaseParamOffloader** (15 connections) — `python/sglang/srt/utils/offloader.py`
- **BaseOffloader** (13 connections) — `python/sglang/srt/utils/offloader.py`
- **_ModuleOffloader** (13 connections) — `python/sglang/srt/utils/offloader.py`
- **OffloaderV2** (11 connections) — `python/sglang/srt/utils/offloader.py`
- **OffloaderV1** (10 connections) — `python/sglang/srt/utils/offloader.py`
- **Module** (9 connections) — `python/sglang/srt/utils/offloader.py`
- **_MetaParamOffloader** (9 connections) — `python/sglang/srt/utils/offloader.py`
- **_ShmCpuParamOffloader** (9 connections) — `python/sglang/srt/utils/offloader.py`
- **_ShardedGpuParamOffloader** (9 connections) — `python/sglang/srt/utils/offloader.py`
- **_CpuParamOffloader** (8 connections) — `python/sglang/srt/utils/offloader.py`
- **.__init__()** (8 connections) — `python/sglang/srt/utils/offloader.py`
- **_SubmoduleAccessor** (7 connections) — `python/sglang/srt/utils/offloader.py`
- **_WhitelistParamNamesCreator** (7 connections) — `python/sglang/srt/utils/offloader.py`
- **NoopOffloader** (7 connections) — `python/sglang/srt/utils/offloader.py`
- **.wrap_modules()** (7 connections) — `python/sglang/srt/utils/offloader.py`
- **_move_param_to_meta()** (7 connections) — `python/sglang/srt/utils/offloader.py`
- **Tensor** (7 connections) — `python/sglang/srt/utils/offloader.py`
- **.post_init()** (7 connections) — `python/sglang/srt/utils/offloader.py`
- **get_naive_distributed()** (6 connections) — `python/sglang/srt/distributed/naive_distributed.py`
- **._malloc_raw()** (6 connections) — `python/sglang/srt/utils/host_shared_memory.py`
- **create_offloader_from_server_args()** (6 connections) — `python/sglang/srt/utils/offloader.py`
- **.__init__()** (6 connections) — `python/sglang/srt/utils/offloader.py`
- *... and 59 more nodes in this community*

## Relationships

- [[Compressed-Tensors Quant Linear]] (17 shared connections)
- [[CLI Arg Parsing & Deprecation]] (16 shared connections)
- [[Aibrix KV Cache Storage]] (4 shared connections)
- [[Context-Parallel Attention]] (4 shared connections)
- [[Hybrid Attention Backend]] (3 shared connections)
- [[Community 42]] (1 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (1 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (1 shared connections)

## Source Files

- `python/sglang/srt/distributed/naive_distributed.py`
- `python/sglang/srt/utils/common.py`
- `python/sglang/srt/utils/host_shared_memory.py`
- `python/sglang/srt/utils/offloader.py`

## Audit Trail

- EXTRACTED: 279 (69%)
- INFERRED: 128 (31%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*