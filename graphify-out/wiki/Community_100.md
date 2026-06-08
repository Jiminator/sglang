# Community 100

> 63 nodes

## Key Concepts

- **PyNcclCommunicator** (35 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **NCCLLibrary** (28 connections) — `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py`
- **buffer_type** (24 connections) — `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py`
- **cudaStream_t** (22 connections) — `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py`
- **.NCCL_CHECK()** (19 connections) — `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py`
- **ncclComm_t** (18 connections) — `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py`
- **Tensor** (15 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **._resolve_stream()** (11 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **ncclUniqueId** (11 connections) — `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py`
- **.__init__()** (10 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **Stream** (10 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **ReduceOp** (10 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **ncclDataTypeEnum** (9 connections) — `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py`
- **ncclRedOpTypeEnum** (9 connections) — `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py`
- **ProcessGroup** (8 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **StatelessProcessGroup** (8 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **device** (8 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **.all_reduce()** (7 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **.outplace_all_reduce()** (6 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **.cp_all_gather_into_tensor()** (6 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **.reduce_scatter()** (6 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **.broadcast()** (6 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **pynccl_wrapper.py** (6 connections) — `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py`
- **.all_gather()** (5 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- **.send()** (5 connections) — `python/sglang/srt/distributed/device_communicators/pynccl.py`
- *... and 38 more nodes in this community*

## Relationships

- [[Community 101]] (9 shared connections)
- [[Breakable CUDA Graph (TBO)]] (1 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (1 shared connections)
- [[Community 240]] (1 shared connections)
- [[Mamba2 / Hybrid Linear Attention]] (1 shared connections)
- [[Community 47]] (1 shared connections)

## Source Files

- `python/sglang/srt/distributed/device_communicators/pynccl.py`
- `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py`

## Audit Trail

- EXTRACTED: 252 (63%)
- INFERRED: 150 (37%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*