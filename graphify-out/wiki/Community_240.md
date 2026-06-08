# Community 240

> 27 nodes

## Key Concepts

- **Tensor** (34 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.all_reduce()** (10 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **inplace_all_reduce()** (6 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.recv_tensor_dict()** (6 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **outplace_all_reduce()** (4 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.quant_all_reduce()** (4 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **._all_reduce_out_place()** (4 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **._all_reduce_in_place()** (4 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **._reduce_scatter_tensor()** (4 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.reduce_scatter()** (4 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.cp_all_gather_into_tensor_async()** (4 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.broadcast()** (4 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.recv_object()** (4 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **reg_reduce_scatter_tensor()** (3 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **reg_all_to_all_single()** (3 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **._all_to_all_single()** (3 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.reduce_scatterv()** (3 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.gather()** (3 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.send()** (3 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **User-facing all-reduce function before we actually call the         all-reduce o** (1 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **User-facing quant-all-reduce function similar to all-reduce. (NPU support only)** (1 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **Implement an asynchronous `allgather` operation on a specified stream.         (** (1 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **NOTE: We assume that the input tensor is on the same device across         all t** (1 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **Broadcast the input tensor.         NOTE: `src` is the local rank of the source** (1 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **Receive the input object list from the source rank.** (1 connections) — `python/sglang/srt/distributed/parallel_state.py`
- *... and 2 more nodes in this community*

## Relationships

- [[Breakable CUDA Graph (TBO)]] (23 shared connections)
- [[Community 101]] (14 shared connections)
- [[Community 49]] (2 shared connections)
- [[Community 100]] (1 shared connections)
- [[Community 313]] (1 shared connections)
- [[NCCL Symmetric Memory]] (1 shared connections)
- [[Community 42]] (1 shared connections)
- [[Community 383]] (1 shared connections)

## Source Files

- `python/sglang/srt/distributed/parallel_state.py`

## Audit Trail

- EXTRACTED: 107 (91%)
- INFERRED: 11 (9%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*