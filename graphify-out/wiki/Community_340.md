# Community 340

> 19 nodes

## Key Concepts

- **_shard_tensor()** (6 connections) — `python/sglang/srt/layers/model_parallel.py`
- **model_parallel.py** (5 connections) — `python/sglang/srt/layers/model_parallel.py`
- **RowwiseParallelMaybeWait** (5 connections) — `python/sglang/srt/layers/model_parallel.py`
- **tensor_parallel()** (5 connections) — `python/sglang/srt/layers/model_parallel.py`
- **ColwiseParallelSharded** (4 connections) — `python/sglang/srt/layers/model_parallel.py`
- **Shard** (3 connections) — `python/sglang/srt/layers/model_parallel.py`
- **._partition_linear_fn()** (3 connections) — `python/sglang/srt/layers/model_parallel.py`
- **DeviceMesh** (2 connections) — `python/sglang/srt/layers/model_parallel.py`
- **._partition_linear_fn()** (2 connections) — `python/sglang/srt/layers/model_parallel.py`
- **Tensor** (1 connections) — `python/sglang/srt/layers/model_parallel.py`
- **ColwiseParallel** (1 connections)
- **RowwiseParallel** (1 connections)
- **._prepare_output_fn()** (1 connections) — `python/sglang/srt/layers/model_parallel.py`
- **Module** (1 connections) — `python/sglang/srt/layers/model_parallel.py`
- **Common utilities for torch model parallelism.** (1 connections) — `python/sglang/srt/layers/model_parallel.py`
- **Locally shards a full tensor based on indicated sharding arrangement, and     re** (1 connections) — `python/sglang/srt/layers/model_parallel.py`
- **A version of ColwiseParallel where the local weight has been already     sharded** (1 connections) — `python/sglang/srt/layers/model_parallel.py`
- **A version of RowwiseParallel that waits for the output (establish dependency** (1 connections) — `python/sglang/srt/layers/model_parallel.py`
- **Tensor parallelize the model across the given device mesh.     Args:         mod** (1 connections) — `python/sglang/srt/layers/model_parallel.py`

## Relationships

- [[Hybrid Attention Backend]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/model_parallel.py`

## Audit Trail

- EXTRACTED: 44 (98%)
- INFERRED: 1 (2%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*