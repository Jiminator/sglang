# Community 383

> 16 nodes

## Key Concepts

- **in_the_same_node_as()** (11 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.__init__()** (8 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **quick_all_reduce.py** (5 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **is_full_nvlink()** (3 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py`
- **qr_rocm_arch_available()** (3 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **.init_quick_all_reduce()** (3 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **.create_shared_buffer()** (3 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **.barrier()** (3 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **QuickReduceRegime** (2 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **ProcessGroup** (1 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **device** (1 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **Custom allreduce provides non-destructive acceleration and is         available** (1 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **Creates a shared buffer for quickreduce.         Has to be called after init_cus** (1 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **# TODO: If the dtype is not bfloat16 or then float16,** (1 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **Barrier synchronization among the group.         NOTE: don't use `device_group`** (1 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **This is a collective operation that returns if each rank is in the same node** (1 connections) — `python/sglang/srt/distributed/parallel_state.py`

## Relationships

- [[Community 101]] (7 shared connections)
- [[Community 174]] (3 shared connections)
- [[Breakable CUDA Graph (TBO)]] (3 shared connections)
- [[Community 107]] (2 shared connections)
- [[Community 313]] (1 shared connections)
- [[Community 240]] (1 shared connections)
- [[Community 355]] (1 shared connections)

## Source Files

- `python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py`
- `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- `python/sglang/srt/distributed/parallel_state.py`

## Audit Trail

- EXTRACTED: 40 (83%)
- INFERRED: 8 (17%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*