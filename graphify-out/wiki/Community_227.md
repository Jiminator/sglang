# Community 227

> 29 nodes

## Key Concepts

- **StatelessProcessGroup** (10 connections) — `python/sglang/srt/distributed/utils.py`
- **utils.py** (8 connections) — `python/sglang/srt/distributed/utils.py`
- **_create_global_tcp_store()** (7 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.broadcast_obj()** (6 connections) — `python/sglang/srt/distributed/utils.py`
- **set_global_tcp_store()** (4 connections) — `python/sglang/srt/distributed/utils.py`
- **TCPStore** (4 connections) — `python/sglang/srt/distributed/utils.py`
- **get_global_tcp_store()** (4 connections) — `python/sglang/srt/distributed/utils.py`
- **.send_obj()** (4 connections) — `python/sglang/srt/distributed/utils.py`
- **Any** (4 connections) — `python/sglang/srt/distributed/utils.py`
- **.expire_data()** (4 connections) — `python/sglang/srt/distributed/utils.py`
- **.all_gather_obj()** (4 connections) — `python/sglang/srt/distributed/utils.py`
- **ensure_divisibility()** (3 connections) — `python/sglang/srt/distributed/utils.py`
- **.recv_obj()** (3 connections) — `python/sglang/srt/distributed/utils.py`
- **.barrier()** (3 connections) — `python/sglang/srt/distributed/utils.py`
- **.create()** (3 connections) — `python/sglang/srt/distributed/utils.py`
- **Create a global TCPStore for coordination across ranks.      This function creat** (1 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.__post_init__()** (1 connections) — `python/sglang/srt/distributed/utils.py`
- **Set the global TCPStore instance.      This should be called during distributed** (1 connections) — `python/sglang/srt/distributed/utils.py`
- **Get the existing global TCPStore.      This function provides access to the shar** (1 connections) — `python/sglang/srt/distributed/utils.py`
- **Ensure that numerator is divisible by the denominator.** (1 connections) — `python/sglang/srt/distributed/utils.py`
- **A dataclass to hold a metadata store, and the rank, world_size of the     group.** (1 connections) — `python/sglang/srt/distributed/utils.py`
- **Send an object to a destination rank.** (1 connections) — `python/sglang/srt/distributed/utils.py`
- **Expire data that is older than `data_expiration_seconds` seconds.** (1 connections) — `python/sglang/srt/distributed/utils.py`
- **Receive an object from a source rank.** (1 connections) — `python/sglang/srt/distributed/utils.py`
- **Broadcast an object from a source rank to all other ranks.         It does not c** (1 connections) — `python/sglang/srt/distributed/utils.py`
- *... and 4 more nodes in this community*

## Relationships

- [[Breakable CUDA Graph (TBO)]] (2 shared connections)
- [[Linear Layer Parameters]] (2 shared connections)
- [[Community 101]] (1 shared connections)
- [[Disaggregation Bootstrap & Decode]] (1 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)
- [[Community 80]] (1 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (1 shared connections)

## Source Files

- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/distributed/utils.py`

## Audit Trail

- EXTRACTED: 79 (93%)
- INFERRED: 6 (7%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*