# Community 256

> 25 nodes

## Key Concepts

- **MooncakeTransferEngine** (19 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **mooncake_transfer_engine.py** (5 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **parse_ib_device_config()** (5 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **get_ib_devices_for_gpu()** (5 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **.initialize()** (5 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **.__init__()** (4 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **.batch_register()** (3 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **.batch_transfer_sync()** (3 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **get_mooncake_transfer_engine()** (3 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **.batch_deregister()** (2 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **.transfer_sync()** (2 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **.register()** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **.deregister()** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **.get_session_id()** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **.get_engine()** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **.get_ib_device()** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **Parse IB device config from a shared string, JSON mapping, or JSON file.** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **Parse IB device string and get IB devices for a specific GPU ID.      Supports a** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **Shared Mooncake transfer engine for RDMA/transfer operations.** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **Batch register multiple memory regions.** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **Batch deregister multiple memory regions.** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **Initialize the mooncake instance.** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **Synchronously transfer data to the specified address.** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **Synchronously transfer data to the specified addresses in batches.** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
- **Return the shared MooncakeTransferEngine if initialized, else None.** (1 connections) — `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`

## Relationships

- [[Disaggregation Bootstrap & Decode]] (5 shared connections)
- [[Community 47]] (4 shared connections)
- [[Community 339]] (2 shared connections)
- [[CLI Arg Parsing & Deprecation]] (1 shared connections)
- [[Breakable CUDA Graph (TBO)]] (1 shared connections)
- [[Community 119]] (1 shared connections)

## Source Files

- `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`

## Audit Trail

- EXTRACTED: 59 (84%)
- INFERRED: 11 (16%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*