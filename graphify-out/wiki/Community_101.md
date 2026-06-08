# Community 101

> 62 nodes

## Key Concepts

- **QuickAllReduce** (20 connections) — `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- **PyMscclppCommunicator** (18 connections) — `python/sglang/srt/distributed/device_communicators/pymscclpp.py`
- **NpuCommunicator** (17 connections) — `python/sglang/srt/distributed/device_communicators/npu_communicator.py`
- **TorchSymmMemCommunicator** (17 connections) — `python/sglang/srt/distributed/device_communicators/torch_symm_mem.py`
- **Any** (17 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **HpuCommunicator** (16 connections) — `python/sglang/srt/distributed/device_communicators/hpu_communicator.py`
- **XpuCommunicator** (16 connections) — `python/sglang/srt/distributed/device_communicators/xpu_communicator.py`
- **.__init__()** (16 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **ProcessGroup** (12 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **Stream** (11 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **GraphCaptureContext** (10 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **timedelta** (10 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **Backend** (9 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **Size** (9 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **dtype** (9 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.broadcast_tensor_dict()** (8 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.broadcast_object_list()** (7 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.send_tensor_dict()** (7 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **_split_tensor_dict()** (6 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.broadcast_object()** (5 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.send_object()** (5 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.recv()** (5 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **graph_capture()** (5 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **.graph_capture()** (4 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **Tensor** (3 connections) — `python/sglang/srt/distributed/device_communicators/npu_communicator.py`
- *... and 37 more nodes in this community*

## Relationships

- [[Breakable CUDA Graph (TBO)]] (23 shared connections)
- [[Community 240]] (14 shared connections)
- [[Community 100]] (9 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (8 shared connections)
- [[Community 313]] (8 shared connections)
- [[Community 383]] (7 shared connections)
- [[Community 411]] (5 shared connections)
- [[Community 9594]] (2 shared connections)
- [[Community 42]] (1 shared connections)
- [[Community 9615]] (1 shared connections)
- [[Community 255]] (1 shared connections)
- [[Community 85]] (1 shared connections)

## Source Files

- `python/sglang/srt/distributed/device_communicators/hpu_communicator.py`
- `python/sglang/srt/distributed/device_communicators/npu_communicator.py`
- `python/sglang/srt/distributed/device_communicators/pymscclpp.py`
- `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- `python/sglang/srt/distributed/device_communicators/torch_symm_mem.py`
- `python/sglang/srt/distributed/device_communicators/xpu_communicator.py`
- `python/sglang/srt/distributed/parallel_state.py`

## Audit Trail

- EXTRACTED: 173 (54%)
- INFERRED: 147 (46%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*