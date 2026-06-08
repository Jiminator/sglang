# Community 255

> 25 nodes

## Key Concepts

- **.__init__()** (9 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- **dispatch_custom_allreduce()** (8 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- **.__init__()** (8 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **can_use_custom_all_reduce_v2()** (8 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **ProcessGroup** (7 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- **custom_all_reduce_v2.py** (7 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **custom_all_reduce.py** (5 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- **device** (5 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- **.create_shared_buffer()** (5 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- **._share_list()** (4 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **_maybe_init_config()** (4 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **_use_amd_deterministic_impl()** (3 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- **.capture()** (3 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **._post_init_obj()** (3 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **ModeConfig** (2 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **ProcessGroup** (2 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **device** (2 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **.override_shot()** (2 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **Args:             group: the process group to work on. If None, it will use the** (1 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- **Creates a shared buffer and returns a list of pointers         representing the** (1 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- **Return the CustomAllreduce class to use (aiter on ROCm if enabled).      On AMD** (1 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- **T** (1 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **# NOTE: This result is based on benchmarks on B200 GPUs** (1 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **# NOTE: This result is based on benchmarks on H200 GPUs** (1 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- **# TODO: tune on more GPUs, e.g A100** (1 connections) — `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`

## Relationships

- [[DeepSeek MLA Attention & MoE]] (9 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (8 shared connections)
- [[Community 174]] (5 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (2 shared connections)
- [[Community 45]] (1 shared connections)
- [[Community 101]] (1 shared connections)

## Source Files

- `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`
- `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`

## Audit Trail

- EXTRACTED: 81 (86%)
- INFERRED: 13 (14%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*