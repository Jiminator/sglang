# Community 339

> 19 nodes

## Key Concepts

- **AscendTransferEngine** (15 connections) — `python/sglang/srt/disaggregation/ascend/transfer_engine.py`
- **AscendKVManager** (7 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **.send_kvcache()** (6 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **.initialize()** (5 connections) — `python/sglang/srt/disaggregation/ascend/transfer_engine.py`
- **conn.py** (4 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **.__init__()** (4 connections) — `python/sglang/srt/disaggregation/ascend/transfer_engine.py`
- **.init_engine()** (3 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **AscendKVSender** (3 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **AscendKVReceiver** (3 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **AscendKVBootstrapServer** (3 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **DisaggregationMode** (3 connections) — `python/sglang/srt/disaggregation/ascend/transfer_engine.py`
- **.get_mla_kv_ptrs_with_pp()** (2 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **NDArray** (2 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **int32** (2 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **ThreadPoolExecutor** (2 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **._get_transfer_protocol()** (2 connections) — `python/sglang/srt/disaggregation/ascend/transfer_engine.py`
- **.register_buffer_to_engine()** (1 connections) — `python/sglang/srt/disaggregation/ascend/conn.py`
- **transfer_engine.py** (1 connections) — `python/sglang/srt/disaggregation/ascend/transfer_engine.py`
- **.batch_register()** (1 connections) — `python/sglang/srt/disaggregation/ascend/transfer_engine.py`

## Relationships

- [[Disaggregation Bootstrap & Decode]] (4 shared connections)
- [[Community 68]] (3 shared connections)
- [[Community 256]] (2 shared connections)
- [[Community 119]] (1 shared connections)
- [[Community 199]] (1 shared connections)
- [[Breakable CUDA Graph (TBO)]] (1 shared connections)
- [[Community 47]] (1 shared connections)

## Source Files

- `python/sglang/srt/disaggregation/ascend/conn.py`
- `python/sglang/srt/disaggregation/ascend/transfer_engine.py`

## Audit Trail

- EXTRACTED: 44 (64%)
- INFERRED: 25 (36%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*