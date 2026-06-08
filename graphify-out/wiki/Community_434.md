# Community 434

> 14 nodes

## Key Concepts

- **.update_draining_state()** (6 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **Req** (6 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **._update_draining_loras()** (5 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **AdapterStats** (4 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **._update_adapter_stats()** (4 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **._update_fully_drained_loras()** (4 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **.can_schedule()** (3 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **lora_drainer.py** (2 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **._reset_stats()** (2 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **.is_starving()** (2 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **Update LoRA drainer state based on current waiting queue and running requests.** (1 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **Select LoRA adapters to drain based on starvation detection.          This metho** (1 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **Clear draining state for adapters that have fully drained.          An adapter i** (1 connections) — `python/sglang/srt/lora/lora_drainer.py`
- **Check if a request can be scheduled based on draining state.          If the ada** (1 connections) — `python/sglang/srt/lora/lora_drainer.py`

## Relationships

- [[Grammar Manager & HiCache Clear]] (6 shared connections)
- [[HiCache Controller & Radix Tree]] (2 shared connections)

## Source Files

- `python/sglang/srt/lora/lora_drainer.py`

## Audit Trail

- EXTRACTED: 40 (95%)
- INFERRED: 2 (5%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*