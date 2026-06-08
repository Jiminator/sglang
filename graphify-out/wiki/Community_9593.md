# Community 9593

> 5 nodes

## Key Concepts

- **try_detect_fp4_experts()** (5 connections) — `python/sglang/srt/configs/deepseek_v4.py`
- **probe_routed_expert_weight_dtype()** (3 connections) — `python/sglang/srt/model_loader/weight_utils.py`
- **deepseek_v4.py** (2 connections) — `python/sglang/srt/configs/deepseek_v4.py`
- **True = mxfp4-packed (U8/I8/F4), False = converted FP8 (F8_E4M3),     None when t** (1 connections) — `python/sglang/srt/configs/deepseek_v4.py`
- **Return the safetensors dtype string (e.g. ``F8_E4M3``, ``U8``) of one     routed** (1 connections) — `python/sglang/srt/model_loader/weight_utils.py`

## Relationships

- [[Context-Parallel Attention]] (1 shared connections)
- [[Community 42]] (1 shared connections)
- [[Community 132]] (1 shared connections)
- [[Weight Loading & EPLB]] (1 shared connections)

## Source Files

- `python/sglang/srt/configs/deepseek_v4.py`
- `python/sglang/srt/model_loader/weight_utils.py`

## Audit Trail

- EXTRACTED: 8 (67%)
- INFERRED: 4 (33%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*