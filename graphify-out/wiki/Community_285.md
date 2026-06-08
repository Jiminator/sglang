# Community 285

> 23 nodes

## Key Concepts

- **PetitNvFp4Config** (24 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **.get_quant_method()** (5 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **petit_utils.py** (5 connections) — `python/sglang/srt/layers/quantization/petit_utils.py`
- **.from_config()** (3 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **.override_quantization_method()** (3 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **.is_petit_nvfp4_compatible()** (3 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **prepare_nvfp4_layer_for_petit()** (3 connections) — `python/sglang/srt/layers/quantization/petit_utils.py`
- **apply_petit_nvfp4_linear()** (3 connections) — `python/sglang/srt/layers/quantization/petit_utils.py`
- **verify_petit_nvfp4_supported()** (3 connections) — `python/sglang/srt/layers/quantization/petit_utils.py`
- **petit.py** (2 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **.get_name()** (2 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **.get_supported_act_dtypes()** (2 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **.is_layer_excluded()** (2 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **.__init__()** (2 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **_check_petit_nvfp4_supported()** (2 connections) — `python/sglang/srt/layers/quantization/petit_utils.py`
- **.__init__()** (1 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **.get_min_capability()** (1 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **.get_config_filenames()** (1 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **.get_scaled_act_names()** (1 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **Config class for Petit FP4.** (1 connections) — `python/sglang/srt/layers/quantization/petit.py`
- **Module** (1 connections) — `python/sglang/srt/layers/quantization/petit_utils.py`
- **Tensor** (1 connections) — `python/sglang/srt/layers/quantization/petit_utils.py`
- **# TODO: Use auto-tuning to find the performant solution_id** (1 connections) — `python/sglang/srt/layers/quantization/petit_utils.py`

## Relationships

- [[Compressed-Tensors Quant Linear]] (15 shared connections)
- [[GPTQ / Marlin Quantization]] (3 shared connections)
- [[Weight Loading & EPLB]] (1 shared connections)
- [[Vision-Language Model Configs]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/quantization/petit.py`
- `python/sglang/srt/layers/quantization/petit_utils.py`

## Audit Trail

- EXTRACTED: 58 (81%)
- INFERRED: 14 (19%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*