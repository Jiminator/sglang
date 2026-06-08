# GPTQ / Marlin Quantization

> 188 nodes

## Key Concepts

- **GPTQConfig** (54 connections) — `python/sglang/srt/layers/quantization/gptq/gptq.py`
- **GPTQMarlinConfig** (50 connections) — `python/sglang/srt/layers/quantization/gptq/gptq.py`
- **MoeWNA16Config** (43 connections) — `python/sglang/srt/layers/quantization/moe_wna16.py`
- **AutoRoundConfig** (35 connections) — `python/sglang/srt/layers/quantization/auto_round.py`
- **DummyConfig** (31 connections) — `python/sglang/srt/layers/quantization/__init__.py`
- **QuantizationConfig** (30 connections) — `python/sglang/srt/layers/quantization/__init__.py`
- **Module** (28 connections) — `python/sglang/srt/layers/quantization/gptq/gptq.py`
- **QuantizationConfig** (24 connections)
- **GPTQAscendConfig** (21 connections) — `python/sglang/srt/layers/quantization/gptq/gptq.py`
- **GPTQMarlinMoEMethod** (20 connections) — `python/sglang/srt/layers/quantization/gptq/gptq.py`
- **Any** (19 connections) — `python/sglang/srt/layers/quantization/auto_round.py`
- **GPTQLinearMethod** (19 connections) — `python/sglang/srt/layers/quantization/gptq/gptq.py`
- **GPTQMoEAscendMethod** (19 connections) — `python/sglang/srt/layers/quantization/gptq/gptq.py`
- **dtype** (18 connections) — `python/sglang/srt/layers/quantization/auto_round.py`
- **Module** (18 connections) — `python/sglang/srt/layers/quantization/auto_round.py`
- **GPTQMarlinLinearMethod** (18 connections) — `python/sglang/srt/layers/quantization/gptq/gptq.py`
- **GPTQMarlinLinearScheme** (18 connections) — `python/sglang/srt/layers/quantization/gptq/schemes/gptq_marlin.py`
- **GPTQLinearSchemeBase** (18 connections) — `python/sglang/srt/layers/quantization/gptq/schemes/gptq_scheme.py`
- **_()** (17 connections) — `python/sglang/srt/layers/quantization/gptq/gptq.py`
- **GPTQAscendLinearScheme** (17 connections) — `python/sglang/srt/layers/quantization/gptq/schemes/gptq_linear.py`
- **.apply_gptq_quant_layer()** (15 connections) — `python/sglang/srt/layers/quantization/auto_round.py`
- **__init__.py** (15 connections) — `python/sglang/srt/layers/quantization/gptq/__init__.py`
- **GPTQLinearAscendMethod** (15 connections) — `python/sglang/srt/layers/quantization/gptq/gptq.py`
- **GPTQMoEAscendScheme** (15 connections) — `python/sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py`
- **GPTQMarlinMoEScheme** (15 connections) — `python/sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py`
- *... and 163 more nodes in this community*

## Relationships

- [[Compressed-Tensors Quant Linear]] (132 shared connections)
- [[Linear Layer Parameters]] (63 shared connections)
- [[Community 41]] (37 shared connections)
- [[Vision-Language Model Configs]] (34 shared connections)
- [[Weight Loading & EPLB]] (17 shared connections)
- [[ROCm MoE Quantization]] (11 shared connections)
- [[Community 88]] (8 shared connections)
- [[Community 92]] (6 shared connections)
- [[DeepSeek MLA Attention & MoE]] (4 shared connections)
- [[Community 53]] (3 shared connections)
- [[Community 198]] (3 shared connections)
- [[Community 285]] (3 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py`
- `python/sglang/srt/hardware_backend/npu/quantization/gptq_kernels.py`
- `python/sglang/srt/layers/quantization/__init__.py`
- `python/sglang/srt/layers/quantization/auto_round.py`
- `python/sglang/srt/layers/quantization/gptq/__init__.py`
- `python/sglang/srt/layers/quantization/gptq/gptq.py`
- `python/sglang/srt/layers/quantization/gptq/schemes/__init__.py`
- `python/sglang/srt/layers/quantization/gptq/schemes/gptq_linear.py`
- `python/sglang/srt/layers/quantization/gptq/schemes/gptq_marlin.py`
- `python/sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py`
- `python/sglang/srt/layers/quantization/gptq/schemes/gptq_scheme.py`
- `python/sglang/srt/layers/quantization/gptq_cpu.py`
- `python/sglang/srt/layers/quantization/marlin_utils.py`
- `python/sglang/srt/layers/quantization/mlx.py`
- `python/sglang/srt/layers/quantization/moe_wna16.py`

## Audit Trail

- EXTRACTED: 700 (60%)
- INFERRED: 466 (40%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*