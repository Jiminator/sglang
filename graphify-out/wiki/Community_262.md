# Community 262

> 25 nodes

## Key Concepts

- **Module** (11 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **.apply()** (4 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **.create_weights()** (4 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **dtype** (4 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **.apply()** (4 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **.apply()** (4 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **.get_quant_method()** (4 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **.create_weights()** (3 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **Tensor** (3 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **.process_weights_after_loading()** (3 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **.create_weights()** (3 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **.create_moe_runner()** (3 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **.get_triton_quant_info()** (3 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **.get_supported_act_dtypes()** (3 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **MoeRunnerConfig** (2 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **DispatchOutput** (2 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **CombineInput** (2 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **Create weights for a layer.          The weights will be set as attributes of th** (1 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **Apply the weights in layer to the input tensor.          Expects create_weights** (1 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **Process the weight after loading.          This can be used for example, to tran** (1 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **Create weights for a linear layer.            The weights will be set as attribu** (1 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **Apply the weights in layer to the input tensor.         Expects create_weights t** (1 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **Return a ``TritonMoeQuantInfo`` describing the quantisation state         stored** (1 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **List of supported activation dtypes.** (1 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **Get the quantize method to use for the quantized layer.          Args:** (1 connections) — `python/sglang/srt/layers/quantization/base_config.py`

## Relationships

- [[Compressed-Tensors Quant Linear]] (16 shared connections)
- [[Vision-Language Model Configs]] (2 shared connections)

## Source Files

- `python/sglang/srt/layers/quantization/base_config.py`

## Audit Trail

- EXTRACTED: 64 (91%)
- INFERRED: 6 (9%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*