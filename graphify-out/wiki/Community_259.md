# Community 259

> 25 nodes

## Key Concepts

- **layernorm_gated.py** (10 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **calc_rows_per_block()** (5 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **_layer_norm_fwd()** (5 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **_get_sm_count()** (4 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **rms_norm_gated()** (4 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **layernorm_fn()** (4 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **LayerNorm** (4 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **.__init__()** (4 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **.__init__()** (4 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **.forward()** (3 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **.reset_parameters()** (3 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **.forward()** (3 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **_layer_norm_fwd_1pass_kernel()** (2 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **device** (2 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **LayerNormFn** (2 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **.forward()** (2 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **rms_norm_ref()** (1 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **constexpr** (1 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **.reset_parameters()** (1 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **Get and cache the SM count for a given device.** (1 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * sil** (1 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **If group_size is not None, we do GroupNorm with each group having group_size ele** (1 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * sil** (1 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **If group_size is not None, we do GroupNorm with each group having group_size ele** (1 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`
- **If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * sil** (1 connections) — `python/sglang/srt/layers/attention/fla/layernorm_gated.py`

## Relationships

- [[Qwen3 / Kimi Model Configs]] (4 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)
- [[Community 47]] (1 shared connections)
- [[Community 42]] (1 shared connections)
- [[DeepSeek MLA Attention & MoE]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/attention/fla/layernorm_gated.py`

## Audit Trail

- EXTRACTED: 66 (94%)
- INFERRED: 4 (6%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*