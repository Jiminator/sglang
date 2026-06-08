# Community 279

> 23 nodes

## Key Concepts

- **._handle_model_specific_adjustments()** (25 connections) — `python/sglang/srt/server_args.py`
- **aiter_can_use_preshuffle_paged_mqa()** (5 connections) — `python/sglang/srt/layers/attention/dsa/utils.py`
- **._handle_mamba_radix_cache()** (4 connections) — `python/sglang/srt/server_args.py`
- **get_nvidia_driver_version()** (4 connections) — `python/sglang/srt/utils/common.py`
- **apply_deepseek_v4_defaults()** (3 connections) — `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
- **validate_deepseek_v4_cp()** (3 connections) — `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
- **apply_nemotron_h_defaults()** (3 connections) — `python/sglang/srt/arg_groups/nemotron_h_hook.py`
- **.enable_mamba_extra_buffer()** (3 connections) — `python/sglang/srt/server_args.py`
- **has_fp8_weights_in_checkpoint()** (3 connections) — `python/sglang/srt/utils/common.py`
- **get_nvidia_driver_version_str()** (3 connections) — `python/sglang/srt/utils/common.py`
- **deepseek_v4_hook.py** (2 connections) — `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
- **._set_default_dsa_kv_cache_dtype()** (2 connections) — `python/sglang/srt/server_args.py`
- **.is_attention_backend_not_set()** (2 connections) — `python/sglang/srt/server_args.py`
- **.enable_mamba_extra_buffer_lazy()** (2 connections) — `python/sglang/srt/server_args.py`
- **is_triton_kernels_available()** (2 connections) — `python/sglang/srt/utils/common.py`
- **Apply DeepSeek V4 model-specific server arg defaults and constraints.** (1 connections) — `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
- **Validate DeepSeek V4 context-parallel configuration.** (1 connections) — `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
- **nemotron_h_hook.py** (1 connections) — `python/sglang/srt/arg_groups/nemotron_h_hook.py`
- **Apply NemotronH model-specific server arg defaults and constraints.** (1 connections) — `python/sglang/srt/arg_groups/nemotron_h_hook.py`
- **Whether aiter's preshuffle paged MQA / cache kernels can be used on this runtime** (1 connections) — `python/sglang/srt/layers/attention/dsa/utils.py`
- **Check if a model checkpoint actually contains FP8 (float8_e4m3fn) expert     wei** (1 connections) — `python/sglang/srt/utils/common.py`
- **Return the NVIDIA driver version as a tuple of ints, e.g. (595, 58, 3).     Retu** (1 connections) — `python/sglang/srt/utils/common.py`
- **Return the NVIDIA driver version string, e.g. '595.58.03'.     Returns None on f** (1 connections) — `python/sglang/srt/utils/common.py`

## Relationships

- [[CLI Arg Parsing & Deprecation]] (11 shared connections)
- [[Community 42]] (7 shared connections)
- [[Community 132]] (2 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)
- [[Community 45]] (1 shared connections)
- [[Disaggregation Utils & Cache Tests]] (1 shared connections)
- [[Community 36]] (1 shared connections)
- [[Community 37]] (1 shared connections)
- [[Community 9591]] (1 shared connections)

## Source Files

- `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
- `python/sglang/srt/arg_groups/nemotron_h_hook.py`
- `python/sglang/srt/layers/attention/dsa/utils.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 51 (69%)
- INFERRED: 23 (31%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*