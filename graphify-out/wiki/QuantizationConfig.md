# QuantizationConfig

> God node · 2376 connections · `python/sglang/srt/layers/quantization/base_config.py`

**Community:** [[Vision-Language Model Configs]]

## Connections by Relation

### contains
- [[base_config.py]] `EXTRACTED`

### inherits
- [[ABC]] `EXTRACTED`

### method
- [[.get_from_keys()]] `EXTRACTED`
- [[.get_from_keys_or()]] `EXTRACTED`
- [[.get_quant_method()]] `EXTRACTED`
- [[.from_config()]] `EXTRACTED`
- [[.get_supported_act_dtypes()]] `EXTRACTED`
- [[.apply_weight_name_mapper()]] `EXTRACTED`
- [[.get_config_filenames()]] `EXTRACTED`
- [[.get_min_capability()]] `EXTRACTED`
- [[.get_name()]] `EXTRACTED`
- [[.get_scaled_act_names()]] `EXTRACTED`
- [[._modelopt_override_quantization_method()]] `EXTRACTED`
- [[.override_quantization_method()]] `EXTRACTED`
- [[.__init__()]] `EXTRACTED`
- [[.update_packed_modules_mapping()]] `EXTRACTED`

### rationale_for
- [[Base class for quantization configs.]] `EXTRACTED`

### uses
- [[RowParallelLinear]] `INFERRED`
- [[RadixAttention]] `INFERRED`
- [[ParallelLMHead]] `INFERRED`
- [[VocabParallelEmbedding]] `INFERRED`
- [[PretrainedConfig]] `INFERRED`
- [[QKVParallelLinear]] `INFERRED`
- [[ColumnParallelLinear]] `INFERRED`
- [[MergedColumnParallelLinear]] `INFERRED`
- [[ReplicatedLinear]] `INFERRED`
- [[SiluAndMul]] `INFERRED`
- [[LinearBase]] `INFERRED`
- [[LlamaForCausalLM]] `INFERRED`
- [[Qwen2ForCausalLM]] `INFERRED`
- [[AttentionType]] `INFERRED`
- [[DeepseekV2AttentionMLA]] `INFERRED`
- [[TritonMoeQuantInfo]] `INFERRED`
- [[Fp8Config]] `INFERRED`
- [[DeepseekV2ForCausalLM]] `INFERRED`
- [[Qwen3ForCausalLM]] `INFERRED`
- [[GeluAndMul]] `INFERRED`

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*