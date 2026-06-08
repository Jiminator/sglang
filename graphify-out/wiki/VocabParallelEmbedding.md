# VocabParallelEmbedding

> God node · 1324 connections · `python/sglang/srt/layers/vocab_parallel_embedding.py`

**Community:** [[DeepSeek MLA Attention & MoE]]

## Connections by Relation

### contains
- [[vocab_parallel_embedding.py]] `EXTRACTED`

### inherits
- [[ParallelLMHead]] `EXTRACTED`

### method
- [[.__init__()]] `EXTRACTED`
- [[.forward()]] `EXTRACTED`
- [[._get_indices()]] `EXTRACTED`
- [[.get_sharded_to_full_mapping()]] `EXTRACTED`
- [[.weight_loader()]] `EXTRACTED`
- [[.extra_repr()]] `EXTRACTED`

### rationale_for
- [[Embedding parallelized in the vocabulary dimension.      Adapted from torch.nn.E]] `EXTRACTED`

### references
- [[.tie_weights()]] `EXTRACTED`

### uses
- [[QuantizationConfig]] `INFERRED`
- [[LogitsProcessor]] `INFERRED`
- [[PretrainedConfig]] `INFERRED`
- [[LogitsProcessorOutput]] `INFERRED`
- [[QuantizeMethodBase]] `INFERRED`
- [[LlamaForCausalLM]] `INFERRED`
- [[Qwen2ForCausalLM]] `INFERRED`
- [[DeepseekV2AttentionMLA]] `INFERRED`
- [[DeepseekV2ForCausalLM]] `INFERRED`
- [[Qwen2Model]] `INFERRED`
- [[LlamaMLP]] `INFERRED`
- [[DeepseekV3ForCausalLM]] `INFERRED`
- [[DeepseekV2MLP]] `INFERRED`
- [[LlamaDecoderLayer]] `INFERRED`
- [[Qwen2MoeMLP]] `INFERRED`
- [[FusedMoEWithLoRA]] `INFERRED`
- [[MiMoV2ForCausalLM]] `INFERRED`
- [[Qwen2MoeSparseMoeBlock]] `INFERRED`
- [[SiglipVisionModel]] `INFERRED`
- [[LoRAManager]] `INFERRED`

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*