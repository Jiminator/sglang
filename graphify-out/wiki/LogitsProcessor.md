# LogitsProcessor

> God node · 1537 connections · `python/sglang/srt/layers/logits_processor.py`

**Community:** [[DeepSeek MLA Attention & MoE]]

## Connections by Relation

### contains
- [[logits_processor.py]] `EXTRACTED`

### method
- [[._get_logits()]] `EXTRACTED`
- [[.forward()]] `EXTRACTED`
- [[.compute_logprobs_for_multi_item_scoring()]] `EXTRACTED`
- [[.process_input_logprobs_by_chunk()]] `EXTRACTED`
- [[._compute_lm_head()]] `EXTRACTED`
- [[._get_dllm_logits()]] `EXTRACTED`
- [[._gather_dp_attn_hidden_states()]] `EXTRACTED`
- [[.process_input_logprobs()]] `EXTRACTED`
- [[._gather_attn_tp_logits()]] `EXTRACTED`
- [[.__init__()]] `EXTRACTED`
- [[._scatter_dp_attn_logits()]] `EXTRACTED`
- [[._copy_logits_to_buffer()]] `EXTRACTED`
- [[._get_hidden_states_to_store()]] `EXTRACTED`
- [[._get_pruned_states()]] `EXTRACTED`

### uses
- [[ForwardBatch]] `INFERRED`
- [[VocabParallelEmbedding]] `INFERRED`
- [[PretrainedConfig]] `INFERRED`
- [[ForwardMode]] `INFERRED`
- [[LlamaForCausalLM]] `INFERRED`
- [[CaptureHiddenMode]] `INFERRED`
- [[Qwen2ForCausalLM]] `INFERRED`
- [[DeepseekV2AttentionMLA]] `INFERRED`
- [[DeepseekV2ForCausalLM]] `INFERRED`
- [[Qwen3ForCausalLM]] `INFERRED`
- [[Qwen2Model]] `INFERRED`
- [[DpPaddingMode]] `INFERRED`
- [[LlamaMLP]] `INFERRED`
- [[DeepseekV3ForCausalLM]] `INFERRED`
- [[DeepseekV2MLP]] `INFERRED`
- [[LlamaDecoderLayer]] `INFERRED`
- [[Qwen2MoeMLP]] `INFERRED`
- [[MiMoV2ForCausalLM]] `INFERRED`
- [[Qwen2MoeSparseMoeBlock]] `INFERRED`
- [[Qwen3MoeForCausalLM]] `INFERRED`

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*