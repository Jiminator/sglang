# ForwardBatch

> God node · 2914 connections · `python/sglang/srt/model_executor/forward_batch_info.py`

**Community:** [[Vision-Language Model Configs]]

## Connections by Relation

### contains
- [[forward_batch_info.py]] `EXTRACTED`

### method
- [[.init_new()]] `EXTRACTED`
- [[.prepare_mlp_sync_batch()]] `EXTRACTED`
- [[._compute_mrope_positions()]] `EXTRACTED`
- [[.post_forward_mlp_sync_batch()]] `EXTRACTED`
- [[._init_ngram_embedding_info()]] `EXTRACTED`
- [[.prepare_attn_tp_scatter_input()]] `EXTRACTED`
- [[.adjust_num_token_non_padded_for_attn_tp()]] `EXTRACTED`
- [[.compute_spec_mrope_positions()]] `EXTRACTED`
- [[._maybe_init_non_generation_fields()]] `EXTRACTED`
- [[._pad_inputs_to_size()]] `EXTRACTED`
- [[.contains_mm_inputs()]] `EXTRACTED`
- [[._expand_mrope_from_input()]] `EXTRACTED`
- [[.apply_deprecated_skip_attn_backend_init()]] `EXTRACTED`
- [[.mark_forward_metadata_ready()]] `EXTRACTED`
- [[.merge_mm_inputs()]] `EXTRACTED`
- [[._pad_tensor_to_size()]] `EXTRACTED`
- [[.contains_audio_inputs()]] `EXTRACTED`
- [[.contains_image_inputs()]] `EXTRACTED`
- [[.contains_video_inputs()]] `EXTRACTED`
- [[.needs_forward_metadata_init()]] `EXTRACTED`

### rationale_for
- [[Store all inputs of a forward pass.]] `EXTRACTED`

### references
- [[build_inner_fb_view()]] `EXTRACTED`

### uses
- [[LogitsProcessor]] `INFERRED`
- [[RadixAttention]] `INFERRED`
- [[PretrainedConfig]] `INFERRED`
- [[MultimodalDataItem]] `INFERRED`
- [[MultimodalInputs]] `INFERRED`
- [[Modality]] `INFERRED`
- [[Req]] `INFERRED`
- [[LogitsProcessorOutput]] `INFERRED`
- [[MultiModalityDataPaddingPatternMultimodalTokens]] `INFERRED`
- [[Pooler]] `INFERRED`
- [[ScheduleBatch]] `INFERRED`
- [[LayerCommunicator]] `INFERRED`
- [[PoolingType]] `INFERRED`
- [[LayerScatterModes]] `INFERRED`
- [[SpeculativeAlgorithm]] `INFERRED`
- [[AttentionBackend]] `INFERRED`
- [[SpecInput]] `INFERRED`
- [[LlamaForCausalLM]] `INFERRED`
- [[Qwen2ForCausalLM]] `INFERRED`
- [[FINISH_ABORT]] `INFERRED`

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*