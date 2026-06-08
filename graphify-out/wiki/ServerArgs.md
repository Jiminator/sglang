# ServerArgs

> God node · 1116 connections · `python/sglang/srt/server_args.py`

**Community:** [[CLI Arg Parsing & Deprecation]]

## Connections by Relation

### contains
- [[server_args.py]] `EXTRACTED`

### method
- [[.__post_init__()]] `EXTRACTED`
- [[._handle_model_specific_adjustments()]] `EXTRACTED`
- [[.get_model_config()]] `EXTRACTED`
- [[._handle_gpu_memory_settings()]] `EXTRACTED`
- [[._handle_attention_backend_compatibility()]] `EXTRACTED`
- [[._handle_load_format()]] `EXTRACTED`
- [[.use_mla_backend()]] `EXTRACTED`
- [[.get_attention_backends()]] `EXTRACTED`
- [[._get_default_attn_backend()]] `EXTRACTED`
- [[._handle_hicache()]] `EXTRACTED`
- [[._handle_kv4_compatibility()]] `EXTRACTED`
- [[._validate_ib_devices()]] `EXTRACTED`
- [[.check_server_args()]] `EXTRACTED`
- [[._resolve_io_decode_attention_compatibility()]] `EXTRACTED`
- [[._validate_cutedsl_a2a_token_budget()]] `EXTRACTED`
- [[._handle_deterministic_inference()]] `EXTRACTED`
- [[._handle_encoder_disaggregation()]] `EXTRACTED`
- [[._handle_mamba_radix_cache()]] `EXTRACTED`
- [[._handle_multi_item_scoring()]] `EXTRACTED`
- [[._handle_piecewise_cuda_graph()]] `EXTRACTED`

### rationale_for
- [[The arguments of the server.      NOTE: When you add new arguments, please make]] `EXTRACTED`

### references
- [[get_global_server_args()]] `EXTRACTED`
- [[prepare_server_args()]] `EXTRACTED`
- [[set_global_server_args_for_scheduler()]] `EXTRACTED`
- [[auto_choose_speculative_params()]] `EXTRACTED`
- [[.init_new()]] `EXTRACTED`

### uses
- [[MultimodalDataItem]] `INFERRED`
- [[MultimodalInputs]] `INFERRED`
- [[Modality]] `INFERRED`
- [[Req]] `INFERRED`
- [[ScheduleBatch]] `INFERRED`
- [[ModelConfigForExpertLocation]] `INFERRED`
- [[TokenizerManager]] `INFERRED`
- [[ModelConfig]] `INFERRED`
- [[RuntimeError]] `INFERRED`
- [[SpeculativeAlgorithm]] `INFERRED`
- [[SpecInput]] `INFERRED`
- [[NetworkAddress]] `INFERRED`
- [[Scheduler]] `INFERRED`
- [[UnifiedRadixCache]] `INFERRED`
- [[DllmConfig]] `INFERRED`
- [[FINISH_ABORT]] `INFERRED`
- [[ReasoningParser]] `INFERRED`
- [[BaseFinishReason]] `INFERRED`
- [[UnifiedTreeNode]] `INFERRED`
- [[MultimodalProcessorOutput]] `INFERRED`

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*