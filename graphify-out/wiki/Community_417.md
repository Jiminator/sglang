# Community 417

> 15 nodes

## Key Concepts

- **TokenizerManager** (12 connections) — `python/sglang/srt/managers/template_manager.py`
- **.load_chat_template()** (9 connections) — `python/sglang/srt/managers/template_manager.py`
- **._load_explicit_chat_template()** (8 connections) — `python/sglang/srt/managers/template_manager.py`
- **.initialize_templates()** (5 connections) — `python/sglang/srt/managers/template_manager.py`
- **._load_jinja_template()** (5 connections) — `python/sglang/srt/managers/template_manager.py`
- **.guess_chat_template_from_model_path()** (4 connections) — `python/sglang/srt/managers/template_manager.py`
- **._resolve_hf_chat_template()** (4 connections) — `python/sglang/srt/managers/template_manager.py`
- **._select_named_template()** (3 connections) — `python/sglang/srt/managers/template_manager.py`
- **get_conv_template_by_model_path()** (2 connections) — `python/sglang/srt/parser/conversation.py`
- **chat_template_exists()** (2 connections) — `python/sglang/srt/parser/conversation.py`
- **Load a chat template from various sources.          Args:             tokenizer_** (1 connections) — `python/sglang/srt/managers/template_manager.py`
- **Load explicitly specified chat template.** (1 connections) — `python/sglang/srt/managers/template_manager.py`
- **Infer chat template name from model path.          Args:             model_path:** (1 connections) — `python/sglang/srt/managers/template_manager.py`
- **Initialize all templates based on provided configuration.          Args:** (1 connections) — `python/sglang/srt/managers/template_manager.py`
- **Load a Jinja template file.** (1 connections) — `python/sglang/srt/managers/template_manager.py`

## Relationships

- [[Anthropic/OpenAI API Entrypoints]] (8 shared connections)
- [[Community 370]] (3 shared connections)
- [[Community 224]] (3 shared connections)
- [[Community 170]] (2 shared connections)
- [[Community 464]] (2 shared connections)
- [[Community 348]] (2 shared connections)
- [[Community 47]] (1 shared connections)

## Source Files

- `python/sglang/srt/managers/template_manager.py`
- `python/sglang/srt/parser/conversation.py`

## Audit Trail

- EXTRACTED: 46 (78%)
- INFERRED: 13 (22%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*