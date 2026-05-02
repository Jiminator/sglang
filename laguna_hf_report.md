# `poolside/Laguna-XS.2` — checkpoint loads cleanly on `transformers >= 5.7.0`, silently breaks on `<= 5.6.0`

**TL;DR.** The published `modeling_laguna.py` shim in the model repo declares an MoE parameter layout that does not match the published safetensors layout (split-per-expert vs packed, `shared_expert` vs `shared_experts`, `e_score_correction_bias` under experts vs router). Native `transformers >= 5.7.0` (PR [huggingface/transformers#45673](https://github.com/huggingface/transformers/pull/45673)) papers over this mismatch with `WeightRenaming` rules in `src/transformers/conversion_mapping.py`, so loading just works. On `transformers < 5.7.0`, neither path works: `trust_remote_code=False` rejects the config, and `trust_remote_code=True` randomly re-initializes ~13 B params via `nn.init.normal_` (silent on meta device, slow-hang on CPU/GPU). The model card's "supported in Transformers v5.7.0 and later" line implies a soft compatibility floor; in practice it is a hard correctness floor.

This report documents the issue empirically so it can be triaged either as (a) a model-card wording fix, (b) an in-repo `modeling_laguna.py` fix to be self-contained, or (c) closed as already-resolved upstream.

## Environment

| | |
|---|---|
| Model | `poolside/Laguna-XS.2` (snapshot `be45e1e5a306aa9325cb43fa67164d2e26f78c16`) |
| Hardware | NVIDIA H200 (143 GiB free), 1.8 TiB host RAM, CUDA 13.0 |
| Python | 3.12.3 |
| torch | 2.11.0+cu130 |
| transformers tested | `5.6.0` and `5.7.0` (isolated venvs, project venv untouched) |

## Three structural mismatches between the in-repo modeling code and the safetensors checkpoint

Confirmed without loading any weights, by diffing the safetensors index against the parameter graph the in-repo `modeling_laguna.py` builds on meta device:

| # | Checkpoint key (safetensors index) | What the in-repo `modeling_laguna.py` declares |
|---|---|---|
| 1 | `model.layers.{L}.mlp.experts.{0..255}.{gate,up,down}_proj.weight` (split per expert) | `mlp.experts.gate_up_proj` of shape `[256, 2*intermediate, hidden]` and `mlp.experts.down_proj` of shape `[256, hidden, intermediate]` (packed) |
| 2 | `model.layers.{L}.mlp.shared_expert.{gate,up,down}_proj.weight` (singular) | `mlp.shared_experts.{gate,up,down}_proj.weight` (plural — `modeling_laguna.py:235`) |
| 3 | `model.layers.{L}.mlp.experts.e_score_correction_bias` (under experts) | `mlp.gate.e_score_correction_bias` (under router — `modeling_laguna.py:166`) |

Total: 234 parameter tensors in the model graph have no source in the checkpoint, and 30,108 tensors in the checkpoint have no target in the model graph (across 39 MoE layers, 256 experts each).

## Empirical matrix

Same machine, same cached weights, only `transformers` version flipped, `device_map="meta"` so we capture the loader's missing/unexpected report without the CPU `init.normal_` cost.

| transformers | Path | missing | unexpected | Generation | Outcome |
|---|---|---:|---:|---|---|
| **5.6.0** | `trust_remote_code=False` | — | — | — | **Fails at config load**: `ValueError: contains custom code which must be executed` (no native `model_type=laguna`) |
| **5.6.0** | `trust_remote_code=True`  | **234** | **30,108** | garbage | **Loads silently broken**; on `device_map="cpu"` the missing-key re-init via `_init_weights → nn.init.normal_` over the packed `gate_up_proj` tensors (39 × 256 × 1024 × 2048 ≈ 21 B elements) effectively hangs the process |
| **5.7.0** | `trust_remote_code=False` | 0 | 0 | `"Paris.\nThe capital of Germany is"` | **Works** — native `LagunaForCausalLM` |
| **5.7.0** | `trust_remote_code=True`  | 0 | 0 | `"Paris.\nThe capital of Germany is"` | **Works** — `WeightRenaming` rules in `transformers.conversion_mapping` apply class-agnostically; the in-repo file gets the conversion for free |

### The 5.6.0 + remote-code report (transformers' own LOAD REPORT, lightly trimmed)

```
LagunaForCausalLM LOAD REPORT from: poolside/Laguna-XS.2
Key                                                          | Status
-------------------------------------------------------------+-------------
model.layers.{1...39}.mlp.experts.{0...255}.gate_proj.weight | UNEXPECTED
model.layers.{1...39}.mlp.experts.{0...255}.up_proj.weight   | UNEXPECTED
model.layers.{1...39}.mlp.experts.{0...255}.down_proj.weight | UNEXPECTED
model.layers.{1...39}.mlp.shared_expert.gate_proj.weight     | UNEXPECTED
model.layers.{1...39}.mlp.shared_expert.up_proj.weight       | UNEXPECTED
model.layers.{1...39}.mlp.shared_expert.down_proj.weight     | UNEXPECTED
model.layers.{1...39}.mlp.experts.e_score_correction_bias    | UNEXPECTED
model.layers.{1...39}.mlp.experts.gate_up_proj               | MISSING
model.layers.{1...39}.mlp.experts.down_proj                  | MISSING
model.layers.{1...39}.mlp.gate.e_score_correction_bias       | MISSING
model.layers.{1...39}.mlp.shared_experts.gate_proj.weight    | MISSING
model.layers.{1...39}.mlp.shared_experts.up_proj.weight      | MISSING
model.layers.{1...39}.mlp.shared_experts.down_proj.weight    | MISSING
```

That report aligns 1:1 with the three structural mismatches above.

## Where the fix actually lives in transformers `5.7.0`

```python
# src/transformers/conversion_mapping.py — added in PR #45673
WeightRenaming("mlp.experts.e_score_correction_bias", "mlp.gate.e_score_correction_bias"),
WeightRenaming("mlp.shared_expert.",                 "mlp.shared_experts."),
```

Plus the split→packed expert stacking inherited via `LagunaExperts(Qwen3MoeExperts)` in `src/transformers/models/laguna/modular_laguna.py`. These run inside the global `from_pretrained` loader keyed off `config.model_type == "laguna"`, and they apply regardless of whether the model class came from the native module or from the in-repo `modeling_laguna.py` via `trust_remote_code=True`. That is why the in-repo shim "works" on 5.7.0 even though it has zero conversion hooks of its own (verified by grep — no `_load_state_dict_pre_hook`, no `_keys_to_ignore_on_load_missing`, no `_load_pretrained_model` override).

## Why this affects downstream inference engines

[SGLang](https://github.com/sgl-project/sglang) pins `transformers==5.6.0` in `python/pyproject.toml`. Without intervention, SGLang would have two equally bad options for serving Laguna-XS.2:

1. Refuse to load: AutoConfig with `trust_remote_code=False` fails (no native model_type).
2. Use `trust_remote_code=True`: load completes silently with 234 random-init'd MoE parameters and 30,108 discarded checkpoint tensors. The model serves "fluent" but architecturally wrong outputs. (We hit this in CI — the unit test process appeared to hang for ~15 min in `nn.init.normal_`.)

Our PR ([sgl-project/sglang#24204](https://github.com/sgl-project/sglang/pull/24204)) works around this by registering a local `LagunaConfig` shim in SGLang's `_CONFIG_REGISTRY` so AutoConfig succeeds without `trust_remote_code`, and shipping a native `LagunaForCausalLM` that loads the split-per-expert checkpoint directly into SGLang's `FusedMoE`.

[vLLM PR #41129](https://github.com/vllm-project/vllm/pull/41129) takes the same approach (independently arrived at):

- `vllm/transformers_utils/configs/laguna.py` (+120 LoC) — vLLM's own `LagunaConfig(PretrainedConfig)` with `model_type="laguna"` and v4↔v5 `rope_parameters` translation.
- `vllm/transformers_utils/config.py` (+28 LoC) — registers `LagunaConfig` in vLLM's config registry so `AutoConfig.from_pretrained(...)` works without `trust_remote_code`.
- `vllm/model_executor/models/laguna.py` (+886 LoC) — vLLM's own `LagunaForCausalLM` declaring `shared_expert` (singular, matches checkpoint), `e_score_correction_bias` registered under `experts` (matches checkpoint), and a custom `load_weights()` + `get_expert_mapping()` that handles the split-per-expert layout natively.

vLLM's `requirements/common.txt` floor is `transformers >= 4.56.0` — they do **not** require 5.7.0. They make this work because they never call `transformers.AutoModelForCausalLM.from_pretrained` for weight loading; they iterate the safetensors stream themselves and route each tensor through their own model module, which is already named to match the checkpoint. The transformers `WeightRenaming` rules added in 5.7.0 are simply irrelevant to either project, because neither uses transformers' loader.

The implication is that the local `LagunaConfig` shim isn't a transformers-version workaround — it is a permanent design choice for inference engines with their own MoE wiring (FusedMoE, TP-sharded experts), regardless of the transformers floor they support. The transformers fix in 5.7.0 helps `transformers` users; it doesn't change anything for vLLM, SGLang, or similar serving stacks.

Other inference engines that go through `transformers.AutoModelForCausalLM.from_pretrained` (or thin wrappers around it) and are pinned to `<= 5.6.0` will silently produce architecturally broken weights on this checkpoint — which is the user-facing harm worth fixing in the model card.

## Suggested resolutions, in increasing scope

1. **Model card wording (smallest).** Replace
   > Laguna XS.2 is supported in Transformers v5.7.0 and later

   with something like
   > Laguna XS.2 **requires** Transformers v5.7.0 or later. The `modeling_laguna.py` file in this repository is provided for reference only — it relies on weight-rename rules added to `transformers.conversion_mapping` in v5.7.0 and **cannot load this checkpoint correctly under older versions** (load completes but ~13 B parameters are silently random-initialized).

2. **Make `modeling_laguna.py` self-contained.** Add a `_load_state_dict_pre_hook` (or override `_load_pretrained_model`) on `LagunaPreTrainedModel` that does the same renames the transformers loader does — `mlp.shared_expert.` → `mlp.shared_experts.`, `mlp.experts.e_score_correction_bias` → `mlp.gate.e_score_correction_bias`, and stack `mlp.experts.{i}.{gate,up,down}_proj.weight` into the packed `gate_up_proj` / `down_proj` parameters. This makes `trust_remote_code=True` actually work on older transformers, which is the entire point of having an in-repo modeling file.

3. **Delete the in-repo `modeling_laguna.py` (largest).** It has been superseded by native code in transformers `5.7.0` and is dead-code-with-trap-door for any user on an older version. Drop it and rely on the version requirement enforced via the model card and a hard error in `configuration_laguna.py` if `transformers.__version__ < (5, 7, 0)`.

We would prefer (2): it is the smallest fix that restores the contract `trust_remote_code=True` is supposed to provide (a self-contained drop-in modeling module).

## Reproduction artifacts

A self-contained reproducer that exercises the structural diff using only `transformers` + `huggingface_hub` (no SGLang, no GPU, runs in seconds) is published at:

- [`scripts/playground/laguna_hf_bug_repro.py` in `sgl-project/sglang`](https://github.com/sgl-project/sglang/blob/model/laguna-xs2/scripts/playground/laguna_hf_bug_repro.py)

The full empirical matrix above was reproduced by running the same script with the project's `transformers==5.6.0` and again with `transformers==5.7.0` in an isolated venv. We are happy to share the full transcripts on request.
