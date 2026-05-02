"""
Reproducer: poolside/Laguna-XS.2 + HF reference modeling_laguna.py mismatch.

Run with no SGLang, no GPU. Just transformers + huggingface_hub.

What this shows
---------------
The HF reference module (`modeling_laguna.py` in the model repo) declares MoE
parameters that do not match the names actually shipped in the safetensors
checkpoint. As a result, `from_pretrained(..., trust_remote_code=True)` marks
the routed-expert weights, the shared-expert weights, and the router-bias as
MISSING, and `_init_weights` then randomly re-initializes ~13 B parameters via
`nn.init.normal_`. The loaded model produces garbage output and (on slow
inits) appears to hang during initialization.

Three structural mismatches, all visible from the safetensors index without
loading any weights:

  CHECKPOINT KEY                                 |  HF MODULE PARAMETER
  -----------------------------------------------+----------------------------
  mlp.experts.{i}.gate_proj.weight  (256 split)  |  mlp.experts.gate_up_proj
  mlp.experts.{i}.up_proj.weight    (256 split)  |    (packed into gate_up)
  mlp.experts.{i}.down_proj.weight  (256 split)  |  mlp.experts.down_proj
  mlp.shared_expert.*               (SINGULAR)   |  mlp.shared_experts.*  (PLURAL)
  mlp.experts.e_score_correction_bias            |  mlp.gate.e_score_correction_bias
"""

import json
import re
from collections import Counter

from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

REPO = "poolside/Laguna-XS.2"


def checkpoint_keys() -> set[str]:
    path = hf_hub_download(REPO, "model.safetensors.index.json")
    return set(json.load(open(path))["weight_map"].keys())


def model_keys() -> set[str]:
    """Build the HF reference model from config, using meta device so no
    weights are allocated. Just want the parameter graph to compare names."""
    import torch

    config = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return set(name for name, _ in model.named_parameters())


def fingerprint(keys: set[str]) -> set[str]:
    """Collapse expert/layer indices so the diff is structural, not per-expert."""
    out = set()
    for k in keys:
        k = re.sub(r"layers\.\d+\.", "layers.{L}.", k)
        k = re.sub(r"experts\.\d+\.", "experts.{i}.", k)
        out.add(k)
    return out


def main() -> None:
    ckpt = checkpoint_keys()
    model = model_keys()

    print(f"checkpoint keys: {len(ckpt):,}")
    print(f"model keys     : {len(model):,}")
    print()

    ckpt_fp = fingerprint(ckpt)
    model_fp = fingerprint(model)

    missing = sorted(model_fp - ckpt_fp)
    unexpected = sorted(ckpt_fp - model_fp)

    print(f"--- in HF model graph but NOT in checkpoint (will be RANDOM-INIT) ---")
    for k in missing:
        print(" ", k)
    print()
    print(f"--- in checkpoint but NOT in HF model graph (will be DISCARDED) ---")
    for k in unexpected:
        print(" ", k)
    print()

    n_missing_params = sum(1 for k in model if fingerprint({k}).pop() in missing)
    n_unexpected_files = sum(1 for k in ckpt if fingerprint({k}).pop() in unexpected)
    print(
        f"=> {n_missing_params:,} parameter tensors in the model graph will be "
        f"re-initialized, and {n_unexpected_files:,} tensors from the checkpoint "
        "will be silently discarded."
    )


if __name__ == "__main__":
    main()
