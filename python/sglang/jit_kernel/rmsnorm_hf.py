"""JIT-compiled RMSNorm with HuggingFace `LlamaRMSNorm` semantics.

Semantics: `out = weight * cast_dtype(rsqrt(mean(x^2) + eps) * x)` — the cast
from fp32 to the activation dtype happens BEFORE the weight multiply (i.e. the
weight multiply is performed in fp16 / bf16, not fp32). This differs from
`sgl_kernel.rmsnorm`, which does the weight multiply in fp32 and casts only
at the end.

The distinction matters for downstream accuracy under weight-only quantization
(e.g. int4wo): the two rounding paths differ by ~1 ULP, which compounds across
32+ transformer layers and can flip borderline MMLU questions. The HF rounding
path is required for parity with Hugging Face's reference forward pass when
SGLang uses the ``transformers`` model backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# The kernel uses a 512-thread block with each thread handling
# `ceil(hidden_size / 512)` elements. For cleanest indexing (and because all
# common Llama-family hidden sizes satisfy this), we require the hidden
# dimension to be a positive multiple of 512. Callers should fall back to a
# native implementation for other sizes.
_BLOCK_SIZE = 512


def is_supported_rmsnorm_hf_hidden_size(hidden_size: int) -> bool:
    """Return True iff the JIT `rmsnorm_hf` kernel supports this hidden size."""
    return hidden_size >= _BLOCK_SIZE and hidden_size % _BLOCK_SIZE == 0


@cache_once
def _jit_rmsnorm_hf_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "rmsnorm_hf",
        *args,
        cuda_files=["elementwise/rmsnorm_hf.cuh"],
        cuda_wrappers=[("rmsnorm_hf", f"RMSNormHFKernel<{args}>::run")],
    )


def rmsnorm_hf(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RMSNorm with HuggingFace (cast-before-weight-multiply) semantics.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape ``(num_tokens, hidden_size)``, CUDA, fp16 or bf16.
        ``hidden_size`` must satisfy :func:`is_supported_rmsnorm_hf_hidden_size`.
    weight : torch.Tensor
        Weight tensor, shape ``(hidden_size,)``, same dtype/device as ``input``.
    eps : float
        Epsilon for numerical stability.
    out : Optional[torch.Tensor]
        Pre-allocated output tensor (same shape/dtype as ``input``). If ``None``
        a new tensor is allocated.

    Returns
    -------
    torch.Tensor
        Normalized tensor, shape ``(num_tokens, hidden_size)``.
    """
    if not input.is_cuda:
        raise RuntimeError("rmsnorm_hf: input must be a CUDA tensor")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"rmsnorm_hf: input must be fp16 or bf16, got {input.dtype}"
        )
    if input.dim() != 2:
        raise RuntimeError(
            f"rmsnorm_hf: input must be 2D (num_tokens, hidden_size), got shape {tuple(input.shape)}"
        )
    hidden_size = input.size(-1)
    if not is_supported_rmsnorm_hf_hidden_size(hidden_size):
        raise RuntimeError(
            f"rmsnorm_hf: unsupported hidden_size={hidden_size} "
            f"(must be a positive multiple of {_BLOCK_SIZE})"
        )

    if out is None:
        out = torch.empty_like(input)

    module = _jit_rmsnorm_hf_module(hidden_size, input.dtype)
    module.rmsnorm_hf(input, weight, out, eps)
    return out
