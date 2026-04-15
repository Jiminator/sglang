"""
JIT-compiled RMSNorm with HuggingFace semantics.

HF semantics: out = weight * cast_dtype(normalize_fp32(x))
Standard:     out = cast_dtype(normalize_fp32(x) * weight_fp32)

The cast-before-weight-multiply order is required for the transformers
backend to produce numerically identical outputs to HF LlamaRMSNorm.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# Reuse the same size/kernel-class selection logic as norm.py
_RMSNORM_WARP_SIZES = frozenset({64, 128, 256})
_RMSNORM_MAX_HIDDEN_SIZE = 16384
_RMSNORM_HALF_BLOCK_MIN_SIZE = 2048


def _is_supported_rmsnorm_hf_hidden_size(d: int) -> bool:
    return d in _RMSNORM_WARP_SIZES or (
        (d > 256 and d % 256 == 0 and d <= 8192)
        or (d >= 8192 and d % 512 == 0 and d <= 16384)
    )


def _rmsnorm_hf_kernel_class(hidden_size: int) -> str:
    if hidden_size in _RMSNORM_WARP_SIZES:
        return "RMSNormHFWarpKernel"
    if hidden_size >= _RMSNORM_HALF_BLOCK_MIN_SIZE:
        if hidden_size % 512 == 0:
            return "RMSNormHFHalfKernel"
    return "RMSNormHFKernel"


@cache_once
def _jit_rmsnorm_hf_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    kernel_class = f"{_rmsnorm_hf_kernel_class(hidden_size)}<{args}>"
    return load_jit(
        "rmsnorm_hf",
        *args,
        cuda_files=["elementwise/rmsnorm_hf.cuh"],
        cuda_wrappers=[("rmsnorm_hf", f"{kernel_class}::run")],
    )


def rmsnorm_hf(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    RMSNorm with HuggingFace semantics: cast normalized x to dtype BEFORE weight multiply.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape (batch_size, hidden_size). Must be fp16 or bf16.
    weight : torch.Tensor
        Weight tensor, shape (hidden_size,). Must match input dtype.
    eps : float
        Epsilon for numerical stability.
    out : Optional[torch.Tensor]
        Pre-allocated output tensor (same shape/dtype as input).

    Returns
    -------
    torch.Tensor
        Normalized tensor, shape (batch_size, hidden_size).
    """
    if not input.is_cuda:
        raise RuntimeError("rmsnorm_hf: input must be a CUDA tensor")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"rmsnorm_hf: input must be fp16 or bf16, got {input.dtype}"
        )
    hidden_size = input.size(-1)
    if not _is_supported_rmsnorm_hf_hidden_size(hidden_size):
        raise RuntimeError(
            f"rmsnorm_hf: unsupported hidden_size={hidden_size}"
        )

    if out is None:
        out = torch.empty_like(input)

    module = _jit_rmsnorm_hf_module(hidden_size, input.dtype)
    module.rmsnorm_hf(input, weight, out, eps)
    return out
