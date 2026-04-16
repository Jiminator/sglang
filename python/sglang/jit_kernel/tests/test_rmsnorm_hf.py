"""Unit tests for the JIT `rmsnorm_hf` kernel.

The kernel implements HuggingFace `LlamaRMSNorm` semantics:
``out = weight * cast_dtype(rsqrt(mean(x^2) + eps) * x)``. Unlike the
standard RMSNorm (which multiplies by the weight in fp32 and casts at the
end), this variant casts to the activation dtype BEFORE the weight multiply.
We verify both correctness against a PyTorch reference and the hidden-size
support predicate.
"""

import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.rmsnorm_hf import (
    is_supported_rmsnorm_hf_hidden_size,
    rmsnorm_hf,
)
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


EPS = 1e-5
DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16]


def hf_rmsnorm_reference(
    x: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    """HF LlamaRMSNorm semantics: fp32 normalize, cast to dtype, multiply weight."""
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return w * x_normed.to(x.dtype)


BS_LIST = get_ci_test_range(
    [1, 2, 4, 7, 16, 64, 128, 512, 1024, 4096],
    [1, 16, 1024],
)
HIDDEN_SIZE_LIST = get_ci_test_range(
    [512, 1024, 2048, 3072, 4096, 8192, 16384],
    [512, 4096, 16384],
)


@pytest.mark.parametrize(
    "batch_size,hidden_size",
    list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST)),
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_hf_matches_hf_reference(
    batch_size: int, hidden_size: int, dtype: torch.dtype
) -> None:
    """The kernel output must match the HF reference up to 1-ULP reduction-order drift.

    The kernel's fp32 sum-of-squares uses a 512-thread scalar-strided reduction;
    PyTorch's ``.mean()`` uses its own reduction order. Both are equally valid
    fp32 reductions but differ on some elements by a single ULP after the final
    cast to fp16/bf16. The tolerance below (1e-2) covers 1 ULP for bf16 (which
    has coarser precision than fp16) and is the convention used in
    ``test_rmsnorm.py``.
    """
    torch.manual_seed(0)
    x = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    w = torch.randn(hidden_size, device=DEVICE, dtype=dtype)

    out = rmsnorm_hf(x, w, EPS)
    ref = hf_rmsnorm_reference(x, w, EPS)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_hf_does_not_match_fp32_weight_multiply(dtype: torch.dtype) -> None:
    """Guard against regressing to standard RMSNorm semantics.

    The whole point of this kernel is that the weight multiply happens in the
    activation dtype (fp16/bf16), NOT fp32. This check constructs a reference
    with the fp32 weight multiply (what ``sgl_kernel.rmsnorm`` does) and
    confirms that the kernel does not match it. If the semantics ever silently
    drift back to fp32-weight-multiply, this test will fail.
    """
    torch.manual_seed(0)
    hidden_size = 4096
    # Use large magnitude inputs so the two semantics diverge by more than noise.
    x = torch.randn(8, hidden_size, device=DEVICE, dtype=dtype) * 4
    w = torch.randn(hidden_size, device=DEVICE, dtype=dtype) * 4

    out = rmsnorm_hf(x, w, EPS)

    # Standard semantics: multiply weight in fp32, cast at the end.
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    standard_ref = (x_fp32 * torch.rsqrt(variance + EPS) * w.to(torch.float32)).to(dtype)

    # The kernel must differ from the standard-semantics reference on at least
    # some elements (bit-level divergence is the whole point).
    assert not torch.equal(out, standard_ref), (
        "rmsnorm_hf matched fp32-weight-multiply output; HF semantics may have regressed."
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_hf_out_param(dtype: torch.dtype) -> None:
    """The out= parameter must be used in-place and returned."""
    torch.manual_seed(0)
    x = torch.randn(8, 4096, device=DEVICE, dtype=dtype)
    w = torch.randn(4096, device=DEVICE, dtype=dtype)
    out = torch.empty_like(x)

    result = rmsnorm_hf(x, w, EPS, out=out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, hf_rmsnorm_reference(x, w, EPS), atol=0.0, rtol=0.0)


def test_rmsnorm_hf_cpu_input_rejected() -> None:
    x = torch.randn(4, 4096, dtype=torch.float16)
    w = torch.randn(4096, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="CUDA"):
        rmsnorm_hf(x, w, EPS)


def test_rmsnorm_hf_unsupported_dtype_rejected() -> None:
    x = torch.randn(4, 4096, device=DEVICE, dtype=torch.float32)
    w = torch.randn(4096, device=DEVICE, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="fp16 or bf16"):
        rmsnorm_hf(x, w, EPS)


def test_rmsnorm_hf_unsupported_hidden_size_rejected() -> None:
    # 4097 is not a multiple of 512.
    x = torch.randn(4, 4097, device=DEVICE, dtype=torch.float16)
    w = torch.randn(4097, device=DEVICE, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="unsupported hidden_size"):
        rmsnorm_hf(x, w, EPS)


@pytest.mark.parametrize(
    ("hidden_size", "expected"),
    [
        (128, False),    # too small
        (256, False),    # too small
        (511, False),    # not a multiple of 512
        (512, True),
        (1024, True),
        (3072, True),    # Llama 3.2 3B
        (4096, True),    # Llama 3.1 8B
        (8192, True),    # Llama 3 70B
        (4097, False),
    ],
)
def test_is_supported_hidden_size(hidden_size: int, expected: bool) -> None:
    assert is_supported_rmsnorm_hf_hidden_size(hidden_size) is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
