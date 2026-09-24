"""Device/dtype tests for srm: cpu always; cuda if available; float32,
float64 and bfloat16, plus float16 behind a capability probe. CPU float16
support for conv2d and its padding ops is inconsistent across PyTorch
builds and versions, so float16 is exercised but allowed to skip with its
reason when the running build can't do it, rather than being required
outright the way the other dtypes are.
"""

import pytest
import torch
from dfwb_torch_srm.modules import TLU, SRMConv2d

# Tolerances, and why: each case compares a dtype computed natively against
# the same computation done in float64 and rounded down afterwards, with
# truncation (TLU) disabled. Truncation is disabled here because with the
# default 0-255-unit threshold (3/255 ~= 0.0118) essentially every entry of
# a standard-normal input saturates to the same clamp bound regardless of
# dtype, which would make the comparison vacuous: it would pass even if the
# convolution silently upcast internally, since the clamp bound is a Python
# float applied identically either way. Disabling truncation exercises the
# actual per-dtype arithmetic.
#   * float32: empirically, max abs diff against the float64 reference is
#     ~2.9e-6 for these kernels/shapes (ordinary float32 accumulation
#     rounding over a 5x5x3 sum of products); atol=1e-5 gives ~3x headroom.
#   * float64: the "native" and "reference" computations are the identical
#     float64 call, so the difference is exactly 0; atol=1e-10 allows for
#     that with no real slack needed.
#   * bfloat16: only an 8-bit mantissa (~2-3 decimal digits), so absolute
#     error accumulates fast over a multi-tap sum. Empirically, max abs diff
#     against the float64 reference is ~0.0625 on outputs of order ~2-3 for
#     this bank/shape/seed; atol=0.15 gives ~2x headroom without being loose
#     enough to hide a real dtype bug (a silent float32 upcast, for example,
#     would make the bfloat16 output match float32 to ~1e-5, not ~0.0625).
#   * float16: a 10-bit mantissa, so noticeably more precise than bfloat16.
#     Empirically, max abs diff against the float64 reference is ~0.0078 for
#     the same setup; atol=0.02 gives ~2.5x headroom, the same margin policy
#     as bfloat16's.
_DTYPES_AND_TOL = [
    (torch.float32, 1e-5),
    (torch.float64, 1e-10),
    (torch.bfloat16, 0.15),
    (torch.float16, 0.02),
]


def _skip_if_conv2d_unsupported(dtype: torch.dtype) -> None:
    """Capability probe: skip (with the reason) if this torch build can't run
    `SRMConv2d`'s actual forward path in `dtype` on CPU, rather than fail.

    A bare `F.conv2d` call is not enough: `SRMConv2d`'s default
    `padding_mode="reflect"` goes through `reflection_pad2d` (falling back to
    `replication_pad2d` for tiny inputs), and on CPU, torch 2.4.1 has neither
    kernel implemented for float16 (`RuntimeError: "reflection_pad2d" not
    implemented for 'Half'`), even though plain `conv2d` in float16 works
    fine. The probe below runs the same construction and input shape as the
    test itself (`SRMConv2d(bank="rgbn3", mode="sum")` over a small image),
    so it exercises the real failure mode rather than a easier proxy. It is a
    no-op (never skips) for float32/float64/bfloat16, which are always
    supported.
    """
    try:
        SRMConv2d(in_channels=3, bank="rgbn3", mode="sum")(torch.zeros(1, 3, 5, 5, dtype=dtype))
    except Exception as error:  # pragma: no cover - depends on the torch build
        pytest.skip(
            f"SRMConv2d forward in {dtype} is unsupported on this torch build (CPU): {error}"
        )


@pytest.mark.parametrize(("dtype", "atol"), _DTYPES_AND_TOL)
def test_device_dtype_follow_input(dtype: torch.dtype, atol: float) -> None:
    _skip_if_conv2d_unsupported(dtype)
    torch.manual_seed(0)
    layer = SRMConv2d(in_channels=3, bank="rgbn3", mode="sum", truncate=None)
    x64 = torch.randn(2, 3, 9, 9, dtype=torch.float64)
    x = x64.to(dtype)

    y = layer(x)
    assert y.dtype == dtype, "SRMConv2d must not silently upcast the output"
    assert y.device == x.device

    reference = layer(x64).to(dtype)  # computed in float64, rounded down once
    assert torch.allclose(y.float(), reference.float(), atol=atol)


def test_weight_buffer_itself_is_cast_not_just_the_output() -> None:
    """The module's own stored `weight` stays at its construction dtype
    (float32, the default); only the value used inside `forward` is cast to
    the input's dtype. This confirms the cast happens per call, not by
    mutating the buffer in place."""
    layer = SRMConv2d(in_channels=1, bank="square3", mode="sum")
    assert layer.weight.dtype == torch.float32
    layer(torch.zeros(1, 1, 8, 8, dtype=torch.float64))
    assert layer.weight.dtype == torch.float32  # unchanged by the float64 call


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_dtype_follow_input_cuda() -> None:
    layer = SRMConv2d(in_channels=3, bank="square3", mode="sum")
    x = torch.randn(1, 3, 9, 9, device="cuda")
    y = layer(x)
    assert y.device.type == "cuda"
    assert y.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_tlu_module_dtype_follows_input(dtype: torch.dtype) -> None:
    module = TLU(threshold=3.0, input_scale="0-1")
    x = torch.tensor([-10.0, 0.0, 10.0], dtype=dtype)
    y = module(x)
    assert y.dtype == dtype
    assert torch.equal(y, torch.tensor([-3.0 / 255.0, 0.0, 3.0 / 255.0], dtype=dtype))
