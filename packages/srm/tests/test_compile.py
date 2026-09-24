"""torch.compile parity tests for srm.

`SRMConv2d`'s control flow branches only on Python-level config (`mode`,
`padding`, `padding_mode`, `bool`s) and static shapes, never on tensor
values, so it is expected to be torch.compile-friendly (dfwb-torch.md's
engineering rules).
"""

import pytest
import torch
from dfwb_torch_srm.modules import SRMConv2d

try:
    import torch._dynamo as _dynamo

    _COMPILE_SUPPORTED = _dynamo.is_dynamo_supported()
    _SKIP_REASON = "" if _COMPILE_SUPPORTED else "torch._dynamo reports no support on this platform"
except ImportError as _import_error:  # pragma: no cover - defensive, not expected on torch>=2.4
    _COMPILE_SUPPORTED = False
    _SKIP_REASON = f"torch._dynamo unavailable: {_import_error}"


@pytest.mark.skipif(not _COMPILE_SUPPORTED, reason=_SKIP_REASON)
@pytest.mark.parametrize("trainable", [False, True])
def test_compile_matches_eager(trainable: bool) -> None:
    torch.manual_seed(0)
    layer = SRMConv2d(in_channels=3, bank="srm30", mode="depthwise", trainable=trainable)
    compiled = torch.compile(layer)
    x = torch.randn(2, 3, 9, 9)

    eager_out = layer(x)
    compiled_out = compiled(x)

    assert eager_out.shape == compiled_out.shape
    # Not necessarily bit-identical: inductor may fuse/reorder float ops
    # differently from eager, so this checks numerical, not exact, parity.
    assert torch.allclose(eager_out, compiled_out, atol=1e-5)
