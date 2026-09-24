"""Device/dtype tests for srm: cpu always; cuda/mps if available; every float dtype."""

import torch
from dfwb_torch_srm.modules import Srm


def test_dtype_follows_input() -> None:
    """Placeholder so a freshly rendered package has a passing test.

    Replace with the real device/dtype matrix for srm.
    """
    module = Srm()
    x = torch.zeros(2, dtype=torch.float64)
    assert module(x).dtype == torch.float64
