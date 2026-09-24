"""torch.compile / torch.jit.script parity tests for srm."""

import torch
from dfwb_torch_srm.modules import Srm


def test_module_is_scriptable_placeholder() -> None:
    """Placeholder so a freshly rendered package has a passing test.

    Replace with real torch.compile / torch.jit.script parity checks for
    srm once forward() does something non-trivial.
    """
    module = Srm()
    assert isinstance(module, torch.nn.Module)
