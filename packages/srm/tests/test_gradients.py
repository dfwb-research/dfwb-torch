"""Gradient tests for srm: gradcheck (float64) where differentiable."""

import torch
from dfwb_torch_srm.modules import Srm


def test_module_is_an_nn_module() -> None:
    """Placeholder so a freshly rendered package has a passing test.

    Replace with real `torch.autograd.gradcheck` coverage for srm.
    """
    module = Srm()
    assert isinstance(module, torch.nn.Module)
