"""Shape contract tests for srm.

Cover batch=1, an odd/non-square image, a 1x1 image, an empty batch and any
channel-count variants the op supports.
"""

import torch
from dfwb_torch_srm.modules import Srm


def test_output_shape_matches_input() -> None:
    """Placeholder so a freshly rendered package has a passing test.

    Replace with the real shape matrix for srm.
    """
    module = Srm()
    x = torch.zeros(1, 3, 5, 7)
    assert module(x).shape == x.shape
