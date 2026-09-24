"""nn.Module wrappers for dfwb_torch_srm."""

import torch
from torch import nn

from dfwb_torch_srm.functional import identity


class Srm(nn.Module):
    """Placeholder module for srm.

    Replace this with the real module(s) for: SRM and high-pass residual filters. Device and dtype
    follow the input; this module holds no learnable state.

    Example:
        >>> import torch
        >>> m = Srm()
        >>> m(torch.arange(3)).tolist()
        [0, 1, 2]
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the placeholder op.

        Args:
            x: Input tensor of any shape.

        Returns:
            The same tensor, unchanged.
        """
        return identity(x)
