"""30-second quickstart for dfwb-torch-srm.

Run: python examples/quickstart.py
"""

import torch
from dfwb_torch_srm import SRMConv2d


def main() -> None:
    """Build an SRM layer over the 30-kernel bank and run one image through it."""
    layer = SRMConv2d(in_channels=3, bank="srm30", mode="gray")
    x = torch.rand(1, 3, 64, 64)
    y = layer(x)
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}")


if __name__ == "__main__":
    main()
