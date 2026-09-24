"""30-second quickstart for dfwb-torch-srm.

Run: python examples/quickstart.py
"""

import torch
from dfwb_torch_srm import Srm


def main() -> None:
    """Run the quickstart example."""
    module = Srm()
    x = torch.zeros(1, 3, 8, 8)
    y = module(x)
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}")


if __name__ == "__main__":
    main()
