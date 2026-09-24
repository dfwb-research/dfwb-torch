"""dfwb.plugins entry point for dfwb_torch_srm.

MUST NOT import torch or dfwb at module level: the framework's plugin
discovery loads this module to read metadata before deciding whether to
import the rest of the package.
"""

from typing import Any

DFWB_PLUGIN_API = ">=1.0,<2"


def register(api: Any) -> None:
    """Register srm's layers with the framework.

    Args:
        api: The plugin registration API (provides `.layers.add(...)`, etc).
            Config kwargs are validated by the framework against the target's
            `__init__` signature and annotations, so no `params=` is given
            here.
    """
    api.layers.add(
        "srm",
        target="dfwb_torch_srm.modules:Srm",
        summary="SRM and high-pass residual filters",
        requires=("torch",),
    )
