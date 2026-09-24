"""SRM and high-pass residual filters.

The public API (`BANKS`, `KERNELS`, `SRMConv2d`, `TLU`, `srm_conv2d`,
`srm_kernels`) is re-exported lazily, via module `__getattr__` (PEP 562),
rather than imported eagerly here. Python always runs a package's
`__init__.py` before importing any of its submodules (including
`dfwb_plugin`), so an eager `import torch` here -- pulled in transitively by
`functional`, `kernels` or `modules` -- would defeat `dfwb_plugin.py`'s own
promise not to need torch at module level: the framework's plugin discovery
loads `dfwb_torch_srm.dfwb_plugin` for every installed plugin before
deciding whether to import a target class (03-contracts.md C1: `target` is
a "lazy import path"). `torch` is only imported the first time one of these
names is actually accessed.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-checking only, never executed
    from dfwb_torch_srm.functional import srm_conv2d as srm_conv2d
    from dfwb_torch_srm.kernels import BANKS as BANKS
    from dfwb_torch_srm.kernels import KERNELS as KERNELS
    from dfwb_torch_srm.kernels import srm_kernels as srm_kernels
    from dfwb_torch_srm.modules import TLU as TLU
    from dfwb_torch_srm.modules import SRMConv2d as SRMConv2d

__version__ = "0.1.0rc1"

__all__ = [
    "BANKS",
    "KERNELS",
    "TLU",
    "SRMConv2d",
    "__version__",
    "srm_conv2d",
    "srm_kernels",
]

# name -> (submodule, attribute); the submodule is only imported on first use.
_LAZY: dict[str, tuple[str, str]] = {
    "srm_conv2d": ("dfwb_torch_srm.functional", "srm_conv2d"),
    "BANKS": ("dfwb_torch_srm.kernels", "BANKS"),
    "KERNELS": ("dfwb_torch_srm.kernels", "KERNELS"),
    "srm_kernels": ("dfwb_torch_srm.kernels", "srm_kernels"),
    "TLU": ("dfwb_torch_srm.modules", "TLU"),
    "SRMConv2d": ("dfwb_torch_srm.modules", "SRMConv2d"),
}


def __getattr__(name: str) -> Any:
    """Resolve a lazy re-export on first access, then cache it on the module.

    Example:
        >>> from dfwb_torch_srm import SRMConv2d
        >>> SRMConv2d.__name__
        'SRMConv2d'
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
