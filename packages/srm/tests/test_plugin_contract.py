"""Contract test for dfwb_torch_srm's dfwb.plugins registration.

Uses a recording mock instead of the real framework, so this test runs
without dfwb installed (dfwb_plugin.py itself must not import torch or dfwb
at module level -- 03-contracts.md C1, dfwb-torch.md "Plugin hook").

Also checks, dynamically, that the whole import path the framework's plugin
discovery takes (`import dfwb_torch_srm.dfwb_plugin`, which Python resolves
by first importing the `dfwb_torch_srm` package itself) never needs torch,
and that the rendered conformance script and the quickstart example both
work end to end.
"""

import importlib
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent

_SUMMARY = "SRM / high-pass residual filter bank (Fridrich & Kodovský 2012)"

# Every registry on the C1 `PluginAPI` dataclass (03-contracts.md), so the
# mock supports `.add()` wherever a plugin might call it.
_REGISTRIES = (
    "layers",
    "transforms",
    "backbones",
    "temporal_pools",
    "heads",
    "losses",
    "metrics",
    "eval_suites",
    "face_backends",
    "inventory_builders",
    "protocol_packs",
    "detectors",
    "detector_sources",
    "callbacks",
)


@dataclass
class _Registry:
    added: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def add(self, key: str, **kwargs: object) -> None:
        self.added.append((key, kwargs))


class _RecordingApi:
    """Records `.add()` calls on every C1 registry."""

    version = (1, 0)
    layers: _Registry
    transforms: _Registry
    backbones: _Registry
    temporal_pools: _Registry
    heads: _Registry
    losses: _Registry
    metrics: _Registry
    eval_suites: _Registry
    face_backends: _Registry
    inventory_builders: _Registry
    protocol_packs: _Registry
    detectors: _Registry
    detector_sources: _Registry
    callbacks: _Registry

    def __init__(self) -> None:
        for name in _REGISTRIES:
            setattr(self, name, _Registry())


def test_register_adds_exactly_one_srm_layer() -> None:
    from dfwb_torch_srm.dfwb_plugin import register

    api = _RecordingApi()
    register(api)

    assert api.layers.added == [
        (
            "srm",
            {
                "target": "dfwb_torch_srm.modules:SRMConv2d",
                "summary": _SUMMARY,
                "requires": ("torch",),
            },
        )
    ]
    for name in _REGISTRIES:
        if name != "layers":
            assert getattr(api, name).added == [], f"unexpected call on {name!r}"


def test_register_target_is_importable() -> None:
    from dfwb_torch_srm.dfwb_plugin import register

    api = _RecordingApi()
    register(api)

    _key, kwargs = api.layers.added[0]
    module_path, _, attr = str(kwargs["target"]).partition(":")
    target_module = importlib.import_module(module_path)
    assert hasattr(target_module, attr)


# --- blocked-torch dynamic import -------------------------------------------

_BLOCKED_TORCH_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys

    class _BlockTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError(f"blocked: {name}")
            return None

    sys.meta_path.insert(0, _BlockTorch())

    import dfwb_torch_srm.dfwb_plugin as plugin

    class _Registry:
        def __init__(self):
            self.added = []

        def add(self, key, **kwargs):
            self.added.append((key, kwargs))

    class _Api:
        def __init__(self):
            self.layers = _Registry()

    api = _Api()
    plugin.register(api)
    assert api.layers.added[0][0] == "srm"
    print("OK")
    """
)


def test_dfwb_plugin_imports_and_registers_with_torch_blocked() -> None:
    """The framework loads every installed plugin's `dfwb_plugin` module to
    read its metadata before deciding whether to import the (possibly heavy)
    target class -- `target` is a "lazy import path" (C1). A meta-path
    finder that raises on any `torch` import simulates torch not being
    installed; both the import and `register()` must still succeed.
    """
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED_TORCH_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# --- conformance and the quickstart example ---------------------------------


def test_check_package_passes() -> None:
    script = _REPO_ROOT / "scripts" / "check_package.py"
    result = subprocess.run(
        [sys.executable, str(script), str(_PACKAGE_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_quickstart_example_runs_and_prints_the_expected_shape() -> None:
    script = _PACKAGE_DIR / "examples" / "quickstart.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=_PACKAGE_DIR,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "input (1, 3, 64, 64) -> output (1, 30, 64, 64)" in result.stdout
