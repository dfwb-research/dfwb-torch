#!/usr/bin/env python3
"""Check a rendered dfwb-torch package for template conformance.

Usage: check_package.py PATH...
"""

import argparse
import ast
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

_REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "pyproject.toml",
    "src/{module}/__init__.py",
    "src/{module}/py.typed",
    "src/{module}/functional.py",
    "src/{module}/modules.py",
    "src/{module}/dfwb_plugin.py",
    "tests/test_shapes.py",
    "tests/test_gradients.py",
    "tests/test_reference.py",
    "tests/test_device_dtype.py",
    "tests/test_compile.py",
    "tests/test_plugin_contract.py",
    "examples/quickstart.py",
)

_DEP_NAME_RE = re.compile(r"[<>=!~;\[\s]")


def _module_name(name: str) -> str:
    return "dfwb_torch_" + name.replace("-", "_")


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text())
    except SyntaxError:
        return None


def _check_required_files(pkg_dir: Path, module: str) -> list[str]:
    problems = []
    for template in _REQUIRED_FILES:
        rel = template.format(module=module)
        if not (pkg_dir / rel).is_file():
            problems.append(f"missing required file: {rel}")
    return problems


def _check_dependencies(dependencies: list[object], raw_text: str) -> list[str]:
    problems: list[str] = []
    lines = raw_text.splitlines()
    for dep in dependencies:
        dep_str = str(dep)
        base = _DEP_NAME_RE.split(dep_str, maxsplit=1)[0].strip()
        if base == "torch":
            continue
        if base != "numpy":
            problems.append(f"non-torch runtime dependency without justification: {dep_str!r}")
            continue
        if not any(dep_str in line and "#" in line for line in lines):
            problems.append(
                f"numpy dependency needs an inline comment explaining why it's needed: {dep_str!r}"
            )
    return problems


def _check_pyproject(pkg_dir: Path, name: str, module: str) -> list[str]:
    pyproject_path = pkg_dir / "pyproject.toml"
    if not pyproject_path.is_file():
        return ["missing required file: pyproject.toml"]

    raw = pyproject_path.read_text()
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:
        return [f"pyproject.toml does not parse: {exc}"]

    problems: list[str] = []
    project = data.get("project", {})

    expected_name = f"dfwb-torch-{name}"
    if project.get("name") != expected_name:
        problems.append(
            f"pyproject [project].name is {project.get('name')!r}, expected {expected_name!r}"
        )

    if project.get("license") != "MIT":
        problems.append(
            f"pyproject [project].license is {project.get('license')!r}, expected 'MIT'"
        )

    if project.get("requires-python") != ">=3.12":
        problems.append(
            "pyproject [project].requires-python is "
            f"{project.get('requires-python')!r}, expected '>=3.12'"
        )

    entry_points = project.get("entry-points", {}).get("dfwb.plugins", {})
    expected_target = f"{module}.dfwb_plugin:register"
    if entry_points.get(name) != expected_target:
        problems.append(
            'pyproject [project.entry-points."dfwb.plugins"] has no '
            f"{name!r} = {expected_target!r} entry (found {entry_points!r})"
        )

    problems.extend(_check_dependencies(project.get("dependencies", []), raw))

    return problems


def _check_no_dfwb_imports(pkg_dir: Path) -> list[str]:
    problems: list[str] = []
    src_dir = pkg_dir / "src"
    if not src_dir.is_dir():
        return problems
    for py_file in sorted(src_dir.rglob("*.py")):
        rel = py_file.relative_to(pkg_dir)
        tree = _parse(py_file)
        if tree is None:
            problems.append(f"{rel}: does not parse as Python")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "dfwb" or alias.name.startswith("dfwb."):
                        problems.append(f"{rel}: imports {alias.name!r} (must not depend on dfwb)")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and (node.module == "dfwb" or node.module.startswith("dfwb."))
            ):
                problems.append(f"{rel}: imports from {node.module!r} (must not depend on dfwb)")
    return problems


def _check_plugin_no_top_level_torch(pkg_dir: Path, module: str) -> list[str]:
    plugin_path = pkg_dir / "src" / module / "dfwb_plugin.py"
    if not plugin_path.is_file():
        return []
    rel = plugin_path.relative_to(pkg_dir)
    tree = _parse(plugin_path)
    if tree is None:
        return [f"{rel}: does not parse as Python"]

    problems: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    problems.append(
                        f"{rel}: top-level 'import {alias.name}' "
                        "(dfwb_plugin.py must not import torch at module level)"
                    )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "torch" or node.module.startswith("torch."))
        ):
            problems.append(
                f"{rel}: top-level 'from {node.module} import ...' "
                "(dfwb_plugin.py must not import torch at module level)"
            )
    return problems


_BLOCKED_IMPORT_SCRIPT = """\
import sys
import importlib.abc


class _Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        for blocked in ("torch", "dfwb"):
            if name == blocked or name.startswith(blocked + "."):
                raise ImportError(f"blocked: {{name}}")
        return None


sys.meta_path.insert(0, _Blocked())

import {module}.dfwb_plugin as plugin


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
print("OK")
"""


def _check_plugin_imports_without_torch(pkg_dir: Path, module: str) -> list[str]:
    """A dynamic check: importing `<module>.dfwb_plugin` -- what the framework's
    plugin discovery actually does -- must not need torch or dfwb, even
    transitively through the package's own `__init__.py`. Python always runs
    a package's `__init__.py` before importing any of its submodules, so an
    eager import there (of a torch- or dfwb-needing submodule) would defeat
    `dfwb_plugin.py`'s own promise not to need torch at module level, even
    though the static AST check above (`_check_plugin_no_top_level_torch`)
    finds nothing wrong with `dfwb_plugin.py` itself.

    Runs the import in a subprocess with a meta-path finder that raises on
    any `torch`/`dfwb` import, so the check is independent of whether torch
    or dfwb actually happen to be installed in the environment running this
    script.
    """
    plugin_path = pkg_dir / "src" / module / "dfwb_plugin.py"
    if not plugin_path.is_file():
        return []
    rel = plugin_path.relative_to(pkg_dir)

    src_dir = pkg_dir / "src"
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src_dir) if not existing else f"{src_dir}{os.pathsep}{existing}"
    # Deterministic, uncoloured tracebacks regardless of the parent
    # environment (Python >=3.13 colourises tracebacks based on these).
    env["PYTHON_COLORS"] = "0"
    env["NO_COLOR"] = "1"
    env.pop("FORCE_COLOR", None)

    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED_IMPORT_SCRIPT.format(module=module)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        lines = (result.stderr or result.stdout).strip().splitlines()
        message = lines[-1] if lines else f"exit code {result.returncode}"
        return [f"{rel}: importing dfwb_plugin pulls in torch ({message})"]
    return []


def _check_version_defined(pkg_dir: Path, module: str) -> list[str]:
    init_path = pkg_dir / "src" / module / "__init__.py"
    if not init_path.is_file():
        return []
    rel = init_path.relative_to(pkg_dir)
    tree = _parse(init_path)
    if tree is None:
        return [f"{rel}: does not parse as Python"]

    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__version__":
                return []
    return [f"{rel}: __version__ is not defined"]


def check_package(pkg_dir: Path) -> list[str]:
    """Return the list of conformance problems for the package at ``pkg_dir``."""
    name = pkg_dir.name
    module = _module_name(name)

    problems: list[str] = []
    problems.extend(_check_required_files(pkg_dir, module))
    problems.extend(_check_pyproject(pkg_dir, name, module))
    problems.extend(_check_no_dfwb_imports(pkg_dir))
    problems.extend(_check_plugin_no_top_level_torch(pkg_dir, module))
    problems.extend(_check_version_defined(pkg_dir, module))
    problems.extend(_check_plugin_imports_without_torch(pkg_dir, module))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    args = parser.parse_args(argv)

    ok = True
    for path in args.paths:
        for problem in check_package(path):
            print(f"{path}: {problem}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
