"""Tests for scripts/check_package.py (T5 conformance checks)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_PACKAGE = REPO_ROOT / "scripts" / "new_package.py"
CHECK_PACKAGE = REPO_ROOT / "scripts" / "check_package.py"
TEMPLATE_DIR = REPO_ROOT / "template"

_NAME = "chk-fixture"
_MODULE = "dfwb_torch_chk_fixture"


def _baseline(tmp_path: Path) -> Path:
    """Render a fresh, fully conformant package to mutate for each test."""
    root = tmp_path / "repo"
    (root / "packages").mkdir(parents=True)
    shutil.copytree(TEMPLATE_DIR, root / "template")
    result = subprocess.run(
        [sys.executable, str(NEW_PACKAGE), _NAME, "--summary", "A fixture", "--root", str(root)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return root / "packages" / _NAME


def _check(pkg_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECK_PACKAGE), str(pkg_dir)],
        capture_output=True,
        text=True,
    )


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"fixture assumption broken: {old!r} not found in {path}"
    path.write_text(text.replace(old, new, 1))


def test_baseline_package_is_conformant(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    result = _check(pkg_dir)
    assert result.returncode == 0, result.stdout
    assert result.stdout == ""


def test_missing_reference_py_is_not_a_violation(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    (pkg_dir / "src" / _MODULE / "_reference.py").unlink()
    result = _check(pkg_dir)
    assert result.returncode == 0, result.stdout


def test_flags_missing_required_file(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    (pkg_dir / "tests" / "test_gradients.py").unlink()
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "tests/test_gradients.py" in result.stdout
    assert str(pkg_dir) in result.stdout


def test_flags_missing_py_typed(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    (pkg_dir / "src" / _MODULE / "py.typed").unlink()
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "py.typed" in result.stdout


def test_flags_wrong_package_name(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    _edit(
        pkg_dir / "pyproject.toml",
        'name = "dfwb-torch-chk-fixture"',
        'name = "dfwb-torch-wrong-name"',
    )
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "name" in result.stdout
    assert "wrong-name" in result.stdout


def test_flags_wrong_licence(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    _edit(pkg_dir / "pyproject.toml", 'license = "MIT"', 'license = "Apache-2.0"')
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "license" in result.stdout


def test_flags_wrong_requires_python(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    _edit(
        pkg_dir / "pyproject.toml",
        'requires-python = ">=3.12"',
        'requires-python = ">=3.10"',
    )
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "requires-python" in result.stdout


def test_flags_wrong_entry_point(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    _edit(
        pkg_dir / "pyproject.toml",
        'chk-fixture = "dfwb_torch_chk_fixture.dfwb_plugin:register"',
        'chk-fixture = "dfwb_torch_chk_fixture.dfwb_plugin:regsiter"',
    )
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "entry" in result.stdout.lower() or "dfwb.plugins" in result.stdout


def test_flags_non_torch_dependency_without_justification(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    _edit(
        pkg_dir / "pyproject.toml",
        'dependencies = ["torch>=2.4"]',
        'dependencies = ["torch>=2.4", "requests>=2"]',
    )
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "requests" in result.stdout


def test_flags_numpy_dependency_without_comment(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    _edit(
        pkg_dir / "pyproject.toml",
        'dependencies = ["torch>=2.4"]',
        'dependencies = ["torch>=2.4", "numpy>=1.24"]',
    )
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "numpy" in result.stdout


def test_allows_numpy_dependency_with_comment(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    _edit(
        pkg_dir / "pyproject.toml",
        'dependencies = ["torch>=2.4"]',
        "dependencies = [\n"
        '    "torch>=2.4",\n'
        '    "numpy>=1.24",  # needed for the reference implementation\n'
        "]",
    )
    result = _check(pkg_dir)
    assert result.returncode == 0, result.stdout


def test_flags_dfwb_import_in_src(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    _edit(
        pkg_dir / "src" / _MODULE / "functional.py",
        "import torch",
        "import dfwb\nimport torch",
    )
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "dfwb" in result.stdout
    assert "functional.py" in result.stdout


def test_flags_top_level_torch_import_in_plugin(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    _edit(
        pkg_dir / "src" / _MODULE / "dfwb_plugin.py",
        "from typing import Any",
        "from typing import Any\n\nimport torch",
    )
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "dfwb_plugin.py" in result.stdout
    assert "torch" in result.stdout


def test_flags_missing_dunder_version(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    _edit(
        pkg_dir / "src" / _MODULE / "__init__.py",
        '__version__ = "0.1.0"',
        "",
    )
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "__version__" in result.stdout


def test_checks_multiple_paths_and_prefixes_each_problem(tmp_path: Path) -> None:
    pkg_dir = _baseline(tmp_path)
    (pkg_dir / "tests" / "test_gradients.py").unlink()

    other_root = tmp_path / "other-repo"
    (other_root / "packages").mkdir(parents=True)
    shutil.copytree(TEMPLATE_DIR, other_root / "template")
    subprocess.run(
        [
            sys.executable,
            str(NEW_PACKAGE),
            "other-fixture",
            "--summary",
            "Another fixture",
            "--root",
            str(other_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    other_pkg = other_root / "packages" / "other-fixture"

    result = subprocess.run(
        [sys.executable, str(CHECK_PACKAGE), str(pkg_dir), str(other_pkg)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert str(pkg_dir) in result.stdout
    assert str(other_pkg) not in result.stdout


@pytest.mark.parametrize("bad_name", ["not-a-real-package"])
def test_missing_pyproject_reports_one_problem_set(tmp_path: Path, bad_name: str) -> None:
    pkg_dir = tmp_path / bad_name
    pkg_dir.mkdir()
    result = _check(pkg_dir)
    assert result.returncode == 1
    assert "pyproject.toml" in result.stdout
