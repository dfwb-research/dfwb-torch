"""Tests for scripts/new_package.py."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_PACKAGE = REPO_ROOT / "scripts" / "new_package.py"
CHECK_PACKAGE = REPO_ROOT / "scripts" / "check_package.py"
TEMPLATE_DIR = REPO_ROOT / "template"

_SUMMARY = "A test filter bank"


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "packages").mkdir(parents=True)
    shutil.copytree(TEMPLATE_DIR, root / "template")
    return root


def _render(root: Path, name: str, summary: str = _SUMMARY) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(NEW_PACKAGE), name, "--summary", summary, "--root", str(root)],
        capture_output=True,
        text=True,
    )


def _expected_files(name: str) -> set[str]:
    module = name.replace("-", "_")
    src = f"src/dfwb_torch_{module}"
    return {
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "pyproject.toml",
        f"{src}/__init__.py",
        f"{src}/py.typed",
        f"{src}/functional.py",
        f"{src}/modules.py",
        f"{src}/_reference.py",
        f"{src}/dfwb_plugin.py",
        "tests/test_shapes.py",
        "tests/test_gradients.py",
        "tests/test_reference.py",
        "tests/test_device_dtype.py",
        "tests/test_compile.py",
        "tests/test_plugin_contract.py",
        "examples/quickstart.py",
    }


def test_render_creates_every_template_file(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    result = _render(root, "my-thing")
    assert result.returncode == 0, result.stderr

    pkg_dir = root / "packages" / "my-thing"
    actual = {str(p.relative_to(pkg_dir)) for p in pkg_dir.rglob("*") if p.is_file()}
    assert actual == _expected_files("my-thing")


def test_render_leaves_no_placeholders_unsubstituted(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _render(root, "my-thing")
    pkg_dir = root / "packages" / "my-thing"
    for f in pkg_dir.rglob("*"):
        if f.is_file():
            text = f.read_text()
            for placeholder in ("$name", "$module", "$Module", "$summary", "$year"):
                assert placeholder not in text, f"{f}: leftover {placeholder}"


def test_render_prints_created_files(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    result = _render(root, "my-thing")
    printed = set(result.stdout.split())
    assert "packages/my-thing/pyproject.toml" in printed
    assert "packages/my-thing/src/dfwb_torch_my_thing/modules.py" in printed


def test_rendered_pyproject_parses_and_has_entry_point(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _render(root, "my-thing")
    pyproject = root / "packages" / "my-thing" / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    assert data["project"]["name"] == "dfwb-torch-my-thing"
    entry_points = data["project"]["entry-points"]["dfwb.plugins"]
    assert entry_points["my-thing"] == "dfwb_torch_my_thing.dfwb_plugin:register"


def test_rendered_package_passes_check_package(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _render(root, "my-thing")
    pkg_dir = root / "packages" / "my-thing"
    result = subprocess.run(
        [sys.executable, str(CHECK_PACKAGE), str(pkg_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_refuses_existing_package(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    first = _render(root, "my-thing")
    assert first.returncode == 0, first.stderr

    second = _render(root, "my-thing")
    assert second.returncode != 0
    assert "already exists" in second.stderr


@pytest.mark.parametrize(
    "bad_name",
    ["My-Thing", "my_thing", "-my-thing", "my-thing-", "UPPER", "123abc", "my--thing"],
)
def test_refuses_non_kebab_name(tmp_path: Path, bad_name: str) -> None:
    root = _make_root(tmp_path)
    result = _render(root, bad_name)
    assert result.returncode != 0
    assert not (root / "packages" / bad_name).exists()


@pytest.mark.parametrize(
    "bad_summary",
    [
        'has a "quote" in it',
        "has a backslash \\ in it",
        "line one\nline two",
        "",
        "x" * 101,
        "has a `backtick` in it",
    ],
    ids=["quote", "backslash", "newline", "empty", "too-long", "backtick"],
)
def test_refuses_unsafe_summary(tmp_path: Path, bad_summary: str) -> None:
    root = _make_root(tmp_path)
    result = _render(root, "my-thing", bad_summary)
    assert result.returncode == 1
    assert "error: --summary" in result.stderr


def test_refuses_unsafe_summary_creates_no_directory(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    result = _render(root, "my-thing", 'a "bad" summary')
    assert result.returncode == 1
    assert not (root / "packages" / "my-thing").exists()
    assert list((root / "packages").iterdir()) == []
