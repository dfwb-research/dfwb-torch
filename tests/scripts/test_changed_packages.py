"""Tests for scripts/changed_packages.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGED_PACKAGES = REPO_ROOT / "scripts" / "changed_packages.py"

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env=_GIT_ENV,
    )


def _init_repo(root: Path) -> str:
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["packages/*"]\n')
    (root / "uv.lock").write_text("# lock\n")
    (root / "template").mkdir()
    (root / "template" / "placeholder.tmpl").write_text("placeholder\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "placeholder.py").write_text("# placeholder\n")
    for pkg in ("pkg-a", "pkg-b"):
        pkg_dir = root / "packages" / pkg
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "file.txt").write_text(f"{pkg}\n")

    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    result = _git(root, "rev-parse", "HEAD")
    return result.stdout.strip()


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHANGED_PACKAGES), "--root", str(root), *args],
        capture_output=True,
        text=True,
    )


def test_change_to_one_package_reports_only_that_package(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    base = _init_repo(root)

    (root / "packages" / "pkg-a" / "file.txt").write_text("changed\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "touch pkg-a")

    result = _run(root, "--base", base)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["pkg-a"]


def test_change_to_template_reports_all_packages(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    base = _init_repo(root)

    (root / "template" / "placeholder.tmpl").write_text("changed\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "touch template")

    result = _run(root, "--base", base)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["pkg-a", "pkg-b"]


def test_change_to_scripts_reports_all_packages(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    base = _init_repo(root)

    (root / "scripts" / "placeholder.py").write_text("changed\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "touch scripts")

    result = _run(root, "--base", base)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["pkg-a", "pkg-b"]


def test_change_to_root_pyproject_reports_all_packages(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    base = _init_repo(root)

    (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["packages/*"]\nx=1\n')
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "touch root pyproject")

    result = _run(root, "--base", base)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["pkg-a", "pkg-b"]


def test_change_to_uv_lock_reports_all_packages(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    base = _init_repo(root)

    (root / "uv.lock").write_text("# lock\nx=1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "touch uv.lock")

    result = _run(root, "--base", base)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["pkg-a", "pkg-b"]


def test_no_changes_reports_empty_list(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    base = _init_repo(root)

    (root / "README.md").write_text("unrelated\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "unrelated change")

    result = _run(root, "--base", base)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_all_flag_reports_every_package_regardless_of_diff(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    base = _init_repo(root)

    result = _run(root, "--base", base, "--all")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["pkg-a", "pkg-b"]


def test_bad_base_ref_prints_clean_error_and_exits_2(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repo(root)

    result = _run(root, "--base", "not-a-real-ref")
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("error: cannot diff against not-a-real-ref:")
    assert "Traceback" not in result.stderr
