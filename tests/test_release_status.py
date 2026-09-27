"""What each package's docs say about its own release: its README and the root README's row.

While a package's ``__version__`` is a pre-release, its own README.md (also the PyPI project
description) says it is not on PyPI yet, and the root README's ``## Packages`` table row for it
says so too. The commit that bumps a package to a final release rewrites both; the check below
fails until it has (a final version is one with no a, b, rc or dev segment).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.version import Version

REPO = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO / "packages"

# Anything that tells a reader a package is not released yet.
_UNRELEASED = re.compile(
    r"\bpre-release\b|\bunreleased\b|\bin development\b|\bon PyPI yet\b"
    r"|\bnot (?:yet )?(?:published|released) on PyPI\b"
    r"|\bnothing is (?:yet )?(?:published|released) on PyPI\b|\bnot on PyPI\b",
    re.IGNORECASE,
)


def is_final(version: str) -> bool:
    return not Version(version).is_prerelease


def unreleased_claims(text: str) -> list[str]:
    return [match.group(0) for match in _UNRELEASED.finditer(" ".join(text.split()))]


def package_names() -> list[str]:
    return sorted(p.name for p in PACKAGES_DIR.iterdir() if p.is_dir())


def module_name(name: str) -> str:
    return "dfwb_torch_" + name.replace("-", "_")


def package_version(name: str) -> str:
    init = PACKAGES_DIR / name / "src" / module_name(name) / "__init__.py"
    text = init.read_text("utf-8")
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
    assert match, f"{init}: __version__ not found"
    return match.group(1)


def packages_table_row(root_readme_text: str, name: str) -> str:
    """The root README's ``## Packages`` table row for ``name``, as one line."""
    for line in root_readme_text.splitlines():
        if line.startswith("|") and f"[`{name}`]" in line:
            return line
    raise AssertionError(f"root README has no packages-table row for {name!r}")


def release_status_problems(version: str, package_readme: str, table_row: str) -> list[str]:
    """Problems with a package's release-status text, given its version.

    Empty while ``version`` is a pre-release (its README/table row may say so). Once it is
    final, neither text may still call the package unreleased or not on PyPI.
    """
    if not is_final(version):
        return []
    problems = []
    readme_claims = unreleased_claims(package_readme)
    if readme_claims:
        problems.append(f"README.md still says {readme_claims} but the version is {version}")
    row_claims = unreleased_claims(table_row)
    if row_claims:
        problems.append(
            f"the packages-table row still says {row_claims} but the version is {version}"
        )
    return problems


@pytest.mark.parametrize(
    ("version", "final"),
    [
        ("0.1.0", True),
        ("1.2.3.post1", True),
        ("0.1.0rc1", False),
        ("0.1.0b2", False),
        ("0.2.0a1", False),
        ("0.2.0.dev1", False),
    ],
)
def test_which_versions_are_final(version, final):
    assert is_final(version) is final


@pytest.mark.parametrize(
    "text",
    [
        "`dfwb-torch-srm` is not on PyPI yet. Install it straight from its git URL:",
        "| [`srm`](packages/srm) | ... | pre-release (`0.1.0rc1`), not yet on PyPI | ... |",
    ],
    ids=["package-readme", "packages-table-row"],
)
def test_the_pre_release_status_text_is_what_the_check_catches(text):
    assert unreleased_claims(text)


def test_release_status_problems_catches_a_stale_final_release():
    problems = release_status_problems(
        "0.1.0",
        "`dfwb-torch-srm` is not on PyPI yet. Install it straight from its git URL:",
        "| [`srm`](packages/srm) | ... | pre-release (`0.1.0rc1`), not yet on PyPI | ... |",
    )
    assert problems


def test_release_status_problems_accepts_an_updated_final_release():
    problems = release_status_problems(
        "0.1.0",
        "pip install dfwb-torch-srm",
        "| [`srm`](packages/srm) | ... | released (`0.1.0`) | ... |",
    )
    assert not problems


def test_release_status_problems_skips_a_pre_release_that_still_says_so():
    problems = release_status_problems(
        "0.1.0rc1",
        "`dfwb-torch-srm` is not on PyPI yet. Install it straight from its git URL:",
        "| [`srm`](packages/srm) | ... | pre-release (`0.1.0rc1`), not yet on PyPI | ... |",
    )
    assert not problems


def test_every_final_package_has_updated_status_text():
    root_readme = (REPO / "README.md").read_text("utf-8")
    for name in package_names():
        version = package_version(name)
        package_readme = (PACKAGES_DIR / name / "README.md").read_text("utf-8")
        row = packages_table_row(root_readme, name)
        problems = release_status_problems(version, package_readme, row)
        assert not problems, f"{name}: {problems}"
