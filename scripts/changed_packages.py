#!/usr/bin/env python3
"""Print the packages/ directories touched between a base ref and HEAD, as JSON.

Usage: changed_packages.py --base REF [--all]

A change to template/, scripts/, the root pyproject.toml or uv.lock means
every package under packages/ is reported touched.
"""

import argparse
import json
import subprocess
from pathlib import Path

_GLOBAL_TRIGGER_FILES = {"pyproject.toml", "uv.lock"}
_GLOBAL_TRIGGER_PREFIXES = ("template/", "scripts/")


def _list_all_packages(root: Path) -> list[str]:
    packages_dir = root / "packages"
    if not packages_dir.is_dir():
        return []
    return sorted(p.name for p in packages_dir.iterdir() if p.is_dir())


def _changed_files(base: str, root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base, "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def changed_packages(base: str, root: Path, *, all_packages: bool = False) -> list[str]:
    """Return the sorted package directory names touched since ``base``.

    Args:
        base: The git ref to diff from.
        root: The repo root containing ``packages/``, ``template/`` and ``scripts/``.
        all_packages: If true, skip the diff and return every package.

    Returns:
        The sorted list of touched (and still-existing) package directory names.
    """
    all_names = _list_all_packages(root)
    if all_packages:
        return all_names

    touched: set[str] = set()
    for changed in _changed_files(base, root):
        if changed in _GLOBAL_TRIGGER_FILES or changed.startswith(_GLOBAL_TRIGGER_PREFIXES):
            return all_names
        if changed.startswith("packages/"):
            parts = changed.split("/")
            if len(parts) >= 2 and parts[1]:
                touched.add(parts[1])

    return sorted(touched.intersection(all_names))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="git ref to diff from")
    parser.add_argument("--all", action="store_true", help="ignore the diff; list every package")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repo root containing packages/ (default: this repo)",
    )
    args = parser.parse_args(argv)

    names = changed_packages(args.base, args.root, all_packages=args.all)
    print(json.dumps(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
