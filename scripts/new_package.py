#!/usr/bin/env python3
"""Render template/ into packages/<name> for a new dfwb-torch package.

Usage: new_package.py NAME --summary "..." [--root DIR]
"""

import argparse
import re
import string
import sys
from datetime import UTC, datetime
from pathlib import Path

_KEBAB_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


def _validate_name(name: str) -> None:
    if not _KEBAB_RE.match(name):
        raise ValueError(
            f"{name!r} is not a valid package name: use lower-kebab-case "
            "(lowercase letters, digits and single hyphens, starting with a letter)"
        )


def _module_name(name: str) -> str:
    return name.replace("-", "_")


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("-"))


def render_package(name: str, summary: str, root: Path, *, year: str | None = None) -> list[Path]:
    """Render ``template/`` into ``root/packages/<name>``.

    Args:
        name: The new package's lower-kebab-case name (e.g. ``"srm"``).
        summary: A one-line description, substituted for ``$summary``.
        root: The repo root containing ``template/`` and ``packages/``.
        year: The year substituted for ``$year`` (default: the current UTC year).

    Returns:
        The created files, as paths relative to ``root``, sorted.

    Raises:
        ValueError: ``name`` is not lower-kebab-case, there is no ``template/``
            directory under ``root``, or ``packages/<name>`` already exists.
    """
    _validate_name(name)

    template_dir = root / "template"
    if not template_dir.is_dir():
        raise ValueError(f"no template/ directory under {root}")

    target_dir = root / "packages" / name
    if target_dir.exists():
        raise ValueError(f"packages/{name} already exists")

    mapping = {
        "name": name,
        "module": _module_name(name),
        "Module": _class_name(name),
        "summary": summary,
        "year": year if year is not None else str(datetime.now(tz=UTC).year),
    }

    created: list[Path] = []
    for src in sorted(template_dir.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(template_dir)
        rendered_rel = Path(string.Template(str(rel)).substitute(mapping))
        if rendered_rel.suffix == ".tmpl":
            rendered_rel = rendered_rel.with_suffix("")
        dest = target_dir / rendered_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(string.Template(src.read_text()).substitute(mapping))
        created.append(dest.relative_to(root))

    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="package name, lower-kebab-case (e.g. 'srm')")
    parser.add_argument("--summary", required=True, help="one-line package summary")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repo root containing template/ and packages/ (default: this repo)",
    )
    args = parser.parse_args(argv)

    try:
        created = render_package(args.name, args.summary, args.root)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
