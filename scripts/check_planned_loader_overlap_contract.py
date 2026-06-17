#!/usr/bin/env python3
"""Static guard for planned offload loop ordering.

Higher-level model code should use the explicit-context planned-loader surface:

    loader.prefetch_with_ctx(0, ctx)
    handle = loader.await_block(i, ctx)
    loader.prefetch_next_with_ctx(i, ctx)
    ... block math ...
    loader.mark_active_block_done(ctx)

The legacy prefetch()/prefetch_next() calls are still present on the loader
classes for compatibility and low-level tests, but model ports should not use
them. With TurboPlannedLoader they defer GPU dispatch until await_block(), which
can collapse copy/compute overlap and skip the slot-reuse compute-done fence.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_ROOTS = (
    Path("serenitymojo/models"),
    Path("serenitymojo/pipeline"),
    Path("serenitymojo/sampling"),
    Path("serenitymojo/training"),
)

LEGACY_PREFETCH_RE = re.compile(r"\.\s*(prefetch|prefetch_next)\s*\(")


def iter_mojo_files(roots: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".mojo":
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.mojo")))
    return sorted(set(files))


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def find_legacy_calls(path: Path) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = strip_comment(line)
        if LEGACY_PREFETCH_RE.search(code):
            findings.append((lineno, line.rstrip()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reject legacy planned-loader prefetch order in model code."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ROOTS),
        help="Mojo files or directories to scan.",
    )
    args = parser.parse_args()

    roots = tuple(args.paths)
    failures: list[tuple[Path, int, str]] = []
    for path in iter_mojo_files(roots):
        for lineno, line in find_legacy_calls(path):
            failures.append((path, lineno, line))

    if failures:
        print("planned-loader overlap contract: FAIL")
        print("Use prefetch_with_ctx()/prefetch_next_with_ctx() in model loops.")
        for path, lineno, line in failures:
            print(f"{path}:{lineno}: {line}")
        return 1

    print(
        "planned-loader overlap contract: PASS "
        f"({len(iter_mojo_files(roots))} Mojo files scanned)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
