#!/usr/bin/env python3
"""Regenerate line-accurate source maps for checked-in browser JavaScript.

The production repository intentionally carries executable JavaScript without
the historical TypeScript tree. Embedding the current source keeps DevTools and
incident debugging accurate after a direct production-JS repair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def identity_mappings(line_count: int) -> str:
    if line_count <= 0:
        return ""
    # First line: generated column 0, source 0, original line 0, column 0.
    # Following lines advance the original line by one.
    return ";".join(["AAAA", *(["AACA"] * (line_count - 1))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("javascript", nargs="+", type=Path)
    args = parser.parse_args()
    for javascript in args.javascript:
        source = javascript.read_text(encoding="utf-8")
        document = {
            "version": 3,
            "file": javascript.name,
            "sourceRoot": "",
            "sources": [f"embedded/{javascript.name}"],
            "sourcesContent": [source],
            "names": [],
            "mappings": identity_mappings(source.count("\n") + 1),
        }
        destination = Path(f"{javascript}.map")
        destination.write_text(
            json.dumps(document, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        print(destination)


if __name__ == "__main__":
    main()
