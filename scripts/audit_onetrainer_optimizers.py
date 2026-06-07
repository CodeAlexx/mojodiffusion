#!/usr/bin/env python3
"""Audit local OneTrainer preset optimizer identifiers.

This is a report-first guard for the Mojo optimizer dispatch map. It reads only
local JSON presets/configs and fails if a new explicit optimizer appears outside
the currently mapped target set.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOTS = (
    Path("/home/alex/OneTrainer/training_presets"),
    Path("/home/alex/OneTrainer/configs"),
    Path("/home/alex/training_presets"),
)
SUPPORTED_EXPLICIT = {"ADAMW", "ADAFACTOR"}
DEFAULT_NAME = "ADAMW"


def _optimizer_name(data: object) -> str:
    if not isinstance(data, dict):
        return "<invalid-root>"
    optimizer = data.get("optimizer", "<missing>")
    if optimizer is None:
        return "<missing>"
    if isinstance(optimizer, dict):
        value = optimizer.get("optimizer", "<missing>")
        return "<missing>" if value is None else str(value)
    return str(optimizer)


def main() -> int:
    by_name: dict[str, list[Path]] = defaultdict(list)
    parse_errors: list[tuple[Path, str]] = []

    for root in ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - report all local parse issues.
                parse_errors.append((path, str(exc)))
                continue
            by_name[_optimizer_name(data)].append(path)

    print("Local OneTrainer optimizer audit")
    print(f"default/missing resolves to: {DEFAULT_NAME}")
    for name in sorted(by_name):
        resolved = DEFAULT_NAME if name == "<missing>" else name
        print(f"{name:>12} -> {resolved:<9} files={len(set(by_name[name]))}")
        for path in sorted(set(by_name[name]))[:8]:
            print(f"  {path}")
        extra = len(set(by_name[name])) - 8
        if extra > 0:
            print(f"  ... {extra} more")

    if parse_errors:
        print("Parse errors:")
        for path, message in parse_errors:
            print(f"  {path}: {message}")
        return 1

    unsupported = sorted(
        name
        for name in by_name
        if name != "<missing>" and name not in SUPPORTED_EXPLICIT
    )
    if unsupported:
        print("Unsupported explicit optimizer identifiers:")
        for name in unsupported:
            print(f"  {name}")
        return 1

    print("PASS: explicit target optimizers are mapped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
