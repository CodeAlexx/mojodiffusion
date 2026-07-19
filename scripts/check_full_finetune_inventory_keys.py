#!/usr/bin/env python3
"""Check Klein and Z-Image full-finetune inventory keys against local artifacts.

This is a report-first guard for model-specific full-finetune inventories. It
does not import Mojo, create runtime contexts, load tensor payloads, or touch
accelerators. Klein is checked from safetensors header JSON only; Z-Image is
checked from the checkpoint index weight_map only.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

KLEIN_CHECKPOINT = Path(
    "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
)
KLEIN_INVENTORY = REPO / "serenitymojo/models/klein/full_finetune_inventory.mojo"

ZIMAGE_INDEX = Path(
    "/home/alex/.serenity/models/zimage_base/transformer/"
    "diffusion_pytorch_model.safetensors.index.json"
)
ZIMAGE_INVENTORY = REPO / "serenitymojo/models/zimage/full_finetune_inventory.mojo"
ZIMAGE_EXPECTED_COUNT = 521

VALID_MODELS = ("klein", "zimage")


class InventoryError(ValueError):
    pass


@dataclass(frozen=True)
class Inventory:
    model: str
    source: Path
    keys: tuple[str, ...]
    parse_note: str


@dataclass(frozen=True)
class ArtifactKeys:
    source: Path
    keys: tuple[str, ...]
    kind: str
    note: str


@dataclass
class ModelReport:
    model: str
    ok: bool
    lines: list[str]
    strict_failures: list[str]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_text_required(path: Path) -> str:
    if not path.exists():
        raise InventoryError(f"missing inventory source: {path}")
    return path.read_text(encoding="utf-8")


def strip_line_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def function_body(text: str, name: str, path: Path) -> str:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if re.match(rf"^def\s+{re.escape(name)}\b", line):
            start = idx + 1
            break
    if start is None:
        raise InventoryError(f"{rel(path)}: missing function {name}")

    body: list[str] = []
    for line in lines[start:]:
        if line and not line[0].isspace():
            break
        body.append(line)
    if not body:
        raise InventoryError(f"{rel(path)}: empty function body {name}")
    return "\n".join(body)


def comptime_int(text: str, name: str, path: Path) -> int:
    match = re.search(rf"^\s*comptime\s+{re.escape(name)}\s*=\s*(\d+)\s*$", text, re.M)
    if not match:
        raise InventoryError(f"{rel(path)}: missing integer comptime {name}")
    return int(match.group(1))


def require_in_order(body: str, needles: tuple[str, ...], label: str) -> None:
    pos = -1
    for needle in needles:
        next_pos = body.find(needle, pos + 1)
        if next_pos < 0:
            raise InventoryError(f"{label}: missing or reordered marker {needle!r}")
        pos = next_pos


def literal_appends(body: str) -> tuple[str, ...]:
    return tuple(
        re.findall(r'out\.append\(\s*String\("([^"]+)"\)\s*\)', body)
    )


def suffix_appends(body: str, base_var: str) -> tuple[str, ...]:
    return tuple(
        re.findall(
            rf'out\.append\(\s*{re.escape(base_var)}\s*\+\s*String\("([^"]+)"\)\s*\)',
            body,
        )
    )


# Direct Mojo import is not available to this Python guard. These inventory
# sources are deterministic String append/range-loop builders, so the parser
# accepts only that narrow shape and fails closed if the source is reorganized.
def build_klein_inventory() -> Inventory:
    path = KLEIN_INVENTORY
    text = strip_line_comments(read_text_required(path))
    num_double = comptime_int(text, "KLEIN_FULL_FT_NUM_DOUBLE", path)
    num_single = comptime_int(text, "KLEIN_FULL_FT_NUM_SINGLE", path)

    main = function_body(text, "klein_full_finetune_tensor_names", path)
    require_in_order(
        main,
        (
            'out.append(String("img_in.weight"))',
            "for i in range(KLEIN_FULL_FT_NUM_DOUBLE):",
            "_append_klein_double_block(out, i)",
            "for i in range(KLEIN_FULL_FT_NUM_SINGLE):",
            "_append_klein_single_block(out, i)",
        ),
        rel(path),
    )
    base_keys = literal_appends(main)

    double_block = function_body(text, "_append_klein_double_block", path)
    require_in_order(
        double_block,
        (
            'String("double_blocks.") + String(block_idx) + String(".")',
            '_append_klein_double_stream(out, p + String("img"))',
            '_append_klein_double_stream(out, p + String("txt"))',
        ),
        rel(path),
    )
    streams = tuple(
        re.findall(
            r'_append_klein_double_stream\(\s*out\s*,\s*p\s*\+\s*String\("([^"]+)"\)\s*\)',
            double_block,
        )
    )
    if not streams:
        raise InventoryError(f"{rel(path)}: no Klein double streams parsed")

    double_stream = function_body(text, "_append_klein_double_stream", path)
    double_suffixes = suffix_appends(double_stream, "prefix")
    if not double_suffixes:
        raise InventoryError(f"{rel(path)}: no Klein double stream suffixes parsed")

    single_block = function_body(text, "_append_klein_single_block", path)
    require_in_order(single_block, ('String("single_blocks.") + String(block_idx)',), rel(path))
    single_suffixes = suffix_appends(single_block, "p")
    if not single_suffixes:
        raise InventoryError(f"{rel(path)}: no Klein single block suffixes parsed")

    keys = list(base_keys)
    for block in range(num_double):
        for stream in streams:
            prefix = f"double_blocks.{block}.{stream}"
            keys.extend(prefix + suffix for suffix in double_suffixes)
    for block in range(num_single):
        prefix = f"single_blocks.{block}"
        keys.extend(prefix + suffix for suffix in single_suffixes)

    return Inventory(
        model="klein",
        source=path,
        keys=tuple(keys),
        parse_note=(
            "parsed deterministic Mojo String appends/range loops; direct Mojo "
            "import is avoided for this Python-only header guard"
        ),
    )


def build_zimage_inventory() -> Inventory:
    path = ZIMAGE_INVENTORY
    text = strip_line_comments(read_text_required(path))
    num_nr = comptime_int(text, "ZIMAGE_FULL_FT_NUM_NR", path)
    num_cr = comptime_int(text, "ZIMAGE_FULL_FT_NUM_CR", path)
    main_depth = comptime_int(text, "ZIMAGE_FULL_FT_MAIN_DEPTH", path)

    main = function_body(text, "zimage_full_finetune_tensor_names", path)
    require_in_order(
        main,
        (
            "_append_zimage_aux(out)",
            "for i in range(ZIMAGE_FULL_FT_NUM_NR):",
            '_append_zimage_block(out, String("noise_refiner.") + String(i))',
            "for i in range(ZIMAGE_FULL_FT_NUM_CR):",
            '_append_zimage_unmodulated_refiner(out, String("context_refiner.") + String(i))',
            "for i in range(ZIMAGE_FULL_FT_MAIN_DEPTH):",
            '_append_zimage_block(out, String("layers.") + String(i))',
        ),
        rel(path),
    )

    aux = function_body(text, "_append_zimage_aux", path)
    aux_keys = literal_appends(aux)
    if not aux_keys:
        raise InventoryError(f"{rel(path)}: no Z-Image aux keys parsed")

    block = function_body(text, "_append_zimage_block", path)
    block_suffixes = suffix_appends(block, "prefix")
    if not block_suffixes:
        raise InventoryError(f"{rel(path)}: no Z-Image block suffixes parsed")

    refiner = function_body(text, "_append_zimage_unmodulated_refiner", path)
    refiner_suffixes = suffix_appends(refiner, "prefix")
    if not refiner_suffixes:
        raise InventoryError(f"{rel(path)}: no Z-Image context refiner suffixes parsed")

    keys = list(aux_keys)
    for block_idx in range(num_nr):
        prefix = f"noise_refiner.{block_idx}"
        keys.extend(prefix + suffix for suffix in block_suffixes)
    for block_idx in range(num_cr):
        prefix = f"context_refiner.{block_idx}"
        keys.extend(prefix + suffix for suffix in refiner_suffixes)
    for block_idx in range(main_depth):
        prefix = f"layers.{block_idx}"
        keys.extend(prefix + suffix for suffix in block_suffixes)

    return Inventory(
        model="zimage",
        source=path,
        keys=tuple(keys),
        parse_note=(
            "parsed deterministic Mojo String appends/range loops; direct Mojo "
            "import is avoided for this Python-only index guard"
        ),
    )


def safetensors_header_keys(path: Path) -> ArtifactKeys:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise InventoryError(f"safetensors file is too small: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = handle.read(header_len)
    if len(header_bytes) != header_len:
        raise InventoryError(f"safetensors header is truncated: {path}")
    header = json.loads(header_bytes.decode("utf-8"))
    return ArtifactKeys(
        source=path,
        keys=tuple(k for k in header if k != "__metadata__"),
        kind="safetensors-header",
        note=f"read {header_len} header bytes only",
    )


def zimage_index_keys(path: Path) -> ArtifactKeys:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise InventoryError(f"missing object weight_map in {path}")
    return ArtifactKeys(
        source=path,
        keys=tuple(str(k) for k in weight_map),
        kind="safetensors-index",
        note="read index weight_map only",
    )


def duplicates(keys: tuple[str, ...]) -> list[str]:
    counts = Counter(keys)
    return sorted(key for key, count in counts.items() if count > 1)


def context_refiner_adaln(keys: tuple[str, ...]) -> list[str]:
    return sorted(
        key
        for key in keys
        if key.startswith("context_refiner.") and "adaLN_modulation" in key
    )


def sample(items: list[str], limit: int = 6) -> str:
    if not items:
        return "none"
    shown = ", ".join(items[:limit])
    if len(items) > limit:
        shown += f", ... (+{len(items) - limit})"
    return shown


def compare_inventory(model: str, inventory: Inventory, artifact: ArtifactKeys) -> ModelReport:
    inv_keys = inventory.keys
    artifact_keys = artifact.keys
    inv_set = set(inv_keys)
    artifact_set = set(artifact_keys)
    dupes = duplicates(inv_keys)
    missing_from_artifact = sorted(inv_set - artifact_set)
    missing_from_inventory = sorted(artifact_set - inv_set)

    strict_failures: list[str] = []
    if dupes:
        strict_failures.append(f"duplicate inventory keys: {sample(dupes)}")
    if missing_from_artifact:
        strict_failures.append(
            f"inventory keys missing from {artifact.kind}: {sample(missing_from_artifact)}"
        )
    if missing_from_inventory:
        strict_failures.append(
            f"{artifact.kind} keys missing from inventory: {sample(missing_from_inventory)}"
        )

    lines = [
        (
            f"[full-ft-inventory] {model}: inventory={len(inv_keys)} "
            f"unique={len(inv_set)} artifact={len(artifact_keys)} "
            f"duplicates={len(dupes)} missing_from_artifact={len(missing_from_artifact)} "
            f"missing_from_inventory={len(missing_from_inventory)}"
        ),
        f"[full-ft-inventory] {model}: inventory_source={rel(inventory.source)}",
        f"[full-ft-inventory] {model}: artifact_source={artifact.source} ({artifact.note})",
        f"[full-ft-inventory] {model}: parse={inventory.parse_note}",
    ]

    if missing_from_artifact:
        lines.append(
            f"[full-ft-inventory] {model}: missing_from_artifact_sample={sample(missing_from_artifact)}"
        )
    if missing_from_inventory:
        lines.append(
            f"[full-ft-inventory] {model}: missing_from_inventory_sample={sample(missing_from_inventory)}"
        )

    return ModelReport(
        model=model,
        ok=not strict_failures,
        lines=lines,
        strict_failures=strict_failures,
    )


def check_klein() -> ModelReport:
    try:
        inventory = build_klein_inventory()
    except Exception as exc:
        return ModelReport("klein", False, [f"[full-ft-inventory] klein: ERROR {exc}"], [str(exc)])

    try:
        artifact = safetensors_header_keys(KLEIN_CHECKPOINT)
    except FileNotFoundError:
        msg = f"missing required checkpoint: {KLEIN_CHECKPOINT}"
        return ModelReport("klein", False, [f"[full-ft-inventory] klein: WARN {msg}"], [msg])
    except Exception as exc:
        return ModelReport("klein", False, [f"[full-ft-inventory] klein: ERROR {exc}"], [str(exc)])

    return compare_inventory("klein", inventory, artifact)


def check_zimage() -> ModelReport:
    try:
        inventory = build_zimage_inventory()
    except Exception as exc:
        return ModelReport("zimage", False, [f"[full-ft-inventory] zimage: ERROR {exc}"], [str(exc)])

    try:
        artifact = zimage_index_keys(ZIMAGE_INDEX)
    except FileNotFoundError:
        msg = f"missing required index: {ZIMAGE_INDEX}"
        return ModelReport("zimage", False, [f"[full-ft-inventory] zimage: WARN {msg}"], [msg])
    except Exception as exc:
        return ModelReport("zimage", False, [f"[full-ft-inventory] zimage: ERROR {exc}"], [str(exc)])

    report = compare_inventory("zimage", inventory, artifact)

    inventory_adaln = context_refiner_adaln(inventory.keys)
    artifact_adaln = context_refiner_adaln(artifact.keys)
    report.lines.append(
        "[full-ft-inventory] zimage: "
        f"expected_inventory_count={ZIMAGE_EXPECTED_COUNT} "
        f"context_refiner_adaln_inventory={len(inventory_adaln)} "
        f"context_refiner_adaln_index={len(artifact_adaln)}"
    )

    if len(inventory.keys) != ZIMAGE_EXPECTED_COUNT:
        report.strict_failures.append(
            f"Z-Image inventory count {len(inventory.keys)} != {ZIMAGE_EXPECTED_COUNT}"
        )
    if inventory_adaln:
        report.strict_failures.append(
            "stale context_refiner adaLN_modulation inventory keys: "
            f"{sample(inventory_adaln)}"
        )
        report.lines.append(
            f"[full-ft-inventory] zimage: stale_inventory_context_refiner_adaln_sample={sample(inventory_adaln)}"
        )
    if artifact_adaln:
        report.strict_failures.append(
            "stale context_refiner adaLN_modulation index keys: "
            f"{sample(artifact_adaln)}"
        )
        report.lines.append(
            f"[full-ft-inventory] zimage: stale_index_context_refiner_adaln_sample={sample(artifact_adaln)}"
        )
    report.ok = not report.strict_failures
    return report


def parse_models(raw: str) -> tuple[str, ...]:
    models: list[str] = []
    for part in raw.split(","):
        model = part.strip().lower()
        if not model:
            continue
        if model == "all":
            for known in VALID_MODELS:
                if known not in models:
                    models.append(known)
            continue
        if model not in VALID_MODELS:
            raise argparse.ArgumentTypeError(
                f"unknown model {model!r}; valid models: {', '.join(VALID_MODELS)}"
            )
        if model not in models:
            models.append(model)
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    return tuple(models)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report full-finetune inventory key/header agreement for Klein and Z-Image."
        )
    )
    parser.add_argument(
        "--models",
        type=parse_models,
        default=parse_models("klein,zimage"),
        help="comma-separated subset: klein,zimage,all (default: klein,zimage)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "exit nonzero on missing required artifacts, duplicate inventory keys, "
            "inventory/artifact key mismatches, stale Z-Image context_refiner adaLN keys, "
            "or wrong Z-Image inventory count"
        ),
    )
    args = parser.parse_args(argv)

    checks = {
        "klein": check_klein,
        "zimage": check_zimage,
    }

    reports = [checks[model]() for model in args.models]
    for report in reports:
        for line in report.lines:
            print(line)

    failures = [
        f"{report.model}: {failure}"
        for report in reports
        for failure in report.strict_failures
    ]

    if failures:
        print("[full-ft-inventory] strict_failures:")
        for failure in failures:
            print(f"  - {failure}")
    elif args.strict:
        print(f"[full-ft-inventory] strict PASS models={','.join(args.models)}")
    else:
        print(f"[full-ft-inventory] report PASS models={','.join(args.models)}")

    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
