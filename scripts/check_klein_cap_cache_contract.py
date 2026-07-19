#!/usr/bin/env python3
"""Header-only Klein sample cap-cache contract guard.

Klein sampling currently relies on precomputed Qwen3 caption embeddings stored
by `io/cap_cache.mojo`. This checker validates the sample JSON and cap-cache
headers without creating a CUDA context or loading tensor bodies.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "serenitymojo/configs/klein9b.json"
CAP_MAGIC = 0x4B4C4E4341505631
STD_BF16_TAG = 8
BF16_BYTE_SIZE = 2
KLEIN9B_CAP_SHAPES = ((1, 512, 12288), (512, 12288))
KLEIN4B_CAP_SHAPES = ((1, 512, 7680), (512, 7680))


@dataclass(frozen=True)
class CapHeader:
    path: Path
    exists: bool
    magic: int | None = None
    dtype_tag: int | None = None
    shape: tuple[int, ...] = ()
    file_size: int = 0
    expected_size: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.exists
            and self.magic == CAP_MAGIC
            and self.dtype_tag == STD_BF16_TAG
            and self.shape in KLEIN9B_CAP_SHAPES
            and self.file_size == self.expected_size
            and not self.error
        )


def fail(message: str) -> None:
    raise SystemExit(f"[klein-cap-cache] FAIL {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail(f"JSON root must be object: {path}")
    return data


def resolve_validation_prompts(config_path: Path) -> Path:
    cfg = load_json(config_path)
    path = cfg.get("validation_prompts_file")
    if not isinstance(path, str) or not path:
        fail(f"{config_path} missing validation_prompts_file")
    return Path(path)


def _numel(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


def read_cap_header(path: Path) -> CapHeader:
    if not path.exists():
        return CapHeader(path=path, exists=False, error="missing")
    try:
        file_size = path.stat().st_size
        if file_size < 24:
            return CapHeader(path=path, exists=True, file_size=file_size, error="too small")
        with path.open("rb") as fh:
            fixed = fh.read(24)
            if len(fixed) != 24:
                return CapHeader(path=path, exists=True, file_size=file_size, error="short fixed header read")
            magic, dtype_tag, rank = struct.unpack("<qqq", fixed)
        if rank <= 0 or rank > 8:
            return CapHeader(
                path=path,
                exists=True,
                magic=magic,
                dtype_tag=dtype_tag,
                file_size=file_size,
                error=f"implausible rank {rank}",
            )
        header_size = 24 + 8 * rank
        if file_size < header_size:
            return CapHeader(
                path=path,
                exists=True,
                magic=magic,
                dtype_tag=dtype_tag,
                file_size=file_size,
                error="truncated header",
            )
        with path.open("rb") as fh:
            fh.seek(24)
            dims = fh.read(8 * rank)
        if len(dims) != 8 * rank:
            return CapHeader(
                path=path,
                exists=True,
                magic=magic,
                dtype_tag=dtype_tag,
                file_size=file_size,
                error="short dims read",
            )
        shape = struct.unpack("<" + "q" * rank, dims)
        if any(dim <= 0 for dim in shape):
            return CapHeader(
                path=path,
                exists=True,
                magic=magic,
                dtype_tag=dtype_tag,
                shape=tuple(int(dim) for dim in shape),
                file_size=file_size,
                error="nonpositive dim",
            )
        expected_size = header_size + _numel(tuple(int(dim) for dim in shape)) * BF16_BYTE_SIZE
        return CapHeader(
            path=path,
            exists=True,
            magic=magic,
            dtype_tag=dtype_tag,
            shape=tuple(int(dim) for dim in shape),
            file_size=file_size,
            expected_size=expected_size,
        )
    except Exception as exc:  # noqa: BLE001 - surface local artifact damage.
        return CapHeader(path=path, exists=True, error=str(exc))


def prompt_caps(sample_path: Path) -> list[tuple[str, Path, Path]]:
    data = load_json(sample_path)
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        fail(f"{sample_path} has no prompts")
    out: list[tuple[str, Path, Path]] = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            fail(f"{sample_path} prompt {index} is not object")
        label = str(prompt.get("id") or prompt.get("label") or f"prompt_{index}")
        caps = prompt.get("caps")
        if not isinstance(caps, dict):
            fail(f"{sample_path} prompt {label} missing caps object")
        pos = caps.get("positive")
        neg = caps.get("negative")
        if not isinstance(pos, str) or not isinstance(neg, str) or not pos or not neg:
            fail(f"{sample_path} prompt {label} missing caps positive/negative paths")
        out.append((label, Path(pos), Path(neg)))
    return out


def print_header(label: str, role: str, header: CapHeader) -> None:
    status = "PASS" if header.ok else "WARN"
    print(f"[klein-cap-cache] {status} {label}:{role} {header.path}")
    if not header.exists:
        print("  missing")
        return
    print(
        "  "
        f"magic={hex(header.magic or 0)} dtype_tag={header.dtype_tag} "
        f"shape={list(header.shape)} bytes={header.file_size} expected={header.expected_size}"
    )
    if header.error:
        print(f"  error={header.error}")
    elif header.magic != CAP_MAGIC:
        print("  error=bad magic; expected KLNCAPV1")
    elif header.dtype_tag != STD_BF16_TAG:
        print("  error=expected BF16 dtype tag 8")
    elif header.shape not in KLEIN9B_CAP_SHAPES:
        print(
            "  error=expected Klein 9B cap shape "
            "[1,512,12288] or [512,12288]; 4B shapes are recognized but not "
            "accepted by the current 9B sampler CLI"
        )
    elif header.file_size != header.expected_size:
        print("  error=file size does not match header shape/dtype")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 unless all configured Klein 9B cap-cache files exist and have BF16 512x12288 headers",
    )
    args = parser.parse_args()

    sample_path = args.samples if args.samples is not None else resolve_validation_prompts(args.config)
    prompts = prompt_caps(sample_path)
    print(f"[klein-cap-cache] sample_file={sample_path} prompts={len(prompts)}")
    print("[klein-cap-cache] scope=header-only; no CUDA context, no tensor body load")

    blockers: list[str] = []
    for label, pos_path, neg_path in prompts:
        for role, path in (("positive", pos_path), ("negative", neg_path)):
            header = read_cap_header(path)
            print_header(label, role, header)
            if not header.ok:
                blockers.append(f"{label}:{role}:{path}")
            if header.exists and header.shape in KLEIN4B_CAP_SHAPES:
                blockers.append(f"{label}:{role}:{path}: 4B cap shape not accepted by current 9B CLI")

    if blockers:
        print("[klein-cap-cache] status=BLOCKED")
        print("[klein-cap-cache] blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
        if args.strict:
            return 2
    else:
        print("[klein-cap-cache] status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
