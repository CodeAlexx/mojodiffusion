#!/usr/bin/env python3
"""Preflight a Krea2 train cache before running a bounded real-cache smoke."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from safetensors import safe_open


KEY_RE = re.compile(r"^clean\.(\d+)$")


def fail(msg: str) -> None:
    raise SystemExit(f"[krea2-cache-contract] FAIL: {msg}")


def sample_indices(keys: list[str]) -> list[int]:
    out: list[int] = []
    for key in keys:
        match = KEY_RE.match(key)
        if match is not None:
            out.append(int(match.group(1)))
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--lh", type=int, required=True, help="expected latent height")
    parser.add_argument("--lw", type=int, required=True, help="expected latent width")
    parser.add_argument("--ltmax", type=int, required=True, help="candidate caption bucket")
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="reject known synthetic fixture metadata",
    )
    args = parser.parse_args()

    if not args.cache.exists():
        fail(f"missing cache: {args.cache}")
    if args.lh <= 0 or args.lw <= 0 or args.ltmax <= 0:
        fail("lh, lw, and ltmax must be positive")
    if args.min_samples <= 0:
        fail("min-samples must be positive")

    with safe_open(str(args.cache), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if args.require_real:
            evidence = metadata.get("evidence_level", "").lower()
            if "synthetic" in evidence or "fixture" in evidence:
                fail(f"cache metadata marks synthetic fixture: {metadata!r}")

        keys = list(handle.keys())
        indices = sample_indices(keys)
        if len(indices) < args.min_samples:
            fail(f"sample count {len(indices)} < min-samples {args.min_samples}")
        if indices != list(range(len(indices))):
            fail("clean.<i> keys must be dense from 0")

        text_lens: list[int] = []
        context_max = 0
        for idx in indices:
            clean_key = f"clean.{idx}"
            context_key = f"context.{idx}"
            text_len_key = f"text_len.{idx}"
            for key in (clean_key, context_key, text_len_key):
                if key not in keys:
                    fail(f"missing key {key}")

            clean = handle.get_slice(clean_key)
            clean_shape = list(clean.get_shape())
            clean_dtype = str(clean.get_dtype())
            expected_clean = [1, 16, args.lh, args.lw]
            if clean_shape != expected_clean:
                fail(f"{clean_key} shape {clean_shape} != {expected_clean}")
            if clean_dtype != "BF16":
                fail(f"{clean_key} dtype {clean_dtype} != BF16")

            context = handle.get_slice(context_key)
            context_shape = list(context.get_shape())
            context_dtype = str(context.get_dtype())
            if len(context_shape) != 4:
                fail(f"{context_key} must be rank 4, got {context_shape}")
            if context_shape[0] != 1 or context_shape[2:] != [12, 2560]:
                fail(f"{context_key} shape {context_shape} incompatible with Krea2 context")
            if context_dtype != "BF16":
                fail(f"{context_key} dtype {context_dtype} != BF16")
            if context_shape[1] > context_max:
                context_max = context_shape[1]

            text_len = handle.get_tensor(text_len_key).reshape(-1)
            if text_len.numel() != 1:
                fail(f"{text_len_key} must be scalar-like")
            lt = int(float(text_len[0]))
            if lt <= 0:
                fail(f"{text_len_key} must be positive, got {lt}")
            if lt != context_shape[1]:
                fail(f"{text_len_key}={lt} != {context_key} length {context_shape[1]}")
            text_lens.append(lt)

    fit = sum(1 for lt in text_lens if lt <= args.ltmax)
    max_lt = max(text_lens)
    min_lt = min(text_lens)
    if fit < args.min_samples:
        fail(
            f"only {fit}/{len(text_lens)} samples fit LTMAX={args.ltmax}; "
            f"min_lt={min_lt} max_lt={max_lt}"
        )
    if max_lt > args.ltmax:
        fail(
            f"LTMAX={args.ltmax} is too small for full cache; "
            f"min_lt={min_lt} max_lt={max_lt} fit={fit}/{len(text_lens)}"
        )

    print(
        "[krea2-cache-contract] PASS:",
        f"samples={len(text_lens)}",
        f"latent=[1,16,{args.lh},{args.lw}]",
        f"text_len_min={min_lt}",
        f"text_len_max={max_lt}",
        f"context_max={context_max}",
        f"ltmax={args.ltmax}",
        f"metadata={metadata if metadata else '{}'}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
