#!/usr/bin/env python3
"""Static guard for the LTX/LTX2 runtime dtype contract.

This is intentionally narrow. It catches the production-runtime anti-patterns
that have repeatedly broken LTX2 inference:

- loading BF16 checkpoint tensors and storing them as F32,
- using F32 view loaders in LTX2 runtime files,
- hardcoding F32 latent/noise randn,
- forcing streamed/block weights through `.to_f32(ctx)`,
- casting audio mel tensors to F32 at the model boundary.

If a production F32 boundary is truly required, add a nearby
`dtype-contract: allow-f32-boundary` comment with the exact reference reason.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

SIDECAR_FILES = (
    "serenitymojo/pipeline/ltx2_t2v_av_hq.mojo",
    "serenitymojo/pipeline/ltx2_t2v_av_mvp.mojo",
    "serenitymojo/sampling/ltx2_sampling.mojo",
    "serenitymojo/sampling/ltx2_guidance.mojo",
)

ALL_RUNTIME_FILES = SIDECAR_FILES + (
    "serenitymojo/models/dit/ltx2_connector.mojo",
    "serenitymojo/models/dit/ltx2_dit.mojo",
    "serenitymojo/models/vocoder/ltx2_vocoder.mojo",
    "serenitymojo/models/ltx2/weights.mojo",
)

ALLOW_MARKER = "dtype-contract: allow-f32-boundary"
RNG_ALLOW_MARKERS = (
    "rng-contract: mojo-native-not-pytorch-parity",
    "rng-contract: uses-pytorch-oracle-noise",
    "rng-contract: proven-pytorch-equivalent",
)


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    message: str


RULES = (
    Rule(
        "f32_view_loader",
        re.compile(r"Tensor\.from_view_as_f32\s*\("),
        "runtime safetensors view loaded as F32 storage",
    ),
    Rule(
        "bf16_to_f32_loader",
        re.compile(
            r"cast_tensor\s*\(\s*Tensor\.from_view_as_bf16\s*\([\s\S]{0,240}?STDtype\.F32",
            re.DOTALL,
        ),
        "BF16 checkpoint tensor promoted to F32 storage",
    ),
    Rule(
        "f32_randn",
        re.compile(r"randn\s*\([\s\S]{0,240}?STDtype\.F32", re.DOTALL),
        "random latent/noise generation hardcoded to F32",
    ),
    Rule(
        "block_to_f32",
        re.compile(r"\.to_f32\s*\(\s*ctx\s*\)"),
        "LTX2 block weights forced to F32 storage",
    ),
    Rule(
        "decode_audio_to_f32_mel",
        re.compile(
            r"var\s+mel\w*\s*=\s*cast_tensor\s*\([\s\S]{0,240}?decode_audio[\s\S]{0,240}?STDtype\.F32",
            re.DOTALL,
        ),
        "audio VAE mel output cast to F32 at vocoder boundary",
    ),
    Rule(
        "connector_context_to_f32",
        re.compile(
            r"cast_tensor\s*\(\s*(?:video|audio)_(?:pre|context|ctx)[\w]*\s*,\s*STDtype\.F32",
            re.DOTALL,
        ),
        "connector context promoted to F32 before LTX2 connector",
    ),
)

RNG_RULES = (
    Rule(
        "mojo_randn_contract",
        re.compile(r"\brandn\s*\("),
        "Mojo randn is not PyTorch same-seed parity unless marked as oracle/proven/not-parity",
    ),
)


def line_for(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def marker_near(
    text: str,
    offset: int,
    markers: tuple[str, ...],
    before: int = 4,
    after: int = 3,
) -> bool:
    line = line_for(text, offset)
    lines = text.splitlines()
    start = max(0, line - before)
    end = min(len(lines), line + after)
    return any(any(marker in lines[i] for marker in markers) for i in range(start, end))


def allowed_near(text: str, offset: int) -> bool:
    return marker_near(text, offset, (ALLOW_MARKER,))


def files_for_scope(scope: str) -> tuple[str, ...]:
    if scope == "sidecar":
        return SIDECAR_FILES
    if scope == "all":
        return ALL_RUNTIME_FILES
    raise ValueError(scope)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("sidecar", "all"),
        default="sidecar",
        help="Files to check. sidecar is pipeline/sampling only; all includes broader LTX2 runtime files.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    for rel in files_for_scope(args.scope):
        path = REPO / rel
        if not path.exists():
            continue
        text = path.read_text()
        for rule in RULES:
            for match in rule.pattern.finditer(text):
                if allowed_near(text, match.start()):
                    continue
                failures.append(
                    f"{rel}:{line_for(text, match.start())}: {rule.name}: {rule.message}"
                )
        for rule in RNG_RULES:
            for match in rule.pattern.finditer(text):
                if marker_near(text, match.start(), RNG_ALLOW_MARKERS, before=8, after=4):
                    continue
                failures.append(
                    f"{rel}:{line_for(text, match.start())}: {rule.name}: {rule.message}"
                )

    if failures:
        print(f"LTX2 dtype/RNG contract violations (scope={args.scope}):")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(f"LTX2 dtype/RNG contract static guard: pass (scope={args.scope})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
