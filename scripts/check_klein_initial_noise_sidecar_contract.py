#!/usr/bin/env python3
"""Static Klein sampler initial-noise sidecar readiness report.

This guard is intentionally no-CUDA and source-only. It reads the Klein sampler
and reports whether the source exposes the markers needed for parity-only
initial-noise sidecar injection while preserving the default production BF16
randn path.

Default mode exits 0 as a report. Use --strict to exit 2 when markers are
missing.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
KLEIN_SAMPLER = REPO / "serenitymojo/sampling/klein_sampler.mojo"
MAX_SOURCE_BYTES = 512 * 1024


@dataclass(frozen=True)
class Source:
    path: Path
    text: str | None
    error: str | None = None

    @property
    def rel(self) -> str:
        try:
            return str(self.path.relative_to(REPO))
        except ValueError:
            return str(self.path)

    def has(self, marker: str) -> bool:
        return self.text is not None and marker in self.text

    def line(self, marker: str) -> int | None:
        if self.text is None:
            return None
        index = self.text.find(marker)
        if index < 0:
            return None
        return self.text.count("\n", 0, index) + 1


@dataclass(frozen=True)
class MarkerGroup:
    name: str
    markers: tuple[str, ...]


@dataclass(frozen=True)
class ContractCheck:
    label: str
    groups: tuple[MarkerGroup, ...]
    detail: str


@dataclass(frozen=True)
class CheckResult:
    check: ContractCheck
    missing_groups: tuple[str, ...]
    refs: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_groups


SIDECAR_MARKERS = (
    "initial-noise-sidecar: parity-only",
    "parity-only initial-noise",
    "parity-only initial noise",
    "parity only initial-noise",
    "parity only initial noise",
    "initial-noise sidecar",
    "initial noise sidecar",
    "initial_noise_sidecar",
)

CHECKS: tuple[ContractCheck, ...] = (
    ContractCheck(
        label="parity-only initial-noise tensor path",
        groups=(
            MarkerGroup("parity-only sidecar marker", SIDECAR_MARKERS),
            MarkerGroup(
                "Tensor sidecar path",
                (
                    "initial_noise_sidecar: Tensor",
                    "initial_noise: Tensor",
                    "initial_noise_tensor: Tensor",
                    "initial_noise_tokens_from_sidecar",
                    "_initial_noise_from_sidecar",
                    "_initial_noise_tokens_from_sidecar",
                    "noise_sidecar: Tensor",
                    "oracle_noise: Tensor",
                    "sidecar_noise: Tensor",
                ),
            ),
        ),
        detail="Sampler exposes an explicit parity-only Tensor path for oracle initial noise.",
    ),
    ContractCheck(
        label="sidecar shape validation",
        groups=(
            MarkerGroup("sidecar marker", SIDECAR_MARKERS),
            MarkerGroup(
                "shape inspection",
                (
                    "initial_noise_sidecar.shape()",
                    "initial_noise.shape()",
                    "initial_noise_tensor.shape()",
                    "noise_sidecar.shape()",
                    "oracle_noise.shape()",
                    "sidecar_noise.shape()",
                ),
            ),
            MarkerGroup(
                "accepted shape [1,in_ch,LH,LW] or [N_IMG,in_ch]",
                (
                    "[1,in_ch,LH,LW]",
                    "[1, in_ch, LH, LW]",
                    "[N_IMG,in_ch]",
                    "[N_IMG, in_ch]",
                    "sh[0] == 1 and sh[1] == in_ch and sh[2] == LH and sh[3] == LW",
                    "sh[0] == N_IMG and sh[1] == in_ch",
                    "shape[0] != 1",
                    "shape[0] != N_IMG",
                ),
            ),
            MarkerGroup(
                "fail-loud shape error",
                (
                    "initial-noise sidecar shape",
                    "initial noise sidecar shape",
                    "raise Error",
                ),
            ),
        ),
        detail="Sidecar input is checked against [1,in_ch,LH,LW] or [N_IMG,in_ch].",
    ),
    ContractCheck(
        label="sidecar dtype preservation",
        groups=(
            MarkerGroup("sidecar marker", SIDECAR_MARKERS),
            MarkerGroup(
                "dtype inspection",
                (
                    "initial_noise_sidecar.dtype()",
                    "initial_noise.dtype()",
                    "initial_noise_tensor.dtype()",
                    "noise_sidecar.dtype()",
                    "oracle_noise.dtype()",
                    "sidecar_noise.dtype()",
                ),
            ),
            MarkerGroup(
                "preserve-dtype marker",
                (
                    "initial-noise-sidecar: preserve-dtype",
                    "preserve sidecar dtype",
                    "preserves sidecar dtype",
                    "preserves the sidecar dtype",
                    "sidecar dtype is preserved",
                    "It is never dtype-cast here",
                    "never dtype-cast",
                    "return Tensor(out_buf^, sh^, initial_noise",
                    "return Tensor(out_buf^, sh^, noise_sidecar",
                    "return Tensor(out_buf^, sh^, oracle_noise",
                    "return Tensor(out_buf^, sh^, sidecar_noise",
                ),
            ),
        ),
        detail="Sidecar path keeps the sidecar tensor dtype instead of forcing BF16/F32.",
    ),
    ContractCheck(
        label="default production BF16 randn path",
        groups=(
            MarkerGroup("initial noise helper", ("def _initial_noise_tokens",)),
            MarkerGroup(
                "NCHW randn shape",
                (
                    "nchw.append(1); nchw.append(in_ch); nchw.append(LH); nchw.append(LW)",
                    "[1,in_ch,LH,LW] ->",
                    "[1, in_ch, LH, LW]",
                ),
            ),
            MarkerGroup(
                "BF16 randn",
                (
                    "randn(nchw^, seed, STDtype.BF16, ctx)",
                    "randn(nchw^, seed, STDtype.BF16",
                ),
            ),
            MarkerGroup(
                "token reshape [N_IMG,in_ch]",
                (
                    "sh.append(N_IMG); sh.append(in_ch)",
                    "[N_IMG, in_ch]",
                    "[N_IMG,in_ch]",
                ),
            ),
            MarkerGroup(
                "default denoise call",
                ("_initial_noise_tokens[N_IMG, LH, LW](cfg.in_channels, seed, ctx)",),
            ),
        ),
        detail="Production sampling still uses Mojo BF16 randn by default.",
    ),
)


def read_source(path: Path, max_bytes: int) -> Source:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return Source(path, None, "missing source file")
    if size > max_bytes:
        return Source(path, None, f"source size {size} exceeds max {max_bytes} bytes")
    try:
        return Source(path, path.read_text(encoding="utf-8"))
    except OSError as exc:
        return Source(path, None, str(exc))


def evaluate(source: Source, check: ContractCheck) -> CheckResult:
    if source.text is None:
        return CheckResult(check, ("source readable",), ())

    missing: list[str] = []
    refs: list[str] = []
    for group in check.groups:
        for marker in group.markers:
            if source.has(marker):
                line = source.line(marker)
                if line is not None:
                    refs.append(f"{source.rel}:{line}")
                break
        else:
            missing.append(group.name)
    return CheckResult(check, tuple(missing), tuple(refs))


def print_report(source: Source, results: tuple[CheckResult, ...], strict: bool) -> None:
    print(f"Klein initial-noise sidecar contract: {source.rel}")
    if source.error is not None:
        print(f"  SOURCE missing - {source.error}")

    for result in results:
        status = "PASS" if result.ok else "MISSING"
        if result.ok:
            refs = ", ".join(result.refs[:4])
            if len(result.refs) > 4:
                refs += ", ..."
            print(f"  {status} {result.check.label} - {result.check.detail} {refs}")
        else:
            missing = ", ".join(result.missing_groups)
            print(f"  {status} {result.check.label} - missing: {missing}")

    missing_count = sum(1 for result in results if not result.ok)
    mode = "strict" if strict else "report-only"
    print(f"Result: {mode}; missing checks={missing_count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 when any readiness marker group is missing.",
    )
    args = parser.parse_args()

    source = read_source(KLEIN_SAMPLER, MAX_SOURCE_BYTES)
    results = tuple(evaluate(source, check) for check in CHECKS)
    print_report(source, results, args.strict)

    if args.strict and any(not result.ok for result in results):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
