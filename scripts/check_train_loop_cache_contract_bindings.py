#!/usr/bin/env python3
"""Report train-loop cache/preflight binding readiness.

Default mode is report-only and exits 0 so the port can use this as an
inventory while known gaps are being closed. Use --strict to fail missing
bindings or unsafe F32 cache-boundary markers. Broader host-F32 carriers are
reported separately as dtype debt; they are not automatically cache-boundary
failures.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO / "serenitymojo/training"


@dataclass(frozen=True)
class LoopTarget:
    label: str
    path: Path


@dataclass(frozen=True)
class MarkerHit:
    line: int
    text: str


@dataclass(frozen=True)
class MarkerReport:
    name: str
    hits: tuple[MarkerHit, ...]


@dataclass(frozen=True)
class LoopReport:
    target: LoopTarget
    exists: bool
    train_config_import: int | None
    train_config_read: int | None
    train_config_validate: int | None
    only_cache_line: int | None
    device_context_line: int | None
    sample_helper: int | None
    sample_read: int | None
    sample_validate: int | None
    sample_decision: int | None
    preflight_import: int | None
    preflight_create: int | None
    preflight_validate: int | None
    direct_text_contract: int | None
    direct_vae_contract: int | None
    cache_f32_markers: tuple[MarkerReport, ...]
    host_f32_markers: tuple[MarkerReport, ...]

    @property
    def train_config_ok(self) -> bool:
        return (
            self.train_config_import is not None
            and self.train_config_read is not None
            and self.train_config_validate is not None
        )

    @property
    def only_cache_order_ok(self) -> bool:
        if self.device_context_line is None:
            return True
        return self.only_cache_line is not None and self.only_cache_line < self.device_context_line

    @property
    def sample_cadence_ok(self) -> bool:
        return (
            self.sample_helper is not None
            and self.sample_read is not None
            and self.sample_validate is not None
            and self.sample_decision is not None
        )

    @property
    def cache_contract_ok(self) -> bool:
        preflight_ok = self.preflight_create is not None and self.preflight_validate is not None
        direct_ok = self.direct_text_contract is not None and self.direct_vae_contract is not None
        return preflight_ok or direct_ok

    @property
    def host_f32_marker_count(self) -> int:
        return sum(len(marker.hits) for marker in self.host_f32_markers)

    @property
    def cache_f32_marker_count(self) -> int:
        return sum(len(marker.hits) for marker in self.cache_f32_markers)

    def strict_failures(self, *, strict_host_f32: bool = False) -> tuple[str, ...]:
        failures: list[str] = []
        if not self.exists:
            failures.append("missing train-loop file")
            return tuple(failures)
        if not self.train_config_ok:
            failures.append("TrainConfig is not both imported/read/validated")
        if not self.only_cache_order_ok:
            failures.append("only_cache does not return before DeviceContext construction")
        if not self.sample_cadence_ok:
            failures.append("sample cadence helpers are not fully wired")
        if not self.cache_contract_ok:
            failures.append("missing explicit text/VAE cache contract or preflight binding")
        if self.cache_f32_marker_count:
            failures.append("unsafe F32 cache-boundary markers are present")
        if strict_host_f32 and self.host_f32_marker_count:
            failures.append("broader host-F32 carriers are present")
        return tuple(failures)


TARGETS: tuple[LoopTarget, ...] = (
    LoopTarget("Qwen", TRAINING_DIR / "train_qwenimage_real.mojo"),
    LoopTarget("Ernie", TRAINING_DIR / "train_ernie_real.mojo"),
    LoopTarget("Anima", TRAINING_DIR / "train_anima_real.mojo"),
    LoopTarget("Klein", TRAINING_DIR / "train_klein_real.mojo"),
    LoopTarget("Z-Image", TRAINING_DIR / "train_zimage_real.mojo"),
    LoopTarget("Chroma", TRAINING_DIR / "train_chroma_real.mojo"),
    LoopTarget("Flux.1", TRAINING_DIR / "train_flux_real.mojo"),
    LoopTarget("SD3.5", TRAINING_DIR / "train_sd35_real.mojo"),
    LoopTarget("SDXL", TRAINING_DIR / "train_sdxl_real.mojo"),
)

TARGET_ALIASES: dict[str, str] = {
    "qwen": "Qwen",
    "qwenimage": "Qwen",
    "qwen-image": "Qwen",
    "ernie": "Ernie",
    "anima": "Anima",
    "klein": "Klein",
    "flux2": "Klein",
    "flux2-klein": "Klein",
    "zimage": "Z-Image",
    "z-image": "Z-Image",
    "chroma": "Chroma",
    "flux": "Flux.1",
    "flux1": "Flux.1",
    "flux.1": "Flux.1",
    "flux1-dev": "Flux.1",
    "sd35": "SD3.5",
    "sd3.5": "SD3.5",
    "sd3_5": "SD3.5",
    "sdxl": "SDXL",
}


CACHE_F32_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("_cache_f32", re.compile(r"\b_cache_f32\b")),
    ("_read_cache_f32", re.compile(r"\b_read_cache_f32\b")),
    ("_load_host_f32", re.compile(r"\b_load_host_f32\b")),
    ("_load_host returning List[Float32]", re.compile(r"\bdef\s+_load_host\b.*List\[Float32\]")),
    ("_load_cache_tensor casts F32", re.compile(r"\bdef\s+_load_cache_tensor\b|cast_tensor\([^#]*STDtype\.F32")),
    ("cache tensor to_host", re.compile(r"\b(?:_cache_tensor|s\.(?:latent|text_embedding)|text_emb_cache)\b.*\.to_host\(\s*ctx\s*\)")),
)

HOST_F32_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("List[Float32]", re.compile(r"\bList\[Float32\]")),
    ("Tensor.from_host F32", re.compile(r"\bTensor\.from_host\([^#]*STDtype\.F32")),
    ("cast_tensor F32", re.compile(r"\bcast_tensor\([^#]*STDtype\.F32")),
    (".to_host(ctx)", re.compile(r"\.to_host\(\s*ctx\s*\)")),
)


def code_part(line: str) -> str:
    return line.split("#", 1)[0]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def first_line(lines: list[str], pattern: str | re.Pattern[str]) -> int | None:
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    for i, line in enumerate(lines, start=1):
        if compiled.search(code_part(line)):
            return i
    return None


def first_non_import_line(lines: list[str], pattern: str | re.Pattern[str]) -> int | None:
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    for i, line in enumerate(lines, start=1):
        code = code_part(line)
        if " import " in code or code.lstrip().startswith("from "):
            continue
        if compiled.search(code):
            return i
    return None


def first_line_at_or_after(
    lines: list[str],
    start_line: int | None,
    pattern: str | re.Pattern[str],
) -> int | None:
    if start_line is None:
        return first_line(lines, pattern)
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    for i, line in enumerate(lines, start=1):
        if i < start_line:
            continue
        if compiled.search(code_part(line)):
            return i
    return None


def marker_hits(lines: list[str], pattern: re.Pattern[str]) -> tuple[MarkerHit, ...]:
    hits: list[MarkerHit] = []
    for i, line in enumerate(lines, start=1):
        code = code_part(line)
        if pattern.search(code):
            hits.append(MarkerHit(i, code.strip()))
    return tuple(hits)


def scan_loop(target: LoopTarget) -> LoopReport:
    if not target.path.exists():
        return LoopReport(
            target=target,
            exists=False,
            train_config_import=None,
            train_config_read=None,
            train_config_validate=None,
            only_cache_line=None,
            device_context_line=None,
            sample_helper=None,
            sample_read=None,
            sample_validate=None,
            sample_decision=None,
            preflight_import=None,
            preflight_create=None,
            preflight_validate=None,
            direct_text_contract=None,
            direct_vae_contract=None,
            cache_f32_markers=(),
            host_f32_markers=(),
        )

    lines = target.path.read_text(encoding="utf-8").splitlines()
    cache_markers = tuple(
        MarkerReport(name, marker_hits(lines, pattern))
        for name, pattern in CACHE_F32_MARKERS
    )
    host_markers = tuple(
        MarkerReport(name, marker_hits(lines, pattern))
        for name, pattern in HOST_F32_MARKERS
    )
    main_line = first_line(lines, r"^\s*def\s+main\s*\(")

    direct_sample_read = first_non_import_line(lines, r"\bread_sample_cadence_config\s*\(")
    direct_sample_validate = first_non_import_line(lines, r"\bvalidate_step_sample_cadence\s*\(")
    shared_sample_policy = first_non_import_line(
        lines,
        r"\bot_sample_cadence_from_train_config\s*\(",
    )

    return LoopReport(
        target=target,
        exists=True,
        train_config_import=first_line(lines, r"serenitymojo\.training\.train_config"),
        train_config_read=first_line_at_or_after(lines, main_line, r"\bread_model_config\s*\("),
        train_config_validate=first_line_at_or_after(
            lines,
            main_line,
            r"^\s*validate_[A-Za-z0-9_]+_train_config\s*\(",
        ),
        only_cache_line=first_line_at_or_after(lines, main_line, r"\b(?:cfg|train_cfg)\.only_cache\b"),
        device_context_line=first_line_at_or_after(
            lines,
            main_line,
            r"\b(?:var|let)\s+\w+\s*=\s*DeviceContext\s*\(",
        ),
        sample_helper=first_non_import_line(lines, r"^\s*def\s+[A-Za-z0-9_]*sample_cadence[A-Za-z0-9_]*\s*\("),
        sample_read=direct_sample_read or shared_sample_policy,
        sample_validate=direct_sample_validate or shared_sample_policy,
        sample_decision=first_non_import_line(
            lines,
            r"\b(?:should_sample_completed_step|next_sample_completed_step)\s*\(",
        ),
        preflight_import=first_line(lines, r"serenitymojo\.training\.onetrainer_cache_preflight"),
        preflight_create=first_non_import_line(lines, r"\bcreate_onetrainer_cache_preflight_plan\s*\("),
        preflight_validate=first_non_import_line(lines, r"\bvalidate_onetrainer_cache_preflight_plan\s*\("),
        direct_text_contract=first_line(lines, r"\b(?:onetrainer_conditioning_contract|ot_text_conditioning|OT_TEXT_CACHE)\b"),
        direct_vae_contract=first_line(lines, r"\b(?:vae_encoder_contract|ot_vae_encoder|OT_VAE_ENCODER)\b"),
        cache_f32_markers=cache_markers,
        host_f32_markers=host_markers,
    )


def line_ref(line: int | None) -> str:
    return "missing" if line is None else f"line {line}"


def status(ok: bool) -> str:
    return "PASS" if ok else "MISSING"


def format_marker(marker: MarkerReport, limit: int) -> str:
    refs = ", ".join(f"{hit.line}: {hit.text}" for hit in marker.hits[:limit])
    if len(marker.hits) > limit:
        refs += f", ... +{len(marker.hits) - limit} more"
    return f"{marker.name} x{len(marker.hits)} [{refs}]"


def print_report(reports: list[LoopReport], marker_limit: int, *, strict_host_f32: bool) -> None:
    print("train-loop cache/preflight binding report")
    print(f"repo: {REPO}")
    print(
        "mode: report-only by default; pass --strict to fail gaps"
        + ("; --strict-host-f32 active" if strict_host_f32 else "")
    )
    print("")

    for report in reports:
        print(f"{report.target.label} ({rel(report.target.path)})")
        if not report.exists:
            print("  file: MISSING")
            print("")
            continue

        print(
            "  TrainConfig: "
            f"{status(report.train_config_ok)} "
            f"import={line_ref(report.train_config_import)} "
            f"read={line_ref(report.train_config_read)} "
            f"validate={line_ref(report.train_config_validate)}"
        )
        if report.device_context_line is None:
            order_detail = "DeviceContext construction not found"
        else:
            order_detail = (
                f"only_cache={line_ref(report.only_cache_line)} "
                f"DeviceContext={line_ref(report.device_context_line)}"
            )
        print(f"  only_cache before DeviceContext: {status(report.only_cache_order_ok)} {order_detail}")
        print(
            "  sample cadence helpers: "
            f"{status(report.sample_cadence_ok)} "
            f"helper={line_ref(report.sample_helper)} "
            f"read={line_ref(report.sample_read)} "
            f"validate={line_ref(report.sample_validate)} "
            f"decision={line_ref(report.sample_decision)}"
        )
        print(
            "  text/VAE cache preflight binding: "
            f"{status(report.cache_contract_ok)} "
            f"preflight_import={line_ref(report.preflight_import)} "
            f"create={line_ref(report.preflight_create)} "
            f"validate={line_ref(report.preflight_validate)} "
            f"text_contract={line_ref(report.direct_text_contract)} "
            f"vae_contract={line_ref(report.direct_vae_contract)}"
        )
        found_cache_markers = [marker for marker in report.cache_f32_markers if marker.hits]
        if found_cache_markers:
            print(f"  unsafe F32 cache-boundary markers: WARN total={report.cache_f32_marker_count}")
            for marker in found_cache_markers:
                print(f"    {format_marker(marker, marker_limit)}")
        else:
            print("  unsafe F32 cache-boundary markers: PASS none")
        found_host_markers = [marker for marker in report.host_f32_markers if marker.hits]
        if found_host_markers:
            print(f"  broader host-F32 carriers: WARN total={report.host_f32_marker_count}")
            for marker in found_host_markers:
                print(f"    {format_marker(marker, marker_limit)}")
        else:
            print("  broader host-F32 carriers: PASS none")
        print("")

    missing_files = sum(1 for report in reports if not report.exists)
    missing_contracts = sum(1 for report in reports if report.exists and not report.cache_contract_ok)
    cache_marker_loops = sum(1 for report in reports if report.cache_f32_marker_count)
    host_marker_loops = sum(1 for report in reports if report.host_f32_marker_count)
    strict_fail_loops = sum(
        1 for report in reports if report.strict_failures(strict_host_f32=strict_host_f32)
    )
    print(
        "summary: "
        f"loops={len(reports)} missing_files={missing_files} "
        f"missing_text_vae_preflight={missing_contracts} "
        f"loops_with_unsafe_cache_f32_markers={cache_marker_loops} "
        f"loops_with_host_f32_markers={host_marker_loops} "
        f"strict_fail_loops={strict_fail_loops}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report train-loop TrainConfig/cache/preflight binding status.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero when any required binding is missing or unsafe F32 cache-boundary markers are present",
    )
    parser.add_argument(
        "--strict-host-f32",
        action="store_true",
        help=(
            "with --strict, also fail selected loops that still have broader "
            "host-F32 carriers"
        ),
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "comma-separated model filter: qwen, ernie, anima, klein, zimage, "
            "chroma, flux1, sd35, sdxl; default all"
        ),
    )
    parser.add_argument(
        "--marker-limit",
        type=int,
        default=3,
        help="number of line samples to print per host-F32 marker",
    )
    return parser.parse_args()


def selected_targets(spec: str) -> list[LoopTarget]:
    raw = [part.strip().lower() for part in spec.split(",") if part.strip()]
    if not raw or raw == ["all"]:
        return list(TARGETS)

    by_label = {target.label: target for target in TARGETS}
    selected: list[LoopTarget] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for item in raw:
        label = TARGET_ALIASES.get(item)
        if label is None:
            unknown.append(item)
            continue
        if label not in seen:
            selected.append(by_label[label])
            seen.add(label)
    if unknown:
        valid = ", ".join(sorted(TARGET_ALIASES))
        raise ValueError(f"unknown model filter(s): {', '.join(unknown)}; valid aliases: {valid}")
    return selected


def main() -> int:
    args = parse_args()
    marker_limit = max(0, args.marker_limit)
    try:
        targets = selected_targets(args.models)
    except ValueError as exc:
        print(f"[train-loop-cache] FAIL: {exc}")
        return 1
    reports = [scan_loop(target) for target in targets]
    print_report(reports, marker_limit, strict_host_f32=args.strict_host_f32)

    if args.strict:
        failures = [
            (report, report.strict_failures(strict_host_f32=args.strict_host_f32))
            for report in reports
            if report.strict_failures(strict_host_f32=args.strict_host_f32)
        ]
        if failures:
            print("")
            print("STRICT FAIL")
            for report, reasons in failures:
                print(f"  {report.target.label}: {', '.join(reasons)}")
            return 1
        print("")
        print("STRICT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
