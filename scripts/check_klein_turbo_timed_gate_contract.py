#!/usr/bin/env python3
"""Static contract for the Klein Turbo timed gate.

This does not replace the GPU run. It prevents the gate source from degrading
back into a structural-only smoke by requiring the local timed evidence fields:
copy-stream/default-stream modes, wall-clock timing, CUDA memory sampling, and a
hard parity check.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "serenitymojo/pipeline/klein_turbo_timed_gate.mojo"
LOADER = ROOT / "serenitymojo/offload/turbo_loader.mojo"
PLANNED = ROOT / "serenitymojo/offload/turbo_planned_loader.mojo"
KLEIN = ROOT / "serenitymojo/models/dit/klein_dit.mojo"
RUNNER = ROOT / "scripts/run_klein_turbo_timed_gate.py"


REQUIRED_GATE_TOKENS = (
    "perf_counter_ns",
    "cu_mem_get_info",
    "default_stream",
    "copy_stream",
    "load_with_copy_mode",
    "forward_seconds",
    "observed_peak_vram_mib",
    "speedup_default_over_copy",
    "_cosine_sim",
    "_check(cos >= Float64(0.999)",
    "KLEIN9B TURBO TIMED GATE: PASS",
)

REQUIRED_LOADER_TOKENS = (
    "open_with_copy_mode",
    "use_default_stream_copy: Bool",
    "self.use_default_stream_copy",
    "_h2d_dma_copy_raw_stream",
    "CUDA(ctx.stream())",
    "def copy_mode(self) -> String",
    "return not self.use_default_stream_copy",
)

REQUIRED_PLANNED_TOKENS = (
    "TurboPlannedLoader.open_with_copy_mode",
    "TurboBlockLoader.open_with_copy_mode",
)

REQUIRED_KLEIN_TOKENS = (
    "def load_with_copy_mode(",
    "TurboPlannedLoader.open_with_copy_mode",
)

REQUIRED_RUNNER_TOKENS = (
    "serenity.klein_turbo_timed_gate.external.v1",
    "nvidia-smi",
    "external_wall_seconds",
    "external_peak_vram_delta_mib",
    "speedup_default_over_copy",
    "accepted_speed_claim",
    "KLEIN9B TURBO TIMED GATE: PASS",
    "--min-speedup",
)


def require(path: Path, tokens: tuple[str, ...]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [token for token in tokens if token not in text]


def main() -> int:
    failures: list[str] = []
    for path, tokens in [
        (GATE, REQUIRED_GATE_TOKENS),
        (LOADER, REQUIRED_LOADER_TOKENS),
        (PLANNED, REQUIRED_PLANNED_TOKENS),
        (KLEIN, REQUIRED_KLEIN_TOKENS),
        (RUNNER, REQUIRED_RUNNER_TOKENS),
    ]:
        missing = require(path, tokens)
        if missing:
            failures.append(f"{path.relative_to(ROOT)} missing: {', '.join(missing)}")

    if failures:
        print("klein turbo timed gate contract: FAIL")
        for failure in failures:
            print(failure)
        return 1

    print("klein turbo timed gate contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
