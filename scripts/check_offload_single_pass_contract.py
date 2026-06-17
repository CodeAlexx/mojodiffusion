#!/usr/bin/env python3
"""Static guard for explicit single-pass offload config.

The non-CFG inference path should advertise single-branch block residency with
`OffloadConfig.single_pass()`, not the older `synchronous_single()` name. The
legacy method stays as a compatibility alias, but product model loops should use
the explicit contract so they do not drift back into CFG-paired bookkeeping.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "serenitymojo/offload/plan.mojo"
PLAN_SMOKE = ROOT / "serenitymojo/offload/plan_smoke.mojo"
PLANNED_SMOKE = ROOT / "serenitymojo/offload/planned_loader_smoke.mojo"

MODEL_FILES = (
    ROOT / "serenitymojo/models/dit/klein_dit.mojo",
    ROOT / "serenitymojo/models/dit/qwenimage_dit.mojo",
    ROOT / "serenitymojo/models/dit/sensenova_u1.mojo",
    ROOT / "serenitymojo/models/lance/lance_t2v.mojo",
)


PLAN_REQUIRED = (
    "def single_pass() -> OffloadConfig:",
    "return OffloadConfig(1, 1, DTypePolicy.preserve(), BranchSchedule.single())",
    "def synchronous_single() -> OffloadConfig:",
    "return OffloadConfig.single_pass()",
)

SMOKE_REQUIRED = (
    (PLAN_SMOKE, "var single = OffloadConfig.single_pass()"),
    (PLAN_SMOKE, "klein sync single alias visits"),
    (PLANNED_SMOKE, "var single = OffloadConfig.single_pass()"),
    (PLANNED_SMOKE, "klein single branch visits"),
)


def main() -> int:
    failures: list[str] = []
    plan_text = PLAN.read_text(encoding="utf-8")
    for token in PLAN_REQUIRED:
        if token not in plan_text:
            failures.append(f"{PLAN.relative_to(ROOT)} missing: {token}")

    for path, token in SMOKE_REQUIRED:
        text = path.read_text(encoding="utf-8")
        if token not in text:
            failures.append(f"{path.relative_to(ROOT)} missing: {token}")

    for path in MODEL_FILES:
        text = path.read_text(encoding="utf-8")
        if "OffloadConfig.single_pass()" not in text:
            failures.append(f"{path.relative_to(ROOT)} missing OffloadConfig.single_pass()")
        if "OffloadConfig.synchronous_single()" in text:
            failures.append(
                f"{path.relative_to(ROOT)} still uses OffloadConfig.synchronous_single()"
            )

    if failures:
        print("offload single-pass contract: FAIL")
        for failure in failures:
            print(failure)
        return 1

    print("offload single-pass contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
