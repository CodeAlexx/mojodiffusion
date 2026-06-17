#!/usr/bin/env python3
"""Static guard for TurboPlannedLoader dtype-boundary safety.

TurboBlockLoader raw-copies packed checkpoint bytes. Unlike PlannedBlockLoader,
it cannot satisfy `OffloadConfig.force_bf16()` by converting F32/F16 tensors
before H2D. This guard keeps the fail-loud check on every Turbo path that can
copy or expose block bytes.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TURBO_PLANNED = ROOT / "serenitymojo/offload/turbo_planned_loader.mojo"


REQUIRED_TOKENS = (
    "from serenitymojo.io.dtype import STDtype",
    "def _assert_raw_copy_dtype_safe(self, index: Int) raises:",
    "self._config.dtype_policy != DTypePolicy.force_bf16()",
    "tv.dtype != STDtype.BF16",
    "force_bf16 requires BF16",
    "PlannedBlockLoader.load_block_as_bf16",
    "converting Turbo staging path",
)

FORBIDDEN_TOKENS = (
    "pass  # Klein9B: BF16 on disk confirmed",
    "caller is responsible for ensuring on-disk",
)

ORDER_CHECKS = (
    (
        "pin_residents",
        "self._assert_raw_copy_dtype_safe(i)",
        "_h2d_dma_copy(",
    ),
    (
        "prefetch_with_ctx",
        "self._assert_raw_copy_dtype_safe(index)",
        "self._turbo.prefetch(prefix, ctx)",
    ),
    (
        "await_block",
        "self._assert_raw_copy_dtype_safe(index)",
        "var block = self._turbo.await_block(load_prefix, ctx)",
    ),
)


def method_slice(text: str, method_name: str) -> str:
    needle = f"    def {method_name}"
    start = text.find(needle)
    if start < 0:
        return ""
    next_start = text.find("\n    def ", start + len(needle))
    if next_start < 0:
        return text[start:]
    return text[start:next_start]


def main() -> int:
    text = TURBO_PLANNED.read_text(encoding="utf-8")
    failures: list[str] = []

    for token in REQUIRED_TOKENS:
        if token not in text:
            failures.append(f"missing token: {token}")
    for token in FORBIDDEN_TOKENS:
        if token in text:
            failures.append(f"forbidden stale token remains: {token}")

    for method_name, before, after in ORDER_CHECKS:
        body = method_slice(text, method_name)
        if not body:
            failures.append(f"missing method: {method_name}")
            continue
        before_idx = body.find(before)
        after_idx = body.find(after)
        if before_idx < 0:
            failures.append(f"{method_name} missing guard call: {before}")
        if after_idx < 0:
            failures.append(f"{method_name} missing guarded operation: {after}")
        if before_idx >= 0 and after_idx >= 0 and before_idx > after_idx:
            failures.append(f"{method_name}: dtype guard must run before {after}")

    if failures:
        print("turbo dtype guard contract: FAIL")
        for failure in failures:
            print(failure)
        return 1

    print("turbo dtype guard contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
