#!/usr/bin/env python3
"""Static guard for the OneTrainer product cache preflight contract.

This guard is intentionally narrow. It checks that the no-CUDA product preflight
is wired to the shared OneTrainer text-conditioning and VAE encoder/cache
contracts, that only_cache cannot pass without raw VAE cache readiness, and that
the product dry-run/smoke surfaces expose the cache preflight fields.
"""

from __future__ import annotations

from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]

CACHE_PREFLIGHT = REPO / "serenitymojo/training/onetrainer_cache_preflight.mojo"
CACHE_PREFLIGHT_SMOKE = REPO / "serenitymojo/training/onetrainer_cache_preflight_smoke.mojo"
PRODUCT_RUN = REPO / "serenitymojo/training/onetrainer_product_run.mojo"
PRODUCT_RUN_SMOKE = REPO / "serenitymojo/training/onetrainer_product_run_smoke.mojo"
TRAIN_DRY_RUN = REPO / "serenitymojo/training/onetrainer_train_dry_run.mojo"

TEXT_CONTRACT = REPO / "serenitymojo/models/text_encoder/onetrainer_conditioning_contract.mojo"
VAE_CONTRACT = REPO / "serenitymojo/sampling/vae_encoder_contract.mojo"

ALLOWED_SOURCE_ROOTS = (
    "/home/alex/OneTrainer",
    "/home/alex/OneTrainer-anima-ref",
)


def read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def require(path: Path, label: str, needles: list[str]) -> None:
    text = read(path)
    missing = [needle for needle in needles if needle not in text]
    if missing:
        details = "\n".join(f"  missing: {needle!r}" for needle in missing)
        raise AssertionError(f"{label}: {path}\n{details}")


def require_regex(path: Path, label: str, pattern: str) -> None:
    text = read(path)
    if re.search(pattern, text, re.S) is None:
        raise AssertionError(f"{label}: {path}: missing pattern {pattern!r}")


def forbid_regex(path: Path, label: str, patterns: list[str]) -> None:
    text = read(path)
    found: list[str] = []
    for pattern in patterns:
        if re.search(pattern, text):
            found.append(pattern)
    if found:
        details = "\n".join(f"  forbidden pattern: {pattern!r}" for pattern in found)
        raise AssertionError(f"{label}: {path}\n{details}")


def require_local_reference_scope(paths: list[Path]) -> None:
    """Forbid external OneTrainer/Anima reference paths in this product preflight.

    The module does not need to open OneTrainer sources; those are already
    checked by the shared conditioning/VAE static guards. If a future edit adds a
    source reference here, keep it local to the user's checked-out trees.
    """

    url_re = re.compile(r"https?://[^\s\"')]+", re.I)
    path_re = re.compile(r"/[^\s\"')]+")
    bad_refs: list[str] = []

    for path in paths:
        text = read(path)
        for match in url_re.findall(text):
            lowered = match.lower()
            if "onetrainer" in lowered or "anima" in lowered:
                bad_refs.append(f"{path}: external URL {match}")
        for match in path_re.findall(text):
            if "OneTrainer" not in match:
                continue
            if not any(match.startswith(root) for root in ALLOWED_SOURCE_ROOTS):
                bad_refs.append(f"{path}: non-local OneTrainer path {match}")

    if bad_refs:
        raise AssertionError("reference scope must stay local:\n" + "\n".join(bad_refs))


def main() -> int:
    # Existence gates are explicit so a deleted module or smoke fails before
    # product code can silently skip the cache preflight.
    for path in [
        CACHE_PREFLIGHT,
        CACHE_PREFLIGHT_SMOKE,
        PRODUCT_RUN,
        PRODUCT_RUN_SMOKE,
        TRAIN_DRY_RUN,
        TEXT_CONTRACT,
        VAE_CONTRACT,
    ]:
        read(path)

    require_local_reference_scope(
        [
            CACHE_PREFLIGHT,
            CACHE_PREFLIGHT_SMOKE,
            PRODUCT_RUN,
            PRODUCT_RUN_SMOKE,
            TRAIN_DRY_RUN,
        ]
    )

    require(
        TEXT_CONTRACT,
        "shared text-conditioning contract surface",
        [
            "struct OneTrainerTextCacheReadinessContract",
            "default_ot_text_conditioning_plan",
            "ot_text_conditioning_cache_readiness_contract",
            "validate_ot_text_conditioning_cache_readiness",
            "preserve_checkpoint_or_train_dtype_at_tensor_boundaries",
        ],
    )
    require(
        VAE_CONTRACT,
        "shared VAE encoder/cache contract surface",
        [
            "struct OneTrainerVaeEncoderContract",
            "default_ot_vae_encoder_contract",
            "validate_ot_vae_encoder_contract",
            "require_raw_cache_encode_ready",
            "OT_VAE_ENCODER_RAW_CACHE_READY",
            "OT_VAE_ENCODER_PARTIAL_PREPARED_ONLY",
        ],
    )

    require(
        CACHE_PREFLIGHT,
        "cache preflight imports shared contracts",
        [
            "from serenitymojo.models.text_encoder.onetrainer_conditioning_contract import (",
            "OT_TEXT_CACHE_USE_SAMPLE",
            "OT_TEXT_CACHE_USE_TRAIN",
            "default_ot_text_conditioning_plan",
            "ot_text_conditioning_cache_readiness_contract",
            "validate_ot_text_conditioning_cache_readiness",
            "validate_ot_text_conditioning_plan",
            "from serenitymojo.sampling.vae_encoder_contract import (",
            "OT_VAE_ENCODER_PARTIAL_PREPARED_ONLY",
            "OT_VAE_ENCODER_RAW_CACHE_READY",
            "OT_VAE_ENCODER_UNSUPPORTED",
            "default_ot_vae_encoder_contract",
            "validate_ot_vae_encoder_contract",
        ],
    )
    require(
        CACHE_PREFLIGHT,
        "cache preflight plan fields",
        [
            "struct OTCachePreflightPlan",
            "var text_train_required_fields: String",
            "var text_sample_required_fields: String",
            "var text_train_requires_mask: Bool",
            "var text_sample_requires_mask: Bool",
            "var text_requires_runtime_ids: Bool",
            "var vae_raw_cache_status: Int",
            "var vae_prepared_encode_status: Int",
            "var only_cache_requested: Bool",
            "var dtype_policy: String",
            "var fail_loud_policy: String",
            "preserve_checkpoint_or_train_dtype_at_tensor_boundaries",
            "raise_before_product_cache_or_train_when_required_contract_is_missing",
        ],
    )
    require(
        CACHE_PREFLIGHT,
        "only_cache raw VAE cache readiness guard",
        [
            "def raw_vae_cache_ready(self) -> Bool:",
            "self.vae_raw_cache_status == OT_VAE_ENCODER_RAW_CACHE_READY",
            "if plan.only_cache_requested and not plan.raw_vae_cache_ready():",
            "OneTrainer only_cache requires raw VAE cache encode readiness",
            "raw_status=",
            "prepared_status=",
        ],
    )
    require_regex(
        CACHE_PREFLIGHT,
        "only_cache guard must fail before product cache/train",
        r"if\s+plan\.only_cache_requested\s+and\s+not\s+plan\.raw_vae_cache_ready\(\):\s*raise Error",
    )

    require(
        CACHE_PREFLIGHT_SMOKE,
        "cache preflight smoke covers raw VAE only_cache readiness",
        [
            "_expect_only_cache_raw_status",
            '_expect_only_cache_raw_status(String("qwenimage"), True)',
            '_expect_only_cache_raw_status(String("STABLE_DIFFUSION_35"), True)',
            '_expect_model_raises(String("STABLE_DIFFUSION_3"))',
            '_expect_only_cache_raw_status(String("klein"), False)',
            '_expect_only_cache_raw_status(String("FLUX_2_DEV"), False)',
            '_expect_only_cache_raw_status(String("flux"), False)',
            '_expect_only_cache_raw_status(String("chroma"), False)',
            "onetrainer_cache_preflight_smoke PASS",
        ],
    )

    f32_storage_markers = [
        r"\bDType\.float32\b",
        r"\bSTDtype\.F32\b",
        r"\bFloat32Tensor\b",
        r"\bTensor\s*\[[^\]]*(?:Float32|float32|F32)",
        r"\bto_f32\b",
        r"\bfloat32\b",
        r"\bFloat32\b",
        r"\bF32\b",
    ]
    forbid_regex(CACHE_PREFLIGHT, "cache preflight dtype storage boundary", f32_storage_markers)
    forbid_regex(CACHE_PREFLIGHT_SMOKE, "cache preflight smoke dtype storage boundary", f32_storage_markers)

    require(
        PRODUCT_RUN,
        "product run wires cache preflight",
        [
            "from serenitymojo.training.onetrainer_cache_preflight import (",
            "OTCachePreflightPlan",
            "create_onetrainer_cache_preflight_plan",
            "onetrainer_cache_preflight_summary",
            "validate_onetrainer_cache_preflight_plan",
            "var cache_preflight: OTCachePreflightPlan",
            "var cache_preflight = create_onetrainer_cache_preflight_plan(cfg)",
            "validate_onetrainer_cache_preflight_plan(cache_preflight)",
            "validate_onetrainer_cache_preflight_plan(plan.cache_preflight)",
            "onetrainer_cache_preflight_summary(plan.cache_preflight)",
        ],
    )
    require(
        TRAIN_DRY_RUN,
        "dry run prints cache preflight fields",
        [
            "plan.cache_preflight.text_train_required_fields",
            "plan.cache_preflight.text_sample_required_fields",
            "plan.cache_preflight.raw_vae_cache_ready()",
            "plan.cache_preflight.prepared_only()",
            "text_train_cache_fields:",
            "text_sample_cache_fields:",
            "raw_vae_cache_ready:",
            "prepared_vae_only:",
        ],
    )
    require(
        PRODUCT_RUN_SMOKE,
        "product smoke checks cache preflight fields",
        [
            "plan.cache_preflight.model_type",
            "plan.cache_preflight.text_train_required_fields",
            "plan.cache_preflight.text_sample_required_fields",
            "plan.cache_preflight.vae_cache_channels",
            "plan.cache_preflight.raw_vae_cache_ready()",
            "plan.cache_preflight.prepared_only()",
            "Flux2 dev has no product runner",
            "num_attention_heads==48 has no product runner",
            "only_cache raw VAE not ready",
            "onetrainer_product_run_smoke PASS",
        ],
    )

    print("OneTrainer cache preflight contract static guard PASS")
    print("  checked files: cache preflight module/smoke, product run, dry run, product smoke")
    print("  evidence type: static/no-CUDA only, not tensor cache parity")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"OneTrainer cache preflight contract static guard FAIL: {exc}")
        raise SystemExit(1)
