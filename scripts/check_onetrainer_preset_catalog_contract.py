#!/usr/bin/env python3
"""Static guard for the OneTrainer-style Mojo preset catalog.

This is no-CUDA and does not prove training parity. It guards the product-layer
catalog against alias/reference drift and makes sure unsafe aliases stay
fail-loud instead of silently routing to the wrong runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "serenitymojo/training/onetrainer_preset_catalog.mojo"
PRODUCT_RUN = REPO / "serenitymojo/training/onetrainer_product_run.mojo"
SMOKE = REPO / "serenitymojo/training/onetrainer_product_run_smoke.mojo"


@dataclass(frozen=True)
class AliasExpectation:
    alias: str
    preset_id: str
    model_type: str
    config_file: str
    reference: Path
    product_wired: bool


def ot_preset(name: str) -> Path:
    return Path("/home/alex/OneTrainer/training_presets") / name


def ot_config(name: str) -> Path:
    return Path("/home/alex/OneTrainer/configs") / name


def anima_preset(name: str) -> Path:
    return Path("/home/alex/OneTrainer-anima-ref/training_presets") / name


PRODUCT_WIRED: tuple[AliasExpectation, ...] = (
    AliasExpectation("qwen_lora_16gb", "qwen_lora_16gb", "qwenimage", "qwenimage.json", ot_preset("#qwen LoRA 16GB.json"), True),
    AliasExpectation("qwen_lora_24gb", "qwen_lora_24gb", "qwenimage", "qwenimage.json", ot_preset("#qwen LoRA 24GB.json"), True),
    AliasExpectation("ernie_lora_8gb", "ernie_lora_8gb", "ernie_image", "ernie_image.json", ot_preset("#ernie LoRA 8GB.json"), True),
    AliasExpectation("ernie_lora_16gb", "ernie_lora_16gb", "ernie_image", "ernie_image.json", ot_preset("#ernie LoRA 16GB.json"), True),
    AliasExpectation("anima_lora", "anima_lora", "anima", "anima.json", anima_preset("#anima LoRA.json"), True),
    AliasExpectation("sd35_lora", "sd35_lora", "STABLE_DIFFUSION_35", "sd35.json", ot_config("sd35m_100step_baseline.json"), True),
    AliasExpectation("sd3-5", "sd35_lora", "STABLE_DIFFUSION_35", "sd35.json", ot_config("sd35m_100step_baseline.json"), True),
    AliasExpectation("sdxl_1_0_lora", "sdxl_1_0_lora", "sdxl", "sdxl.json", ot_preset("#sdxl 1.0 LoRA.json"), True),
    AliasExpectation("flux1_dev", "flux1_dev_lora", "flux", "flux.json", ot_preset("#flux LoRA.json"), True),
    AliasExpectation("flux1_dev_lora", "flux1_dev_lora", "flux", "flux.json", ot_preset("#flux LoRA.json"), True),
    AliasExpectation("flux2_lora_8gb", "flux2_lora_8gb", "klein", "klein9b.json", ot_preset("#flux2 LoRA 8GB.json"), True),
    AliasExpectation("flux2_lora_16gb", "flux2_klein9b_lora_16gb", "klein", "klein9b.json", ot_preset("#flux2 LoRA 16GB.json"), True),
    AliasExpectation("klein9b_lora_8gb", "flux2_lora_8gb", "klein", "klein9b.json", ot_preset("#flux2 LoRA 8GB.json"), True),
    AliasExpectation("klein9b_lora_16gb", "flux2_klein9b_lora_16gb", "klein", "klein9b.json", ot_preset("#flux2 LoRA 16GB.json"), True),
    AliasExpectation("chroma_lora_8gb", "chroma_lora_8gb", "chroma", "chroma.json", ot_preset("#chroma LoRA 8GB.json"), True),
    AliasExpectation("chroma_lora_16gb", "chroma_lora_16gb", "chroma", "chroma.json", ot_preset("#chroma LoRA 16GB.json"), True),
    AliasExpectation("chroma_lora_24gb", "chroma_lora_24gb", "chroma", "chroma.json", ot_preset("#chroma LoRA 24GB.json"), True),
    AliasExpectation("zimage_lora_8gb", "zimage_lora_8gb", "zimage", "zimage.json", ot_preset("#z-image LoRA 8GB.json"), True),
    AliasExpectation("z-image_lora_8gb", "zimage_lora_8gb", "zimage", "zimage.json", ot_preset("#z-image LoRA 8GB.json"), True),
    AliasExpectation("zimage_lora_16gb", "zimage_lora_16gb", "zimage", "zimage.json", ot_preset("#z-image LoRA 16GB.json"), True),
    AliasExpectation("z-image_lora_16gb", "zimage_lora_16gb", "zimage", "zimage.json", ot_preset("#z-image LoRA 16GB.json"), True),
)


NON_WIRED: tuple[AliasExpectation, ...] = (
    AliasExpectation("qwen_finetune_16gb", "qwen_finetune_16gb", "qwenimage", "qwenimage.json", ot_preset("#qwen Finetune 16GB.json"), False),
    AliasExpectation("qwen_finetune_24gb", "qwen_finetune_24gb", "qwenimage", "qwenimage.json", ot_preset("#qwen Finetune 24GB.json"), False),
    AliasExpectation("anima_finetune", "anima_finetune", "anima", "anima.json", anima_preset("#anima Finetune.json"), False),
    AliasExpectation("sdxl_1_0_finetune", "sdxl_1_0_finetune", "sdxl", "sdxl.json", ot_preset("#sdxl 1.0.json"), False),
    AliasExpectation("sdxl_inpaint_lora", "sdxl_1_0_inpaint_lora", "STABLE_DIFFUSION_XL_10_BASE_INPAINTING", "sdxl.json", ot_preset("#sdxl 1.0 inpaint LoRA.json"), False),
    AliasExpectation("flux2_finetune_16gb", "flux2_finetune_16gb", "klein", "klein9b.json", ot_preset("#flux2 Finetune 16GB.json"), False),
    AliasExpectation("flux2_finetune_24gb", "flux2_finetune_24gb", "klein", "klein9b.json", ot_preset("#flux2 Finetune 24GB.json"), False),
    AliasExpectation("klein4b", "klein4b_not_product_wired", "klein", "klein4b.json", ot_preset("#flux2 LoRA 8GB.json"), False),
    AliasExpectation("chroma_finetune_8gb", "chroma_finetune_8gb", "chroma", "chroma.json", ot_preset("#chroma Finetune 8GB.json"), False),
    AliasExpectation("chroma_finetune_16gb", "chroma_finetune_16gb", "chroma", "chroma.json", ot_preset("#chroma Finetune 16GB.json"), False),
    AliasExpectation("chroma_finetune_24gb", "chroma_finetune_24gb", "chroma", "chroma.json", ot_preset("#chroma Finetune 24GB.json"), False),
    AliasExpectation("zimage_finetune_16gb", "zimage_finetune_16gb", "zimage", "zimage.json", ot_preset("#z-image Finetune 16GB.json"), False),
    AliasExpectation("zimage_finetune_24gb", "zimage_finetune_24gb", "zimage", "zimage.json", ot_preset("#z-image Finetune 24GB.json"), False),
    AliasExpectation("zimage_deturbo_lora_8gb", "zimage_deturbo_lora_8gb", "zimage", "zimage.json", ot_preset("#z-image DeTurbo LoRA 8GB.json"), False),
    AliasExpectation("zimage_deturbo_lora_16gb", "zimage_deturbo_lora_16gb", "zimage", "zimage.json", ot_preset("#z-image DeTurbo LoRA 16GB.json"), False),
)


def require(cond: bool, msg: str, failures: list[str]) -> None:
    if not cond:
        failures.append(msg)


def check_expectation(exp: AliasExpectation, text: str, failures: list[str]) -> None:
    require(f'String("{exp.alias}")' in text, f"missing alias {exp.alias}", failures)
    require(exp.preset_id in text, f"missing preset id {exp.preset_id}", failures)
    require(exp.model_type in text, f"missing model_type {exp.model_type} for {exp.alias}", failures)
    require(exp.config_file in text, f"missing config file {exp.config_file} for {exp.alias}", failures)
    require(exp.reference.name in text, f"missing reference {exp.reference.name} for {exp.alias}", failures)
    require(exp.reference.exists(), f"reference file missing on disk: {exp.reference}", failures)
    local_config = REPO / "serenitymojo/configs" / exp.config_file
    require(local_config.exists(), f"local config missing on disk: {local_config}", failures)


def main() -> int:
    catalog = CATALOG.read_text(encoding="utf-8")
    product_run = PRODUCT_RUN.read_text(encoding="utf-8")
    smoke = SMOKE.read_text(encoding="utf-8")
    failures: list[str] = []

    for exp in PRODUCT_WIRED + NON_WIRED:
        check_expectation(exp, catalog, failures)

    for alias in ("sd3", "sd3_lora", "STABLE_DIFFUSION_3"):
        require(f'String("{alias}")' in catalog, f"plain SD3 alias not explicitly blocked: {alias}", failures)
        require(f'String("{alias}")' in smoke, f"plain SD3 alias not covered by product smoke: {alias}", failures)

    require("if not entry.product_run_wired:" in product_run, "product preset entrypoint does not enforce product_run_wired", failures)
    for marker in (
        "recipe_family",
        "variant_kind",
        "vram_tier_gb",
        "def _vram_tier_gb",
        "def _variant_kind",
    ):
        require(marker in catalog, f"preset catalog missing recipe metadata marker: {marker}", failures)
    for marker in (
        "materialize_onetrainer_preset_config",
        "/tmp/mojo-ot-presets",
        "resolved_config_materialized",
        "mojo_onetrainer_preset_id",
        "mojo_onetrainer_reference_path",
        "ot_sample_cadence_from_train_config",
    ):
        require(marker in product_run, f"product run missing resolved preset config marker: {marker}", failures)
    require(
        "read_sample_cadence_config" not in product_run,
        "product run must use shared OneTrainer sample cadence policy, not local parsing",
        failures,
    )
    for marker in (
        "resolved config materialized",
        "materialized command",
        "/tmp/mojo-ot-presets/",
    ):
        require(marker in smoke, f"product smoke missing resolved preset coverage: {marker}", failures)
    for alias in ("klein4b", "zimage_deturbo_lora_16gb", "sdxl_inpaint_lora"):
        require(f'String("{alias}")' in smoke, f"non-product-wired alias not covered by product smoke: {alias}", failures)

    print("[preset-catalog] product-wired aliases:", len(PRODUCT_WIRED))
    print("[preset-catalog] non-product-wired aliases:", len(NON_WIRED))
    if failures:
        print("[preset-catalog] FAIL")
        for failure in failures:
            print("[preset-catalog] ", failure)
        return 1
    print("[preset-catalog] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
