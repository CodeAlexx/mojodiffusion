# training/onetrainer_preset_catalog.mojo
#
# Repo-local catalog for the OneTrainer preset names this port is targeting.
# OneTrainer presets are recipe defaults, not datasets; users still provide the
# concept file for a real run. The catalog maps those preset names to the local
# Mojo config files that the product entrypoint can read.

from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.training.train_config import (
    TRAINING_METHOD_FINE_TUNE,
    TRAINING_METHOD_LORA,
    TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_OPTIMIZER_ADAMW,
)


comptime OT_PRESET_CONFIG_ROOT = "/home/alex/mojodiffusion/serenitymojo/configs"
comptime OT_REF_PRESET_ROOT = "/home/alex/OneTrainer/training_presets"
comptime OT_REF_CONFIG_ROOT = "/home/alex/OneTrainer/configs"
comptime OT_ANIMA_REF_PRESET_ROOT = "/home/alex/OneTrainer-anima-ref/training_presets"


@fieldwise_init
struct OTPresetCatalogEntry(Copyable, Movable):
    var preset_id: String
    var display_name: String
    var recipe_family: String
    var variant_kind: String
    var vram_tier_gb: Int
    var model_type: String
    var config_path: String
    var ot_reference_path: String
    var ot_reference_kind: String
    var training_method: Int
    var optimizer: Int
    var requires_user_concepts: Bool
    var product_run_wired: Bool


def _config_path(file_name: String) -> String:
    return String(OT_PRESET_CONFIG_ROOT) + String("/") + file_name


def _ot_preset_path(file_name: String) -> String:
    return String(OT_REF_PRESET_ROOT) + String("/") + file_name


def _ot_config_path(file_name: String) -> String:
    return String(OT_REF_CONFIG_ROOT) + String("/") + file_name


def _anima_ref_preset_path(file_name: String) -> String:
    return String(OT_ANIMA_REF_PRESET_ROOT) + String("/") + file_name


def _recipe_family(model_type: String) -> String:
    if model_type == String("qwenimage"):
        return String("qwen")
    if model_type == String("ernie_image"):
        return String("ernie")
    if model_type == String("anima"):
        return String("anima")
    if model_type == String("STABLE_DIFFUSION_35"):
        return String("sd35")
    if model_type == String("sdxl") or model_type == String("STABLE_DIFFUSION_XL_10_BASE_INPAINTING"):
        return String("sdxl")
    if model_type == String("flux"):
        return String("flux1")
    if model_type == String("klein"):
        return String("flux2_klein")
    if model_type == String("chroma"):
        return String("chroma")
    if model_type == String("zimage"):
        return String("zimage")
    return model_type.copy()


def _vram_tier_gb(preset_id: String) -> Int:
    if (
        preset_id == String("ernie_lora_8gb")
        or preset_id == String("flux2_lora_8gb")
        or preset_id == String("chroma_lora_8gb")
        or preset_id == String("chroma_finetune_8gb")
        or preset_id == String("zimage_lora_8gb")
        or preset_id == String("zimage_deturbo_lora_8gb")
    ):
        return 8
    if (
        preset_id == String("qwen_lora_16gb")
        or preset_id == String("qwen_finetune_16gb")
        or preset_id == String("ernie_lora_16gb")
        or preset_id == String("flux2_klein9b_lora_16gb")
        or preset_id == String("flux2_finetune_16gb")
        or preset_id == String("chroma_lora_16gb")
        or preset_id == String("chroma_finetune_16gb")
        or preset_id == String("zimage_lora_16gb")
        or preset_id == String("zimage_deturbo_lora_16gb")
        or preset_id == String("zimage_finetune_16gb")
    ):
        return 16
    if (
        preset_id == String("qwen_lora_24gb")
        or preset_id == String("qwen_finetune_24gb")
        or preset_id == String("flux2_finetune_24gb")
        or preset_id == String("chroma_lora_24gb")
        or preset_id == String("chroma_finetune_24gb")
        or preset_id == String("zimage_finetune_24gb")
    ):
        return 24
    return 0


def _variant_kind(preset_id: String, training_method: Int) -> String:
    if training_method == TRAINING_METHOD_FINE_TUNE:
        return String("finetune")
    if preset_id == String("sdxl_1_0_inpaint_lora"):
        return String("inpaint_lora")
    if (
        preset_id == String("zimage_deturbo_lora_8gb")
        or preset_id == String("zimage_deturbo_lora_16gb")
    ):
        return String("deturbo_lora")
    return String("lora")


def _entry(
    preset_id: String,
    display_name: String,
    model_type: String,
    config_file: String,
    reference_path: String,
    reference_kind: String,
    training_method: Int,
    optimizer: Int,
    product_run_wired: Bool,
) -> OTPresetCatalogEntry:
    return OTPresetCatalogEntry(
        preset_id,
        display_name,
        _recipe_family(model_type),
        _variant_kind(preset_id, training_method),
        _vram_tier_gb(preset_id),
        model_type,
        _config_path(config_file),
        reference_path,
        reference_kind,
        training_method,
        optimizer,
        True,
        product_run_wired,
    )


def default_onetrainer_preset_entry(preset: String) raises -> OTPresetCatalogEntry:
    """Resolve a OneTrainer-style preset id/alias to a local Mojo config.

    Plain SD3 is intentionally rejected. The active target is SD3.5
    (`STABLE_DIFFUSION_35`), while OneTrainer's stock `#sd 3` preset is
    `STABLE_DIFFUSION_3`.
    """

    if (
        preset == String("qwen")
        or preset == String("qwen_lora")
        or preset == String("qwen_lora_16gb")
        or preset == String("qwenimage")
        or preset == String("QWEN")
    ):
        return _entry(
            String("qwen_lora_16gb"),
            String("Qwen Image LoRA 16GB"),
            String("qwenimage"),
            String("qwenimage.json"),
            _ot_preset_path(String("#qwen LoRA 16GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("qwen_lora_24gb"):
        return _entry(
            String("qwen_lora_24gb"),
            String("Qwen Image LoRA 24GB"),
            String("qwenimage"),
            String("qwenimage.json"),
            _ot_preset_path(String("#qwen LoRA 24GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("qwen_finetune") or preset == String("qwen_finetune_16gb"):
        return _entry(
            String("qwen_finetune_16gb"),
            String("Qwen Image Finetune 16GB"),
            String("qwenimage"),
            String("qwenimage.json"),
            _ot_preset_path(String("#qwen Finetune 16GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    if preset == String("qwen_finetune_24gb"):
        return _entry(
            String("qwen_finetune_24gb"),
            String("Qwen Image Finetune 24GB"),
            String("qwenimage"),
            String("qwenimage.json"),
            _ot_preset_path(String("#qwen Finetune 24GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    if (
        preset == String("ernie")
        or preset == String("ernie_lora")
        or preset == String("ernie_lora_16gb")
        or preset == String("ernie_image")
        or preset == String("ERNIE")
    ):
        return _entry(
            String("ernie_lora_16gb"),
            String("ERNIE Image LoRA 16GB"),
            String("ernie_image"),
            String("ernie_image.json"),
            _ot_preset_path(String("#ernie LoRA 16GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("ernie_lora_8gb"):
        return _entry(
            String("ernie_lora_8gb"),
            String("ERNIE Image LoRA 8GB"),
            String("ernie_image"),
            String("ernie_image.json"),
            _ot_preset_path(String("#ernie LoRA 8GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("anima") or preset == String("anima_lora") or preset == String("ANIMA"):
        return _entry(
            String("anima_lora"),
            String("Anima LoRA"),
            String("anima"),
            String("anima.json"),
            _anima_ref_preset_path(String("#anima LoRA.json")),
            String("onetrainer_anima_ref_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("anima_finetune"):
        return _entry(
            String("anima_finetune"),
            String("Anima Finetune"),
            String("anima"),
            String("anima.json"),
            _anima_ref_preset_path(String("#anima Finetune.json")),
            String("onetrainer_anima_ref_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    if (
        preset == String("sd35")
        or preset == String("sd3.5")
        or preset == String("sd3-5")
        or preset == String("sd35_lora")
        or preset == String("STABLE_DIFFUSION_35")
    ):
        return _entry(
            String("sd35_lora"),
            String("Stable Diffusion 3.5 LoRA"),
            String("STABLE_DIFFUSION_35"),
            String("sd35.json"),
            _ot_config_path(String("sd35m_100step_baseline.json")),
            String("onetrainer_local_baseline_config_not_stock_sd3_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if (
        preset == String("sd3")
        or preset == String("sd3_lora")
        or preset == String("STABLE_DIFFUSION_3")
    ):
        raise Error("OneTrainer preset catalog: plain SD3 is not a target; use sd35_lora")
    if (
        preset == String("sdxl")
        or preset == String("sdxl_lora")
        or preset == String("sdxl_1_0_lora")
        or preset == String("STABLE_DIFFUSION_XL_10_BASE")
    ):
        return _entry(
            String("sdxl_1_0_lora"),
            String("SDXL 1.0 LoRA"),
            String("sdxl"),
            String("sdxl.json"),
            _ot_preset_path(String("#sdxl 1.0 LoRA.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("sdxl_finetune") or preset == String("sdxl_1_0_finetune"):
        return _entry(
            String("sdxl_1_0_finetune"),
            String("SDXL 1.0 Finetune"),
            String("sdxl"),
            String("sdxl.json"),
            _ot_preset_path(String("#sdxl 1.0.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAMW,
            False,
        )
    if (
        preset == String("sdxl_inpaint_lora")
        or preset == String("sdxl_1_0_inpaint_lora")
        or preset == String("STABLE_DIFFUSION_XL_10_BASE_INPAINTING")
    ):
        return _entry(
            String("sdxl_1_0_inpaint_lora"),
            String("SDXL 1.0 Inpaint LoRA"),
            String("STABLE_DIFFUSION_XL_10_BASE_INPAINTING"),
            String("sdxl.json"),
            _ot_preset_path(String("#sdxl 1.0 inpaint LoRA.json")),
            String("onetrainer_inpaint_preset_not_local_base_config"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            False,
        )
    if (
        preset == String("flux")
        or preset == String("flux1")
        or preset == String("flux1_dev")
        or preset == String("flux1_lora")
        or preset == String("flux1_dev_lora")
        or preset == String("flux_lora")
        or preset == String("FLUX_DEV_1")
    ):
        return _entry(
            String("flux1_dev_lora"),
            String("FLUX.1 dev LoRA"),
            String("flux"),
            String("flux.json"),
            _ot_preset_path(String("#flux LoRA.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if (
        preset == String("flux2")
        or preset == String("flux2_lora")
        or preset == String("flux2_lora_16gb")
        or preset == String("flux2_klein9b_lora_16gb")
        or preset == String("klein")
        or preset == String("klein9b")
        or preset == String("klein9b_lora_16gb")
        or preset == String("FLUX_2")
    ):
        return _entry(
            String("flux2_klein9b_lora_16gb"),
            String("FLUX.2 Klein 9B LoRA 16GB"),
            String("klein"),
            String("klein9b.json"),
            _ot_preset_path(String("#flux2 LoRA 16GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if (
        preset == String("flux2_lora_8gb")
        or preset == String("flux2_klein9b_lora_8gb")
        or preset == String("klein9b_lora_8gb")
    ):
        return _entry(
            String("flux2_lora_8gb"),
            String("FLUX.2 Klein 9B LoRA 8GB"),
            String("klein"),
            String("klein9b.json"),
            _ot_preset_path(String("#flux2 LoRA 8GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("klein4b"):
        return _entry(
            String("klein4b_not_product_wired"),
            String("FLUX.2 Klein 4B"),
            String("klein"),
            String("klein4b.json"),
            _ot_preset_path(String("#flux2 LoRA 8GB.json")),
            String("onetrainer_training_preset_alias_not_9b_runner"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            False,
        )
    if preset == String("flux2_finetune") or preset == String("flux2_finetune_16gb"):
        return _entry(
            String("flux2_finetune_16gb"),
            String("FLUX.2 Klein 9B Finetune 16GB"),
            String("klein"),
            String("klein9b.json"),
            _ot_preset_path(String("#flux2 Finetune 16GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    if (
        preset == String("flux2_finetune_24gb")
        or preset == String("flux2_klein9b_finetune_24gb")
        or preset == String("klein9b_finetune_24gb")
    ):
        return _entry(
            String("flux2_finetune_24gb"),
            String("FLUX.2 Klein 9B Finetune 24GB"),
            String("klein"),
            String("klein9b.json"),
            _ot_preset_path(String("#flux2 Finetune 24GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    if (
        preset == String("chroma")
        or preset == String("chroma_lora")
        or preset == String("chroma_lora_16gb")
        or preset == String("CHROMA_1")
    ):
        return _entry(
            String("chroma_lora_16gb"),
            String("Chroma1-HD LoRA 16GB"),
            String("chroma"),
            String("chroma.json"),
            _ot_preset_path(String("#chroma LoRA 16GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("chroma_lora_8gb"):
        return _entry(
            String("chroma_lora_8gb"),
            String("Chroma1-HD LoRA 8GB"),
            String("chroma"),
            String("chroma.json"),
            _ot_preset_path(String("#chroma LoRA 8GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("chroma_lora_24gb"):
        return _entry(
            String("chroma_lora_24gb"),
            String("Chroma1-HD LoRA 24GB"),
            String("chroma"),
            String("chroma.json"),
            _ot_preset_path(String("#chroma LoRA 24GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("chroma_finetune") or preset == String("chroma_finetune_16gb"):
        return _entry(
            String("chroma_finetune_16gb"),
            String("Chroma1-HD Finetune 16GB"),
            String("chroma"),
            String("chroma.json"),
            _ot_preset_path(String("#chroma Finetune 16GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    if preset == String("chroma_finetune_8gb"):
        return _entry(
            String("chroma_finetune_8gb"),
            String("Chroma1-HD Finetune 8GB"),
            String("chroma"),
            String("chroma.json"),
            _ot_preset_path(String("#chroma Finetune 8GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    if preset == String("chroma_finetune_24gb"):
        return _entry(
            String("chroma_finetune_24gb"),
            String("Chroma1-HD Finetune 24GB"),
            String("chroma"),
            String("chroma.json"),
            _ot_preset_path(String("#chroma Finetune 24GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    if (
        preset == String("zimage")
        or preset == String("z-image")
        or preset == String("zimage_lora")
        or preset == String("zimage_lora_16gb")
        or preset == String("z-image_lora_16gb")
        or preset == String("Z_IMAGE")
    ):
        return _entry(
            String("zimage_lora_16gb"),
            String("Z-Image LoRA 16GB"),
            String("zimage"),
            String("zimage.json"),
            _ot_preset_path(String("#z-image LoRA 16GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("zimage_lora_8gb") or preset == String("z-image_lora_8gb"):
        return _entry(
            String("zimage_lora_8gb"),
            String("Z-Image LoRA 8GB"),
            String("zimage"),
            String("zimage.json"),
            _ot_preset_path(String("#z-image LoRA 8GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            True,
        )
    if preset == String("zimage_deturo_lora_8gb") or preset == String("zimage_deturo_lora_16gb"):
        raise Error("OneTrainer preset catalog: use zimage_deturo_lora_8gb/deturo? typo; expected deturbo")
    if preset == String("zimage_deturbo_lora_8gb") or preset == String("z-image_deturbo_lora_8gb"):
        return _entry(
            String("zimage_deturbo_lora_8gb"),
            String("Z-Image DeTurbo LoRA 8GB"),
            String("zimage"),
            String("zimage.json"),
            _ot_preset_path(String("#z-image DeTurbo LoRA 8GB.json")),
            String("onetrainer_turbo_preset_not_local_base_config"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            False,
        )
    if preset == String("zimage_deturbo_lora_16gb") or preset == String("z-image_deturbo_lora_16gb"):
        return _entry(
            String("zimage_deturbo_lora_16gb"),
            String("Z-Image DeTurbo LoRA 16GB"),
            String("zimage"),
            String("zimage.json"),
            _ot_preset_path(String("#z-image DeTurbo LoRA 16GB.json")),
            String("onetrainer_turbo_preset_not_local_base_config"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
            False,
        )
    if preset == String("zimage_finetune") or preset == String("zimage_finetune_16gb"):
        return _entry(
            String("zimage_finetune_16gb"),
            String("Z-Image Finetune 16GB"),
            String("zimage"),
            String("zimage.json"),
            _ot_preset_path(String("#z-image Finetune 16GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    if preset == String("z-image_finetune") or preset == String("z-image_finetune_16gb"):
        return _entry(
            String("zimage_finetune_16gb"),
            String("Z-Image Finetune 16GB"),
            String("zimage"),
            String("zimage.json"),
            _ot_preset_path(String("#z-image Finetune 16GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    if preset == String("zimage_finetune_24gb") or preset == String("z-image_finetune_24gb"):
        return _entry(
            String("zimage_finetune_24gb"),
            String("Z-Image Finetune 24GB"),
            String("zimage"),
            String("zimage.json"),
            _ot_preset_path(String("#z-image Finetune 24GB.json")),
            String("onetrainer_training_preset"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    raise Error(String("OneTrainer preset catalog: unknown preset '") + preset + String("'"))


def validate_onetrainer_preset_entry(entry: OTPresetCatalogEntry) raises:
    if entry.preset_id == String("") or entry.display_name == String(""):
        raise Error("OneTrainer preset catalog: preset identity is missing")
    if entry.model_type == String("STABLE_DIFFUSION_3"):
        raise Error("OneTrainer preset catalog: plain SD3 is not a target")
    if not path_exists(entry.config_path):
        raise Error(String("OneTrainer preset catalog: local config missing: ") + entry.config_path)
    if entry.ot_reference_path != String("") and not path_exists(entry.ot_reference_path):
        raise Error(
            String("OneTrainer preset catalog: reference preset/config missing: ")
            + entry.ot_reference_path
        )
    if entry.product_run_wired:
        if entry.training_method != TRAINING_METHOD_LORA:
            raise Error("OneTrainer preset catalog: product-wired presets must be LoRA")
        if entry.optimizer != TRAIN_OPTIMIZER_ADAMW:
            raise Error("OneTrainer preset catalog: product-wired presets must use AdamW")
        if entry.variant_kind != String("lora"):
            raise Error("OneTrainer preset catalog: product-wired presets must be base LoRA variants")
    if entry.vram_tier_gb < 0:
        raise Error("OneTrainer preset catalog: invalid negative VRAM tier")


def onetrainer_preset_config_path(preset: String) raises -> String:
    var entry = default_onetrainer_preset_entry(preset)
    validate_onetrainer_preset_entry(entry)
    return entry.config_path.copy()


def onetrainer_preset_reference_path(preset: String) raises -> String:
    var entry = default_onetrainer_preset_entry(preset)
    validate_onetrainer_preset_entry(entry)
    return entry.ot_reference_path.copy()


def onetrainer_preset_summary(entry: OTPresetCatalogEntry) -> String:
    return (
        entry.preset_id
        + String(" family=")
        + entry.recipe_family
        + String(" kind=")
        + entry.variant_kind
        + String(" vram_gb=")
        + String(entry.vram_tier_gb)
        + String(" model=")
        + entry.model_type
        + String(" config=")
        + entry.config_path
        + String(" ref=")
        + entry.ot_reference_path
        + String(" method=")
        + String(entry.training_method)
        + String(" product_wired=")
        + String(entry.product_run_wired)
    )
