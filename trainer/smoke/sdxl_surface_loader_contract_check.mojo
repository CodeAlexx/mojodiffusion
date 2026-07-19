# SDXL loader/conversion surface contract gate.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelLoader/StableDiffusionXL*.py
#   /home/alex/Serenity/modules/modelLoader/stableDiffusionXL/*.py

from serenity_trainer.modelLoader.stableDiffusionXL.StableDiffusionXLModelLoader import (
    SDXL_LOAD_AUTO,
    SDXL_LORA_ROUTE_SAFETENSORS,
    StableDiffusionXLEmbeddingName,
    StableDiffusionXLModelHandle,
    StableDiffusionXLModelNames,
    StableDiffusionXLQuantizationConfig,
    StableDiffusionXLWeightDtypes,
    stable_diffusion_xl_lora_conversion_plan,
)
from serenity_trainer.modelLoader.stableDiffusionXL.StableDiffusionXLEmbeddingLoader import (
    StableDiffusionXLEmbeddingLoader,
)
from serenity_trainer.modelLoader.stableDiffusionXL.StableDiffusionXLLoRALoader import (
    StableDiffusionXLLoRALoader,
)
from serenity_trainer.modelLoader.StableDiffusionXLFineTuneModelLoader import (
    StableDiffusionXLFineTuneModelLoader,
)
from serenity_trainer.modelLoader.StableDiffusionXLLoRAModelLoader import (
    StableDiffusionXLLoRAModelLoader,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE,
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
)


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(": got ") + got + String(", expected ") + expected)


def _names() -> StableDiffusionXLModelNames:
    return StableDiffusionXLModelNames(
        String("/models/sdxl"),
        String("/models/sdxl-vae"),
        String("/models/sdxl-lora.safetensors"),
        StableDiffusionXLEmbeddingName(String("emb-1"), String("/models/sdxl-embedding.safetensors")),
    )


def main() raises:
    var names = _names()
    var dtypes = StableDiffusionXLWeightDtypes.bf16()
    var quantization = StableDiffusionXLQuantizationConfig.default_values()
    _expect_string("names base", names.base_model, String("/models/sdxl"))
    _expect_string("names vae", names.vae_model, String("/models/sdxl-vae"))
    _expect_string("names lora", names.lora, String("/models/sdxl-lora.safetensors"))
    _expect_string("weight train dtype", dtypes.train_dtype, String("BF16"))
    _expect_string("weight unet dtype", dtypes.unet, String("BF16"))
    _expect_string("weight te2 dtype", dtypes.text_encoder_2, String("BF16"))
    _expect_string("quant preset", quantization.layer_filter_preset, String("full"))
    _expect_int("quant svd rank", quantization.svd_rank, 16)

    var ft_loader = StableDiffusionXLFineTuneModelLoader()
    var load_plan = ft_loader.load(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE, names, dtypes, quantization)
    _expect_int("ft route", load_plan.route, SDXL_LOAD_AUTO)
    _expect_string("ft spec", load_plan.model_spec, String("resources/sd_model_spec/sd_xl_base_1.0.json"))
    _expect_string("ft sd config", load_plan.sd_config_name, String("resources/model_config/stable_diffusion_xl/sd_xl_base.yaml"))
    _expect_bool("ft internal first", load_plan.tries_internal_first, True)
    _expect_bool("ft diffusers second", load_plan.tries_diffusers_second, True)
    _expect_bool("ft safetensors third", load_plan.tries_safetensors_third, True)
    _expect_bool("ft single file", load_plan.single_file_supported, True)
    _expect_string("ft safetensors pipeline", load_plan.safetensors_single_file_pipeline_class, String("StableDiffusionXLPipeline.from_single_file"))
    _expect_string("ft tokenizer 1", load_plan.tokenizer_1_subfolder, String("tokenizer"))
    _expect_string("ft tokenizer 2", load_plan.tokenizer_2_subfolder, String("tokenizer_2"))
    _expect_string("ft scheduler", load_plan.scheduler_class, String("DDIMScheduler"))
    _expect_string("ft unet class", load_plan.unet_class, String("UNet2DConditionModel"))
    _expect_string("ft unet dtype", load_plan.unet_storage_dtype, String("BF16"))
    _expect_bool("ft vae override", load_plan.vae_override_supported, True)
    _expect_bool("ft unet quantization", load_plan.unet_quantization_supported, True)
    _expect_bool("ft preserves dtype", load_plan.preserves_storage_dtype_at_boundaries, True)

    var lora_loader = StableDiffusionXLLoRAModelLoader()
    var lora_load_plan = lora_loader.load(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING, names, dtypes, quantization)
    _expect_string("lora spec", lora_load_plan.model_spec, String("resources/sd_model_spec/sd_xl_base_1.0_inpainting-lora.json"))
    _expect_string("lora sd config", lora_load_plan.sd_config_name, String("resources/model_config/stable_diffusion_xl/sd_xl_base-inpainting.yaml"))
    _expect_string("lora path", lora_load_plan.lora_model, String("/models/sdxl-lora.safetensors"))
    _expect_bool("lora base loader", lora_load_plan.base_loader_invoked, True)
    _expect_bool("lora loader", lora_load_plan.lora_loader_invoked, True)
    _expect_bool("lora embedding loader", lora_load_plan.embedding_loader_invoked, True)
    _expect_bool("lora safetensors inpaint", lora_load_plan.inpainting_pipeline_for_safetensors, True)
    _expect_bool("lora preserves dtype", lora_load_plan.preserves_storage_dtype_at_boundaries, True)

    var direct_model = StableDiffusionXLModelHandle(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE)
    var direct_lora_loader = StableDiffusionXLLoRALoader()
    var direct_lora_plan = direct_lora_loader.load(direct_model, names)
    var direct_embedding_loader = StableDiffusionXLEmbeddingLoader()
    var direct_embedding_plan = direct_embedding_loader.load(direct_model, String("/models/sdxl"), names)
    _expect_int("direct lora route", direct_lora_plan.route, SDXL_LORA_ROUTE_SAFETENSORS)
    _expect_string("direct lora namespace", direct_lora_plan.converted_target_namespace, String("diffusers"))
    _expect_bool("direct lora convert keys", direct_lora_plan.has_convert_key_sets, True)
    _expect_bool("direct lora preserves dtype", direct_lora_plan.preserves_tensor_storage_dtype, True)
    _expect_bool("direct model lora loaded", direct_model.lora_loaded, True)
    _expect_bool("direct model embedding loaded", direct_model.embedding_loaded, True)
    _expect_string("direct embedding clip_l", direct_embedding_plan.key_clip_l, String("clip_l"))
    _expect_string("direct embedding clip_g", direct_embedding_plan.key_clip_g, String("clip_g"))
    _expect_bool("direct embedding preserves dtype", direct_embedding_plan.preserves_tensor_storage_dtype, True)

    var conversion = stable_diffusion_xl_lora_conversion_plan()
    _expect_bool("conversion key sets", conversion.has_convert_key_sets, True)
    _expect_string("conversion source namespaces", conversion.source_namespaces, String("omi,diffusers,legacy_diffusers"))
    _expect_string("conversion load target", conversion.load_target_namespace, String("diffusers"))
    _expect_string("conversion safetensors target", conversion.safetensors_save_target_namespace, String("legacy_diffusers"))
    _expect_string("conversion internal target", conversion.internal_save_target_namespace, String("omi"))
    _expect_string("conversion unet diffusers", conversion.unet_diffusers_prefix, String("lora_unet"))
    _expect_string("conversion clip_l diffusers", conversion.clip_l_diffusers_prefix, String("lora_te1"))
    _expect_string("conversion clip_g diffusers", conversion.clip_g_diffusers_prefix, String("lora_te2"))
    _expect_int("conversion range", conversion.range_upper_bound, 100)
    _expect_bool("conversion unet resnet rules", conversion.has_unet_resnet_rules, True)
    _expect_bool("conversion clip projection rules", conversion.has_clip_projection_rules, True)

    print("SDXL SURFACE LOADER CONTRACT OK")
