# SDXL saver surface contract gate.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSaver/StableDiffusionXL*.py
#   /home/alex/Serenity/modules/modelSaver/stableDiffusionXL/*.py

from serenity_trainer.modelSaver.StableDiffusionXLEmbeddingModelSaver import (
    StableDiffusionXLEmbeddingModelSaver,
)
from serenity_trainer.modelSaver.StableDiffusionXLFineTuneModelSaver import (
    StableDiffusionXLFineTuneModelSaver,
)
from serenity_trainer.modelSaver.StableDiffusionXLLoRAModelSaver import (
    StableDiffusionXLLoRAModelSaver,
)
from serenity_trainer.modelSaver.stableDiffusionXL.StableDiffusionXLEmbeddingSaver import (
    stable_diffusion_xl_embedding_keys,
)
from serenity_trainer.modelSaver.stableDiffusionXL.StableDiffusionXLLoRASaver import (
    SDXL_FMT_INTERNAL,
    SDXL_FMT_SAFETENSORS,
    stable_diffusion_xl_lora_bundle_embedding_keys,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE,
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


def main() raises:
    var ft_saver = StableDiffusionXLFineTuneModelSaver()
    var save_plan = ft_saver.save_plan(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE, SDXL_FMT_SAFETENSORS, String("/tmp/sdxl.safetensors"), String("BF16"))
    _expect_string("model saver route", save_plan.route_name, String("original_safetensors_checkpoint"))
    _expect_string("model saver converter", save_plan.converter_name, String("convert_sdxl_diffusers_to_ckpt"))
    _expect_string("model saver dtype", save_plan.dtype_override, String("BF16"))
    _expect_bool("model saver diffusers", save_plan.saves_diffusers_pipeline, False)
    _expect_bool("model saver safetensors", save_plan.saves_original_safetensors_checkpoint, True)
    _expect_bool("model saver converter enabled", save_plan.uses_diffusers_to_ckpt_converter, True)
    _expect_bool("model saver deep copy", save_plan.deep_copy_pipeline_when_dtype_override, True)
    _expect_bool("model saver yaml", save_plan.writes_yaml_config_sidecar, True)
    _expect_bool("model saver vae state", save_plan.includes_vae_state_dict, True)
    _expect_bool("model saver unet state", save_plan.includes_unet_state_dict, True)
    _expect_bool("model saver te1 state", save_plan.includes_text_encoder_1_state_dict, True)
    _expect_bool("model saver te2 state", save_plan.includes_text_encoder_2_state_dict, True)

    var lora_saver = StableDiffusionXLLoRAModelSaver()
    var lora_save_plan = lora_saver.save_plan(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE, SDXL_FMT_INTERNAL, String("/tmp/sdxl-internal"))
    _expect_string("lora saver route", lora_save_plan.route_name, String("internal_lora"))
    _expect_string("lora saver namespace", lora_save_plan.target_key_namespace, String("omi"))
    _expect_string("lora saver internal destination", lora_save_plan.internal_destination, String("/tmp/sdxl-internal/lora/lora.safetensors"))
    _expect_bool("lora saver writes safetensors", lora_save_plan.writes_safetensors, True)
    _expect_bool("lora saver convert keys", lora_save_plan.has_convert_key_sets, True)
    _expect_bool("lora saver te1", lora_save_plan.includes_text_encoder_1_lora, True)
    _expect_bool("lora saver te2", lora_save_plan.includes_text_encoder_2_lora, True)
    _expect_bool("lora saver unet", lora_save_plan.includes_unet_lora, True)
    _expect_bool("lora saver preloaded state", lora_save_plan.includes_preloaded_lora_state_dict, True)
    _expect_bool("lora saver bundle embeddings", lora_save_plan.can_bundle_additional_embeddings, True)
    _expect_bool("lora saver preserves without override", lora_save_plan.preserves_storage_dtype_without_override, True)

    var embedding_saver = StableDiffusionXLEmbeddingModelSaver()
    var emb_plan = embedding_saver.save_multiple_plan(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE, SDXL_FMT_SAFETENSORS, String("/tmp/sdxl"))
    var emb_keys = stable_diffusion_xl_embedding_keys()
    var bundle_keys = stable_diffusion_xl_lora_bundle_embedding_keys(String("token"))
    _expect_string("embedding saver route", emb_plan.route_name, String("embedding_safetensors"))
    _expect_bool("embedding saver multiple", emb_plan.is_multiple, True)
    _expect_string("embedding saver clip_l", emb_plan.key_clip_l, String("clip_l"))
    _expect_string("embedding saver clip_g", emb_plan.key_clip_g, String("clip_g"))
    _expect_bool("embedding saver diffusers unsupported", emb_plan.diffusers_supported, False)
    _expect_bool("embedding saver preserves without override", emb_plan.preserves_storage_dtype_without_override, True)
    _expect_int("embedding key count", len(emb_keys), 4)
    _expect_string("embedding key clip_l", emb_keys[0], String("clip_l"))
    _expect_string("embedding key clip_g", emb_keys[1], String("clip_g"))
    _expect_int("bundle key count", len(bundle_keys), 4)
    _expect_string("bundle key clip_l", bundle_keys[0], String("bundle_emb.token.clip_l"))
    _expect_string("bundle key clip_g_out", bundle_keys[3], String("bundle_emb.token.clip_g_out"))

    print("SDXL SURFACE SAVER CONTRACT OK")
