# Flux2/Klein loader surface contract gate.
#
# Serenity references only:
#   modules/modelLoader/Flux2ModelLoader.py:38-207
#       internal -> diffusers -> safetensors load order; meta.json internal
#       probe; diffusers submodules; transformer/VAE overrides; unsupported
#       single-file Flux2 error.
#   modules/modelLoader/Flux2ModelLoader.py:100-128
#       num_attention_heads == 48 uses Pixtral/Mistral Dev branch; otherwise
#       Qwen2/Qwen3 Klein branch and relinks lm_head to embed_tokens.
#   modules/modelLoader/Flux2ModelLoader.py:217-246
#       Flux2LoRALoader returns None convert key sets; generated LoRA and
#       fine-tune wrappers use flux_2.0-lora.json and flux_2.0.json.
#   modules/model/Flux2Model.py:152-160,232-236
#       is_dev() is heads == 48, is_klein() is not is_dev(), and pipeline class
#       is Flux2Pipeline for Dev else Flux2KleinPipeline.
#
# This is build-only/source-contract coverage. It makes no numeric parity claim.

from serenity_trainer.modelLoader.Flux2ModelLoader import (
    FLUX2_LOAD_AUTO,
    FLUX2_LORA_ROUTE_MIXIN,
    Flux2FineTuneModelLoader,
    Flux2LoRALoader,
    Flux2LoRAModelLoader,
    Flux2ModelLoader,
    Flux2ModelNames,
    flux2_default_model_spec_name,
    flux2_internal_probe_file,
    flux2_internal_route_delegates_to_diffusers,
    flux2_is_dev_num_attention_heads,
    flux2_is_klein_num_attention_heads,
    flux2_klein_relinks_lm_head_to_embed_tokens,
    flux2_load_tries_diffusers_second,
    flux2_load_tries_internal_first,
    flux2_load_tries_safetensors_third,
    flux2_lora_default_model_spec_name,
    flux2_lora_loader_has_convert_key_sets,
    flux2_pipeline_class_for_heads,
    flux2_prepares_text_encoder_submodule_from_base,
    flux2_prepares_transformer_submodule_from_base,
    flux2_prepares_vae_submodule_from_base,
    flux2_preserves_storage_dtype_at_boundaries,
    flux2_safetensors_error,
    flux2_scheduler_class,
    flux2_scheduler_subfolder,
    flux2_single_file_supported,
    flux2_text_encoder_class_for_heads,
    flux2_text_encoder_subfolder,
    flux2_text_encoder_uses_fallback_train_dtype,
    flux2_tokenizer_class_for_heads,
    flux2_tokenizer_subfolder,
    flux2_transformer_class,
    flux2_transformer_override_avoids_float32_load,
    flux2_transformer_override_config_source,
    flux2_transformer_override_default_torch_dtype,
    flux2_transformer_override_from_single_file,
    flux2_transformer_override_gguf_compute_dtype,
    flux2_transformer_override_supported,
    flux2_transformer_quantization_supported,
    flux2_transformer_subfolder,
    flux2_vae_class,
    flux2_vae_override_from_model_name,
    flux2_vae_override_supported,
    flux2_vae_subfolder,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_FLUX_2


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(" got=") + String(got)
            + String(" expected=") + String(expected)
        )


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(
            name + String(" got=") + String(got)
            + String(" expected=") + String(expected)
        )


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(" got=") + got + String(" expected=") + expected)


def main() raises:
    _expect_string(
        "ft spec",
        flux2_default_model_spec_name(MODEL_TYPE_FLUX_2),
        String("resources/sd_model_spec/flux_2.0.json"),
    )
    _expect_string(
        "lora spec",
        flux2_lora_default_model_spec_name(MODEL_TYPE_FLUX_2),
        String("resources/sd_model_spec/flux_2.0-lora.json"),
    )

    var ft_loader = Flux2FineTuneModelLoader()
    var ft_contract = ft_loader.contract()
    _expect_string("ft factory", ft_contract.factory_name, String("make_fine_tune_model_loader"))
    _expect_string("ft model class", ft_contract.model_class, String("Flux2Model"))
    _expect_string("ft model loader", ft_contract.model_loader_class, String("Flux2ModelLoader"))
    _expect_bool("ft has lora loader", ft_contract.has_lora_loader(), False)
    _expect_bool("ft has embedding loader", ft_contract.has_embedding_loader(), False)

    var lora_wrapper = Flux2LoRAModelLoader()
    var lora_contract = lora_wrapper.contract()
    _expect_string("lora factory", lora_contract.factory_name, String("make_lora_model_loader"))
    _expect_string("lora model class", lora_contract.model_class, String("Flux2Model"))
    _expect_string("lora base loader", lora_contract.model_loader_class, String("Flux2ModelLoader"))
    _expect_string("lora leaf loader", lora_contract.lora_loader_class, String("Flux2LoRALoader"))
    _expect_bool("lora has lora loader", lora_contract.has_lora_loader(), True)
    _expect_bool("lora has embedding loader", lora_contract.has_embedding_loader(), False)

    var base_loader = Flux2ModelLoader()
    _expect_int("load route", base_loader.route(), FLUX2_LOAD_AUTO)
    _expect_bool("internal first", flux2_load_tries_internal_first(), True)
    _expect_bool("diffusers second", flux2_load_tries_diffusers_second(), True)
    _expect_bool("safetensors third", flux2_load_tries_safetensors_third(), True)
    _expect_bool("internal delegates", flux2_internal_route_delegates_to_diffusers(), True)
    _expect_string("internal probe", flux2_internal_probe_file(), String("meta.json"))
    _expect_bool("single file unsupported", flux2_single_file_supported(), False)
    _expect_string(
        "single file error",
        flux2_safetensors_error(),
        String("Loading of single file Flux2 models not supported. Use the diffusers model instead. Optionally, transformer-only safetensor files can be loaded by overriding the transformer."),
    )

    _expect_string("tokenizer subfolder", flux2_tokenizer_subfolder(), String("tokenizer"))
    _expect_string("text encoder subfolder", flux2_text_encoder_subfolder(), String("text_encoder"))
    _expect_string("scheduler subfolder", flux2_scheduler_subfolder(), String("scheduler"))
    _expect_string("transformer subfolder", flux2_transformer_subfolder(), String("transformer"))
    _expect_string("vae subfolder", flux2_vae_subfolder(), String("vae"))
    _expect_string("transformer class", flux2_transformer_class(), String("Flux2Transformer2DModel"))
    _expect_string("scheduler class", flux2_scheduler_class(), String("FlowMatchEulerDiscreteScheduler"))
    _expect_string("vae class", flux2_vae_class(), String("AutoencoderKLFlux2"))

    var names = Flux2ModelNames(
        String("/models/flux2"),
        String("/models/flux2-transformer.safetensors"),
        String("/models/flux2-vae"),
        String("/models/flux2-lora.safetensors"),
    )
    _expect_bool("override skips transformer prep", flux2_prepares_transformer_submodule_from_base(names), False)
    _expect_bool("override skips vae prep", flux2_prepares_vae_submodule_from_base(names), False)
    _expect_bool("always prepares text encoder", flux2_prepares_text_encoder_submodule_from_base(), True)
    _expect_bool("transformer override supported", flux2_transformer_override_supported(), True)
    _expect_bool("transformer override single file", flux2_transformer_override_from_single_file(names), True)
    _expect_string("override config", flux2_transformer_override_config_source(), String("base_model"))
    _expect_string("override default dtype", flux2_transformer_override_default_torch_dtype(), String("BF16"))
    _expect_string("override gguf compute", flux2_transformer_override_gguf_compute_dtype(), String("BF16"))
    _expect_bool("override avoids f32", flux2_transformer_override_avoids_float32_load(), True)
    _expect_bool("transformer quantization", flux2_transformer_quantization_supported(), True)
    _expect_bool("vae override", flux2_vae_override_supported(), True)
    _expect_bool("vae override model name", flux2_vae_override_from_model_name(names), True)
    _expect_bool("text encoder fallback dtype", flux2_text_encoder_uses_fallback_train_dtype(), True)
    _expect_bool("storage dtype preserved", flux2_preserves_storage_dtype_at_boundaries(), True)

    var base_names = Flux2ModelNames(String("/models/flux2"), String(), String(), String())
    _expect_bool("base prepares transformer", flux2_prepares_transformer_submodule_from_base(base_names), True)
    _expect_bool("base prepares vae", flux2_prepares_vae_submodule_from_base(base_names), True)
    _expect_bool("base no transformer override", flux2_transformer_override_from_single_file(base_names), False)
    _expect_bool("base no vae override", flux2_vae_override_from_model_name(base_names), False)

    _expect_bool("dev heads", flux2_is_dev_num_attention_heads(48), True)
    _expect_bool("klein heads", flux2_is_klein_num_attention_heads(24), True)
    _expect_bool("48 not klein", flux2_is_klein_num_attention_heads(48), False)
    _expect_string("dev tokenizer", flux2_tokenizer_class_for_heads(48), String("PixtralProcessor.tokenizer"))
    _expect_string("dev text encoder", flux2_text_encoder_class_for_heads(48), String("Mistral3ForConditionalGeneration"))
    _expect_string("dev pipeline", flux2_pipeline_class_for_heads(48), String("Flux2Pipeline"))
    _expect_string("klein tokenizer", flux2_tokenizer_class_for_heads(24), String("Qwen2Tokenizer"))
    _expect_string("klein text encoder", flux2_text_encoder_class_for_heads(24), String("Qwen3ForCausalLM"))
    _expect_string("klein pipeline", flux2_pipeline_class_for_heads(24), String("Flux2KleinPipeline"))
    _expect_bool("klein tied weight relink", flux2_klein_relinks_lm_head_to_embed_tokens(), True)

    _expect_bool("lora convert key sets", flux2_lora_loader_has_convert_key_sets(), False)
    var lora_loader = Flux2LoRALoader()
    var lora_plan = lora_loader.load(names)
    _expect_int("lora route", lora_plan.route, FLUX2_LORA_ROUTE_MIXIN)
    _expect_bool("lora delegates", lora_plan.delegates_to_lora_mixin, True)
    _expect_bool("lora has convert keys", lora_plan.has_convert_key_sets, False)
    _expect_string("lora convert keys name", lora_plan.convert_key_sets_name, String("None"))
    _expect_bool("wrapper lora invoked", lora_wrapper.lora_loader_invoked(names), True)
    _expect_bool("wrapper embedding absent", lora_wrapper.embedding_loader_invoked(), False)

    print("FLUX2 SURFACE LOADER CONTRACT OK")
