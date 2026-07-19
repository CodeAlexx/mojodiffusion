# SD3/SD3.5 loader/saver/sampler surface smoke.
#
# This is not a parity gate and does not generate an image. It only instantiates
# build-only SD3 plan surfaces and create/factory dispatch records.

from serenity_trainer.modelLoader.stableDiffusion3.StableDiffusion3ModelLoader import (
    SD3_LOAD_AUTO,
    StableDiffusion3EmbeddingName,
    StableDiffusion3ModelNames,
    StableDiffusion3QuantizationConfig,
    StableDiffusion3WeightDtypes,
)
from serenity_trainer.modelLoader.stableDiffusion3.StableDiffusion3LoRALoader import (
    stable_diffusion3_lora_loader_has_convert_key_sets,
)
from serenity_trainer.modelLoader.StableDiffusion3EmbeddingModelLoader import StableDiffusion3EmbeddingModelLoader
from serenity_trainer.modelLoader.StableDiffusion3FineTuneModelLoader import StableDiffusion3FineTuneModelLoader
from serenity_trainer.modelLoader.StableDiffusion3LoRAModelLoader import StableDiffusion3LoRAModelLoader
from serenity_trainer.modelSampler.StableDiffusion3Sampler import (
    StableDiffusion3SampleConfig,
    StableDiffusion3Sampler,
)
from serenity_trainer.modelSaver.StableDiffusion3FineTuneModelSaver import StableDiffusion3FineTuneModelSaver
from serenity_trainer.modelSaver.StableDiffusion3EmbeddingModelSaver import StableDiffusion3EmbeddingModelSaver
from serenity_trainer.modelSaver.StableDiffusion3LoRAModelSaver import StableDiffusion3LoRAModelSaver
from serenity_trainer.modelSaver.stableDiffusion3.StableDiffusion3LoRASaver import (
    SD3_FMT_INTERNAL,
    SD3_FMT_SAFETENSORS,
    stable_diffusion3_lora_bundle_embedding_keys,
)
from serenity_trainer.util.create import create_model_loader, create_model_sampler, create_model_saver
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_3,
    MODEL_TYPE_STABLE_DIFFUSION_35,
)
from serenity_trainer.util.enum.TrainingMethod import TM_EMBEDDING, TM_FINE_TUNE, TM_LORA


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(": got ") + got + String(", expected ") + expected)


def main() raises:
    var names = StableDiffusion3ModelNames(
        String("/models/sd3"),
        String(),
        String("/models/sd3-lora.safetensors"),
        StableDiffusion3EmbeddingName.empty(),
        True,
        True,
        True,
    )
    var dtypes = StableDiffusion3WeightDtypes.bf16()
    var quantization = StableDiffusion3QuantizationConfig.default_values()
    _expect_string("names base", names.base_model, String("/models/sd3"))
    _expect_bool("names include te1", names.include_text_encoder_1, True)
    _expect_bool("names include te2", names.include_text_encoder_2, True)
    _expect_bool("names include te3", names.include_text_encoder_3, True)
    _expect_string("weight transformer dtype", dtypes.transformer, String("BF16"))
    _expect_string("weight t5 dtype", dtypes.text_encoder_3, String("BF16"))
    _expect_string("quant preset", quantization.layer_filter_preset, String("full"))

    var ft_loader = StableDiffusion3FineTuneModelLoader()
    var ft_plan = ft_loader.load(MODEL_TYPE_STABLE_DIFFUSION_3, names, dtypes, quantization)
    _expect_int("ft route", ft_plan.route, SD3_LOAD_AUTO)
    _expect_string("ft spec", ft_plan.model_spec, String("resources/sd_model_spec/sd_3_2b_1.0.json"))
    _expect_string("ft scheduler", ft_plan.scheduler_class, String("FlowMatchEulerDiscreteScheduler"))
    _expect_string("ft transformer", ft_plan.transformer_class, String("SD3Transformer2DModel"))
    _expect_bool("ft base loader", ft_plan.base_loader_invoked, True)
    _expect_bool("ft lora loader", ft_plan.lora_loader_invoked, False)
    _expect_bool("ft embedding loader", ft_plan.embedding_loader_invoked, True)
    _expect_bool("ft preserves dtype", ft_plan.preserves_storage_dtype_at_boundaries, True)
    _expect_bool("ft t5 fallback dtype", ft_plan.text_encoder_3_uses_fallback_train_dtype, True)
    print("sd3 ft loader =", ft_plan.model_spec, " transformer =", ft_plan.transformer_class)

    var lora_loader = StableDiffusion3LoRAModelLoader()
    var lora_plan = lora_loader.load(MODEL_TYPE_STABLE_DIFFUSION_35, names, dtypes, quantization)
    _expect_string("sd35 lora spec", lora_plan.model_spec, String("resources/sd_model_spec/sd_3.5_1.0-lora.json"))
    _expect_string("sd35 lora path", lora_plan.lora_model, String("/models/sd3-lora.safetensors"))
    _expect_bool("sd35 lora base loader", lora_plan.base_loader_invoked, True)
    _expect_bool("sd35 lora loader", lora_plan.lora_loader_invoked, True)
    _expect_bool("sd35 lora embedding loader", lora_plan.embedding_loader_invoked, True)
    print("sd35 lora loader =", lora_plan.model_spec, " lora invoked =", lora_plan.lora_loader_invoked)

    var emb_loader = StableDiffusion3EmbeddingModelLoader()
    var emb_plan = emb_loader.load(MODEL_TYPE_STABLE_DIFFUSION_3, names, dtypes, quantization)
    _expect_string("sd3 embedding spec", emb_plan.model_spec, String("resources/sd_model_spec/sd_3_2b_1.0-embedding.json"))
    _expect_bool("sd3 embedding base loader", emb_plan.base_loader_invoked, True)
    _expect_bool("sd3 embedding loader", emb_plan.embedding_loader_invoked, True)
    _expect_bool("sd3 lora convert available", stable_diffusion3_lora_loader_has_convert_key_sets(), True)
    print("sd3 embedding loader =", emb_plan.model_spec, " embedding invoked =", emb_plan.embedding_loader_invoked)
    print("sd3 lora convert keys =", stable_diffusion3_lora_loader_has_convert_key_sets())

    var sampler = StableDiffusion3Sampler(MODEL_TYPE_STABLE_DIFFUSION_3)
    var sample_config = StableDiffusion3SampleConfig(
        String("prompt"),
        String("negative"),
        1025,
        1025,
        1,
        False,
        4,
        Float32(4.0),
        0,
    )
    var sample_plan = sampler.sample(sample_config, String("/tmp/sd3.png"))
    _expect_int("sample height", sample_plan.height, 1024)
    _expect_int("sample width", sample_plan.width, 1024)
    _expect_int("sample latent h", sample_plan.latent_h, 128)
    _expect_int("sample latent w", sample_plan.latent_w, 128)
    _expect_int("sample latent channels", sample_plan.latent_channels, 16)
    _expect_int("sample cfg batch", sample_plan.batch_size, 2)
    _expect_bool("sample negative prompt", sample_plan.always_uses_negative_prompt, True)
    _expect_bool("sample unscaled latents", sample_plan.scales_latents_before_transformer, False)
    _expect_string("sample decode formula", sample_plan.decode_formula, String("(latent_image / vae.config.scaling_factor) + vae.config.shift_factor"))
    print("sd3 sample =", sample_plan.height, "x", sample_plan.width, " batch =", sample_plan.batch_size)

    var ft_saver = StableDiffusion3FineTuneModelSaver()
    var save_plan = ft_saver.save_plan(
        MODEL_TYPE_STABLE_DIFFUSION_35,
        SD3_FMT_SAFETENSORS,
        String("/tmp/sd35.safetensors"),
        String("BF16"),
    )
    _expect_string("sd35 saver route", save_plan.route_name, String("original_safetensors_checkpoint"))
    _expect_string("sd35 saver converter", save_plan.converter_name, String("convert_sd3_diffusers_to_ckpt"))
    _expect_string("sd35 saver dtype", save_plan.dtype_override, String("BF16"))
    _expect_bool("sd35 saver ckpt", save_plan.saves_original_safetensors_checkpoint, True)
    _expect_bool("sd35 saver converter enabled", save_plan.uses_diffusers_to_ckpt_converter, True)
    _expect_bool("sd35 saver deep copy", save_plan.deep_copy_pipeline_when_dtype_override, True)
    print("sd35 saver =", save_plan.route_name, " converter =", save_plan.converter_name)

    var lora_saver = StableDiffusion3LoRAModelSaver()
    var lora_save_plan = lora_saver.save_plan(
        MODEL_TYPE_STABLE_DIFFUSION_3,
        SD3_FMT_INTERNAL,
        String("/tmp/sd3-internal"),
    )
    _expect_string("sd3 lora saver route", lora_save_plan.route_name, String("internal_lora"))
    _expect_string("sd3 lora saver namespace", lora_save_plan.target_key_namespace, String("omi"))
    _expect_string("sd3 lora saver internal destination", lora_save_plan.internal_destination, String("/tmp/sd3-internal/lora/lora.safetensors"))
    _expect_bool("sd3 lora saver writes safetensors", lora_save_plan.writes_safetensors, True)
    _expect_bool("sd3 lora saver convert keys", lora_save_plan.has_convert_key_sets, True)
    print("sd3 lora saver =", lora_save_plan.route_name, " keys =", lora_save_plan.target_key_namespace)

    var emb_saver = StableDiffusion3EmbeddingModelSaver()
    var emb_save_plan = emb_saver.save_multiple_plan(
        MODEL_TYPE_STABLE_DIFFUSION_3,
        SD3_FMT_SAFETENSORS,
        String("/tmp/sd3-emb.safetensors"),
    )
    _expect_string("sd3 embedding saver route", emb_save_plan.route_name, String("embedding_safetensors"))
    _expect_bool("sd3 embedding saver multiple", emb_save_plan.is_multiple, True)
    _expect_string("sd3 embedding key clip_l", emb_save_plan.key_clip_l, String("clip_l"))
    _expect_string("sd3 embedding key t5_out", emb_save_plan.key_t5_out, String("t5_out"))
    print("sd3 embedding saver =", emb_save_plan.route_name, " multiple =", emb_save_plan.is_multiple)

    var bundle_keys = stable_diffusion3_lora_bundle_embedding_keys(String("tok"))
    _expect_int("bundle key count", len(bundle_keys), 6)
    _expect_string("bundle key clip_l", bundle_keys[0], String("bundle_emb.tok.clip_l"))
    _expect_string("bundle key t5_out", bundle_keys[5], String("bundle_emb.tok.t5_out"))
    print("sd3 bundle embedding keys =", len(bundle_keys))

    var reg_loader = create_model_loader(MODEL_TYPE_STABLE_DIFFUSION_35, TM_LORA)
    var reg_saver = create_model_saver(MODEL_TYPE_STABLE_DIFFUSION_3, TM_FINE_TUNE)
    var reg_sampler = create_model_sampler(MODEL_TYPE_STABLE_DIFFUSION_35, TM_FINE_TUNE)
    var reg_embedding = create_model_loader(MODEL_TYPE_STABLE_DIFFUSION_3, TM_EMBEDDING)
    _expect_string("factory lora loader", reg_loader.implementation, String("StableDiffusion3LoRAModelLoader"))
    _expect_string("factory ft saver", reg_saver.implementation, String("StableDiffusion3FineTuneModelSaver"))
    _expect_string("factory sampler", reg_sampler.implementation, String("StableDiffusion3Sampler"))
    _expect_int("factory sampler training method", reg_sampler.training_method, -1)
    _expect_string("factory embedding loader", reg_embedding.implementation, String("StableDiffusion3EmbeddingModelLoader"))
    print(reg_loader.implementation, reg_saver.implementation, reg_sampler.implementation, reg_embedding.implementation)
    print("SD3 SURFACE CHECK OK")
