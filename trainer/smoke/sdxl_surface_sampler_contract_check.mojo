# SDXL sampler surface contract gate.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSampler/StableDiffusionXLSampler.py

from serenity_trainer.modelSampler.StableDiffusionXLSampler import (
    StableDiffusionXLSampleConfig,
    StableDiffusionXLSampler,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
)


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(": got ") + got + String(", expected ") + expected)


def _expect_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var diff = _abs(got - expected)
    if diff > tol:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected) + String(", |d| ") + String(diff))


def main() raises:
    var sampler = StableDiffusionXLSampler(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING)
    var sample_config = StableDiffusionXLSampleConfig(
        String("prompt"),
        String("negative"),
        1025,
        1025,
        123,
        False,
        4,
        Float32(7.5),
        0,
        True,
        String("/tmp/base.png"),
        String("/tmp/mask.png"),
        1,
        2,
        True,
    )
    var sample_plan = sampler.sample(sample_config, String("/tmp/sdxl.png"))
    _expect_int("sample file type", sample_plan.file_type, 0)
    _expect_string("sample destination", sample_plan.destination, String("/tmp/sdxl.png"))
    _expect_int("sample height", sample_plan.height, 1024)
    _expect_int("sample width", sample_plan.width, 1024)
    _expect_int("sample latent h", sample_plan.latent_h, 128)
    _expect_int("sample latent w", sample_plan.latent_w, 128)
    _expect_int("sample latent channels", sample_plan.latent_channels, 4)
    _expect_string("sample latent source", sample_plan.latent_channels_source, String("latent_conditioning_image.shape[1]"))
    _expect_int("sample unet input channels", sample_plan.unet_input_channels, 9)
    _expect_int("sample vae scale", sample_plan.vae_scale_factor, 8)
    _expect_int("sample cfg batch", sample_plan.batch_size, 2)
    _expect_int("sample seed", sample_plan.seed, 123)
    _expect_bool("sample random seed", sample_plan.random_seed, False)
    _expect_string("sample seed source", sample_plan.seed_source, String("torch.Generator.manual_seed(seed)"))
    _expect_close("sample cfg scale", sample_plan.cfg_scale, Float32(7.5), Float32(1e-6))
    _expect_close("sample cfg rescale", sample_plan.cfg_rescale, Float32(0.7), Float32(1e-6))
    _expect_bool("sample negative prompt", sample_plan.uses_negative_prompt, True)
    _expect_int("sample noise scheduler", sample_plan.noise_scheduler, 0)
    _expect_int("sample diffusion steps", sample_plan.diffusion_steps, 4)
    _expect_int("sample timestep min", sample_plan.timestep_count_min, 3)
    _expect_int("sample timestep max", sample_plan.timestep_count_max, 4)
    _expect_bool("sample force last", sample_plan.force_last_timestep, True)
    _expect_bool("sample inpaint model", sample_plan.inpainting_model_type, True)
    _expect_bool("sample inpaint", sample_plan.sample_inpainting, True)
    _expect_bool("sample prepares conditioning", sample_plan.prepares_conditioning_image, True)
    _expect_bool("sample erodes mask", sample_plan.erodes_mask_before_encoding, True)
    _expect_bool("sample appends mask conditioning", sample_plan.appends_mask_and_conditioning_latents, True)
    _expect_string("sample initial noise dtype", sample_plan.initial_noise_dtype, String("model.train_dtype.torch_dtype()"))
    _expect_string("sample latent dtype", sample_plan.latent_state_dtype, String("model.train_dtype.torch_dtype()"))
    _expect_string("sample prompt dtype", sample_plan.prompt_embedding_dtype, String("model.train_dtype.torch_dtype()"))
    _expect_bool("sample pooled text", sample_plan.pooled_text_embedding_used, True)
    _expect_bool("sample scheduler scales", sample_plan.scheduler_scales_model_input, True)
    _expect_bool("sample generator kwargs", sample_plan.extra_step_kwargs_may_include_generator, True)
    _expect_string("sample decode dtype", sample_plan.decode_input_dtype, String("model.vae_train_dtype.torch_dtype()"))
    _expect_string("sample decode formula", sample_plan.decode_formula, String("vae.decode(latent_image / vae.config.scaling_factor)"))
    _expect_string("sample output type", sample_plan.postprocess_output_type, String("pil"))
    _expect_int("sample te1 skip", sample_plan.text_encoder_1_layer_skip, 1)
    _expect_int("sample te2 skip", sample_plan.text_encoder_2_layer_skip, 2)

    print("SDXL SURFACE SAMPLER CONTRACT OK")
