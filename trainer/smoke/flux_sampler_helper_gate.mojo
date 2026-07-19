# FLUX.1 sampler helper parity gate.
#
# This is bounded to deterministic helper math mirrored from Serenity
# FluxSampler.py and FluxModel.py. It does not run text encoders, transformer
# inference, random noise, scheduler stepping, VAE decode, or image saving, and
# is not end-to-end sampler parity.

from serenity_trainer.model.FluxModel import (
    FluxSchedulerShiftConfig,
    calculate_timestep_shift,
    flux_scheduler_mu_from_shift,
    flux_transformer_timestep_input,
)
from serenity_trainer.modelSampler.FluxSampler import (
    FluxSampleConfig,
    flux_quantize_resolution,
    flux_sample_plan,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_FLUX_DEV_1,
    MODEL_TYPE_FLUX_FILL_DEV_1,
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
        raise Error(name + String(": unexpected bool"))


def _expect_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var diff = _abs(got - expected)
    if diff > tol:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
            + String(", |d| ") + String(diff)
        )


def _config() -> FluxSampleConfig:
    return FluxSampleConfig(
        String("prompt"),
        String("negative"),
        1025,
        1025,
        7,
        False,
        30,
        Float32(3.5),
        0,
        False,
        String(),
        String(),
    )


def main() raises:
    var base_plan = flux_sample_plan(MODEL_TYPE_FLUX_DEV_1, _config(), String("/tmp/flux.png"))

    _expect_int("quantize 1025", flux_quantize_resolution(1025, 64), 1024)
    _expect_int("quantize half-even 1056", flux_quantize_resolution(1056, 64), 1024)
    _expect_int("height", base_plan.height, 1024)
    _expect_int("width", base_plan.width, 1024)
    _expect_int("latent h", base_plan.latent_h, 128)
    _expect_int("latent w", base_plan.latent_w, 128)
    _expect_int("packed h", base_plan.packed_h, 64)
    _expect_int("packed w", base_plan.packed_w, 64)
    _expect_int("latent channels", base_plan.latent_channels, 16)
    _expect_int("packed channels", base_plan.packed_latent_channels, 64)
    _expect_int("batch size", base_plan.batch_size, 1)
    _expect_bool("negative prompt unused", base_plan.negative_prompt_used, False)
    _expect_bool("cfg does not duplicate latents", base_plan.cfg_batch_uses_negative_prompt, False)
    _expect_bool("base not inpainting", base_plan.inpainting_model_type, False)
    _expect_bool("base no conditioning", base_plan.appends_conditioning_latents_and_mask, False)
    _expect_bool("packs before denoise", base_plan.packs_latents_before_denoise, True)
    _expect_bool("unpacks before decode", base_plan.unpacks_latents_before_decode, True)

    var fill_config = _config()
    fill_config.sample_inpainting = True
    fill_config.base_image_path = String("/tmp/base.png")
    fill_config.mask_image_path = String("/tmp/mask.png")
    var fill_plan = flux_sample_plan(MODEL_TYPE_FLUX_FILL_DEV_1, fill_config, String("/tmp/fill.png"))
    _expect_bool("fill model", fill_plan.inpainting_model_type, True)
    _expect_bool("fill sample inpainting", fill_plan.sample_inpainting, True)
    _expect_bool("fill conditioning", fill_plan.appends_conditioning_latents_and_mask, True)
    _expect_int("fill mask channels", fill_plan.fill_mask_channels, 256)
    _expect_int("fill concat dim", fill_plan.fill_conditioning_concat_dim, -1)

    var shift_cfg = FluxSchedulerShiftConfig(256, 4096, Float32(0.5), Float32(1.15))
    var shift = calculate_timestep_shift(base_plan.latent_h, base_plan.latent_w, shift_cfg)
    var mu = flux_scheduler_mu_from_shift(shift)
    var model_t = flux_transformer_timestep_input(Float32(500.0))
    _expect_close("shift", shift, Float32(3.1581929), Float32(2e-5))
    _expect_close("mu", mu, Float32(1.15), Float32(2e-5))
    _expect_close("model_t", model_t, Float32(0.5), Float32(1e-6))

    print("FLUX SAMPLER HELPER GATE OK")
    print(
        "plan =", base_plan.height, "x", base_plan.width,
        " latent =", base_plan.latent_h, "x", base_plan.latent_w,
        " packed =", base_plan.packed_h * base_plan.packed_w,
        "x", base_plan.packed_latent_channels,
    )
    print(
        "shift =", shift,
        " mu =", mu,
        " model_t =", model_t,
        " fill_mask_channels =", fill_plan.fill_mask_channels,
    )
