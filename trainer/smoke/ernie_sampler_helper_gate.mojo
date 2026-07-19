# Ernie sampler helper parity gate.
#
# This is intentionally bounded to deterministic helper math mirrored from
# Serenity ErnieSampler.py and ErnieModel.py. It does not run tokenizers, text
# encoders, transformer inference, random noise, scheduler tensor stepping, VAE
# decode, postprocess, or image saving, and is not end-to-end sampler parity.

from serenity_trainer.modelSampler.ErnieSampler import (
    ErnieSampleConfig,
    ernie_cfg_batch_size,
    ernie_cfg_combine_value,
    ernie_euler_update_value,
    ernie_latent_contract_for_image,
    ernie_make_schedule,
    ernie_quantize_resolution,
    ernie_sample_plan,
    ernie_use_cfg,
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


def main() raises:
    var sample_config = ErnieSampleConfig(
        String("prompt"),
        String("negative"),
        1025,
        1000,
        123,
        False,
        4,
        Float32(4.0),
        0,
    )
    var plan = ernie_sample_plan(sample_config, String("/tmp/ernie-helper.png"))

    _expect_int("quantize 1025", ernie_quantize_resolution(1025, 64), 1024)
    _expect_int("quantize 1000", ernie_quantize_resolution(1000, 64), 1024)
    _expect_int("quantize half-even 1056", ernie_quantize_resolution(1056, 64), 1024)
    _expect_int("plan height", plan.height, 1024)
    _expect_int("plan width", plan.width, 1024)
    _expect_int("latent height", plan.latent_h, 128)
    _expect_int("latent width", plan.latent_w, 128)
    _expect_int("latent channels", plan.latent_channels, 32)
    _expect_bool("patchified latents", plan.patchified_latents, True)
    _expect_int("cfg batch", plan.batch_size, 2)
    _expect_bool("uses negative prompt", plan.uses_negative_prompt, True)
    _expect_bool("cfg path off at 1", ernie_use_cfg(Float32(1.0)), False)
    _expect_bool("cfg path on above 1", ernie_use_cfg(Float32(1.0001)), True)
    _expect_int("cfg batch off", ernie_cfg_batch_size(Float32(1.0)), 1)
    _expect_int("cfg batch on", ernie_cfg_batch_size(Float32(4.0)), 2)
    _expect_close(
        "cfg combine",
        ernie_cfg_combine_value(Float32(3.0), Float32(1.0), Float32(2.0)),
        Float32(5.0),
        Float32(1e-6),
    )

    var contract = ernie_latent_contract_for_image(1024, 512, 2)
    _expect_int("contract batch", contract.batch_size, 2)
    _expect_int("contract latent channels", contract.latent_channels, 32)
    _expect_int("contract latent h", contract.latent_h, 128)
    _expect_int("contract latent w", contract.latent_w, 64)
    _expect_int("contract patch channels", contract.patchified_channels, 128)
    _expect_int("contract patch h", contract.patchified_h, 64)
    _expect_int("contract patch w", contract.patchified_w, 32)
    _expect_int("contract patch seq", contract.patchified_seq_len, 2048)

    var schedule = ernie_make_schedule(4)
    _expect_int("schedule timesteps", len(schedule.timesteps), 4)
    _expect_int("schedule sigmas", len(schedule.sigmas), 5)
    _expect_close("sigma[0]", schedule.sigmas[0], Float32(1.0), Float32(1e-6))
    _expect_close("sigma[1]", schedule.sigmas[1], Float32(0.75), Float32(1e-6))
    _expect_close("sigma[2]", schedule.sigmas[2], Float32(0.5), Float32(1e-6))
    _expect_close("sigma[3]", schedule.sigmas[3], Float32(0.25), Float32(1e-6))
    _expect_close("sigma terminal", schedule.sigmas[4], Float32(0.0), Float32(1e-6))
    _expect_close("timestep[0]", schedule.timesteps[0], Float32(1000.0), Float32(1e-4))
    _expect_close("timestep[1]", schedule.timesteps[1], Float32(750.0), Float32(1e-4))
    _expect_close("timestep[3]", schedule.timesteps[3], Float32(250.0), Float32(1e-4))

    var updated = ernie_euler_update_value(
        Float32(0.25),
        Float32(0.5),
        schedule.sigmas[0],
        schedule.sigmas[1],
    )
    _expect_close("euler update", updated, Float32(0.125), Float32(1e-6))

    print("ERNIE SAMPLER HELPER GATE OK")
    print(
        "plan =", plan.height, "x", plan.width,
        " latent =", plan.latent_h, "x", plan.latent_w,
        " patch =", contract.patchified_h, "x", contract.patchified_w,
        "x", contract.patchified_channels,
        " cfg_batch =", plan.batch_size,
    )
    print(
        "sigma1 =", schedule.sigmas[1],
        " timestep1 =", schedule.timesteps[1],
        " euler =", updated,
    )
