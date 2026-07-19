# Qwen sampler helper parity gate.
#
# This is intentionally bounded to deterministic helper math mirrored from
# Serenity QwenSampler.py, QwenModel.py, and BaseQwenSetup.py. It does not
# run text encoding, transformer inference, random noise, VAE decode, or image
# postprocess, and it is not an end-to-end image parity claim.

from serenity_trainer.modelSampler.QwenSampler import (
    QwenSampleConfig,
    QwenSamplerSchedulerConfig,
    qwen_batch_size,
    qwen_cfg_combine_value,
    qwen_euler_update_value,
    qwen_latent_contract_for_image,
    qwen_make_schedule,
    qwen_sample_plan,
    qwen_use_cfg,
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
    var sample_config = QwenSampleConfig(
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
    var plan = qwen_sample_plan(sample_config, String("/tmp/qwen-helper.png"))

    _expect_int("quantized height", plan.height, 1024)
    _expect_int("quantized width", plan.width, 1024)
    _expect_int("latent height", plan.latent_h, 128)
    _expect_int("latent width", plan.latent_w, 128)
    _expect_int("latent channels", plan.latent_channels, 16)
    _expect_int("packed seq len", plan.packed_seq_len, 4096)
    _expect_int("packed channels", plan.packed_channels, 64)
    _expect_int("img_shapes frame", plan.img_shape_frame, 1)
    _expect_int("img_shapes height", plan.img_shape_h, 64)
    _expect_int("img_shapes width", plan.img_shape_w, 64)

    _expect_bool("cfg path off at 1", qwen_use_cfg(Float32(1.0)), False)
    _expect_bool("cfg path on above 1", qwen_use_cfg(Float32(1.0001)), True)
    _expect_int("cfg batch", qwen_batch_size(Float32(4.0)), 2)
    _expect_int("no-cfg batch", qwen_batch_size(Float32(1.0)), 1)
    _expect_close(
        "cfg combine",
        qwen_cfg_combine_value(Float32(3.0), Float32(1.0), Float32(2.0)),
        Float32(5.0),
        Float32(1e-6),
    )

    var contract = qwen_latent_contract_for_image(1024, 512, 2)
    _expect_int("contract batch", contract.batch_size, 2)
    _expect_int("contract latent h", contract.latent_height, 128)
    _expect_int("contract latent w", contract.latent_width, 64)
    _expect_int("contract packed seq", contract.packed_seq_len, 2048)
    _expect_int("contract packed channels", contract.packed_channels, 64)

    var shift_cfg = QwenSamplerSchedulerConfig(
        256, 4096, Float32(0.5), Float32(1.15)
    )
    var schedule = qwen_make_schedule(4, plan.latent_h, plan.latent_w, shift_cfg)
    _expect_int("schedule timesteps", len(schedule.timesteps), 4)
    _expect_int("schedule sigmas", len(schedule.sigmas), 5)
    _expect_close("schedule shift", schedule.shift, Float32(3.1581929), Float32(2e-5))
    _expect_close("schedule mu", schedule.mu, Float32(1.15), Float32(2e-5))
    _expect_close("sigma[0]", schedule.sigmas[0], Float32(1.0), Float32(1e-6))
    _expect_close("sigma[1]", schedule.sigmas[1], Float32(0.8634974), Float32(2e-5))
    _expect_close("sigma[2]", schedule.sigmas[2], Float32(0.6129789), Float32(2e-5))
    _expect_close("sigma[3]", schedule.sigmas[3], Float32(0.00315139), Float32(2e-6))
    _expect_close("sigma terminal", schedule.sigmas[4], Float32(0.0), Float32(1e-6))
    _expect_close("timestep[1]", schedule.timesteps[1], Float32(863.49744), Float32(2e-2))

    var updated = qwen_euler_update_value(
        Float32(0.25),
        Float32(0.5),
        schedule.sigmas[0],
        schedule.sigmas[1],
    )
    _expect_close("euler update", updated, Float32(0.1817487), Float32(2e-5))

    print("QWEN SAMPLER HELPER GATE OK")
    print(
        "plan =", plan.height, "x", plan.width,
        " latent =", plan.latent_h, "x", plan.latent_w,
        " packed =", plan.packed_seq_len, "x", plan.packed_channels,
        " batch =", plan.batch_size,
    )
    print(
        "schedule shift =", schedule.shift,
        " mu =", schedule.mu,
        " sigma1 =", schedule.sigmas[1],
        " timestep1 =", schedule.timesteps[1],
        " euler =", updated,
    )
