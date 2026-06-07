# Compile/run smoke for host-side FLUX.1-dev schedule and pack contracts.
#
# This deliberately does not create a DeviceContext or run model inference.

from serenitymojo.sampling.flux1_dev import (
    Flux1DevScheduler,
    build_flux1_packed_latent_plan,
    build_flux1_sigma_schedule,
    flux1_cfg_batch_size,
    flux1_dynamic_shift,
    flux1_euler_update_value,
    flux1_guidance_embed_value,
    flux1_latent_spatial_dim,
    flux1_model_timestep_from_scheduler_timestep,
    flux1_model_timestep_from_sigma,
    flux1_mu,
    flux1_packed_spatial_dim,
    flux1_scheduler_timestep_from_sigma,
    validate_flux1_flow_match_scheduler_config,
)


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _check_equal(name: String, actual: Int, expected: Int) raises:
    if actual != expected:
        raise Error(
            name
            + " mismatch: actual="
            + String(actual)
            + " expected="
            + String(expected)
        )


def _check_close(name: String, actual: Float64, expected: Float64) raises:
    if _abs(actual - expected) > 1.0e-5:
        raise Error(
            name
            + " mismatch: actual="
            + String(actual)
            + " expected="
            + String(expected)
        )


def _check_strictly_less(name: String, actual: Float64, upper_bound: Float64) raises:
    if actual >= upper_bound:
        raise Error(
            name
            + " expected strictly less than "
            + String(upper_bound)
            + ", got "
            + String(actual)
        )


def _expect_scheduler_config_block() raises:
    var raised = False
    try:
        validate_flux1_flow_match_scheduler_config(
            String("FlowMatchEulerDiscreteScheduler"),
            1000,
            3.0,
            256,
            4096,
            0.5,
            1.15,
            False,
        )
    except e:
        raised = True
        print("  scheduler config drift blocked as expected:", String(e))
    if not raised:
        raise Error("expected FLUX.1-dev scheduler config drift to be blocked")


def main() raises:
    validate_flux1_flow_match_scheduler_config(
        String("FlowMatchEulerDiscreteScheduler"),
        1000,
        3.0,
        256,
        4096,
        0.5,
        1.15,
        True,
    )
    _expect_scheduler_config_block()

    var plan = build_flux1_packed_latent_plan(1024, 1024, 512)
    plan.validate_dev_1024_contract()
    _check_equal("latent_h", plan.latent_h, 128)
    _check_equal("latent_w", plan.latent_w, 128)
    _check_equal("packed_h", plan.packed_h, 64)
    _check_equal("packed_w", plan.packed_w, 64)
    _check_equal("image_tokens", plan.image_tokens, 4096)
    _check_equal("packed_channels", plan.packed_channels, 64)
    _check_equal("total_sequence", plan.total_sequence, 4608)
    _check_equal("latent_h/packed_h", plan.latent_h, plan.packed_h * plan.patch_size)
    _check_equal("latent_w/packed_w", plan.latent_w, plan.packed_w * plan.patch_size)
    _check_equal(
        "packed channel patch area",
        plan.packed_channels,
        plan.latent_channels * plan.patch_size * plan.patch_size,
    )
    _check_equal("sequence sum", plan.total_sequence, plan.text_tokens + plan.image_tokens)

    var odd = build_flux1_packed_latent_plan(1025, 768, 512)
    _check_equal("odd latent_h", odd.latent_h, 96)
    _check_equal("odd latent_w", odd.latent_w, 130)
    _check_equal("odd image_tokens", odd.image_tokens, 3120)
    _check_equal("packed dim 1025", flux1_packed_spatial_dim(1025), 65)
    _check_equal("latent dim 1025", flux1_latent_spatial_dim(1025), 130)

    _check_close("mu(256)", flux1_mu(256), 0.5)
    _check_close("mu(4096)", flux1_mu(4096), 1.15)
    _check_close("dynamic shift(4096)", flux1_dynamic_shift(4096), 3.15819291)
    _check_equal("cfg batch size", flux1_cfg_batch_size(), 1)
    _check_close("guidance embed", Float64(flux1_guidance_embed_value(3.5)), 3.5)
    _check_close(
        "scheduler timestep",
        Float64(flux1_scheduler_timestep_from_sigma(0.25)),
        250.0,
    )
    _check_close(
        "model timestep from scheduler",
        Float64(flux1_model_timestep_from_scheduler_timestep(250.0)),
        0.25,
    )
    _check_close(
        "model timestep from sigma",
        Float64(flux1_model_timestep_from_sigma(0.25)),
        0.25,
    )
    _check_close(
        "scalar Euler update",
        Float64(flux1_euler_update_value(2.0, -4.0, 0.75, 0.25)),
        4.0,
    )

    var sigmas = build_flux1_sigma_schedule(20, 4096)
    _check_equal("schedule len", len(sigmas), 21)
    _check_close("sigma[0]", Float64(sigmas[0]), 1.0)
    _check_close("sigma[1]", Float64(sigmas[1]), 0.98360808)
    _check_close("sigma[10]", Float64(sigmas[10]), 0.75951092)
    _check_close("sigma[19]", Float64(sigmas[19]), 0.14252935)
    _check_close("sigma[20]", Float64(sigmas[20]), 0.0)
    for i in range(20):
        _check_strictly_less(
            String("sigma monotonic ") + String(i),
            Float64(sigmas[i + 1]),
            Float64(sigmas[i]),
        )

    var sched = Flux1DevScheduler(3, 4096)
    _check_close("sched.mu", sched.mu, 1.15)
    _check_close("sched.timestep(1)", Float64(sched.timestep(1)), 0.86332049)
    _check_close("sched.scheduler_timestep(1)", Float64(sched.scheduler_timestep(1)), 863.32049)
    _check_close("sched.model_timestep(1)", Float64(sched.model_timestep(1)), 0.86332049)
    _check_close("sched.dt(0)", Float64(sched.dt(0)), -0.13667951)
    _check_close(
        "sched.dt adjacent delta",
        Float64(sched.dt(1)),
        Float64(sched.timestep(2) - sched.timestep(1)),
    )
    print("FLUX.1-dev schedule/pack smoke PASS")
