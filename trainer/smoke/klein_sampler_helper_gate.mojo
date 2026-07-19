# Tiny Flux2/Klein sampler helper gate.
#
# Serenity reference:
#   modules/modelSampler/Flux2Sampler.py:73 chooses CFG batch size,
#   :96-101 prepares the mu/custom-sigma scheduler, :113-135 performs guidance,
#   CFG combine, and one scheduler step.
#
# This gate intentionally covers only the Mojo helpers currently implemented in
# modelSampler/Flux2Sampler.mojo. It is not an end-to-end sampler parity test.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenity_trainer.modelSampler.Flux2Sampler import (
    make_flux2_denoise_state,
    flux2_batch_size,
    flux2_cfg_combine,
    flux2_euler_step,
    flux2_guidance_value,
    flux2_step_t_embedder,
    flux2_step_t_model,
    flux2_use_cfg,
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
    var ctx = DeviceContext()

    var latent_vals = List[Float32]()
    for i in range(32):
        latent_vals.append(Float32(i) * Float32(0.01))
    var latent = Tensor.from_host(latent_vals^, [1, 4, 8], STDtype.F32, ctx)
    var state = make_flux2_denoise_state(latent^, 4, 4)

    _expect_int("timesteps", len(state.scheduler.timesteps), 4)
    _expect_int("sigmas", len(state.scheduler.sigmas), 5)
    _expect_close(
        "t_model * 1000",
        flux2_step_t_model(state, 0) * Float32(1000.0),
        flux2_step_t_embedder(state, 0),
        Float32(1e-3),
    )

    _expect_bool("cfg path", flux2_use_cfg(Float32(2.0), False), True)
    _expect_int("cfg batch", flux2_batch_size(Float32(2.0), False), 2)
    _expect_bool("guidance disables cfg batch", flux2_use_cfg(Float32(2.0), True), False)
    _expect_int("guidance batch", flux2_batch_size(Float32(2.0), True), 1)

    var guidance = flux2_guidance_value(Float32(3.5), True)
    if not guidance:
        raise Error("guidance value missing")
    _expect_close("guidance embedder value", guidance.value(), Float32(3500.0), Float32(1e-3))
    var no_guidance = flux2_guidance_value(Float32(3.5), False)
    if no_guidance:
        raise Error("guidance value present when guidance_embeds=False")

    var pos = Tensor.from_host([Float32(1.0), Float32(3.0)], [2], STDtype.F32, ctx)
    var neg = Tensor.from_host([Float32(0.5), Float32(1.0)], [2], STDtype.F32, ctx)
    var cfg = flux2_cfg_combine(pos, neg, Float32(2.0), ctx).to_host(ctx)
    _expect_close("cfg[0]", cfg[0], Float32(1.5), Float32(1e-6))
    _expect_close("cfg[1]", cfg[1], Float32(5.0), Float32(1e-6))

    var pred_vals = List[Float32]()
    for _ in range(32):
        pred_vals.append(Float32(1.0))
    var pred = Tensor.from_host(pred_vals^, [1, 4, 8], STDtype.BF16, ctx)
    flux2_euler_step(state, pred, 0, ctx)
    if state.latent.dtype() != STDtype.BF16:
        raise Error("Euler step did not cast prev_sample back to model_output dtype")
    var stepped = state.latent.to_host(ctx)
    if _abs(stepped[0]) <= Float32(0.0):
        raise Error("Euler step left latent[0] unchanged")

    print("KLEIN FLUX2 SAMPLER HELPERS OK: steps =", len(state.scheduler.timesteps), " cfg[1] =", cfg[1])
