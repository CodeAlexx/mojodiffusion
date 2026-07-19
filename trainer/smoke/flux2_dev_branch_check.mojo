# Flux2 dev branch + sampler helper gate.
#
# Serenity reference only:
#   modules/model/Flux2Model.py:232-236:
#       is_dev() => transformer.config.num_attention_heads == 48
#       is_klein() => not is_dev()
#   modules/modelSampler/Flux2Sampler.py:73, 95-101, 113-135:
#       CFG batch branching, empirical-mu scheduler prep, normalized transformer
#       timestep input, guidance tensor branch, and Euler dt sign.
#
# This is a bounded structural/helper gate. It does not run Serenity, PyTorch,
# text encoders, transformer inference, random noise, loss, gradients, optimizer
# steps, VAE decode, image saving, or speed checks. Passing this gate is not
# numeric train/sample parity.

from serenity_trainer.modelSampler.FlowMatchEulerDiscreteScheduler import (
    flux2_compute_empirical_mu,
    make_flux2_scheduler,
)
from serenity_trainer.modelSampler.Flux2Sampler import (
    flux2_batch_size,
    flux2_guidance_value,
    flux2_use_cfg,
)
from serenity_trainer.modelSetup.BaseFlux2Setup import (
    calculate_timestep_shift,
    model_t_from_timestep,
    sigma_from_timestep,
    timestep_from_sigma,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_FLUX_2,
    model_type_is_flux_2,
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


def _flux2_is_dev_runtime_branch(num_attention_heads: Int) -> Bool:
    return num_attention_heads == 48


def _flux2_is_klein_runtime_branch(num_attention_heads: Int) -> Bool:
    return not _flux2_is_dev_runtime_branch(num_attention_heads)


def main() raises:
    _expect_bool("ModelType.FLUX_2", model_type_is_flux_2(MODEL_TYPE_FLUX_2), True)
    _expect_bool("Flux2 dev branch heads=48", _flux2_is_dev_runtime_branch(48), True)
    _expect_bool("Flux2 dev branch heads=32", _flux2_is_dev_runtime_branch(32), False)
    _expect_bool("Flux2 Klein branch heads=32", _flux2_is_klein_runtime_branch(32), True)
    _expect_bool("Flux2 Klein branch heads=48", _flux2_is_klein_runtime_branch(48), False)

    # Flux2Sampler.py:73. Dev and Klein share this sampler branch; the loaded
    # checkpoint's guidance_embeds value chooses CFG batch vs guidance injection.
    _expect_bool("cfg path disabled at 1", flux2_use_cfg(Float32(1.0), False), False)
    _expect_bool("cfg path uses neg prompt", flux2_use_cfg(Float32(3.5), False), True)
    _expect_bool("guidance_embeds disables cfg batch", flux2_use_cfg(Float32(3.5), True), False)
    _expect_int("cfg batch off", flux2_batch_size(Float32(1.0), False), 1)
    _expect_int("cfg batch on", flux2_batch_size(Float32(3.5), False), 2)
    _expect_int("guidance batch", flux2_batch_size(Float32(3.5), True), 1)

    var guidance = flux2_guidance_value(Float32(3.5), True)
    if not guidance:
        raise Error("guidance value missing when guidance_embeds=True")
    _expect_close("guidance embedder integer input", guidance.value(), Float32(3500.0), Float32(1e-4))
    var no_guidance = flux2_guidance_value(Float32(3.5), False)
    if no_guidance:
        raise Error("guidance value present when guidance_embeds=False")

    # 1024x1024 sample: raw VAE latent is 128x128, Flux2 patchify halves it to
    # 64x64, then pack_latents gives image_seq_len = 4096 (Flux2Sampler.py:84-96).
    var diffusion_steps = 28
    var image_seq_len = 4096
    var mu = flux2_compute_empirical_mu(image_seq_len, diffusion_steps)
    _expect_close("empirical mu 4096/28", mu, Float32(2.1514432), Float32(5e-5))

    var scheduler = make_flux2_scheduler(diffusion_steps, image_seq_len)
    _expect_int("scheduler timesteps", len(scheduler.timesteps), diffusion_steps)
    _expect_int("scheduler sigmas", len(scheduler.sigmas), diffusion_steps + 1)
    _expect_close("sigma[0]", scheduler.sigmas[0], Float32(1.0), Float32(1e-6))
    _expect_close("sigma[1]", scheduler.sigmas[1], Float32(0.9957105), Float32(5e-5))
    _expect_close("sigma[last]", scheduler.sigmas[diffusion_steps - 1], Float32(0.2415146), Float32(5e-5))
    _expect_close("sigma terminal", scheduler.sigmas[diffusion_steps], Float32(0.0), Float32(1e-6))
    _expect_close("timestep[1]", scheduler.timesteps[1], Float32(995.7105), Float32(5e-2))
    _expect_close(
        "transformer timestep input",
        scheduler.timesteps[1] / Float32(1000.0),
        Float32(0.9957105),
        Float32(5e-5),
    )

    var dt0 = scheduler.sigmas[1] - scheduler.sigmas[0]
    if dt0 >= Float32(0.0):
        raise Error("Euler dt sign must be negative for descending sigmas")
    var sample_after_step0 = Float32(0.25) + dt0 * Float32(0.5)
    _expect_close("Euler scalar sign", sample_after_step0, Float32(0.2478552), Float32(5e-5))

    # Flux2Model.calculate_timestep_shift and BaseFlux2Setup scalar helpers. These
    # are deterministic math checks, not CPU PyTorch numeric parity.
    _expect_close("training timestep shift 64x64", calculate_timestep_shift(64, 64), Float32(1.8776106), Float32(5e-5))
    _expect_close("sigma from timestep", sigma_from_timestep(499), Float32(0.5), Float32(1e-6))
    _expect_close("model t from timestep", model_t_from_timestep(500), Float32(0.5), Float32(1e-6))
    _expect_int("timestep from sigma", timestep_from_sigma(Float32(0.5)), 499)

    print("FLUX2 DEV BRANCH/SAMPLER HELPER GATE OK")
    print(
        "dev_heads=48",
        "klein_heads=32",
        "seq_len =", image_seq_len,
        "steps =", diffusion_steps,
        "mu =", mu,
        "sigma1 =", scheduler.sigmas[1],
    )
