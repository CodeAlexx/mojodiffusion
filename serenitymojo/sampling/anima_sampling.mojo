# sampling/anima_sampling.mojo - Anima linear FlowMatch scheduler helpers.
#
# Anima's image path uses a linear sigma schedule from 1.0 to 0.0, textbook CFG,
# and direct-velocity Euler updates with no timestep *1000 scaling.

from std.gpu.host import DeviceContext

from serenitymojo.ops.tensor_algebra import add, mul_scalar, sub
from serenitymojo.tensor import Tensor


def anima_sigma(index: Int, num_steps: Int) raises -> Float32:
    if num_steps <= 0:
        raise Error("anima_sigma: num_steps must be > 0")
    if index < 0 or index > num_steps:
        raise Error("anima_sigma: index out of range")
    return 1.0 - Float32(index) / Float32(num_steps)


def build_anima_sigma_schedule(num_steps: Int) raises -> List[Float32]:
    if num_steps <= 0:
        raise Error("build_anima_sigma_schedule: num_steps must be > 0")
    var out = List[Float32]()
    for i in range(num_steps + 1):
        out.append(anima_sigma(i, num_steps))
    return out^


def anima_model_timestep_from_sigma(sigma: Float32) -> Float32:
    return sigma


def anima_cfg(
    v_cond: Tensor, v_uncond: Tensor, guidance_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Textbook classifier-free guidance: uncond + scale * (cond - uncond)."""
    if v_cond.numel() != v_uncond.numel():
        raise Error("anima_cfg: shape mismatch")
    if v_cond.dtype() != v_uncond.dtype():
        raise Error("anima_cfg: dtype mismatch")
    var diff = sub(v_cond, v_uncond, ctx)
    var scaled = mul_scalar(diff, guidance_scale, ctx)
    return add(v_uncond, scaled, ctx)


def anima_euler_step(
    latent: Tensor, velocity: Tensor, dt: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Euler update: latent + velocity * (sigma_next - sigma)."""
    if latent.numel() != velocity.numel():
        raise Error("anima_euler_step: shape mismatch")
    if latent.dtype() != velocity.dtype():
        raise Error("anima_euler_step: dtype mismatch")
    var delta = mul_scalar(velocity, dt, ctx)
    return add(latent, delta, ctx)


struct AnimaLinearFlowScheduler(Movable):
    var _sigmas: List[Float32]
    var num_steps: Int

    def __init__(out self, num_steps: Int) raises:
        self._sigmas = build_anima_sigma_schedule(num_steps)
        self.num_steps = num_steps

    @staticmethod
    def default_30() raises -> AnimaLinearFlowScheduler:
        return AnimaLinearFlowScheduler(30)

    def sigmas(self) -> List[Float32]:
        return self._sigmas.copy()

    def sigma(self, i: Int) raises -> Float32:
        if i < 0 or i > self.num_steps:
            raise Error("AnimaLinearFlowScheduler.sigma: step out of range")
        return self._sigmas[i]

    def model_timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("AnimaLinearFlowScheduler.model_timestep: step out of range")
        return anima_model_timestep_from_sigma(self._sigmas[i])

    def dt(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("AnimaLinearFlowScheduler.dt: step out of range")
        return self._sigmas[i + 1] - self._sigmas[i]

    def step(
        self, latent: Tensor, velocity: Tensor, i: Int, ctx: DeviceContext
    ) raises -> Tensor:
        return anima_euler_step(latent, velocity, self.dt(i), ctx)
