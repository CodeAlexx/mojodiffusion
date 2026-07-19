# sampling/lens_flowmatch.mojo — Microsoft Lens FlowMatch scalar scheduler.
#
# 1:1 port of the host-side Lens FlowMatch schedule from Serenity
# (modules/modelSampler/LensSampler.py:__sample_base + lens/pipeline.py:38
# compute_empirical_mu and the diffusers FlowMatchEulerDiscreteScheduler
# set_timesteps(sigmas=linspace(1,1/N,N), mu) call). Lens differs from the shared
# Z-Image/FLUX schedulers in one important contract: the public shifted sigma
# list has exactly N values (`linspace(1.0, 1.0 / N, N)` after dynamic
# exponential shift), and the final Euler step uses `sigma_next = 0.0` outside
# that list.
#
# Includes the host schedule and the GPU tensor Euler step. Tensor ops may use
# F32 arithmetic internally, but the latent carrier stays in its storage dtype.

from std.gpu.host import DeviceContext
from std.math import exp

from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, mul_scalar
from serenitymojo.tensor import Tensor


def lens_image_seq_len(width: Int, height: Int) raises -> Int:
    """Image-only sequence length used by Lens `compute_empirical_mu`.

    Lens uses latent image tokens only: `(height / 16) * (width / 16)`.
    Do not use the full DiT text+image sequence for the scheduler shift.
    """
    if width <= 0 or height <= 0:
        raise Error("lens_image_seq_len: dimensions must be > 0")
    if width % 16 != 0:
        raise Error("lens_image_seq_len: width must divide by 16")
    if height % 16 != 0:
        raise Error("lens_image_seq_len: height must divide by 16")
    return (height // 16) * (width // 16)


def lens_compute_empirical_mu(image_seq_len: Int, num_steps: Int) raises -> Float64:
    """Lens empirical dynamic-shift parameter.

    Byte-for-byte port of Serenity `compute_empirical_mu`
    (lens/pipeline.py:38). For 1024x1024 and 20 steps, `image_seq_len=4096` and
    `mu=2.1980220725551165`.
    """
    if image_seq_len <= 0:
        raise Error("lens_compute_empirical_mu: image_seq_len must be > 0")
    if num_steps <= 0:
        raise Error("lens_compute_empirical_mu: num_steps must be > 0")

    var a1: Float64 = 8.73809524e-05
    var b1: Float64 = 1.89833333
    var a2: Float64 = 0.00016927
    var b2: Float64 = 0.45666666
    var seq = Float64(image_seq_len)

    if image_seq_len > 4300:
        return a2 * seq + b2

    var m_200 = a2 * seq + b2
    var m_10 = a1 * seq + b1
    var a = (m_200 - m_10) / 190.0
    var b = m_200 - 200.0 * a
    return a * Float64(num_steps) + b


def build_lens_raw_sigmas(num_steps: Int) raises -> List[Float32]:
    """Build Lens raw sigmas: `linspace(1.0, 1.0 / N, N)`.

    Returns exactly `num_steps` values. The terminal zero is not part of the
    public list; callers use `0.0` only as `sigma_next` on the final step.
    """
    if num_steps <= 0:
        raise Error("build_lens_raw_sigmas: num_steps must be > 0")

    var out = List[Float32]()
    if num_steps == 1:
        out.append(1.0)
        return out^

    var n = Float64(num_steps)
    var start: Float64 = 1.0
    var end = 1.0 / n
    var step = (end - start) / Float64(num_steps - 1)
    for i in range(num_steps):
        out.append(Float32(start + step * Float64(i)))

    # Pin the endpoints exactly (defensive against linspace rounding).
    out[0] = Float32(start)
    out[num_steps - 1] = Float32(end)
    return out^


def lens_exponential_shift(sigma: Float32, mu: Float64) -> Float32:
    """Diffusers FlowMatchEuler exponential time shift."""
    var s = Float64(sigma)
    var exp_mu = exp(mu)
    return Float32((exp_mu * s) / (exp_mu * s + (1.0 - s)))


def lens_euler_step(
    latents: Tensor,
    noise_pred: Tensor,
    sigma_curr: Float32,
    sigma_next: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """One Lens FlowMatch Euler step.

    `dt * noise_pred` is rounded in the latent storage dtype. F32 is limited to
    scalar `dt` and tensor-op internals; the latent tensor is not stored in F32.
    """
    var target_dtype = latents.dtype()
    var dt = sigma_next - sigma_curr
    var pred = cast_tensor(noise_pred, target_dtype, ctx)
    var delta = mul_scalar(pred, dt, ctx)
    return add(latents, delta, ctx)


def build_lens_shifted_sigmas(
    num_steps: Int, image_seq_len: Int
) raises -> List[Float32]:
    """Build Lens shifted sigmas after empirical `mu`."""
    var mu = lens_compute_empirical_mu(image_seq_len, num_steps)
    var raw = build_lens_raw_sigmas(num_steps)
    var out = List[Float32]()
    for i in range(len(raw)):
        out.append(lens_exponential_shift(raw[i], mu))
    return out^


struct LensFlowMatchScheduler(Movable):
    """Host scalar schedule for Microsoft Lens FlowMatch-Euler."""

    var _sigmas: List[Float32]
    var num_steps: Int
    var image_seq_len: Int
    var mu: Float64

    def __init__(out self, num_steps: Int, image_seq_len: Int) raises:
        self._sigmas = build_lens_shifted_sigmas(num_steps, image_seq_len)
        self.num_steps = num_steps
        self.image_seq_len = image_seq_len
        self.mu = lens_compute_empirical_mu(image_seq_len, num_steps)

    @staticmethod
    def for_resolution(width: Int, height: Int, num_steps: Int) raises -> LensFlowMatchScheduler:
        return LensFlowMatchScheduler(num_steps, lens_image_seq_len(width, height))

    def sigmas(self) -> List[Float32]:
        """Per-step shifted sigmas, length `num_steps`."""
        return self._sigmas.copy()

    def timestep(self, i: Int) raises -> Float32:
        """Model timestep for denoise step `i`; Lens passes sigma directly."""
        if i < 0 or i >= self.num_steps:
            raise Error("LensFlowMatchScheduler.timestep: step out of range")
        return self._sigmas[i]

    def sigma_next(self, i: Int) raises -> Float32:
        """Next sigma for Euler step `i`; final step targets 0.0."""
        if i < 0 or i >= self.num_steps:
            raise Error("LensFlowMatchScheduler.sigma_next: step out of range")
        if i + 1 < self.num_steps:
            return self._sigmas[i + 1]
        return 0.0

    def dt(self, i: Int) raises -> Float32:
        """Euler scalar `sigma_next - sigma_curr`, normally negative."""
        return self.sigma_next(i) - self.timestep(i)

    def step(
        self, latents: Tensor, noise_pred: Tensor, i: Int, ctx: DeviceContext
    ) raises -> Tensor:
        return lens_euler_step(latents, noise_pred, self.timestep(i), self.sigma_next(i), ctx)
