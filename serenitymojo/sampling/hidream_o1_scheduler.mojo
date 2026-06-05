# sampling/hidream_o1_scheduler.mojo — HiDream-O1 FlashFlowMatchEuler scheduler.
#
# Reference, read line-by-line:
#   /home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/scheduler.rs
#   /home/alex/HiDream-O1-Image/models/flash_scheduler.py (cited inline in .rs)
#
# Two modes:
#   Flash   (Dev, 28 step): stochastic Euler, re-injects scaled noise each step.
#     timesteps = DEFAULT_TIMESTEPS_DEV (hardcoded 28 values).
#     sigmas    = [t/1000 for t in timesteps] + [0.0]; shift=1.0.
#     step (flash_scheduler.py:340-356):
#       denoised = sample - model_output*sigma
#       noise    = clamp(noise, +/- k*std(noise))  if k>0
#       sample'  = sigma_next*noise*s_noise + (1-sigma_next)*denoised
#   Default (Full, N step): deterministic stock diffusers Euler; shift=3.0.
#     sigmas via linspace + shift transform; step:
#       prev = sample + (sigma_next - sigma) * model_output
#
# model_output is the post-CFG, post-NEGATION velocity (pipeline applies the
# `model_output = -v_guided` sign flip; scheduler.rs:179-181 + pipeline F3).
# t_pixeldit = 1 - t/1000 is computed pipeline-side, not here.
#
# Reused foundation ops: ops/tensor_algebra.{add,sub,mul_scalar}, ops/cast.
# Pure scalar schedule math on host (F32); tensor carriers preserve sample dtype.
# Mojo 1.0.0b1.

from std.math import sqrt as fsqrt

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar
from serenitymojo.ops.cast import cast_tensor
from std.gpu.host import DeviceContext


# Hardcoded 28-step Dev timestep list (scheduler.rs:51-54 / pipeline.py:25-28).
def default_timesteps_dev() -> List[Int]:
    var ts = List[Int]()
    var raw = [
        999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764,
        737, 707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
    ]
    for v in raw:
        ts.append(v)
    return ts^


comptime SCHED_FLASH = 0
comptime SCHED_DEFAULT = 1


struct HiDreamO1Scheduler(Movable):
    """FlashFlowMatchEulerDiscreteScheduler. Use dev_28step() for Dev (Flash,
    stochastic) and full_n_step(n, shift) for Full (Default, deterministic)."""

    var timesteps: List[Float32]  # per-step, descending. len = num_inference_steps
    var sigmas: List[Float32]  # len = num_inference_steps + 1 (trailing 0.0)
    var shift: Float32
    var mode: Int  # SCHED_FLASH | SCHED_DEFAULT

    def __init__(
        out self,
        var timesteps: List[Float32],
        var sigmas: List[Float32],
        shift: Float32,
        mode: Int,
    ):
        self.timesteps = timesteps^
        self.sigmas = sigmas^
        self.shift = shift
        self.mode = mode

    @staticmethod
    def dev_28step() raises -> HiDreamO1Scheduler:
        var ts_i = default_timesteps_dev()
        var timesteps = List[Float32]()
        var sigmas = List[Float32]()
        for v in ts_i:
            timesteps.append(Float32(v))
            sigmas.append(Float32(v) / Float32(1000.0))
        sigmas.append(Float32(0.0))
        return HiDreamO1Scheduler(timesteps^, sigmas^, Float32(1.0), SCHED_FLASH)

    @staticmethod
    def full_n_step(n: Int, shift: Float32) raises -> HiDreamO1Scheduler:
        # flash_scheduler.py:215-240 (scheduler.rs:124-169).
        var sigma_min = Float32(1.0) / Float32(1000.0)
        var sigma_max = Float32(1.0)
        var timesteps0 = List[Float32]()
        if n == 1:
            timesteps0.append(sigma_max * Float32(1000.0))
        else:
            var lo = sigma_min * Float32(1000.0)
            var hi = sigma_max * Float32(1000.0)
            for i in range(n):
                var alpha = Float32(i) / Float32(n - 1)
                timesteps0.append(hi + (lo - hi) * alpha)
        var sigmas = List[Float32]()
        for i in range(n):
            sigmas.append(timesteps0[i] / Float32(1000.0))
        # shift transform.
        for i in range(n):
            var s = sigmas[i]
            sigmas[i] = (shift * s) / (Float32(1.0) + (shift - Float32(1.0)) * s)
        # recompute timesteps from shifted sigmas.
        var timesteps = List[Float32]()
        for i in range(n):
            timesteps.append(sigmas[i] * Float32(1000.0))
        sigmas.append(Float32(0.0))
        return HiDreamO1Scheduler(timesteps^, sigmas^, shift, SCHED_DEFAULT)

    def num_inference_steps(self) -> Int:
        return len(self.timesteps)

    def timestep(self, i: Int) raises -> Float32:
        return self.timesteps[i]

    def needs_step_noise(self) -> Bool:
        return self.mode == SCHED_FLASH

    # Single denoise step. model_output: post-CFG, post-negation velocity
    # [1, L, 3072]. sample: current z patches. noise: pre-drawn N(0,1) of the
    # same shape (Flash only; pass a zeros tensor / ignored for Default).
    # s_noise: noise scaling (Dev constant 7.5). noise_clip_std: +/- k*std clip
    # (Dev 2.5; <=0 disables). Tensor ops use F32 arithmetic internally and
    # store the sample dtype; host std/debug values are F32 scalars.
    def step(
        self,
        model_output: Tensor,
        step_index: Int,
        sample: Tensor,
        noise: Tensor,
        s_noise: Float32,
        noise_clip_std: Float32,
        ctx: DeviceContext,
    ) raises -> Tensor:
        if step_index >= len(self.timesteps):
            raise Error("HiDreamO1Scheduler.step: step_index out of range")
        var sigma = self.sigmas[step_index]
        var sigma_next = self.sigmas[step_index + 1]
        var out_dtype = sample.dtype()
        var mo_step = cast_tensor(model_output, out_dtype, ctx)

        if self.mode == SCHED_FLASH:
            # denoised = sample - model_output * sigma
            var mo_sigma = mul_scalar(mo_step, sigma, ctx)
            var denoised = sub(sample, mo_sigma, ctx)

            var noise_step = cast_tensor(noise, out_dtype, ctx)
            # Optional +/- k*sample-std clip (scheduler.rs:257-281). Compute std
            # host-side; small relative to model fwd cost.
            if noise_clip_std > Float32(0.0):
                var host = noise_step.to_host(ctx)
                var nf = Float32(len(host))
                var mean = Float32(0.0)
                for x in host:
                    mean += x
                mean /= nf
                var var_acc = Float32(0.0)
                for x in host:
                    var_acc += (x - mean) * (x - mean)
                var std = fsqrt(var_acc / nf)
                var clip_val = noise_clip_std * std
                var clamped = List[Float32]()
                for x in host:
                    var c = x
                    if c > clip_val:
                        c = clip_val
                    if c < -clip_val:
                        c = -clip_val
                    clamped.append(c)
                noise_step = Tensor.from_host(
                    clamped, noise_step.shape(), out_dtype, ctx
                )

            # sample' = sigma_next*noise*s_noise + (1-sigma_next)*denoised
            var weight_noise = sigma_next * s_noise
            var weight_den = Float32(1.0) - sigma_next
            var term_noise = mul_scalar(noise_step, weight_noise, ctx)
            var term_den = mul_scalar(denoised, weight_den, ctx)
            return add(term_noise, term_den, ctx)
        else:
            # prev = sample + (sigma_next - sigma) * model_output
            var dsigma = sigma_next - sigma
            var term = mul_scalar(mo_step, dsigma, ctx)
            return add(sample, term, ctx)
