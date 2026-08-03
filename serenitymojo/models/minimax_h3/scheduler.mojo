# serenitymojo/models/minimax_h3/scheduler.mojo
#
# MiniMax-H3 rectified-flow Euler scheduler (eta = 0) — unit 2 of the H3 port.
#
# Source (op-for-op port, diffusers PR huggingface/diffusers#14355 at head
# e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1):
#   src/diffusers/schedulers/scheduling_minimax_h3.py
#     MiniMaxH3Scheduler.set_timesteps      :127-173
#     MiniMaxH3Scheduler.index_for_timestep :175-194
#     MiniMaxH3Scheduler.scale_noise        :196-225
#     MiniMaxH3Scheduler.step               :227-288
#
# Three things make this NOT a flow-match Euler scheduler, per the reference's
# own module docstring, and each one is a place a "reasonable" port goes wrong:
#
#   1. The velocity sign is REVERSED. H3's transformer predicts a data-ward
#      velocity, so `x0 = x_t + sigma * v`, not `x_t - sigma * v`.
#   2. Timesteps are `t = 1 - sigma` in [0, 1] with t = 1 meaning CLEAN — the
#      opposite direction from the usual `sigma * 1000` convention, and the
#      AdaLN consumes this convention directly.
#   3. The grid is `linspace(1, 0, num_inference_steps)`: the terminal zero is
#      part of the requested step count, so N sigmas drive N-1 model calls.
#
# Two more traps that only show up under a bit-exact gate:
#
#   * `torch.linspace` on CPU fills from BOTH ENDS — `start + i*step` below the
#     halfway point and `end - (N-1-i)*step` at or above it — so a one-sided
#     fill drifts in the back half. Reproduced in `_linspace_1_to_0`.
#   * `step()` draws sigma from two different sources on purpose: the denoised
#     estimate uses `1 - timestep` (what the transformer was conditioned on)
#     while the Euler ratio uses the stored sigma grid. For sigma < 0.5 the
#     float32 round trip `1 - (1 - sigma)` is not exact, and the reference keeps
#     them apart. Do not "simplify" this.
#
# Float32 throughout: the reference builds the grid in float32 on CPU and
# evaluates the blend in float32. H3 runs TWO of these per request, shift 12.0
# for video and 3.0 for audio; the modality is not a property of the scheduler.
#
# Host-side lists: the schedule itself is scalar and the blend is elementwise,
# so this is exact and cheap here. The elementwise application moves to device
# kernels when the sampler is wired; the scalar schedule does not change.

from std.collections import List
from std.math import fma
from std.benchmark import black_box

# The released checkpoints' shifts (scheduler_config.json, written by the
# converter from `model_index.json` `_minimax_h3.sigma_shift_scales`).
comptime MINIMAX_H3_VIDEO_SHIFT = Float32(12.0)
comptime MINIMAX_H3_AUDIO_SHIFT = Float32(3.0)


def _round_f32(var x: Float32) -> Float32:
    """Optimization barrier forcing one float32 rounding.

    Mojo contracts `a + b*c` into a fused multiply-add — MEASURED: fusing only
    the `denoised` expression reproduces this port's pre-fix trajectory
    mismatch exactly, 67 of 464 values on `video_30`, including the first
    differing element. The reference runs the multiply and the add as separate
    torch kernels with a temporary between them, i.e. two roundings, so the
    fused form is 1 ulp off. `@no_inline` makes the compiler materialize the
    value across the call and stops the contraction.

    Scalar host code on tiny arrays, so the call overhead is irrelevant here.
    When the blend moves to a device kernel, that kernel must not contract
    either."""
    # MEASURED 2026-08-03: `@no_inline` did NOTHING. The call site is tagged
    # `call contract float` with an IDENTICAL contract-flag count at -O0 and
    # -O3 — @no_inline blocks code DUPLICATION, not the fast-math permission
    # to fuse. -O0 merely never exploited it, which is why this "worked".
    # `black_box` is a real inline-asm memory clobber the optimizer cannot
    # prove is a pure identity, so it holds at every level.
    return black_box(x)


def _linspace_1_to_0(steps: Int) -> List[Float32]:
    """`torch.linspace(1.0, 0.0, steps, dtype=float32)`, CPU semantics.

    ATen's vectorized CPU kernel fills from both ends around
    `halfway = steps // 2` and uses a FUSED multiply-add, i.e. ONE rounding per
    element: `fma(step, i, start)` below the halfway point and
    `fma(-step, steps-1-i, end)` at or above it, with
    `step = (end - start) / (steps - 1)` computed in float32.

    Determined by measurement, not assumption: a one-sided fill, a separately
    rounded mul-then-add, and a float64 grid cast down each miss torch at
    several indices, while this form is bit-exact for every step count tried
    (2, 8, 29, 30, 50, 100, 400)."""
    var start = Float32(1.0)
    var end = Float32(0.0)
    var step = (end - start) / Float32(steps - 1)
    var halfway = steps // 2
    var out = List[Float32]()
    for i in range(steps):
        if i < halfway:
            out.append(fma(step, Float32(i), start))
        else:
            out.append(fma(-step, Float32(steps - i - 1), end))
    return out^


struct MiniMaxH3Scheduler(Movable):
    """Rectified-flow Euler scheduler with an exponential sigma shift.

    `step_index` and `begin_index` use -1 for the reference's `None`."""

    var shift: Float32
    var sigmas: List[Float32]
    var timesteps: List[Float32]
    var step_index: Int
    var begin_index: Int

    def __init__(out self, shift: Float32) raises:
        if shift <= 0.0:
            raise Error("MiniMax-H3 scheduler: `shift` must be positive")
        self.shift = shift
        self.sigmas = List[Float32]()
        self.timesteps = List[Float32]()
        self.step_index = -1
        self.begin_index = -1

    def set_shift(mut self, shift: Float32) raises:
        """Override the sigma shift; call before `set_timesteps`."""
        if shift <= 0.0:
            raise Error("MiniMax-H3 scheduler: `shift` must be positive")
        self.shift = shift

    def set_begin_index(mut self, begin_index: Int):
        self.begin_index = begin_index

    def set_timesteps(mut self, num_inference_steps: Int) raises:
        """Build the sigma / timestep schedule.

        `linspace(1, 0, N)` pushed through `sigma' = s*sigma / (1 + (s-1)*sigma)`
        with consecutive duplicates collapsed. The terminal 0 is already in the
        grid (the shift maps 0 to exactly 0), so N sigmas drive N-1 model
        evaluations and `timesteps = 1 - sigmas[:-1]`."""
        if num_inference_steps < 2:
            raise Error(
                "MiniMax-H3 scheduler: `num_inference_steps` must be >= 2"
            )

        var base = _linspace_1_to_0(num_inference_steps)
        var shifted = List[Float32]()
        var shift_minus_one = self.shift - Float32(1.0)
        for i in range(len(base)):
            var b = base[i]
            # Was UNPROTECTED. Only bites when shift-1 is not an exact power of
            # two: shift 12.0 -> 11.0 breaks (video, 4/30 sigmas), shift 3.0 -> 2.0
            # is exact so audio never rounded — which is why audio always passed.
            var num = _round_f32(self.shift * b)
            var denom_term = _round_f32(shift_minus_one * b)
            var denom = _round_f32(Float32(1.0) + denom_term)
            shifted.append(num / denom)

        # The shift compresses the grid near sigma = 1; collapse the float32
        # collisions it can create (`torch.unique_consecutive`).
        var sigmas = List[Float32]()
        for i in range(len(shifted)):
            if i == 0 or shifted[i] != shifted[i - 1]:
                sigmas.append(shifted[i])

        var timesteps = List[Float32]()
        for i in range(len(sigmas) - 1):
            timesteps.append(Float32(1.0) - sigmas[i])

        self.sigmas = sigmas^
        self.timesteps = timesteps^
        self.step_index = -1
        self.begin_index = -1

    def num_inference_steps(self) -> Int:
        """Model evaluations the schedule drives — one fewer than the sigmas."""
        return len(self.timesteps)

    def index_for_timestep(self, timestep: Float32) raises -> Int:
        """Map a timestep value to its index. The schedule is strictly
        increasing in t, so the match is unique."""
        for i in range(len(self.timesteps)):
            if self.timesteps[i] == timestep:
                return i
        raise Error(
            "MiniMax-H3 scheduler: timestep is not in `timesteps`; pass a value"
            " taken from the schedule"
        )

    def scale_noise(
        self, sample: List[Float32], timestep: Float32, noise: List[Float32]
    ) raises -> List[Float32]:
        """Rectified-flow forward process in H3's t convention:
        `x_t = t*x0 + (1 - t)*noise`, with t = 1 returning `sample` unchanged.

        H3 noises its conditioning anchors with this, where t is the noise-aug
        level rather than a schedule entry, so `timestep` is taken at face value
        and is NOT looked up in the schedule."""
        if len(sample) != len(noise):
            raise Error("MiniMax-H3 scheduler: sample and noise length mismatch")
        var one_minus_timestep = _round_f32(Float32(1.0) - timestep)
        var out = List[Float32]()
        for i in range(len(sample)):
            var a = _round_f32(timestep * sample[i])
            var bb = _round_f32(one_minus_timestep * noise[i])
            out.append(_round_f32(a + bb))
        return out^

    def step(
        mut self,
        model_output: List[Float32],
        timestep: Float32,
        sample: List[Float32],
    ) raises -> List[Float32]:
        """One Euler (eta = 0) step.

        `x0 = x_t + (1 - t)*v` — note the PLUS, the opposite of the usual
        flow-match convention — then the blend `x_next = r*x_t + (1 - r)*x0`
        with `r = sigma_next / sigma` taken from the sigma grid."""
        if len(model_output) != len(sample):
            raise Error(
                "MiniMax-H3 scheduler: model_output and sample length mismatch"
            )
        if self.step_index < 0:
            if self.begin_index < 0:
                self.step_index = self.index_for_timestep(timestep)
            else:
                self.step_index = self.begin_index
        if self.step_index + 1 >= len(self.sigmas):
            raise Error("MiniMax-H3 scheduler: schedule exhausted")

        # sigma for the denoised estimate comes from the TIMESTEP the
        # transformer was conditioned on, not from the grid (see header).
        var sigma_from_timestep = Float32(1.0) - timestep
        var sigma = self.sigmas[self.step_index]
        var sigma_next = self.sigmas[self.step_index + 1]
        var ratio = sigma_next / sigma

        # Each multiply is rounded to float32 BEFORE its add. The reference runs
        # these as separate torch kernels with a temporary between them, so a
        # fused multiply-add — one rounding instead of two — lands 1 ulp away.
        # Measured: a fully fused form misses the reference trajectory, a
        # separately rounded one reproduces it exactly.
        var one_minus_ratio = Float32(1.0) - ratio
        var out = List[Float32]()
        for i in range(len(sample)):
            var scaled_velocity = _round_f32(sigma_from_timestep * model_output[i])
            var denoised = _round_f32(sample[i] + scaled_velocity)
            var kept = _round_f32(ratio * sample[i])
            var moved = _round_f32(one_minus_ratio * denoised)
            out.append(_round_f32(kept + moved))

        self.step_index += 1
        return out^
