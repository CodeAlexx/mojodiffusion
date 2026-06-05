# sampling/flow_match.mojo — Z-Image flow-matching (rectified-flow) scheduler.
#
# Pure-Mojo port of the scheduler the Z-Image inference bin uses:
#   * Sigma schedule  — inference-flame/src/sampling/schedules.rs
#         t_i  = 1 - i/N                       for i in 0..=N   (N+1 values)
#         shift!=1:  sigma_i = shift*t_i / (1 + (shift-1)*t_i)
#     A linearly-spaced 1.0 -> 0.0 schedule with the flow-matching static shift.
#   * Euler update    — inference-flame/src/sampling/euler.rs
#         x_next = x + v * (sigma_next - sigma)
#     where v is the model's predicted velocity at sigma_i. dt is NEGATIVE
#     (sigma decreases), so this walks x from noise (sigma=1) to data (sigma=0).
#   * CFG combine     — inference-flame/src/sampling/euler.rs (the CODE form):
#         pred = pred_cond + cfg_scale * (pred_cond - pred_uncond)
#
# IMPORTANT for diffusers ZImagePipeline: this module's `cfg()` returns the raw
# CFG-combined transformer output. Diffusers then does `noise_pred = -noise_pred`
# before `FlowMatchEulerDiscreteScheduler.step`. Do not hide that sign flip in
# `cfg()` or `NextDiT.forward`; pipeline callers that match diffusers must pass
# `-cfg(...)` into the Euler update. See
# serenitymojo/docs/ZIMAGE_DENOISE_SIGN_CONVENTION.md.
#
# Z-Image BASE uses 30-50 steps (CFG 3-5); turbo=8. The schedule + update are
# identical across step counts — only N differs. Default shift = 3.0 (bin).
#
# Latent arithmetic goes through serenitymojo/ops/tensor_algebra (scalar mul +
# tensor add/sub) — never hand-rolled. The schedule itself is a tiny host-side
# F32 array (N+1 floats); only the per-step latent update touches the GPU.
#
# Mojo 1.0.0b1. Inference-only. No autograd, no Python at runtime.

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul, div, mul_scalar
from serenitymojo.io.dtype import STDtype
from std.gpu.host import DeviceContext
from std.math import exp, sqrt


# F32 machine epsilon — matches the Rust `f32::EPSILON` shift-skip guard.
comptime _F32_EPS: Float32 = 1.1920929e-07


def build_sigma_schedule(num_steps: Int, shift: Float32) raises -> List[Float32]:
    """Exact port of `build_sigma_schedule` (schedules.rs).

    Returns `num_steps + 1` sigmas, descending from 1.0 to 0.0. With `shift==1`
    this is the plain linear schedule `1 - i/N`; otherwise each value is bent by
    the flow-matching static shift `shift*t / (1 + (shift-1)*t)`.
    """
    if num_steps <= 0:
        raise Error("build_sigma_schedule: num_steps must be > 0")
    var out = List[Float32]()
    var n_f = Float32(num_steps)
    for i in range(num_steps + 1):
        out.append(1.0 - Float32(i) / n_f)
    # Apply the shift in-place (skip the no-op shift==1 case, matching Rust's
    # `(shift - 1.0).abs() > f32::EPSILON` guard exactly).
    var shift_delta = shift - 1.0
    if shift_delta < 0.0:
        shift_delta = -shift_delta
    if shift_delta > _F32_EPS:
        for i in range(len(out)):
            var t = out[i]
            out[i] = shift * t / (1.0 + (shift - 1.0) * t)
    return out^


def cfg(
    v_cond: Tensor, v_uncond: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Classifier-free guidance combine, matching euler.rs's CODE form `pred = v_cond + scale*(v_cond - v_uncond)`.

        pred = v_cond + scale * (v_cond - v_uncond)

    NOTE: this is the reference's actual raw CFG code, NOT the textbook
    `v_uncond + scale*(v_cond - v_uncond)`. They differ by a constant offset of
    `v_cond - v_uncond` (i.e. textbook uses guidance weight `scale` on the
    delta added to v_uncond; the Z-Image code adds it to v_cond). For diffusers
    ZImagePipeline parity, negate this raw CFG output before scheduler step.

    This is the Z-IMAGE CFG. Qwen-Image uses the textbook form + a norm rescale —
    see `cfg_qwen` below; do NOT use this for Qwen.
    """
    var diff = sub(v_cond, v_uncond, ctx)  # v_cond - v_uncond
    var scaled = mul_scalar(diff, scale, ctx)  # scale * (v_cond - v_uncond)
    return add(v_cond, scaled, ctx)  # v_cond + scale*(...)


# ─────────────────────────────────────────────────────────────────────────────
# Qwen-Image variant.
#
# Qwen-Image is the same flow-matching family as Z-Image (Euler update is
# IDENTICAL: x_next = x + v*(sigma_next - sigma)) but the SIGMA SCHEDULE and the
# CFG COMBINE differ. References (verified line-by-line):
#   * inference-flame/src/bin/qwenimage_gen.rs  (the Rust inference bin)
#   * diffusers FlowMatchEulerDiscreteScheduler  (set_timesteps + time_shift +
#     stretch_shift_to_terminal) with use_dynamic_shifting=True,
#     time_shift_type="exponential", shift_terminal=0.02
#   * diffusers pipeline_qwenimage.py __call__ (true-CFG + norm rescale)
#   * scheduler_config.json (qwen-image-2512): base_shift=0.5, max_shift=0.9,
#     base_image_seq_len=256, max_image_seq_len=8192, shift_terminal=0.02.
#
# Schedule differences vs Z-Image:
#   1. base sigmas = linspace(1.0, 1/N, N)  (N values, NOT N+1; the trailing 0.0
#      is appended AFTER the shift). Z-Image uses 1 - i/N over N+1 values.
#   2. dynamic EXPONENTIAL shift driven by `mu` (resolution-dependent):
#        mu = calculate_shift(seq_len)   (see qwen_mu below)
#        sigma = exp(mu) / (exp(mu) + (1/sigma - 1))      [time_shift_type=exp, shift=1]
#      Z-Image uses the STATIC shift  shift*t/(1+(shift-1)*t)  with shift=3.0.
#   3. stretch-to-terminal so the last pre-0 sigma == shift_terminal (0.02):
#        one_minus_z = 1 - sigma; scale = one_minus_z[-1] / (1 - shift_terminal)
#        sigma = 1 - one_minus_z / scale
#      Z-Image has no terminal stretch.
#
# CFG difference vs Z-Image (pipeline_qwenimage.py:704-708):
#   comb = v_uncond + scale*(v_cond - v_uncond)     # TEXTBOOK (Z-Image adds to cond)
#   out  = comb * (||v_cond||_lastdim / ||comb||_lastdim)   # per-row norm rescale
#
# Default steps = 50 (pipeline default). true_cfg_scale default = 4.0.

# Qwen-Image scheduler_config.json (qwen-image-2512) values.
comptime _QWEN_BASE_SHIFT: Float32 = 0.5
comptime _QWEN_MAX_SHIFT: Float32 = 0.9
comptime _QWEN_BASE_SEQ: Float32 = 256.0
comptime _QWEN_MAX_SEQ: Float32 = 8192.0
comptime _QWEN_SHIFT_TERMINAL: Float32 = 0.02


def qwen_mu(seq_len: Float32) -> Float32:
    """Resolution-dependent shift parameter `mu` (pipeline_qwenimage.calculate_shift).

        m  = (max_shift - base_shift) / (max_seq - base_seq)
        b  = base_shift - m*base_seq
        mu = seq_len*m + b

    `seq_len` is the packed token count (latent_h/2 * latent_w/2). The config's
    `max_image_seq_len`/`base_image_seq_len` and `max_shift`/`base_shift` win over
    the diffusers function defaults (1.15) because the pipeline passes them via
    scheduler.config.get(...). The Rust bin hardcodes the same config values.
    """
    var m = (_QWEN_MAX_SHIFT - _QWEN_BASE_SHIFT) / (_QWEN_MAX_SEQ - _QWEN_BASE_SEQ)
    var b = _QWEN_BASE_SHIFT - m * _QWEN_BASE_SEQ
    return seq_len * m + b


def build_qwen_sigma_schedule(
    num_steps: Int, seq_len: Float32
) raises -> List[Float32]:
    """Qwen-Image dynamic-exponential sigma schedule.

    Returns `num_steps + 1` sigmas (the leading 1.0 region, the bent body, then
    a trailing 0.0). Differs from Z-Image's static-shift schedule in every step
    except the elementwise Euler update that consumes it.
    """
    if num_steps <= 0:
        raise Error("build_qwen_sigma_schedule: num_steps must be > 0")
    var out = List[Float32]()
    # 1. linspace(1.0, 1/N, N)  -> N values (descending). With N steps the step
    #    between samples is (1 - 1/N)/(N-1); guard N==1 -> single value 1.0.
    var n_f = Float32(num_steps)
    if num_steps == 1:
        out.append(1.0)
    else:
        var lo = 1.0 / n_f
        var span = 1.0 - lo
        var denom = Float32(num_steps - 1)
        for i in range(num_steps):
            out.append(1.0 - (Float32(i) / denom) * span)
    # 2. exponential time-shift (shift=1.0 so (1/s - 1)^1 = 1/s - 1).
    var mu = qwen_mu(seq_len)
    var exp_mu = exp(mu)
    for i in range(len(out)):
        var s = out[i]
        out[i] = exp_mu / (exp_mu + (1.0 / s - 1.0))
    # 3. stretch-to-terminal so out[-1] lands on shift_terminal (0.02).
    var last = out[len(out) - 1]
    var one_minus_last = 1.0 - last
    var oml_abs = one_minus_last if one_minus_last >= 0.0 else -one_minus_last
    if oml_abs > 1e-12:
        var scale = one_minus_last / (1.0 - _QWEN_SHIFT_TERMINAL)
        for i in range(len(out)):
            var o = 1.0 - out[i]
            out[i] = 1.0 - o / scale
    # 4. append terminal 0.0  -> num_steps + 1 values.
    out.append(0.0)
    return out^


def cfg_qwen(
    v_cond: Tensor, v_uncond: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Qwen-Image true-CFG combine + per-row norm rescale (pipeline_qwenimage.py:704-708).

        comb = v_uncond + scale*(v_cond - v_uncond)        # TEXTBOOK form
        out  = comb * (||v_cond||_lastdim / ||comb||_lastdim)

    The L2 norm reduces the LAST dim (keepdim) and broadcasts back, so the
    per-token magnitude of the combined prediction is rescaled to match the
    conditional prediction's magnitude. Inputs are [B, seq, dim]; the norm is
    over `dim`. This differs from Z-Image's `cfg` in BOTH the combine (textbook
    vs cond-anchored) AND the extra norm rescale.

    The per-row L2 norm reduces the LAST dim. serenitymojo's ops layer has no
    device last-dim reduction yet, so the norm + ratio are computed host-side
    (a tiny [B, seq] worth of scalars) then broadcast back through the on-device
    `div`/`mul`. This is the schedule-style host-assist pattern, not a hot-path
    full-tensor roundtrip of activations: only the per-row norms cross the bus.
    When a device `sum(dim=-1, keepdim)` lands, swap `_l2_ratio_lastdim` for an
    on-device sqrt(sum(x^2)) and the rest is unchanged.
    """
    var diff = sub(v_cond, v_uncond, ctx)  # v_cond - v_uncond
    var scaled = mul_scalar(diff, scale, ctx)  # scale * (...)
    var comb = add(v_uncond, scaled, ctx)  # v_uncond + scale*(...)  TEXTBOOK
    # ratio[..., 0] = ||v_cond||_lastdim / ||comb||_lastdim, broadcast over dim.
    var ratio = _l2_ratio_lastdim(v_cond, comb, ctx)
    return mul(comb, ratio, ctx)


def _l2_ratio_lastdim(
    cond: Tensor, comb: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """ratio = ||cond||_lastdim / ||comb||_lastdim, shaped [..., 1] for broadcast.

    Both inputs are [..., dim]; we reduce the trailing `dim`. Returns a tensor
    with the last dim collapsed to 1 so `mul(comb, ratio)` broadcasts it back.
    Computed host-side (see `cfg_qwen` note) — only the row count of scalars moves.
    The reduction uses F32 host statistics, then stores the tiny ratio tensor in
    the prediction storage dtype so the full latent/prediction carrier stays BF16
    on BF16 inference paths.
    """
    if cond.dtype() != comb.dtype():
        raise Error("_l2_ratio_lastdim: cond/comb dtype mismatch")
    var cs = cond.shape()
    var ms = comb.shape()
    if len(cs) == 0 or len(ms) == 0:
        raise Error("_l2_ratio_lastdim: inputs must have >= 1 dim")
    var dim = cs[len(cs) - 1]
    if ms[len(ms) - 1] != dim:
        raise Error("_l2_ratio_lastdim: last dims must match")
    var cond_h = cond.to_host(ctx)
    var comb_h = comb.to_host(ctx)
    var n = len(cond_h)
    if len(comb_h) != n:
        raise Error("_l2_ratio_lastdim: cond/comb numel mismatch")
    var rows = n // dim
    var ratio_shape = List[Int]()
    for i in range(len(cs) - 1):
        ratio_shape.append(cs[i])
    ratio_shape.append(1)
    var ratio_h = List[Float32]()
    for r in range(rows):
        var cs_sum: Float32 = 0.0
        var ms_sum: Float32 = 0.0
        var base = r * dim
        for j in range(dim):
            var cv = cond_h[base + j]
            var mv = comb_h[base + j]
            cs_sum += cv * cv
            ms_sum += mv * mv
        var cn = sqrt(cs_sum)
        var mn = sqrt(ms_sum)
        ratio_h.append(cn / mn)
    return Tensor.from_host(ratio_h, ratio_shape^, cond.dtype(), ctx)


struct Scheduler(Movable):
    """Flow-matching (rectified-flow Euler) scheduler for Z-Image inference.

    Holds the precomputed sigma schedule. `step(latent, velocity, i)` performs
    one Euler update `x + v*(sigma[i+1] - sigma[i])`. The model supplies the
    velocity (already CFG-combined via `cfg()` if guidance is on).
    """

    var _sigmas: List[Float32]
    var num_steps: Int
    var shift: Float32

    def __init__(out self, num_steps: Int, shift: Float32 = 3.0) raises:
        """Build a scheduler for `num_steps` denoising steps at the given
        `shift` (Z-Image bin default = 3.0)."""
        self._sigmas = build_sigma_schedule(num_steps, shift)
        self.num_steps = num_steps
        self.shift = shift

    def __init__(
        out self, var sigmas: List[Float32], num_steps: Int, shift: Float32
    ):
        """Internal: build a scheduler directly from a precomputed sigma table.
        Used by `Scheduler.qwen` so the Qwen schedule reuses the same Euler
        `step()` machinery."""
        self._sigmas = sigmas^
        self.num_steps = num_steps
        self.shift = shift

    @staticmethod
    def qwen(num_steps: Int, seq_len: Float32) raises -> Scheduler:
        """Build a Qwen-Image scheduler (dynamic-exponential shift driven by the
        packed-token `seq_len`). The Euler `step()` update is identical to
        Z-Image's — only the sigma table changes (see `build_qwen_sigma_schedule`).
        `shift` is stored as `qwen_mu(seq_len)` for inspection (not a static shift).
        Default `num_steps` for Qwen-Image is 50."""
        var sigmas = build_qwen_sigma_schedule(num_steps, seq_len)
        return Scheduler(sigmas^, num_steps, qwen_mu(seq_len))

    def sigmas(self) -> List[Float32]:
        """The sigma schedule (num_steps + 1 values, 1.0 -> 0.0), copied out."""
        return self._sigmas.copy()

    def timesteps(self) -> List[Float32]:
        """Per-step timesteps fed to the model = the sigma at each step start.

        For Z-Image the model's timestep input IS the sigma value (euler.rs
        builds the timestep tensor directly from `sigma`), so the `num_steps`
        timesteps are `sigmas[0 .. num_steps-1]` (the trailing 0.0 sigma is the
        final target, never a model input)."""
        var out = List[Float32]()
        for i in range(self.num_steps):
            out.append(self._sigmas[i])
        return out^

    def sigma(self, i: Int) -> Float32:
        """Sigma at schedule index i (0 <= i <= num_steps)."""
        return self._sigmas[i]

    def step(
        self, latent: Tensor, velocity: Tensor, i: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """One rectified-flow Euler update for step `i` (0-based) `x_next = latent + velocity*(sigma[i+1] - sigma[i])`.

            x_next = latent + velocity * (sigma[i+1] - sigma[i])

        `velocity` is the model's velocity prediction at `sigma[i]` (CFG-combined
        beforehand if guidance is used). `dt` is negative since sigma descends.
        """
        if i < 0 or i >= self.num_steps:
            raise Error(
                String("Scheduler.step: i=")
                + String(i)
                + " out of range [0, "
                + String(self.num_steps)
                + ")"
            )
        var dt = self._sigmas[i + 1] - self._sigmas[i]
        var scaled = mul_scalar(velocity, dt, ctx)  # v * dt
        return add(latent, scaled, ctx)  # x + v*dt
