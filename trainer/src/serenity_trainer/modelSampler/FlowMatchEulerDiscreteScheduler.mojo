# FlowMatchEulerDiscreteScheduler.mojo — 1:1 port of the diffusers scheduler
#   that Serenity's ZImageSampler drives:
#   venv/src/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
#   (FlowMatchEulerDiscreteScheduler.__init__ / set_timesteps / step).
#
# Z-Image scheduler_config.json (Tongyi-MAI/Z-Image, scheduler/scheduler_config.json):
#   num_train_timesteps = 1000
#   use_dynamic_shifting = false
#   shift                = 6.0
# Every other ctor flag defaults: base_shift=0.5, max_shift=1.15,
#   base_image_seq_len=256, max_image_seq_len=4096, invert_sigmas=False,
#   shift_terminal=None, use_karras/exponential/beta_sigmas=False,
#   time_shift_type="exponential", stochastic_sampling=False.
# Under that config the only active code paths are the plain shifted-linear
# sigma schedule and the deterministic Euler update — exactly the branches ported
# below. The karras/exponential/beta/dynamic/stochastic/per-token/invert branches
# are NOT reachable for Z-Image and are intentionally omitted.
#
# ── EXACT MATH (diffusers line cites) ─────────────────────────────────────────
#
# __init__ (scheduling_flow_match_euler_discrete.py:126-143):
#   timesteps = linspace(1, N, N)[::-1]              # N=num_train_timesteps
#   sigmas    = timesteps / N
#   sigmas    = shift*sigmas / (1 + (shift-1)*sigmas)   (:132, use_dynamic_shifting False)
#   sigma_max = sigmas[0]   = shift*1/(1+(shift-1)*1)        = 1.0          (:143)
#   sigma_min = sigmas[-1]  = shift*(1/N)/(1+(shift-1)*(1/N))               (:142)
#
# _sigma_to_t (:237-238):   t = sigma * N
#
# set_timesteps(n) (:282-384, default branches only):
#   timesteps = linspace(_sigma_to_t(sigma_max), _sigma_to_t(sigma_min), n)  (:335-339)
#   sigmas    = timesteps / N                                                 (:340)
#   sigmas    = shift*sigmas / (1 + (shift-1)*sigmas)                         (:350)
#   timesteps = sigmas * N                                                    (:367)
#   sigmas    = cat([sigmas, zeros(1)])   # terminal sigma 0                  (:379)
#   → self.sigmas has length n+1, self.timesteps has length n.
#
# step(model_output, t, sample) (:425-524, default branches only):
#   sigma      = self.sigmas[step_index]                                      (:501)
#   sigma_next = self.sigmas[step_index + 1]                                  (:502)
#   dt         = sigma_next - sigma                                           (:506)
#   prev_sample = sample + dt * model_output    (stochastic_sampling False)   (:513)
#   step_index += 1                                                           (:516)
#
# Verified numerically against the real diffusers (shift=6, n=8) — sigmas &
# timesteps match to f32 (see parity/zi_sampler_ref.safetensors sigmas/timesteps).
#
# DTYPE: schedule math is host-scalar F32. The Euler tensor update mirrors
# diffusers exactly — `sample` is upcast to F32 (:486), the whole update
# `sample + dt*model_output` is accumulated in F32 in one shot (:513), and the
# prev_sample is cast back to model_output.dtype once at the end (:519). This is
# a true F32 accumulation, NOT a BF16-storage two-op path (which would double-
# round dt*model_output then the sum).

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import add, mul_scalar
from serenitymojo.ops.cast import cast_tensor, cast_tensor_if_needed


comptime ZIMAGE_NUM_TRAIN_TIMESTEPS = 1000
comptime ZIMAGE_DEFAULT_SHIFT = Float32(6.0)


# ── the scheduler state (sigmas length n+1, timesteps length n) ───────────────
struct FlowMatchEulerDiscreteScheduler(Movable):
    var num_train_timesteps: Int
    var shift: Float32
    var sigmas: List[Float32]      # length num_inference_steps + 1 (terminal 0)
    var timesteps: List[Float32]   # length num_inference_steps
    var step_index: Int

    def __init__(out self, num_train_timesteps: Int, shift: Float32):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigmas = List[Float32]()
        self.timesteps = List[Float32]()
        self.step_index = 0

    # ── set_timesteps (diffusers :282-384, default config branches) ──────────
    def set_timesteps(mut self, num_inference_steps: Int):
        var n = num_inference_steps
        var N = Float32(self.num_train_timesteps)
        var shift = self.shift

        # __init__ sigma_max/sigma_min (diffusers :142-143). sigma_max == 1.0
        # exactly because shift*1/(1+(shift-1)*1) = shift/shift = 1.
        var sigma_max = Float32(1.0)
        var s_min_base = Float32(1.0) / N
        var sigma_min = shift * s_min_base / (Float32(1.0) + (shift - Float32(1.0)) * s_min_base)

        # _sigma_to_t(sigma_max/min) = sigma * N  (diffusers :237-238, :335-339)
        var t_start = sigma_max * N        # = N
        var t_end = sigma_min * N          # = sigma_min * N

        self.sigmas = List[Float32]()
        self.timesteps = List[Float32]()

        # 1. timesteps = linspace(t_start, t_end, n)   (diffusers :335-339)
        # numpy linspace endpoint=True: step = (end-start)/(n-1); n==1 → [start].
        for i in range(n):
            var t_lin: Float32
            if n == 1:
                t_lin = t_start
            else:
                var frac = Float32(i) / Float32(n - 1)
                t_lin = t_start + frac * (t_end - t_start)
            # 2. sigmas = timesteps / N            (diffusers :340)
            var sig = t_lin / N
            # 3. shift   (diffusers :350, use_dynamic_shifting False)
            sig = shift * sig / (Float32(1.0) + (shift - Float32(1.0)) * sig)
            self.sigmas.append(sig)
            # 4. timesteps = sigmas * N            (diffusers :367)
            self.timesteps.append(sig * N)

        # 6. append terminal sigma 0   (diffusers :379, invert_sigmas False)
        self.sigmas.append(Float32(0.0))
        self.step_index = 0

    # ── step (diffusers :425-524, deterministic Euler branch) ────────────────
    # prev_sample = sample + (sigma_next - sigma) * model_output   (:506-513)
    # `index` is the loop index i (diffusers tracks step_index internally and
    # advances by 1 per call; we pass it explicitly for a stateless tensor op).
    #
    # PRECISION (diffusers :486, :513, :519): diffusers upcasts `sample` to F32,
    # computes the WHOLE Euler update `sample + dt*model_output` in F32 (one
    # accumulation, no intermediate rounding), then casts the result back to
    # `model_output.dtype` exactly once. We replicate that here: cast both
    # operands to F32, do mul_scalar + add in F32 (the f32 elementwise kernels
    # keep full precision), then cast the prev_sample back to model_output's
    # dtype. This avoids the BF16 double-round (dt*model_output rounded to BF16,
    # then the sum rounded again) that a BF16-storage two-op path would incur.
    def step(
        self, model_output: Tensor, sample: Tensor, index: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var sigma = self.sigmas[index]
        var sigma_next = self.sigmas[index + 1]
        var dt = sigma_next - sigma                                  # diffusers :506
        # Upcast to F32 (diffusers :486 upcasts `sample`; `dt`*bf16 also promotes
        # to f32 in torch). Compute the update entirely in F32.
        var mo_f32 = cast_tensor(model_output, STDtype.F32, ctx)
        var sample_f32 = cast_tensor(sample, STDtype.F32, ctx)       # diffusers :486
        var delta = mul_scalar(mo_f32, dt, ctx)                      # dt * model_output (f32)
        var prev_f32 = add(sample_f32, delta, ctx)                   # sample + dt*model_output (f32)  :513
        # Cast prev_sample back to model_output's dtype (diffusers :519).
        return cast_tensor_if_needed(prev_f32^, model_output.dtype(), ctx)


# Convenience: build + set_timesteps in one call (matches Serenity
# ZImageSampler.py:88 `noise_scheduler.set_timesteps(diffusion_steps)` on the
# Z-Image-configured scheduler).
def make_zimage_scheduler(
    diffusion_steps: Int, shift: Float32 = ZIMAGE_DEFAULT_SHIFT
) raises -> FlowMatchEulerDiscreteScheduler:
    var sch = FlowMatchEulerDiscreteScheduler(ZIMAGE_NUM_TRAIN_TIMESTEPS, shift)
    sch.set_timesteps(diffusion_steps)
    return sch^


from std.math import exp as _exp


# ══════════════════════════════════════════════════════════════════════════════
# FLUX.2 (Klein) sampler scheduler path — DISTINCT from the Z-Image default path.
#
# Flux2Sampler.py (:96-101) drives the scheduler with BOTH an empirical mu AND a
# CUSTOM sigmas array, i.e. the diffusers DYNAMIC-SHIFTING + custom-sigmas branch:
#   image_seq_len = packed latent seq len  (Flux2Sampler.py:95)
#   mu     = compute_empirical_mu(image_seq_len, diffusion_steps)   (:96)
#   sigmas = np.linspace(1.0, 1/diffusion_steps, diffusion_steps)   (:100)
#   noise_scheduler.set_timesteps(diffusion_steps, mu=mu, sigmas=sigmas)  (:101)
#
# diffusers set_timesteps (scheduling_flow_match_euler_discrete.py:282-385), with
# sigmas provided and use_dynamic_shifting=True (mu passed; the FLUX.2 scheduler
# config sets use_dynamic_shifting=True — confirmed by the sampler passing mu):
#   sigmas = np.array(sigmas)                                       (:357)  → the linspace
#   sigmas = time_shift(mu, 1.0, sigmas)                           (:347-348, dynamic)
#     time_shift_type "exponential" (default):
#       _time_shift_exponential(mu, sigma=1.0, t):                 (:647-649)
#         exp(mu) / (exp(mu) + (1/t - 1)**1.0)
#   timesteps = sigmas * num_train_timesteps                       (:380, not is_timesteps_provided)
#   sigmas = cat([sigmas, zeros(1)])  (invert_sigmas False)        (:379)
#   → self.sigmas length n+1, self.timesteps length n.
#
# compute_empirical_mu (pipeline_flux2.py:159-176), VERBATIM:
#   a1,b1 = 8.73809524e-05, 1.89833333
#   a2,b2 = 0.00016927, 0.45666666
#   if image_seq_len > 4300: return a2*image_seq_len + b2
#   m_200 = a2*image_seq_len + b2
#   m_10  = a1*image_seq_len + b1
#   a = (m_200 - m_10) / 190.0
#   b = m_200 - 200.0*a
#   return a*num_steps + b
#
# The Euler step() is SHARED with the Z-Image path (diffusers :425-524 default
# branch) — same prev = sample + (sigma_next - sigma)*model_output, F32-accumulated.
comptime FLUX2_NUM_TRAIN_TIMESTEPS = 1000


def flux2_compute_empirical_mu(image_seq_len: Int, num_steps: Int) -> Float32:
    var a1 = Float32(8.73809524e-05)
    var b1 = Float32(1.89833333)
    var a2 = Float32(0.00016927)
    var b2 = Float32(0.45666666)
    var isl = Float32(image_seq_len)
    if image_seq_len > 4300:
        return a2 * isl + b2
    var m_200 = a2 * isl + b2
    var m_10 = a1 * isl + b1
    var a = (m_200 - m_10) / Float32(190.0)
    var b = m_200 - Float32(200.0) * a
    return a * Float32(num_steps) + b


# Flux2 set_timesteps: dynamic-shift (mu) + custom linspace sigmas. Populates a
# FlowMatchEulerDiscreteScheduler in place (sigmas len n+1, timesteps len n).
def flux2_set_timesteps(
    mut sch: FlowMatchEulerDiscreteScheduler, num_inference_steps: Int, mu: Float32
):
    var n = num_inference_steps
    var N = Float32(sch.num_train_timesteps)
    var exp_mu = _exp(mu)

    sch.sigmas = List[Float32]()
    sch.timesteps = List[Float32]()

    # sigmas = np.linspace(1.0, 1/n, n)  (Flux2Sampler.py:100). numpy linspace
    # endpoint=True: step=(end-start)/(n-1); n==1 → [1.0].
    var t_start = Float32(1.0)
    var t_end = Float32(1.0) / Float32(n)
    for i in range(n):
        var sig: Float32
        if n == 1:
            sig = t_start
        else:
            var frac = Float32(i) / Float32(n - 1)
            sig = t_start + frac * (t_end - t_start)
        # dynamic time_shift exponential: exp(mu)/(exp(mu) + (1/sig - 1)^1.0)
        var inv = (Float32(1.0) / sig) - Float32(1.0)
        sig = exp_mu / (exp_mu + inv)
        sch.sigmas.append(sig)
        sch.timesteps.append(sig * N)   # timesteps = sigmas * N

    sch.sigmas.append(Float32(0.0))     # terminal sigma 0
    sch.step_index = 0


# Convenience: build a FLUX.2 scheduler and run its (mu, custom-sigmas) timestep
# prep. `image_seq_len` is the PACKED latent sequence length (Flux2Sampler.py:95).
def make_flux2_scheduler(
    diffusion_steps: Int, image_seq_len: Int
) raises -> FlowMatchEulerDiscreteScheduler:
    var mu = flux2_compute_empirical_mu(image_seq_len, diffusion_steps)
    # shift is unused on the dynamic path (time_shift uses mu, not self.shift); pass
    # the FlowMatchEuler default 1.0 for the ctor.
    var sch = FlowMatchEulerDiscreteScheduler(FLUX2_NUM_TRAIN_TIMESTEPS, Float32(1.0))
    flux2_set_timesteps(sch, diffusion_steps, mu)
    return sch^
