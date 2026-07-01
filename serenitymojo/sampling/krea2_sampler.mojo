# sampling/krea2_sampler.mojo — Krea-2 (krea2) flow-matching sampler.
#
# Reference: ai-toolkit krea2 src/pipeline.py — `timesteps()` (138-160) +
# `Krea2Pipeline.__call__` (185-260). Krea-2 is a PLAIN flow-matching model:
#   time runs t=1 (pure noise) -> t=0 (clean); velocity = noise - clean;
#   x_t = (1-t)*clean + t*noise. NO time flip / negation (pipeline.py:6-11).
#
# This module ports the three host/tensor pieces of the sampler (the DiT forward
# itself lives in models/dit/krea2_dit.krea2_forward):
#
#   1. krea2_timesteps  — the resolution-aware EXPONENTIAL-shift schedule.
#        ts = linspace(1, 0, steps+1)                       # steps+1 values, 1 -> 0
#        mu = slope*seq_len + (y1 - slope*x1)  (slope=(y2-y1)/(x2-x1))   [if mu unset]
#        ts = exp(mu) / (exp(mu) + (1/ts - 1)^sigma)         # sigma=1.0 default
#      where seq_len = gh*gw (PACKED image-token count = (H/8/2)*(W/8/2)),
#      x1 = (minres//(8*patch))^2 = (minres//16)^2, x2 = (maxres//16)^2,
#      y1=0.5, y2=1.15, minres=256, maxres=1280 (krea2.py model_kwargs defaults).
#      This is DISTINCT from build_qwen_sigma_schedule: krea2 uses steps+1 values
#      of a 1->0 linspace (NOT 1.0..1/N over N then append 0), has NO terminal
#      stretch, and uses the (1/ts - 1)^sigma power form (sigma=1).
#
#   2. krea2_cfg        — v_cond + scale*(v_cond - v_uncond)  (pipeline.py:251).
#      Identical CODE form to sampling/flow_match.cfg (the Z-Image cond-anchored
#      combine, NOT the qwen textbook+rescale). Reused via a thin wrapper so the
#      krea2 sampler is self-documenting.
#
#   3. krea2_euler_step — latents + (t_prev - t_cur)*v  (pipeline.py:254).
#      t descends (1->0) so (t_prev - t_cur) < 0; the latent accumulator stays F32
#      (pipeline.py:225 keeps latents F32; only the model FEED is bf16). This step
#      is numerically the same Euler update as flow_match (x + v*(sig_next-sig)).
#
# The schedule is tiny host-side F32 work; only the CFG combine and Euler update
# touch the GPU (through ops/tensor_algebra). Mojo 1.0.0b1. Inference-only.

from std.gpu.host import DeviceContext
from std.math import exp, log

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar


# krea2.py model_kwargs schedule defaults (krea2.py:206-211 / pipeline.py:142-145).
comptime KREA2_SCHED_Y1: Float32 = 0.5
comptime KREA2_SCHED_Y2: Float32 = 1.15
comptime KREA2_SCHED_MIN_RES: Int = 256
comptime KREA2_SCHED_MAX_RES: Int = 1280
comptime KREA2_SCHED_SIGMA: Float32 = 1.0
# align = vae_scale_factor(8) * patch(2) = 16  (pipeline.py:236).
comptime KREA2_ALIGN: Int = 16


def krea2_packed_seq_len(height: Int, width: Int) raises -> Int:
    """Packed image-token count gh*gw = (H/8/2)*(W/8/2) (pipeline.py:215-216).

    gh = height // (ae_scale * patch) = height // 16; gw = width // 16. This is
    the `seq_len` fed to the mu interpolation (NOT the padded main-block S).
    """
    if height <= 0 or width <= 0:
        raise Error("krea2_packed_seq_len: dims must be > 0")
    if height % KREA2_ALIGN != 0 or width % KREA2_ALIGN != 0:
        raise Error(
            "krea2_packed_seq_len: H and W must be divisible by 16 (8*patch)"
        )
    var gh = height // KREA2_ALIGN
    var gw = width // KREA2_ALIGN
    return gh * gw


def _krea2_pow(base: Float32, p: Float32) -> Float32:
    """base^p for the schedule's `(1/ts - 1)^sigma`. sigma defaults to 1.0 (the
    common path) -> returns base unchanged; the general form uses exp(p*log(base))
    only when base > 0. Mirrors torch's `** sigma` on the strictly-positive
    `(1/ts - 1)` (ts in (0,1) at the interior points)."""
    if p == Float32(1.0):
        return base
    if base <= Float32(0.0):
        return Float32(0.0)
    return exp(p * log(base))


def krea2_mu(seq_len: Float32) -> Float32:
    """Resolution-aware shift parameter mu (pipeline.py:156-158).

        x1 = (min_res // 16)^2 ; x2 = (max_res // 16)^2
        slope = (y2 - y1) / (x2 - x1)
        mu    = slope*seq_len + (y1 - slope*x1)

    seq_len is the PACKED image-token count gh*gw. With the krea2 defaults
    (min_res=256 -> x1=256, max_res=1280 -> x2=6400, y1=0.5, y2=1.15) this is a
    line through (256, 0.5) and (6400, 1.15).
    """
    var x1 = Float32((KREA2_SCHED_MIN_RES // KREA2_ALIGN) * (KREA2_SCHED_MIN_RES // KREA2_ALIGN))
    var x2 = Float32((KREA2_SCHED_MAX_RES // KREA2_ALIGN) * (KREA2_SCHED_MAX_RES // KREA2_ALIGN))
    var slope = (KREA2_SCHED_Y2 - KREA2_SCHED_Y1) / (x2 - x1)
    return slope * seq_len + (KREA2_SCHED_Y1 - slope * x1)


def krea2_timesteps(
    seq_len: Int,
    steps: Int,
    mu_override: Float32 = Float32(0.0),
    use_mu_override: Bool = False,
) raises -> List[Float32]:
    """Krea-2 exponential-shift flow-matching schedule (pipeline.py:138-160).

    Returns `steps + 1` time values descending 1 -> 0:
        ts = linspace(1, 0, steps+1)
        ts = exp(mu) / (exp(mu) + (1/ts - 1)^sigma)        # sigma = 1.0
    with mu = krea2_mu(seq_len) unless `use_mu_override` pins a constant mu (the
    distilled turbo checkpoint was trained at a fixed mu=1.15; pipeline.py:152-153).

    Endpoints: ts[0] -> exp(mu)/(exp(mu)+(1/1 - 1)) = exp(mu)/exp(mu) = 1.0;
    ts[steps] -> 1/ts = +inf so the term -> 0... torch evaluates 1/0 = inf, and
    exp(mu)/(exp(mu)+inf) = 0.0. We special-case the exact 1.0 and 0.0 endpoints
    to avoid the 1/0 division (matching torch's inf-limit -> 0.0).
    """
    if steps <= 0:
        raise Error("krea2_timesteps: steps must be > 0")
    var mu = mu_override if use_mu_override else krea2_mu(Float32(seq_len))
    var exp_mu = exp(mu)
    var n_f = Float32(steps)
    var out = List[Float32]()
    for i in range(steps + 1):
        # linspace(1, 0, steps+1): ts_i = 1 - i/steps.
        var ts = Float32(1.0) - Float32(i) / n_f
        if ts >= Float32(1.0):
            out.append(Float32(1.0))          # endpoint: exp(mu)/exp(mu) = 1.0
        elif ts <= Float32(0.0):
            out.append(Float32(0.0))          # endpoint: 1/ts -> inf -> 0.0
        else:
            var inner = _krea2_pow(Float32(1.0) / ts - Float32(1.0), KREA2_SCHED_SIGMA)
            out.append(exp_mu / (exp_mu + inner))
    return out^


def krea2_cfg(
    v_cond: Tensor, v_uncond: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Krea-2 CFG combine (pipeline.py:251): v = v_cond + scale*(v_cond - v_uncond).

    This is the cond-anchored CODE form (identical to sampling/flow_match.cfg),
    NOT the textbook `v_uncond + scale*(v_cond - v_uncond)`. Krea-2's pipeline
    uses exactly this; do not substitute the qwen variant.
    """
    var diff = sub(v_cond, v_uncond, ctx)          # v_cond - v_uncond
    var scaled = mul_scalar(diff, scale, ctx)      # scale * (...)
    return add(v_cond, scaled, ctx)                # v_cond + scale*(...)


def krea2_euler_step(
    latents: Tensor, v: Tensor, t_cur: Float32, t_prev: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Krea-2 Euler update (pipeline.py:254): latents + (t_prev - t_cur)*v.

    `latents` and `v` must be the SAME dtype. Elementwise kernels may use F32
    scalar math internally, but the latent carrier stays in `latents.dtype()`.
    t descends 1 -> 0 across the schedule so (t_prev - t_cur) < 0 — the latent
    walks from noise (t=1) to clean (t=0). Returns the next latent in
    `latents`' dtype.
    """
    var dt = t_prev - t_cur
    var step = mul_scalar(v, dt, ctx)              # (t_prev - t_cur) * v
    return add(latents, step, ctx)                 # latents + (...)
