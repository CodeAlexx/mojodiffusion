# 1:1 port of Serenity modules/modelSetup/mixin/ModelSetupNoiseMixin.py
# Source of truth: /home/alex/Serenity/modules/modelSetup/mixin/ModelSetupNoiseMixin.py
#
# Ported in this file:
#   _compute_and_cache_offset_noise_psi_schedule  (reference trainer L19-74)  — generalized
#       offset-noise psi_t schedule (Algorithm 1 of the "Generalized Diffusion
#       Model with Adjusted Offset Noise" paper). Used by _create_noise.
#   _create_noise                                  (reference trainer L77-119) — base randn +
#       offset_noise (+ generalized psi_t) + perturbation_noise branches.
#   _get_timestep_discrete                         (reference trainer L121-212)— deterministic /
#       UNIFORM / LOGIT_NORMAL / HEAVY_TAIL continuous branches + the timestep
#       shift, and the discrete COS_MAP / SIGMOID / INVERTED_PARABOLA branches
#       (linspace + linspace_derivative + multinomial).
#   _get_timestep_continuous                       (reference trainer L214-238)— deterministic
#       torch.full(0.5) branch + the discrete_timesteps=10000 / +1 / .float()/10000
#       flow-matching branch (delegates to the shared discrete host math).
#
# DTYPE policy: BF16 storage, F32 compute. The full-size noise tensors live on
# device and go through serenitymojo ops. The timestep computation is per-batch
# scalar math (batch_size values); torch does it on F64/F32 — we mirror it on the
# host in F64 to match torch's element-for-element order of operations, then emit
# an INT32 device tensor (torch returns .int(); torch.long internally for the
# discrete multinomial path, cast to int at the end — reference trainer L210/L212).
#
# RNG STREAM DIFFERENCE (documented, per port constraints):
#   Serenity draws from torch.Generator(seed): torch.randn / torch.rand /
#   torch.normal / torch.multinomial. serenitymojo's RNG is a ChaCha12/PCG32
#   stream (rand-0.8.5 StdRng-compatible), NOT torch's Philox. The numeric VALUES
#   therefore will NOT bit-match torch; the FORMULAS / order-of-operations / dtype
#   / branch structure are identical. Numeric parity is verified against fixed
#   dumped tensors where the RNG draw matters (per task verification protocol).
#   - Full-tensor Gaussian noise: serenitymojo.ops.random.randn (Box-Muller).
#   - Per-batch uniform U(0,1): serenitymojo host primitive _standard_f64 of one
#     ChaCha word (matches torch.rand role).
#   - Per-batch N(mu,sigma): serenitymojo host primitive _standard_normal_at
#     (Box-Muller) scaled+shifted (matches torch.normal role).
#   - multinomial: faithful port of the aten CPU multinomial kernel — unnormalized
#     cumsum(weights), one U(0,1) per draw, lower_bound (binary search, >= tie
#     semantics) on cumdist against uniform_sample*sum, clamped to [0, n-1]
#     (replacement=True). See get_timestep_discrete discrete branch (SKEPTIC P0-1).

from std.gpu.host import DeviceContext
from std.math import exp, sqrt, cos

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, mul_scalar, mul

# Host RNG primitives from the serenitymojo numerical foundation
# (training/schedule.mojo). These are the rand-0.8.5/ChaCha12 host draws; we reuse
# them as the RNG source for the per-batch uniform / normal / multinomial draws.
from serenitymojo.training.schedule import (
    _expand_key,
    _standard_f64,
    _standard_normal_at,
    _chacha12_word_from_key,
)

# TimestepDistribution enum values (1:1 with
# Serenity modules/util/enum/TimestepDistribution.py).
comptime TSD_UNIFORM = 0
comptime TSD_SIGMOID = 1
comptime TSD_LOGIT_NORMAL = 2
comptime TSD_HEAVY_TAIL = 3
comptime TSD_COS_MAP = 4
comptime TSD_INVERTED_PARABOLA = 5

comptime _PI = Float64(3.141592653589793238462643383279502884)


# Pack host Int values into a fresh INT32 device tensor (reference trainer returns timestep.int()
# / torch.long → .int(); these are exact integer values). Tensor.from_host only
# packs float compute dtypes, so the integer buffer is built directly here,
# mirroring the from_host H2D pattern.
def _int32_tensor_from_host(
    values: List[Int], var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var n = len(values)
    var nbytes = n * 4
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var ip = host.unsafe_ptr().bitcast[Int32]()
    for i in range(n):
        ip[i] = Int32(values[i])
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, shape^, STDtype.I32)


# Pack host Int values into a fresh INT64 device tensor. Used by the deterministic
# branch of _get_timestep_discrete, which in reference trainer returns torch.tensor(...,
# dtype=torch.long) WITHOUT the trailing .int() (reference trainer L135-139 vs L212 — the .int()
# at L212 only applies to the non-deterministic return). See SKEPTIC P3-6.
def _int64_tensor_from_host(
    values: List[Int], var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var n = len(values)
    var nbytes = n * 8
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var ip = host.unsafe_ptr().bitcast[Int64]()
    for i in range(n):
        ip[i] = Int64(values[i])
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, shape^, STDtype.I64)


# ─────────────────────────────────────────────────────────────────────────────
# _compute_and_cache_offset_noise_psi_schedule  (reference trainer L19-74)
#
# Computes the time-dependent psi_t coefficients for generalized offset noise.
# Pure host F64 math (torch does betas.to(torch.float64) at reference trainer L28, all ops F64).
# Caching (self._offset_noise_psi_schedule, reference trainer L25-26/L73) is handled by the
# caller in Mojo (no `self`); this is the pure compute. Returns the psi schedule
# as a host List[Float64] of length T (indexable by timestep, reference trainer L103 psi_t =
# psi_schedule[timestep]).
# ─────────────────────────────────────────────────────────────────────────────
def compute_offset_noise_psi_schedule(betas_host: List[Float32]) -> List[Float64]:
    var T = len(betas_host)

    # betas = betas.to(torch.float64)                                  (reference trainer L28)
    var betas = List[Float64]()
    for i in range(T):
        betas.append(Float64(betas_host[i]))

    # alphas = 1.0 - betas                                            (reference trainer L30)
    var alphas = List[Float64]()
    for i in range(T):
        alphas.append(Float64(1.0) - betas[i])

    # alphas_cumprod = torch.cumprod(alphas, dim=0)                   (reference trainer L31)
    var alphas_cumprod = List[Float64]()
    var acc = Float64(1.0)
    for i in range(T):
        acc = acc * alphas[i]
        alphas_cumprod.append(acc)

    # alphas_cumprod_prev = cat([tensor([1.0]), alphas_cumprod[:-1]]) (reference trainer L34)
    var alphas_cumprod_prev = List[Float64]()
    alphas_cumprod_prev.append(Float64(1.0))
    for i in range(T - 1):
        alphas_cumprod_prev.append(alphas_cumprod[i])

    # gammas = torch.zeros(T)                                          (reference trainer L37)
    var gammas = List[Float64]()
    for _ in range(T):
        gammas.append(Float64(0.0))

    # Step 1: gammas[0] = 1.0                                          (reference trainer L40)
    gammas[0] = Float64(1.0)

    # cumulative_sum_term = gammas[0] / sqrt(alphas_cumprod_prev[0])   (reference trainer L43)
    var cumulative_sum_term = gammas[0] / sqrt(alphas_cumprod_prev[0])

    # Step 2-4: for t in range(1, T)                                  (reference trainer L46-58)
    for t in range(1, T):
        var alpha_t = alphas[t]                                       # reference trainer L47
        var alpha_cumprod_tm1 = alphas_cumprod_prev[t]                # reference trainer L48
        # c_t_denominator = alpha_t * (1 - alpha_cumprod_tm1)          (reference trainer L51)
        var c_t_denominator = alpha_t * (Float64(1.0) - alpha_cumprod_tm1)
        # c_t = (1 - alpha_t) * sqrt(alpha_cumprod_tm1) / c_t_denominator  (reference trainer L52)
        var c_t = (Float64(1.0) - alpha_t) * sqrt(alpha_cumprod_tm1) / c_t_denominator
        # gammas[t] = c_t * cumulative_sum_term                        (reference trainer L55)
        gammas[t] = c_t * cumulative_sum_term
        # cumulative_sum_term += gammas[t] / sqrt(alphas_cumprod_prev[t])  (reference trainer L58)
        cumulative_sum_term = cumulative_sum_term + gammas[t] / sqrt(alphas_cumprod_prev[t])

    # Step 5: psi_T_denominator = sqrt(1 - alphas_cumprod[-1])         (reference trainer L61)
    var psi_T_denominator = sqrt(Float64(1.0) - alphas_cumprod[T - 1])
    # psi_T = cumulative_sum_term / psi_T_denominator                 (reference trainer L62)
    var psi_T = cumulative_sum_term / psi_T_denominator

    # Step 6-8: gammas_normalized = gammas / psi_T                    (reference trainer L65)
    var gammas_normalized = List[Float64]()
    for i in range(T):
        gammas_normalized.append(gammas[i] / psi_T)

    # terms = gammas_normalized / sqrt(alphas_cumprod_prev)           (reference trainer L69)
    # s_cumulative = torch.cumsum(terms, dim=0)                       (reference trainer L70)
    # psi_schedule = s_cumulative / sqrt(1 - alphas_cumprod)          (reference trainer L71)
    var psi_schedule = List[Float64]()
    var s_cumulative = Float64(0.0)
    for i in range(T):
        var term = gammas_normalized[i] / sqrt(alphas_cumprod_prev[i])
        s_cumulative = s_cumulative + term
        psi_schedule.append(s_cumulative / sqrt(Float64(1.0) - alphas_cumprod[i]))

    return psi_schedule^


# ─────────────────────────────────────────────────────────────────────────────
# _create_noise  (reference trainer L77-119)
#
# torch.randn base noise + offset_noise branch (with optional generalized psi_t)
# + perturbation_noise branch. All three branches ported, gated on config values.
#
# Mojo signature passes the config scalars directly (no TrainConfig struct here):
#   offset_noise_weight        = config.offset_noise_weight
#   generalized_offset_noise   = config.generalized_offset_noise
#   perturbation_noise_weight  = config.perturbation_noise_weight
# `timestep` (host int per batch element) and `betas_host` are only consulted on
# the generalized branch (reference trainer L101). `psi_schedule` is the cached host schedule
# from compute_offset_noise_psi_schedule (caller caches; passed in here). When the
# generalized branch is not taken the caller may pass an empty psi_schedule.
#
# `seed` / `seed_offset` / `seed_perturb` select the deterministic randn draws
# (torch.Generator advances between the three torch.randn calls; we use distinct
# seeds — RNG stream difference documented in the module header).
# ─────────────────────────────────────────────────────────────────────────────
def create_noise(
    source_tensor: Tensor,
    offset_noise_weight: Float32,
    generalized_offset_noise: Bool,
    perturbation_noise_weight: Float32,
    timestep: List[Int],
    psi_schedule: List[Float64],
    seed: UInt64,
    seed_offset: UInt64,
    seed_perturb: UInt64,
    ctx: DeviceContext,
) raises -> Tensor:
    var dt = source_tensor.dtype()
    var shape = source_tensor.shape()
    var ndim = len(shape)

    # noise = torch.randn(source_tensor.shape, dtype=source_tensor.dtype)  (reference trainer L85-90)
    var noise = randn(shape.copy(), seed, dt, ctx)

    # if config.offset_noise_weight > 0:                              (reference trainer L92)
    if offset_noise_weight > Float32(0.0):
        # offset_noise shape = (shape[0], shape[1], *[1]*(ndim-2))    (reference trainer L93-98)
        var off_shape = List[Int]()
        off_shape.append(shape[0])
        off_shape.append(shape[1])
        for _ in range(ndim - 2):
            off_shape.append(1)
        var offset_noise = randn(off_shape.copy(), seed_offset, dt, ctx)

        # if config.generalized_offset_noise and timestep is not None
        #        and betas is not None:                               (reference trainer L101)
        if generalized_offset_noise and len(timestep) > 0 and len(psi_schedule) > 0:
            # psi_t = psi_schedule[timestep]                          (reference trainer L103)
            # psi_t = psi_t.view(psi_t.shape[0], *[1]*(ndim-1))       (reference trainer L104)
            var B = len(timestep)
            var psi_vals = List[Float32]()
            for i in range(B):
                psi_vals.append(Float32(psi_schedule[timestep[i]]))
            var psi_view_shape = List[Int]()
            psi_view_shape.append(B)
            for _ in range(ndim - 1):
                psi_view_shape.append(1)
            var psi_t = Tensor.from_host(psi_vals, psi_view_shape.copy(), dt, ctx)

            # noise = noise + (psi_t * config.offset_noise_weight * offset_noise)
            #                                                          (reference trainer L106)
            var scaled_off = mul_scalar(offset_noise, offset_noise_weight, ctx)
            var psi_scaled = mul(psi_t, scaled_off, ctx)
            noise = add(noise, psi_scaled, ctx)
        else:
            # noise = noise + (config.offset_noise_weight * offset_noise)  (reference trainer L108)
            var scaled_off = mul_scalar(offset_noise, offset_noise_weight, ctx)
            noise = add(noise, scaled_off, ctx)

    # if config.perturbation_noise_weight > 0:                        (reference trainer L110)
    if perturbation_noise_weight > Float32(0.0):
        # perturbation_noise = torch.randn(source_tensor.shape, ...)  (reference trainer L111-116)
        var perturbation_noise = randn(shape.copy(), seed_perturb, dt, ctx)
        # noise = noise + (config.perturbation_noise_weight * perturbation_noise)
        #                                                              (reference trainer L117)
        var scaled_pert = mul_scalar(perturbation_noise, perturbation_noise_weight, ctx)
        noise = add(noise, scaled_pert, ctx)

    return noise^


# ─────────────────────────────────────────────────────────────────────────────
# Host RNG helpers (per-batch scalar draws). These mirror the role of
# torch.rand / torch.normal under torch.Generator. We advance the ChaCha word
# position per draw, deterministic from `seed`.
# ─────────────────────────────────────────────────────────────────────────────

# One U(0,1) draw at the given pair index (one ChaCha word at word_pos = pair).
# Matches torch.rand role (RNG stream differs — see module header).
def _uniform_at(ks: List[UInt32], pair: UInt64) -> Float64:
    var word_pos = pair
    var block = word_pos // 16
    var offset = Int(word_pos % 16)
    var w = _chacha12_word_from_key(
        ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], block, offset
    )
    return _standard_f64(w)


# ─────────────────────────────────────────────────────────────────────────────
# _get_timestep_discrete  (reference trainer L121-212)
#
# Returns batch_size int32 timesteps. Computed on host in F64 to match torch's
# element-wise order of operations, then uploaded as an INT32 device tensor
# (torch returns timestep.int(), reference trainer L212). The discrete (multinomial) branch
# returns torch.long internally then .int() — we emit int32 directly.
#
# Config scalars passed in (1:1 with TrainConfig fields):
#   min_noising_strength  (default 0.0)   max_noising_strength (default 1.0)
#   timestep_distribution (TSD_* enum)    noising_weight (default 0.0)
#   noising_bias (default 0.0)            timestep_shift via `shift`
# If shift is passed as a NaN sentinel the caller must have resolved
# config.timestep_shift first (reference trainer L130-131); here `shift` is always resolved.
#
# `weights_cache` mirrors self.__weights (reference trainer L188/L193/L202): the discrete weight
# vector is computed once and cached by the caller. Pass an empty list to force
# (re)computation; the (possibly newly computed) weights are returned via the
# out-tuple so the caller can cache them.
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct TimestepResult(Movable):
    var timestep: Tensor            # int32 device tensor, shape (batch_size,) or (1,)
    var weights: List[Float64]      # discrete weights cache (empty for continuous)


# Host-int result of the NON-deterministic _get_timestep_discrete numeric path
# (reference trainer L140-210, before the final .int()/device upload). Returned as host List[Int]
# so both get_timestep_discrete (device int32 tensor) and get_timestep_continuous
# (reference trainer L229-237: `+ 1`, `.float() / 10000`) can consume the SAME verified math
# without a device→host readback. `weights` carries the (possibly newly computed)
# discrete weights cache (reference trainer L188/193/202 self.__weights), empty for continuous.
@fieldwise_init
struct TimestepHostVals(Movable):
    var values: List[Int]
    var weights: List[Float64]


# Faithful port of the NON-deterministic body of _get_timestep_discrete
# (reference trainer L140-210). Returns host int values (post `+ min_timestep`, pre `.int()` cast
# which is exact for these integer values). Shared by get_timestep_discrete and
# get_timestep_continuous so the verified continuous/discrete math lives once.
def _get_timestep_discrete_host(
    num_train_timesteps: Int,
    batch_size: Int,
    timestep_distribution: Int,
    min_noising_strength: Float64,
    max_noising_strength: Float64,
    noising_weight: Float64,
    noising_bias: Float64,
    shift: Float64,
    weights_cache: List[Float64],
    seed: UInt64,
) raises -> TimestepHostVals:
    # min_timestep = int(num_train_timesteps * config.min_noising_strength)  (reference trainer L141)
    # (params are Float64 — original config F64 values passed by the caller,
    #  no F32→F64 widening, eliminating the boundary-rounding hazard. SKEPTIC #5)
    var min_timestep = Int(Float64(num_train_timesteps) * min_noising_strength)
    # max_timestep = int(num_train_timesteps * config.max_noising_strength)  (reference trainer L142)
    var max_timestep = Int(Float64(num_train_timesteps) * max_noising_strength)
    # num_timestep = max_timestep - min_timestep                      (reference trainer L143)
    var num_timestep = max_timestep - min_timestep

    var ks = _expand_key(seed)
    var shift64 = shift
    var N64 = Float64(num_train_timesteps)

    # Continuous branches: UNIFORM / LOGIT_NORMAL / HEAVY_TAIL        (reference trainer L145-172)
    if (
        timestep_distribution == TSD_UNIFORM
        or timestep_distribution == TSD_LOGIT_NORMAL
        or timestep_distribution == TSD_HEAVY_TAIL
    ):
        var out_vals = List[Int]()
        for b in range(batch_size):
            var timestep_b = Float64(0.0)

            if timestep_distribution == TSD_UNIFORM:
                # timestep = min_timestep + (max_timestep - min_timestep) * rand  (reference trainer L152-153)
                var u = _uniform_at(ks, UInt64(b))
                timestep_b = Float64(min_timestep) + Float64(max_timestep - min_timestep) * u
            elif timestep_distribution == TSD_LOGIT_NORMAL:
                # bias = config.noising_bias                          (reference trainer L155)
                var bias = Float64(noising_bias)
                # scale = config.noising_weight + 1.0                 (reference trainer L156)
                var scale = Float64(noising_weight) + Float64(1.0)
                # normal = torch.normal(bias, scale, size=(batch_size,))  (reference trainer L158)
                var draw = _standard_normal_at(
                    ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7], UInt64(b)
                )
                var normal = bias + scale * draw.z
                # logit_normal = normal.sigmoid()                     (reference trainer L159)
                var logit_normal = Float64(1.0) / (Float64(1.0) + exp(-normal))
                # timestep = logit_normal * num_timestep + min_timestep  (reference trainer L160)
                timestep_b = logit_normal * Float64(num_timestep) + Float64(min_timestep)
            else:  # TSD_HEAVY_TAIL                                   (reference trainer L161-170)
                # scale = config.noising_weight                       (reference trainer L162)
                var scale = Float64(noising_weight)
                # u = torch.rand(size=(batch_size,))                  (reference trainer L164-168)
                var u = _uniform_at(ks, UInt64(b))
                # u = 1.0 - u - scale * (cos(pi/2 * u)**2 - 1.0 + u)   (reference trainer L169)
                var c = cos(_PI / Float64(2.0) * u)
                u = Float64(1.0) - u - scale * (c * c - Float64(1.0) + u)
                # timestep = u * num_timestep + min_timestep          (reference trainer L170)
                timestep_b = u * Float64(num_timestep) + Float64(min_timestep)

            # timestep = N*shift*timestep / ((shift-1)*timestep + N)  (reference trainer L172)
            var shifted = (
                N64 * shift64 * timestep_b
                / ((shift64 - Float64(1.0)) * timestep_b + N64)
            )
            # return timestep.int()  (truncate toward zero)           (reference trainer L212)
            out_vals.append(Int(shifted))

        return TimestepHostVals(out_vals^, List[Float64]())

    # Discrete branches: COS_MAP / SIGMOID / INVERTED_PARABOLA        (reference trainer L173-210)
    else:
        # torch.linspace(0, 1, num_timestep) requires num_timestep >= 1
        # (steps == 0 raises "number of steps must be non-negative" is allowed
        #  but yields an empty tensor, and torch.multinomial on an empty weights
        #  vector then raises). Mirror torch by raising on num_timestep < 1
        #  instead of silently returning idx = -1 (SKEPTIC P2-4).
        if num_timestep < 1:
            raise Error(
                "num_timestep must be >= 1 for the discrete timestep "
                "distribution (max_noising_strength must exceed "
                "min_noising_strength)"
            )
        var weights = List[Float64]()
        # if self.__weights is None: ... (compute once, cached)       (reference trainer L188/193/202)
        if len(weights_cache) == 0:
            # linspace = torch.linspace(0, 1, num_timestep)           (reference trainer L180)
            # linspace = linspace / (shift - shift*linspace + linspace)  (reference trainer L181)
            # linspace_derivative = torch.linspace(0, 1, num_timestep) (reference trainer L183)
            # linspace_derivative = shift / (shift + ld - ld*shift).pow(2)  (reference trainer L184)
            var linspace = List[Float64]()
            var linspace_deriv = List[Float64]()
            for k in range(num_timestep):
                # torch.linspace(0,1,n): step = 1/(n-1); endpoint inclusive.
                var raw: Float64
                if num_timestep == 1:
                    raw = Float64(0.0)
                else:
                    raw = Float64(k) / Float64(num_timestep - 1)
                # shifted linspace (reference trainer L181)
                var ls = raw / (shift64 - shift64 * raw + raw)
                linspace.append(ls)
                # derivative of inverse shift (reference trainer L184): uses RAW linspace value
                var denom = shift64 + raw - (raw * shift64)
                linspace_deriv.append(shift64 / (denom * denom))

            for k in range(num_timestep):
                var w: Float64
                if timestep_distribution == TSD_COS_MAP:
                    # weights = 2 / (pi - 2*pi*ls + 2*pi*ls**2)        (reference trainer L189)
                    var ls = linspace[k]
                    w = Float64(2.0) / (
                        _PI - Float64(2.0) * _PI * ls + Float64(2.0) * _PI * ls * ls
                    )
                    # weights *= linspace_derivative                  (reference trainer L190)
                    w = w * linspace_deriv[k]
                elif timestep_distribution == TSD_SIGMOID:
                    # bias = config.noising_bias + 0.5                (reference trainer L194)
                    var bias = Float64(noising_bias) + Float64(0.5)
                    # weight = config.noising_weight                  (reference trainer L195)
                    var weight = Float64(noising_weight)
                    # weights = linspace / (shift - shift*linspace + linspace)  (reference trainer L197)
                    var ls = linspace[k]
                    var ww = ls / (shift64 - shift64 * ls + ls)
                    # weights = 1 / (1 + exp(-weight*(weights - bias)))  (reference trainer L198)
                    ww = Float64(1.0) / (Float64(1.0) + exp(-weight * (ww - bias)))
                    # weights *= linspace_derivative                  (reference trainer L199)
                    w = ww * linspace_deriv[k]
                else:  # TSD_INVERTED_PARABOLA                        (reference trainer L201-207)
                    # bias = config.noising_bias + 0.5                (reference trainer L203)
                    var bias = Float64(noising_bias) + Float64(0.5)
                    # weight = config.noising_weight                  (reference trainer L204)
                    var weight = Float64(noising_weight)
                    # weights = clamp(-weight*((linspace - bias)**2) + 2, min=0)  (reference trainer L206)
                    var ls = linspace[k]
                    var diff = ls - bias
                    var ww = -weight * (diff * diff) + Float64(2.0)
                    if ww < Float64(0.0):
                        ww = Float64(0.0)
                    # weights *= linspace_derivative                  (reference trainer L207)
                    w = ww * linspace_deriv[k]
                weights.append(w)
        else:
            for k in range(len(weights_cache)):
                weights.append(weights_cache[k])

        # samples = torch.multinomial(weights, num_samples=batch_size,
        #     replacement=True) + min_timestep                       (reference trainer L209)
        # timestep = samples.to(dtype=torch.long)                    (reference trainer L210)
        #
        # Faithful port of the aten CPU multinomial kernel
        # (aten/src/ATen/native/cpu/MultinomialKernel.cpp,
        #  multinomial_with_replacement_apply):
        #   1. cumdist[k] = cumsum(weights)               (unnormalized prefix sums)
        #   2. sum = cumdist[n-1]
        #   3. per draw: uniform_sample = rand() in [0,1)  (NOT scaled by sum)
        #   4. binary search lower_bound on cumdist for the first index `idx`
        #      with `uniform_sample * sum <= cumdist[idx]` — equivalently aten
        #      compares the running cumulative against `uniform_sample * sum`
        #      via the lower_bound loop:
        #          while (lo < hi):
        #              mid = (lo + hi) / 2
        #              if cumdist[mid] < uniform_sample * sum: lo = mid + 1
        #              else: hi = mid
        #      returning `lo` (first idx with cumdist[idx] >= uniform_sample*sum).
        #   5. aten clamps the result into [0, n-1] (sample_idx).
        # The `>=` (lower_bound) tie semantics and the scaled-uniform comparison
        # both matter for exact parity — see SKEPTIC P0-1.
        var n = len(weights)
        # cumdist (prefix sums, unnormalized)                        (aten)
        var cumdist = List[Float64]()
        var run = Float64(0.0)
        for k in range(n):
            run = run + weights[k]
            cumdist.append(run)
        var sum = cumdist[n - 1]

        var out_vals = List[Int]()
        for b in range(batch_size):
            # uniform_sample in [0,1) (aten draws one uniform per sample)
            var uniform_sample = _uniform_at(ks, UInt64(b))
            # binary search: lower_bound for first idx where
            #   cumdist[idx] >= uniform_sample * sum
            var target = uniform_sample * sum
            var lo = 0
            var hi = n
            while lo < hi:
                var mid = (lo + hi) // 2
                if cumdist[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            var sample_idx = lo
            # aten clamps the sampled index into [0, n-1]
            if sample_idx < 0:
                sample_idx = 0
            elif sample_idx > n - 1:
                sample_idx = n - 1
            # + min_timestep (reference trainer L209); .int() at return (reference trainer L212)
            out_vals.append(sample_idx + min_timestep)

        return TimestepHostVals(out_vals^, weights^)


# ─────────────────────────────────────────────────────────────────────────────
# _get_timestep_discrete  (reference trainer L121-212)
#
# Device-tensor wrapper. Deterministic branch returns an int64 (torch.long) tensor
# of shape (1,) — torch.tensor(int(N*0.5)-1, dtype=torch.long).unsqueeze(0)
# (reference trainer L135-139); NO .int() (the .int() at reference trainer L212 is only on the non-deterministic
# return — SKEPTIC P3-6). Non-deterministic branch delegates to the shared host
# computation and uploads as an INT32 device tensor (timestep.int(), reference trainer L212).
# ─────────────────────────────────────────────────────────────────────────────
def get_timestep_discrete(
    num_train_timesteps: Int,
    deterministic: Bool,
    batch_size: Int,
    timestep_distribution: Int,
    min_noising_strength: Float64,
    max_noising_strength: Float64,
    noising_weight: Float64,
    noising_bias: Float64,
    shift: Float64,
    weights_cache: List[Float64],
    seed: UInt64,
    ctx: DeviceContext,
) raises -> TimestepResult:
    # if deterministic:                                               (reference trainer L133-139)
    if deterministic:
        # int(num_train_timesteps * 0.5) - 1  (-1 = zero-based)       (reference trainer L136)
        var v = Int(Float64(num_train_timesteps) * Float64(0.5)) - 1
        var vals = List[Int]()
        vals.append(v)
        var out_shape = List[Int]()
        out_shape.append(1)  # .unsqueeze(0)                          (reference trainer L139)
        # reference trainer returns dtype=torch.long (int64) on this branch — the .int() at
        # reference trainer L212 is only on the non-deterministic return (SKEPTIC P3-6).
        var t = _int64_tensor_from_host(vals, out_shape.copy(), ctx)
        return TimestepResult(t^, List[Float64]())

    var host = _get_timestep_discrete_host(
        num_train_timesteps,
        batch_size,
        timestep_distribution,
        min_noising_strength,
        max_noising_strength,
        noising_weight,
        noising_bias,
        shift,
        weights_cache,
        seed,
    )
    # return timestep.int()  → INT32 device tensor                    (reference trainer L212)
    var out_shape = List[Int]()
    out_shape.append(batch_size)
    var t = _int32_tensor_from_host(host.values, out_shape.copy(), ctx)
    return TimestepResult(t^, host.weights^)


# ─────────────────────────────────────────────────────────────────────────────
# _get_timestep_continuous  (reference trainer L214-238)
#
# deterministic → torch.full((batch_size,), 0.5)  → FLOAT32 tensor of 0.5
#                 (reference trainer L221-226).
# else          → discrete = _get_timestep_discrete(
#                     num_train_timesteps=10000, deterministic=False, ...) + 1
#                 (reference trainer L228-235); the `+ 1` applies to the INT tensor (reference trainer L235),
#                 then continuous = discrete.float() / 10000  (reference trainer L237) → FLOAT32.
#
# Returns a FLOAT32 device tensor of shape (batch_size,) (torch returns float32 on
# both branches; torch.full default dtype is float32, discrete.float() is float32).
# Reuses the shared host computation (SAME verified continuous/discrete math) so
# the `+ 1` and `/ 10000` are applied to the exact integer timesteps.
# ─────────────────────────────────────────────────────────────────────────────
def get_timestep_continuous(
    deterministic: Bool,
    batch_size: Int,
    timestep_distribution: Int,
    min_noising_strength: Float64,
    max_noising_strength: Float64,
    noising_weight: Float64,
    noising_bias: Float64,
    shift: Float64,
    weights_cache: List[Float64],
    seed: UInt64,
    ctx: DeviceContext,
) raises -> TimestepResult:
    # if deterministic:                                               (reference trainer L221-226)
    if deterministic:
        # torch.full(size=(batch_size,), fill_value=0.5)  → float32   (reference trainer L222-226)
        var vals = List[Float32]()
        for _ in range(batch_size):
            vals.append(Float32(0.5))
        var out_shape = List[Int]()
        out_shape.append(batch_size)
        var t = Tensor.from_host(vals, out_shape.copy(), STDtype.F32, ctx)
        return TimestepResult(t^, List[Float64]())

    # discrete_timesteps = 10000                                      (reference trainer L228)
    var discrete_timesteps = 10000
    # discrete = _get_timestep_discrete(num_train_timesteps=10000,
    #     deterministic=False, ...) + 1                               (reference trainer L229-235)
    var host = _get_timestep_discrete_host(
        discrete_timesteps,
        batch_size,
        timestep_distribution,
        min_noising_strength,
        max_noising_strength,
        noising_weight,
        noising_bias,
        shift,
        weights_cache,
        seed,
    )
    # continuous = discrete.float() / discrete_timesteps              (reference trainer L237)
    # The `+ 1` (reference trainer L235) is applied to the INT timestep before the float divide.
    # torch: discrete is int32, discrete.float() is float32, the Python-int
    # `discrete_timesteps` is promoted to float32, so the divide is FLOAT32/FLOAT32
    # — done here in Float32 (not Float64-then-round) to match torch bit-for-bit.
    var out_vals = List[Float32]()
    for i in range(len(host.values)):
        var discrete_plus1 = host.values[i] + 1
        out_vals.append(Float32(discrete_plus1) / Float32(discrete_timesteps))

    var out_shape = List[Int]()
    out_shape.append(batch_size)
    var t = Tensor.from_host(out_vals, out_shape.copy(), STDtype.F32, ctx)
    return TimestepResult(t^, host.weights^)
