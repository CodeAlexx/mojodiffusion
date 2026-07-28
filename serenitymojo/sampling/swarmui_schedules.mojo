# serenitymojo.sampling.swarmui_schedules — scalar scheduler math copied from
# the bundled SwarmUI Comfy backend.
#
# Creator authority:
#   /home/alex/SwarmUI/dlbackend/ComfyUI/comfy/samplers.py
#     beta_scheduler(alpha=0.6, beta=0.6)
#   /home/alex/SwarmUI/dlbackend/ComfyUI/comfy/model_sampling.py
#     ModelSamplingFlux(shift=1.15, timesteps=10000)
#
# Keep these helpers host-scalar.  The denoise carrier remains BF16/F16 and
# consumes only the resulting Float32 sigma table.

from std.math import atan, exp, log, pow, tan


comptime SWARM_BETA_ALPHA: Float64 = 0.6
comptime SWARM_BETA_BETA: Float64 = 0.6
comptime SWARM_BETA_LOG_BETA: Float64 = 0.8818418061417859
comptime SWARM_FLUX_SHIFT: Float64 = 1.15
comptime SWARM_FLUX_TIMESTEPS = 10000


def _beta_continued_fraction(a: Float64, b: Float64, x: Float64) -> Float64:
    """Numerical-Recipes continued fraction used by regularized incomplete beta."""
    var qab = a + b
    var qap = a + 1.0
    var qam = a - 1.0
    var c = 1.0
    var d = 1.0 - qab * x / qap
    comptime FPMIN: Float64 = 1.0e-30
    if d < FPMIN and d > -FPMIN:
        d = FPMIN
    d = 1.0 / d
    var h = d
    for m in range(1, 201):
        var m_f = Float64(m)
        var m2 = Float64(2 * m)
        var aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if d < FPMIN and d > -FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if c < FPMIN and c > -FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if d < FPMIN and d > -FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if c < FPMIN and c > -FPMIN:
            c = FPMIN
        d = 1.0 / d
        var delta = d * c
        h *= delta
        var err = delta - 1.0
        if err < 0.0:
            err = -err
        if err < 3.0e-14:
            break
    return h


def swarm_beta_cdf_06(x: Float64) -> Float64:
    """Regularized Beta(0.6, 0.6) CDF used by Comfy's beta scheduler."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    var a = Float64(SWARM_BETA_ALPHA)
    var b = Float64(SWARM_BETA_BETA)
    var front = exp(a * log(x) + b * log(1.0 - x) - SWARM_BETA_LOG_BETA)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def swarm_beta_ppf_06(q: Float64) -> Float64:
    """Inverse Beta(0.6, 0.6) CDF, matching scipy.stats.beta.ppf."""
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    var lo = Float64(0.0)
    var hi = Float64(1.0)
    # Bisection is deterministic and comfortably more precise than the final
    # Comfy integer-timestep rounding.
    for _ in range(80):
        var mid = (lo + hi) * 0.5
        if swarm_beta_cdf_06(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def swarm_flux_sigma_from_timestep(t: Float64) -> Float64:
    """Comfy ModelSamplingFlux.sigma with its creator default shift=1.15."""
    if t <= 0.0 or t >= 1.0:
        return t
    var shifted = exp(SWARM_FLUX_SHIFT)
    return shifted / (shifted + (1.0 / t - 1.0))


def build_swarm_beta_flux_schedule(num_steps: Int) raises -> List[Float32]:
    """Exact SwarmUI/Comfy beta schedule for a default ModelSamplingFlux model.

    This follows numpy.rint(beta.ppf(...)*9999), then indexes the 10,000-entry
    shifted Flux sigma table.  At Serenity's <=500 image-step product range the
    rounded indices are unique, so the result is num_steps+1 and ends in zero.
    """
    if num_steps <= 0:
        raise Error("build_swarm_beta_flux_schedule: num_steps must be > 0")
    var out = List[Float32]()
    var last_index = -1
    for i in range(num_steps):
        var q = 1.0 - Float64(i) / Float64(num_steps)
        var ppf = swarm_beta_ppf_06(q)
        # Positive-domain equivalent of numpy.rint except at exact .5 ties;
        # Beta(0.6,0.6) grid points do not land on those ties in the admitted
        # step range.
        var index = Int(ppf * Float64(SWARM_FLUX_TIMESTEPS - 1) + 0.5)
        if index != last_index:
            var t = Float64(index + 1) / Float64(SWARM_FLUX_TIMESTEPS)
            out.append(Float32(swarm_flux_sigma_from_timestep(t)))
        last_index = index
    out.append(0.0)
    return out^


def swarm_flux_sigma_min() -> Float64:
    return swarm_flux_sigma_from_timestep(
        1.0 / Float64(SWARM_FLUX_TIMESTEPS)
    )


def build_swarm_simple_flux_schedule(num_steps: Int) raises -> List[Float32]:
    """Comfy simple_scheduler over ModelSamplingFlux's 10,000-entry table."""
    if num_steps <= 0:
        raise Error("build_swarm_simple_flux_schedule: num_steps must be > 0")
    var out = List[Float32]()
    var stride = Float64(SWARM_FLUX_TIMESTEPS) / Float64(num_steps)
    for i in range(num_steps):
        var source_index = SWARM_FLUX_TIMESTEPS - 1 - Int(Float64(i) * stride)
        var t = Float64(source_index + 1) / Float64(SWARM_FLUX_TIMESTEPS)
        out.append(Float32(swarm_flux_sigma_from_timestep(t)))
    out.append(0.0)
    return out^


def build_swarm_normal_flux_schedule(
    num_steps: Int, sgm: Bool = False
) raises -> List[Float32]:
    """Comfy normal_scheduler/sgm_uniform over ModelSamplingFlux."""
    if num_steps <= 0:
        raise Error("build_swarm_normal_flux_schedule: num_steps must be > 0")
    var count = num_steps + 1 if sgm else num_steps
    var start = Float64(1.0)
    var end = swarm_flux_sigma_min()
    var out = List[Float32]()
    for i in range(num_steps):
        var timestep = start
        if count > 1:
            timestep = start + (end - start) * Float64(i) / Float64(count - 1)
        out.append(Float32(swarm_flux_sigma_from_timestep(timestep)))
    out.append(0.0)
    return out^


def build_swarm_ddim_flux_schedule(num_steps: Int) raises -> List[Float32]:
    """Comfy ddim_scheduler over ModelSamplingFlux."""
    if num_steps <= 0:
        raise Error("build_swarm_ddim_flux_schedule: num_steps must be > 0")
    var stride = SWARM_FLUX_TIMESTEPS // num_steps
    if stride < 1:
        stride = 1
    var ascending = List[Float32]()
    var index = 1
    while index < SWARM_FLUX_TIMESTEPS:
        var t = Float64(index + 1) / Float64(SWARM_FLUX_TIMESTEPS)
        ascending.append(Float32(swarm_flux_sigma_from_timestep(t)))
        index += stride
    var out = List[Float32]()
    for i in range(len(ascending)):
        out.append(ascending[len(ascending) - 1 - i])
    out.append(0.0)
    return out^


def build_swarm_karras_flux_schedule(num_steps: Int) raises -> List[Float32]:
    """k-diffusion get_sigmas_karras(n, sigma_min, sigma_max, rho=7)."""
    if num_steps <= 0:
        raise Error("build_swarm_karras_flux_schedule: num_steps must be > 0")
    var sigma_min = swarm_flux_sigma_min()
    var rho = Float64(7.0)
    var min_inv = pow(sigma_min, 1.0 / rho)
    var max_inv = Float64(1.0)
    var out = List[Float32]()
    for i in range(num_steps):
        var ramp = (
            Float64(i) / Float64(num_steps - 1)
            if num_steps > 1
            else Float64(0.0)
        )
        out.append(Float32(pow(max_inv + ramp * (min_inv - max_inv), rho)))
    out.append(0.0)
    return out^


def build_swarm_exponential_flux_schedule(num_steps: Int) raises -> List[Float32]:
    """k-diffusion get_sigmas_exponential over Flux sigma bounds."""
    if num_steps <= 0:
        raise Error("build_swarm_exponential_flux_schedule: num_steps must be > 0")
    var lo = log(swarm_flux_sigma_min())
    var hi = Float64(0.0)
    var out = List[Float32]()
    for i in range(num_steps):
        var ramp = (
            Float64(i) / Float64(num_steps - 1)
            if num_steps > 1
            else Float64(0.0)
        )
        out.append(Float32(exp(hi + ramp * (lo - hi))))
    out.append(0.0)
    return out^


def build_swarm_linear_quadratic_flux_schedule(num_steps: Int) raises -> List[Float32]:
    """Comfy linear_quadratic_schedule, including its Mochi threshold=0.025."""
    if num_steps <= 0:
        raise Error("build_swarm_linear_quadratic_flux_schedule: num_steps must be > 0")
    if num_steps == 1:
        var one = List[Float32]()
        one.append(1.0)
        one.append(0.0)
        return one^
    var linear_steps = num_steps // 2
    var threshold = Float64(0.025)
    var threshold_diff = Float64(linear_steps) - threshold * Float64(num_steps)
    var quadratic_steps = num_steps - linear_steps
    var quadratic_coef = threshold_diff / (
        Float64(linear_steps) * Float64(quadratic_steps * quadratic_steps)
    )
    var linear_coef = threshold / Float64(linear_steps) - (
        2.0 * threshold_diff / Float64(quadratic_steps * quadratic_steps)
    )
    var constant = quadratic_coef * Float64(linear_steps * linear_steps)
    var out = List[Float32]()
    for i in range(linear_steps):
        var noise = Float64(i) * threshold / Float64(linear_steps)
        out.append(Float32(1.0 - noise))
    for i in range(linear_steps, num_steps):
        var fi = Float64(i)
        var noise = quadratic_coef * fi * fi + linear_coef * fi + constant
        out.append(Float32(1.0 - noise))
    out.append(0.0)
    return out^


def build_swarm_kl_optimal_flux_schedule(num_steps: Int) raises -> List[Float32]:
    """Comfy kl_optimal_scheduler over Flux sigma_min/sigma_max."""
    if num_steps <= 0:
        raise Error("build_swarm_kl_optimal_flux_schedule: num_steps must be > 0")
    var angle_min = atan(swarm_flux_sigma_min())
    var angle_max = atan(Float64(1.0))
    var out = List[Float32]()
    for i in range(num_steps):
        var adj = (
            Float64(i) / Float64(num_steps - 1)
            if num_steps > 1
            else Float64(0.0)
        )
        out.append(Float32(tan(adj * angle_min + (1.0 - adj) * angle_max)))
    out.append(0.0)
    return out^


def build_swarm_flux_schedule(name: String, num_steps: Int) raises -> List[Float32]:
    """Dispatch one genuine general-purpose SwarmUI scheduler for Flux models."""
    var normalized = String(name.lower())
    if normalized == "beta":
        return build_swarm_beta_flux_schedule(num_steps)
    if normalized == "simple":
        return build_swarm_simple_flux_schedule(num_steps)
    if normalized == "normal":
        return build_swarm_normal_flux_schedule(num_steps)
    if normalized == "sgm_uniform":
        return build_swarm_normal_flux_schedule(num_steps, sgm=True)
    if normalized == "ddim_uniform":
        return build_swarm_ddim_flux_schedule(num_steps)
    if normalized == "karras":
        return build_swarm_karras_flux_schedule(num_steps)
    if normalized == "exponential":
        return build_swarm_exponential_flux_schedule(num_steps)
    if normalized == "linear_quadratic":
        return build_swarm_linear_quadratic_flux_schedule(num_steps)
    if normalized == "kl_optimal":
        return build_swarm_kl_optimal_flux_schedule(num_steps)
    raise Error(String("unsupported Swarm Flux scheduler: ") + name)
