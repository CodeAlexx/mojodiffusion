# 1:1 port of Serenity modules/util/loss/vb_loss.py
# Source of truth: /home/alex/Serenity/modules/util/loss/vb_loss.py
#   (Modified from OpenAI GLIDE/ADM/IDDPM gaussian_diffusion.py.)
#
# The variational-lower-bound term. Serenity computes this on the per-element
# loss tensors (host statistic, F32) — consistent with the bf16-storage policy.
# torch.where / element-wise pow / exp / tanh / log have no single foundation
# kernel chained the way vb needs, so the elementwise math is done host-side on
# the F32 views. The diffusion-schedule coefficient TABLES (one value per
# timestep, see DiffusionScheduleCoefficients.py) are passed as List[Float32]
# and indexed by the per-item timestep, mirroring `__extract_into_tensor`
# (vb_loss.py:179-187: res = tensor[timesteps], then broadcast over trailing
# dims — here we broadcast by indexing the same coeff for every inner element).

from std.math import sqrt, exp, tanh, log
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor


# ─── DiffusionScheduleCoefficients view ───────────────────────────────────────
# Holds the per-timestep coefficient tables vb_loss reads. Each List has length
# num_timesteps. Mirrors the fields used in vb_loss.py (DiffusionScheduleCoefficients.py).
struct VbCoefficients(Copyable, Movable):
    var betas: List[Float32]
    var posterior_log_variance_clipped: List[Float32]
    var posterior_mean_coef1: List[Float32]
    var posterior_mean_coef2: List[Float32]
    var sqrt_recip_alphas_cumprod: List[Float32]
    var sqrt_recipm1_alphas_cumprod: List[Float32]

    fn __init__(
        out self,
        betas: List[Float32],
        posterior_log_variance_clipped: List[Float32],
        posterior_mean_coef1: List[Float32],
        posterior_mean_coef2: List[Float32],
        sqrt_recip_alphas_cumprod: List[Float32],
        sqrt_recipm1_alphas_cumprod: List[Float32],
    ):
        self.betas = betas
        self.posterior_log_variance_clipped = posterior_log_variance_clipped
        self.posterior_mean_coef1 = posterior_mean_coef1
        self.posterior_mean_coef2 = posterior_mean_coef2
        self.sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod


comptime _SQRT_2_OVER_PI = Float32(0.7978845608028654)  # sqrt(2/pi)
comptime _LOG2 = Float32(0.6931471805599453)            # np.log(2.0)


# normal_kl  (vb_loss.py:14-31): elementwise KL between two gaussians.
fn _normal_kl(
    mean1: Float32, logvar1: Float32, mean2: Float32, logvar2: Float32
) -> Float32:
    var d = mean1 - mean2
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + exp(logvar1 - logvar2)
        + (d * d) * exp(-logvar2)
    )


# approx_standard_normal_cdf  (vb_loss.py:34-39)
fn _approx_standard_normal_cdf(x: Float32) -> Float32:
    return 0.5 * (1.0 + tanh(_SQRT_2_OVER_PI * (x + 0.044715 * (x * x * x))))


fn _clamp_min(v: Float32, lo: Float32) -> Float32:
    return v if v > lo else lo


# discretized_gaussian_log_likelihood  (vb_loss.py:42-70)
fn _discretized_gaussian_log_likelihood(
    x: Float32, means: Float32, log_scales: Float32
) -> Float32:
    var centered_x = x - means
    var inv_stdv = exp(-log_scales)
    var plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    var cdf_plus = _approx_standard_normal_cdf(plus_in)
    var min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    var cdf_min = _approx_standard_normal_cdf(min_in)
    var log_cdf_plus = log(_clamp_min(cdf_plus, 1e-12))
    var log_one_minus_cdf_min = log(_clamp_min(1.0 - cdf_min, 1e-12))
    var cdf_delta = cdf_plus - cdf_min
    # torch.where(x < -0.999, log_cdf_plus,
    #     torch.where(x > 0.999, log_one_minus_cdf_min, log(cdf_delta.clamp(1e-12))))
    if x < -0.999:
        return log_cdf_plus
    elif x > 0.999:
        return log_one_minus_cdf_min
    else:
        return log(_clamp_min(cdf_delta, 1e-12))


# __predict_x_0_from_eps  (vb_loss.py:118-128)
fn _predict_x_0_from_eps(
    coefficients: VbCoefficients, x_t: Float32, t: Int, eps: Float32
) -> Float32:
    return (
        coefficients.sqrt_recip_alphas_cumprod[t] * x_t
        - coefficients.sqrt_recipm1_alphas_cumprod[t] * eps
    )


# __q_posterior_mean_variance  (vb_loss.py:73-96): returns (mean, log_var).
fn _q_posterior_mean(
    coefficients: VbCoefficients, x_0: Float32, x_t: Float32, t: Int
) -> Float32:
    return (
        coefficients.posterior_mean_coef1[t] * x_0
        + coefficients.posterior_mean_coef2[t] * x_t
    )


# __p_mean_variance  (vb_loss.py:99-115): returns (predicted_mean, predicted_log_variance).
fn _p_mean_variance(
    coefficients: VbCoefficients,
    x_t: Float32,
    t: Int,
    frozen_predicted_eps: Float32,
    predicted_var_values: Float32,
) -> (Float32, Float32):
    var min_log = coefficients.posterior_log_variance_clipped[t]
    var max_log = log(coefficients.betas[t])  # __extract_into_tensor(log(betas), t)
    # frac = (predicted_var_values + 1) / 2     (vb_loss.py:109)
    var frac = (predicted_var_values + 1.0) / 2.0
    var predicted_log_variance = frac * max_log + (1.0 - frac) * min_log

    var predicted_x_0 = _predict_x_0_from_eps(
        coefficients, x_t, t, frozen_predicted_eps
    )
    var predicted_mean = _q_posterior_mean(coefficients, predicted_x_0, x_t, t)
    return (predicted_mean, predicted_log_variance)


# __vb_terms_bpd  (vb_loss.py:131-176): per-element variational-bound term (bits).
fn _vb_term(
    coefficients: VbCoefficients,
    x_0: Float32,
    x_t: Float32,
    t: Int,
    frozen_predicted_eps: Float32,
    predicted_var_values: Float32,
) -> Float32:
    # true posterior mean + clipped log-var (vb_loss.py:147-152)
    var true_mean = _q_posterior_mean(coefficients, x_0, x_t, t)
    var true_log_variance_clipped = coefficients.posterior_log_variance_clipped[t]

    var p = _p_mean_variance(
        coefficients, x_t, t, frozen_predicted_eps, predicted_var_values
    )
    var predicted_mean = p[0]
    var predicted_log_variance = p[1]

    # kl = normal_kl(...) / np.log(2.0)        (vb_loss.py:160-163)
    var kl = _normal_kl(
        true_mean, true_log_variance_clipped, predicted_mean, predicted_log_variance
    )
    kl = kl / _LOG2

    # decoder_nll = -discretized_gaussian_log_likelihood(x_0, predicted_mean,
    #     0.5*predicted_log_variance) / np.log(2.0)   (vb_loss.py:165-169)
    var decoder_nll = -_discretized_gaussian_log_likelihood(
        x_0, predicted_mean, 0.5 * predicted_log_variance
    )
    decoder_nll = decoder_nll / _LOG2

    # output = torch.where((t == 0), decoder_nll, kl)   (vb_loss.py:175)
    if t == 0:
        return decoder_nll
    else:
        return kl


# vb_losses  (vb_loss.py:190-212). predicted_eps is detached (frozen) — eps does
# not contribute grads here; only the variance prediction is learned via the VB.
# Coefficient tables come from DiffusionScheduleCoefficients (passed as a view).
# `t` is the per-batch-item timestep (length = batch).
def vb_losses(
    coefficients: VbCoefficients,
    x_0: Tensor,
    x_t: Tensor,
    t: List[Int],
    predicted_eps: Tensor,
    predicted_var_values: Tensor,
    ctx: DeviceContext,
) -> Tensor:
    var shape = x_0.shape()
    var n = x_0.numel()
    var batch = shape[0]
    var inner = n // batch

    # NOTE: x_0, x_t, predicted_eps, predicted_var_values must all share
    # x_0.shape() — they are indexed by the same flat `i`. The caller splits the
    # 2*C model output into eps/var BEFORE this call so `var_h` matches the eps
    # shape (vb_loss.py: predicted_var_values arrives already split). Only the
    # per-timestep coefficient tables broadcast (via `t[b]` in _vb_term).
    var x0_h = x_0.to_host(ctx)
    var xt_h = x_t.to_host(ctx)
    var eps_h = predicted_eps.to_host(ctx)  # .detach() — frozen (vb_loss.py:210)
    var var_h = predicted_var_values.to_host(ctx)

    var out = List[Float32]()
    for i in range(n):
        var b = i // inner
        var ti = t[b]
        out.append(
            _vb_term(coefficients, x0_h[i], xt_h[i], ti, eps_h[i], var_h[i])
        )

    return Tensor.from_host(out, shape, x_0.dtype(), ctx)
