# sampling/dpmpp_2m.mojo — DPM++ 2M exponential-multistep scheduler step.
#
# Pure-Mojo port of the 2nd-order multistep DPM++ (data-prediction,
# flow-matching variant) from the EDv2 / inference-flame clean-room solver:
#   inference-flame/src/sampling/exponential_multistep.rs  ::  dpmpp_2m_step
#
# Reference: Lu et al. 2022, "DPM-Solver++" (arXiv:2211.01095), §4.
#
# Math (rectified flow, α = 1-σ, log-SNR λ(σ) = log((1-σ)/σ)):
#   x_next  = (σ_next/σ)·x - α_next·(e^{-h} - 1)·denoised_correction
#           = (σ_next/σ)·x + α_next·(-h).expm1()·(-denoised_correction)
# with
#   denoised_correction = (1 + 1/(2r))·denoised - (1/(2r))·denoised_prev,
#   h = λ_next - λ,  r = (λ - λ_prev) / h.
#
# One NFE per step. Uses one previous `denoised` (and its λ) for the 2nd-order
# correction; if the history is empty (first step) or h_prev is non-positive
# the step degrades to a 1st-order data-prediction step (DDIM / Euler-on-data):
#   x_next = σ_ratio·x + α_next·(1 - e^{-h})·denoised.
#
# This module works in the DATA-PREDICTION (denoised, x0) form. The model
# produces velocity v; `denoised = x - σ·v` (rectified-flow convention
# v = noise - data, x_σ = (1-σ)·data + σ·noise). The caller converts v→denoised
# before calling `step` (helper `denoised_from_velocity` provided).
#
# The scalar coefficient logic is IDENTICAL whether applied to a single f64
# scalar (the convergence-order parity gate) or elementwise to a device tensor
# (inference). The tensor path uses serenitymojo.ops.tensor_algebra (mul_scalar
# + add/sub); only the per-step latent update touches the GPU. The history of
# past `denoised` tensors is boxed in TArc (ArcPointer[Tensor]) because Tensor
# is move-only and cannot live in a plain List (MOJO_CONVENTIONS §2a).
#
# Mojo 0.26.x. Inference-only. No autograd, no Python at runtime.

from collections import List, Optional
from std.math import exp, log
from std.gpu.host import DeviceContext
from memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar
from serenitymojo.io.dtype import STDtype

# Tensor is move-only; box it for the history List (MOJO_CONVENTIONS §2a).
comptime TArc = ArcPointer[Tensor]


# ─────────────────────────────────────────────────────────────────────────────
# Scalar helpers (f64 interior — mirrors the Rust reference scalar path).
# ─────────────────────────────────────────────────────────────────────────────

comptime _LAMBDA_CLAMP: Float64 = 30.0


def _expm1_f64(z: Float64) -> Float64:
    """e^z - 1, accurate near z=0 via Taylor (std.math has no expm1).

    Matches Rust's `f64::exp_m1` to ~1e-15: for |z| < 1e-5 the closed form
    `exp(z)-1` loses all leading digits to cancellation, so use the series
    z + z²/2 + z³/6 + z⁴/24 + z⁵/120 (6 terms is ample for |z| < 1e-5).
    """
    var az = z if z >= 0.0 else -z
    if az < 1.0e-5:
        var z2 = z * z
        return (
            z
            + z2 * 0.5
            + z2 * z * (1.0 / 6.0)
            + z2 * z2 * (1.0 / 24.0)
            + z2 * z2 * z * (1.0 / 120.0)
        )
    return exp(z) - 1.0


def lambda_from_sigma_f64(sigma: Float64) -> Float64:
    """Log-SNR λ(σ) = log((1-σ)/σ), clamped to ±30 (matches the scalar ref).

    Increases monotonically as σ decreases, so h = λ_next - λ > 0 for the
    denoising direction (σ: 1 → 0).
    """
    var lam = log((1.0 - sigma) / sigma)
    if lam > _LAMBDA_CLAMP:
        return _LAMBDA_CLAMP
    if lam < -_LAMBDA_CLAMP:
        return -_LAMBDA_CLAMP
    return lam


struct _Dpmpp2mCoeffs(Copyable, Movable):
    """The three scalar coefficients of one DPM++ 2M update:

        x_next = c_x·x + c_d·denoised + c_p·denoised_prev

    For the 1st-order fallback `c_p == 0.0` and `denoised_prev` is unused.
    Computed entirely from the scalar (σ, σ_next, λ_prev) schedule data, so the
    same struct drives both the scalar gate and the tensor `step`.
    """

    var c_x: Float64
    var c_d: Float64
    var c_p: Float64
    var used_history: Bool

    def __init__(out self, c_x: Float64, c_d: Float64, c_p: Float64, used_history: Bool):
        self.c_x = c_x
        self.c_d = c_d
        self.c_p = c_p
        self.used_history = used_history


def dpmpp_2m_coeffs(
    sigma: Float64,
    sigma_next: Float64,
    have_history: Bool,
    lambda_prev: Float64,
) -> _Dpmpp2mCoeffs:
    """Scalar DPM++ 2M coefficients for one step (verbatim from
    `dpmpp_2m_step` in exponential_multistep.rs).

    `have_history` is False on the first step (empty history) → 1st-order.
    `lambda_prev` is the log-SNR stored alongside the previous denoised; it is
    ignored when `have_history` is False.
    """
    var lam = lambda_from_sigma_f64(sigma)
    var lam_next = lambda_from_sigma_f64(sigma_next)
    var h = lam_next - lam
    var alpha_next = 1.0 - sigma_next
    var sigma_ratio = sigma_next / sigma
    var em1 = _expm1_f64(-h)  # e^{-h} - 1 ≤ 0 (h > 0)

    # First step / empty history → 1st-order data-prediction step.
    #   x_next = σ_ratio·x + α_next·(1 - e^{-h})·denoised = σ_ratio·x - α_next·em1·denoised
    if not have_history:
        return _Dpmpp2mCoeffs(sigma_ratio, -alpha_next * em1, 0.0, False)

    var h_prev = lam - lambda_prev
    # r must be positive; numerical noise → drop to 1st-order.
    if not (h_prev > 0.0 and h > 0.0):
        return _Dpmpp2mCoeffs(sigma_ratio, -alpha_next * em1, 0.0, False)

    var r = h_prev / h
    var inv_2r = 0.5 / r
    var c_d = -alpha_next * em1 * (1.0 + inv_2r)
    var c_p = alpha_next * em1 * inv_2r  # = -(-em1)·inv_2r·α_next
    return _Dpmpp2mCoeffs(sigma_ratio, c_d, c_p, True)


# ─────────────────────────────────────────────────────────────────────────────
# Tensor helpers.
# ─────────────────────────────────────────────────────────────────────────────


def denoised_from_velocity(
    x: Tensor, v: Tensor, sigma: Float32, ctx: DeviceContext
) raises -> Tensor:
    """denoised (x0) = x - σ·v  (rectified-flow convention v = noise - data)."""
    var sv = mul_scalar(v, sigma, ctx)
    return sub(x, sv, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Multistep history ring buffer of (denoised, λ).
# ─────────────────────────────────────────────────────────────────────────────


struct MultistepHistory(Movable):
    """Bounded most-recent-first history of past (denoised, λ) pairs.

    `push` evicts the oldest entry when full. `get(back)` returns the entry
    `back` steps back (0 = newest). Tensors are boxed in TArc since Tensor is
    move-only (MOJO_CONVENTIONS §2a). DPM++ 2M only needs capacity 1, but the
    buffer is general so a future 3rd-order scheme can reuse it.
    """

    var _denoised: List[TArc]
    var _lambdas: List[Float64]
    var _capacity: Int

    def __init__(out self, capacity: Int):
        self._denoised = List[TArc]()
        self._lambdas = List[Float64]()
        self._capacity = capacity if capacity > 0 else 1

    def len(self) -> Int:
        return len(self._denoised)

    def is_empty(self) -> Bool:
        return len(self._denoised) == 0

    def push(mut self, var denoised: Tensor, lam_val: Float64):
        """Insert `(denoised, lam_val)` as the newest entry, evicting the oldest
        if the buffer is full. Stored newest-LAST so `get(0)` reads the tail.

        `lam_val` is the log-SNR λ of `denoised` (param renamed from `lambda`,
        which is a Mojo reserved keyword)."""
        self._denoised.append(TArc(denoised^))
        self._lambdas.append(lam_val)
        if len(self._denoised) > self._capacity:
            # Drop the oldest (front). List has no pop_front; rebuild the tail.
            var nd = List[TArc]()
            var nl = List[Float64]()
            for i in range(1, len(self._denoised)):
                nd.append(self._denoised[i])
                nl.append(self._lambdas[i])
            self._denoised = nd^
            self._lambdas = nl^

    def lambda_back(self, back: Int) raises -> Float64:
        """λ of the entry `back` steps back (0 = newest)."""
        var n = len(self._denoised)
        if back >= n:
            raise Error("MultistepHistory.lambda_back: index out of range")
        return self._lambdas[n - 1 - back]

    def denoised_back(self, back: Int, ctx: DeviceContext) raises -> Tensor:
        """A clone of the denoised tensor `back` steps back (0 = newest)."""
        var n = len(self._denoised)
        if back >= n:
            raise Error("MultistepHistory.denoised_back: index out of range")
        return self._denoised[n - 1 - back][].clone(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# The DPM++ 2M step (tensor form).
# ─────────────────────────────────────────────────────────────────────────────


def dpmpp_2m_step(
    x: Tensor,
    denoised: Tensor,
    sigma: Float32,
    sigma_next: Float32,
    history: MultistepHistory,
    ctx: DeviceContext,
) raises -> Tensor:
    """One DPM++ 2M update on a device tensor.

        x_next = c_x·x + c_d·denoised  (+ c_p·denoised_prev if history present)

    `denoised` is the x0-prediction at `sigma` (use `denoised_from_velocity`).
    Coefficients come from the scalar `dpmpp_2m_coeffs` (f64) so the tensor path
    matches the parity gate byte-for-byte at the scalar layer. The caller is
    responsible for `history.push(denoised, λ)` AFTER the step.
    """
    var have_hist = not history.is_empty()
    var lam_prev: Float64 = 0.0
    if have_hist:
        lam_prev = history.lambda_back(0)
    var c = dpmpp_2m_coeffs(
        Float64(sigma), Float64(sigma_next), have_hist, lam_prev
    )

    # x_next = c_x·x + c_d·denoised
    var term_x = mul_scalar(x, Float32(c.c_x), ctx)
    var term_d = mul_scalar(denoised, Float32(c.c_d), ctx)
    var acc = add(term_x, term_d, ctx)
    if c.used_history:
        var d_prev = history.denoised_back(0, ctx)
        var term_p = mul_scalar(d_prev, Float32(c.c_p), ctx)
        acc = add(acc, term_p, ctx)
    return acc^
