# sampling/unipc.mojo — UniPC multistep predictor-corrector scheduler (bh2).
#
# Pure-Mojo port of the bh2, solver_order=2, predict_x0=true UniPC multistep
# scheduler from the EDv2 / inference-flame reference:
#   inference-flame/src/sampling/cosmos_unipc.rs
#     (itself a verbatim port of diffusers' FlowUniPCMultistepScheduler,
#      cosmos_predict2/_src/predict2/models/fm_solvers_unipc.py)
#
# Configuration (the ONLY one implemented — matches the Cosmos reference):
#   * solver_order      = 2
#   * solver_type       = "bh2"     (B_h = expm1(hh))
#   * prediction_type   = flow / predict_x0 = True
#   * lower_order_final  = True
#   * final_sigmas_type = "zero"
#   * disable_corrector = []  (corrector always on after step 0)
#
# AGENT-DEFAULT NOTE (flagged per task): I ported the **bh2** solver_type
# (B_h = expm1(hh)) because that is exactly the variant the Cosmos reference
# wires (cosmos_unipc.rs:8 `solver_type = "bh2"`). The diffusers default is
# "bh2" as well. bh1 (B_h = hh) is NOT ported. The parametrization is the
# **(alpha, sigma)** flow-matching form with log-SNR lambda = log(alpha/sigma),
# alpha = 1 - sigma — again matching the Cosmos reference, NOT a sigma-only form.
#
# At solver_order=2 the predictor short-circuits to rhos_p=[0.5] and the
# order-1 corrector short-circuits to rhos_c=[0.5] (fm_solvers_unipc.py:441-444,
# :579-582). The order-2 corrector solves the full 2x2 R·rhos = b via a CPU f64
# Gauss-Jordan (linsolve). The toy-ODE scalar integration and the rhos
# coefficient checks are exercised in sampling/parity/unipc_parity.mojo.
#
# Tensor math goes through serenitymojo.ops.tensor_algebra (mul_scalar/add/sub).
# The small linear solve and all coefficient prep stay on host in f64. The ring
# buffer of converted model outputs is boxed in TArc (Tensor is move-only,
# MOJO_CONVENTIONS §2a).
#
# Mojo 0.26.x. Inference-only. No autograd, no Python at runtime.

from collections import List, Optional
from std.math import exp, log
from std.gpu.host import DeviceContext
from memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar
from serenitymojo.io.dtype import STDtype

comptime TArc = ArcPointer[Tensor]


# ─────────────────────────────────────────────────────────────────────────────
# Scalar helpers (host f64 — mirror the Rust reference exactly).
# ─────────────────────────────────────────────────────────────────────────────


def _expm1_f64(z: Float64) -> Float64:
    """e^z - 1 with a near-zero Taylor guard (std.math has no expm1)."""
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


def alpha_from_sigma(sigma: Float64) -> Float64:
    """Flow-matching `_sigma_to_alpha_sigma_t`: alpha = 1 - sigma."""
    return 1.0 - sigma


def build_unipc_sigma_schedule(
    num_inference_steps: Int, shift: Float64, num_train_timesteps: Int
) raises -> List[Float64]:
    """Sigma schedule mirroring cosmos_unipc.rs `new` (= diffusers set_timesteps).

        sigma_max = (N_train - 1)/N_train,  sigma_min = 0
        sigmas = linspace(sigma_max, sigma_min, n_inf+1)[:-1]   # n_inf values
        sigmas = shift*s / (1 + (shift-1)*s)                    # flow shift
        sigmas = concat([sigmas, 0.0])                          # final zero

    Returns `num_inference_steps + 1` values (descending, last == 0).
    """
    if num_inference_steps <= 0:
        raise Error("build_unipc_sigma_schedule: num_inference_steps must be > 0")
    var n_train = Float64(num_train_timesteps)
    var sigma_max = (n_train - 1.0) / n_train
    var sigma_min = 0.0
    var out = List[Float64]()
    for i in range(num_inference_steps):
        var t = sigma_max + (sigma_min - sigma_max) * (
            Float64(i) / Float64(num_inference_steps)
        )
        var shifted = shift * t / (1.0 + (shift - 1.0) * t)
        out.append(shifted)
    out.append(0.0)
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# bh2 coefficient prep (rks, b_vec, R, rhos) — verbatim from cosmos_unipc.rs.
# ─────────────────────────────────────────────────────────────────────────────


def _linsolve_f64(r: List[List[Float64]], b: List[Float64]) raises -> List[Float64]:
    """Solve R·x = b for square R (f64 Gauss-Jordan, partial pivoting).

    Mirrors cosmos_unipc.rs `linsolve_f64`. Used by the order-2 corrector.
    """
    var k = len(b)
    if len(r) != k:
        raise Error("_linsolve_f64: shape mismatch")
    # Augmented matrix [R | b].
    var aug = List[List[Float64]]()
    for i in range(k):
        if len(r[i]) != k:
            raise Error("_linsolve_f64: non-square row")
        var row = List[Float64]()
        for j in range(k):
            row.append(r[i][j])
        row.append(b[i])
        aug.append(row^)
    for col in range(k):
        var piv = col
        var piv_val = aug[col][col] if aug[col][col] >= 0.0 else -aug[col][col]
        for row in range(col + 1, k):
            var v = aug[row][col] if aug[row][col] >= 0.0 else -aug[row][col]
            if v > piv_val:
                piv_val = v
                piv = row
        if piv_val < 1.0e-30:
            raise Error("_linsolve_f64: singular matrix")
        if piv != col:
            for j in range(k + 1):
                var t = aug[piv][j]
                aug[piv][j] = aug[col][j]
                aug[col][j] = t
        var pivot = aug[col][col]
        for j in range(col, k + 1):
            aug[col][j] = aug[col][j] / pivot
        for row in range(k):
            if row == col:
                continue
            var factor = aug[row][col]
            var af = factor if factor >= 0.0 else -factor
            if af < 1.0e-30:
                continue
            for j in range(col, k + 1):
                aug[row][j] = aug[row][j] - factor * aug[col][j]
    var x = List[Float64]()
    for i in range(k):
        x.append(aug[i][k])
    return x^


struct UniPcBh2Coeffs(Copyable, Movable):
    """Scalar bh2 coefficients for one predictor/corrector update.

    Holds the scalar scaling factors and the `rks` / `rhos` vectors so both the
    tensor `step` and the parity gate consume identical numbers. `rhos` is the
    full solved vector for order==2 (the predictor ignores it and uses [0.5];
    the corrector uses it directly).
    """

    var b_h: Float64
    var alpha_t: Float64
    var sigma_t: Float64
    var sigma_s0: Float64
    var alpha_s0: Float64
    var h_phi_1: Float64
    var rks: List[Float64]
    var rhos: List[Float64]

    def __init__(
        out self,
        b_h: Float64,
        alpha_t: Float64,
        sigma_t: Float64,
        sigma_s0: Float64,
        alpha_s0: Float64,
        h_phi_1: Float64,
        var rks: List[Float64],
        var rhos: List[Float64],
    ):
        self.b_h = b_h
        self.alpha_t = alpha_t
        self.sigma_t = sigma_t
        self.sigma_s0 = sigma_s0
        self.alpha_s0 = alpha_s0
        self.h_phi_1 = h_phi_1
        self.rks = rks^
        self.rhos = rhos^


def _log_safe(v: Float64) -> Float64:
    # log of a positive value; -inf-ish sentinel for v<=0 (boundary sigma=0).
    if v > 0.0:
        return log(v)
    return -1.0e30


def compute_bh2_coefficients(
    sigmas: List[Float64],
    step_index: Int,
    order: Int,
    is_corrector: Bool,
) raises -> UniPcBh2Coeffs:
    """Port of cosmos_unipc.rs `compute_bh2_coefficients` (predict_x0=true).

    Predictor: sigma_t = sigmas[step_index+1], sigma_s0 = sigmas[step_index],
               si = step_index - i.
    Corrector: sigma_t = sigmas[step_index],   sigma_s0 = sigmas[step_index-1],
               si = step_index - (i+1).
    Returns rks (length `order`, last == 1.0) and the solved rhos (order==2:
    full 2x2 solve; order==1: placeholder, caller short-circuits to [0.5]).
    """
    var idx_t: Int
    var idx_s0: Int
    if is_corrector:
        idx_t = step_index
        idx_s0 = step_index - 1
    else:
        idx_t = step_index + 1
        idx_s0 = step_index
    var sigma_t = sigmas[idx_t]
    var sigma_s0 = sigmas[idx_s0]
    var alpha_t = alpha_from_sigma(sigma_t)
    var alpha_s0 = alpha_from_sigma(sigma_s0)

    var lambda_t = _log_safe(alpha_t) - _log_safe(sigma_t)
    var lambda_s0 = _log_safe(alpha_s0) - _log_safe(sigma_s0)
    var h = lambda_t - lambda_s0

    # rks: for i in 1..order, rk = (lambda_si - lambda_s0)/h; then append 1.0.
    var rks = List[Float64]()
    for i in range(1, order):
        var si: Int
        if is_corrector:
            si = step_index - (i + 1)
        else:
            si = step_index - i
        if si < 0:
            si = 0
        var sigma_si = sigmas[si]
        var alpha_si = alpha_from_sigma(sigma_si)
        var lambda_si = _log_safe(alpha_si) - _log_safe(sigma_si)
        rks.append((lambda_si - lambda_s0) / h)
    rks.append(1.0)

    # b vector (predict_x0 → hh = -h). bh2: B_h = expm1(hh).
    var hh = -h
    var h_phi_1 = _expm1_f64(hh)
    var b_h = _expm1_f64(hh)

    var h_phi_k = h_phi_1 / hh - 1.0
    var factorial_i = 1.0
    var b_vec = List[Float64]()
    for i in range(1, order + 1):
        b_vec.append(h_phi_k * factorial_i / b_h)
        factorial_i = factorial_i * Float64(i + 1)
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i

    # R rows = [rks**0, rks**1, ..., rks**(order-1)].
    var r_mat = List[List[Float64]]()
    for i in range(1, order + 1):
        var row = List[Float64]()
        for j in range(len(rks)):
            # rks[j] ** (i-1)
            var p = 1.0
            for _ in range(i - 1):
                p = p * rks[j]
            row.append(p)
        r_mat.append(row^)

    var rhos = List[Float64]()
    if order == 1:
        rhos.append(0.0)  # placeholder; corrector caller short-circuits to 0.5
    else:
        rhos = _linsolve_f64(r_mat, b_vec)

    return UniPcBh2Coeffs(
        b_h, alpha_t, sigma_t, sigma_s0, alpha_s0, h_phi_1, rks^, rhos^
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stateful tensor scheduler.
# ─────────────────────────────────────────────────────────────────────────────


struct UniPcMultistepScheduler(Movable):
    """FlowUniPCMultistepScheduler, bh2 / solver_order=2 / predict_x0=true.

    Stateful: call `step(model_output, sample)` once per inference step in
    order; `step_index` advances internally. Mirrors cosmos_unipc.rs `step`.
    """

    var num_train_timesteps: Int
    var num_inference_steps: Int
    var solver_order: Int
    var shift: Float64
    var _sigmas: List[Float64]
    # Ring buffer of converted model outputs, length solver_order.
    # _outputs[solver_order-1] is the most recent ("m0"). None == empty slot.
    var _outputs: List[Optional[TArc]]
    var _lower_order_nums: Int
    var _last_sample: Optional[TArc]
    var _step_index: Int
    var _this_order: Int

    def __init__(
        out self,
        num_train_timesteps: Int,
        num_inference_steps: Int,
        shift: Float64,
        solver_order: Int,
    ) raises:
        if solver_order != 2:
            raise Error(
                "UniPcMultistepScheduler: only solver_order=2 is implemented"
            )
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.solver_order = solver_order
        self.shift = shift
        self._sigmas = build_unipc_sigma_schedule(
            num_inference_steps, shift, num_train_timesteps
        )
        self._outputs = List[Optional[TArc]]()
        for _ in range(solver_order):
            self._outputs.append(None)
        self._lower_order_nums = 0
        self._last_sample = None
        self._step_index = 0
        self._this_order = 0

    def sigmas(self) -> List[Float64]:
        return self._sigmas.copy()

    def step_index(self) -> Int:
        return self._step_index

    def this_order(self) -> Int:
        return self._this_order

    def lower_order_nums(self) -> Int:
        return self._lower_order_nums

    def _convert_model_output(
        self, model_output: Tensor, sample: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        # predict_x0: x0_pred = sample - sigma_t * model_output.
        var sigma_t = Float32(self._sigmas[self._step_index])
        var scaled = mul_scalar(model_output, sigma_t, ctx)
        return sub(sample, scaled, ctx)

    def _predictor(
        self, sample: Tensor, order: Int, ctx: DeviceContext
    ) raises -> Tensor:
        # m0 = most recent converted output.
        if not self._outputs[self.solver_order - 1]:
            raise Error("UniPc predictor: m0 is None")
        var m0 = self._outputs[self.solver_order - 1].value()[].clone(ctx)

        var c = compute_bh2_coefficients(
            self._sigmas, self._step_index, order, False
        )

        # x_t_ = sigma_t/sigma_s0 * x - alpha_t*h_phi_1 * m0
        var coef_x = Float32(c.sigma_t / c.sigma_s0)
        var coef_m0 = Float32(c.alpha_t * c.h_phi_1)
        var term_x = mul_scalar(sample, coef_x, ctx)
        var term_m0 = mul_scalar(m0, coef_m0, ctx)
        var x_t_ = sub(term_x, term_m0, ctx)

        if order == 1:
            return x_t_^

        # order == 2: D1s[0] = (m1 - m0)/rks[0]; rhos_p = [0.5] (shortcut).
        if not self._outputs[self.solver_order - 2]:
            raise Error("UniPc predictor: m1 is None for order 2")
        var m1 = self._outputs[self.solver_order - 2].value()[].clone(ctx)
        var rk = c.rks[0]
        var diff = sub(m1, m0, ctx)
        var d1 = mul_scalar(diff, Float32(1.0 / rk), ctx)
        # pred_res = 0.5 * D1s[0]
        var pred_res = mul_scalar(d1, 0.5, ctx)
        # x_t = x_t_ - alpha_t*B_h * pred_res
        var coef_pred = Float32(c.alpha_t * c.b_h)
        var term = mul_scalar(pred_res, coef_pred, ctx)
        return sub(x_t_, term, ctx)

    def _corrector(
        self,
        this_model_output: Tensor,
        last_sample: Tensor,
        order: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        if not self._outputs[self.solver_order - 1]:
            raise Error("UniPc corrector: m0 is None")
        var m0 = self._outputs[self.solver_order - 1].value()[].clone(ctx)

        var c = compute_bh2_coefficients(
            self._sigmas, self._step_index, order, True
        )

        # x_t_ = sigma_t/sigma_s0 * last_sample - alpha_t*h_phi_1 * m0
        var coef_x = Float32(c.sigma_t / c.sigma_s0)
        var coef_m0 = Float32(c.alpha_t * c.h_phi_1)
        var term_x = mul_scalar(last_sample, coef_x, ctx)
        var term_m0 = mul_scalar(m0, coef_m0, ctx)
        var x_t_ = sub(term_x, term_m0, ctx)

        # rhos_c: order==1 → [0.5]; order==2 → full 2x2 solve (c.rhos).
        var rhos_c = List[Float64]()
        if order == 1:
            rhos_c.append(0.5)
        else:
            for i in range(len(c.rhos)):
                rhos_c.append(c.rhos[i])

        # corr_res = rhos_c[0]*D1s[0] for order==2; None for order==1.
        # D1_t = this_model_output - m0; total = corr_res + rhos_c[-1]*D1_t.
        var d1_t = sub(this_model_output, m0, ctx)
        var rho_last = Float32(rhos_c[len(rhos_c) - 1])
        var total = mul_scalar(d1_t, rho_last, ctx)
        if order == 2:
            if not self._outputs[self.solver_order - 2]:
                raise Error("UniPc corrector: m1 is None for order 2")
            var m1 = self._outputs[self.solver_order - 2].value()[].clone(ctx)
            var rk = c.rks[0]
            var diff = sub(m1, m0, ctx)
            var d1 = mul_scalar(diff, Float32(1.0 / rk), ctx)
            var corr_res = mul_scalar(d1, Float32(rhos_c[0]), ctx)
            total = add(corr_res, total, ctx)

        # x_t = x_t_ - alpha_t*B_h * total
        var coef_total = Float32(c.alpha_t * c.b_h)
        var term_total = mul_scalar(total, coef_total, ctx)
        return sub(x_t_, term_total, ctx)

    def step(
        mut self, model_output: Tensor, sample: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        if self._step_index >= self.num_inference_steps:
            raise Error("UniPc step: step_index out of range")

        var use_corrector = self._step_index > 0 and self._last_sample.__bool__()

        var mo_convert = self._convert_model_output(model_output, sample, ctx)

        var sample_after: Tensor
        if use_corrector:
            var last = self._last_sample.value()[].clone(ctx)
            sample_after = self._corrector(
                mo_convert, last, self._this_order, ctx
            )
        else:
            sample_after = sample.clone(ctx)

        # Shift ring buffer left, append the new converted output.
        for i in range(self.solver_order - 1):
            if self._outputs[i + 1]:
                self._outputs[i] = Optional(self._outputs[i + 1].value())
            else:
                self._outputs[i] = None
        self._outputs[self.solver_order - 1] = Optional(TArc(mo_convert^))

        # this_order = min(solver_order, len(timesteps) - step_index)
        # then min(this_order, lower_order_nums + 1).
        var this_order = self.solver_order
        var remaining = self.num_inference_steps - self._step_index
        if remaining < this_order:
            this_order = remaining
        if self._lower_order_nums + 1 < this_order:
            this_order = self._lower_order_nums + 1
        if this_order == 0:
            raise Error("UniPc step: computed this_order=0")
        self._this_order = this_order

        # last_sample = the (possibly corrected) sample.
        self._last_sample = Optional(TArc(sample_after.clone(ctx)))

        # Predictor.
        var prev_sample = self._predictor(sample_after, this_order, ctx)

        if self._lower_order_nums < self.solver_order:
            self._lower_order_nums += 1
        self._step_index += 1
        return prev_sample^
