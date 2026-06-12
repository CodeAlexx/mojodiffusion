# training/adafactor.mojo — host-math Adafactor for 2D LoRA matrices (T1.C).
#
# Reference (the implementation SimpleTuner actually invokes): SimpleTuner key
# "torch-adafactor" -> torch.optim.Adafactor
#   * registered: /home/alex/SimpleTuner/simpletuner/helpers/training/
#     optimizer_param.py:153-162 with default_settings beta2_decay=-0.8,
#     eps=(None, 1e-3), d=1.0, weight_decay=0.0 (lr explicit from args).
#     The old transformers-Adafactor key "adafactor" was REMOVED upstream
#     (optimizer_param.py:879) — torch's is the live one.
#   * math: torch/optim/_adafactor.py:330-416 (_single_tensor_adafactor,
#     torch 2.10.0). Per step t (1-based), per 2D param:
#       one_minus_beta2_t = t ** beta2_decay                       # :379
#       rho_t  = min(lr, 1/sqrt(t))                                # :380
#       alpha  = max(eps2, RMS(p)) * rho_t                         # :381
#       if wd != 0: p *= (1 - lr*wd)        # "stepweight decay"   # :384-385
#       row_var.lerp_(mean(g*g, dim=-1), one_minus_beta2_t)        # :393-396
#       col_var.lerp_(mean(g*g, dim=-2), one_minus_beta2_t)        # :398-401
#       var_est = (row_var @ col_var) / clamp(mean(row_var), eps1) # :402-403
#       update = rsqrt(clamp(var_est, min=eps1^2)) * g             # :413-414
#       denom  = max(1, ||update||_2 / (sqrt(numel)*d))            # :415
#       p += (-alpha/denom) * update                               # :416
#     eps1 = None in SimpleTuner's defaults -> torch.finfo(param.dtype).eps
#     (:372-373); for F32 that is 2^-23 = 1.1920928955078125e-07.
#
# Notes vs torch:
#   * There is NO relative-step / scale_parameter mode in torch's Adafactor
#     (that was the transformers variant); lr is explicit and rho_t clips it
#     at 1/sqrt(t). We mirror that exactly.
#   * update clipping is the d-threshold RMS clip at :415 (d=1.0 default).
#   * Scalar bookkeeping (rho_t, alpha, denom) is Float64 like torch's python
#     floats / .item() calls; element math is Float32 like the f32 tensors.
#     Reductions accumulate in Float64 (torch's vectorized f32 reduction order
#     is unreproducible anyway; gate is rel<=1e-5 / cos>=0.999999).
#
# Host-math first (List[Float32] params/grads — the `_adamw_host_list` style,
# see training/lora_adamw_plain_fused.mojo header); fused GPU path comes later
# behind the same dispatch (wiring design in the T1.C report).
#
# Parity gate: training/tests/optimizer_parity.mojo vs
# /tmp/optimizer_oracle.safetensors (gen_optimizer_oracle.py, 20 steps).
#
# Mojo 1.0.0b1.

from std.math import sqrt


comptime ADAFACTOR_EPS1_F32 = Float64(1.1920928955078125e-07)
"""torch.finfo(torch.float32).eps — what eps[0]=None resolves to for F32
params (_adafactor.py:372-373). Pass eps1 <= 0 to adafactor_step_2d to get
this default."""


struct AdafactorState(Movable):
    """Per-parameter factored second-moment state for a 2D [rows, cols]
    matrix: row_var [rows] (mean of g^2 over cols), col_var [cols] (mean of
    g^2 over rows), and the 1-based step count. Mirrors torch state
    row_var/col_var/step (_adafactor.py:98-109)."""

    var row_var: List[Float32]
    var col_var: List[Float32]
    var rows: Int
    var cols: Int
    var step: Int

    def __init__(out self, rows: Int, cols: Int):
        self.row_var = List[Float32]()
        for _ in range(rows):
            self.row_var.append(Float32(0.0))
        self.col_var = List[Float32]()
        for _ in range(cols):
            self.col_var.append(Float32(0.0))
        self.rows = rows
        self.cols = cols
        self.step = 0


def adafactor_step_2d(
    mut p: List[Float32],
    g: List[Float32],
    mut state: AdafactorState,
    lr: Float64,
    beta2_decay: Float64,
    eps1_in: Float64,
    eps2: Float64,
    d: Float64,
    weight_decay: Float64,
) raises:
    """One Adafactor step on one 2D param (row-major [state.rows,
    state.cols]). Mutates p and state in place. eps1_in <= 0 selects the
    torch F32 default (ADAFACTOR_EPS1_F32). SimpleTuner "torch-adafactor"
    settings: beta2_decay=-0.8, eps1_in=-1 (None), eps2=1e-3, d=1.0,
    weight_decay=0.0."""
    var rows = state.rows
    var cols = state.cols
    var n = rows * cols
    if len(p) != n or len(g) != n:
        raise Error("adafactor_step_2d: p/g length != rows*cols")
    var eps1 = eps1_in
    if eps1 <= 0.0:
        eps1 = ADAFACTOR_EPS1_F32

    state.step += 1
    var t = Float64(state.step)
    var one_minus_beta2_t = t**beta2_decay  # _adafactor.py:379
    var rho_t = min(lr, 1.0 / sqrt(t))  # :380

    # alpha = max(eps2, RMS(p)) * rho_t (:381) — uses p BEFORE weight decay.
    var p_sq = Float64(0.0)
    for i in range(n):
        p_sq += Float64(p[i]) * Float64(p[i])
    var p_rms = sqrt(p_sq) / sqrt(Float64(n))
    var alpha = max(eps2, p_rms) * rho_t

    # Stepweight decay (:384-385): p *= (1 - lr*wd), f32.
    if weight_decay != 0.0:
        var wmul = Float32(1.0 - lr * weight_decay)
        for i in range(n):
            p[i] = p[i] * wmul

    # row/col second-moment factors: lerp(old, mean(g^2), one_minus_beta2_t)
    # i.e. old + w*(mean - old) (:393-401).
    var w = Float32(one_minus_beta2_t)
    for r in range(rows):
        var s = Float64(0.0)
        for c in range(cols):
            var gv = Float64(g[r * cols + c])
            s += gv * gv
        var row_mean = Float32(s / Float64(cols))
        state.row_var[r] = state.row_var[r] + w * (row_mean - state.row_var[r])
    for c in range(cols):
        var s = Float64(0.0)
        for r in range(rows):
            var gv = Float64(g[r * cols + c])
            s += gv * gv
        var col_mean = Float32(s / Float64(rows))
        state.col_var[c] = state.col_var[c] + w * (col_mean - state.col_var[c])

    # var_est = outer(row_var, col_var) / clamp(mean(row_var), min=eps1)
    # (:402-403), then update = rsqrt(clamp(var_est, min=eps1^2)) * g
    # (:413-414) with d-threshold RMS clipping (:415-416).
    var rv_sum = Float64(0.0)
    for r in range(rows):
        rv_sum += Float64(state.row_var[r])
    var rv_mean = Float32(rv_sum / Float64(rows))
    if Float64(rv_mean) < eps1:
        rv_mean = Float32(eps1)
    var eps1_sq = Float32(eps1 * eps1)

    var update = List[Float32](capacity=n)
    var u_sq = Float64(0.0)
    for r in range(rows):
        for c in range(cols):
            var ve = (state.row_var[r] * state.col_var[c]) / rv_mean
            if ve < eps1_sq:
                ve = eps1_sq
            var u = (Float32(1.0) / sqrt(ve)) * g[r * cols + c]
            update.append(u)
            u_sq += Float64(u) * Float64(u)
    var u_rms = sqrt(u_sq) / (sqrt(Float64(n)) * d)
    var denom = max(1.0, u_rms)
    var coeff = Float32(-(alpha / denom))
    for i in range(n):
        p[i] = p[i] + coeff * update[i]
