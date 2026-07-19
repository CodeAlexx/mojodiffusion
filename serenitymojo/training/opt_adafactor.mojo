# training/opt_adafactor.mojo — Adafactor (Shazeer & Stern 2018), host-F32.
#
# NEW STANDALONE MODULE. Mirrors optim.mojo's struct+step idiom (read-only
# reference) and the EXACT F32 algorithm of EriDiffusion-v2
# optimizers.rs::Adafactor::step (~line 354), pinned against its inline
# reference test (adafactor_5_steps_matches_inline_reference, ~line 2197).
# transformers.Adafactor defaults: beta1=None (no first moment), decay_rate=-0.8,
# clip_threshold=1.0, eps=(1e-30, 1e-3), relative_step=False.
#
# Per-step (factored second moment for rank>=2, per-element for rank<=1):
#   beta2t = 1 - step^decay_rate            # decay_rate<0 → beta2t→1 as step grows
#   g_sq = g*g + eps_grad                    # eps_grad = 1e-30
#   FACTORED (ndim>=2, treat as [R, C] = [prod(dims[..-1]), dims[-1]]):
#     mean_last[r]   = mean_c g_sq[r,c]               # over last dim
#     mean_second[c] = mean_r g_sq[r,c]               # over 2nd-to-last dim
#     row = beta2t*row + (1-beta2t)*mean_last
#     col = beta2t*col + (1-beta2t)*mean_second
#     r_factor[r] = rsqrt( row[r] / mean(row) )
#     c_factor[c] = rsqrt( col[c] )
#     update[r,c] = r_factor[r] * c_factor[c] * g[r,c]
#   PER-ELEMENT (ndim<=1):
#     v = beta2t*v + (1-beta2t)*g_sq ; update = rsqrt(v) * g
#   THEN (both paths):
#     rms = sqrt(mean(update^2)) ; update /= max(rms/clip_threshold, 1.0)
#     lr_eff = lr * max(eps_param, RMS(p))  if scale_parameter else lr
#     update *= lr_eff
#     if wd != 0:  p *= (1 - wd*lr_eff)      # decoupled, scales by lr_eff
#     p -= update
#
# IMPLEMENTATION NOTE: the factored row/col reductions + outer-product broadcast
# are shape-dependent (no single elementwise kernel). They are computed on the
# HOST in F32 (parity-grade, mirrors optim.mojo's grad-clip host-reduction
# idiom), then the updated param is uploaded. This matches the Rust F32 math
# exactly. State (row/col/v) is held by the CALLER as host List[Float32] and
# passed by `mut` — there is no persistent device state for this optimizer.
#
# AGENT-DEFAULT: eps_param defaults to 1e-3 if caller passes eps==0 (matches
# Adafactor::with_options). decay_rate=-0.8 and clip_threshold=1.0 are fixed
# (the upstream defaults; not exposed as args, same as the Rust struct).
# Mojo 0.26.x. Host-F32 path; no GPU kernel (shape-dependent reductions).

from std.math import sqrt, exp, log
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DECAY_RATE = Float32(-0.8)
comptime _CLIP_THRESHOLD = Float32(1.0)
comptime _EPS_GRAD = Float32(1.0e-30)


# step^decay_rate with decay_rate = -0.8 (host pow via exp/log; step >= 1).
def _step_pow(step: Int, p: Float32) -> Float32:
    var b = Float32(step)
    return exp(p * log(b))


def _rms_host(x: List[Float32]) -> Float32:
    var acc = Float32(0.0)
    var n = len(x)
    for i in range(n):
        acc += x[i] * x[i]
    return sqrt(acc / Float32(n))


# ── Adafactor step — FACTORED (rank>=2) path ─────────────────────────────────
# Param treated as [R, C] with R = numel/C, C = last_dim. `row` (len R) and
# `col` (len C) are the persistent factored second-moment estimators, owned by
# the caller and updated IN PLACE. `param_host`/`grad_host` are F32 host arrays.
# Returns the updated param as a host List (caller uploads / keeps it).
def adafactor_step_factored(
    mut param_host: List[Float32],
    grad_host: List[Float32],
    mut row: List[Float32],
    mut col: List[Float32],
    R: Int,
    C: Int,
    step: Int,
    lr: Float32,
    eps_param: Float32,
    weight_decay: Float32,
    scale_parameter: Bool,
) raises:
    var n = R * C
    if len(param_host) != n or len(grad_host) != n:
        raise Error("adafactor_step_factored: param/grad numel != R*C")
    if len(row) != R or len(col) != C:
        raise Error("adafactor_step_factored: row len != R or col len != C")
    if step < 1:
        raise Error("adafactor_step_factored: step must be >= 1")

    var beta2t = Float32(1.0) - _step_pow(step, _DECAY_RATE)
    var one_m = Float32(1.0) - beta2t

    # g_sq = g*g + eps_grad
    var g_sq = List[Float32]()
    for i in range(n):
        g_sq.append(grad_host[i] * grad_host[i] + _EPS_GRAD)

    # mean over last dim (per row) and over 2nd-to-last (per col)
    var mean_last = List[Float32]()
    for r in range(R):
        var acc = Float32(0.0)
        for c in range(C):
            acc += g_sq[r * C + c]
        mean_last.append(acc / Float32(C))
    var mean_second = List[Float32]()
    for c in range(C):
        var acc = Float32(0.0)
        for r in range(R):
            acc += g_sq[r * C + c]
        mean_second.append(acc / Float32(R))

    # EMA update of row/col estimators (in place)
    for r in range(R):
        row[r] = beta2t * row[r] + one_m * mean_last[r]
    for c in range(C):
        col[c] = beta2t * col[c] + one_m * mean_second[c]

    # r_factor = rsqrt(row / mean(row)) ; c_factor = rsqrt(col)
    var row_mean = Float32(0.0)
    for r in range(R):
        row_mean += row[r]
    row_mean /= Float32(R)
    var r_factor = List[Float32]()
    for r in range(R):
        r_factor.append(Float32(1.0) / sqrt(row[r] / row_mean))
    var c_factor = List[Float32]()
    for c in range(C):
        c_factor.append(Float32(1.0) / sqrt(col[c]))

    # update[r,c] = r_factor[r] * c_factor[c] * g[r,c]
    var update = List[Float32]()
    for r in range(R):
        for c in range(C):
            update.append(r_factor[r] * c_factor[c] * grad_host[r * C + c])

    _finish(param_host, update, lr, eps_param, weight_decay, scale_parameter)


# ── Adafactor step — GENERAL rank>=2 (leading dims kept separate) ────────────
# The plain `adafactor_step_factored` flattens ALL leading dims into one R, so
# its row_mean is a single global mean over R = numel/C. That matches the Rust
# `mean_dim` semantics ONLY for rank-2 tensors. For rank>=3 the Rust reference
# (optimizers.rs:354) keeps the leading dims separate:
#   grad_mean_last   = g_sq.mean_dim([last])   -> shape dims[..-1]   (per L,R row)
#   grad_mean_second = g_sq.mean_dim([second]) -> shape dims[..-2]+[last] (per L,C)
#   r_factor = rsqrt(row / row.mean(-1, keepdim))  -> row_mean is PER (L) block,
#                                                     reduced only over R.
# We model the tensor as [L, R, C] with L = prod(dims[..-2]), R = dims[-2],
# C = dims[-1]. row is [L,R], col is [L,C]. Each L-block is an independent
# Adafactor factored update. For L=1 this reduces exactly to the rank-2 path,
# so the rank-2 results are unchanged.
def adafactor_step_factored_nd(
    mut param_host: List[Float32],
    grad_host: List[Float32],
    mut row: List[Float32],   # len L*R
    mut col: List[Float32],   # len L*C
    L: Int,
    R: Int,
    C: Int,
    step: Int,
    lr: Float32,
    eps_param: Float32,
    weight_decay: Float32,
    scale_parameter: Bool,
) raises:
    var n = L * R * C
    if len(param_host) != n or len(grad_host) != n:
        raise Error("adafactor_step_factored_nd: param/grad numel != L*R*C")
    if len(row) != L * R or len(col) != L * C:
        raise Error("adafactor_step_factored_nd: row len != L*R or col len != L*C")
    if step < 1:
        raise Error("adafactor_step_factored_nd: step must be >= 1")

    var beta2t = Float32(1.0) - _step_pow(step, _DECAY_RATE)
    var one_m = Float32(1.0) - beta2t

    var update = List[Float32]()
    for _ in range(n):
        update.append(0.0)

    # Process each leading-dim block independently (these are the dims the Rust
    # mean_dim keeps separate).
    for l in range(L):
        var base = l * R * C
        # mean over last dim (per row r) and over 2nd-to-last (per col c)
        var mean_last = List[Float32]()
        for r in range(R):
            var acc = Float32(0.0)
            for c in range(C):
                var g = grad_host[base + r * C + c]
                acc += g * g + _EPS_GRAD
            mean_last.append(acc / Float32(C))
        var mean_second = List[Float32]()
        for c in range(C):
            var acc = Float32(0.0)
            for r in range(R):
                var g = grad_host[base + r * C + c]
                acc += g * g + _EPS_GRAD
            mean_second.append(acc / Float32(R))

        # EMA update of this block's row/col estimators (in place)
        var row_base = l * R
        var col_base = l * C
        for r in range(R):
            row[row_base + r] = beta2t * row[row_base + r] + one_m * mean_last[r]
        for c in range(C):
            col[col_base + c] = beta2t * col[col_base + c] + one_m * mean_second[c]

        # row_mean is the mean over R WITHIN this block (Rust mean_dim(-1)).
        var row_mean = Float32(0.0)
        for r in range(R):
            row_mean += row[row_base + r]
        row_mean /= Float32(R)
        var r_factor = List[Float32]()
        for r in range(R):
            r_factor.append(Float32(1.0) / sqrt(row[row_base + r] / row_mean))
        var c_factor = List[Float32]()
        for c in range(C):
            c_factor.append(Float32(1.0) / sqrt(col[col_base + c]))

        for r in range(R):
            for c in range(C):
                update[base + r * C + c] = (
                    r_factor[r] * c_factor[c] * grad_host[base + r * C + c]
                )

    _finish(param_host, update, lr, eps_param, weight_decay, scale_parameter)


# ── Adafactor step — PER-ELEMENT (rank<=1) path ──────────────────────────────
# `v` is the persistent per-element second moment (len n), owned by caller,
# updated IN PLACE.
def adafactor_step_elementwise(
    mut param_host: List[Float32],
    grad_host: List[Float32],
    mut v: List[Float32],
    step: Int,
    lr: Float32,
    eps_param: Float32,
    weight_decay: Float32,
    scale_parameter: Bool,
) raises:
    var n = len(param_host)
    if len(grad_host) != n or len(v) != n:
        raise Error("adafactor_step_elementwise: param/grad/v numel mismatch")
    if step < 1:
        raise Error("adafactor_step_elementwise: step must be >= 1")

    var beta2t = Float32(1.0) - _step_pow(step, _DECAY_RATE)
    var one_m = Float32(1.0) - beta2t

    var update = List[Float32]()
    for i in range(n):
        var g_sq = grad_host[i] * grad_host[i] + _EPS_GRAD
        v[i] = beta2t * v[i] + one_m * g_sq
        update.append((Float32(1.0) / sqrt(v[i])) * grad_host[i])

    _finish(param_host, update, lr, eps_param, weight_decay, scale_parameter)


# ── shared tail: RMS clip → lr_eff → decoupled WD → subtract ─────────────────
def _finish(
    mut param_host: List[Float32],
    mut update: List[Float32],
    lr: Float32,
    eps_param: Float32,
    weight_decay: Float32,
    scale_parameter: Bool,
):
    var n = len(param_host)

    # RMS clipping: update /= max(rms/clip_threshold, 1.0)
    var rms = _rms_host(update)
    var scale_div = rms / _CLIP_THRESHOLD
    if scale_div < 1.0:
        scale_div = 1.0
    for i in range(n):
        update[i] = update[i] / scale_div

    # lr_eff = lr * max(eps_param, RMS(p))  if scale_parameter else lr
    var lr_eff = lr
    if scale_parameter:
        var p_rms = _rms_host(param_host)
        if p_rms < eps_param:
            p_rms = eps_param
        lr_eff = lr * p_rms

    for i in range(n):
        update[i] = update[i] * lr_eff

    # decoupled WD (scales by lr_eff) then subtract update
    for i in range(n):
        if weight_decay != 0.0:
            param_host[i] = param_host[i] * (Float32(1.0) - weight_decay * lr_eff)
        param_host[i] = param_host[i] - update[i]


# ── eps_param default helper (eps==0 → 1e-3, mirrors with_options) ───────────
def adafactor_eps_param(eps: Float32) -> Float32:
    if eps == 0.0:
        return Float32(1.0e-3)
    return eps
