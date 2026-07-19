# adafactor.mojo — pure-Mojo port of Serenity's Adafactor
# (modules/util/optimizer/adafactor_extensions.py + the transformers Adafactor
# base it patches: transformers/optimization.py::Adafactor).
#
# MEASURED semantics (adafactor_extensions.py:17-97, optimization.py:804-925):
#   _get_options(group, shape): factored = len(shape) >= 2 ;
#                               use_first_moment = (beta1 is not None)   (opt.py:815-819)
#   State (zeros_like(grad)/zeros(...).to(grad) → BF16 per dtype policy):
#     factored:  exp_avg_sq_row = zeros(shape[:-1]) ; exp_avg_sq_col = zeros(shape[:-2]+shape[-1:])
#     else:      exp_avg_sq     = zeros_like(grad)
#     if use_first_moment: exp_avg = zeros_like(grad)
#     RMS = 0 (recomputed every step)
#   Per step (adafactor_extensions.py:57-95):
#     step += 1
#     RMS = _rms(p) = ||p||_2 / sqrt(numel)                     (opt.py:821-823, :58)
#     lr  = _get_lr(group, state)                               (opt.py:804-813, :59)
#     beta2t = 1 - step ** decay_rate                           (:61)
#     update = grad**2 + eps[0]                                 (:62)
#     factored:
#       row.mul_(beta2t).add_(update.mean(dim=-1), 1-beta2t)    (:67)
#       col.mul_(beta2t).add_(update.mean(dim=-2), 1-beta2t)    (:68)
#       update = _approx_sq_grad(row, col) ; update *= grad     (:71-72, opt.py:825-831)
#         r_factor = rsqrt(row / row.mean(dim=-1,keepdim)) [.,:,None]
#         c_factor = rsqrt(col) [.,None,:]
#         approx   = r_factor * c_factor
#     else:
#       v.mul_(beta2t).add_(update, 1-beta2t)                   (:76)
#       update = rsqrt(v) * grad                                (:77)
#     update /= max(_rms(update)/clip_threshold, 1.0)           (:79)
#     update *= lr                                              (:80)
#     if use_first_moment:
#       exp_avg.mul_(beta1).add_(update, 1-beta1) ; update = exp_avg   (:83-85)
#     if weight_decay != 0: p += -(weight_decay*lr) * p         (:87-88)
#     p += -update                                             (:90)
#     bf16 + stochastic_rounding → copy_stochastic_(p, p_fp32)  (:92-95)
#
# DTYPE POLICY: p, exp_avg, exp_avg_sq(_row/_col) are ALL BF16 STORAGE. No F32
# master, no F32 moment state. The factored row/col means and the two _rms(.)
# norms are SCALAR/host reductions (allowed in F32). Every per-element value is
# computed in an F32 register and written back to BF16 via the same SR helper as
# adamw. Stochastic rounding mirrors copy_stochastic_ (bf16_stochastic_rounding.py).
#
# This step is orchestrated host-side (reductions over the trailing two axes plus
# two global norms make a single fused GPU kernel impractical, and the reductions
# are exactly the "scalar reductions / host stats" the dtype policy permits in
# F32). BF16 in, F32 compute, BF16 out — zero persistent F32 tensors.

from std.gpu.host import DeviceContext
from std.builtin.dtype import DType
from std.math import sqrt, floor, log, pow
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.torch_bf16 import torch_bf16_rne_value

comptime _LN2 = Float64(0.69314718055994530942)
comptime _U24 = Float32(1.0) / Float32(16777216.0)  # 1/2^24 → uniform [0,1)


# PCG-style hash → uniform UInt32 (identical to adamw.mojo::_pcg_hash; the RNG
# behind copy_stochastic_'s random 16-bit mantissa add, bf16_stochastic_rounding.py:24).
def _pcg_hash(x: UInt32) -> UInt32:
    var state = x * UInt32(747796405) + UInt32(2891336453)
    var shift = (state >> UInt32(28)) + UInt32(4)
    var word = ((state >> shift) ^ state) * UInt32(277803737)
    return (word >> UInt32(22)) ^ word


# Stochastic round f32 -> bf16 (identical to adamw.mojo::_sr_bf16). `u` uniform
# [0,1). Unbiased: E[result] == v, matching copy_stochastic_'s random-mantissa
# trick (bf16_stochastic_rounding.py:33-40). Float math only (no scalar f32
# bit-reinterpret in Mojo 1.0.0b1 kernels).
def _sr_bf16(v: Float32, u: Float32) -> BFloat16:
    if not (v == v):  # NaN
        return v.cast[DType.bfloat16]()
    if v == Float32(0.0):
        return BFloat16(0.0)
    var sign = Float32(1.0)
    var a = v
    if a < Float32(0.0):
        sign = Float32(-1.0)
        a = -a
    if a < Float32(1.0e-38):
        return v.cast[DType.bfloat16]()
    var av = Float64(a)
    var e = Int(floor(log(av) / _LN2))            # binade
    var step_ulp = pow(Float64(2.0), Float64(e - 7))  # bf16 ULP (7 mantissa bits)
    var y = av / step_ulp
    var kf = floor(y)
    var frac = y - kf
    var k = Int(kf)
    if Float64(u) < frac:
        k += 1
    var q = Float32(Float64(k) * step_ulp)
    if sign < Float32(0.0):
        q = -q
    return q.cast[DType.bfloat16]()


# round-to-nearest-even bf16 (no SR path) — used when stochastic_rounding=False,
# matching torch's p.copy_(p_fp32) (adafactor_extensions.py:94-95). Routed through
# the CUDA-parity RNE helper (cvt.rn.bf16.f32 semantics) exactly as adamw.mojo does,
# since Mojo's native cast[bfloat16] differs by one BF16 quantum on some values
# (torch_bf16.mojo:4-7).
def _rne_bf16(v: Float32) -> BFloat16:
    return torch_bf16_rne_value(v)


# host helper: ||x||_2 / sqrt(numel)  (transformers _rms, opt.py:821-823).
# Reduction performed in F32 (scalar host stat).
def _rms_host(x: List[Float32]) -> Float32:
    var n = len(x)
    if n == 0:
        return Float32(0.0)
    var ss = Float64(0.0)
    for i in range(n):
        var xv = Float64(x[i])
        ss += xv * xv
    var norm = sqrt(ss)
    return Float32(norm / sqrt(Float64(n)))


# _get_lr port (opt.py:804-813). param_state RMS passed in; step is 1-based.
#   rel_step_sz = lr
#   if relative_step:
#       min_step = 1e-6*step if warmup_init else 1e-2
#       rel_step_sz = min(min_step, 1/sqrt(step))
#   param_scale = 1.0 ; if scale_parameter: param_scale = max(eps[1], RMS)
#   return param_scale * rel_step_sz
def _get_lr(
    lr: Float32,
    step: Int,
    rms: Float32,
    eps_param_scale: Float32,   # eps[1] (transformers param_scale floor), opt.py:812
    relative_step: Bool,
    warmup_init: Bool,
    scale_parameter: Bool,
) -> Float32:
    var rel_step_sz = lr
    if relative_step:
        var min_step = Float32(1.0e-2)
        if warmup_init:
            min_step = Float32(1.0e-6) * Float32(step)
        var inv = Float32(1.0) / sqrt(Float32(step))
        rel_step_sz = min_step if min_step < inv else inv
    var param_scale = Float32(1.0)
    if scale_parameter:
        param_scale = eps_param_scale if eps_param_scale > rms else rms
    return param_scale * rel_step_sz


# Host-side single-tensor Adafactor step. All tensors BF16, mutated IN PLACE.
#
# shape: full grad/param shape (row-major). For factored (len>=2):
#   row state has shape[:-1]  → numel = numel(p) / shape[-1]
#   col state has shape[:-2]+shape[-1:] → numel = numel(p) / shape[-2]
# `row`/`col` are ignored when not factored; `exp_avg_sq` is ignored when factored.
# `exp_avg` is ignored when use_first_moment is False (beta1 is None → pass 0.0
# AND use_first_moment=False).
#
# step: 1-based AFTER increment (caller passes the post-increment value, matching
# state["step"] += 1 at :57 before its use).
def adafactor_step(
    p: Tensor,
    g: Tensor,
    exp_avg: Tensor,          # first-moment EMA (BF16), used iff use_first_moment
    exp_avg_sq: Tensor,       # non-factored 2nd moment (BF16), used iff not factored
    row: Tensor,              # factored row EMA (BF16), used iff factored
    col: Tensor,              # factored col EMA (BF16), used iff factored
    shape: List[Int],
    step: Int,
    lr: Float32,
    eps1: Float32,            # eps[0]  (default 1e-30)
    eps2: Float32,            # eps[1]  (default 1e-3)
    clip_threshold: Float32,  # default 1.0
    decay_rate: Float32,      # default -0.8
    beta1: Float32,           # only read when use_first_moment
    weight_decay: Float32,
    use_first_moment: Bool,
    relative_step: Bool,
    scale_parameter: Bool,
    warmup_init: Bool,
    stochastic_rounding: Bool,
    seed: UInt32,
    ctx: DeviceContext,
) raises:
    if p.dtype() != STDtype.BF16 or g.dtype() != STDtype.BF16:
        raise Error("adafactor_step: param/grad must be BF16 (port dtype policy)")
    if step < 1:
        raise Error("adafactor_step: step must be >= 1 (1-based)")

    var ndim = len(shape)
    var factored = ndim >= 2     # _get_options (opt.py:817)
    var n = p.numel()
    var ncheck = 1
    for d in range(ndim):
        ncheck *= shape[d]
    if ncheck != n:
        raise Error("adafactor_step: shape does not match param numel")
    if g.numel() != n:
        raise Error("adafactor_step: grad numel mismatch")

    # Pull operands to host F32 registers (BF16 storage → F32 compute).
    var pf = p.to_host(ctx)
    var gf = g.to_host(ctx)

    # RMS of the *current* param, then lr (adafactor_extensions.py:58-59).
    var rms_p = _rms_host(pf)
    var lr_eff = _get_lr(lr, step, rms_p, eps2, relative_step, warmup_init, scale_parameter)

    # beta2t = 1 - step ** decay_rate  (:61). step^decay_rate via exp/log.
    var beta2t = Float32(1.0) - Float32(
        pow(Float64(step), Float64(decay_rate))
    )
    var one_m_b2 = Float32(1.0) - beta2t

    # update_in = grad**2 + eps[0]  (:62)
    var upd = List[Float32](capacity=n)
    for i in range(n):
        upd.append(gf[i] * gf[i] + eps1)

    # `update` will be overwritten by the normalized step direction below.
    var update = List[Float32](capacity=n)
    for _ in range(n):
        update.append(Float32(0.0))

    if factored:
        # Trailing two axes are the matrix; everything before is batched.
        var last = shape[ndim - 1]          # columns
        var second = shape[ndim - 2]        # rows
        var outer = n // (last * second)    # product of leading dims
        var rows_n = n // last              # numel of row state (shape[:-1])
        var cols_n = n // second            # numel of col state (shape[:-2]+last)

        var rf = row.to_host(ctx)           # BF16 → F32
        var cf = col.to_host(ctx)
        if len(rf) != rows_n or len(cf) != cols_n:
            raise Error("adafactor_step: factored row/col state numel mismatch")

        # row mean over dim=-1 (mean of `upd` across `last`) then EMA (:67).
        # row index maps to (outer, second): rows_n = outer*second.
        for ridx in range(rows_n):
            var base = ridx * last
            var acc = Float64(0.0)
            for c in range(last):
                acc += Float64(upd[base + c])
            var mean_row = Float32(acc / Float64(last))
            rf[ridx] = beta2t * rf[ridx] + one_m_b2 * mean_row

        # col mean over dim=-2 (mean of `upd` across `second`) then EMA (:68).
        # col state index = o*last + c, mean over the `second` rows of block o.
        for o in range(outer):
            for c in range(last):
                var acc = Float64(0.0)
                for r in range(second):
                    acc += Float64(upd[(o * second + r) * last + c])
                var mean_col = Float32(acc / Float64(second))
                var cidx = o * last + c
                cf[cidx] = beta2t * cf[cidx] + one_m_b2 * mean_col

        # _approx_sq_grad (opt.py:825-831):
        #   r_factor = rsqrt(row / row.mean(dim=-1,keepdim))[..,None]
        #   c_factor = rsqrt(col)[..,None,:]
        #   approx = r_factor * c_factor ; update = approx * grad  (:71-72)
        # row.mean(dim=-1) is the mean of row-state across `second` (the trailing
        # axis of the row tensor, whose shape is shape[:-1] = leading.. * second).
        for o in range(outer):
            # mean of row block (the `second` entries) for this outer index
            var racc = Float64(0.0)
            for r in range(second):
                racc += Float64(rf[o * second + r])
            var rmean = Float32(racc / Float64(second))
            for r in range(second):
                var r_factor = Float32(1.0) / sqrt(rf[o * second + r] / rmean)
                var pbase = (o * second + r) * last
                var cbase = o * last
                for c in range(last):
                    var c_factor = Float32(1.0) / sqrt(cf[cbase + c])
                    update[pbase + c] = r_factor * c_factor * gf[pbase + c]

        # write factored state back to BF16 (in place) with SR.
        _store_bf16(row, rf, stochastic_rounding, seed ^ UInt32(0x9E3779B1), ctx)
        _store_bf16(col, cf, stochastic_rounding, seed ^ UInt32(0x85EBCA77), ctx)
    else:
        # non-factored: exp_avg_sq EMA then rsqrt*grad (:76-77).
        var vf = exp_avg_sq.to_host(ctx)
        if len(vf) != n:
            raise Error("adafactor_step: exp_avg_sq numel mismatch")
        for i in range(n):
            vf[i] = beta2t * vf[i] + one_m_b2 * upd[i]
            update[i] = (Float32(1.0) / sqrt(vf[i])) * gf[i]
        _store_bf16(exp_avg_sq, vf, stochastic_rounding, seed ^ UInt32(0xC2B2AE35), ctx)

    # update /= max(_rms(update)/clip_threshold, 1.0)  (:79)
    var rms_u = _rms_host(update)
    var clip = rms_u / clip_threshold
    var denom_clip = clip if clip > Float32(1.0) else Float32(1.0)
    var inv_clip = Float32(1.0) / denom_clip
    for i in range(n):
        update[i] = update[i] * inv_clip * lr_eff   # also *lr (:80)

    # first moment (:83-85)
    if use_first_moment:
        var ea = exp_avg.to_host(ctx)
        if len(ea) != n:
            raise Error("adafactor_step: exp_avg numel mismatch")
        var one_m_b1 = Float32(1.0) - beta1
        for i in range(n):
            ea[i] = beta1 * ea[i] + one_m_b1 * update[i]
            update[i] = ea[i]
        _store_bf16(exp_avg, ea, stochastic_rounding, seed ^ UInt32(0x27D4EB2F), ctx)

    # weight decay (:87-88) then p += -update (:90)
    var wd_factor = -weight_decay * lr_eff
    for i in range(n):
        var newp = pf[i]
        if weight_decay != Float32(0.0):
            newp = newp + wd_factor * newp
        newp = newp - update[i]
        pf[i] = newp

    _store_bf16(p, pf, stochastic_rounding, seed, ctx)


# Quantize an F32 host buffer to BF16 (with optional stochastic rounding) and
# copy it back IN PLACE into `t`'s existing device buffer. Mirrors
# copy_stochastic_(p, p_fp32) / p.copy_(p_fp32) (bf16_stochastic_rounding.py:40,
# adafactor_extensions.py:93-95). Writes the bf16 elements into `t.buf` so the
# tensor's identity (buffer + autograd id) is preserved — same in-place contract
# adamw.mojo relies on. Uses the same H2D staging pattern as Tensor.from_host but
# targets the existing buffer rather than allocating a new one.
def _store_bf16(
    t: Tensor,
    src: List[Float32],
    stochastic: Bool,
    seed: UInt32,
    ctx: DeviceContext,
) raises:
    # Guard the dtype contract: this writes n*2 bytes into t.buf. If t were F32
    # (n*4 bytes) the copy would silently overwrite only the first half of the
    # buffer (corrupt state, no error). All Adafactor state must be BF16.
    if t.dtype() != STDtype.BF16:
        raise Error("_store_bf16: target tensor must be BF16 (port dtype policy)")
    var n = len(src)
    var nbytes = n * 2  # BF16 = 2 bytes/elem
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var bp = host.unsafe_ptr().bitcast[BFloat16]()
    if stochastic:
        for i in range(n):
            var rnd = _pcg_hash(seed ^ UInt32(i))
            var u = Float32(Int(rnd >> UInt32(8))) * _U24   # uniform [0,1)
            bp[i] = _sr_bf16(src[i], u)
    else:
        for i in range(n):
            bp[i] = _rne_bf16(src[i])
    ctx.enqueue_copy(dst_buf=t.buf, src_buf=host)
    ctx.synchronize()
