# CAME.mojo — pure-Mojo port of Serenity's CAME optimizer
# (modules/util/optimizer/CAME.py + bf16_stochastic_rounding.py).
#
# CAME = Confidence-guided Adaptive Memory Efficient optimization
# (https://github.com/yangluo7/CAME). It is Adafactor-shaped (factored second
# moment) PLUS a confidence-guided "instability" term (factored res_row/res_col)
# and optional cautious masking.
#
# MEASURED semantics (CAME.py::step_parameter, :83-187). betas = (b1, b2, b3):
#   factored = len(shape) >= 2                                  (:67-69)
#   State (zeros_like(grad) / zeros(...).type_as(grad) → BF16):
#     exp_avg = zeros_like(grad)                                (:101)
#     factored: exp_avg_sq_row = zeros(shape[:-1])
#               exp_avg_sq_col = zeros(shape[:-2]+shape[-1:])   (:103-106)
#               exp_avg_res_row = zeros(shape[:-1])
#               exp_avg_res_col = zeros(shape[:-2]+shape[-1:])  (:108-111)
#     else:     exp_avg_sq = zeros_like(grad)                   (:113)
#     RMS = 0 (recomputed each step)                            (:115)
#   Per step (:117-187):
#     step += 1                                                 (:117)
#     RMS  = _rms(p) = ||p||_2 / sqrt(numel)                    (:71-72,:118)
#     update = grad**2 + eps[0]                                 (:120)
#     factored:
#       sq_row.mul_(b2).add_(update.mean(dim=-1), 1-b2)         (:125-127)
#       sq_col.mul_(b2).add_(update.mean(dim=-2), 1-b2)         (:128-130)
#       update = _approx_sq_grad(sq_row, sq_col) ; update *= grad   (:133-134)
#         r_factor = rsqrt(row / row.mean(dim=-1,keepdim))[..,None]  (:74-81)
#         c_factor = rsqrt(col)[..,None,:] ; approx = r_factor*c_factor
#     else:
#       sq.mul_(b2).add_(update, 1-b2) ; update = rsqrt(sq)*grad     (:136-139)
#     update /= max(_rms(update)/clip_threshold, 1.0)           (:141-143)
#     exp_avg.mul_(b1).add_(update, 1-b1)                       (:145-146)
#     # confidence-guided instability
#     res = (update - exp_avg)**2 + eps[1]                      (:150)
#     factored:
#       res_row.mul_(b3).add_(res.mean(dim=-1), 1-b3)           (:156-158)
#       res_col.mul_(b3).add_(res.mean(dim=-2), 1-b3)           (:159-161)
#       res_approx = _approx_sq_grad(res_row, res_col)
#       update = res_approx * exp_avg                           (:164-165)
#     else:
#       update = exp_avg.clone()                                (:167)
#     if use_cautious:                                          (:169-172)
#       mask = (update*grad > 0).to(dtype)
#       mask /= mask.mean().clamp_(min=1e-3) ; update *= mask
#     if weight_decay != 0: p += -(wd*lr) * p                   (:174-181)
#     update *= lr ; p += -update                              (:183-187)
#
# DTYPE POLICY (matches Serenity + adafactor.mojo): p, exp_avg, all factored
# row/col state are BF16 STORAGE. No F32 master, no F32 moment state. Factored
# means and the two _rms() norms are SCALAR/host reductions (allowed in F32).
# Every per-element value is computed in an F32 register and written back to BF16
# via the shared SR helper. BF16 in, F32 compute, BF16 out — zero persistent F32.
#
# This step is orchestrated host-side: factored reductions over the trailing two
# axes plus the two global RMS norms (and the cautious-mask mean) make a single
# fused GPU kernel impractical, exactly as in adafactor.mojo. Stochastic rounding
# mirrors add_stochastic_ (CAME.py:176,185 → bf16_stochastic_rounding.py:45-57).

from std.gpu.host import DeviceContext
from std.builtin.dtype import DType
from std.math import sqrt, floor, log, pow
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.torch_bf16 import torch_bf16_rne_value

# Shared SR helpers (1:1 of bf16_stochastic_rounding.py): _sr_bf16 is the
# copy_stochastic_ equivalent (used by add_stochastic_), _pcg_hash the random
# 16-bit source.
from serenity_trainer.util.bf16_stochastic_rounding import _sr_bf16, _pcg_hash

comptime _U24 = Float32(1.0) / Float32(16777216.0)  # 1/2^24 → uniform [0,1)


# round-to-nearest-even bf16 (non-SR path) — matches torch's p.data.add_(-update)
# write-back when stochastic_rounding=False (CAME.py:178-181,:186-187). Routed
# through the CUDA-parity RNE helper exactly as adamw.mojo / adafactor.mojo, since
# Mojo's native cast[bfloat16] differs by one BF16 quantum on some values.
def _rne_bf16(v: Float32) -> BFloat16:
    return torch_bf16_rne_value(v)


# host helper: ||x||_2 / sqrt(numel)  (CAME._rms, CAME.py:71-72). F32 reduction.
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


# Host-side single-tensor CAME step. All tensors BF16, mutated IN PLACE.
#
# shape: full grad/param shape (row-major). For factored (len>=2):
#   row state has shape[:-1]              → numel = numel(p) / shape[-1]
#   col state has shape[:-2]+shape[-1:]   → numel = numel(p) / shape[-2]
# `sq_row/sq_col/res_row/res_col` are ignored when not factored; `exp_avg_sq` is
# ignored when factored. `exp_avg` is always used (CAME always keeps a first
# moment — :145-146).
#
# step: 1-based AFTER increment (caller passes the post-increment value, matching
# state["step"] += 1 at :117 before its use).
def came_step(
    p: Tensor,
    g: Tensor,
    exp_avg: Tensor,        # first-moment EMA (BF16), always used
    exp_avg_sq: Tensor,     # non-factored 2nd moment (BF16), used iff not factored
    sq_row: Tensor,         # factored row 2nd moment (BF16), used iff factored
    sq_col: Tensor,         # factored col 2nd moment (BF16), used iff factored
    res_row: Tensor,        # factored row instability (BF16), used iff factored
    res_col: Tensor,        # factored col instability (BF16), used iff factored
    shape: List[Int],
    step: Int,
    lr: Float32,
    eps1: Float32,          # eps[0]  (default 1e-30)
    eps2: Float32,          # eps[1]  (default 1e-16)
    clip_threshold: Float32,  # default 1.0
    beta1: Float32,         # betas[0] (default 0.9)
    beta2: Float32,         # betas[1] (default 0.999)
    beta3: Float32,         # betas[2] (default 0.9999)
    weight_decay: Float32,
    use_cautious: Bool,
    stochastic_rounding: Bool,
    seed: UInt32,
    ctx: DeviceContext,
) raises:
    if p.dtype() != STDtype.BF16 or g.dtype() != STDtype.BF16:
        raise Error("came_step: param/grad must be BF16 (port dtype policy)")
    if step < 1:
        raise Error("came_step: step must be >= 1 (1-based)")

    var ndim = len(shape)
    var factored = ndim >= 2     # _get_options (CAME.py:67-69)
    var n = p.numel()
    var ncheck = 1
    for d in range(ndim):
        ncheck *= shape[d]
    if ncheck != n:
        raise Error("came_step: shape does not match param numel")
    if g.numel() != n:
        raise Error("came_step: grad numel mismatch")

    # Pull operands to host F32 registers (BF16 storage → F32 compute). CAME casts
    # grad to float at :87-89; param is read both for RMS and for the update.
    var pf = p.to_host(ctx)
    var gf = g.to_host(ctx)

    # RMS of current param (:118) — computed but only stored in state (RMS is not
    # consumed by the update in CAME; kept for parity with the source).
    var _rms_p = _rms_host(pf)

    # update_in = grad**2 + eps[0]  (:120)
    var upd = List[Float32](capacity=n)
    for i in range(n):
        upd.append(gf[i] * gf[i] + eps1)

    var one_m_b2 = Float32(1.0) - beta2

    # `update` = normalized adaptive step direction (overwritten below).
    var update = List[Float32](capacity=n)
    for _ in range(n):
        update.append(Float32(0.0))

    # geometry of trailing two axes (only meaningful when factored)
    var last = shape[ndim - 1] if factored else 1     # columns
    var second = shape[ndim - 2] if factored else 1   # rows
    var outer = (n // (last * second)) if factored else 1

    if factored:
        var srf = sq_row.to_host(ctx)   # shape[:-1]
        var scf = sq_col.to_host(ctx)   # shape[:-2]+shape[-1:]
        var rows_n = n // last
        var cols_n = n // second
        if len(srf) != rows_n or len(scf) != cols_n:
            raise Error("came_step: factored sq_row/sq_col state numel mismatch")

        # sq_row EMA over dim=-1 (mean of upd across `last`)  (:125-127)
        for ridx in range(rows_n):
            var base = ridx * last
            var acc = Float64(0.0)
            for c in range(last):
                acc += Float64(upd[base + c])
            var mean_row = Float32(acc / Float64(last))
            srf[ridx] = beta2 * srf[ridx] + one_m_b2 * mean_row

        # sq_col EMA over dim=-2 (mean of upd across `second`)  (:128-130)
        for o in range(outer):
            for c in range(last):
                var acc = Float64(0.0)
                for r in range(second):
                    acc += Float64(upd[(o * second + r) * last + c])
                var mean_col = Float32(acc / Float64(second))
                scf[o * last + c] = beta2 * scf[o * last + c] + one_m_b2 * mean_col

        # _approx_sq_grad(sq_row, sq_col) ; update *= grad  (:74-81,:133-134)
        #   r_factor = rsqrt(row / row.mean(dim=-1,keepdim))[..,None]
        #   c_factor = rsqrt(col)[..,None,:]
        for o in range(outer):
            var racc = Float64(0.0)
            for r in range(second):
                racc += Float64(srf[o * second + r])
            var rmean = Float32(racc / Float64(second))
            for r in range(second):
                var r_factor = Float32(1.0) / sqrt(srf[o * second + r] / rmean)
                var pbase = (o * second + r) * last
                var cbase = o * last
                for c in range(last):
                    var c_factor = Float32(1.0) / sqrt(scf[cbase + c])
                    update[pbase + c] = r_factor * c_factor * gf[pbase + c]

        _store_bf16(sq_row, srf, stochastic_rounding, seed ^ UInt32(0x9E3779B1), ctx)
        _store_bf16(sq_col, scf, stochastic_rounding, seed ^ UInt32(0x85EBCA77), ctx)
    else:
        # non-factored: exp_avg_sq EMA then rsqrt*grad  (:136-139)
        var vf = exp_avg_sq.to_host(ctx)
        if len(vf) != n:
            raise Error("came_step: exp_avg_sq numel mismatch")
        for i in range(n):
            vf[i] = beta2 * vf[i] + one_m_b2 * upd[i]
            update[i] = (Float32(1.0) / sqrt(vf[i])) * gf[i]
        _store_bf16(exp_avg_sq, vf, stochastic_rounding, seed ^ UInt32(0xC2B2AE35), ctx)

    # update /= max(_rms(update)/clip_threshold, 1.0)  (:141-143)
    var rms_u = _rms_host(update)
    var clip = rms_u / clip_threshold
    var denom_clip = clip if clip > Float32(1.0) else Float32(1.0)
    var inv_clip = Float32(1.0) / denom_clip
    for i in range(n):
        update[i] = update[i] * inv_clip

    # exp_avg.mul_(b1).add_(update, 1-b1)  (:145-146)
    var ea = exp_avg.to_host(ctx)
    if len(ea) != n:
        raise Error("came_step: exp_avg numel mismatch")
    var one_m_b1 = Float32(1.0) - beta1
    for i in range(n):
        ea[i] = beta1 * ea[i] + one_m_b1 * update[i]

    # Confidence-guided strategy: instability res = (update - exp_avg)**2 + eps[1]
    # (:150). NOTE: `update` here is the post-clip adaptive step, `exp_avg` the
    # freshly updated first moment.
    var res = List[Float32](capacity=n)
    for i in range(n):
        var d = update[i] - ea[i]
        res.append(d * d + eps2)

    if factored:
        var rrf = res_row.to_host(ctx)
        var rcf = res_col.to_host(ctx)
        var rows_n = n // last
        var cols_n = n // second
        if len(rrf) != rows_n or len(rcf) != cols_n:
            raise Error("came_step: factored res_row/res_col state numel mismatch")

        var one_m_b3 = Float32(1.0) - beta3
        # res_row EMA over dim=-1  (:156-158)
        for ridx in range(rows_n):
            var base = ridx * last
            var acc = Float64(0.0)
            for c in range(last):
                acc += Float64(res[base + c])
            var mean_row = Float32(acc / Float64(last))
            rrf[ridx] = beta3 * rrf[ridx] + one_m_b3 * mean_row

        # res_col EMA over dim=-2  (:159-161)
        for o in range(outer):
            for c in range(last):
                var acc = Float64(0.0)
                for r in range(second):
                    acc += Float64(res[(o * second + r) * last + c])
                var mean_col = Float32(acc / Float64(second))
                rcf[o * last + c] = beta3 * rcf[o * last + c] + one_m_b3 * mean_col

        # res_approx = _approx_sq_grad(res_row, res_col) ; update = res_approx * exp_avg
        # (:164-165)
        for o in range(outer):
            var racc = Float64(0.0)
            for r in range(second):
                racc += Float64(rrf[o * second + r])
            var rmean = Float32(racc / Float64(second))
            for r in range(second):
                var r_factor = Float32(1.0) / sqrt(rrf[o * second + r] / rmean)
                var pbase = (o * second + r) * last
                var cbase = o * last
                for c in range(last):
                    var c_factor = Float32(1.0) / sqrt(rcf[cbase + c])
                    update[pbase + c] = r_factor * c_factor * ea[pbase + c]

        _store_bf16(res_row, rrf, stochastic_rounding, seed ^ UInt32(0x27D4EB2F), ctx)
        _store_bf16(res_col, rcf, stochastic_rounding, seed ^ UInt32(0x165667B1), ctx)
    else:
        # update = exp_avg.clone()  (:167)
        for i in range(n):
            update[i] = ea[i]

    # write first moment back (it was mutated in place by .mul_/.add_, :146)
    _store_bf16(exp_avg, ea, stochastic_rounding, seed ^ UInt32(0xD3A2646C), ctx)

    # Cautious masking (:169-172): mask = (update*grad > 0); mask /= clamp(mean, 1e-3);
    # update *= mask. mask.mean() is a scalar host reduction.
    if use_cautious:
        var mask = List[Float32](capacity=n)
        var msum = Float64(0.0)
        for i in range(n):
            var mv = Float32(1.0) if (update[i] * gf[i] > Float32(0.0)) else Float32(0.0)
            mask.append(mv)
            msum += Float64(mv)
        var mmean = Float32(msum / Float64(n))
        var mmean_c = mmean if mmean > Float32(1.0e-3) else Float32(1.0e-3)
        var inv_mmean = Float32(1.0) / mmean_c
        for i in range(n):
            update[i] = update[i] * (mask[i] * inv_mmean)

    # weight decay applied to p BEFORE the update (:174-181):
    #   p += -(weight_decay*lr) * p   (add_stochastic_(p, p, alpha=-wd*lr))
    # then update *= lr ; p += -update  (:183-187).
    var wd_alpha = -weight_decay * lr
    for i in range(n):
        var newp = pf[i]
        if weight_decay != Float32(0.0):
            newp = newp + wd_alpha * newp
        newp = newp - lr * update[i]
        pf[i] = newp

    _store_bf16(p, pf, stochastic_rounding, seed, ctx)


# Quantize an F32 host buffer to BF16 (optional stochastic rounding) and copy it
# back IN PLACE into `t`'s existing device buffer. Mirrors add_stochastic_(p, ...)
# (bf16_stochastic_rounding.py:45-57 → copy_stochastic_ :40) and the plain
# p.data.add_ write-back (CAME.py:178-181,:186-187). Preserves the tensor's
# buffer + autograd id (in-place contract), same as adafactor.mojo::_store_bf16.
def _store_bf16(
    t: Tensor,
    src: List[Float32],
    stochastic: Bool,
    seed: UInt32,
    ctx: DeviceContext,
) raises:
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
