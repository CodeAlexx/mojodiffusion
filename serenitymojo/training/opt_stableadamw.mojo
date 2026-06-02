# training/opt_stableadamw.mojo — StableAdamW (Wortsman et al. 2023), host-F32.
#
# NEW STANDALONE MODULE. Mirrors optim.mojo's AdamW struct+step style (read-only
# reference) and the EXACT formula of EriDiffusion-v2 optimizers.rs::StableAdamW
# ::step (~line 1211) + debias_beta (~line 1342). pytorch_optimizer.StableAdamW.
#
# Per-step (weight_decouple=True, no Kahan — params are F32 so Kahan degenerates):
#   beta1_hat = debias_beta(beta1, t) ; beta2_hat = debias_beta(beta2, t)
#   beta1_comp = 1 - beta1_hat        ; (debias_beta = (b^t - b)/(b^t - 1))
#   eps_p2 = eps*eps
#   m = m*(1-beta1_comp) + g*beta1_comp                # lerp_(grad, beta1_comp)
#   v = v*beta2_hat + g*g*(1-beta2_hat)
#   rms = sqrt( mean_i( g_i^2 / max(v_i, eps_p2) ) )  clipped to >= 1.0
#   lr_eff = lr / rms                                  # PER-TENSOR scalar
#   if wd != 0:  p *= (1 - wd*lr_eff)                  # decoupled, BEFORE step
#   p -= lr_eff * m / (sqrt(v) + eps)
#
# debias_beta computed in F64 (catastrophic-cancellation guard, matches Rust).
# At t=1: beta1_comp=1, beta2_hat=0 → first step uses raw grad / grad².
#
# rms requires a per-tensor reduction (mean over ALL elements). Computed via
# host readback after the m/v update (parity-grade, NOT the hot path — same
# idiom optim.mojo's grad-clip uses), then lr_eff is passed to the param-update
# kernel as a precomputed scalar. m/v live on device throughout.
#
# AGENT-DEFAULT: caller passes all hyperparameters; this module hardcodes none.
# Mojo 0.26.x, NVIDIA GPU. F32 device buffers; LayoutTensor flat-index kernels.

from std.math import sqrt
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── moment update: m = m*(1-b1c) + g*b1c ; v = v*b2h + g*g*(1-b2h) ────────────
def _moment_kernel(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    m: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    beta1_comp: Float32,
    beta2_hat: Float32,
    one_minus_beta2_hat: Float32,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var gv = rebind[Scalar[DType.float32]](g[idx])
        var mv = rebind[Scalar[DType.float32]](m[idx])
        var vv = rebind[Scalar[DType.float32]](v[idx])
        var m_new = mv * (1.0 - beta1_comp) + gv * beta1_comp
        var v_new = vv * beta2_hat + gv * gv * one_minus_beta2_hat
        m[idx] = rebind[m.element_type](m_new)
        v[idx] = rebind[v.element_type](v_new)


# ── param update with precomputed per-tensor lr_eff ──────────────────────────
def _update_kernel(
    p: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    m: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    lr_eff: Float32,
    eps: Float32,
    weight_decay: Float32,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var pv = rebind[Scalar[DType.float32]](p[idx])
        var mv = rebind[Scalar[DType.float32]](m[idx])
        var vv = rebind[Scalar[DType.float32]](v[idx])
        if weight_decay != 0.0:
            pv = pv * (1.0 - weight_decay * lr_eff)
        var denom = sqrt(vv) + eps
        pv = pv - lr_eff * mv / denom
        p[idx] = rebind[p.element_type](pv)


def _require_f32(name: String, t: Tensor) raises:
    if t.dtype() != STDtype.F32:
        raise Error(
            String("opt_stableadamw: ") + name + " must be F32 (got "
            + t.dtype().name() + ")"
        )


# ── debias_beta in F64: (b^t - b) / (b^t - 1) ────────────────────────────────
def _debias_beta(beta: Float32, t: Int) -> Float64:
    var b = Float64(beta)
    var bn = Float64(1.0)
    for _ in range(t):
        bn *= b
    var num = bn - b
    var den = bn - 1.0
    if den < 0.0:
        if -den < 1.0e-30:
            return Float64(beta)
    else:
        if den < 1.0e-30:
            return Float64(beta)
    return num / den


# ── StableAdamW step (in place on param/m/v) ─────────────────────────────────
def stableadamw_step(
    mut param: Tensor,
    grad: Tensor,
    mut m: Tensor,
    mut v: Tensor,
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    ctx: DeviceContext,
) raises:
    """One StableAdamW step. `param`, `m`, `v` updated IN PLACE; `grad` read-only.
    `t` is the 1-based per-param step counter. `m`, `v` start at zeros. All F32,
    same numel.

    Caller idiom (no moves):
        stableadamw_step(p, g, m, v, t, lr, b1, b2, eps, wd, ctx)"""
    _require_f32("param", param)
    _require_f32("grad", grad)
    _require_f32("m", m)
    _require_f32("v", v)
    var n = param.numel()
    if grad.numel() != n or m.numel() != n or v.numel() != n:
        raise Error("stableadamw_step: param/grad/m/v numel mismatch")
    if t < 1:
        raise Error("stableadamw_step: t must be >= 1 (1-based)")

    var beta1_hat = _debias_beta(beta1, t)
    var beta2_hat64 = _debias_beta(beta2, t)
    var beta1_comp = Float32(1.0 - beta1_hat)
    var beta2_hat = Float32(beta2_hat64)
    var one_minus_beta2_hat = Float32(1.0 - beta2_hat64)
    var eps_p2 = eps * eps

    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK

    # 1) moment update (m, v) on device.
    var gt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad.buf.unsafe_ptr().bitcast[Float32](), rl)
    var mt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        m.buf.unsafe_ptr().bitcast[Float32](), rl)
    var vt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        v.buf.unsafe_ptr().bitcast[Float32](), rl)
    ctx.enqueue_function[_moment_kernel, _moment_kernel](
        gt, mt, vt, beta1_comp, beta2_hat, one_minus_beta2_hat, n,
        grid_dim=grid, block_dim=_BLOCK,
    )
    ctx.synchronize()

    # 2) rms = sqrt(mean_i( g_i^2 / max(v_i, eps_p2) )).clip_min(1.0)
    #    Host readback of grad + updated v (parity-grade, mirrors grad-clip path).
    var gh = grad.to_host(ctx)
    var vh = v.to_host(ctx)
    var acc = Float32(0.0)
    for i in range(n):
        var vi = vh[i]
        if vi < eps_p2:
            vi = eps_p2
        acc += gh[i] * gh[i] / vi
    var rms_inner = acc / Float32(n)
    if rms_inner < 0.0:
        rms_inner = 0.0
    var rms = sqrt(rms_inner)
    if rms < 1.0:
        rms = 1.0
    var lr_eff = lr / rms

    # 3) param update with the per-tensor lr_eff.
    var pt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        param.buf.unsafe_ptr().bitcast[Float32](), rl)
    ctx.enqueue_function[_update_kernel, _update_kernel](
        pt, mt, vt, lr_eff, eps, weight_decay, n,
        grid_dim=grid, block_dim=_BLOCK,
    )
    ctx.synchronize()
