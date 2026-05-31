# training/optim.mojo — optimizers (AdamW, SGD) + global-norm grad clip.
#
# Phase T4 of FULL_PORT_TRAINING_PLAN.md. F32 throughout (master-weight path).
# Parity-gated: one optimizer step on a fixed (param, grad, moment) tuple must
# match the PyTorch reference at cos >= 0.999 (training/parity/optim_*).
#
# ── AdamW decoupled weight decay (Loshchilov & Hutter, 2017) ─────────────────
# Ported to match flame-core/src/adam.rs EXACTLY (and torch.optim.AdamW, NOT
# Adam+L2). The per-element update is:
#     m    = beta1*m + (1-beta1)*g
#     v    = beta2*v + (1-beta2)*g*g
#     mhat = m / (1 - beta1^t)
#     vhat = v / (1 - beta2^t)
#     p    = p - lr * mhat / (sqrt(vhat) + eps)
#     if weight_decay > 0:  p -= lr * weight_decay * p     # DECOUPLED, on p
# The weight-decay term is applied to `p` DIRECTLY after the Adam step. It must
# NOT be folded into `g` before the moments — that is the Adam+L2 form, which
# contaminates the adaptive rate and destroyed klein LoRA_A training (see the
# receipt at the top of flame-core/src/adam.rs). The parity oracle exercises
# weight_decay > 0 so this distinction is tested.
#
# ── SGD ──────────────────────────────────────────────────────────────────────
# PyTorch SGD-with-momentum (no Nesterov, no dampening):
#     buf = momentum*buf + g          (buf starts at 0 → buf = g on step 1,
#                                       matching torch's buffer init)
#     p   = p - lr * buf
#     if weight_decay > 0:  p -= lr * weight_decay * p     # DECOUPLED, on p
# (The gated SGD case uses weight_decay = 0; torch's SGD weight_decay is COUPLED
# L2 into the grad, so the decoupled form here only matches torch at wd = 0.)
#
# ── API: in-place via `mut` (Tensor is move-only) ────────────────────────────
# Tensor is NOT Copyable and Mojo 1.0.0b1 forbids moving an individual field out
# of a consumed struct, so the optimizers take their persistent tensors by `mut`
# and update the existing device buffers IN PLACE (the move-only-friendly
# analogue of flame-core's `&mut Tensor`). The caller keeps `var p`, `var m`,
# `var v` (or `var buf`) and re-passes them every step — no per-step moves, no
# result-struct unpacking. `grad` is read-only.
#
# ── Grad clip ────────────────────────────────────────────────────────────────
# `clip_grad_global_norm(g1, g2, max_norm)` computes total_norm = sqrt(sum of
# per-tensor sum-of-squares) over the two grads and, if total_norm > max_norm,
# scales both IN PLACE by max_norm/total_norm; returns total_norm. Mirrors
# flame-core gradient_clip.rs::clip_grads_by_norm + torch clip_grad_norm_
# (2-norm). The 2-tensor case is the gated one (the deliverable allows a single-
# or two-tensor case); the global L2 norm over N tensors is the obvious
# generalization (sum the per-tensor sum-of-squares, sqrt at the end).
#
# Mojo 1.0.0b1, NVIDIA GPU. F32 device buffers; LayoutTensor flat-index kernels.
# LayoutTensor views are built INLINE at each call site (a `def` helper cannot
# return a LayoutTensor — origin inference — so this follows linalg_backward).

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


# ── AdamW elementwise kernel (F32 param/grad/m/v) ────────────────────────────
# Reads grad + old (m,v,p), writes new (m,v,p) in place. Bias correction
# (bc1 = 1 - beta1^t, bc2 = 1 - beta2^t) is passed pre-computed by the host so
# the kernel needs no `pow`. DECOUPLED WD applied to p AFTER the Adam step.
def _adamw_kernel(
    p: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    m: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    bc1: Float32,
    bc2: Float32,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var gv = rebind[Scalar[DType.float32]](g[idx])
        var pv = rebind[Scalar[DType.float32]](p[idx])
        var mi = beta1 * rebind[Scalar[DType.float32]](m[idx]) + (1.0 - beta1) * gv
        var vi = beta2 * rebind[Scalar[DType.float32]](v[idx]) + (1.0 - beta2) * gv * gv
        m[idx] = rebind[m.element_type](mi)
        v[idx] = rebind[v.element_type](vi)
        var m_hat = mi / bc1
        var v_hat = vi / bc2
        pv = pv - lr * m_hat / (sqrt(v_hat) + eps)
        if weight_decay > 0.0:
            pv = pv - lr * weight_decay * pv
        p[idx] = rebind[p.element_type](pv)


# ── SGD elementwise kernel (momentum buffer + optional decoupled WD) ─────────
def _sgd_kernel(
    p: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    buf: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    lr: Float32,
    momentum: Float32,
    weight_decay: Float32,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var gv = rebind[Scalar[DType.float32]](g[idx])
        var pv = rebind[Scalar[DType.float32]](p[idx])
        var b = momentum * rebind[Scalar[DType.float32]](buf[idx]) + gv
        buf[idx] = rebind[buf.element_type](b)
        pv = pv - lr * b
        if weight_decay > 0.0:
            pv = pv - lr * weight_decay * pv
        p[idx] = rebind[p.element_type](pv)


# ── scale a flat F32 buffer in place ─────────────────────────────────────────
def _scale_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], scale: Float32, n: Int
):
    var idx = Int(global_idx.x)
    if idx < n:
        x[idx] = rebind[x.element_type](
            rebind[Scalar[DType.float32]](x[idx]) * scale
        )


# ── F32 dtype guard ──────────────────────────────────────────────────────────
def _require_f32(name: String, t: Tensor) raises:
    if t.dtype() != STDtype.F32:
        raise Error(
            String("optim: ") + name + " must be F32 (got " + t.dtype().name() + ")"
        )


# ── AdamW step (in place on param/m/v) ───────────────────────────────────────
def adamw_step(
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
    """One AdamW step (decoupled WD). `param`, `m`, `v` are updated IN PLACE;
    `grad` is read-only. `t` is the 1-based step counter (bias correction uses
    t). All tensors must be F32, same numel.

    Caller idiom (no moves):
        adamw_step(p, g, m, v, t, lr, b1, b2, eps, wd, ctx)  # p,m,v mutated"""
    _require_f32("param", param)
    _require_f32("grad", grad)
    _require_f32("m", m)
    _require_f32("v", v)
    var n = param.numel()
    if grad.numel() != n or m.numel() != n or v.numel() != n:
        raise Error("adamw_step: param/grad/m/v numel mismatch")
    if t < 1:
        raise Error("adamw_step: t must be >= 1 (1-based step counter)")

    # Bias correction (host-side integer power, matches flame-core: 1 - beta^t).
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p

    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var pt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        param.buf.unsafe_ptr().bitcast[Float32](), rl)
    var gt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad.buf.unsafe_ptr().bitcast[Float32](), rl)
    var mt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        m.buf.unsafe_ptr().bitcast[Float32](), rl)
    var vt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        v.buf.unsafe_ptr().bitcast[Float32](), rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_adamw_kernel, _adamw_kernel](
        pt, gt, mt, vt, lr, beta1, beta2, eps, weight_decay, bc1, bc2, n,
        grid_dim=grid, block_dim=_BLOCK,
    )
    ctx.synchronize()


# ── SGD step (in place on param/momentum_buf) ────────────────────────────────
def sgd_step(
    mut param: Tensor,
    grad: Tensor,
    mut momentum_buf: Tensor,
    lr: Float32,
    momentum: Float32,
    weight_decay: Float32,
    ctx: DeviceContext,
) raises:
    """One SGD-with-momentum step (decoupled WD). buf = momentum*buf + g; then
    p = p - lr*buf; then optional p -= lr*wd*p. `momentum_buf` starts at zeros
    (the first step yields buf = g, matching torch's buffer init). `param` and
    `momentum_buf` are updated IN PLACE; `grad` is read-only."""
    _require_f32("param", param)
    _require_f32("grad", grad)
    _require_f32("momentum_buf", momentum_buf)
    var n = param.numel()
    if grad.numel() != n or momentum_buf.numel() != n:
        raise Error("sgd_step: param/grad/momentum numel mismatch")

    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var pt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        param.buf.unsafe_ptr().bitcast[Float32](), rl)
    var gt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad.buf.unsafe_ptr().bitcast[Float32](), rl)
    var bt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        momentum_buf.buf.unsafe_ptr().bitcast[Float32](), rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_sgd_kernel, _sgd_kernel](
        pt, gt, bt, lr, momentum, weight_decay, n,
        grid_dim=grid, block_dim=_BLOCK,
    )
    ctx.synchronize()


# ── global-norm gradient clipping (2-tensor case, in place) ──────────────────
def clip_grad_global_norm(
    mut g1: Tensor,
    mut g2: Tensor,
    max_norm: Float32,
    ctx: DeviceContext,
) raises -> Float32:
    """Scale g1,g2 IN PLACE by min(1, max_norm/total_norm); returns total_norm
    (the pre-clip global L2 norm).

    total_norm = sqrt(sum over BOTH tensors of sum(g^2)) — matches flame-core
    compute_grad_norm (per-tensor L2 summed in quadrature) and torch
    clip_grad_norm_. Both grads must be F32."""
    _require_f32("g1", g1)
    _require_f32("g2", g2)

    # Sum of squares over both grads (host readback — parity-grade, not hot).
    var total_sq = Float64(0.0)
    var h1 = g1.to_host(ctx)
    for j in range(len(h1)):
        var x = Float64(h1[j])
        total_sq += x * x
    var h2 = g2.to_host(ctx)
    for j in range(len(h2)):
        var x = Float64(h2[j])
        total_sq += x * x
    var total_norm = Float32(sqrt(total_sq))

    var scale = Float32(1.0)
    if total_norm > max_norm and total_norm > 0.0:
        scale = max_norm / total_norm

    if scale != 1.0:
        var n1 = g1.numel()
        var rl1 = RuntimeLayout[_DYN1].row_major(IndexList[1](n1))
        var gt1 = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            g1.buf.unsafe_ptr().bitcast[Float32](), rl1)
        var grid1 = (n1 + _BLOCK - 1) // _BLOCK
        ctx.enqueue_function[_scale_kernel, _scale_kernel](
            gt1, scale, n1, grid_dim=grid1, block_dim=_BLOCK)
        var n2 = g2.numel()
        var rl2 = RuntimeLayout[_DYN1].row_major(IndexList[1](n2))
        var gt2 = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            g2.buf.unsafe_ptr().bitcast[Float32](), rl2)
        var grid2 = (n2 + _BLOCK - 1) // _BLOCK
        ctx.enqueue_function[_scale_kernel, _scale_kernel](
            gt2, scale, n2, grid_dim=grid2, block_dim=_BLOCK)
        ctx.synchronize()

    return total_norm
