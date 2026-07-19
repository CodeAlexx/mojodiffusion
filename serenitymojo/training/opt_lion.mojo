# training/opt_lion.mojo — Lion optimizer (Chen et al., 2023), host-F32 master.
#
# NEW STANDALONE MODULE. Mirrors the AdamW/SGD struct+step style of
# serenitymojo/training/optim.mojo (read-only reference) and the EXACT formula
# of EriDiffusion-v2 training_features/optimizers.rs::Lion::step (~line 1083).
#
# Lion sign-of-momentum update. 1-tensor state per parameter, no eps:
#   c   = beta1 * m + (1 - beta1) * g          # interpolated direction
#   dir = sign(c)
#   m   = beta2 * m + (1 - beta2) * g          # EMA momentum (uses OLD m)
#   if weight_decay != 0:  p *= (1 - weight_decay * lr)   # DECOUPLED, BEFORE step
#   p   = p - lr * dir
#
# ORDERING NOTE (load-bearing, matches optimizers.rs:1124-1128): the decoupled
# weight-decay multiply `p *= (1 - wd*lr)` is applied to `p` BEFORE subtracting
# `lr*sign(c)`. Both `c` (direction) and `m` are computed from the OLD `m`.
#
# Defaults (Lion paper / OptimizerKind::default_betas Lion special-case):
#   beta1 = 0.9, beta2 = 0.99 (NOT 0.999). lr ~ adam_lr / 3..10. wd ~ adam_wd*3..10.
#   AGENT-DEFAULT: caller passes hyperparameters; this module hardcodes nothing.
#
# Mojo 0.26.x, NVIDIA GPU. F32 device buffers; LayoutTensor flat-index kernels.
# In-place via `mut` (Tensor is move-only). Mirrors optim.mojo conventions
# exactly: LayoutTensor views built INLINE (a `def` helper can't return one).

from std.math import copysign
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── Lion elementwise kernel (F32 param/grad/m) ───────────────────────────────
# Reads grad + old (m,p), writes new (m,p) in place. sign(0) := 0 (matches the
# IEEE/torch-sign convention torch.sign uses; Rust Tensor::sign mirrors it).
def _lion_kernel(
    p: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    m: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    weight_decay: Float32,
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var gv = rebind[Scalar[DType.float32]](g[idx])
        var mv = rebind[Scalar[DType.float32]](m[idx])
        var pv = rebind[Scalar[DType.float32]](p[idx])

        # interpolated update direction from OLD m
        var c = beta1 * mv + (1.0 - beta1) * gv
        var direction = Float32(0.0)
        if c > 0.0:
            direction = 1.0
        elif c < 0.0:
            direction = -1.0
        # c == 0 → direction 0 (torch.sign(0)=0)

        # EMA momentum from OLD m
        var m_new = beta2 * mv + (1.0 - beta2) * gv
        m[idx] = rebind[m.element_type](m_new)

        # decoupled WD BEFORE the sign step (matches optimizers.rs:1124-1128)
        if weight_decay != 0.0:
            pv = pv * (1.0 - weight_decay * lr)
        pv = pv - lr * direction
        p[idx] = rebind[p.element_type](pv)


def _require_f32(name: String, t: Tensor) raises:
    if t.dtype() != STDtype.F32:
        raise Error(
            String("opt_lion: ") + name + " must be F32 (got " + t.dtype().name() + ")"
        )


# ── Lion step (in place on param/m) ──────────────────────────────────────────
def lion_step(
    mut param: Tensor,
    grad: Tensor,
    mut m: Tensor,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    weight_decay: Float32,
    ctx: DeviceContext,
) raises:
    """One Lion step. `param`, `m` updated IN PLACE; `grad` read-only. `m` starts
    at zeros. All tensors must be F32, same numel.

    Caller idiom (no moves):
        lion_step(p, g, m, lr, b1, b2, wd, ctx)   # p, m mutated"""
    _require_f32("param", param)
    _require_f32("grad", grad)
    _require_f32("m", m)
    var n = param.numel()
    if grad.numel() != n or m.numel() != n:
        raise Error("lion_step: param/grad/m numel mismatch")

    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var pt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        param.buf.unsafe_ptr().bitcast[Float32](), rl)
    var gt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        grad.buf.unsafe_ptr().bitcast[Float32](), rl)
    var mt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        m.buf.unsafe_ptr().bitcast[Float32](), rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_lion_kernel, _lion_kernel](
        pt, gt, mt, lr, beta1, beta2, weight_decay, n,
        grid_dim=grid, block_dim=_BLOCK,
    )
    ctx.synchronize()
