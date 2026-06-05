# ops/loss_swiglu_backward.mojo — BACKWARD for MSELoss, HuberLoss, FusedSwiGLU.
#
# Phase T (FULL_PORT_TRAINING_PLAN.md): the training-loss + MLP-activation
# backward kernels. MSE is THE diffusion training loss (flow-matching v-MSE);
# SwiGLU is the activation in every DiT MLP. These three backward passes are on
# the critical path for full fine-tune.
#
# ── Math ─────────────────────────────────────────────────────────────────────
# MSE (mean reduction, matches torch.nn.functional.mse_loss(reduction='mean')):
#     loss   = mean((pred - target)^2)            over N = numel elements
#     d_pred = 2 * (pred - target) / N
#
# Huber / smooth-L1 (mean reduction, matches torch huber_loss(delta=δ)):
#     x      = pred - target
#     loss_i = 0.5*x^2             if |x| <= δ
#            = δ*(|x| - 0.5*δ)     otherwise
#     d_pred = clamp(x, -δ, δ) / N
#       (because dloss_i/dx = x for |x|<=δ, else δ*sign(x); clamp captures both)
#
# SwiGLU (ported verbatim from flame-core kernels/swiglu_backward.cu):
#     y      = silu(gate) * up,   silu(g) = g * sigmoid(g)
#     d_up   = grad_out * silu(gate)
#     d_gate = grad_out * up * silu'(gate)
#     silu'(g) = sig + g*sig*(1 - sig)         (sig = sigmoid(g))
#              = sig * (1 + g*(1 - sig))        (the spec's spelling; identical)
#
# All interior math is F32. BF16/F16 public storage paths keep full tensors in
# their storage dtype; kernels cast scalar elements to F32 and write gradients
# back to the input dtype. Only scalar reduction values remain F32.
#
# Mojo 1.0.0b1, NVIDIA GPU. Mirrors activations.mojo / attention_backward.mojo
# gather/scatter + per-element kernel scaffolding.

from std.math import exp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


@always_inline
def _sigmoid_f32(v: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-v))


# ── MSE backward: d_pred[i] = 2 * (pred[i] - target[i]) / N ───────────────────
def _mse_bwd_kernel[dtype: DType](
    pred: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    tgt: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    inv_n2: Float32,  # = 2.0 / N
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var p = rebind[Scalar[dtype]](pred[i]).cast[DType.float32]()
        var t = rebind[Scalar[dtype]](tgt[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](((p - t) * inv_n2).cast[dtype]())


def mse_backward(pred: Tensor, target: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Gradient of mean((pred-target)^2) wrt pred. d_pred = 2*(pred-target)/N.

    pred/target: same shape and dtype. Returns d_pred in pred storage dtype.
    Matches torch.nn.functional.mse_loss(reduction='mean') autograd.
    """
    if pred.dtype() != target.dtype():
        raise Error("mse_backward: pred/target dtype mismatch")
    if pred.numel() != target.numel():
        raise Error("mse_backward: pred/target numel mismatch")
    var n = pred.numel()
    if n == 0:
        raise Error("mse_backward: empty input")
    var inv_n2 = Float32(2.0) / Float32(n)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](pred.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = pred.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[Float32](), rl)
        var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[Float32](), rl)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[
            _mse_bwd_kernel[DType.float32], _mse_bwd_kernel[DType.float32]
        ](P, T, O, inv_n2, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var T = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[
            _mse_bwd_kernel[DType.bfloat16], _mse_bwd_kernel[DType.bfloat16]
        ](P, T, O, inv_n2, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var P = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[Float16](), rl)
        var T = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[Float16](), rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[
            _mse_bwd_kernel[DType.float16], _mse_bwd_kernel[DType.float16]
        ](P, T, O, inv_n2, n, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, pred.shape(), pred.dtype())


# ── MAE / L1 backward: d_pred[i] = sign(pred[i]-target[i]) / N ────────────────
# (Wave 2A Domain-3 l1-mae-loss-backward item.)
# MAE (mean reduction, matches torch.nn.functional.l1_loss(reduction='mean')):
#     loss   = mean(|pred - target|)
#     d_pred = sign(pred - target) / N      (sign(0) = 0, matches torch subgrad)
def _mae_bwd_kernel[dtype: DType](
    pred: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    tgt: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    inv_n: Float32,  # = 1.0 / N
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var p = rebind[Scalar[dtype]](pred[i]).cast[DType.float32]()
        var t = rebind[Scalar[dtype]](tgt[i]).cast[DType.float32]()
        var x = p - t
        var s = Float32(0.0)
        if x > Float32(0.0):
            s = Float32(1.0)
        elif x < Float32(0.0):
            s = Float32(-1.0)
        o[i] = rebind[o.element_type]((s * inv_n).cast[dtype]())


def mae_backward(pred: Tensor, target: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Gradient of mean(|pred-target|) wrt pred. d_pred = sign(pred-target)/N.

    pred/target: same shape and dtype. Returns d_pred in pred storage dtype.
    Matches torch.nn.functional.l1_loss(reduction='mean') autograd (sign(0)=0).
    """
    if pred.dtype() != target.dtype():
        raise Error("mae_backward: pred/target dtype mismatch")
    if pred.numel() != target.numel():
        raise Error("mae_backward: pred/target numel mismatch")
    var n = pred.numel()
    if n == 0:
        raise Error("mae_backward: empty input")
    var inv_n = Float32(1.0) / Float32(n)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](pred.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = pred.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[Float32](), rl)
        var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[Float32](), rl)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[
            _mae_bwd_kernel[DType.float32], _mae_bwd_kernel[DType.float32]
        ](P, T, O, inv_n, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var T = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[
            _mae_bwd_kernel[DType.bfloat16], _mae_bwd_kernel[DType.bfloat16]
        ](P, T, O, inv_n, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var P = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[Float16](), rl)
        var T = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[Float16](), rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[
            _mae_bwd_kernel[DType.float16], _mae_bwd_kernel[DType.float16]
        ](P, T, O, inv_n, n, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, pred.shape(), pred.dtype())


# ── Huber backward: d_pred[i] = clamp(pred[i]-target[i], -δ, δ) / N ───────────
def _huber_bwd_kernel[dtype: DType](
    pred: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    tgt: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    delta: Float32,
    inv_n: Float32,  # = 1.0 / N
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var p = rebind[Scalar[dtype]](pred[i]).cast[DType.float32]()
        var t = rebind[Scalar[dtype]](tgt[i]).cast[DType.float32]()
        var x = p - t
        var c = x
        if c > delta:
            c = delta
        elif c < -delta:
            c = -delta
        o[i] = rebind[o.element_type]((c * inv_n).cast[dtype]())


def huber_backward(
    pred: Tensor, target: Tensor, delta: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Gradient of huber/smooth-L1 (mean) wrt pred. d_pred = clamp(x,-δ,δ)/N,
    x = pred - target. Matches torch.nn.functional.huber_loss(delta=δ) autograd.

    pred/target: same shape and dtype. delta > 0. Returns d_pred in pred dtype.
    """
    if pred.dtype() != target.dtype():
        raise Error("huber_backward: pred/target dtype mismatch")
    if pred.numel() != target.numel():
        raise Error("huber_backward: pred/target numel mismatch")
    if delta <= 0.0:
        raise Error("huber_backward: delta must be > 0")
    var n = pred.numel()
    if n == 0:
        raise Error("huber_backward: empty input")
    var inv_n = Float32(1.0) / Float32(n)
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](pred.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = pred.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[Float32](), rl)
        var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[Float32](), rl)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[
            _huber_bwd_kernel[DType.float32], _huber_bwd_kernel[DType.float32]
        ](P, T, O, delta, inv_n, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var T = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[
            _huber_bwd_kernel[DType.bfloat16], _huber_bwd_kernel[DType.bfloat16]
        ](P, T, O, delta, inv_n, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var P = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[Float16](), rl)
        var T = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[Float16](), rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[
            _huber_bwd_kernel[DType.float16], _huber_bwd_kernel[DType.float16]
        ](P, T, O, delta, inv_n, n, grid_dim=grid, block_dim=_BLOCK)
    return Tensor(out_buf^, pred.shape(), pred.dtype())


# ── SwiGLU backward ──────────────────────────────────────────────────────────
struct SwigluGrads(Movable):
    """Backward outputs of `swiglu_backward`: gradients wrt gate and up."""

    var d_gate: Tensor
    var d_up: Tensor

    def __init__(out self, var d_gate: Tensor, var d_up: Tensor):
        self.d_gate = d_gate^
        self.d_up = d_up^


def _swiglu_bwd_kernel[dtype: DType](
    grad_out: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    gate: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    up: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    d_gate: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    d_up: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var g = rebind[Scalar[dtype]](grad_out[i]).cast[DType.float32]()
        var x = rebind[Scalar[dtype]](gate[i]).cast[DType.float32]()
        var u = rebind[Scalar[dtype]](up[i]).cast[DType.float32]()
        var sig = _sigmoid_f32(x)
        var silu_x = x * sig
        var dsilu = sig + x * sig * (1.0 - sig)
        d_up[i] = rebind[d_up.element_type]((g * silu_x).cast[dtype]())
        d_gate[i] = rebind[d_gate.element_type]((g * dsilu * u).cast[dtype]())


def swiglu_backward(
    grad_out: Tensor, gate: Tensor, up: Tensor, ctx: DeviceContext
) raises -> SwigluGrads:
    """Backward of y = silu(gate) * up.
        d_up   = grad_out * silu(gate)
        d_gate = grad_out * up * silu'(gate),  silu'(g) = sig + g*sig*(1-sig)
    Ports flame-core kernels/swiglu_backward.cu verbatim.

    grad_out/gate/up: same shape and dtype. Returns gradients in storage dtype.
    """
    if grad_out.dtype() != gate.dtype() or gate.dtype() != up.dtype():
        raise Error("swiglu_backward: grad_out/gate/up dtype mismatch")
    if grad_out.numel() != gate.numel() or gate.numel() != up.numel():
        raise Error("swiglu_backward: grad_out/gate/up numel mismatch")
    var n = gate.numel()
    if n == 0:
        raise Error("swiglu_backward: empty input")
    var dg_buf = ctx.enqueue_create_buffer[DType.uint8](gate.nbytes())
    var du_buf = ctx.enqueue_create_buffer[DType.uint8](up.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = gate.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var GO = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), rl)
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            gate.buf.unsafe_ptr().bitcast[Float32](), rl)
        var U = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            up.buf.unsafe_ptr().bitcast[Float32](), rl)
        var DG = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            dg_buf.unsafe_ptr().bitcast[Float32](), rl)
        var DU = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            du_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[
            _swiglu_bwd_kernel[DType.float32], _swiglu_bwd_kernel[DType.float32]
        ](GO, G, U, DG, DU, n, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var GO = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            gate.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var U = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            up.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var DG = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            dg_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var DU = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            du_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[
            _swiglu_bwd_kernel[DType.bfloat16], _swiglu_bwd_kernel[DType.bfloat16]
        ](GO, G, U, DG, DU, n, grid_dim=grid, block_dim=_BLOCK)
    else:
        var GO = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), rl)
        var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            gate.buf.unsafe_ptr().bitcast[Float16](), rl)
        var U = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            up.buf.unsafe_ptr().bitcast[Float16](), rl)
        var DG = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            dg_buf.unsafe_ptr().bitcast[Float16](), rl)
        var DU = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            du_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[
            _swiglu_bwd_kernel[DType.float16], _swiglu_bwd_kernel[DType.float16]
        ](GO, G, U, DG, DU, n, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    var dg_t = Tensor(dg_buf^, gate.shape(), gate.dtype())
    var du_t = Tensor(du_buf^, up.shape(), up.dtype())
    return SwigluGrads(dg_t^, du_t^)
