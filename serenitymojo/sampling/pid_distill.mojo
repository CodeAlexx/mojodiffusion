# sampling/pid_distill.mojo — PiD 4-step distilled flow-match pixel sampler.
#
# Faithful port of pid/_src/models/pid_distill_model.py:
#   _velocity_to_x0  (prediction_type="velocity"):  x0 = x_t - t*v   (t in [0,1])
#   _student_sample_loop (student_sample_type="sde", t_list =
#       [0.999, 0.866, 0.634, 0.342, 0.0], fm_timescale=1000):
#     x = noise
#     for (t_cur, t_next) in zip(t_list[:-1], t_list[1:]):
#         v = net(x, t_cur*timescale, ...)           # velocity
#         if t_next > 0:
#             x0 = x - t_cur*v
#             eps = randn_like(x)
#             x  = (1-t_next)*x0 + t_next*eps         # SDE re-noise
#         else:
#             x  = x - t_cur*v                        # final x0
#     return clamp(x, -1, 1)
#
# The net forward itself is models/pid/pid_net.pid_net_forward; this module
# provides the scalar schedule + the velocity->x0 and SDE-renoise GPU helpers.
# B=1, pixel-space [B,3,H,W]. Mojo 1.0.0b1.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
def student_t_list() -> List[Float32]:
    """The distilled 4-step schedule (released SD3/Flux ckpts)."""
    var t = List[Float32]()
    t.append(Float32(0.999))
    t.append(Float32(0.866))
    t.append(Float32(0.634))
    t.append(Float32(0.342))
    t.append(Float32(0.0))
    return t^


def fm_timescale() -> Float32:
    return Float32(1000.0)


# ── x0 = x_t - t*v  (velocity prediction -> clean sample) ───────────────────
def _vel_to_x0_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    v: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int, t: Float32,
):
    var i = Int(global_idx.x)
    if i < n:
        # F32 is the sampler arithmetic workspace; storage remains `dtype`.
        var xv = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        var vv = rebind[Scalar[dtype]](v[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((xv - t * vv).cast[dtype]())


def velocity_to_x0(x: Tensor, v: Tensor, t: Float32, ctx: DeviceContext) raises -> Tensor:
    """Convert velocity to x0 as x - t*v, preserving x/v storage dtype."""
    if x.dtype() != v.dtype():
        raise Error("velocity_to_x0: x/v dtype mismatch")
    if x.numel() != v.numel():
        raise Error("velocity_to_x0: x/v numel mismatch")
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), rl)
        var V = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](v.buf.unsafe_ptr().bitcast[Float32](), rl)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[_vel_to_x0_kernel[DType.float32], _vel_to_x0_kernel[DType.float32]](
            X, V, O, n, t, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](v.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[_vel_to_x0_kernel[DType.bfloat16], _vel_to_x0_kernel[DType.bfloat16]](
            X, V, O, n, t, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.float16:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float16](), rl)
        var V = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](v.buf.unsafe_ptr().bitcast[Float16](), rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[_vel_to_x0_kernel[DType.float16], _vel_to_x0_kernel[DType.float16]](
            X, V, O, n, t, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        raise Error("velocity_to_x0: unsupported storage dtype")
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── SDE re-noise: x = (1-t_next)*x0 + t_next*eps ────────────────────────────
def _renoise_kernel[dtype: DType](
    x0: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    eps: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int, t_next: Float32,
):
    var i = Int(global_idx.x)
    if i < n:
        # F32 is the sampler arithmetic workspace; storage remains `dtype`.
        var x0v = rebind[Scalar[dtype]](x0[i]).cast[DType.float32]()
        var ev = rebind[Scalar[dtype]](eps[i]).cast[DType.float32]()
        var rv = (Float32(1.0) - t_next) * x0v + t_next * ev
        o[i] = rebind[o.element_type](rv.cast[dtype]())


def sde_renoise(x0: Tensor, eps: Tensor, t_next: Float32, ctx: DeviceContext) raises -> Tensor:
    if x0.dtype() != eps.dtype():
        raise Error("sde_renoise: x0/eps dtype mismatch")
    if x0.numel() != eps.numel():
        raise Error("sde_renoise: x0/eps numel mismatch")
    var n = x0.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x0.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = x0.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](x0.buf.unsafe_ptr().bitcast[Float32](), rl)
        var E = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](eps.buf.unsafe_ptr().bitcast[Float32](), rl)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[_renoise_kernel[DType.float32], _renoise_kernel[DType.float32]](
            X, E, O, n, t_next, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](x0.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var E = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](eps.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[_renoise_kernel[DType.bfloat16], _renoise_kernel[DType.bfloat16]](
            X, E, O, n, t_next, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.float16:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](x0.buf.unsafe_ptr().bitcast[Float16](), rl)
        var E = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](eps.buf.unsafe_ptr().bitcast[Float16](), rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[_renoise_kernel[DType.float16], _renoise_kernel[DType.float16]](
            X, E, O, n, t_next, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        raise Error("sde_renoise: unsupported storage dtype")
    ctx.synchronize()
    return Tensor(out_buf^, x0.shape(), x0.dtype())


# ── clamp to [-1,1] (final output) ──────────────────────────────────────────
def _clamp_kernel[dtype: DType](
    x: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        # F32 compare workspace; final clamped sample stores in `dtype`.
        var v = rebind[Scalar[dtype]](x[i]).cast[DType.float32]()
        var c = v
        if c > Float32(1.0):
            c = Float32(1.0)
        if c < Float32(-1.0):
            c = Float32(-1.0)
        o[i] = rebind[o.element_type](c.cast[dtype]())


def clamp_unit(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float32](), rl)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[_clamp_kernel[DType.float32], _clamp_kernel[DType.float32]](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[_clamp_kernel[DType.bfloat16], _clamp_kernel[DType.bfloat16]](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.float16:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[Float16](), rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[_clamp_kernel[DType.float16], _clamp_kernel[DType.float16]](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        raise Error("clamp_unit: unsupported storage dtype")
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())
