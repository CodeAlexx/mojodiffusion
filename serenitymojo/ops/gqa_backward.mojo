# ops/gqa_backward.mojo — BACKWARD for GQA repeat_kv (grouped-sum).
#
# Forward partner: the repeat_kv used by acestep_dit.mojo::_repeat_kv (and the
# Rust repeat_kv, acestep.rs:1151). Forward (BSHD [1,S,Hkv,Dh] -> [1,S,H,Dh]):
#   dst head `head` reads src kv-head `head // n_rep`  (PyTorch repeat_kv order;
#   src kv-head g is broadcast across the n_rep contiguous dst heads
#   [g*n_rep .. g*n_rep+n_rep-1]).
# Backward (given d_dst [1,S,H,Dh]):
#   d_src[1,S,Hkv,Dh]:  d_src[t,g,d] = sum_{r in [0,n_rep)} d_dst[t, g*n_rep+r, d]
# i.e. SUM the n_rep repeated copies back onto each source kv head. This is the
# grad-routing inverse of the head-broadcast. BF16/F16 inputs are read as their
# storage dtype, summed in F32 scalars where needed, and stored back to the same
# dtype.
#
# Mojo 1.0.0b1, NVIDIA GPU. `def` not `fn`.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── F32 forward (mirror of acestep_dit._repeat_kv, F32 storage for training) ──
# dst head `head` reads src kv-head `head // n_rep` (one thread per DST element).
def _repeat_kv_fwd_kernel[dtype: DType](
    src: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [S*Hkv*Dh]
    dst: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [S*H*Dh]
    seq: Int, h: Int, h_kv: Int, dh: Int, n_rep: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * h * dh
    if idx < total:
        var d_i = idx % dh
        var rest = idx // dh
        var head = rest % h
        var t = rest // h
        var kvh = head // n_rep
        var src_idx = (t * h_kv + kvh) * dh + d_i
        dst[idx] = rebind[dst.element_type](src[src_idx])


def repeat_kv_f32(
    x: Tensor, s: Int, h_kv: Int, n_rep: Int, dh: Int, ctx: DeviceContext
) raises -> Tensor:
    """GQA repeat_kv forward. x [1,S,h_kv,Dh] -> [1,S,h_kv*n_rep,Dh]."""
    if n_rep == 1:
        var dev0 = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev0, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev0^, x.shape(), x.dtype())
    var h = h_kv * n_rep
    var out_n = s * h * dh
    var src_n = s * h_kv * dh
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * x.dtype().byte_size()
    )
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](src_n))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var dt = x.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var SRC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        var DST = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[
            _repeat_kv_fwd_kernel[DType.float32],
            _repeat_kv_fwd_kernel[DType.float32],
        ](SRC, DST, s, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var SRC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        var DST = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[
            _repeat_kv_fwd_kernel[DType.bfloat16],
            _repeat_kv_fwd_kernel[DType.bfloat16],
        ](SRC, DST, s, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK)
    else:
        var SRC = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        var DST = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[
            _repeat_kv_fwd_kernel[DType.float16],
            _repeat_kv_fwd_kernel[DType.float16],
        ](SRC, DST, s, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK)
    # sync removed (single-stream ordering; was kernel-trailing host stall)
    var sh = [1, s, h, dh]
    return Tensor(out_buf^, sh^, x.dtype())


# One thread per SOURCE element [t, g, d]; sum the n_rep dst copies.
def _repeat_kv_bwd_kernel[dtype: DType](
    d_dst: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [S*H*Dh]
    d_src: LayoutTensor[dtype, _DYN1, MutAnyOrigin],   # [S*Hkv*Dh]
    seq: Int, h_kv: Int, dh: Int, n_rep: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * h_kv * dh
    if idx < total:
        var d_i = idx % dh
        var rest = idx // dh
        var g = rest % h_kv
        var t = rest // h_kv
        var h = h_kv * n_rep
        var acc: Float32 = 0.0
        for r in range(n_rep):
            var head = g * n_rep + r
            var src_idx = (t * h + head) * dh + d_i
            acc += rebind[Scalar[dtype]](d_dst[src_idx]).cast[DType.float32]()
        d_src[idx] = rebind[d_src.element_type](acc.cast[dtype]())


def repeat_kv_backward(
    d_dst: Tensor, s: Int, h_kv: Int, n_rep: Int, dh: Int, ctx: DeviceContext
) raises -> Tensor:
    """Backward of GQA repeat_kv. d_dst [1,S,H,Dh] (H = h_kv*n_rep).
    Returns d_src [1,S,h_kv,Dh] = grouped sum over the n_rep repeated heads."""
    if n_rep == 1:
        var dev0 = ctx.enqueue_create_buffer[DType.uint8](d_dst.nbytes())
        ctx.enqueue_copy(dst_buf=dev0, src_buf=d_dst.buf)
        ctx.synchronize()
        return Tensor(dev0^, d_dst.shape(), d_dst.dtype())
    var src_n = s * h_kv * dh
    var dst_n = s * h_kv * n_rep * dh
    var dsh = d_dst.shape()
    if d_dst.numel() != dst_n:
        raise Error("repeat_kv_backward: d_dst numel mismatch")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        src_n * d_dst.dtype().byte_size()
    )
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](src_n))
    var dst_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](dst_n))
    var grid = (src_n + _BLOCK - 1) // _BLOCK
    var dt = d_dst.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var DDST = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            d_dst.buf.unsafe_ptr().bitcast[Float32](), dst_rl
        )
        var DSRC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        ctx.enqueue_function[
            _repeat_kv_bwd_kernel[DType.float32],
            _repeat_kv_bwd_kernel[DType.float32],
        ](DDST, DSRC, s, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var DDST = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            d_dst.buf.unsafe_ptr().bitcast[BFloat16](), dst_rl
        )
        var DSRC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        ctx.enqueue_function[
            _repeat_kv_bwd_kernel[DType.bfloat16],
            _repeat_kv_bwd_kernel[DType.bfloat16],
        ](DDST, DSRC, s, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK)
    else:
        var DDST = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            d_dst.buf.unsafe_ptr().bitcast[Float16](), dst_rl
        )
        var DSRC = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        ctx.enqueue_function[
            _repeat_kv_bwd_kernel[DType.float16],
            _repeat_kv_bwd_kernel[DType.float16],
        ](DDST, DSRC, s, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK)
    # sync removed (single-stream ordering; was kernel-trailing host stall)
    var sh = [1, s, h_kv, dh]
    return Tensor(out_buf^, sh^, d_dst.dtype())
