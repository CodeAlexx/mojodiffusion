# ops/pixelshuffle.mojo — VAE upsample interleaving: depth_to_space (3D) +
# pixel_shuffle / pixel_unshuffle. Pure reshape+permute (index remapping), no
# learnable params, no math — values pass through unchanged, only relocated.
#
# LTX2 P-d2s (LTX2_PORT_PLAN_2026-05-28 §P-d2s). Mirrors the Rust video-VAE
# decoder `depth_to_space` and `unpatchify` (ltx2_vae.rs:312-322, 370-385):
#
#   depth_to_space_3d(x, (p1,p2,p3)):
#       [B, C*p1*p2*p3, F, H, W] -> [B, C, F*p1, H*p2, W*p3]
#     Rust: reshape [B,C,p1,p2,p3,F,H,W] -> permute [0,1,5,2,6,3,7,4]
#           -> reshape [B,C, F*p1, H*p2, W*p3].
#     Channel split is c-major:  ct = ((c*p1 + i1)*p2 + i2)*p3 + i3.
#     Output coords:  fo = f*p1 + i1,  ho = h*p2 + i2,  wo = w*p3 + i3.
#     If `drop_first_temporal` (temporal stride==2 in the decoder), the first
#     output frame is dropped (causal-duplication artifact, ltx2_vae.rs:301-306)
#     -> [B, C, F*p1 - 1, H*p2, W*p3].
#
#   pixel_unshuffle(x, r):  [B, C, H, W] -> [B, C*r*r, H/r, W/r]
#     channel c_out = (c*r + i)*r + j  reads input (h*r + i, w*r + j).
#   pixel_shuffle(x, r):    [B, C*r*r, H, W] -> [B, C, H*r, W*r]   (exact inverse)
#     output (ho, wo) = (h*r + i, w*r + j) reads channel (c*r + i)*r + j.
#
# Kernel style mirrors ops/layout.mojo (patchify/unpatchify): runtime _DYN1
# layouts, three dtype branches, one thread per OUTPUT element, cast-on-store.
# No reduction (gather only) -> no F32 accumulation; bit-exact relocation.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ─────────────────────────────────────────────────────────────────────────────
# depth_to_space_3d. One thread per OUTPUT element. Output flat layout
# [B, C, FO, HO, WO] where FO=F*p1 (then optionally minus the dropped frame),
# HO=H*p2, WO=W*p3. The output flat index iterates over the *kept* frames; we
# add `drop` to the decoded fo so the gather reads the correct source frame.
# ─────────────────────────────────────────────────────────────────────────────
def _d2s_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    B: Int, C: Int, F: Int, H: Int, W: Int,
    p1: Int, p2: Int, p3: Int, FOk: Int, drop: Int,
):
    var idx = Int(global_idx.x)
    var HO = H * p2
    var WO = W * p3
    var total = B * C * FOk * HO * WO
    if idx < total:
        var wo = idx % WO
        var t0 = idx // WO
        var ho = t0 % HO
        var t1 = t0 // HO
        var fo_k = t1 % FOk
        var t2 = t1 // FOk
        var c = t2 % C
        var b = t2 // C
        var fo = fo_k + drop
        var f = fo // p1
        var i1 = fo % p1
        var h = ho // p2
        var i2 = ho % p2
        var w = wo // p3
        var i3 = wo % p3
        var ct = ((c * p1 + i1) * p2 + i2) * p3 + i3
        var Ctot = C * p1 * p2 * p3
        var in_off = (((b * Ctot + ct) * F + f) * H + h) * W + w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[in_off]))


def _d2s_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    B: Int, C: Int, F: Int, H: Int, W: Int,
    p1: Int, p2: Int, p3: Int, FOk: Int, drop: Int,
):
    var idx = Int(global_idx.x)
    var HO = H * p2
    var WO = W * p3
    var total = B * C * FOk * HO * WO
    if idx < total:
        var wo = idx % WO
        var t0 = idx // WO
        var ho = t0 % HO
        var t1 = t0 // HO
        var fo_k = t1 % FOk
        var t2 = t1 // FOk
        var c = t2 % C
        var b = t2 // C
        var fo = fo_k + drop
        var f = fo // p1
        var i1 = fo % p1
        var h = ho // p2
        var i2 = ho % p2
        var w = wo // p3
        var i3 = wo % p3
        var ct = ((c * p1 + i1) * p2 + i2) * p3 + i3
        var Ctot = C * p1 * p2 * p3
        var in_off = (((b * Ctot + ct) * F + f) * H + h) * W + w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](x[in_off]))


def _d2s_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    B: Int, C: Int, F: Int, H: Int, W: Int,
    p1: Int, p2: Int, p3: Int, FOk: Int, drop: Int,
):
    var idx = Int(global_idx.x)
    var HO = H * p2
    var WO = W * p3
    var total = B * C * FOk * HO * WO
    if idx < total:
        var wo = idx % WO
        var t0 = idx // WO
        var ho = t0 % HO
        var t1 = t0 // HO
        var fo_k = t1 % FOk
        var t2 = t1 // FOk
        var c = t2 % C
        var b = t2 // C
        var fo = fo_k + drop
        var f = fo // p1
        var i1 = fo % p1
        var h = ho // p2
        var i2 = ho % p2
        var w = wo // p3
        var i3 = wo % p3
        var ct = ((c * p1 + i1) * p2 + i2) * p3 + i3
        var Ctot = C * p1 * p2 * p3
        var in_off = (((b * Ctot + ct) * F + f) * H + h) * W + w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](x[in_off]))


def depth_to_space_3d(
    x: Tensor,
    p1: Int,
    p2: Int,
    p3: Int,
    drop_first_temporal: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    """depth_to_space_3d: [B, C*p1*p2*p3, F,H,W] -> [B, C, F*p1, H*p2, W*p3].

    Trades channels for spatio-temporal resolution. Channel split is c-major
    (ct = ((c*p1+i1)*p2+i2)*p3+i3), matching the Rust permute [0,1,5,2,6,3,7,4]
    (ltx2_vae.rs:312-322). If `drop_first_temporal` (used when the temporal
    stride p1==2 in the decoder, ltx2_vae.rs:301-306), the first output frame is
    dropped -> F*p1 - 1 output frames."""
    var xs = x.shape()
    if len(xs) != 5:
        raise Error("depth_to_space_3d: x must be rank-5 [B,C*p1*p2*p3,F,H,W]")
    var B = xs[0]
    var Ctot = xs[1]
    var F = xs[2]
    var H = xs[3]
    var W = xs[4]
    var prod = p1 * p2 * p3
    if p1 <= 0 or p2 <= 0 or p3 <= 0 or Ctot % prod != 0:
        raise Error("depth_to_space_3d: stride product must divide channel dim")
    var C = Ctot // prod
    var FO = F * p1
    var drop = 1 if drop_first_temporal else 0
    if drop_first_temporal and FO < 1:
        raise Error("depth_to_space_3d: cannot drop frame from empty temporal axis")
    var FOk = FO - drop
    var HO = H * p2
    var WO = W * p3
    var total = B * C * FOk * HO * WO

    var dt = x.dtype().to_mojo_dtype()
    var out_bytes = total * x.dtype().byte_size()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_bytes)
    var in_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var grid = (total + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), in_rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[_d2s_kernel_f32, _d2s_kernel_f32](
            X, O, B, C, F, H, W, p1, p2, p3, FOk, drop,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), in_rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[_d2s_kernel_bf16, _d2s_kernel_bf16](
            X, O, B, C, F, H, W, p1, p2, p3, FOk, drop,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), in_rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[_d2s_kernel_f16, _d2s_kernel_f16](
            X, O, B, C, F, H, W, p1, p2, p3, FOk, drop,
            grid_dim=grid, block_dim=_BLOCK,
        )
    ctx.synchronize()
    return Tensor(out_buf^, [B, C, FOk, HO, WO], x.dtype())


# ─────────────────────────────────────────────────────────────────────────────
# pixel_unshuffle. One thread per OUTPUT element. Output [B, C*r*r, H/r, W/r].
# Decode (b, c_out, ho, wo); c_out = (c*r + i)*r + j -> read input
# (h = ho*r + i, w = wo*r + j) at channel c. in_off = ((b*C + c)*H + h)*W + w.
# ─────────────────────────────────────────────────────────────────────────────
def _unshuffle_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, r: Int,
):
    var idx = Int(global_idx.x)
    var Co = C * r * r
    var Ho = H // r
    var Wo = W // r
    var total = B * Co * Ho * Wo
    if idx < total:
        var wo = idx % Wo
        var t0 = idx // Wo
        var ho = t0 % Ho
        var t1 = t0 // Ho
        var c_out = t1 % Co
        var b = t1 // Co
        var c = c_out // (r * r)
        var rem = c_out % (r * r)
        var i = rem // r
        var j = rem % r
        var h = ho * r + i
        var w = wo * r + j
        var in_off = ((b * C + c) * H + h) * W + w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[in_off]))


def _unshuffle_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, r: Int,
):
    var idx = Int(global_idx.x)
    var Co = C * r * r
    var Ho = H // r
    var Wo = W // r
    var total = B * Co * Ho * Wo
    if idx < total:
        var wo = idx % Wo
        var t0 = idx // Wo
        var ho = t0 % Ho
        var t1 = t0 // Ho
        var c_out = t1 % Co
        var b = t1 // Co
        var c = c_out // (r * r)
        var rem = c_out % (r * r)
        var i = rem // r
        var j = rem % r
        var h = ho * r + i
        var w = wo * r + j
        var in_off = ((b * C + c) * H + h) * W + w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](x[in_off]))


def _unshuffle_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, r: Int,
):
    var idx = Int(global_idx.x)
    var Co = C * r * r
    var Ho = H // r
    var Wo = W // r
    var total = B * Co * Ho * Wo
    if idx < total:
        var wo = idx % Wo
        var t0 = idx // Wo
        var ho = t0 % Ho
        var t1 = t0 // Ho
        var c_out = t1 % Co
        var b = t1 // Co
        var c = c_out // (r * r)
        var rem = c_out % (r * r)
        var i = rem // r
        var j = rem % r
        var h = ho * r + i
        var w = wo * r + j
        var in_off = ((b * C + c) * H + h) * W + w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](x[in_off]))


def pixel_unshuffle(x: Tensor, r: Int, ctx: DeviceContext) raises -> Tensor:
    """pixel_unshuffle: [B, C, H, W] -> [B, C*r*r, H/r, W/r].

    channel c_out = (c*r + i)*r + j gathers input (h*r+i, w*r+j). r must divide
    both H and W. Exact inverse of `pixel_shuffle`."""
    var xs = x.shape()
    if len(xs) != 4:
        raise Error("pixel_unshuffle: x must be rank-4 [B,C,H,W]")
    var B = xs[0]
    var C = xs[1]
    var H = xs[2]
    var W = xs[3]
    if r <= 0 or H % r != 0 or W % r != 0:
        raise Error("pixel_unshuffle: r must divide H and W")
    var Co = C * r * r
    var Ho = H // r
    var Wo = W // r
    var total = B * Co * Ho * Wo

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var grid = (total + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_unshuffle_kernel_f32, _unshuffle_kernel_f32](
            X, O, B, C, H, W, r, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_unshuffle_kernel_bf16, _unshuffle_kernel_bf16](
            X, O, B, C, H, W, r, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_unshuffle_kernel_f16, _unshuffle_kernel_f16](
            X, O, B, C, H, W, r, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, [B, Co, Ho, Wo], x.dtype())


# ─────────────────────────────────────────────────────────────────────────────
# pixel_shuffle. Inverse of pixel_unshuffle. One thread per OUTPUT element.
# Output [B, C, H*r, W*r]. Decode (b, c, ho, wo); (h,i)=(ho//r, ho%r),
# (w,j)=(wo//r, wo%r); c_in = (c*r + i)*r + j; read input [b, c_in, h, w].
# ─────────────────────────────────────────────────────────────────────────────
def _shuffle_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, r: Int,
):
    var idx = Int(global_idx.x)
    var Cin = C * r * r
    var Ho = H * r
    var Wo = W * r
    var total = B * C * Ho * Wo
    if idx < total:
        var wo = idx % Wo
        var t0 = idx // Wo
        var ho = t0 % Ho
        var t1 = t0 // Ho
        var c = t1 % C
        var b = t1 // C
        var h = ho // r
        var i = ho % r
        var w = wo // r
        var j = wo % r
        var c_in = (c * r + i) * r + j
        var in_off = ((b * Cin + c_in) * H + h) * W + w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[in_off]))


def _shuffle_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, r: Int,
):
    var idx = Int(global_idx.x)
    var Cin = C * r * r
    var Ho = H * r
    var Wo = W * r
    var total = B * C * Ho * Wo
    if idx < total:
        var wo = idx % Wo
        var t0 = idx // Wo
        var ho = t0 % Ho
        var t1 = t0 // Ho
        var c = t1 % C
        var b = t1 // C
        var h = ho // r
        var i = ho % r
        var w = wo // r
        var j = wo % r
        var c_in = (c * r + i) * r + j
        var in_off = ((b * Cin + c_in) * H + h) * W + w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](x[in_off]))


def _shuffle_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, r: Int,
):
    var idx = Int(global_idx.x)
    var Cin = C * r * r
    var Ho = H * r
    var Wo = W * r
    var total = B * C * Ho * Wo
    if idx < total:
        var wo = idx % Wo
        var t0 = idx // Wo
        var ho = t0 % Ho
        var t1 = t0 // Ho
        var c = t1 % C
        var b = t1 // C
        var h = ho // r
        var i = ho % r
        var w = wo // r
        var j = wo % r
        var c_in = (c * r + i) * r + j
        var in_off = ((b * Cin + c_in) * H + h) * W + w
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](x[in_off]))


def pixel_shuffle(x: Tensor, r: Int, ctx: DeviceContext) raises -> Tensor:
    """pixel_shuffle: [B, C*r*r, H, W] -> [B, C, H*r, W*r]. Exact inverse of
    `pixel_unshuffle`. Output (ho,wo)=(h*r+i, w*r+j) reads channel (c*r+i)*r+j."""
    var xs = x.shape()
    if len(xs) != 4:
        raise Error("pixel_shuffle: x must be rank-4 [B,C*r*r,H,W]")
    var B = xs[0]
    var Cin = xs[1]
    var H = xs[2]
    var W = xs[3]
    if r <= 0 or Cin % (r * r) != 0:
        raise Error("pixel_shuffle: r*r must divide channel dim")
    var C = Cin // (r * r)
    var Ho = H * r
    var Wo = W * r
    var total = B * C * Ho * Wo

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](x.numel()))
    var grid = (total + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_shuffle_kernel_f32, _shuffle_kernel_f32](
            X, O, B, C, H, W, r, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_shuffle_kernel_bf16, _shuffle_kernel_bf16](
            X, O, B, C, H, W, r, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_shuffle_kernel_f16, _shuffle_kernel_f16](
            X, O, B, C, H, W, r, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, [B, C, Ho, Wo], x.dtype())
