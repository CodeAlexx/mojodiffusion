# ops/layout.mojo — DiT patchify / unpatchify + interleaved-SwiGLU deinterleave.
#
# patchify(x, p):    image [B, C, H, W] -> sequence [B, L, C*p*p] where
#                    L = (H/p)*(W/p). Patch grid is raster-scanned (row-major
#                    over patch rows then patch cols); WITHIN a patch the
#                    channels-major flatten is (c, ph, pw) — i.e. for each
#                    channel the p×p block is laid out row-major. This matches
#                    the standard DiT/Flux patch-embed `Rearrange
#                    'b c (h p1) (w p2) -> b (h w) (c p1 p2)'`.
# unpatchify(seq, ..):  exact inverse, [B, L, C*p*p] -> [B, C, H, W].
#
# deinterleave_pair(x): split last dim [..., 2K] into two [..., K] tensors —
#                    even columns (0,2,4,...) and odd columns (1,3,5,...). Used
#                    to un-interleave a fused [gate,up] tensor stored as
#                    g0,u0,g1,u1,... before SwiGLU. Returns (evens, odds).
#
# Kernel style mirrors ops/norm.mojo: runtime _DYN1 layouts, three dtype
# branches, one thread per OUTPUT element, cast-on-store. No reduction here
# (pure gather/scatter index math), so no F32 accumulation needed — values pass
# through unchanged, only relocated.
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
# patchify. One thread per OUTPUT element. Output [B, L, C*p*p]; decode the
# output flat index into (b, l, f), where l = patch index (gh*GW + gw) and
# f = (c*p + ph)*p + pw. Map back to input offset in [B,C,H,W]:
#   ih = gh*p + ph,  iw = gw*p + pw
#   in_off = ((b*C + c)*H + ih)*W + iw
# ─────────────────────────────────────────────────────────────────────────────
def _patchify_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, p: Int, GH: Int, GW: Int,
):
    var idx = Int(global_idx.x)
    var L = GH * GW
    var F = C * p * p
    var total = B * L * F
    if idx < total:
        var f = idx % F
        var rem = idx // F
        var l = rem % L
        var b = rem // L
        var gh = l // GW
        var gw = l % GW
        var pw = f % p
        var t = f // p
        var ph = t % p
        var c = t // p
        var ih = gh * p + ph
        var iw = gw * p + pw
        var in_off = ((b * C + c) * H + ih) * W + iw
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[in_off]))


def _patchify_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, p: Int, GH: Int, GW: Int,
):
    var idx = Int(global_idx.x)
    var L = GH * GW
    var F = C * p * p
    var total = B * L * F
    if idx < total:
        var f = idx % F
        var rem = idx // F
        var l = rem % L
        var b = rem // L
        var gh = l // GW
        var gw = l % GW
        var pw = f % p
        var t = f // p
        var ph = t % p
        var c = t // p
        var ih = gh * p + ph
        var iw = gw * p + pw
        var in_off = ((b * C + c) * H + ih) * W + iw
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](x[in_off]))


def _patchify_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, p: Int, GH: Int, GW: Int,
):
    var idx = Int(global_idx.x)
    var L = GH * GW
    var F = C * p * p
    var total = B * L * F
    if idx < total:
        var f = idx % F
        var rem = idx // F
        var l = rem % L
        var b = rem // L
        var gh = l // GW
        var gw = l % GW
        var pw = f % p
        var t = f // p
        var ph = t % p
        var c = t // p
        var ih = gh * p + ph
        var iw = gw * p + pw
        var in_off = ((b * C + c) * H + ih) * W + iw
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](x[in_off]))


def patchify(x: Tensor, patch: Int, ctx: DeviceContext) raises -> Tensor:
    """DiT patchify: image [B,C,H,W] -> sequence [B, (H/p)*(W/p), C*p*p].

    `patch` (p) must divide both H and W. Within-patch flatten is (c, ph, pw)
    (channels-major), matching 'b c (h p1) (w p2) -> b (h w) (c p1 p2)'."""
    var xshape = x.shape()
    if len(xshape) != 4:
        raise Error("patchify: x must be rank-4 [B,C,H,W]")
    var B = xshape[0]
    var C = xshape[1]
    var H = xshape[2]
    var W = xshape[3]
    var p = patch
    if p <= 0 or H % p != 0 or W % p != 0:
        raise Error("patchify: patch must divide H and W")
    var GH = H // p
    var GW = W // p
    var L = GH * GW
    var F = C * p * p
    var total = B * L * F

    var dt = x.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var grid = (total + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_patchify_kernel_f32, _patchify_kernel_f32](
            X, O, B, C, H, W, p, GH, GW, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_patchify_kernel_bf16, _patchify_kernel_bf16](
            X, O, B, C, H, W, p, GH, GW, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_patchify_kernel_f16, _patchify_kernel_f16](
            X, O, B, C, H, W, p, GH, GW, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, [B, L, F], x.dtype())


# ─────────────────────────────────────────────────────────────────────────────
# unpatchify. Inverse of patchify. One thread per OUTPUT element. Output
# [B,C,H,W]; decode (b, c, ih, iw); recover (gh, ph)=(ih//p, ih%p),
# (gw, pw)=(iw//p, iw%p); l = gh*GW+gw; f = (c*p+ph)*p+pw; read seq[b, l, f].
# ─────────────────────────────────────────────────────────────────────────────
def _unpatchify_kernel_f32(
    seq: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, p: Int, GH: Int, GW: Int,
):
    var idx = Int(global_idx.x)
    var total = B * C * H * W
    if idx < total:
        var iw = idx % W
        var rem = idx // W
        var ih = rem % H
        rem = rem // H
        var c = rem % C
        var b = rem // C
        var gh = ih // p
        var ph = ih % p
        var gw = iw // p
        var pw = iw % p
        var L = GH * GW
        var F = C * p * p
        var l = gh * GW + gw
        var f = (c * p + ph) * p + pw
        var seq_off = (b * L + l) * F + f
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](seq[seq_off]))


def _unpatchify_kernel_bf16(
    seq: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, p: Int, GH: Int, GW: Int,
):
    var idx = Int(global_idx.x)
    var total = B * C * H * W
    if idx < total:
        var iw = idx % W
        var rem = idx // W
        var ih = rem % H
        rem = rem // H
        var c = rem % C
        var b = rem // C
        var gh = ih // p
        var ph = ih % p
        var gw = iw // p
        var pw = iw % p
        var L = GH * GW
        var F = C * p * p
        var l = gh * GW + gw
        var f = (c * p + ph) * p + pw
        var seq_off = (b * L + l) * F + f
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](seq[seq_off]))


def _unpatchify_kernel_f16(
    seq: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, p: Int, GH: Int, GW: Int,
):
    var idx = Int(global_idx.x)
    var total = B * C * H * W
    if idx < total:
        var iw = idx % W
        var rem = idx // W
        var ih = rem % H
        rem = rem // H
        var c = rem % C
        var b = rem // C
        var gh = ih // p
        var ph = ih % p
        var gw = iw // p
        var pw = iw % p
        var L = GH * GW
        var F = C * p * p
        var l = gh * GW + gw
        var f = (c * p + ph) * p + pw
        var seq_off = (b * L + l) * F + f
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](seq[seq_off]))


def unpatchify(
    seq: Tensor, channels: Int, height: Int, width: Int, patch: Int, ctx: DeviceContext
) raises -> Tensor:
    """Inverse of patchify: sequence [B, L, C*p*p] -> image [B, C, H, W].

    The output geometry can't be inferred from the sequence alone, so C/H/W and
    the patch size are passed explicitly. Requires L == (H/p)*(W/p) and
    last_dim == C*p*p."""
    var sshape = seq.shape()
    if len(sshape) != 3:
        raise Error("unpatchify: seq must be rank-3 [B, L, C*p*p]")
    var B = sshape[0]
    var L = sshape[1]
    var Fdim = sshape[2]
    var C = channels
    var H = height
    var W = width
    var p = patch
    if p <= 0 or H % p != 0 or W % p != 0:
        raise Error("unpatchify: patch must divide H and W")
    var GH = H // p
    var GW = W // p
    if L != GH * GW:
        raise Error("unpatchify: L != (H/p)*(W/p)")
    if Fdim != C * p * p:
        raise Error("unpatchify: last dim != C*p*p")
    var total = B * C * H * W

    var dt = seq.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](seq.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var grid = (total + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            seq.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_unpatchify_kernel_f32, _unpatchify_kernel_f32](
            S, O, B, C, H, W, p, GH, GW, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            seq.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_unpatchify_kernel_bf16, _unpatchify_kernel_bf16](
            S, O, B, C, H, W, p, GH, GW, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var S = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            seq.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_unpatchify_kernel_f16, _unpatchify_kernel_f16](
            S, O, B, C, H, W, p, GH, GW, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, [B, C, H, W], seq.dtype())


# ─────────────────────────────────────────────────────────────────────────────
# deinterleave_pair. Input [..., 2K] (flattened rows × 2K). Output evens [...,K]
# (cols 0,2,4,...) and odds [...,K] (cols 1,3,5,...). One thread per OUTPUT-half
# element: evens[r, j] = in[r, 2j]; odds[r, j] = in[r, 2j+1].
# ─────────────────────────────────────────────────────────────────────────────
def _deinterleave_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ev: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    od: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int, K: Int,
):
    var idx = Int(global_idx.x)
    var half = rows * K
    if idx < half:
        var r = idx // K
        var j = idx % K
        var base = r * (2 * K)
        ev[idx] = rebind[ev.element_type](rebind[Scalar[DType.float32]](x[base + 2 * j]))
        od[idx] = rebind[od.element_type](rebind[Scalar[DType.float32]](x[base + 2 * j + 1]))


def _deinterleave_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    ev: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    od: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    rows: Int, K: Int,
):
    var idx = Int(global_idx.x)
    var half = rows * K
    if idx < half:
        var r = idx // K
        var j = idx % K
        var base = r * (2 * K)
        ev[idx] = rebind[ev.element_type](rebind[Scalar[DType.bfloat16]](x[base + 2 * j]))
        od[idx] = rebind[od.element_type](rebind[Scalar[DType.bfloat16]](x[base + 2 * j + 1]))


def _deinterleave_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    ev: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    od: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    rows: Int, K: Int,
):
    var idx = Int(global_idx.x)
    var half = rows * K
    if idx < half:
        var r = idx // K
        var j = idx % K
        var base = r * (2 * K)
        ev[idx] = rebind[ev.element_type](rebind[Scalar[DType.float16]](x[base + 2 * j]))
        od[idx] = rebind[od.element_type](rebind[Scalar[DType.float16]](x[base + 2 * j + 1]))


def deinterleave_pair(
    x: Tensor, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    """Split last dim [..., 2K] into (evens [...,K], odds [...,K]). evens are
    cols 0,2,4,...; odds are cols 1,3,5,... (interleaved-SwiGLU un-fuse)."""
    var xshape = x.shape()
    var rank = len(xshape)
    if rank < 1:
        raise Error("deinterleave_pair: x must have rank >= 1")
    var last = xshape[rank - 1]
    if last % 2 != 0:
        raise Error("deinterleave_pair: last dim must be even")
    var K = last // 2
    var rows = 1
    for i in range(rank - 1):
        rows *= xshape[i]
    var half = rows * K
    var out_shape = List[Int]()
    for i in range(rank - 1):
        out_shape.append(xshape[i])
    out_shape.append(K)

    var dt = x.dtype().to_mojo_dtype()
    var bsz = x.dtype().byte_size()
    var ev_buf = ctx.enqueue_create_buffer[DType.uint8](half * bsz)
    var od_buf = ctx.enqueue_create_buffer[DType.uint8](half * bsz)
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * last))
    var h_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](half))
    var grid = (half + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), x_rl
        )
        var EV = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            ev_buf.unsafe_ptr().bitcast[Float32](), h_rl
        )
        var OD = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            od_buf.unsafe_ptr().bitcast[Float32](), h_rl
        )
        ctx.enqueue_function[_deinterleave_kernel_f32, _deinterleave_kernel_f32](
            X, EV, OD, rows, K, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
        )
        var EV = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            ev_buf.unsafe_ptr().bitcast[BFloat16](), h_rl
        )
        var OD = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            od_buf.unsafe_ptr().bitcast[BFloat16](), h_rl
        )
        ctx.enqueue_function[_deinterleave_kernel_bf16, _deinterleave_kernel_bf16](
            X, EV, OD, rows, K, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), x_rl
        )
        var EV = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            ev_buf.unsafe_ptr().bitcast[Float16](), h_rl
        )
        var OD = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            od_buf.unsafe_ptr().bitcast[Float16](), h_rl
        )
        ctx.enqueue_function[_deinterleave_kernel_f16, _deinterleave_kernel_f16](
            X, EV, OD, rows, K, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return (Tensor(ev_buf^, out_shape.copy(), x.dtype()), Tensor(od_buf^, out_shape^, x.dtype()))
