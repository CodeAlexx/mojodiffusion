# ops/vec_permute0213.mojo — VECTORIZED specialized [0,2,1,3] permute.
#
# NEW STANDALONE kernel. Does NOT replace ops/tensor_algebra.mojo `permute`; it
# is a faster sibling for the ONE permutation that dominates DiT attention: the
# [B,S,H,Dh] <-> [B,H,S,Dh] reshape (perm = [0,2,1,3]). Parity is gated against
# the GENERAL scalar `permute` (vec_permute0213_parity.mojo).
#
# Why it's faster: in perm [0,2,1,3] the innermost axis (Dh) is NOT permuted, so
# for a fixed (b, h, s) the whole Dh run is contiguous in BOTH source and dest.
# The general permute issues one scalar gather per element (a div/mod chain over
# 6 padded dims per element). Here, when Dh % 4 == 0, we copy each Dh run with
# width-4 SIMD load/store and skip the per-element index arithmetic entirely —
# the same "copy contiguous runs as vec4" idea behind flame-core's deinterleave
# / float2 kernels (FLAME_KERNELS.md ops/deinterleave.rs).
#
# Layout: input x is [B, S, H, Dh] row-major; output is [B, H, S, Dh] row-major.
#   src offset of (b,s,h,:) = ((b*S + s)*H + h)*Dh
#   dst offset of (b,h,s,:) = ((b*H + h)*S + s)*Dh
# One thread per (b,h,s) run; the thread streams Dh/4 vec4 chunks.
# Requirement: Dh % 4 == 0 (Klein/Z-Image Dh=128/64 satisfy). Else RAISE — the
# caller falls back to the general scalar permute (AGENT-DEFAULT: raise, no
# silent slow tail).
#
# Mojo 1.0.0b1, NVIDIA GPU. Non-F32 input falls back to the dtype-preserving
# general permute.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import permute as _general_permute


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _VW = 4


# One thread per OUTPUT vec4 chunk: full coalescing (consecutive threads write
# consecutive 16-byte output spans) while the per-element div/mod index math is
# amortized over 4 elements and the copy is a single 16-byte load/store. nchunks
# = B*H*S*(Dh/4). Each chunk lives entirely within one (b,h,s) Dh-run because
# Dh % 4 == 0, so the source offset is a clean run-base + intra-run offset.
def _vec_permute0213_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    B: Int, S: Int, H: Int, Dh: Int,
    nchunks: Int,   # = B*H*S*(Dh/4)
):
    var chunk = Int(global_idx.x)
    if chunk >= nchunks:
        return
    var dhv = Dh // _VW
    # decode (b, h, s, kv) from the OUTPUT layout [B,H,S,Dh/4]
    var kv = chunk % dhv
    var rem = chunk // dhv
    var s = rem % S
    rem = rem // S
    var h = rem % H
    var b = rem // H
    var src_off = (((b * S + s) * H + h) * Dh) + kv * _VW
    var dst_off = chunk * _VW
    o.ptr.store[width=_VW](dst_off, x.ptr.load[width=_VW](src_off))


def vec_permute0213(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Specialized [0,2,1,3] permute: [B,S,H,Dh] -> [B,H,S,Dh]."""
    var xshape = x.shape()
    if len(xshape) != 4:
        raise Error("vec_permute0213: x must be rank-4 [B,S,H,Dh]")
    if x.dtype() != STDtype.F32:
        var perm = [0, 2, 1, 3]
        return _general_permute(x, perm, ctx)
    var B = xshape[0]
    var S = xshape[1]
    var H = xshape[2]
    var Dh = xshape[3]
    if Dh % _VW != 0:
        raise Error(
            String("vec_permute0213: Dh must be a multiple of 4 (got ")
            + String(Dh) + ") — use the general permute"
        )
    var nchunks = B * H * S * (Dh // _VW)
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var grid = (nchunks + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_vec_permute0213_kernel, _vec_permute0213_kernel](
        X, O, B, S, H, Dh, nchunks, grid_dim=grid, block_dim=_BLOCK
    )
    var oshape = List[Int]()
    oshape.append(B); oshape.append(H); oshape.append(S); oshape.append(Dh)
    return Tensor(out_buf^, oshape^, STDtype.F32)
