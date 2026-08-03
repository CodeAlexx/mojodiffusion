# serenitymojo/pipeline/minimax_h3_video_vae_blend.mojo
#
# Shared linear cross-fade for MiniMax-H3's VAE seam blending. ONE
# implementation, `dim`-parameterized, used by BOTH the temporal chunk
# layer (pipeline/minimax_h3_video_vae_temporal.mojo, blends along the
# frame axis) and the spatial tiling layer
# (pipeline/minimax_h3_video_vae_spatial_tiling.mojo, blends along H then
# W) — klvae.py's own `AutoencoderKL.blend` (:220-250) is ALSO one function
# reused for both temporal (`dim=-3`) and spatial (`dim=-2`/`dim=-1`)
# seams; this mirrors that reuse rather than duplicating the math per axis.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, concat, mul, slice


def minimax_h3_video_blend(
    a: Tensor, b: Tensor, blend_extent: Int, dim: Int, ctx: DeviceContext,
) raises -> Tensor:
    """Linear cross-fade the LAST `blend_extent` positions of `a` along
    `dim` with the FIRST `blend_extent` positions of `b`, `b`'s remaining
    positions appended untouched. Matches `AutoencoderKL.blend` exactly:
    `weight_a = 1 - i/extent`, `weight_b = i/extent` for `i` in
    `[0, extent)`; `extent` is clamped to `min(a.shape[dim], b.shape[dim],
    blend_extent)`, and `extent <= 0` returns `b` unchanged (blend_extent==0
    is the reference's own "no overlap" case, not an error)."""
    var ash = a.shape()
    var bsh = b.shape()
    var rank = len(ash)
    if dim < 0 or dim >= rank:
        raise Error("MiniMax-H3 video blend: dim out of range")
    if len(bsh) != rank:
        raise Error("MiniMax-H3 video blend: a/b rank mismatch")

    var extent = blend_extent
    if ash[dim] < extent:
        extent = ash[dim]
    if bsh[dim] < extent:
        extent = bsh[dim]
    if extent <= 0:
        return b.clone(ctx)

    var a_overlap = slice(a, dim, ash[dim] - extent, extent, ctx)
    var b_overlap = slice(b, dim, 0, extent, ctx)

    var wb_host = List[Float32](capacity=extent)
    var wa_host = List[Float32](capacity=extent)
    for i in range(extent):
        var t = Float32(i) / Float32(extent)
        wb_host.append(t)
        wa_host.append(Float32(1.0) - t)
    var wshape = List[Int](capacity=rank)
    for i in range(rank):
        wshape.append(extent if i == dim else 1)
    var wb = Tensor.from_host(wb_host, wshape.copy(), a.dtype(), ctx)
    var wa = Tensor.from_host(wa_host, wshape^, a.dtype(), ctx)

    var blended = add(mul(a_overlap, wa, ctx), mul(b_overlap, wb, ctx), ctx)
    if extent < bsh[dim]:
        var b_rest = slice(b, dim, extent, bsh[dim] - extent, ctx)
        return concat(dim, ctx, blended, b_rest)
    return blended^
