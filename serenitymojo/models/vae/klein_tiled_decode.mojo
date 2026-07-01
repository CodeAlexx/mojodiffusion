# klein_tiled_decode.mojo — Klein (FLUX.2) VAE tiled decode (3x3 overlap +
# feathered blend) for PACKED latents.
#
# The monolithic 1024² decode (KleinVaeDecoder[64,64].decode of a packed
# [1,128,64,64] latent) peaks past 24 GB beside the per-job working set
# (MEASURED: all 20 denoise steps run, then CUDA_OUT_OF_MEMORY at the decode,
# peak ~22 GB). Each LATENT/2 quadrant (KleinVaeDecoder[32,32], packed
# [1,128,32,32] -> [1,3,512,512]) is a small alloc that fits the pool. Crops sit
# at latent stride TILE/2 (positions 0/half/TILE) so adjacent image tiles overlap
# by half a tile; a separable feathered cross-fade (horizontal per row, then
# vertical) erases the conv-boundary seams. Tile vars are reassigned per row so
# prior tiles free → retained-memory peak stays near the single-tile working set.
#
# PACKED latent (flux2-family): the Klein latent carries 128 packed channels
# (LATENT_CH=32 unpatchified 2x2 inside the decoder). Tiling slices ONLY the
# spatial dims (2 = height, 3 = width); the 128-channel dim is untouched — exactly
# as flux_tiled_decode slices its 16-channel packed latent. So the per-tile crops
# are valid packed latents the decoder consumes unchanged. The 16× spatial upscale
# (vs flux's 8×) is irrelevant to the blend, which operates on the decoded image
# tensors (TILE→16·TILE px) and is byte-identical to flux_tiled_decode /
# sdxl_tiled_decode / zimage_tiled_decode (VAE-agnostic image-space cross-fade).
#
# Blend math (_weight_tensor/_xfade/_blend3) is verbatim from flux_tiled_decode;
# only the decoder instance differs (KleinVaeDecoder vs LdmVaeDecoder). Each tile
# is cast to F32 before blending, exactly as the flux/sdxl precedents do.
#
# Mojo 1.0.0b1: `def` not `fn`.

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, mul, slice, concat


# ── feathered cross-fade weight (ramps 1→0 or 0→1 along `dim`), F32 ──────────
# Shape [1,1,n,1] for a height(dim 2) blend, [1,1,1,n] for a width(dim 3) blend,
# so it broadcasts over channels and the non-blend spatial axis.
def _weight_tensor(n: Int, dim: Int, ascending: Bool, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for i in range(n):
        var t = (Float32(i) + 0.5) / Float32(n)
        h.append(t if ascending else (1.0 - t))
    var sh = List[Int]()
    sh.append(1)
    sh.append(1)
    if dim == 2:
        sh.append(n)
        sh.append(1)
    else:
        sh.append(1)
        sh.append(n)
    return Tensor.from_host(h^, sh^, STDtype.F32, ctx)


# ── cross-fade two equal-shaped overlap slabs along `dim` (left fades out) ────
def _xfade(left: Tensor, right: Tensor, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var n = left.shape()[dim]
    var wl = _weight_tensor(n, dim, False, ctx)
    var wr = _weight_tensor(n, dim, True, ctx)
    return add(mul(left, wl, ctx), mul(right, wr, ctx), ctx)


# ── blend 3 equal tiles (size T along `dim`) placed at offsets 0, T/2, T ──────
# Output size 2T: [pure t0 | xfade(t0,t1) | xfade(t1,t2) | pure t2].
def _blend3(t0: Tensor, t1: Tensor, t2: Tensor, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var t = t0.shape()[dim]
    var s = t // 2
    var ov = t - s
    var a = slice(t0, dim, 0, s, ctx)
    var b = _xfade(slice(t0, dim, s, ov, ctx), slice(t1, dim, 0, ov, ctx), dim, ctx)
    var c = _xfade(slice(t1, dim, ov, ov, ctx), slice(t2, dim, 0, ov, ctx), dim, ctx)
    var d = slice(t2, dim, ov, s, ctx)
    return concat(dim, ctx, a, b, c, d)


# ── tiled Klein VAE decode — 3x3 OVERLAPPING packed-latent crops + blend ──────
# packed latent NCHW [1,128,LATENT_H,LATENT_W] -> image NCHW
# [1,3,16*LATENT_H,16*LATENT_W]. Decoder supplied at the TILE shape (LATENT/2),
# reused for all 9 crops at latent stride TILE/2. The 128-channel (dim 1) is never
# sliced — only the spatial dims 2/3 — so each crop is a valid packed latent.
def klein_tiled_decode_with_decoder[
    LATENT_H: Int, LATENT_W: Int, TILE_H: Int, TILE_W: Int
](
    latent: Tensor, dec: KleinVaeDecoder[TILE_H, TILE_W], ctx: DeviceContext
) raises -> Tensor:
    comptime assert TILE_H == LATENT_H // 2, "tile height must be half latent height"
    comptime assert TILE_W == LATENT_W // 2, "tile width must be half latent width"
    var half = TILE_H // 2                          # latent stride
    # row 0 crop [0:TILE_H], blend its 3 columns
    var r = slice(latent, 2, 0, TILE_H, ctx)
    var a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row0 = _blend3(a, b, c, 3, ctx)
    # row 1 crop [half:half+TILE_H] (reassign a/b/c → prior tiles freed)
    r = slice(latent, 2, half, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row1 = _blend3(a, b, c, 3, ctx)
    # row 2 crop [TILE_H:2*TILE_H]
    r = slice(latent, 2, TILE_H, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row2 = _blend3(a, b, c, 3, ctx)
    # vertical feathered blend of the 3 full-width rows
    return _blend3(row0, row1, row2, 2, ctx)


def klein_tiled_decode_nested_2x[
    LATENT_H: Int, LATENT_W: Int, OUTER_H: Int, OUTER_W: Int,
    INNER_H: Int, INNER_W: Int,
](
    latent: Tensor, dec: KleinVaeDecoder[INNER_H, INNER_W], ctx: DeviceContext
) raises -> Tensor:
    comptime assert OUTER_H == LATENT_H // 2, "outer tile height must be half latent height"
    comptime assert OUTER_W == LATENT_W // 2, "outer tile width must be half latent width"
    comptime assert INNER_H == OUTER_H // 2, "inner tile height must be half outer height"
    comptime assert INNER_W == OUTER_W // 2, "inner tile width must be half outer width"
    var half = OUTER_H // 2
    # 2K decode cannot use monolithic 1024px VAE crops while the trainer is
    # resident. Decode each outer crop through the 512px tile path, then apply
    # the same 3x3 feathered blend at the outer grid.
    var r = slice(latent, 2, 0, OUTER_H, ctx)
    var a = klein_tiled_decode_with_decoder[
        OUTER_H, OUTER_W, INNER_H, INNER_W
    ](slice(r, 3, 0, OUTER_W, ctx), dec, ctx)
    var b = klein_tiled_decode_with_decoder[
        OUTER_H, OUTER_W, INNER_H, INNER_W
    ](slice(r, 3, half, OUTER_W, ctx), dec, ctx)
    var c = klein_tiled_decode_with_decoder[
        OUTER_H, OUTER_W, INNER_H, INNER_W
    ](slice(r, 3, OUTER_W, OUTER_W, ctx), dec, ctx)
    var row0 = _blend3(a, b, c, 3, ctx)

    r = slice(latent, 2, half, OUTER_H, ctx)
    a = klein_tiled_decode_with_decoder[
        OUTER_H, OUTER_W, INNER_H, INNER_W
    ](slice(r, 3, 0, OUTER_W, ctx), dec, ctx)
    b = klein_tiled_decode_with_decoder[
        OUTER_H, OUTER_W, INNER_H, INNER_W
    ](slice(r, 3, half, OUTER_W, ctx), dec, ctx)
    c = klein_tiled_decode_with_decoder[
        OUTER_H, OUTER_W, INNER_H, INNER_W
    ](slice(r, 3, OUTER_W, OUTER_W, ctx), dec, ctx)
    var row1 = _blend3(a, b, c, 3, ctx)

    r = slice(latent, 2, OUTER_H, OUTER_H, ctx)
    a = klein_tiled_decode_with_decoder[
        OUTER_H, OUTER_W, INNER_H, INNER_W
    ](slice(r, 3, 0, OUTER_W, ctx), dec, ctx)
    b = klein_tiled_decode_with_decoder[
        OUTER_H, OUTER_W, INNER_H, INNER_W
    ](slice(r, 3, half, OUTER_W, ctx), dec, ctx)
    c = klein_tiled_decode_with_decoder[
        OUTER_H, OUTER_W, INNER_H, INNER_W
    ](slice(r, 3, OUTER_W, OUTER_W, ctx), dec, ctx)
    var row2 = _blend3(a, b, c, 3, ctx)
    return _blend3(row0, row1, row2, 2, ctx)


# Convenience: load the TILE-shaped KleinVaeDecoder and tiled-decode a packed
# latent. `vae_path` is the same file cfg.vae the monolithic
# KleinVaeDecoder[LH,LW].load consumes.
def klein_tiled_decode[
    LATENT_H: Int, LATENT_W: Int
](latent: Tensor, vae_path: String, ctx: DeviceContext) raises -> Tensor:
    comptime if LATENT_H >= 128 or LATENT_W >= 128:
        comptime OUTER_H = LATENT_H // 2
        comptime OUTER_W = LATENT_W // 2
        comptime INNER_H = OUTER_H // 2
        comptime INNER_W = OUTER_W // 2
        var dec = KleinVaeDecoder[INNER_H, INNER_W].load(vae_path, ctx)
        return klein_tiled_decode_nested_2x[
            LATENT_H, LATENT_W, OUTER_H, OUTER_W, INNER_H, INNER_W
        ](latent, dec, ctx)
    else:
        comptime TILE_H = LATENT_H // 2
        comptime TILE_W = LATENT_W // 2
        var dec = KleinVaeDecoder[TILE_H, TILE_W].load(vae_path, ctx)
        return klein_tiled_decode_with_decoder[
            LATENT_H, LATENT_W, TILE_H, TILE_W
        ](latent, dec, ctx)
