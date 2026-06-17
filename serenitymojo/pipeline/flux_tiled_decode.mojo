# flux_tiled_decode.mojo — shared FLUX VAE tiled-decode (3x3 overlap + feathered
# blend). Extracted verbatim from flux_sample_cli.mojo so the CLI and the
# tile-seam parity gate (flux_tiled_decode_parity.mojo) exercise ONE
# implementation.
#
# After the offloaded DiT, the caching allocator pool is at a high water mark and
# a single 1024² VAE decode buffer OOMs. Each crop decodes a LATENT/2 quadrant
# (small alloc that fits the pool). Crops sit at latent stride TILE/2 (positions
# 0/half/TILE) so adjacent image tiles overlap; a separable feathered cross-fade
# (horizontal per row, then vertical) erases seams. Tile vars are reassigned per
# row so prior tiles free → retained-memory peak stays near the 2x2 working path.

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.ldm_decoder import load_flux1_ldm_decoder
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


# ── blend 5 equal tiles placed at offsets 0, 3T/4, 6T/4, 9T/4, 12T/4 ───────
# Output size 4T with T/4 overlaps. This keeps the same feathered-overlap
# semantics as _blend3 while using 256px VAE tiles for lower peak memory.
def _blend5(
    t0: Tensor, t1: Tensor, t2: Tensor, t3: Tensor, t4: Tensor,
    dim: Int, ctx: DeviceContext,
) raises -> Tensor:
    var t = t0.shape()[dim]
    var s = (t * 3) // 4
    var ov = t - s
    var a = slice(t0, dim, 0, s, ctx)
    var b = _xfade(slice(t0, dim, s, ov, ctx), slice(t1, dim, 0, ov, ctx), dim, ctx)
    var c = slice(t1, dim, ov, s - ov, ctx)
    var d = _xfade(slice(t1, dim, s, ov, ctx), slice(t2, dim, 0, ov, ctx), dim, ctx)
    var e = slice(t2, dim, ov, s - ov, ctx)
    var f = _xfade(slice(t2, dim, s, ov, ctx), slice(t3, dim, 0, ov, ctx), dim, ctx)
    var g = slice(t3, dim, ov, s - ov, ctx)
    var h = _xfade(slice(t3, dim, s, ov, ctx), slice(t4, dim, 0, ov, ctx), dim, ctx)
    var i = slice(t4, dim, ov, t - ov, ctx)
    return concat(dim, ctx, a, b, c, d, e, f, g, h, i)


# ── tiled VAE decode — 3x3 OVERLAPPING latent crops + feathered blend ────────
# latent NCHW [1,16,LATENT_H,LATENT_W] -> image NCHW [1,3,8*LATENT_H,8*LATENT_W].
# Decoder is instantiated once at the TILE shape (LATENT/2) and reused for all 9
# crops at latent stride TILE/2.
def flux_tiled_decode[
    LATENT_H: Int, LATENT_W: Int
](latent: Tensor, vae_path: String, ctx: DeviceContext) raises -> Tensor:
    comptime TILE_H = LATENT_H // 2
    comptime TILE_W = LATENT_W // 2
    var dec = load_flux1_ldm_decoder[TILE_H, TILE_W](vae_path, ctx)
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


# Lower-memory server decode: 5x5 overlapping latent crops, each LATENT/4.
# Slower than the 3x3 path but leaves roughly 4x less per-crop VAE activation
# pressure, which is the difference between OOM and a manifest-backed product
# artifact on a 24 GB card with another small GPU context present.
def flux_tiled_decode_5x5_lowmem[
    LATENT_H: Int, LATENT_W: Int
](latent: Tensor, vae_path: String, ctx: DeviceContext) raises -> Tensor:
    comptime TILE_H = LATENT_H // 4
    comptime TILE_W = LATENT_W // 4
    var dec = load_flux1_ldm_decoder[TILE_H, TILE_W](vae_path, ctx)
    var stride = (TILE_H * 3) // 4

    var r = slice(latent, 2, 0, TILE_H, ctx)
    var a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var b = cast_tensor(dec.decode(slice(r, 3, stride, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var c = cast_tensor(dec.decode(slice(r, 3, stride * 2, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var d = cast_tensor(dec.decode(slice(r, 3, stride * 3, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var e = cast_tensor(dec.decode(slice(r, 3, stride * 4, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row0 = _blend5(a, b, c, d, e, 3, ctx)

    r = slice(latent, 2, stride, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, stride, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, stride * 2, TILE_W, ctx), ctx), STDtype.F32, ctx)
    d = cast_tensor(dec.decode(slice(r, 3, stride * 3, TILE_W, ctx), ctx), STDtype.F32, ctx)
    e = cast_tensor(dec.decode(slice(r, 3, stride * 4, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row1 = _blend5(a, b, c, d, e, 3, ctx)

    r = slice(latent, 2, stride * 2, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, stride, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, stride * 2, TILE_W, ctx), ctx), STDtype.F32, ctx)
    d = cast_tensor(dec.decode(slice(r, 3, stride * 3, TILE_W, ctx), ctx), STDtype.F32, ctx)
    e = cast_tensor(dec.decode(slice(r, 3, stride * 4, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row2 = _blend5(a, b, c, d, e, 3, ctx)

    r = slice(latent, 2, stride * 3, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, stride, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, stride * 2, TILE_W, ctx), ctx), STDtype.F32, ctx)
    d = cast_tensor(dec.decode(slice(r, 3, stride * 3, TILE_W, ctx), ctx), STDtype.F32, ctx)
    e = cast_tensor(dec.decode(slice(r, 3, stride * 4, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row3 = _blend5(a, b, c, d, e, 3, ctx)

    r = slice(latent, 2, stride * 4, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, stride, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, stride * 2, TILE_W, ctx), ctx), STDtype.F32, ctx)
    d = cast_tensor(dec.decode(slice(r, 3, stride * 3, TILE_W, ctx), ctx), STDtype.F32, ctx)
    e = cast_tensor(dec.decode(slice(r, 3, stride * 4, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row4 = _blend5(a, b, c, d, e, 3, ctx)

    return _blend5(row0, row1, row2, row3, row4, 2, ctx)
