# zimage_tiled_decode.mojo — Z-Image VAE tiled decode (3x3 overlap + feathered
# blend), the FLUX precedent (flux_tiled_decode.mojo) ported to the Z-Image VAE
# decoder so a 1024² (128-latent) decode fits beside the resident DiT (F2).
#
# THE PROBLEM (Phase-4 F2). A whole-frame 128-latent ZImageDecoder[128,128]
# decode peaks ~23.6 GB; beside the resident DiT (13.2 GB) on a 24 GB GPU it
# OOMs. Each 64-latent crop (ZImageDecoder[64,64]) is a small alloc that fits.
# Crops sit at latent stride 32 (positions 0/32/64) so adjacent image tiles
# overlap by half a tile; a separable feathered cross-fade (horizontal per row,
# then vertical) erases the conv-boundary seams. Tile vars are reassigned per
# row so prior tiles free → retained-memory peak stays near the single-tile
# working set.
#
# Blend math (feather weights, _xfade, _blend3) is byte-identical to
# flux_tiled_decode — it is VAE-agnostic (operates on decoded image tensors);
# only the decoder instance differs (ZImageDecoder vs the FLUX LDM decoder).

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, mul, slice, concat


# ── feathered cross-fade weight (ramps 1→0 or 0→1 along `dim`), F32 ──────────
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


# ── tiled Z-Image VAE decode — 3x3 OVERLAPPING latent crops + feathered blend ─
# latent NCHW [1,16,LATENT_H,LATENT_W] -> image NCHW [1,3,8*LATENT_H,8*LATENT_W].
# Decoder is instantiated once at the TILE shape (LATENT/2) and reused for all 9
# crops at latent stride TILE/2.
def zimage_tiled_decode[
    LATENT_H: Int, LATENT_W: Int
](latent: Tensor, vae_dir: String, ctx: DeviceContext) raises -> Tensor:
    comptime TILE_H = LATENT_H // 2
    comptime TILE_W = LATENT_W // 2
    var dec = ZImageDecoder[TILE_H, TILE_W].load(vae_dir, ctx)
    var half = TILE_H // 2                          # latent stride
    # row 0 crop [0:TILE_H], blend its 3 columns
    var r = slice(latent, 2, 0, TILE_H, ctx)
    var a = cast_tensor(dec.decode(cast_tensor(slice(r, 3, 0, TILE_W, ctx), STDtype.BF16, ctx), ctx), STDtype.F32, ctx)
    var b = cast_tensor(dec.decode(cast_tensor(slice(r, 3, half, TILE_W, ctx), STDtype.BF16, ctx), ctx), STDtype.F32, ctx)
    var c = cast_tensor(dec.decode(cast_tensor(slice(r, 3, TILE_W, TILE_W, ctx), STDtype.BF16, ctx), ctx), STDtype.F32, ctx)
    var row0 = _blend3(a, b, c, 3, ctx)
    # row 1 crop [half:half+TILE_H] (reassign a/b/c → prior tiles freed)
    r = slice(latent, 2, half, TILE_H, ctx)
    a = cast_tensor(dec.decode(cast_tensor(slice(r, 3, 0, TILE_W, ctx), STDtype.BF16, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(cast_tensor(slice(r, 3, half, TILE_W, ctx), STDtype.BF16, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(cast_tensor(slice(r, 3, TILE_W, TILE_W, ctx), STDtype.BF16, ctx), ctx), STDtype.F32, ctx)
    var row1 = _blend3(a, b, c, 3, ctx)
    # row 2 crop [TILE_H:2*TILE_H]
    r = slice(latent, 2, TILE_H, TILE_H, ctx)
    a = cast_tensor(dec.decode(cast_tensor(slice(r, 3, 0, TILE_W, ctx), STDtype.BF16, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(cast_tensor(slice(r, 3, half, TILE_W, ctx), STDtype.BF16, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(cast_tensor(slice(r, 3, TILE_W, TILE_W, ctx), STDtype.BF16, ctx), ctx), STDtype.F32, ctx)
    var row2 = _blend3(a, b, c, 3, ctx)
    # vertical feathered blend of the 3 full-width rows
    return _blend3(row0, row1, row2, 2, ctx)
