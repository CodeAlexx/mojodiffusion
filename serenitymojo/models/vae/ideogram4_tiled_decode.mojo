# ideogram4_tiled_decode.mojo — Ideogram-4 VAE tiled decode.
#
# The monolithic Ideogram 1024x1024 VAE decode can OOM after the fp8 DiT peak on
# a 24 GB GPU. Reuse the same 3x3 overlap + feathered blend pattern as SDXL,
# Z-Image, and FLUX: decode half-latent crops with a TILE-shaped decoder and
# blend in image space. The only model-specific parts are LATENT_CH=32 and the
# Ideogram-4 AutoencoderKLFlux2 decoder loader.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.ldm_decoder import LdmVaeDecoder, load_ideogram4_vae_decoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, mul, slice, concat


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


def _blend3(t0: Tensor, t1: Tensor, t2: Tensor, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var t = t0.shape()[dim]
    var s = t // 2
    var ov = t - s
    var a = slice(t0, dim, 0, s, ctx)
    var b = _xfade(slice(t0, dim, s, ov, ctx), slice(t1, dim, 0, ov, ctx), dim, ctx)
    var c = _xfade(slice(t1, dim, ov, ov, ctx), slice(t2, dim, 0, ov, ctx), dim, ctx)
    var d = slice(t2, dim, ov, s, ctx)
    return concat(dim, ctx, a, b, c, d)


def ideogram4_tiled_decode_with_decoder[
    LATENT_H: Int, LATENT_W: Int, TILE_H: Int, TILE_W: Int
](
    latent: Tensor, dec: LdmVaeDecoder[TILE_H, TILE_W, 32], ctx: DeviceContext
) raises -> Tensor:
    comptime assert TILE_H == LATENT_H // 2, "tile height must be half latent height"
    comptime assert TILE_W == LATENT_W // 2, "tile width must be half latent width"
    var half = TILE_H // 2

    var r = slice(latent, 2, 0, TILE_H, ctx)
    var a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row0 = _blend3(a, b, c, 3, ctx)

    r = slice(latent, 2, half, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row1 = _blend3(a, b, c, 3, ctx)

    r = slice(latent, 2, TILE_H, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, half, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, TILE_W, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row2 = _blend3(a, b, c, 3, ctx)

    return _blend3(row0, row1, row2, 2, ctx)


def ideogram4_tiled_decode[
    LATENT_H: Int, LATENT_W: Int
](latent: Tensor, vae_path: String, ctx: DeviceContext) raises -> Tensor:
    comptime TILE_H = LATENT_H // 2
    comptime TILE_W = LATENT_W // 2
    var dec = load_ideogram4_vae_decoder[TILE_H, TILE_W](vae_path, ctx)
    return ideogram4_tiled_decode_with_decoder[
        LATENT_H, LATENT_W, TILE_H, TILE_W
    ](latent, dec, ctx)
