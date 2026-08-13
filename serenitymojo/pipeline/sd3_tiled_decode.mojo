# sd3_tiled_decode.mojo -- low-memory SD3 VAE tiled decode for server workers.
#
# The monolithic 1024px embedded SD3 VAE decode is right on the 24 GB edge after
# a large-model all-admitted worker-swap run. This path decodes a 5x5 grid of
# quarter-latent rectangular tiles and feather-blends them to the exact requested
# canvas. Endpoint-balanced offsets handle dimensions whose quarter-tile stride
# is not integral (notably latent 168 and 104).

from std.collections import List
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.ldm_decoder import load_sd3_embedded_ldm_decoder
from serenitymojo.models.dit.sd3_contract import sd3_lowmem_tile_start
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, mul, slice, concat
from serenitymojo.tensor import Tensor


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


def _stitch_at(
    left: Tensor, right: Tensor, dim: Int, start: Int, ctx: DeviceContext
) raises -> Tensor:
    """Append an equal-sized tile at an explicit output offset with overlap fade."""
    var left_len = left.shape()[dim]
    var tile_len = right.shape()[dim]
    var overlap = left_len - start
    if start <= 0 or start >= left_len or overlap <= 0 or overlap >= tile_len:
        raise Error("sd3 tiled decode: invalid stitch offset")
    var prefix = slice(left, dim, 0, start, ctx)
    var blended = _xfade(
        slice(left, dim, start, overlap, ctx),
        slice(right, dim, 0, overlap, ctx),
        dim,
        ctx,
    )
    var tail = slice(right, dim, overlap, tile_len - overlap, ctx)
    return concat(dim, ctx, prefix, blended, tail)


def _stitch5(
    t0: Tensor, t1: Tensor, t2: Tensor, t3: Tensor, t4: Tensor,
    dim: Int, start1: Int, start2: Int, start3: Int, start4: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var out = _stitch_at(t0, t1, dim, start1, ctx)
    out = _stitch_at(out, t2, dim, start2, ctx)
    out = _stitch_at(out, t3, dim, start3, ctx)
    return _stitch_at(out, t4, dim, start4, ctx)


def sd3_tiled_decode_5x5_lowmem[
    LATENT_H: Int, LATENT_W: Int
](latent: Tensor, model_path: String, ctx: DeviceContext) raises -> Tensor:
    comptime TILE_H = LATENT_H // 4
    comptime TILE_W = LATENT_W // 4
    comptime assert LATENT_H % 4 == 0, "SD3 tiled latent H must divide by four"
    comptime assert LATENT_W % 4 == 0, "SD3 tiled latent W must divide by four"
    var dec = load_sd3_embedded_ldm_decoder[TILE_H, TILE_W](model_path, ctx)
    var h1 = sd3_lowmem_tile_start(LATENT_H, TILE_H, 1)
    var h2 = sd3_lowmem_tile_start(LATENT_H, TILE_H, 2)
    var h3 = sd3_lowmem_tile_start(LATENT_H, TILE_H, 3)
    var h4 = sd3_lowmem_tile_start(LATENT_H, TILE_H, 4)
    var w1 = sd3_lowmem_tile_start(LATENT_W, TILE_W, 1)
    var w2 = sd3_lowmem_tile_start(LATENT_W, TILE_W, 2)
    var w3 = sd3_lowmem_tile_start(LATENT_W, TILE_W, 3)
    var w4 = sd3_lowmem_tile_start(LATENT_W, TILE_W, 4)
    # Decode output is 8x latent, so stitch offsets are scaled to image pixels.
    var oh1 = h1 * 8
    var oh2 = h2 * 8
    var oh3 = h3 * 8
    var oh4 = h4 * 8
    var ow1 = w1 * 8
    var ow2 = w2 * 8
    var ow3 = w3 * 8
    var ow4 = w4 * 8

    var r = slice(latent, 2, 0, TILE_H, ctx)
    var a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var b = cast_tensor(dec.decode(slice(r, 3, w1, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var c = cast_tensor(dec.decode(slice(r, 3, w2, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var d = cast_tensor(dec.decode(slice(r, 3, w3, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var e = cast_tensor(dec.decode(slice(r, 3, w4, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row0 = _stitch5(a, b, c, d, e, 3, ow1, ow2, ow3, ow4, ctx)

    r = slice(latent, 2, h1, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, w1, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, w2, TILE_W, ctx), ctx), STDtype.F32, ctx)
    d = cast_tensor(dec.decode(slice(r, 3, w3, TILE_W, ctx), ctx), STDtype.F32, ctx)
    e = cast_tensor(dec.decode(slice(r, 3, w4, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row1 = _stitch5(a, b, c, d, e, 3, ow1, ow2, ow3, ow4, ctx)

    r = slice(latent, 2, h2, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, w1, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, w2, TILE_W, ctx), ctx), STDtype.F32, ctx)
    d = cast_tensor(dec.decode(slice(r, 3, w3, TILE_W, ctx), ctx), STDtype.F32, ctx)
    e = cast_tensor(dec.decode(slice(r, 3, w4, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row2 = _stitch5(a, b, c, d, e, 3, ow1, ow2, ow3, ow4, ctx)

    r = slice(latent, 2, h3, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, w1, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, w2, TILE_W, ctx), ctx), STDtype.F32, ctx)
    d = cast_tensor(dec.decode(slice(r, 3, w3, TILE_W, ctx), ctx), STDtype.F32, ctx)
    e = cast_tensor(dec.decode(slice(r, 3, w4, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row3 = _stitch5(a, b, c, d, e, 3, ow1, ow2, ow3, ow4, ctx)

    r = slice(latent, 2, h4, TILE_H, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE_W, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, w1, TILE_W, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, w2, TILE_W, ctx), ctx), STDtype.F32, ctx)
    d = cast_tensor(dec.decode(slice(r, 3, w3, TILE_W, ctx), ctx), STDtype.F32, ctx)
    e = cast_tensor(dec.decode(slice(r, 3, w4, TILE_W, ctx), ctx), STDtype.F32, ctx)
    var row4 = _stitch5(a, b, c, d, e, 3, ow1, ow2, ow3, ow4, ctx)

    return _stitch5(row0, row1, row2, row3, row4, 2, oh1, oh2, oh3, oh4, ctx)
