# sd3_tiled_decode.mojo -- low-memory SD3 VAE tiled decode for server workers.
#
# The monolithic 1024px embedded SD3 VAE decode is right on the 24 GB edge after
# a large-model all-admitted worker-swap run. This path decodes 5x5 overlapping
# 256px image tiles and feather-blends them into the same 1024px output shape.

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.ldm_decoder import load_sd3_embedded_ldm_decoder
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


def sd3_tiled_decode_5x5_lowmem[
    LATENT_H: Int, LATENT_W: Int
](latent: Tensor, model_path: String, ctx: DeviceContext) raises -> Tensor:
    comptime TILE_H = LATENT_H // 4
    comptime TILE_W = LATENT_W // 4
    var dec = load_sd3_embedded_ldm_decoder[TILE_H, TILE_W](model_path, ctx)
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
