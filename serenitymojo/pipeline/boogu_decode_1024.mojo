# boogu_decode_1024.mojo — TILED VAE decode of the dumped Boogu 1024 final latent.
# The monolithic 1024 decode OOMs (conv im2col at 1024x1024 + mid-attn over 16384
# positions > 24GB). Mirror the proven ideogram4/flux/zimage tiled pattern: decode
# 9 overlapping TILE=64x64 latent crops (each -> 512x512, ~2-3GB, the size that
# already worked at 256) and feathered-blend in image space. Blend helpers reused
# from ideogram4_tiled_decode (VAE-agnostic). Run in a fresh process:
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/pipeline/boogu_decode_1024.mojo
from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder
from serenitymojo.models.vae.ideogram4_tiled_decode import _blend3
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.image.png import save_png, ValueRange

comptime LATENT_BIN = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_dumps/c8_final_latent_1024_mojo.bin"
comptime VAE_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/vae"
comptime OUT_PNG = "/home/alex/mojodiffusion/output/boogu_t2i_1024_mojo.png"
comptime LAT = 128          # latent H=W
comptime TILE = 64          # = LAT // 2
comptime HALF = 32          # = TILE // 2


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(n // 4):
        out.append(fp[i])
    buf.free()
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("=== Boogu 1024 TILED VAE decode (9x 64x64 -> 512x512, blended) ===")
    var lat_h = _read_bin_f32(String(LATENT_BIN))
    if len(lat_h) != 16 * LAT * LAT:
        raise Error("latent numel " + String(len(lat_h)))
    var latent = Tensor.from_host(lat_h, [1, 16, LAT, LAT], STDtype.F32, ctx)
    var dec = ZImageDecoder[TILE, TILE].load(String(VAE_DIR), ctx)

    # 3 row-bands (rows 0/HALF/TILE) x 3 col-crops (cols 0/HALF/TILE), each TILE^2.
    var r = slice(latent, 2, 0, TILE, ctx)
    var a = cast_tensor(dec.decode(slice(r, 3, 0, TILE, ctx), ctx), STDtype.F32, ctx)
    var b = cast_tensor(dec.decode(slice(r, 3, HALF, TILE, ctx), ctx), STDtype.F32, ctx)
    var c = cast_tensor(dec.decode(slice(r, 3, TILE, TILE, ctx), ctx), STDtype.F32, ctx)
    var row0 = _blend3(a, b, c, 3, ctx)

    r = slice(latent, 2, HALF, TILE, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, HALF, TILE, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, TILE, TILE, ctx), ctx), STDtype.F32, ctx)
    var row1 = _blend3(a, b, c, 3, ctx)

    r = slice(latent, 2, TILE, TILE, ctx)
    a = cast_tensor(dec.decode(slice(r, 3, 0, TILE, ctx), ctx), STDtype.F32, ctx)
    b = cast_tensor(dec.decode(slice(r, 3, HALF, TILE, ctx), ctx), STDtype.F32, ctx)
    c = cast_tensor(dec.decode(slice(r, 3, TILE, TILE, ctx), ctx), STDtype.F32, ctx)
    var row2 = _blend3(a, b, c, 3, ctx)

    var img = _blend3(row0, row1, row2, 2, ctx)   # [1,3,1024,1024]
    var sh = img.shape()
    print("  decoded image [", sh[0], ",", sh[1], ",", sh[2], ",", sh[3], "]")
    save_png(img, String(OUT_PNG), ctx, ValueRange.SIGNED)
    print("  wrote", OUT_PNG)
