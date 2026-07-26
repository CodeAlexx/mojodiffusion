# sampling/realesrgan_fast_cli.mojo — Real-ESRGAN x4 FAST upscaler CLI (pure Mojo).
# Uses SRVGGNetCompact (realesr-general-x4v3): ~34 convs vs RRDBNet's 345, so it
# is markedly faster (real-time-ish) at slightly softer quality — the "fast path".
# Reads a PNG/JPEG/WebP, upscales x4, writes a PNG.
#   argv[1] input image          (default: demo)
#   argv[2] output PNG           (default: /tmp/realesrgan_fast_out.png)
#   argv[3] weights .safetensors (default: models/srvgg/srvgg_x4v3.safetensors)
#
# Tiling is identical to the RRDBNet CLI: fixed 128x128 -> 512x512 tile driven
# with a 96px core + 16px context per side; interior tiles see real neighbour
# pixels (seam-free after core crop), true borders use edge-replicate.
#
#   pixi run mojo build -I . -I vendor/mojo-libs -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -lpng16 -Xlinker -lturbojpeg \
#     serenitymojo/sampling/realesrgan_fast_cli.mojo -o /tmp/realesrgan_fast_cli
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/realesrgan_fast_cli in.png out.png

from std.sys import argv
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.serve.image_io import decode_image_any, image_to_signed_nchw
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.realesrgan.srvggnet import load_srvggnet, srvggnet_forward

comptime TILE = 128
comptime PAD = 16
comptime CORE = TILE - 2 * PAD    # 96
comptime SCALE = 4
comptime OTILE = TILE * SCALE     # 512
comptime OPAD = PAD * SCALE       # 64
comptime OCORE = CORE * SCALE     # 384


def _clampi(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _clamp01(v: Float32) -> Float32:
    if v < Float32(0.0):
        return Float32(0.0)
    if v > Float32(1.0):
        return Float32(1.0)
    return v


def main() raises:
    var a = argv()
    var in_path = String(a[1]) if len(a) > 1 else String(
        "/home/alex/models/realesrgan/demo_in.png")
    var out_path = String(a[2]) if len(a) > 2 else String("/tmp/realesrgan_fast_out.png")
    var model = String(a[3]) if len(a) > 3 else String(
        "/home/alex/models/srvgg/srvgg_x4v3.safetensors")

    var ctx = DeviceContext()
    print("[realesrgan-fast] loading weights:", model)
    var w = load_srvggnet(model, ctx)

    # ── decode → unit [0,1] CHW host buffer ──
    var img = decode_image_any(in_path)
    var H = img.height
    var W = img.width
    var plane = H * W
    var signed = image_to_signed_nchw(img)              # [1,3,H,W] in [-1,1], CHW flat
    var src = List[Float32](capacity=3 * plane)
    for i in range(3 * plane):
        src.append((signed[i] + Float32(1.0)) * Float32(0.5))   # [-1,1] → [0,1]
    print("[realesrgan-fast] input", W, "x", H, "→ output", W * SCALE, "x", H * SCALE)

    # ── output canvas (unit CHW) ──
    var OW = W * SCALE
    var OH = H * SCALE
    var oplane = OH * OW
    var dst = List[Float32](capacity=3 * oplane)
    for _ in range(3 * oplane):
        dst.append(Float32(0.0))

    var n_ty = (H + CORE - 1) // CORE
    var n_tx = (W + CORE - 1) // CORE
    var tiles = n_ty * n_tx
    print("[realesrgan-fast] tiles:", tiles, "(", n_ty, "x", n_tx, ")")

    var tile = List[Float32](capacity=TILE * TILE * 3)
    for _ in range(TILE * TILE * 3):
        tile.append(Float32(0.0))

    var done = 0
    for ty in range(n_ty):
        var ry0 = ty * CORE
        for tx in range(n_tx):
            var cx0 = tx * CORE
            for h in range(TILE):
                var sy = _clampi(ry0 - PAD + h, 0, H - 1)
                for ww in range(TILE):
                    var sx = _clampi(cx0 - PAD + ww, 0, W - 1)
                    var t0 = (h * TILE + ww) * 3
                    tile[t0 + 0] = src[0 * plane + sy * W + sx]
                    tile[t0 + 1] = src[1 * plane + sy * W + sx]
                    tile[t0 + 2] = src[2 * plane + sy * W + sx]
            var xt = Tensor.from_host(tile, [1, TILE, TILE, 3], STDtype.F32, ctx)
            var out_tensor = srvggnet_forward(w, xt, ctx)  # [1,512,512,3] NHWC
            var oh = out_tensor.to_host(ctx)

            for r in range(OCORE):
                var oy = ry0 * SCALE + r
                if oy >= OH:
                    break
                for c in range(OCORE):
                    var ox = cx0 * SCALE + c
                    if ox >= OW:
                        break
                    var s0 = ((r + OPAD) * OTILE + (c + OPAD)) * 3
                    dst[0 * oplane + oy * OW + ox] = _clamp01(oh[s0 + 0])
                    dst[1 * oplane + oy * OW + ox] = _clamp01(oh[s0 + 1])
                    dst[2 * oplane + oy * OW + ox] = _clamp01(oh[s0 + 2])
            done += 1
        print("[realesrgan-fast] row", ty + 1, "/", n_ty, "done (", done, "/", tiles, "tiles)")

    var out_t = Tensor.from_host(dst, [1, 3, OH, OW], STDtype.F32, ctx)
    save_png(out_t, out_path, ctx, ValueRange.UNIT)
    print("[realesrgan-fast] wrote", out_path, "(", OW, "x", OH, ")")
