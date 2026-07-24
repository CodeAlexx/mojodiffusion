# models/realesrgan/srvggnet_parity.mojo — parity gate for the pure-Mojo
# SRVGGNetCompact (realesr-general-x4v3) x4 forward. Loads the converted
# safetensors, runs the deterministic 128x128x3 fixture, and compares the
# 512x512x3 output against the torch oracle (srvgg/oracle_convert.py).
# Bar: cos >= 0.999 vs oracle_out, on the F32 path.
#
# Arch (SRVGGNetCompact, x4v3): num_feat=64, num_conv=32, upscale=4, act=PReLU.
#   body = conv(3->64) + PReLU, then 32x [conv(64->64) + PReLU], then conv(64->48).
#   tail = PixelShuffle(4) on the 48ch map, + nearest-4x(input) skip.
# NHWC throughout the convs (conv2d is NHWC/RSCF); PixelShuffle runs in NCHW.
#
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/realesrgan/srvggnet_parity.mojo -o /tmp/srvggnet_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/srvggnet_parity

from std.math import sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.activations import prelu
from serenitymojo.ops.tensor_algebra import add
from serenitymojo.ops.pixelshuffle import pixel_shuffle
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc
from serenitymojo.models.vae.decoder2d import nchw_to_nhwc, nhwc_to_nchw
from serenitymojo.models.sdxl.real_weights import load_conv_rscf, load_bias

comptime MODEL = "/home/alex/models/srvgg/srvgg_x4v3.safetensors"
comptime ORACLE = "/home/alex/models/srvgg/oracle_io.safetensors"
comptime NUM_CONV = 32


def _c3[H: Int, W: Int, Cin: Int, Cout: Int](
    x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Tensor:
    return conv2d[1, H, W, Cin, 3, 3, Cout, 1, 1, 1, 1](
        x, w, Optional[Tensor](b.clone(ctx)), ctx
    )


def _cos(a: List[Float32], e: List[Float32]) -> Float64:
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var ne = Float64(0.0)
    for i in range(len(a)):
        var av = Float64(a[i])
        var ev = Float64(e[i])
        dot += av * ev
        na += av * av
        ne += ev * ev
    return dot / (sqrt(na) * sqrt(ne) + Float64(1e-12))


def _maxabs(a: List[Float32], e: List[Float32]) -> Float32:
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - e[i]
        if d < Float32(0.0):
            d = -d
        if d > m:
            m = d
    return m


def main() raises:
    var ctx = DeviceContext()
    var st = SafeTensors.open(MODEL)

    # conv weights/biases at body indices 0,2,...,66 (34 convs); prelu alphas at
    # body indices 1,3,...,65 (33). Stored flat in application order.
    var cw = List[ArcPointer[Tensor]]()
    var cb = List[ArcPointer[Tensor]]()
    var pw = List[ArcPointer[Tensor]]()
    var n_conv = NUM_CONV + 2                      # first + mid + last = 34
    for c in range(n_conv):
        var i = 2 * c                              # body index of this conv
        var p = String("body.") + String(i)
        cw.append(ArcPointer[Tensor](load_conv_rscf(st, p + ".weight", ctx)))
        cb.append(ArcPointer[Tensor](load_bias(st, p + ".bias", ctx)))
    for c in range(NUM_CONV + 1):                  # 33 prelus
        var i = 2 * c + 1
        var p = String("body.") + String(i) + ".weight"
        pw.append(ArcPointer[Tensor](load_bias(st, p, ctx)))   # alpha [64] F32
    print("loaded convs =", len(cw), " prelus =", len(pw))

    # oracle input/reference
    var io = SafeTensors.open(ORACLE)
    var inp = load_bias(io, "inp", ctx)            # [1,3,128,128] NCHW F32
    var ref_out = load_bias(io, "out", ctx)        # [1,3,512,512] NCHW F32
    var x = nchw_to_nhwc(inp, ctx)                 # [1,128,128,3]

    # forward — body
    var out = _c3[128, 128, 3, 64](x, cw[0][], cb[0][], ctx)
    out = prelu(out, pw[0][], ctx)
    for k in range(NUM_CONV):
        out = _c3[128, 128, 64, 64](out, cw[1 + k][], cb[1 + k][], ctx)
        out = prelu(out, pw[1 + k][], ctx)
    out = _c3[128, 128, 64, 48](out, cw[n_conv - 1][], cb[n_conv - 1][], ctx)   # last conv

    # tail — PixelShuffle(4) in NCHW, then nearest-4x input skip
    var ps = pixel_shuffle(nhwc_to_nchw(out, ctx), 4, ctx)     # [1,3,512,512] NCHW
    var ps_nhwc = nchw_to_nhwc(ps, ctx)                        # [1,512,512,3]
    var base = upsample_nearest2x_nhwc(upsample_nearest2x_nhwc(x, ctx), ctx)  # [1,512,512,3]
    var res = add(ps_nhwc, base, ctx)                         # NHWC

    var out_nchw = nhwc_to_nchw(res, ctx)                     # [1,3,512,512]
    var got = out_nchw.to_host(ctx)
    var exp = ref_out.to_host(ctx)

    var cos = _cos(got, exp)
    var mx = _maxabs(got, exp)
    var ok = cos >= Float64(0.999)
    print("elems:", len(got), "(exp", len(exp), ")")
    print("cos =", cos, " max_abs =", mx)
    print("RESULT:", "PASS" if ok else "FAIL", " (bar cos>=0.999)")
