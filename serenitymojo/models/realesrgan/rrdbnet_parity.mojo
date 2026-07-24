# models/realesrgan/rrdbnet_parity.mojo — STEP 3 parity gate for the pure-Mojo
# RRDBNet x4plus forward. Loads the converted safetensors weights, runs a fixed
# 128x128x3 tile, and compares the 512x512x3 output against the torch oracle
# (oracle_convert.py). Bar: cos >= 0.999 vs oracle_out, on the F32 path.
#
# Arch (xinntao Real-ESRGAN, x4plus): nf=64, nb=23 RRDB, grow=32, scale=4, no
# pixel_unshuffle. RDB = 5 dense convs + leaky_relu(0.2); x5*0.2 + x. RRDB = 3
# RDB; out*0.2 + x. Trunk: feat + conv_body(body(feat)) (no scale). Upsample:
# 2x (nearest2x + conv + lrelu), then conv_hr(lrelu) + conv_last.
# NHWC throughout (conv2d is NHWC/RSCF); channel-concat is dim=3.
#
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/realesrgan/rrdbnet_parity.mojo -o /tmp/rrdbnet_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/rrdbnet_parity

from std.math import sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.activations import leaky_relu
from serenitymojo.ops.tensor_algebra import concat, add, mul_scalar
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc
from serenitymojo.models.vae.decoder2d import nchw_to_nhwc, nhwc_to_nchw
from serenitymojo.models.sdxl.real_weights import load_conv_rscf, load_bias

comptime MODEL = "/home/alex/models/realesrgan/realesrgan_x4plus.safetensors"
comptime ORACLE = "/home/alex/models/realesrgan/oracle_io.safetensors"


# 3x3, stride 1, pad 1 conv (NHWC/RSCF) with bias. Weight reused across tiles, so
# the bias is cloned into the Optional (conv2d takes weight by borrow).
def _c3[H: Int, W: Int, Cin: Int, Cout: Int](
    x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Tensor:
    return conv2d[1, H, W, Cin, 3, 3, Cout, 1, 1, 1, 1](
        x, w, Optional[Tensor](b.clone(ctx)), ctx
    )


# Residual Dense Block at the 128x128 trunk resolution. `off` = base index into
# the flat weight/bias lists (5 convs per RDB).
def _rdb(
    x: Tensor, w: List[ArcPointer[Tensor]], b: List[ArcPointer[Tensor]], off: Int, ctx: DeviceContext
) raises -> Tensor:
    var x1 = leaky_relu(_c3[128, 128, 64, 32](x, w[off + 0][], b[off + 0][], ctx), ctx)
    var c2 = concat(3, ctx, x, x1)
    var x2 = leaky_relu(_c3[128, 128, 96, 32](c2, w[off + 1][], b[off + 1][], ctx), ctx)
    var c3 = concat(3, ctx, x, x1, x2)
    var x3 = leaky_relu(_c3[128, 128, 128, 32](c3, w[off + 2][], b[off + 2][], ctx), ctx)
    var c4 = concat(3, ctx, x, x1, x2, x3)
    var x4 = leaky_relu(_c3[128, 128, 160, 32](c4, w[off + 3][], b[off + 3][], ctx), ctx)
    var c5 = concat(3, ctx, x, x1, x2, x3, x4)
    var x5 = _c3[128, 128, 192, 64](c5, w[off + 4][], b[off + 4][], ctx)
    return add(x, mul_scalar(x5, Float32(0.2), ctx), ctx)


# Residual-in-Residual Dense Block: 3 RDBs (15 convs), out*0.2 + x.
def _rrdb(
    x: Tensor, w: List[ArcPointer[Tensor]], b: List[ArcPointer[Tensor]], off: Int, ctx: DeviceContext
) raises -> Tensor:
    var o = _rdb(x, w, b, off + 0, ctx)
    o = _rdb(o, w, b, off + 5, ctx)
    o = _rdb(o, w, b, off + 10, ctx)
    return add(x, mul_scalar(o, Float32(0.2), ctx), ctx)


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

    # ── singleton conv weights/biases ──
    var cf_w = load_conv_rscf(st, "conv_first.weight", ctx)
    var cf_b = load_bias(st, "conv_first.bias", ctx)
    var cb_w = load_conv_rscf(st, "conv_body.weight", ctx)
    var cb_b = load_bias(st, "conv_body.bias", ctx)
    var u1_w = load_conv_rscf(st, "conv_up1.weight", ctx)
    var u1_b = load_bias(st, "conv_up1.bias", ctx)
    var u2_w = load_conv_rscf(st, "conv_up2.weight", ctx)
    var u2_b = load_bias(st, "conv_up2.bias", ctx)
    var hr_w = load_conv_rscf(st, "conv_hr.weight", ctx)
    var hr_b = load_bias(st, "conv_hr.bias", ctx)
    var lt_w = load_conv_rscf(st, "conv_last.weight", ctx)
    var lt_b = load_bias(st, "conv_last.bias", ctx)

    # ── body: 23 RRDB × 3 RDB × 5 conv = 345 conv weights/biases (flat) ──
    var bw = List[ArcPointer[Tensor]]()
    var bb = List[ArcPointer[Tensor]]()
    for blk in range(23):
        for rdb in range(1, 4):
            for cv in range(1, 6):
                var p = String("body.") + String(blk) + ".rdb" + String(rdb) + ".conv" + String(cv)
                bw.append(ArcPointer[Tensor](load_conv_rscf(st, p + ".weight", ctx)))
                bb.append(ArcPointer[Tensor](load_bias(st, p + ".bias", ctx)))
    print("loaded weights: body convs =", len(bw))

    # ── oracle input/reference ──
    var io = SafeTensors.open(ORACLE)
    var inp = load_bias(io, "inp", ctx)      # [1,3,128,128] NCHW F32
    var ref_out = load_bias(io, "out", ctx)  # [1,3,512,512] NCHW F32
    var x = nchw_to_nhwc(inp, ctx)           # [1,128,128,3]

    # ── forward ──
    var feat = _c3[128, 128, 3, 64](x, cf_w, cf_b, ctx)
    var body = feat.clone(ctx)
    for blk in range(23):
        body = _rrdb(body, bw, bb, blk * 15, ctx)
    var body_out = _c3[128, 128, 64, 64](body, cb_w, cb_b, ctx)
    feat = add(feat, body_out, ctx)                                   # trunk residual (no scale)

    var u = upsample_nearest2x_nhwc(feat, ctx)                        # [1,256,256,64]
    u = leaky_relu(_c3[256, 256, 64, 64](u, u1_w, u1_b, ctx), ctx)
    u = upsample_nearest2x_nhwc(u, ctx)                               # [1,512,512,64]
    u = leaky_relu(_c3[512, 512, 64, 64](u, u2_w, u2_b, ctx), ctx)
    u = leaky_relu(_c3[512, 512, 64, 64](u, hr_w, hr_b, ctx), ctx)
    var out = _c3[512, 512, 64, 3](u, lt_w, lt_b, ctx)               # [1,512,512,3] NHWC

    var out_nchw = nhwc_to_nchw(out, ctx)                             # [1,3,512,512]
    var got = out_nchw.to_host(ctx)
    var exp = ref_out.to_host(ctx)

    var cos = _cos(got, exp)
    var mx = _maxabs(got, exp)
    var ok = cos >= Float64(0.999)
    print("elems:", len(got), "(exp", len(exp), ")")
    print("cos =", cos, " max_abs =", mx)
    print("RESULT:", "PASS" if ok else "FAIL", " (bar cos>=0.999)")
