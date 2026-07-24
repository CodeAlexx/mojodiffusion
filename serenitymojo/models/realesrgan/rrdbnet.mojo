# models/realesrgan/rrdbnet.mojo — Real-ESRGAN RRDBNet x4plus, pure Mojo.
# Verified bit-accurate to the torch reference (rrdbnet_parity.mojo:
# cos=0.99999999999977, max_abs=1.37e-6). Fixed 128x128 NHWC tile in ->
# 512x512 NHWC out (conv2d shapes are compile-time-static). The CLI tiles
# arbitrary images through this with overlap context; see realesrgan_cli.mojo.
#
# Arch: nf=64, nb=23 RRDB, grow=32, scale=4, no pixel_unshuffle.
#   RDB  = 5 dense convs + leaky_relu(0.2); x5*0.2 + x
#   RRDB = 3 RDB;  out*0.2 + x
#   trunk: feat + conv_body(body(feat))            (no scale)
#   tail:  2x (nearest2x + conv + lrelu), conv_hr(lrelu), conv_last

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.activations import leaky_relu
from serenitymojo.ops.tensor_algebra import concat, add, mul_scalar
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc
from serenitymojo.models.sdxl.real_weights import load_conv_rscf, load_bias

comptime TILE = 128       # fixed conv input tile (H=W)
comptime SCALE = 4        # x4 upscale
comptime OUT = TILE * SCALE  # 512


@fieldwise_init
struct RRDBNetWeights(Movable):
    var cf_w: Tensor
    var cf_b: Tensor
    var cb_w: Tensor
    var cb_b: Tensor
    var u1_w: Tensor
    var u1_b: Tensor
    var u2_w: Tensor
    var u2_b: Tensor
    var hr_w: Tensor
    var hr_b: Tensor
    var lt_w: Tensor
    var lt_b: Tensor
    var bw: List[ArcPointer[Tensor]]   # 345 body conv weights (RSCF)
    var bb: List[ArcPointer[Tensor]]   # 345 body conv biases


def load_rrdbnet(path: String, ctx: DeviceContext) raises -> RRDBNetWeights:
    var st = SafeTensors.open(path)
    var bw = List[ArcPointer[Tensor]]()
    var bb = List[ArcPointer[Tensor]]()
    for blk in range(23):
        for rdb in range(1, 4):
            for cv in range(1, 6):
                var p = String("body.") + String(blk) + ".rdb" + String(rdb) + ".conv" + String(cv)
                bw.append(ArcPointer[Tensor](load_conv_rscf(st, p + ".weight", ctx)))
                bb.append(ArcPointer[Tensor](load_bias(st, p + ".bias", ctx)))
    return RRDBNetWeights(
        load_conv_rscf(st, "conv_first.weight", ctx), load_bias(st, "conv_first.bias", ctx),
        load_conv_rscf(st, "conv_body.weight", ctx),  load_bias(st, "conv_body.bias", ctx),
        load_conv_rscf(st, "conv_up1.weight", ctx),   load_bias(st, "conv_up1.bias", ctx),
        load_conv_rscf(st, "conv_up2.weight", ctx),   load_bias(st, "conv_up2.bias", ctx),
        load_conv_rscf(st, "conv_hr.weight", ctx),    load_bias(st, "conv_hr.bias", ctx),
        load_conv_rscf(st, "conv_last.weight", ctx),  load_bias(st, "conv_last.bias", ctx),
        bw^, bb^,
    )


# 3x3 stride-1 pad-1 conv (NHWC/RSCF) + bias. Bias cloned into the Optional;
# weight reused across tiles so passed by borrow.
def _c3[H: Int, W: Int, Cin: Int, Cout: Int](
    x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Tensor:
    return conv2d[1, H, W, Cin, 3, 3, Cout, 1, 1, 1, 1](
        x, w, Optional[Tensor](b.clone(ctx)), ctx
    )


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


def _rrdb(
    x: Tensor, w: List[ArcPointer[Tensor]], b: List[ArcPointer[Tensor]], off: Int, ctx: DeviceContext
) raises -> Tensor:
    var o = _rdb(x, w, b, off + 0, ctx)
    o = _rdb(o, w, b, off + 5, ctx)
    o = _rdb(o, w, b, off + 10, ctx)
    return add(x, mul_scalar(o, Float32(0.2), ctx), ctx)


# One fixed 128x128x3 NHWC tile (unit [0,1] RGB) -> 512x512x3 NHWC.
def rrdbnet_forward(w: RRDBNetWeights, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var feat = _c3[128, 128, 3, 64](x, w.cf_w, w.cf_b, ctx)
    var body = feat.clone(ctx)
    for blk in range(23):
        body = _rrdb(body, w.bw, w.bb, blk * 15, ctx)
    var body_out = _c3[128, 128, 64, 64](body, w.cb_w, w.cb_b, ctx)
    feat = add(feat, body_out, ctx)                              # trunk residual (no scale)

    var u = upsample_nearest2x_nhwc(feat, ctx)                   # [1,256,256,64]
    u = leaky_relu(_c3[256, 256, 64, 64](u, w.u1_w, w.u1_b, ctx), ctx)
    u = upsample_nearest2x_nhwc(u, ctx)                          # [1,512,512,64]
    u = leaky_relu(_c3[512, 512, 64, 64](u, w.u2_w, w.u2_b, ctx), ctx)
    u = leaky_relu(_c3[512, 512, 64, 64](u, w.hr_w, w.hr_b, ctx), ctx)
    return _c3[512, 512, 64, 3](u, w.lt_w, w.lt_b, ctx)         # [1,512,512,3] NHWC
