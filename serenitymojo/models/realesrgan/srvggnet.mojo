# models/realesrgan/srvggnet.mojo — Real-ESRGAN SRVGGNetCompact x4 (the fast /
# real-time "compact" model, e.g. realesr-general-x4v3). Pure Mojo + MAX.
# Verified bit-accurate to torch (srvggnet_parity.mojo: cos=0.99999999999892,
# max_abs=5.4e-6). Fixed 128x128 NHWC tile in -> 512x512 NHWC out (conv2d shapes
# are compile-time-static). The CLI tiles arbitrary images through this.
#
# Arch: num_feat=64, num_conv=32, upscale=4, act=PReLU (per-channel).
#   body = conv(3->64) + PReLU, then 32x [conv(64->64) + PReLU], then conv(64->48)
#   tail = PixelShuffle(4) on the 48ch map, + nearest-4x(input) skip
# ~34 convs total vs RRDBNet's 345 -> markedly faster, slightly softer output.

from std.memory import ArcPointer
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.activations import prelu
from serenitymojo.ops.tensor_algebra import add
from serenitymojo.ops.pixelshuffle import pixel_shuffle
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc
from serenitymojo.models.vae.decoder2d import nchw_to_nhwc, nhwc_to_nchw
from serenitymojo.models.sdxl.real_weights import load_conv_rscf, load_bias

comptime TILE = 128       # fixed conv input tile (H=W)
comptime SCALE = 4        # x4 upscale
comptime NUM_CONV = 32    # mid conv+PReLU blocks


@fieldwise_init
struct SRVGGNetWeights(Movable):
    var cw: List[ArcPointer[Tensor]]   # 34 conv weights (RSCF): first + 32 mid + last
    var cb: List[ArcPointer[Tensor]]   # 34 conv biases
    var pw: List[ArcPointer[Tensor]]   # 33 PReLU alphas [64] F32


def load_srvggnet(path: String, ctx: DeviceContext) raises -> SRVGGNetWeights:
    var st = SafeTensors.open(path)
    var cw = List[ArcPointer[Tensor]]()
    var cb = List[ArcPointer[Tensor]]()
    var pw = List[ArcPointer[Tensor]]()
    var n_conv = NUM_CONV + 2                      # first + mid + last = 34
    for c in range(n_conv):
        var p = String("body.") + String(2 * c)   # convs at body 0,2,...,66
        cw.append(ArcPointer[Tensor](load_conv_rscf(st, p + ".weight", ctx)))
        cb.append(ArcPointer[Tensor](load_bias(st, p + ".bias", ctx)))
    for c in range(NUM_CONV + 1):                  # prelus at body 1,3,...,65
        var p = String("body.") + String(2 * c + 1) + ".weight"
        pw.append(ArcPointer[Tensor](load_bias(st, p, ctx)))
    return SRVGGNetWeights(cw^, cb^, pw^)


def _c3[H: Int, W: Int, Cin: Int, Cout: Int](
    x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Tensor:
    return conv2d[1, H, W, Cin, 3, 3, Cout, 1, 1, 1, 1](
        x, w, Optional[Tensor](b.clone(ctx)), ctx
    )


# One fixed 128x128x3 NHWC tile (unit [0,1] RGB) -> 512x512x3 NHWC.
def srvggnet_forward(w: SRVGGNetWeights, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var n_conv = NUM_CONV + 2
    var out = _c3[128, 128, 3, 64](x, w.cw[0][], w.cb[0][], ctx)
    out = prelu(out, w.pw[0][], ctx)
    for k in range(NUM_CONV):
        out = _c3[128, 128, 64, 64](out, w.cw[1 + k][], w.cb[1 + k][], ctx)
        out = prelu(out, w.pw[1 + k][], ctx)
    out = _c3[128, 128, 64, 48](out, w.cw[n_conv - 1][], w.cb[n_conv - 1][], ctx)

    # PixelShuffle(4) in NCHW, then nearest-4x input skip (NHWC).
    var ps = pixel_shuffle(nhwc_to_nchw(out, ctx), SCALE, ctx)  # [1,3,512,512] NCHW
    var ps_nhwc = nchw_to_nhwc(ps, ctx)                         # [1,512,512,3]
    var base = upsample_nearest2x_nhwc(upsample_nearest2x_nhwc(x, ctx), ctx)
    return add(ps_nhwc, base, ctx)                              # [1,512,512,3] NHWC
