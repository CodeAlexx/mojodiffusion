# upsample_probe.mojo — SKEPTIC: verify nearest-2x replication on a NON-square,
# multi-channel NHWC tensor against an explicit host reference. diffusers
# Upsample2D uses F.interpolate(scale_factor=2, mode="nearest"), i.e.
#   out[n,oh,ow,c] = in[n, oh//2, ow//2, c]  (each input cell -> 2x2 block).
# Non-square H!=W catches any oh/ow index swap.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc


def main() raises:
    var ctx = DeviceContext()
    comptime N = 1
    comptime H = 3
    comptime W = 5  # non-square
    comptime C = 2
    var xv = List[Float32]()
    for i in range(N * H * W * C):
        xv.append(Float32(i) * 1.0)
    var xs = List[Int]()
    xs.append(N); xs.append(H); xs.append(W); xs.append(C)
    var x = Tensor.from_host(xv, xs^, STDtype.F32, ctx)

    var up = upsample_nearest2x_nhwc(x, ctx)
    var osh = up.shape()
    print("upsample out shape:", osh[0], osh[1], osh[2], osh[3],
          "(expect", N, 2 * H, 2 * W, C, ")")
    var ov = up.to_host(ctx)

    var oh = 2 * H
    var ow = 2 * W
    var maxd: Float32 = 0.0
    var bad = 0
    for ohi in range(oh):
        for owi in range(ow):
            for c in range(C):
                var ih = ohi // 2
                var iw = owi // 2
                var expv = xv[((ih) * W + iw) * C + c]
                var gotv = ov[((ohi) * ow + owi) * C + c]
                var d = gotv - expv
                if d < 0.0:
                    d = -d
                if d > maxd:
                    maxd = d
                if d > 1e-6:
                    bad += 1
    print("upsample max_abs_diff vs host nearest-ref:", maxd, " bad=", bad)
    if maxd < 1e-6 and osh[1] == oh and osh[2] == ow:
        print("UPSAMPLE PROBE PASS")
    else:
        print("UPSAMPLE PROBE FAIL")
