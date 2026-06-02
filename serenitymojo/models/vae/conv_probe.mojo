# conv_probe.mojo — sanity: foundation conv2d at a small shape vs a hand CPU
# reference, including the OIHW->RSCF weight transpose the VAE loader needs.
#
# Throwaway probe (kept under models/vae/ during the build). Confirms:
#   1. conv2d wants RSCF [Kh,Kw,Cin,Cout]; PyTorch weights are OIHW
#      [Cout,Cin,Kh,Kw]. We transpose on the host.
#   2. NHWC input layout.
# Run: pixi run mojo run -I . serenitymojo/models/vae/conv_probe.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.conv import conv2d


def main() raises:
    var ctx = DeviceContext()
    # N=1, H=4, W=4, Cin=2, Cout=3, K=3, stride=1, pad=1.
    comptime N = 1
    comptime H = 4
    comptime W = 4
    comptime Cin = 2
    comptime Cout = 3
    comptime K = 3

    # NHWC input: fill with i index.
    var xn = N * H * W * Cin
    var xv = List[Float32]()
    for i in range(xn):
        xv.append(Float32(i % 7) - 3.0)
    var xshape = List[Int]()
    xshape.append(N); xshape.append(H); xshape.append(W); xshape.append(Cin)
    var x = Tensor.from_host(xv, xshape^, STDtype.F32, ctx)

    # OIHW weight [Cout,Cin,K,K]; transpose to RSCF [K,K,Cin,Cout].
    var w_oihw = List[Float32]()
    for i in range(Cout * Cin * K * K):
        w_oihw.append(Float32((i * 3) % 5) * 0.1 - 0.2)
    var w_rscf = List[Float32]()
    for _ in range(K * K * Cin * Cout):
        w_rscf.append(0.0)
    # OIHW idx = ((o*Cin + ci)*K + kh)*K + kw
    # RSCF idx = ((kh*K + kw)*Cin + ci)*Cout + o
    for o in range(Cout):
        for ci in range(Cin):
            for kh in range(K):
                for kw in range(K):
                    var src = ((o * Cin + ci) * K + kh) * K + kw
                    var dst = ((kh * K + kw) * Cin + ci) * Cout + o
                    w_rscf[dst] = w_oihw[src]
    var wshape = List[Int]()
    wshape.append(K); wshape.append(K); wshape.append(Cin); wshape.append(Cout)
    var wt = Tensor.from_host(w_rscf, wshape^, STDtype.F32, ctx)

    var bv = List[Float32]()
    bv.append(0.5); bv.append(-0.5); bv.append(1.0)
    var bshape = List[Int]()
    bshape.append(Cout)
    var bias = Tensor.from_host(bv, bshape^, STDtype.F32, ctx)

    var out = conv2d[N, H, W, Cin, K, K, Cout, 1, 1, 1, 1](
        x, wt, Optional[Tensor](bias^), ctx
    )
    var ov = out.to_host(ctx)
    var osh = out.shape()
    print("conv out shape", osh[0], osh[1], osh[2], osh[3])
    var cpuref = List[Float32]()
    var rn = N * H * W * Cout
    for _ in range(rn):
        cpuref.append(0.0)
    for h in range(H):
        for w in range(W):
            for o in range(Cout):
                var acc: Float32 = bv[o]
                for kh in range(K):
                    for kw in range(K):
                        var ih = h + kh - 1
                        var iw = w + kw - 1
                        if ih >= 0 and ih < H and iw >= 0 and iw < W:
                            for ci in range(Cin):
                                var xidx = ((ih) * W + iw) * Cin + ci
                                var widx = ((kh * K + kw) * Cin + ci) * Cout + o
                                acc += xv[xidx] * w_rscf[widx]
                var oidx = (h * W + w) * Cout + o
                cpuref[oidx] = acc
    var maxd: Float32 = 0.0
    for i in range(len(ov)):
        var d = ov[i] - cpuref[i]
        if d < 0.0:
            d = -d
        if d > maxd:
            maxd = d
    print("conv max_abs_diff vs CPU cpuref:", maxd)
    if maxd < 1e-3:
        print("CONV PROBE PASS")
    else:
        print("CONV PROBE FAIL")
