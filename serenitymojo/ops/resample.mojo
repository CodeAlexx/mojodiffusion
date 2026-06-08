# ops/resample.mojo — torchaudio.functional.resample (sinc_interp_hann) as a
# strided conv1d. Faithful port of torchaudio _get_sinc_resample_kernel /
# _apply_sinc_resample_kernel (rolloff 0.99, lowpass_filter_width 6, hann window).
# Used by the LTX-2 / NAVA audio path to resample the 48 kHz vocoder output to the
# 16 kHz target rate (wrapped_decode's torchaudio.functional.resample). F32.
from std.math import cos, sin, pi, ceil
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.conv1d import conv1d
from serenitymojo.ops.tensor_algebra import reshape, permute, slice, concat, zeros_device


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


# torchaudio resample: x [B,C,L] -> [B,C,Lout]. F32 in/out. sinc_interp_hann,
# rolloff=0.99, lowpass_filter_width=6.
def resample_hann(
    x: Tensor, orig_freq_in: Int, new_freq_in: Int, ctx: DeviceContext
) raises -> Tensor:
    var g = _gcd(orig_freq_in, new_freq_in)
    var orig = orig_freq_in // g
    var new = new_freq_in // g
    var rolloff = Float64(0.99)
    var lfw = Float64(6.0)
    var base = Float64(orig if orig < new else new) * rolloff
    var width = Int(ceil(lfw * Float64(orig) / base))
    var K = 2 * width + orig
    var scale = base / Float64(orig)

    # kernel [new, 1, K]: phase j, tap n -> torchaudio formula.
    var data = List[Float32]()
    data.resize(new * K, Float32(0.0))
    for j in range(new):
        for n in range(K):
            var idx = Float64(-width + n) / Float64(orig)
            var t = (Float64(-j) / Float64(new) + idx) * base
            if t < -lfw:
                t = -lfw
            elif t > lfw:
                t = lfw
            var w = cos(t * pi / lfw / Float64(2.0))
            w = w * w
            var tp = t * pi
            var kv: Float64
            if tp == Float64(0.0):
                kv = Float64(1.0)
            else:
                kv = sin(tp) / tp
            data[j * K + n] = Float32(kv * w * scale)
    var kernel = Tensor.from_host(data, [new, 1, K], STDtype.F32, ctx)

    # apply: x [B,C,L] -> [B*C,1,L]; zero-pad (width, width+orig); conv1d stride=orig
    #   -> [B*C,new,Lo]; interleave phases -> [B*C, Lo*new]; crop to ceil(new*L/orig).
    var xs = x.shape()
    var B = xs[0]
    var C = xs[1]
    var L = xs[2]
    var bc = B * C
    var xf = reshape(x, [bc, 1, L], ctx)
    var zl = zeros_device([bc, 1, width], STDtype.F32, ctx)
    var zr = zeros_device([bc, 1, width + orig], STDtype.F32, ctx)
    var xp = concat(2, ctx, zl, xf, zr)
    var y = conv1d(xp, kernel, None, orig, 0, 1, 1, ctx)  # [bc, new, Lo]
    var ys = y.shape()
    var Lo = ys[2]
    # interleave: [bc,new,Lo] -> [bc,Lo,new] -> [bc, Lo*new]
    var yt = permute(y, [0, 2, 1], ctx)
    var yflat = reshape(yt, [bc, Lo * new], ctx)
    var target = Int(ceil(Float64(new) * Float64(L) / Float64(orig)))
    var ycrop = slice(yflat, 1, 0, target, ctx)
    return reshape(ycrop, [B, C, target], ctx)
