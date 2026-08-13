# seedvr2_conv_in_parity.mojo — Chunk-1 parity probe for the SeedVR2-3B video VAE
# decoder's inflated CAUSAL-3D `conv_in` (models/vae/seedvr2_vae.causal_conv3d).
#
# Feeds the torch oracle's exact NCDHW input through the pure-Mojo causal conv and
# compares against the torch reference output (both F32).
#
#   weight: decoder.conv_in.weight  [512,16,3,3,3] (OIDHW) — seedvr2_vae.safetensors
#   bias:   decoder.conv_in.bias    [512]
#   in:     conv_in.in  [1,16,4,16,16] NCDHW F32       — vae_decode_oracle.safetensors
#   out:    conv_in.out [1,512,4,16,16] NCDHW F32 (torch reference)
#
# PASS when cos >= 0.999.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/vae/tests/seedvr2_conv_in_parity.mojo

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.models.vae.seedvr2_vae import causal_conv3d

comptime WEIGHTS = "/home/alex/models/seedvr2-3b/seedvr2_vae.safetensors"
comptime ORACLE = "/home/alex/models/seedvr2-3b/vae_decode_oracle.safetensors"


def _cos_maxabs(a: List[Float32], b: List[Float32]) -> Tuple[Float32, Float32]:
    var n = len(a)
    if len(b) < n:
        n = len(b)
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    var maxd: Float32 = 0.0
    for i in range(n):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > maxd:
            maxd = d
    var denom = sqrt(na) * sqrt(nb)
    var cos: Float32 = 0.0
    if denom > 0.0:
        cos = Float32(dot / denom)
    return (cos, maxd)


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))

    var st = SafeTensors.open(WEIGHTS)
    var orc = SafeTensors.open(ORACLE)

    var weight = load_bias(st, "decoder.conv_in.weight", ctx)  # [512,16,3,3,3]
    var bias = load_bias(st, "decoder.conv_in.bias", ctx)      # [512]
    var x_in = load_bias(orc, "conv_in.in", ctx)               # [1,16,4,16,16]

    var ws = weight.shape()
    var xs = x_in.shape()
    print("weight shape:", ws[0], ws[1], ws[2], ws[3], ws[4])
    print("input  shape:", xs[0], xs[1], xs[2], xs[3], xs[4])

    var y = causal_conv3d[3, 3, 3](x_in, weight, bias^, ctx)   # NCDHW [1,512,4,16,16]
    var ys = y.shape()
    print("output shape:", ys[0], ys[1], ys[2], ys[3], ys[4])

    var y_h = y.to_host(ctx)
    var ref_h = load_bias(orc, "conv_in.out", ctx).to_host(ctx)

    var res = _cos_maxabs(y_h, ref_h)
    print("cos=", res[0], " max_abs=", res[1])
    if res[0] >= 0.999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
