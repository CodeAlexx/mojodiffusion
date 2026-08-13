# seedvr2_upsample_parity.mojo — Chunk-3 parity probe for the SeedVR2-3B video
# VAE decoder's Upsample3D (models/vae/seedvr2_vae.upsample3d).
#
# Gates BOTH configurations against the torch oracle (all F32):
#   up0 TEMPORAL+SPATIAL (temporal_up=True, spatial_up=True, SR=2, TR=2):
#     prefix decoder.up_blocks.0.upsamplers.0
#     in  up0_upsampler0.in  [1,512,4,16,16] -> out up0_upsampler0.out [1,512,7,32,32]
#     (D 4*2=8, remove_head -> 7; HxW 16 -> 32)
#   up2 SPATIAL-ONLY (temporal_up=False, SR=2, TR=1):
#     prefix decoder.up_blocks.2.upsamplers.0
#     in  up2_upsampler0.in  [1,256,13,64,64] -> out up2_upsampler0.out [1,256,13,128,128]
#     (D stays 13, no remove_head; HxW 64 -> 128)
#
# PASS only when BOTH cos >= 0.999.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/vae/tests/seedvr2_upsample_parity.mojo

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.models.vae.seedvr2_vae import upsample3d

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


def _run[
    SR: Int, TR: Int
](
    st: SafeTensors, orc: SafeTensors, ctx: DeviceContext,
    prefix: String, in_key: String, out_key: String,
) raises -> Float32:
    var up_w = load_bias(st, prefix + ".upscale_conv.weight", ctx)
    var up_b = load_bias(st, prefix + ".upscale_conv.bias", ctx)
    var conv_w = load_bias(st, prefix + ".conv.weight", ctx)
    var conv_b = load_bias(st, prefix + ".conv.bias", ctx)

    var x_in = load_bias(orc, in_key, ctx)
    var xs = x_in.shape()
    print(prefix, "input shape:", xs[0], xs[1], xs[2], xs[3], xs[4])

    var y = upsample3d[SR, TR](x_in, up_w, up_b, conv_w, conv_b, ctx)
    var ys = y.shape()
    print(prefix, "output shape:", ys[0], ys[1], ys[2], ys[3], ys[4])

    var y_h = y.to_host(ctx)
    var ref_h = load_bias(orc, out_key, ctx).to_host(ctx)
    var res = _cos_maxabs(y_h, ref_h)
    print(prefix, "cos=", res[0], " max_abs=", res[1])
    return res[0]


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))

    var st = SafeTensors.open(WEIGHTS)
    var orc = SafeTensors.open(ORACLE)

    print("=== up0 TEMPORAL+SPATIAL (up_blocks.0.upsamplers.0, SR=2 TR=2) ===")
    var cos_up0 = _run[2, 2](
        st, orc, ctx, "decoder.up_blocks.0.upsamplers.0",
        "up0_upsampler0.in", "up0_upsampler0.out",
    )

    print("=== up2 SPATIAL-ONLY (up_blocks.2.upsamplers.0, SR=2 TR=1) ===")
    var cos_up2 = _run[2, 1](
        st, orc, ctx, "decoder.up_blocks.2.upsamplers.0",
        "up2_upsampler0.in", "up2_upsampler0.out",
    )

    print("SUMMARY: up0 cos=", cos_up0, " up2 cos=", cos_up2)
    if cos_up0 >= 0.999 and cos_up2 >= 0.999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
