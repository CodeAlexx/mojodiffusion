# seedvr2_resnet_parity.mojo — Chunk-2 parity probe for the SeedVR2-3B video VAE
# decoder's ResnetBlock3D (models/vae/seedvr2_vae.resnet_block3d).
#
# Gates BOTH configurations against the torch oracle (all F32):
#   CLEAN (no shortcut, 512->512): decoder.mid_block.resnets.0
#     in  mid_resnet0.in  [1,512,4,16,16] -> out mid_resnet0.out [1,512,4,16,16]
#   SHORTCUT (512->256):           decoder.up_blocks.2.resnets.0
#     in  up2_resnet0.in  [1,512,13,64,64] -> out up2_resnet0.out [1,256,13,64,64]
#
# PASS only when BOTH cos >= 0.999.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/vae/tests/seedvr2_resnet_parity.mojo

from std.gpu.host import DeviceContext
from math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.models.vae.seedvr2_vae import resnet_block3d

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


def _run_block(
    st: SafeTensors, orc: SafeTensors, ctx: DeviceContext,
    prefix: String, in_key: String, out_key: String, has_shortcut: Bool,
) raises -> Float32:
    var norm1_w = load_bias(st, prefix + ".norm1.weight", ctx)
    var norm1_b = load_bias(st, prefix + ".norm1.bias", ctx)
    var conv1_w = load_bias(st, prefix + ".conv1.weight", ctx)
    var conv1_b = load_bias(st, prefix + ".conv1.bias", ctx)
    var norm2_w = load_bias(st, prefix + ".norm2.weight", ctx)
    var norm2_b = load_bias(st, prefix + ".norm2.bias", ctx)
    var conv2_w = load_bias(st, prefix + ".conv2.weight", ctx)
    var conv2_b = load_bias(st, prefix + ".conv2.bias", ctx)

    # shortcut weights: real for the shortcut block, harmless placeholders
    # (norm1_w/norm1_b, never touched) for the clean block.
    var sc_w = norm1_w.clone(ctx)
    var sc_b = norm1_b.clone(ctx)
    if has_shortcut:
        sc_w = load_bias(st, prefix + ".conv_shortcut.weight", ctx)
        sc_b = load_bias(st, prefix + ".conv_shortcut.bias", ctx)

    var x_in = load_bias(orc, in_key, ctx)
    var xs = x_in.shape()
    print(prefix, "input shape:", xs[0], xs[1], xs[2], xs[3], xs[4])

    var y = resnet_block3d(
        x_in, norm1_w, norm1_b, conv1_w, conv1_b, norm2_w, norm2_b,
        conv2_w, conv2_b, has_shortcut, sc_w, sc_b, ctx,
    )
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

    print("=== CLEAN block (mid_block.resnets.0, 512->512) ===")
    var cos_mid = _run_block(
        st, orc, ctx, "decoder.mid_block.resnets.0",
        "mid_resnet0.in", "mid_resnet0.out", False,
    )

    print("=== SHORTCUT block (up_blocks.2.resnets.0, 512->256) ===")
    var cos_up2 = _run_block(
        st, orc, ctx, "decoder.up_blocks.2.resnets.0",
        "up2_resnet0.in", "up2_resnet0.out", True,
    )

    print("SUMMARY: mid cos=", cos_mid, " up2 cos=", cos_up2)
    if cos_mid >= 0.999 and cos_up2 >= 0.999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
