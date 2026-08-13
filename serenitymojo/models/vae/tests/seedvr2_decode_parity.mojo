# seedvr2_decode_parity.mojo — Chunk 5+6 FULL-DECODE parity probe for the
# SeedVR2-3B video VAE decoder (models/vae/seedvr2_vae.decode_seedvr2_vae).
#
# Runs the entire Decoder3D forward from the oracle `latent_in` and gates EVERY
# stage against the torch oracle (all F32) — this is the WIRING test:
#   conv_in -> mid_block3d          vs mid_block.out    [1,512,4,16,16]
#           -> up_block3d[0]        vs up_block_0.out   [1,512,7,32,32]
#           -> up_block3d[1]        vs up_block_1.out   [1,512,13,64,64]
#           -> up_block3d[2]        vs up_block_2.out   [1,256,13,128,128]
#           -> up_block3d[3]        vs up_block_3.out   [1,128,13,128,128]
#   -> conv_norm_out/SiLU/conv_out  vs decode_out       [1,3,13,128,128]  (FINAL)
#
# RESULT: PASS only if the FINAL decode_out cos >= 0.999. All intermediate cos
# are reported too (they localize any wiring bug).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/vae/tests/seedvr2_decode_parity.mojo

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.activations import silu
from serenitymojo.models.vae.seedvr2_vae import (
    causal_conv3d, mid_block3d, up_block3d, _group_norm_per_frame,
)

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


def _gate(
    h: Tensor, orc: SafeTensors, key: String, ctx: DeviceContext
) raises -> Float32:
    var hs = h.shape()
    print(key, "shape:", hs[0], hs[1], hs[2], hs[3], hs[4])
    var h_host = h.to_host(ctx)
    var ref_host = load_bias(orc, key, ctx).to_host(ctx)
    var res = _cos_maxabs(h_host, ref_host)
    print(key, "cos=", res[0], " max_abs=", res[1])
    return res[0]


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))

    var st = SafeTensors.open(WEIGHTS)
    var orc = SafeTensors.open(ORACLE)

    var latent = load_bias(orc, "latent_in", ctx)   # [1,16,4,16,16]
    var ls = latent.shape()
    print("latent_in shape:", ls[0], ls[1], ls[2], ls[3], ls[4])

    # conv_in : 16 -> 512
    var ci_w = load_bias(st, "decoder.conv_in.weight", ctx)
    var ci_b = load_bias(st, "decoder.conv_in.bias", ctx)
    var h = causal_conv3d[3, 3, 3](latent, ci_w, ci_b^, ctx)

    # mid_block
    h = mid_block3d(h, st, ctx)
    var cos_mid = _gate(h, orc, "mid_block.out", ctx)

    # up_blocks
    h = up_block3d[False, True, 2, 2](h, st, 0, ctx)
    var cos_up0 = _gate(h, orc, "up_block_0.out", ctx)

    h = up_block3d[False, True, 2, 2](h, st, 1, ctx)
    var cos_up1 = _gate(h, orc, "up_block_1.out", ctx)

    h = up_block3d[True, True, 2, 1](h, st, 2, ctx)
    var cos_up2 = _gate(h, orc, "up_block_2.out", ctx)

    h = up_block3d[True, False, 2, 1](h, st, 3, ctx)
    var cos_up3 = _gate(h, orc, "up_block_3.out", ctx)

    # conv_norm_out : GN-32, 128 ch, eps 1e-6 -> SiLU -> conv_out : 128 -> 3
    var no_w = load_bias(st, "decoder.conv_norm_out.weight", ctx)
    var no_b = load_bias(st, "decoder.conv_norm_out.bias", ctx)
    h = _group_norm_per_frame(h, no_w, no_b, ctx)
    h = silu(h, ctx)
    var co_w = load_bias(st, "decoder.conv_out.weight", ctx)
    var co_b = load_bias(st, "decoder.conv_out.bias", ctx)
    h = causal_conv3d[3, 3, 3](h, co_w, co_b^, ctx)
    var cos_out = _gate(h, orc, "decode_out", ctx)

    print("SUMMARY:")
    print("  mid_block   cos=", cos_mid)
    print("  up_block_0  cos=", cos_up0)
    print("  up_block_1  cos=", cos_up1)
    print("  up_block_2  cos=", cos_up2)
    print("  up_block_3  cos=", cos_up3)
    print("  decode_out  cos=", cos_out)
    if cos_out >= 0.999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
