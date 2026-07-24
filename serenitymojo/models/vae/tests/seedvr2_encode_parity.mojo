# seedvr2_encode_parity.mojo — FULL-ENCODE parity probe for the SeedVR2-3B video
# VAE encoder (models/vae/seedvr2_vae.encode_seedvr2_vae).
#
# STAGE A (validate the NEW primitive in isolation): gate downsample3d directly:
#   down0_downsampler0.in [1,128,13,128,128] -> .out [1,128,13,64,64]  (spatial only)
#   down1_downsampler0.in [1,256,13,64,64]  -> .out [1,256,7,32,32]    (temporal+spatial)
#
# STAGE B (full wiring): run encode from `video_in` and gate every stage:
#   conv_in -> down_block3d[0]  vs down_block_0.out  [1,128,13,64,64]
#           -> down_block3d[1]  vs down_block_1.out  [1,256,7,32,32]
#           -> down_block3d[2]  vs down_block_2.out  [1,512,4,16,16]
#           -> down_block3d[3]  vs down_block_3.out  [1,512,4,16,16]
#           -> mid_block        vs mid_block.out     [1,512,4,16,16]
#   conv_norm_out/SiLU/conv_out vs conv_out.out      [1,32,4,16,16]
#           -> [:, :16]         vs latent_out        [1,16,4,16,16]  (FINAL)
#
# RESULT: PASS only if the FINAL latent_out cos >= 0.999.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/vae/tests/seedvr2_encode_parity.mojo

from std.gpu.host import DeviceContext
from math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.models.vae.seedvr2_vae import (
    causal_conv3d, downsample3d, down_block3d, encoder_mid_block3d,
    _group_norm_per_frame,
)

comptime WEIGHTS = "/home/alex/models/seedvr2-3b/seedvr2_vae.safetensors"
comptime ORACLE = "/home/alex/models/seedvr2-3b/vae_encode_oracle.safetensors"


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

    # ── STAGE A: validate downsample3d in isolation ───────────────────────────
    print("=== STAGE A: downsample3d sub-gates ===")

    # down0: spatial-only (TEMPORAL_DOWN=False), weight [128,128,1,3,3].
    var d0_in = load_bias(orc, "down0_downsampler0.in", ctx)   # [1,128,13,128,128]
    var d0_w = load_bias(st, "encoder.down_blocks.0.downsamplers.0.conv.weight", ctx)
    var d0_b = load_bias(st, "encoder.down_blocks.0.downsamplers.0.conv.bias", ctx)
    var d0_out = downsample3d[False](d0_in, d0_w, d0_b^, ctx)
    var cos_d0 = _gate(d0_out, orc, "down0_downsampler0.out", ctx)

    # down1: temporal+spatial (TEMPORAL_DOWN=True), weight [256,256,3,3,3].
    var d1_in = load_bias(orc, "down1_downsampler0.in", ctx)   # [1,256,13,64,64]
    var d1_w = load_bias(st, "encoder.down_blocks.1.downsamplers.0.conv.weight", ctx)
    var d1_b = load_bias(st, "encoder.down_blocks.1.downsamplers.0.conv.bias", ctx)
    var d1_out = downsample3d[True](d1_in, d1_w, d1_b^, ctx)
    var cos_d1 = _gate(d1_out, orc, "down1_downsampler0.out", ctx)

    # ── STAGE B: full encode e2e ──────────────────────────────────────────────
    print("=== STAGE B: full encode ===")

    var video = load_bias(orc, "video_in", ctx)   # [1,3,13,128,128]
    var vs = video.shape()
    print("video_in shape:", vs[0], vs[1], vs[2], vs[3], vs[4])

    # conv_in : 3 -> 128
    var ci_w = load_bias(st, "encoder.conv_in.weight", ctx)
    var ci_b = load_bias(st, "encoder.conv_in.bias", ctx)
    var h = causal_conv3d[3, 3, 3](video, ci_w, ci_b^, ctx)

    h = down_block3d[False, True, False](h, st, 0, ctx)
    var cos_db0 = _gate(h, orc, "down_block_0.out", ctx)

    h = down_block3d[True, True, True](h, st, 1, ctx)
    var cos_db1 = _gate(h, orc, "down_block_1.out", ctx)

    h = down_block3d[True, True, True](h, st, 2, ctx)
    var cos_db2 = _gate(h, orc, "down_block_2.out", ctx)

    h = down_block3d[False, False, False](h, st, 3, ctx)
    var cos_db3 = _gate(h, orc, "down_block_3.out", ctx)

    h = encoder_mid_block3d(h, st, ctx)
    var cos_mid = _gate(h, orc, "mid_block.out", ctx)

    # conv_norm_out : GN-32, 512 ch, eps 1e-6 -> SiLU -> conv_out : 512 -> 32
    var no_w = load_bias(st, "encoder.conv_norm_out.weight", ctx)
    var no_b = load_bias(st, "encoder.conv_norm_out.bias", ctx)
    h = _group_norm_per_frame(h, no_w, no_b, ctx)
    h = silu(h, ctx)
    var co_w = load_bias(st, "encoder.conv_out.weight", ctx)
    var co_b = load_bias(st, "encoder.conv_out.bias", ctx)
    h = causal_conv3d[3, 3, 3](h, co_w, co_b^, ctx)
    var cos_co = _gate(h, orc, "conv_out.out", ctx)

    # latent = mean = first 16 channels (double_z).
    var latent = slice(h, 1, 0, 16, ctx)
    var cos_lat = _gate(latent, orc, "latent_out", ctx)

    print("SUMMARY:")
    print("  down0_downsampler0 cos=", cos_d0)
    print("  down1_downsampler0 cos=", cos_d1)
    print("  down_block_0       cos=", cos_db0)
    print("  down_block_1       cos=", cos_db1)
    print("  down_block_2       cos=", cos_db2)
    print("  down_block_3       cos=", cos_db3)
    print("  mid_block          cos=", cos_mid)
    print("  conv_out           cos=", cos_co)
    print("  latent_out         cos=", cos_lat)
    if cos_lat >= 0.999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
