# seedvr2_attn_parity.mojo — Chunk-4 parity probe for the SeedVR2-3B video VAE
# decoder's mid-block spatial self-attention
# (models/vae/seedvr2_vae.mid_attention).
#
# Gates the single-head per-frame attention against the torch oracle (all F32):
#   decoder.mid_block.attentions.0
#     in  mid_attn0.in  [4,512,16,16] -> out mid_attn0.out [4,512,16,16]
#
# PASS when cos >= 0.999. The GroupNorm eps used is reported (chunk-4 reuses the
# chunk-2 per-frame GN eps=1e-6; diffusers Attention.group_norm default is 1e-5).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/vae/tests/seedvr2_attn_parity.mojo

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.models.vae.seedvr2_vae import mid_attention

comptime WEIGHTS = "/home/alex/models/seedvr2-3b/seedvr2_vae.safetensors"
comptime ORACLE = "/home/alex/models/seedvr2-3b/vae_decode_oracle.safetensors"
comptime EPS_USED = "1e-6"


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

    var prefix = String("decoder.mid_block.attentions.0")
    var gn_w = load_bias(st, prefix + ".group_norm.weight", ctx)
    var gn_b = load_bias(st, prefix + ".group_norm.bias", ctx)
    var q_w = load_bias(st, prefix + ".to_q.weight", ctx)
    var q_b = load_bias(st, prefix + ".to_q.bias", ctx)
    var k_w = load_bias(st, prefix + ".to_k.weight", ctx)
    var k_b = load_bias(st, prefix + ".to_k.bias", ctx)
    var v_w = load_bias(st, prefix + ".to_v.weight", ctx)
    var v_b = load_bias(st, prefix + ".to_v.bias", ctx)
    var o_w = load_bias(st, prefix + ".to_out.0.weight", ctx)
    var o_b = load_bias(st, prefix + ".to_out.0.bias", ctx)

    var x_in = load_bias(orc, "mid_attn0.in", ctx)
    var xs = x_in.shape()
    print("mid_attn0 input shape:", xs[0], xs[1], xs[2], xs[3])

    var y = mid_attention(
        x_in, gn_w, gn_b, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, ctx,
    )
    var ys = y.shape()
    print("mid_attn0 output shape:", ys[0], ys[1], ys[2], ys[3])

    var y_h = y.to_host(ctx)
    var ref_h = load_bias(orc, "mid_attn0.out", ctx).to_host(ctx)
    var res = _cos_maxabs(y_h, ref_h)
    print("cos=", res[0], " max_abs=", res[1], " eps=", EPS_USED)
    if res[0] >= 0.999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
