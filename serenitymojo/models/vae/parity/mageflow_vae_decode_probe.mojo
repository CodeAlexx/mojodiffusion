# mageflow_vae_decode_probe.mojo — MageVAE DECODE parity gates vs the offline
# oracle (mageflow_vae_decode_oracle.py). F32 path (clean gate; weights are
# stored BF16 and upcast, matching the oracle's model.to(float32)).
#
# Gates (each stage fed the oracle's input for that stage, so divergence is
# localized):
#   G1 depthwise 3x3 (block-diagonal dense) vs CPU reference   [primitive]
#   G2 t_embedder(t=0)                       vs oracle c_tembed
#   G3 adaLN(block0)(c)                       vs oracle mod_block0
#   G4 s_embedder(cond)                       vs oracle s_sembed
#   G5 DiCoBlock[0](s_sembed, c)              vs oracle s_block0
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/vae/parity/mageflow_vae_decode_probe.mojo

from std.gpu.host import DeviceContext
from math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import permute
from serenitymojo.models.vae.vae_ops import reshape as vae_reshape
from serenitymojo.models.vae.mageflow_vae import (
    DiCoBlock,
    SEmbedder,
    load_f32,
    load_depthwise_rscf,
    cod_attn_forward,
    cod_decode,
    nerf_forward,
    decnet_forward,
    mageflow_decode,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/vae/diffusion_pytorch_model.safetensors"
comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity/mageflow_vae_decode_oracle_f32.safetensors"
comptime ORACLE_BF16 = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity/mageflow_vae_decode_oracle_bf16.safetensors"
comptime C = 384
comptime FFN = 1536
comptime ZC = 128
comptime N = 1
comptime SH = 4  # latent/feature spatial (H = W = 4)


def _cos_maxabs(a: List[Float32], b: List[Float32]) -> Tuple[Float32, Float32, Float32]:
    """returns (cosine, max_abs_diff, ref_std)."""
    var n = len(a)
    if len(b) < n:
        n = len(b)
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    var maxd: Float32 = 0.0
    var sb: Float64 = 0.0
    var sb2: Float64 = 0.0
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
        sb += y
        sb2 += y * y
    var denom = sqrt(na) * sqrt(nb)
    var cos: Float32 = 0.0
    if denom > 0.0:
        cos = Float32(dot / denom)
    var mean = sb / Float64(n)
    var std = Float32(sqrt(sb2 / Float64(n) - mean * mean))
    return (cos, maxd, std)


def _report(tag: String, cos: Float32, maxd: Float32, std: Float32) -> Bool:
    var ok = cos >= 0.999
    var mark: String
    if ok:
        mark = "PASS"
    else:
        mark = "FAIL"
    print(
        tag, " cos=", cos, " max_abs=", maxd, " ref_std=", std, "  [", mark, "]"
    )
    return ok


def _nchw_to_nhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var p = List[Int](); p.append(0); p.append(2); p.append(3); p.append(1)
    return permute(x, p^, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))
    var st = ShardedSafeTensors.open(CKPT)
    var orc = ShardedSafeTensors.open(ORACLE)
    var all_pass = True

    # ── G1: depthwise 3x3 (block-diagonal) vs CPU ────────────────────────────
    # synthetic: C=4, H=W=3. Build a fake depthwise weight [C,1,3,3] + input.
    comptime dC = 4
    comptime dH = 3
    var dw_host = List[Float32]()
    for i in range(dC * 1 * 3 * 3):
        dw_host.append(Float32((i * 7) % 5) * 0.1 - 0.2)
    var dwsh = List[Int](); dwsh.append(dC); dwsh.append(1); dwsh.append(3); dwsh.append(3)
    var dw = Tensor.from_host(dw_host, dwsh^, STDtype.F32, ctx)
    # build block-diagonal RSCF via the SAME loader path (write dw to a temp is
    # overkill; replicate the expansion inline for the synthetic gate).
    var rscf = List[Float32]()
    for _ in range(3 * 3 * dC * dC):
        rscf.append(0.0)
    for co in range(dC):
        for kh in range(3):
            for kw in range(3):
                var src = (co * 3 + kh) * 3 + kw
                var dst = ((kh * 3 + kw) * dC + co) * dC + co
                rscf[dst] = dw_host[src]
    var rsh4 = List[Int](); rsh4.append(3); rsh4.append(3); rsh4.append(dC); rsh4.append(dC)
    var wt = Tensor.from_host(rscf, rsh4^, STDtype.F32, ctx)
    var xin = List[Float32]()
    for i in range(dH * dH * dC):
        xin.append(Float32(i % 7) - 3.0)
    var xsh = List[Int](); xsh.append(1); xsh.append(dH); xsh.append(dH); xsh.append(dC)
    var xt = Tensor.from_host(xin, xsh^, STDtype.F32, ctx)
    var dcv = conv2d[1, dH, dH, dC, 3, 3, dC, 1, 1, 1, 1](
        xt, wt, Optional[Tensor](), ctx
    )
    var dcv_h = dcv.to_host(ctx)
    # CPU depthwise reference (pad1, stride1)
    var refv = List[Float32]()
    for _ in range(dH * dH * dC):
        refv.append(0.0)
    for h in range(dH):
        for w in range(dH):
            for cc in range(dC):
                var acc: Float32 = 0.0
                for kh in range(3):
                    for kw in range(3):
                        var ih = h + kh - 1
                        var iw = w + kw - 1
                        if ih >= 0 and ih < dH and iw >= 0 and iw < dH:
                            var xi = (ih * dH + iw) * dC + cc
                            var wi = (cc * 3 + kh) * 3 + kw
                            acc += xin[xi] * dw_host[wi]
                refv[(h * dH + w) * dC + cc] = acc
    var g1 = _cos_maxabs(dcv_h, refv)
    all_pass = _report("G1 depthwise3x3 ", g1[0], g1[1], g1[2]) and all_pass

    # ── G2: t_embedder(t=0) vs oracle c_tembed ───────────────────────────────
    # emb(t=0) = cat([cos(0)*128, sin(0)*128]) = [1]*128 ++ [0]*128
    var emb = List[Float32]()
    for _ in range(128):
        emb.append(1.0)
    for _ in range(128):
        emb.append(0.0)
    var embsh = List[Int](); embsh.append(1); embsh.append(256)
    var embt = Tensor.from_host(emb, embsh^, STDtype.F32, ctx)
    var w0 = load_f32(st, "pipeline.t_embedder.mlp.0.weight", ctx)
    var b0 = load_f32(st, "pipeline.t_embedder.mlp.0.bias", ctx)
    var w2 = load_f32(st, "pipeline.t_embedder.mlp.2.weight", ctx)
    var b2 = load_f32(st, "pipeline.t_embedder.mlp.2.bias", ctx)
    var c = linear_bias(embt, w0, b0, ctx)
    c = silu(c, ctx)
    c = linear_bias(c, w2, b2, ctx)  # [1,384]
    var c_h = c.to_host(ctx)
    var c_ref = load_f32(orc, "c_tembed", ctx).to_host(ctx)
    var g2 = _cos_maxabs(c_h, c_ref)
    all_pass = _report("G2 t_embedder    ", g2[0], g2[1], g2[2]) and all_pass

    # ── G3: adaLN(block0)(c) vs oracle mod_block0 ────────────────────────────
    var aw = load_f32(st, "pipeline.blocks.0.adaLN_modulation.1.weight", ctx)
    var ab = load_f32(st, "pipeline.blocks.0.adaLN_modulation.1.bias", ctx)
    var c_orc = load_f32(orc, "c_tembed", ctx)  # isolate: use oracle c
    var mod = linear_bias(silu(c_orc, ctx), aw, ab, ctx)  # [1,2304]
    var mod_h = mod.to_host(ctx)
    var mod_ref = load_f32(orc, "mod_block0", ctx).to_host(ctx)
    var g3 = _cos_maxabs(mod_h, mod_ref)
    all_pass = _report("G3 adaLN(block0) ", g3[0], g3[1], g3[2]) and all_pass

    # ── G4: s_embedder(cond) vs oracle s_sembed ──────────────────────────────
    var semb = SEmbedder[C, ZC].load(st, ctx)
    var cond = _nchw_to_nhwc(load_f32(orc, "cond", ctx), ctx)  # [1,4,4,384]
    var s = semb.forward[N, SH, SH](cond, ctx)
    var s_h = s.to_host(ctx)
    var s_ref = _nchw_to_nhwc(load_f32(orc, "s_sembed", ctx), ctx).to_host(ctx)
    var g4 = _cos_maxabs(s_h, s_ref)
    all_pass = _report("G4 s_embedder    ", g4[0], g4[1], g4[2]) and all_pass

    # ── G5: DiCoBlock[0](s_sembed, c) vs oracle s_block0 ─────────────────────
    var blk = DiCoBlock[C, FFN].load(st, "pipeline.blocks.0", ctx)
    var s_in = _nchw_to_nhwc(load_f32(orc, "s_sembed", ctx), ctx)  # [1,4,4,384]
    var b0out = blk.forward[N, SH, SH](s_in, c_orc, ctx)
    var b0_h = b0out.to_host(ctx)
    var b0_ref = _nchw_to_nhwc(load_f32(orc, "s_block0", ctx), ctx).to_host(ctx)
    var g5 = _cos_maxabs(b0_h, b0_ref)
    all_pass = _report("G5 DiCoBlock[0]  ", g5[0], g5[1], g5[2]) and all_pass

    # ── G6: CoD patched-attn (block.1) isolation ─────────────────────────────
    # feed oracle cod_block0_out (= input to AttnBlock), compare cod_attn1_out.
    var attn_in = _nchw_to_nhwc(load_f32(orc, "cod_block0_out", ctx), ctx)  # [1,4,4,384]
    var attn_out = cod_attn_forward(
        attn_in, st, "pipeline.y_embedder.decoder.block.1", 32, ctx
    )
    var attn_h = attn_out.to_host(ctx)
    var attn_ref = _nchw_to_nhwc(load_f32(orc, "cod_attn1_out", ctx), ctx).to_host(ctx)
    var g6 = _cos_maxabs(attn_h, attn_ref)
    all_pass = _report("G6 patched-attn  ", g6[0], g6[1], g6[2]) and all_pass

    # ── G7: CoD decoder cond (latent z -> cond) ──────────────────────────────
    var z_nhwc = _nchw_to_nhwc(load_f32(orc, "z", ctx), ctx)  # [1,4,4,128]
    var cond_out = cod_decode[SH](z_nhwc, st, ctx)  # [1,4,4,384]
    var cond_h = cond_out.to_host(ctx)
    var cond_ref = _nchw_to_nhwc(load_f32(orc, "cond", ctx), ctx).to_host(ctx)
    var g7 = _cos_maxabs(cond_h, cond_ref)
    all_pass = _report("G7 CoD cond      ", g7[0], g7[1], g7[2]) and all_pass

    # ── G8a: nerf x-embedder (cond -> x_xembed), DCT-positional isolation ─────
    var cond_nhwc = _nchw_to_nhwc(load_f32(orc, "cond", ctx), ctx)
    var xemb = nerf_forward[SH](cond_nhwc, st, ctx)  # [16*256,32]
    var xemb_h = xemb.to_host(ctx)
    var xemb_ref = load_f32(orc, "x_xembed", ctx).to_host(ctx)  # [16,256,32] flat
    var g8a = _cos_maxabs(xemb_h, xemb_ref)
    all_pass = _report("G8a nerf x-embed ", g8a[0], g8a[1], g8a[2]) and all_pass

    # ── G8b: dec_net (x_xembed + s_blocklast -> x_decnet) ────────────────────
    var xin2 = load_f32(orc, "x_xembed", ctx)  # [16,256,32]
    var xin_rows = vae_reshape(xin2, [16 * 256, 32], ctx)
    var s_last = _nchw_to_nhwc(load_f32(orc, "s_blocklast", ctx), ctx)  # [1,4,4,384]
    var s_cond = vae_reshape(s_last, [16, 384], ctx)
    var xdec = decnet_forward[SH](xin_rows, s_cond, st, ctx)
    var xdec_h = xdec.to_host(ctx)
    var xdec_ref = load_f32(orc, "x_decnet", ctx).to_host(ctx)
    var g8b = _cos_maxabs(xdec_h, xdec_ref)
    all_pass = _report("G8b dec_net      ", g8b[0], g8b[1], g8b[2]) and all_pass

    # ── G9: FULL RGB decode (latent z -> rgb) ────────────────────────────────
    var z_full = load_f32(orc, "z", ctx)  # [1,128,4,4] NCHW
    var rgb = mageflow_decode[SH](z_full, st, ctx)  # [1,3,64,64] NCHW
    var rgb_h = rgb.to_host(ctx)
    var rgb_ref = load_f32(orc, "rgb", ctx).to_host(ctx)
    var g9 = _cos_maxabs(rgb_h, rgb_ref)
    all_pass = _report("G9 FULL RGB (f32)", g9[0], g9[1], g9[2]) and all_pass
    # also measure vs bf16-autocast oracle rgb (ship dtype divergence)
    var orcb = ShardedSafeTensors.open(ORACLE_BF16)
    var rgb_refb = load_f32(orcb, "rgb", ctx).to_host(ctx)
    var g9b = _cos_maxabs(rgb_h, rgb_refb)
    _ = _report("G9 f32-vs-bf16   ", g9b[0], g9b[1], g9b[2])

    print("")
    if all_pass:
        print("MAGEVAE DECODE PROBE: ALL GATES PASS")
    else:
        print("MAGEVAE DECODE PROBE: SOME GATES FAIL (see above)")
