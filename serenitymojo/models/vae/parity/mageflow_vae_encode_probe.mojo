# mageflow_vae_encode_probe.mojo — MageVAE ENCODE parity gates vs the offline
# oracle (mageflow_vae_encode_oracle.py). F32 path (clean gate; weights are stored
# BF16 and upcast, matching the oracle's model.to(float32)).
#
# Gates (each stage fed the oracle's input for that stage, so divergence is
# localized to the newly-built encoder ops):
#   G1  encoder t_embedder(t=0)               vs oracle c_tembed
#   G2a patchify + head_block[0]              vs oracle cond_head0
#   G2b full head path (+proj_down)           vs oracle cond_projdown
#   G3  fuse_proj fold (z_t=0)                vs oracle s_fuse
#   G4  encoder DiCoBlock[0](s_fuse, c)       vs oracle s_block0
#   G5  norm_out (affine LayerNorm2d)         vs oracle s_normout
#   G6  proj_out (1x1 384->256)               vs oracle out_proj
#   G7  FULL encode (image -> latent mean)    vs oracle mean  (f32 + bf16)
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . -Xlinker -L/usr/lib/x86_64-linux-gnu -Xlinker -lcuda \
#        serenitymojo/models/vae/parity/mageflow_vae_encode_probe.mojo

from std.gpu.host import DeviceContext
from math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.tensor_algebra import permute
from serenitymojo.models.vae.vae_ops import reshape as vae_reshape
from serenitymojo.models.vae.decoder2d import nchw_to_nhwc
from serenitymojo.models.vae.mageflow_vae import (
    DiCoBlock,
    EncoderDiCoBlock,
    load_f32,
    load_conv1x1_as_linear,
    load_conv_rscf_f32,
    fuse_encode,
    mageflow_encode,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/vae/diffusion_pytorch_model.safetensors"
comptime ORACLE = "/home/alex/mojodiffusion/.claude/worktrees/agent-a9640baf1d0ce09c5/serenitymojo/models/vae/parity/mageflow_vae_encode_oracle_f32.safetensors"
comptime ORACLE_BF16 = "/home/alex/mojodiffusion/.claude/worktrees/agent-a9640baf1d0ce09c5/serenitymojo/models/vae/parity/mageflow_vae_encode_oracle_bf16.safetensors"
comptime IH = 64
comptime IW = 64
comptime SH = 4  # latent spatial (IH/16 = IW/16 = 4)


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
    print(tag, " cos=", cos, " max_abs=", maxd, " ref_std=", std, "  [", mark, "]")
    return ok


def _nchw_to_nhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var p = List[Int](); p.append(0); p.append(2); p.append(3); p.append(1)
    return permute(x, p^, ctx)


def _enc_tembed(st: ShardedSafeTensors, ctx: DeviceContext) raises -> Tensor:
    """encoder t_embedder(t=0): emb(0)=[1]*128 ++ [0]*128 then the encoder t-MLP."""
    var emb = List[Float32]()
    for _ in range(128):
        emb.append(1.0)
    for _ in range(128):
        emb.append(0.0)
    var embt = Tensor.from_host(emb, [1, 256], STDtype.F32, ctx)
    var w0 = load_f32(st, "student.dconv_encoder.t_embedder.mlp.0.weight", ctx)
    var b0 = load_f32(st, "student.dconv_encoder.t_embedder.mlp.0.bias", ctx)
    var w2 = load_f32(st, "student.dconv_encoder.t_embedder.mlp.2.weight", ctx)
    var b2 = load_f32(st, "student.dconv_encoder.t_embedder.mlp.2.bias", ctx)
    var c = linear_bias(embt, w0, b0, ctx)
    c = silu(c, ctx)
    return linear_bias(c, w2, b2, ctx)  # [1,384]


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))
    var st = ShardedSafeTensors.open(CKPT)
    var orc = ShardedSafeTensors.open(ORACLE)
    var all_pass = True
    var P = String("student.dconv_encoder")

    # ── G1: encoder t_embedder(t=0) vs oracle c_tembed ───────────────────────
    var c = _enc_tembed(st, ctx)  # [1,384]
    var c_h = c.to_host(ctx)
    var c_ref = load_f32(orc, "c_tembed", ctx).to_host(ctx)
    var g1 = _cos_maxabs(c_h, c_ref)
    all_pass = _report("G1 enc t_embedder ", g1[0], g1[1], g1[2]) and all_pass

    # ── G2a: patchify + head_block[0] vs oracle cond_head0 ───────────────────
    var y = _nchw_to_nhwc(load_f32(orc, "y", ctx), ctx)  # [1,64,64,3]
    var pw = load_conv_rscf_f32(st, P + ".patch_cond_embed.weight", ctx)  # [16,16,3,768]
    var pb = load_f32(st, P + ".patch_cond_embed.bias", ctx)
    var cond = conv2d[1, IH, IW, 3, 16, 16, 768, 16, 16, 0, 0](
        y, pw, Optional[Tensor](pb^), ctx
    )  # [1,4,4,768]
    var hb0 = EncoderDiCoBlock[768, 3072].load(st, P + ".head_blocks.0", ctx)
    var cond_h0 = hb0.forward[1, SH, SH](cond, ctx)
    var ch0_h = cond_h0.to_host(ctx)
    var ch0_ref = _nchw_to_nhwc(load_f32(orc, "cond_head0", ctx), ctx).to_host(ctx)
    var g2a = _cos_maxabs(ch0_h, ch0_ref)
    all_pass = _report("G2a patch+head[0] ", g2a[0], g2a[1], g2a[2]) and all_pass

    # ── G2b: full head path (+head[1]+proj_down) vs oracle cond_projdown ──────
    var hb1 = EncoderDiCoBlock[768, 3072].load(st, P + ".head_blocks.1", ctx)
    var cond_h1 = hb1.forward[1, SH, SH](cond_h0, ctx)  # [1,4,4,768]
    var pdw = load_conv1x1_as_linear(st, P + ".proj_down.weight", ctx)  # [384,768]
    var pdb = load_f32(st, P + ".proj_down.bias", ctx)
    var ch1_rows = vae_reshape(cond_h1, [SH * SH, 768], ctx)
    var pd = linear_bias(ch1_rows, pdw, pdb, ctx)  # [16,384]
    var pd_h = pd.to_host(ctx)
    var pd_ref = _nchw_to_nhwc(load_f32(orc, "cond_projdown", ctx), ctx).to_host(ctx)
    var g2b = _cos_maxabs(pd_h, pd_ref)
    all_pass = _report("G2b head+proj_down", g2b[0], g2b[1], g2b[2]) and all_pass

    # ── G3: fuse_proj fold (z_t=0) vs oracle s_fuse ──────────────────────────
    var cond_pd = _nchw_to_nhwc(load_f32(orc, "cond_projdown", ctx), ctx)  # [1,4,4,384]
    var cond_pd_rows = vae_reshape(cond_pd, [SH * SH, 384], ctx)
    var s_fuse = fuse_encode(cond_pd_rows, st, ctx)  # [16,384]
    var sf_h = s_fuse.to_host(ctx)
    var sf_ref = _nchw_to_nhwc(load_f32(orc, "s_fuse", ctx), ctx).to_host(ctx)
    var g3 = _cos_maxabs(sf_h, sf_ref)
    all_pass = _report("G3 fuse_proj fold ", g3[0], g3[1], g3[2]) and all_pass

    # ── G4: encoder DiCoBlock[0](s_fuse, c) vs oracle s_block0 ────────────────
    var s_in = _nchw_to_nhwc(load_f32(orc, "s_fuse", ctx), ctx)  # [1,4,4,384]
    var blk0 = DiCoBlock[384, 1536].load(st, P + ".blocks.0", ctx)
    var b0out = blk0.forward[1, SH, SH](s_in, c, ctx)
    var b0_h = b0out.to_host(ctx)
    var b0_ref = _nchw_to_nhwc(load_f32(orc, "s_block0", ctx), ctx).to_host(ctx)
    var g4 = _cos_maxabs(b0_h, b0_ref)
    all_pass = _report("G4 enc DiCoBlk[0] ", g4[0], g4[1], g4[2]) and all_pass

    # ── G5: norm_out (affine LayerNorm2d) vs oracle s_normout ────────────────
    var s_last = _nchw_to_nhwc(load_f32(orc, "s_blocklast", ctx), ctx)  # [1,4,4,384]
    var sl_rows = vae_reshape(s_last, [SH * SH, 384], ctx)
    var now = load_f32(st, P + ".norm_out.weight", ctx)
    var nob = load_f32(st, P + ".norm_out.bias", ctx)
    var sn = layer_norm(sl_rows, now, nob, Float32(1e-6), ctx)  # [16,384]
    var sn_h = sn.to_host(ctx)
    var sn_ref = _nchw_to_nhwc(load_f32(orc, "s_normout", ctx), ctx).to_host(ctx)
    var g5 = _cos_maxabs(sn_h, sn_ref)
    all_pass = _report("G5 norm_out       ", g5[0], g5[1], g5[2]) and all_pass

    # ── G6: proj_out (1x1 384->256) vs oracle out_proj ───────────────────────
    var s_no = _nchw_to_nhwc(load_f32(orc, "s_normout", ctx), ctx)  # [1,4,4,384]
    var sno_rows = vae_reshape(s_no, [SH * SH, 384], ctx)
    var pow_ = load_conv1x1_as_linear(st, P + ".proj_out.weight", ctx)  # [256,384]
    var pob = load_f32(st, P + ".proj_out.bias", ctx)
    var op = linear_bias(sno_rows, pow_, pob, ctx)  # [16,256]
    var op_h = op.to_host(ctx)
    var op_ref = _nchw_to_nhwc(load_f32(orc, "out_proj", ctx), ctx).to_host(ctx)
    var g6 = _cos_maxabs(op_h, op_ref)
    all_pass = _report("G6 proj_out       ", g6[0], g6[1], g6[2]) and all_pass

    # ── G7: FULL encode (image -> latent mean) vs oracle mean ────────────────
    var y_full = load_f32(orc, "y", ctx)  # [1,3,64,64] NCHW
    var mean = mageflow_encode[IH, IW](y_full, st, ctx)  # [1,128,4,4] NCHW
    var mean_h = mean.to_host(ctx)
    var mean_ref = load_f32(orc, "mean", ctx).to_host(ctx)
    var g7 = _cos_maxabs(mean_h, mean_ref)
    all_pass = _report("G7 FULL mean (f32)", g7[0], g7[1], g7[2]) and all_pass
    # also vs bf16-autocast oracle mean (ship dtype divergence)
    var orcb = ShardedSafeTensors.open(ORACLE_BF16)
    var mean_refb = load_f32(orcb, "mean", ctx).to_host(ctx)
    var g7b = _cos_maxabs(mean_h, mean_refb)
    _ = _report("G7 f32-vs-bf16    ", g7b[0], g7b[1], g7b[2])

    print("")
    if all_pass:
        print("MAGEVAE ENCODE PROBE: ALL GATES PASS")
    else:
        print("MAGEVAE ENCODE PROBE: SOME GATES FAIL (see above)")
