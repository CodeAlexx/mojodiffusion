# models/nldiffusion/emu3_vq_decoder.mojo — CHUNK D: Emu3p5 VQ-VAE DECODE path.
#
# Pure-Mojo, GPU-compute, INFERENCE-ONLY port of the Emu3p5VisionVQ decode path
# (image token ids -> pixels). Reference authority = NVIDIA creator source:
#   /mnt/disk1/models/NL-Diffusion-Image/emu3_vqvae/modeling_emu3p5visionvq.py
#     Emu3p5VisionVQVectorQuantizer.get_codebook_entry  (L425)
#     Emu3p5VisionVQModel.decode                         (L490: post_quant_conv + Decoder)
#     Emu3p5VisionVQDecoder.forward                      (L373)
#
# DTYPE: the Emu3 VQ-VAE is fp32 (config torch_dtype float32, oracle ran fp32).
# Everything here loads + computes in F32 — matches the oracle exactly.
#
# Config (emu3_vqvae/config.json): ch=256, ch_mult=[1,1,2,2,4], z_channels=256,
# embed_dim=256, num_res_blocks=4, attn_resolutions=[16], out_ch=3, resolution=256.
#   * block_in (lowest res) = ch * ch_mult[-1] = 256*4 = 1024.
#   * 5 up-levels (reversed), num_res_blocks+1 = 5 ResnetBlocks each.
#   * AttnBlock only at curr_res==16 -> the FIRST-processed up-level (weights
#     key `decoder.up.4`, the 1024-ch level) has 5 attn blocks interleaved.
#   * Upsample (nearest 2x + conv3x3) after every level EXCEPT the last
#     (weights key `decoder.up.0`).
#
# The decoder is architecturally the SAME kit as the LDM/VQGAN decoder in
# models/vae/decoder2d.mojo (GroupNorm(32,eps1e-6)->swish ResnetBlock, spatial
# self-attn AttnBlock with scale c^-0.5, nearest-2x Upsample). We REUSE that kit
# VERBATIM (ResnetBlock / AttnBlock / Upsample / nchw<->nhwc / conv-weight RSCF
# loader) and the LDM-key sub-loaders from models/vae/ldm_decoder.mojo
# (_load_resnet_ldm keys `.nin_shortcut`; _load_attn_ldm keys `.norm/.q/.k/.v/
# .proj_out` squeezing the 1x1 conv weights to [C,C] Linear). The ONLY new code
# is this per-model config wiring (5 levels, 5 resnets/level, attn on the top
# level) + get_codebook_entry gather and the `b (h w) d -> b d h w` reshape.
#
# get_codebook_entry: gather rows of `quantize.embedding.weight` [131072,256].
# reshape:            [1, h*w, d] -> [1,h,w,d] -> permute(0,3,1,2) -> [1,d,h,w].
# decode:             post_quant_conv (1x1 256->256) then Decoder.forward.
#
# Comptime-parameterized on the token grid (LH, LW): the oracle uses 16x16 ->
# 256x256 (16x upscale, 4 upsamples). Every intermediate H/W is comptime-derived.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import gather_rows, reshape, permute
from serenitymojo.models.vae.decoder2d import (
    ResnetBlock,
    AttnBlock,
    Upsample,
    nchw_to_nhwc,
    nhwc_to_nchw,
    _load_weight,
    _load_conv_weight_rscf,
    GN_GROUPS,
    GN_EPS,
)
from serenitymojo.models.vae.ldm_decoder import _load_resnet_ldm, _load_attn_ldm
from serenitymojo.models.vae.vae_ops import clone


comptime ZCH = 256      # z_channels / embed_dim
comptime C1024 = 1024   # ch * ch_mult[-1] (block_in, mid, top up-level)
comptime C512 = 512
comptime C256 = 256     # ch * ch_mult[0]
comptime OUT_CH = 3


# ── get_codebook_entry (modeling L425) ────────────────────────────────────────
# z_q = self.embedding(indices)  -> [1, N, embed_dim]. A pure gather; no math.
def emu3_get_codebook_entry(
    ids: List[Int], dir_or_file: String, ctx: DeviceContext
) raises -> Tensor:
    """`nn.Embedding(131072, 256)` lookup. ids length N -> [1, N, 256] (F32)."""
    var st = ShardedSafeTensors.open(dir_or_file)
    var w = _load_weight(st, String("quantize.embedding.weight"), ctx)  # [131072,256]
    var emb = gather_rows(w, ids, ctx)  # [N, 256]
    var n = len(ids)
    return reshape(emb, [1, n, ZCH], ctx)


# ── reshape b (h w) d -> b d h w (oracle: einops.rearrange) ────────────────────
# [1, LH*LW, 256] -> view [1,LH,LW,256] (b,h,w,d) -> permute(0,3,1,2) -> [1,256,LH,LW].
def emu3_codebook_to_z[
    LH: Int, LW: Int
](cb: Tensor, ctx: DeviceContext) raises -> Tensor:
    """[1, LH*LW, 256] -> z NCHW [1, 256, LH, LW]."""
    var nhwc = reshape(cb, [1, LH, LW, ZCH], ctx)  # b h w d
    return permute(nhwc, [0, 3, 1, 2], ctx)         # b d h w


# ── Emu3 VQ decoder ───────────────────────────────────────────────────────────
struct Emu3VQDecoder[LH: Int, LW: Int](Movable):
    # post_quant_conv (1x1, 256 -> 256) + conv_in (3x3, 256 -> 1024).
    var pqc_w: Tensor
    var pqc_b: Tensor
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    # mid @ LH x LW, 1024 ch.
    var mid_res0: ResnetBlock[1, Self.LH, Self.LW, C1024, C1024]
    var mid_attn: AttnBlock[1, Self.LH, Self.LW, C1024]
    var mid_res1: ResnetBlock[1, Self.LH, Self.LW, C1024, C1024]
    # up.4 @ LH: 1024->1024, 5 resnets, 5 attn (curr_res==16), upsample.
    var up4_r0: ResnetBlock[1, Self.LH, Self.LW, C1024, C1024]
    var up4_r1: ResnetBlock[1, Self.LH, Self.LW, C1024, C1024]
    var up4_r2: ResnetBlock[1, Self.LH, Self.LW, C1024, C1024]
    var up4_r3: ResnetBlock[1, Self.LH, Self.LW, C1024, C1024]
    var up4_r4: ResnetBlock[1, Self.LH, Self.LW, C1024, C1024]
    var up4_a0: AttnBlock[1, Self.LH, Self.LW, C1024]
    var up4_a1: AttnBlock[1, Self.LH, Self.LW, C1024]
    var up4_a2: AttnBlock[1, Self.LH, Self.LW, C1024]
    var up4_a3: AttnBlock[1, Self.LH, Self.LW, C1024]
    var up4_a4: AttnBlock[1, Self.LH, Self.LW, C1024]
    var up4_up: Upsample[1, Self.LH, Self.LW, C1024]
    # up.3 @ 2LH: block.0 1024->512 (nin_shortcut), rest 512->512, upsample.
    var up3_r0: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, C1024, C512]
    var up3_r1: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, C512, C512]
    var up3_r2: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, C512, C512]
    var up3_r3: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, C512, C512]
    var up3_r4: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, C512, C512]
    var up3_up: Upsample[1, 2 * Self.LH, 2 * Self.LW, C512]
    # up.2 @ 4LH: 512->512, 5 resnets, upsample.
    var up2_r0: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, C512, C512]
    var up2_r1: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, C512, C512]
    var up2_r2: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, C512, C512]
    var up2_r3: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, C512, C512]
    var up2_r4: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, C512, C512]
    var up2_up: Upsample[1, 4 * Self.LH, 4 * Self.LW, C512]
    # up.1 @ 8LH: block.0 512->256 (nin_shortcut), rest 256->256, upsample.
    var up1_r0: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, C512, C256]
    var up1_r1: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, C256, C256]
    var up1_r2: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, C256, C256]
    var up1_r3: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, C256, C256]
    var up1_r4: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, C256, C256]
    var up1_up: Upsample[1, 8 * Self.LH, 8 * Self.LW, C256]
    # up.0 @ 16LH: 256->256, 5 resnets, NO upsample.
    var up0_r0: ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, C256, C256]
    var up0_r1: ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, C256, C256]
    var up0_r2: ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, C256, C256]
    var up0_r3: ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, C256, C256]
    var up0_r4: ResnetBlock[1, 16 * Self.LH, 16 * Self.LW, C256, C256]
    # head @ 16LH: GroupNorm(256) -> swish -> conv_out (3x3, 256 -> 3).
    var norm_out_w: Tensor
    var norm_out_b: Tensor
    var conv_out_w: Tensor
    var conv_out_b: Tensor

    def __init__(out self, dir_or_file: String, ctx: DeviceContext) raises:
        var st = ShardedSafeTensors.open(dir_or_file)
        var p = String("decoder")
        self.pqc_w = _load_conv_weight_rscf(st, String("post_quant_conv.weight"), ctx)
        self.pqc_b = _load_weight(st, String("post_quant_conv.bias"), ctx)
        self.conv_in_w = _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx)
        self.conv_in_b = _load_weight(st, p + ".conv_in.bias", ctx)
        # mid.
        self.mid_res0 = _load_resnet_ldm[1, Self.LH, Self.LW, C1024, C1024](
            st, p + ".mid.block_1", ctx
        )
        self.mid_attn = _load_attn_ldm[1, Self.LH, Self.LW, C1024](
            st, p + ".mid.attn_1", ctx
        )
        self.mid_res1 = _load_resnet_ldm[1, Self.LH, Self.LW, C1024, C1024](
            st, p + ".mid.block_2", ctx
        )
        # up.4 (top level, attn, upsample).
        self.up4_r0 = _load_resnet_ldm[1, Self.LH, Self.LW, C1024, C1024](
            st, p + ".up.4.block.0", ctx
        )
        self.up4_r1 = _load_resnet_ldm[1, Self.LH, Self.LW, C1024, C1024](
            st, p + ".up.4.block.1", ctx
        )
        self.up4_r2 = _load_resnet_ldm[1, Self.LH, Self.LW, C1024, C1024](
            st, p + ".up.4.block.2", ctx
        )
        self.up4_r3 = _load_resnet_ldm[1, Self.LH, Self.LW, C1024, C1024](
            st, p + ".up.4.block.3", ctx
        )
        self.up4_r4 = _load_resnet_ldm[1, Self.LH, Self.LW, C1024, C1024](
            st, p + ".up.4.block.4", ctx
        )
        self.up4_a0 = _load_attn_ldm[1, Self.LH, Self.LW, C1024](
            st, p + ".up.4.attn.0", ctx
        )
        self.up4_a1 = _load_attn_ldm[1, Self.LH, Self.LW, C1024](
            st, p + ".up.4.attn.1", ctx
        )
        self.up4_a2 = _load_attn_ldm[1, Self.LH, Self.LW, C1024](
            st, p + ".up.4.attn.2", ctx
        )
        self.up4_a3 = _load_attn_ldm[1, Self.LH, Self.LW, C1024](
            st, p + ".up.4.attn.3", ctx
        )
        self.up4_a4 = _load_attn_ldm[1, Self.LH, Self.LW, C1024](
            st, p + ".up.4.attn.4", ctx
        )
        self.up4_up = Upsample[1, Self.LH, Self.LW, C1024].load(
            st, p + ".up.4.upsample", ctx
        )
        # up.3 (1024->512).
        self.up3_r0 = _load_resnet_ldm[1, 2 * Self.LH, 2 * Self.LW, C1024, C512](
            st, p + ".up.3.block.0", ctx
        )
        self.up3_r1 = _load_resnet_ldm[1, 2 * Self.LH, 2 * Self.LW, C512, C512](
            st, p + ".up.3.block.1", ctx
        )
        self.up3_r2 = _load_resnet_ldm[1, 2 * Self.LH, 2 * Self.LW, C512, C512](
            st, p + ".up.3.block.2", ctx
        )
        self.up3_r3 = _load_resnet_ldm[1, 2 * Self.LH, 2 * Self.LW, C512, C512](
            st, p + ".up.3.block.3", ctx
        )
        self.up3_r4 = _load_resnet_ldm[1, 2 * Self.LH, 2 * Self.LW, C512, C512](
            st, p + ".up.3.block.4", ctx
        )
        self.up3_up = Upsample[1, 2 * Self.LH, 2 * Self.LW, C512].load(
            st, p + ".up.3.upsample", ctx
        )
        # up.2 (512->512).
        self.up2_r0 = _load_resnet_ldm[1, 4 * Self.LH, 4 * Self.LW, C512, C512](
            st, p + ".up.2.block.0", ctx
        )
        self.up2_r1 = _load_resnet_ldm[1, 4 * Self.LH, 4 * Self.LW, C512, C512](
            st, p + ".up.2.block.1", ctx
        )
        self.up2_r2 = _load_resnet_ldm[1, 4 * Self.LH, 4 * Self.LW, C512, C512](
            st, p + ".up.2.block.2", ctx
        )
        self.up2_r3 = _load_resnet_ldm[1, 4 * Self.LH, 4 * Self.LW, C512, C512](
            st, p + ".up.2.block.3", ctx
        )
        self.up2_r4 = _load_resnet_ldm[1, 4 * Self.LH, 4 * Self.LW, C512, C512](
            st, p + ".up.2.block.4", ctx
        )
        self.up2_up = Upsample[1, 4 * Self.LH, 4 * Self.LW, C512].load(
            st, p + ".up.2.upsample", ctx
        )
        # up.1 (512->256).
        self.up1_r0 = _load_resnet_ldm[1, 8 * Self.LH, 8 * Self.LW, C512, C256](
            st, p + ".up.1.block.0", ctx
        )
        self.up1_r1 = _load_resnet_ldm[1, 8 * Self.LH, 8 * Self.LW, C256, C256](
            st, p + ".up.1.block.1", ctx
        )
        self.up1_r2 = _load_resnet_ldm[1, 8 * Self.LH, 8 * Self.LW, C256, C256](
            st, p + ".up.1.block.2", ctx
        )
        self.up1_r3 = _load_resnet_ldm[1, 8 * Self.LH, 8 * Self.LW, C256, C256](
            st, p + ".up.1.block.3", ctx
        )
        self.up1_r4 = _load_resnet_ldm[1, 8 * Self.LH, 8 * Self.LW, C256, C256](
            st, p + ".up.1.block.4", ctx
        )
        self.up1_up = Upsample[1, 8 * Self.LH, 8 * Self.LW, C256].load(
            st, p + ".up.1.upsample", ctx
        )
        # up.0 (256->256, no upsample).
        self.up0_r0 = _load_resnet_ldm[1, 16 * Self.LH, 16 * Self.LW, C256, C256](
            st, p + ".up.0.block.0", ctx
        )
        self.up0_r1 = _load_resnet_ldm[1, 16 * Self.LH, 16 * Self.LW, C256, C256](
            st, p + ".up.0.block.1", ctx
        )
        self.up0_r2 = _load_resnet_ldm[1, 16 * Self.LH, 16 * Self.LW, C256, C256](
            st, p + ".up.0.block.2", ctx
        )
        self.up0_r3 = _load_resnet_ldm[1, 16 * Self.LH, 16 * Self.LW, C256, C256](
            st, p + ".up.0.block.3", ctx
        )
        self.up0_r4 = _load_resnet_ldm[1, 16 * Self.LH, 16 * Self.LW, C256, C256](
            st, p + ".up.0.block.4", ctx
        )
        # head.
        self.norm_out_w = _load_weight(st, p + ".norm_out.weight", ctx)
        self.norm_out_b = _load_weight(st, p + ".norm_out.bias", ctx)
        self.conv_out_w = _load_conv_weight_rscf(st, p + ".conv_out.weight", ctx)
        self.conv_out_b = _load_weight(st, p + ".conv_out.bias", ctx)

    def decode(self, z_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """z NCHW [1,256,LH,LW] -> image NCHW [1,3,16*LH,16*LW]. (post_quant_conv
        + Decoder.forward). NO latent rescale (Emu3 decode has none)."""
        # NCHW -> NHWC once.
        var h = nchw_to_nhwc(z_nchw, ctx)  # [1,LH,LW,256]
        # post_quant_conv (1x1, 256 -> 256).
        h = conv2d[1, Self.LH, Self.LW, ZCH, 1, 1, ZCH, 1, 1, 0, 0](
            h, clone(self.pqc_w, ctx),
            Optional[Tensor](clone(self.pqc_b, ctx)), ctx
        )
        # conv_in (3x3, 256 -> 1024).
        h = conv2d[1, Self.LH, Self.LW, ZCH, 3, 3, C1024, 1, 1, 1, 1](
            h, clone(self.conv_in_w, ctx),
            Optional[Tensor](clone(self.conv_in_b, ctx)), ctx
        )
        # mid: Res -> Attn -> Res.
        h = self.mid_res0.forward(h, ctx)
        h = self.mid_attn.forward(h, ctx)
        h = self.mid_res1.forward(h, ctx)
        # up.4: block/attn interleaved (curr_res==16), then upsample.
        h = self.up4_r0.forward(h, ctx); h = self.up4_a0.forward(h, ctx)
        h = self.up4_r1.forward(h, ctx); h = self.up4_a1.forward(h, ctx)
        h = self.up4_r2.forward(h, ctx); h = self.up4_a2.forward(h, ctx)
        h = self.up4_r3.forward(h, ctx); h = self.up4_a3.forward(h, ctx)
        h = self.up4_r4.forward(h, ctx); h = self.up4_a4.forward(h, ctx)
        h = self.up4_up.forward(h, ctx)  # -> 2LH
        # up.3.
        h = self.up3_r0.forward(h, ctx)
        h = self.up3_r1.forward(h, ctx)
        h = self.up3_r2.forward(h, ctx)
        h = self.up3_r3.forward(h, ctx)
        h = self.up3_r4.forward(h, ctx)
        h = self.up3_up.forward(h, ctx)  # -> 4LH
        # up.2.
        h = self.up2_r0.forward(h, ctx)
        h = self.up2_r1.forward(h, ctx)
        h = self.up2_r2.forward(h, ctx)
        h = self.up2_r3.forward(h, ctx)
        h = self.up2_r4.forward(h, ctx)
        h = self.up2_up.forward(h, ctx)  # -> 8LH
        # up.1.
        h = self.up1_r0.forward(h, ctx)
        h = self.up1_r1.forward(h, ctx)
        h = self.up1_r2.forward(h, ctx)
        h = self.up1_r3.forward(h, ctx)
        h = self.up1_r4.forward(h, ctx)
        h = self.up1_up.forward(h, ctx)  # -> 16LH
        # up.0 (no upsample).
        h = self.up0_r0.forward(h, ctx)
        h = self.up0_r1.forward(h, ctx)
        h = self.up0_r2.forward(h, ctx)
        h = self.up0_r3.forward(h, ctx)
        h = self.up0_r4.forward(h, ctx)
        # head: GroupNorm -> swish -> conv_out.
        h = group_norm(h, self.norm_out_w, self.norm_out_b, GN_GROUPS, GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[1, 16 * Self.LH, 16 * Self.LW, C256, 3, 3, OUT_CH, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )
        # NHWC -> NCHW for the caller.
        return nhwc_to_nchw(h, ctx)
