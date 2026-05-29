# models/vae/ldm_decoder.mojo — generic LDM AutoencoderKL decoder (SDXL/SD).
#
# Pure-Mojo, GPU-compute, inference-only port of
#   inference-flame/src/vae/ldm_decoder.rs (read FULL, 802 L).
#
# Same 2D-VAE topology as the Z-Image decoder (block_out_channels =
# (128,256,512,512), 3 resnets/up block, mid = Res+Attn+Res), so it REUSES the
# shared kit in models/vae/decoder2d.mojo VERBATIM (ResnetBlock, AttnBlock,
# Upsample, nchw<->nhwc, conv-weight RSCF loader). The only differences vs the
# Z-Image config:
#   * latent channels = 4 (SDXL/SD) instead of 16 (Z-Image).
#   * a post_quant_conv (1x1 Conv, latent_ch->latent_ch) IS present, applied to
#     the rescaled latent BEFORE conv_in (Z-Image disables it).
#   * scale/shift configurable: SDXL = (0.13025, 0.0); the decode rescale is
#     z = z / scale + shift (ldm_decoder.rs:763-764).
#
# Weight keys (standalone sdxl_vae.safetensors ships DIFFUSERS format, same as
# the Z-Image VAE the kit already consumes natively):
#   post_quant_conv.weight/bias
#   decoder.conv_in.weight/bias
#   decoder.mid_block.resnets.{0,1}.*  decoder.mid_block.attentions.0.*
#   decoder.up_blocks.{0..3}.resnets.{0,1,2}.*  .upsamplers.0.conv.*
#   decoder.conv_norm_out.weight/bias  decoder.conv_out.weight/bias
#
# Diffusers up_blocks are in NATIVE PROCESSING ORDER (0 first .. 3 last); the
# kit processes them in that order — NO LDM relabel. up_blocks.{0,1} are
# 512->512 with upsample; up_blocks.2 is 512->256 with upsample (resnet0 has
# conv_shortcut); up_blocks.3 is 256->128 NO upsample (resnet0 conv_shortcut).
#
# Comptime-parameterized on the latent spatial size (LH, LW): SDXL 1024² ->
# latent 128×128 (downscale 8) -> SDXLLdmDecoder[128,128]. The conv2d foundation
# op needs static shapes and the spatial size changes per upsample.
#
# decode(latent NCHW [1,4,LH,LW]) -> image NCHW [1,3,8*LH,8*LW].

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
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
from serenitymojo.models.vae.vae_ops import clone, reshape as vae_reshape


comptime LATENT_CH = 4
comptime CH0 = 512  # conv_in out / mid
comptime CH_UP2 = 256
comptime CH_UP3 = 128
# SDXL VAE latent normalization (sdxl_infer.rs:262): scale 0.13025, shift 0.0.
comptime SDXL_SCALING = Float32(0.13025)
comptime SDXL_SHIFT = Float32(0.0)


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# Latent rescale z = z/scale + shift (ldm_decoder.rs decode()). Mirrors the
# Z-Image decoder's _rescale but with runtime scale/shift (SDXL shift is 0).
def _rescale_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    inv_scale: Float32,
    shift: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((v * inv_scale + shift).cast[DType.bfloat16]())


def _rescale_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    inv_scale: Float32,
    shift: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](v * inv_scale + shift)


def _rescale(
    x: Tensor, scale: Float32, shift: Float32, ctx: DeviceContext
) raises -> Tensor:
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var inv = Float32(1.0) / scale
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_rescale_kernel_f32, _rescale_kernel_f32](
            X, O, inv, shift, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_rescale_kernel_bf16, _rescale_kernel_bf16](
            X, O, inv, shift, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        raise Error("_rescale: only F32/BF16 supported")
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── SDXLLdmDecoder ────────────────────────────────────────────────────────────
# Identical structure to ZImageDecoder but latent_ch=4 + post_quant_conv (1x1).
@fieldwise_init
struct SDXLLdmDecoder[LH: Int, LW: Int](Movable):
    # post_quant_conv (1x1, latent_ch -> latent_ch), pre-conv_in.
    var pqc_w: Tensor
    var pqc_b: Tensor
    # conv_in (latent_ch -> 512).
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    # mid block @ Self.LH x Self.LW, 512 ch.
    var mid_res0: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var mid_attn: AttnBlock[1, Self.LH, Self.LW, CH0]
    var mid_res1: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    # up0: 512->512 @ LH, upsample.
    var up0_r0: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var up0_r1: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var up0_r2: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var up0_up: Upsample[1, Self.LH, Self.LW, CH0]
    # up1: 512->512 @ 2LH, upsample.
    var up1_r0: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up1_r1: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up1_r2: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up1_up: Upsample[1, 2 * Self.LH, 2 * Self.LW, CH0]
    # up2: 512->256 @ 4LH, upsample (resnet0 has shortcut).
    var up2_r0: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH_UP2]
    var up2_r1: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2]
    var up2_r2: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2]
    var up2_up: Upsample[1, 4 * Self.LH, 4 * Self.LW, CH_UP2]
    # up3: 256->128 @ 8LH, NO upsample (resnet0 has shortcut).
    var up3_r0: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP3]
    var up3_r1: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3]
    var up3_r2: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3]
    # head.
    var norm_out_w: Tensor
    var norm_out_b: Tensor
    var conv_out_w: Tensor
    var conv_out_b: Tensor

    @staticmethod
    def load(
        dir_or_file: String, ctx: DeviceContext
    ) raises -> SDXLLdmDecoder[Self.LH, Self.LW]:
        var st = ShardedSafeTensors.open(dir_or_file)
        var p = String("decoder")
        return SDXLLdmDecoder[Self.LH, Self.LW](
            _load_conv_weight_rscf(st, String("post_quant_conv.weight"), ctx),
            _load_weight(st, String("post_quant_conv.bias"), ctx),
            _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx),
            _load_weight(st, p + ".conv_in.bias", ctx),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".mid_block.resnets.0", ctx
            ),
            AttnBlock[1, Self.LH, Self.LW, CH0].load(
                st, p + ".mid_block.attentions.0", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".mid_block.resnets.1", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.0", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.1", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.2", ctx
            ),
            Upsample[1, Self.LH, Self.LW, CH0].load(
                st, p + ".up_blocks.0.upsamplers.0", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.0", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.1", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.2", ctx
            ),
            Upsample[1, 2 * Self.LH, 2 * Self.LW, CH0].load(
                st, p + ".up_blocks.1.upsamplers.0", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.0", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.1", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.2", ctx
            ),
            Upsample[1, 4 * Self.LH, 4 * Self.LW, CH_UP2].load(
                st, p + ".up_blocks.2.upsamplers.0", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.0", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.1", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.2", ctx
            ),
            _load_weight(st, p + ".conv_norm_out.weight", ctx),
            _load_weight(st, p + ".conv_norm_out.bias", ctx),
            _load_conv_weight_rscf(st, p + ".conv_out.weight", ctx),
            _load_weight(st, p + ".conv_out.bias", ctx),
        )

    def decode(self, latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """latent NCHW [1,4,Self.LH,Self.LW] -> image NCHW [1,3,8*Self.LH,8*Self.LW]."""
        # Rescale (z = z/scale + shift), still NCHW.
        var z = _rescale(latent_nchw, SDXL_SCALING, SDXL_SHIFT, ctx)
        # NCHW -> NHWC once.
        var h = nchw_to_nhwc(z, ctx)  # [1,LH,LW,4]
        # post_quant_conv (1x1, 4 -> 4) on the rescaled latent.
        h = conv2d[1, Self.LH, Self.LW, LATENT_CH, 1, 1, LATENT_CH, 1, 1, 0, 0](
            h, clone(self.pqc_w, ctx),
            Optional[Tensor](clone(self.pqc_b, ctx)), ctx
        )
        # conv_in: 4 -> 512.
        h = conv2d[1, Self.LH, Self.LW, LATENT_CH, 3, 3, CH0, 1, 1, 1, 1](
            h, clone(self.conv_in_w, ctx),
            Optional[Tensor](clone(self.conv_in_b, ctx)), ctx
        )
        # mid block.
        h = self.mid_res0.forward(h, ctx)
        h = self.mid_attn.forward(h, ctx)
        h = self.mid_res1.forward(h, ctx)
        # up0.
        h = self.up0_r0.forward(h, ctx)
        h = self.up0_r1.forward(h, ctx)
        h = self.up0_r2.forward(h, ctx)
        h = self.up0_up.forward(h, ctx)  # -> 2LH
        # up1.
        h = self.up1_r0.forward(h, ctx)
        h = self.up1_r1.forward(h, ctx)
        h = self.up1_r2.forward(h, ctx)
        h = self.up1_up.forward(h, ctx)  # -> 4LH
        # up2.
        h = self.up2_r0.forward(h, ctx)
        h = self.up2_r1.forward(h, ctx)
        h = self.up2_r2.forward(h, ctx)
        h = self.up2_up.forward(h, ctx)  # -> 8LH
        # up3 (no upsample).
        h = self.up3_r0.forward(h, ctx)
        h = self.up3_r1.forward(h, ctx)
        h = self.up3_r2.forward(h, ctx)
        # head: GroupNorm -> silu -> conv_out.
        h = group_norm(h, self.norm_out_w, self.norm_out_b, GN_GROUPS, GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, 3, 3, 3, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )
        # NHWC -> NCHW for the caller / PNG writer.
        return nhwc_to_nchw(h, ctx)


# ── LDM/BFL-format VAE decoder (FLUX.1 + standalone SDXL VAE) ──────────────────
#
# FLUX's ae.safetensors AND the standalone sdxl_vae.safetensors ship the
# ORIGINAL LDM/BFL key layout, NOT the diffusers AutoencoderKL layout the
# decoder2d sub-loaders (ResnetBlock.load / AttnBlock.load) key against. The
# on-disk headers are purely LDM (verified by reading the actual files —
# ae.safetensors 244 tensors; sdxl_vae.safetensors 250 tensors):
#   decoder.conv_in / decoder.conv_out
#   decoder.mid.block_{1,2}.{norm1,conv1,norm2,conv2}
#   decoder.mid.attn_1.{norm,q,k,v,proj_out}        (q/k/v/proj_out are Conv2d 1x1)
#   decoder.up.{0..3}.block.{0..2}.{norm1,conv1,norm2,conv2[,nin_shortcut]}
#   decoder.up.{1..3}.upsample.conv
#   decoder.norm_out
#
# The two files differ ONLY in config, never in topology/keys:
#   * latent channels: FLUX 16, SDXL 4 (decoder.conv_in.weight [512,Cin,3,3]).
#   * scale/shift:      FLUX (0.3611, 0.1159), SDXL (0.13025, 0.0).
#   * post_quant_conv:  FLUX has NONE; SDXL HAS post_quant_conv.{weight[4,4,1,1],
#     bias[4]} applied to the rescaled latent BEFORE conv_in.
#
# References:
#   inference-flame/src/bin/flux1_infer.rs:258
#     LdmVAEDecoder::from_safetensors(VAE_PATH, 16, 0.3611, 0.1159)
#   inference-flame/src/bin/sdxl_infer.rs:262 (scale 0.13025, shift 0.0)
#   inference-flame/src/vae/ldm_decoder.rs LDM key spelling + the up.3->up.0
#     PROCESSING ORDER (ldm_decoder.rs:706 `for ldm_idx in [3,2,1,0]`).
#
# The COMPUTE is byte-identical to ZImageDecoder (same channels, same kit), so
# this struct reuses the decoder2d ResnetBlock / AttnBlock / Upsample *forward*
# paths and every kernel VERBATIM. Only the weight-KEY layout differs, so the
# only new code here is LDM-key loaders that construct those @fieldwise_init
# compute structs directly. No kernel/decoder2d edits.
#
# LDM -> decoder2d processing-slot map (channels match the kit exactly; the
# up.3->up.0 reversal maps LDM up.3 to kit up0 .. LDM up.0 to kit up3):
#   decoder.up.3 (512->512, upsample)        -> kit up0
#   decoder.up.2 (512->512, upsample)        -> kit up1
#   decoder.up.1 (512->256, upsample, r0 sc) -> kit up2
#   decoder.up.0 (256->128, no upsample, r0 sc) -> kit up3
#
# decode(latent NCHW [1,LATENT_CH,LH,LW]) -> image NCHW [1,3,8*LH,8*LW].

comptime FLUX_LATENT_CH = 16
comptime FLUX_SCALING = Float32(0.3611)
comptime FLUX_SHIFT = Float32(0.1159)
comptime SD3_LATENT_CH = 16
comptime SD3_SCALING = Float32(1.5305)
comptime SD3_SHIFT = Float32(0.0609)
comptime SD15_SCALING = Float32(0.18215)
comptime SD15_SHIFT = Float32(0.0)


# Load a ResnetBlock from LDM-format keys. Identical to ResnetBlock.load except
# the shortcut key is `.nin_shortcut` (LDM) instead of `.conv_shortcut`
# (diffusers); everything else (norm1/conv1/norm2/conv2) shares the spelling.
def _load_resnet_ldm[
    N: Int, H: Int, W: Int, Cin: Int, Cout: Int
](
    st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
) raises -> ResnetBlock[N, H, W, Cin, Cout]:
    var n1w = _load_weight(st, prefix + ".norm1.weight", ctx)
    var n1b = _load_weight(st, prefix + ".norm1.bias", ctx)
    var c1w = _load_conv_weight_rscf(st, prefix + ".conv1.weight", ctx)
    var c1b = _load_weight(st, prefix + ".conv1.bias", ctx)
    var n2w = _load_weight(st, prefix + ".norm2.weight", ctx)
    var n2b = _load_weight(st, prefix + ".norm2.bias", ctx)
    var c2w = _load_conv_weight_rscf(st, prefix + ".conv2.weight", ctx)
    var c2b = _load_weight(st, prefix + ".conv2.bias", ctx)

    var has_sc = Cin != Cout
    var scw: Tensor
    var scb: Tensor
    if has_sc:
        scw = _load_conv_weight_rscf(st, prefix + ".nin_shortcut.weight", ctx)
        scb = _load_weight(st, prefix + ".nin_shortcut.bias", ctx)
    else:
        var d = List[Float32]()
        d.append(0.0)
        var ds = List[Int]()
        ds.append(1)
        scw = Tensor.from_host(d.copy(), ds.copy(), STDtype.F32, ctx)
        scb = Tensor.from_host(d, ds^, STDtype.F32, ctx)
    return ResnetBlock[N, H, W, Cin, Cout](
        n1w^, n1b^, c1w^, c1b^, n2w^, n2b^, c2w^, c2b^, has_sc, scw^, scb^
    )


# Squeeze an attention Conv2d-1x1 weight [C,C,1,1] -> Linear [C,C] for the kit's
# `linear`-based AttnBlock (ldm_decoder.rs squeeze_1x1). Same numel -> device copy
# with new metadata.
def _load_attn_proj_ldm(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var w = _load_weight(st, name, ctx)
    var sh = w.shape()
    if len(sh) == 4 and sh[2] == 1 and sh[3] == 1:
        var two = List[Int]()
        two.append(sh[0])
        two.append(sh[1])
        return vae_reshape(w, two^, ctx)
    return w^


# Load the mid-block AttnBlock from LDM-format keys. LDM uses `.norm` (not
# `.group_norm`) and `.q/.k/.v/.proj_out` Conv2d-1x1 (not `.to_q/.../.to_out.0`
# Linear). Squeeze each 1x1 conv to [C,C] so the kit's `linear` path applies.
def _load_attn_ldm[
    N: Int, H: Int, W: Int, C: Int
](
    st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
) raises -> AttnBlock[N, H, W, C]:
    return AttnBlock[N, H, W, C](
        _load_weight(st, prefix + ".norm.weight", ctx),
        _load_weight(st, prefix + ".norm.bias", ctx),
        _load_attn_proj_ldm(st, prefix + ".q.weight", ctx),
        _load_weight(st, prefix + ".q.bias", ctx),
        _load_attn_proj_ldm(st, prefix + ".k.weight", ctx),
        _load_weight(st, prefix + ".k.bias", ctx),
        _load_attn_proj_ldm(st, prefix + ".v.weight", ctx),
        _load_weight(st, prefix + ".v.bias", ctx),
        _load_attn_proj_ldm(st, prefix + ".proj_out.weight", ctx),
        _load_weight(st, prefix + ".proj_out.bias", ctx),
    )


# SD1.5's diffusers VAE snapshot is almost the same key layout as the modern
# diffusers decoder kit, except the mid attention projections use the older
# `query/key/value/proj_attn` spelling instead of `to_q/to_k/to_v/to_out.0`.
def _load_attn_diffusers_legacy[
    N: Int, H: Int, W: Int, C: Int
](
    st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
) raises -> AttnBlock[N, H, W, C]:
    return AttnBlock[N, H, W, C](
        _load_weight(st, prefix + ".group_norm.weight", ctx),
        _load_weight(st, prefix + ".group_norm.bias", ctx),
        _load_weight(st, prefix + ".query.weight", ctx),
        _load_weight(st, prefix + ".query.bias", ctx),
        _load_weight(st, prefix + ".key.weight", ctx),
        _load_weight(st, prefix + ".key.bias", ctx),
        _load_weight(st, prefix + ".value.weight", ctx),
        _load_weight(st, prefix + ".value.bias", ctx),
        _load_weight(st, prefix + ".proj_attn.weight", ctx),
        _load_weight(st, prefix + ".proj_attn.bias", ctx),
    )


@fieldwise_init
struct LdmVaeDecoder[LH: Int, LW: Int, LATENT_CH: Int](Movable):
    # Latent rescale config (z = z/scale + shift): SDXL (0.13025, 0.0),
    # FLUX (0.3611, 0.1159).
    var scale: Float32
    var shift: Float32
    # post_quant_conv (1x1, LATENT_CH -> LATENT_CH), applied pre-conv_in when
    # has_pqc is True (SDXL). FLUX has none -> dummy 1-elem tensors + has_pqc=False.
    var has_pqc: Bool
    var pqc_w: Tensor
    var pqc_b: Tensor
    # conv_in (LATENT_CH -> 512).
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    # mid block @ Self.LH x Self.LW, 512 ch.
    var mid_res0: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var mid_attn: AttnBlock[1, Self.LH, Self.LW, CH0]
    var mid_res1: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    # up0 == LDM up.3: 512->512 @ LH, upsample.
    var up0_r0: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var up0_r1: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var up0_r2: ResnetBlock[1, Self.LH, Self.LW, CH0, CH0]
    var up0_up: Upsample[1, Self.LH, Self.LW, CH0]
    # up1 == LDM up.2: 512->512 @ 2LH, upsample.
    var up1_r0: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up1_r1: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up1_r2: ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0]
    var up1_up: Upsample[1, 2 * Self.LH, 2 * Self.LW, CH0]
    # up2 == LDM up.1: 512->256 @ 4LH, upsample (resnet0 has shortcut).
    var up2_r0: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH_UP2]
    var up2_r1: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2]
    var up2_r2: ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2]
    var up2_up: Upsample[1, 4 * Self.LH, 4 * Self.LW, CH_UP2]
    # up3 == LDM up.0: 256->128 @ 8LH, NO upsample (resnet0 has shortcut).
    var up3_r0: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP3]
    var up3_r1: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3]
    var up3_r2: ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3]
    # head.
    var norm_out_w: Tensor
    var norm_out_b: Tensor
    var conv_out_w: Tensor
    var conv_out_b: Tensor

    @staticmethod
    def load(
        dir_or_file: String,
        scale: Float32,
        shift: Float32,
        has_pqc: Bool,
        ctx: DeviceContext,
    ) raises -> LdmVaeDecoder[Self.LH, Self.LW, Self.LATENT_CH]:
        var st = ShardedSafeTensors.open(dir_or_file)
        var p = String("decoder")
        # post_quant_conv: load when present (SDXL), else fabricate 1-elem dummies
        # (FLUX) that decode() never applies (has_pqc=False).
        var pqc_w: Tensor
        var pqc_b: Tensor
        if has_pqc:
            pqc_w = _load_conv_weight_rscf(st, String("post_quant_conv.weight"), ctx)
            pqc_b = _load_weight(st, String("post_quant_conv.bias"), ctx)
        else:
            var d = List[Float32]()
            d.append(0.0)
            var ds = List[Int]()
            ds.append(1)
            pqc_w = Tensor.from_host(d.copy(), ds.copy(), STDtype.F32, ctx)
            pqc_b = Tensor.from_host(d, ds^, STDtype.F32, ctx)
        return LdmVaeDecoder[Self.LH, Self.LW, Self.LATENT_CH](
            scale,
            shift,
            has_pqc,
            pqc_w^,
            pqc_b^,
            _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx),
            _load_weight(st, p + ".conv_in.bias", ctx),
            _load_resnet_ldm[1, Self.LH, Self.LW, CH0, CH0](
                st, p + ".mid.block_1", ctx
            ),
            _load_attn_ldm[1, Self.LH, Self.LW, CH0](
                st, p + ".mid.attn_1", ctx
            ),
            _load_resnet_ldm[1, Self.LH, Self.LW, CH0, CH0](
                st, p + ".mid.block_2", ctx
            ),
            # up0 <- LDM up.3 (512->512, upsample).
            _load_resnet_ldm[1, Self.LH, Self.LW, CH0, CH0](
                st, p + ".up.3.block.0", ctx
            ),
            _load_resnet_ldm[1, Self.LH, Self.LW, CH0, CH0](
                st, p + ".up.3.block.1", ctx
            ),
            _load_resnet_ldm[1, Self.LH, Self.LW, CH0, CH0](
                st, p + ".up.3.block.2", ctx
            ),
            Upsample[1, Self.LH, Self.LW, CH0].load(
                st, p + ".up.3.upsample", ctx
            ),
            # up1 <- LDM up.2 (512->512, upsample).
            _load_resnet_ldm[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0](
                st, p + ".up.2.block.0", ctx
            ),
            _load_resnet_ldm[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0](
                st, p + ".up.2.block.1", ctx
            ),
            _load_resnet_ldm[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0](
                st, p + ".up.2.block.2", ctx
            ),
            Upsample[1, 2 * Self.LH, 2 * Self.LW, CH0].load(
                st, p + ".up.2.upsample", ctx
            ),
            # up2 <- LDM up.1 (512->256, upsample, block.0 nin_shortcut).
            _load_resnet_ldm[1, 4 * Self.LH, 4 * Self.LW, CH0, CH_UP2](
                st, p + ".up.1.block.0", ctx
            ),
            _load_resnet_ldm[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2](
                st, p + ".up.1.block.1", ctx
            ),
            _load_resnet_ldm[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2](
                st, p + ".up.1.block.2", ctx
            ),
            Upsample[1, 4 * Self.LH, 4 * Self.LW, CH_UP2].load(
                st, p + ".up.1.upsample", ctx
            ),
            # up3 <- LDM up.0 (256->128, NO upsample, block.0 nin_shortcut).
            _load_resnet_ldm[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP3](
                st, p + ".up.0.block.0", ctx
            ),
            _load_resnet_ldm[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3](
                st, p + ".up.0.block.1", ctx
            ),
            _load_resnet_ldm[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3](
                st, p + ".up.0.block.2", ctx
            ),
            _load_weight(st, p + ".norm_out.weight", ctx),
            _load_weight(st, p + ".norm_out.bias", ctx),
            _load_conv_weight_rscf(st, p + ".conv_out.weight", ctx),
            _load_weight(st, p + ".conv_out.bias", ctx),
        )

    @staticmethod
    def load_prefixed_no_pqc(
        dir_or_file: String,
        prefix: String,
        scale: Float32,
        shift: Float32,
        ctx: DeviceContext,
    ) raises -> LdmVaeDecoder[Self.LH, Self.LW, Self.LATENT_CH]:
        """Load LDM-format decoder keys nested under `prefix`, without post_quant_conv."""
        var st = ShardedSafeTensors.open(dir_or_file)
        var d = List[Float32]()
        d.append(0.0)
        var ds = List[Int]()
        ds.append(1)
        var pqc_w = Tensor.from_host(d.copy(), ds.copy(), STDtype.F32, ctx)
        var pqc_b = Tensor.from_host(d, ds^, STDtype.F32, ctx)
        return LdmVaeDecoder[Self.LH, Self.LW, Self.LATENT_CH](
            scale,
            shift,
            False,
            pqc_w^,
            pqc_b^,
            _load_conv_weight_rscf(st, prefix + ".conv_in.weight", ctx),
            _load_weight(st, prefix + ".conv_in.bias", ctx),
            _load_resnet_ldm[1, Self.LH, Self.LW, CH0, CH0](
                st, prefix + ".mid.block_1", ctx
            ),
            _load_attn_ldm[1, Self.LH, Self.LW, CH0](
                st, prefix + ".mid.attn_1", ctx
            ),
            _load_resnet_ldm[1, Self.LH, Self.LW, CH0, CH0](
                st, prefix + ".mid.block_2", ctx
            ),
            _load_resnet_ldm[1, Self.LH, Self.LW, CH0, CH0](
                st, prefix + ".up.3.block.0", ctx
            ),
            _load_resnet_ldm[1, Self.LH, Self.LW, CH0, CH0](
                st, prefix + ".up.3.block.1", ctx
            ),
            _load_resnet_ldm[1, Self.LH, Self.LW, CH0, CH0](
                st, prefix + ".up.3.block.2", ctx
            ),
            Upsample[1, Self.LH, Self.LW, CH0].load(
                st, prefix + ".up.3.upsample", ctx
            ),
            _load_resnet_ldm[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0](
                st, prefix + ".up.2.block.0", ctx
            ),
            _load_resnet_ldm[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0](
                st, prefix + ".up.2.block.1", ctx
            ),
            _load_resnet_ldm[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0](
                st, prefix + ".up.2.block.2", ctx
            ),
            Upsample[1, 2 * Self.LH, 2 * Self.LW, CH0].load(
                st, prefix + ".up.2.upsample", ctx
            ),
            _load_resnet_ldm[1, 4 * Self.LH, 4 * Self.LW, CH0, CH_UP2](
                st, prefix + ".up.1.block.0", ctx
            ),
            _load_resnet_ldm[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2](
                st, prefix + ".up.1.block.1", ctx
            ),
            _load_resnet_ldm[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2](
                st, prefix + ".up.1.block.2", ctx
            ),
            Upsample[1, 4 * Self.LH, 4 * Self.LW, CH_UP2].load(
                st, prefix + ".up.1.upsample", ctx
            ),
            _load_resnet_ldm[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP3](
                st, prefix + ".up.0.block.0", ctx
            ),
            _load_resnet_ldm[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3](
                st, prefix + ".up.0.block.1", ctx
            ),
            _load_resnet_ldm[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3](
                st, prefix + ".up.0.block.2", ctx
            ),
            _load_weight(st, prefix + ".norm_out.weight", ctx),
            _load_weight(st, prefix + ".norm_out.bias", ctx),
            _load_conv_weight_rscf(st, prefix + ".conv_out.weight", ctx),
            _load_weight(st, prefix + ".conv_out.bias", ctx),
        )

    @staticmethod
    def load_sd15_diffusers(
        dir_or_file: String, ctx: DeviceContext
    ) raises -> LdmVaeDecoder[Self.LH, Self.LW, Self.LATENT_CH]:
        """Load the SD1.5 diffusers VAE snapshot.

        This keeps the modern diffusers processing order (`up_blocks.0..3`)
        but handles the legacy mid-attention names used by the local SD1.5
        checkpoint. It is intentionally separate from the LDM/BFL loader above,
        whose key layout is `decoder.mid.block_1`, `decoder.up.3`, etc.
        """
        var st = ShardedSafeTensors.open(dir_or_file)
        var p = String("decoder")
        return LdmVaeDecoder[Self.LH, Self.LW, Self.LATENT_CH](
            SD15_SCALING,
            SD15_SHIFT,
            True,
            _load_conv_weight_rscf(st, String("post_quant_conv.weight"), ctx),
            _load_weight(st, String("post_quant_conv.bias"), ctx),
            _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx),
            _load_weight(st, p + ".conv_in.bias", ctx),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".mid_block.resnets.0", ctx
            ),
            _load_attn_diffusers_legacy[1, Self.LH, Self.LW, CH0](
                st, p + ".mid_block.attentions.0", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".mid_block.resnets.1", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.0", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.1", ctx
            ),
            ResnetBlock[1, Self.LH, Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.0.resnets.2", ctx
            ),
            Upsample[1, Self.LH, Self.LW, CH0].load(
                st, p + ".up_blocks.0.upsamplers.0", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.0", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.1", ctx
            ),
            ResnetBlock[1, 2 * Self.LH, 2 * Self.LW, CH0, CH0].load(
                st, p + ".up_blocks.1.resnets.2", ctx
            ),
            Upsample[1, 2 * Self.LH, 2 * Self.LW, CH0].load(
                st, p + ".up_blocks.1.upsamplers.0", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH0, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.0", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.1", ctx
            ),
            ResnetBlock[1, 4 * Self.LH, 4 * Self.LW, CH_UP2, CH_UP2].load(
                st, p + ".up_blocks.2.resnets.2", ctx
            ),
            Upsample[1, 4 * Self.LH, 4 * Self.LW, CH_UP2].load(
                st, p + ".up_blocks.2.upsamplers.0", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP2, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.0", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.1", ctx
            ),
            ResnetBlock[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, CH_UP3].load(
                st, p + ".up_blocks.3.resnets.2", ctx
            ),
            _load_weight(st, p + ".conv_norm_out.weight", ctx),
            _load_weight(st, p + ".conv_norm_out.bias", ctx),
            _load_conv_weight_rscf(st, p + ".conv_out.weight", ctx),
            _load_weight(st, p + ".conv_out.bias", ctx),
        )

    def decode(self, latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """latent NCHW [1,LATENT_CH,Self.LH,Self.LW] -> image NCHW [1,3,8*Self.LH,8*Self.LW].
        """
        # Rescale (z = z/scale + shift), still NCHW.
        var z = _rescale(latent_nchw, self.scale, self.shift, ctx)
        # NCHW -> NHWC once.
        var h = nchw_to_nhwc(z, ctx)  # [1,LH,LW,LATENT_CH]
        # post_quant_conv (1x1, LATENT_CH -> LATENT_CH), SDXL only.
        if self.has_pqc:
            h = conv2d[
                1, Self.LH, Self.LW, Self.LATENT_CH, 1, 1, Self.LATENT_CH, 1, 1, 0, 0
            ](
                h, clone(self.pqc_w, ctx),
                Optional[Tensor](clone(self.pqc_b, ctx)), ctx
            )
        # conv_in: LATENT_CH -> 512.
        h = conv2d[1, Self.LH, Self.LW, Self.LATENT_CH, 3, 3, CH0, 1, 1, 1, 1](
            h, clone(self.conv_in_w, ctx),
            Optional[Tensor](clone(self.conv_in_b, ctx)), ctx
        )
        # mid block.
        h = self.mid_res0.forward(h, ctx)
        h = self.mid_attn.forward(h, ctx)
        h = self.mid_res1.forward(h, ctx)
        # up0 (LDM up.3).
        h = self.up0_r0.forward(h, ctx)
        h = self.up0_r1.forward(h, ctx)
        h = self.up0_r2.forward(h, ctx)
        h = self.up0_up.forward(h, ctx)  # -> 2LH
        # up1 (LDM up.2).
        h = self.up1_r0.forward(h, ctx)
        h = self.up1_r1.forward(h, ctx)
        h = self.up1_r2.forward(h, ctx)
        h = self.up1_up.forward(h, ctx)  # -> 4LH
        # up2 (LDM up.1).
        h = self.up2_r0.forward(h, ctx)
        h = self.up2_r1.forward(h, ctx)
        h = self.up2_r2.forward(h, ctx)
        h = self.up2_up.forward(h, ctx)  # -> 8LH
        # up3 (LDM up.0, no upsample).
        h = self.up3_r0.forward(h, ctx)
        h = self.up3_r1.forward(h, ctx)
        h = self.up3_r2.forward(h, ctx)
        # head: GroupNorm -> silu -> conv_out.
        h = group_norm(h, self.norm_out_w, self.norm_out_b, GN_GROUPS, GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[1, 8 * Self.LH, 8 * Self.LW, CH_UP3, 3, 3, 3, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )
        # NHWC -> NCHW for the caller / PNG writer.
        return nhwc_to_nchw(h, ctx)


# ── Thin per-model factories ──────────────────────────────────────────────────
# Bind the LDM decoder's config so call sites stay declarative. Both forward to
# the same generalized LdmVaeDecoder; the only differences are the config args.

# FLUX.1: latent_ch 16, scale 0.3611, shift 0.1159, NO post_quant_conv.
def load_flux1_ldm_decoder[
    LH: Int, LW: Int
](dir_or_file: String, ctx: DeviceContext) raises -> LdmVaeDecoder[LH, LW, FLUX_LATENT_CH]:
    return LdmVaeDecoder[LH, LW, FLUX_LATENT_CH].load(
        dir_or_file, FLUX_SCALING, FLUX_SHIFT, False, ctx
    )


# SDXL: latent_ch 4, scale 0.13025, shift 0.0, WITH post_quant_conv.
def load_sdxl_ldm_decoder[
    LH: Int, LW: Int
](dir_or_file: String, ctx: DeviceContext) raises -> LdmVaeDecoder[LH, LW, LATENT_CH]:
    return LdmVaeDecoder[LH, LW, LATENT_CH].load(
        dir_or_file, SDXL_SCALING, SDXL_SHIFT, True, ctx
    )


# SD1.5: latent_ch 4, scale 0.18215, shift 0.0, WITH post_quant_conv.
def load_sd15_ldm_decoder[
    LH: Int, LW: Int
](dir_or_file: String, ctx: DeviceContext) raises -> LdmVaeDecoder[LH, LW, LATENT_CH]:
    return LdmVaeDecoder[LH, LW, LATENT_CH].load_sd15_diffusers(dir_or_file, ctx)


# SD3.5 checkpoints: latent_ch 16, scale 1.5305, shift 0.0609, embedded
# decoder keys, NO post_quant_conv in the local Large/Medium checkpoints.
def load_sd3_embedded_ldm_decoder[
    LH: Int, LW: Int
](dir_or_file: String, ctx: DeviceContext) raises -> LdmVaeDecoder[LH, LW, SD3_LATENT_CH]:
    return LdmVaeDecoder[LH, LW, SD3_LATENT_CH].load_prefixed_no_pqc(
        dir_or_file,
        String("first_stage_model.decoder"),
        SD3_SCALING,
        SD3_SHIFT,
        ctx,
    )
