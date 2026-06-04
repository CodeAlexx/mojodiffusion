# models/vae/ldm_encoder.mojo — generic LDM AutoencoderKL ENCODER (SDXL/SD).
#
# Pure-Mojo, GPU-compute, inference-only port of
#   inference-flame/src/vae/ldm_encoder.rs (read FULL, 728 L)
#   numerical-parity oracle = diffusers AutoencoderKL.encode.
#
# This is the EXACT mirror of the decoder (models/vae/ldm_decoder.mojo): same
# 2D-VAE topology (block_out_channels = (128,256,512,512), layers_per_block=2 →
# 2 resnets per down block, mid = Res+Attn+Res), so it REUSES the shared kit in
# models/vae/decoder2d.mojo VERBATIM (ResnetBlock, AttnBlock, nchw<->nhwc, RSCF
# conv-weight loader) and the LDM-key sub-loaders from ldm_decoder.mojo
# (_load_resnet_ldm, _load_attn_ldm). No kernel/decoder2d edits.
#
# Forward sequence (ldm_encoder.rs encode(), :675-705 + diffusers AutoencoderKL):
#   conv_in   (3 -> 128, 3x3 pad1)
#   down_blocks 0..3, each = 2x ResBlock then (blocks 0,1,2 only) a stride-2
#               downsample: ZeroPad2d((0,1,0,1)) [right+bottom] then Conv2d 3x3
#               stride=2 pad=0.  (ldm_encoder.rs DownBlock::forward :415-426,
#               pad2d_zeros(0,1,0,1).)  Block 3 has NO downsample.
#       channel flow: d0 128->128, d1 128->256, d2 256->512, d3 512->512.
#   mid       Res(512) + Attn(512) + Res(512)   (ldm_encoder.rs MidBlock :294-298)
#   norm_out  GroupNorm(32, eps 1e-6) -> silu   (ldm_encoder.rs :688-689)
#   conv_out  (512 -> 2*latent_ch, 3x3 pad1)    (ldm_encoder.rs :690)
#   quant_conv(2*latent_ch -> 2*latent_ch, 1x1) BEFORE channel split, when present
#               (ldm_encoder.rs :692-695; SDXL has it).
#   -> moments [mu | logvar] of 2*latent_ch channels.
#     mean   = moments[..., :latent_ch]          (DiagonalGaussian.mode())
#     logvar = moments[..., latent_ch:]
#     sample = mu + exp(0.5*clamp(logvar,-30,20))*eps   (DiagonalGaussian.sample())
#
# Diffusers post-encode scaling (the pipeline boundary, NOT inside encode):
#   z = scaling_factor * (z - shift_factor)   (ldm_encoder.rs encode_scaled :714-722)
# Mirrors the decoder rescale z = z/scale + shift exactly inverted.
#
# LAYOUT RULE: foundation conv2d AND group_norm are BOTH NHWC-native, so the
# encoder stays NHWC end-to-end: NCHW->NHWC once at entry, NHWC->NCHW once at the
# latent output. (The decoder does the same.) The asymmetric (0,1,0,1) pad is a
# right+bottom NHWC zero pad before each stride-2 conv.
#
# The on-disk standalone sdxl_vae.safetensors ships ORIGINAL LDM/BFL key layout
# (verified: encoder.down.{i}.block.{j}, encoder.mid.block_{1,2},
# encoder.mid.attn_1.{q,k,v,proj_out} Conv2d-1x1, encoder.norm_out, top-level
# quant_conv) — NOT the diffusers encoder.down_blocks.* layout. So the loaders
# key against the LDM spelling and reuse ldm_decoder's _load_resnet_ldm /
# _load_attn_ldm (which already handle nin_shortcut + Conv2d-1x1 squeeze).
#
# Comptime-parameterized on the latent spatial size (LH, LW): SDXL 1024² ->
# latent 128x128 -> SdxlLdmEncoder = LdmVaeEncoder[128,128,4]. conv2d needs
# static shapes and the spatial size changes per downsample.
#
# encode_moments(image NCHW [1,3,8*LH,8*LW]) -> NHWC moments [1,LH,LW,2*latent_ch].
# encode_mean(...) -> mean latent NCHW [1,latent_ch,LH,LW].
# encode(..., eps_seed) -> sampled latent NCHW [1,latent_ch,LH,LW].
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import slice, concat, zeros_device
from serenitymojo.models.vae.decoder2d import (
    ResnetBlock,
    AttnBlock,
    nchw_to_nhwc,
    nhwc_to_nchw,
    _load_weight,
    _load_conv_weight_rscf,
    GN_GROUPS,
    GN_EPS,
)
from serenitymojo.models.vae.ldm_decoder import (
    _load_resnet_ldm,
    _load_attn_ldm,
)
from serenitymojo.models.vae.vae_ops import clone
from serenitymojo.vae.vae_encode_general import diag_gaussian_sample


comptime ENC_CH0 = 128   # conv_in out / down.0
comptime ENC_CH1 = 256   # down.1 out
comptime ENC_CH2 = 512   # down.2 / down.3 / mid
# SDXL VAE latent normalization (config.json scaling_factor 0.13025, shift 0).
comptime SDXL_ENC_SCALING = Float32(0.13025)
comptime SDXL_ENC_SHIFT = Float32(0.0)
comptime SD15_ENC_SCALING = Float32(0.18215)
comptime SD15_ENC_SHIFT = Float32(0.0)
# SD3/SD3.5 embedded VAE (16 latent ch): scaling_factor 1.5305, shift 0.0609
# (matches ldm_decoder SD3_SCALING / SD3_SHIFT). z = scale * (z - shift).
comptime SD3_ENC_LATENT_CH = 16
comptime SD3_ENC_SCALING = Float32(1.5305)
comptime SD3_ENC_SHIFT = Float32(0.0609)


# Asymmetric right(+1 W) + bottom(+1 H) zero pad on an NHWC tensor — diffusers
# encoder ZeroPad2d((0,1,0,1)) before each stride-2 downsample conv
# (ldm_encoder.rs pad2d_zeros(x, 0, 1, 0, 1) :422). Mirrors zimage_encoder.mojo.
def _pad_rb_nhwc[
    N: Int, H: Int, W: Int, C: Int
](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var zr = zeros_device([N, H, 1, C], x.dtype(), ctx)
    var xw = concat(2, ctx, x, zr)            # [N,H,W+1,C]
    var zb = zeros_device([N, 1, W + 1, C], x.dtype(), ctx)
    return concat(1, ctx, xw, zb)             # [N,H+1,W+1,C]


# ── BF16 weight casting (faithful to Rust `val.to_dtype(DType::BF16)`) ─────────
# The standalone sdxl_vae.safetensors stores F32 weights; the shared decoder2d /
# ldm_decoder loaders (_load_weight / _load_conv_weight_rscf / _load_resnet_ldm /
# _load_attn_ldm) preserve that F32 dtype. The Rust reference casts every encoder
# weight to BF16 on load (ldm_encoder.rs:566-576) and runs the whole forward in
# BF16 (F32 accumulation inside each kernel). So we re-cast each loaded weight to
# BF16 here. This mirrors the decoder's dtype convention (BF16 weights + BF16
# activations) without editing the shared loaders.
def _bf16(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    # cast_tensor materializes a fresh BF16 buffer (no-op clone if already BF16),
    # so it never consumes `t`; the caller's loaded F32 tensor is freed normally.
    return cast_tensor(clone(t, ctx), STDtype.BF16, ctx)


def _bf16_resnet[
    N: Int, H: Int, W: Int, Cin: Int, Cout: Int
](
    r: ResnetBlock[N, H, W, Cin, Cout], ctx: DeviceContext
) raises -> ResnetBlock[N, H, W, Cin, Cout]:
    return ResnetBlock[N, H, W, Cin, Cout](
        _bf16(r.norm1_w, ctx), _bf16(r.norm1_b, ctx),
        _bf16(r.conv1_w, ctx), _bf16(r.conv1_b, ctx),
        _bf16(r.norm2_w, ctx), _bf16(r.norm2_b, ctx),
        _bf16(r.conv2_w, ctx), _bf16(r.conv2_b, ctx),
        r.has_shortcut, _bf16(r.sc_w, ctx), _bf16(r.sc_b, ctx),
    )


def _bf16_attn[
    N: Int, H: Int, W: Int, C: Int
](
    a: AttnBlock[N, H, W, C], ctx: DeviceContext
) raises -> AttnBlock[N, H, W, C]:
    return AttnBlock[N, H, W, C](
        _bf16(a.norm_w, ctx), _bf16(a.norm_b, ctx),
        _bf16(a.q_w, ctx), _bf16(a.q_b, ctx),
        _bf16(a.k_w, ctx), _bf16(a.k_b, ctx),
        _bf16(a.v_w, ctx), _bf16(a.v_b, ctx),
        _bf16(a.o_w, ctx), _bf16(a.o_b, ctx),
    )


@fieldwise_init
struct LdmVaeEncoder[LH: Int, LW: Int, LATENT_CH: Int](Movable):
    """Generic LDM/diffusers AutoencoderKL encoder. LATENT_CH = 4 (SDXL/SD1.5),
    16 (SD3/FLUX). Image [1,3,8*LH,8*LW] -> moments [1,LH,LW,2*LATENT_CH].

    Comptime spatial dims so the mid-attention sequence length (LH*LW) and every
    intermediate conv shape are constants for the comptime-shaped sdpa/conv2d."""

    comptime IH = 8 * Self.LH
    comptime IW = 8 * Self.LW
    comptime H2 = 4 * Self.LH   # after down.0 stride-2
    comptime W2 = 4 * Self.LW
    comptime H4 = 2 * Self.LH   # after down.1 stride-2
    comptime W4 = 2 * Self.LW
    comptime H8 = Self.LH       # after down.2 stride-2 (== mid spatial)
    comptime W8 = Self.LW

    # latent normalization config (z = scale * (z - shift)).
    var scale: Float32
    var shift: Float32
    # whether quant_conv (1x1, 2*LATENT_CH -> 2*LATENT_CH) is present.
    var has_quant: Bool
    var quant_w: Tensor
    var quant_b: Tensor
    # conv_in (3 -> 128).
    var conv_in_w: Tensor
    var conv_in_b: Tensor
    # down.0: 128->128 @ IH, downsample.
    var d0_r0: ResnetBlock[1, Self.IH, Self.IW, ENC_CH0, ENC_CH0]
    var d0_r1: ResnetBlock[1, Self.IH, Self.IW, ENC_CH0, ENC_CH0]
    var d0_ds_w: Tensor
    var d0_ds_b: Tensor
    # down.1: 128->256 @ H2 (r0 nin_shortcut), downsample.
    var d1_r0: ResnetBlock[1, Self.H2, Self.W2, ENC_CH0, ENC_CH1]
    var d1_r1: ResnetBlock[1, Self.H2, Self.W2, ENC_CH1, ENC_CH1]
    var d1_ds_w: Tensor
    var d1_ds_b: Tensor
    # down.2: 256->512 @ H4 (r0 nin_shortcut), downsample.
    var d2_r0: ResnetBlock[1, Self.H4, Self.W4, ENC_CH1, ENC_CH2]
    var d2_r1: ResnetBlock[1, Self.H4, Self.W4, ENC_CH2, ENC_CH2]
    var d2_ds_w: Tensor
    var d2_ds_b: Tensor
    # down.3: 512->512 @ H8, NO downsample.
    var d3_r0: ResnetBlock[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2]
    var d3_r1: ResnetBlock[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2]
    # mid @ H8, 512 ch.
    var mid_res0: ResnetBlock[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2]
    var mid_attn: AttnBlock[1, Self.H8, Self.W8, ENC_CH2]
    var mid_res1: ResnetBlock[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2]
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
        has_quant: Bool,
        ctx: DeviceContext,
    ) raises -> LdmVaeEncoder[Self.LH, Self.LW, Self.LATENT_CH]:
        """Load LDM-format encoder keys + top-level quant_conv from the standalone
        VAE file (e.g. sdxl_vae.safetensors)."""
        var st = ShardedSafeTensors.open(dir_or_file)
        var p = String("encoder")
        comptime ZC2 = 2 * Self.LATENT_CH

        # quant_conv: load when present (SDXL), else 1-elem dummies (never applied).
        var quant_w: Tensor
        var quant_b: Tensor
        if has_quant:
            quant_w = _bf16(_load_conv_weight_rscf(st, String("quant_conv.weight"), ctx), ctx)
            quant_b = _bf16(_load_weight(st, String("quant_conv.bias"), ctx), ctx)
        else:
            var d = List[Float32]()
            d.append(0.0)
            var ds = List[Int]()
            ds.append(1)
            quant_w = Tensor.from_host(d.copy(), ds.copy(), STDtype.BF16, ctx)
            quant_b = Tensor.from_host(d, ds^, STDtype.BF16, ctx)

        return LdmVaeEncoder[Self.LH, Self.LW, Self.LATENT_CH](
            scale,
            shift,
            has_quant,
            quant_w^,
            quant_b^,
            _bf16(_load_conv_weight_rscf(st, p + ".conv_in.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".conv_in.bias", ctx), ctx),
            # down.0 (128->128, downsample).
            _bf16_resnet[1, Self.IH, Self.IW, ENC_CH0, ENC_CH0](
                _load_resnet_ldm[1, Self.IH, Self.IW, ENC_CH0, ENC_CH0](
                    st, p + ".down.0.block.0", ctx
                ), ctx
            ),
            _bf16_resnet[1, Self.IH, Self.IW, ENC_CH0, ENC_CH0](
                _load_resnet_ldm[1, Self.IH, Self.IW, ENC_CH0, ENC_CH0](
                    st, p + ".down.0.block.1", ctx
                ), ctx
            ),
            _bf16(_load_conv_weight_rscf(st, p + ".down.0.downsample.conv.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".down.0.downsample.conv.bias", ctx), ctx),
            # down.1 (128->256, r0 nin_shortcut, downsample).
            _bf16_resnet[1, Self.H2, Self.W2, ENC_CH0, ENC_CH1](
                _load_resnet_ldm[1, Self.H2, Self.W2, ENC_CH0, ENC_CH1](
                    st, p + ".down.1.block.0", ctx
                ), ctx
            ),
            _bf16_resnet[1, Self.H2, Self.W2, ENC_CH1, ENC_CH1](
                _load_resnet_ldm[1, Self.H2, Self.W2, ENC_CH1, ENC_CH1](
                    st, p + ".down.1.block.1", ctx
                ), ctx
            ),
            _bf16(_load_conv_weight_rscf(st, p + ".down.1.downsample.conv.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".down.1.downsample.conv.bias", ctx), ctx),
            # down.2 (256->512, r0 nin_shortcut, downsample).
            _bf16_resnet[1, Self.H4, Self.W4, ENC_CH1, ENC_CH2](
                _load_resnet_ldm[1, Self.H4, Self.W4, ENC_CH1, ENC_CH2](
                    st, p + ".down.2.block.0", ctx
                ), ctx
            ),
            _bf16_resnet[1, Self.H4, Self.W4, ENC_CH2, ENC_CH2](
                _load_resnet_ldm[1, Self.H4, Self.W4, ENC_CH2, ENC_CH2](
                    st, p + ".down.2.block.1", ctx
                ), ctx
            ),
            _bf16(_load_conv_weight_rscf(st, p + ".down.2.downsample.conv.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".down.2.downsample.conv.bias", ctx), ctx),
            # down.3 (512->512, NO downsample).
            _bf16_resnet[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                _load_resnet_ldm[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                    st, p + ".down.3.block.0", ctx
                ), ctx
            ),
            _bf16_resnet[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                _load_resnet_ldm[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                    st, p + ".down.3.block.1", ctx
                ), ctx
            ),
            # mid.
            _bf16_resnet[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                _load_resnet_ldm[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                    st, p + ".mid.block_1", ctx
                ), ctx
            ),
            _bf16_attn[1, Self.H8, Self.W8, ENC_CH2](
                _load_attn_ldm[1, Self.H8, Self.W8, ENC_CH2](
                    st, p + ".mid.attn_1", ctx
                ), ctx
            ),
            _bf16_resnet[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                _load_resnet_ldm[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                    st, p + ".mid.block_2", ctx
                ), ctx
            ),
            # head.
            _bf16(_load_weight(st, p + ".norm_out.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".norm_out.bias", ctx), ctx),
            _bf16(_load_conv_weight_rscf(st, p + ".conv_out.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".conv_out.bias", ctx), ctx),
        )

    @staticmethod
    def load_prefixed_no_quant(
        dir_or_file: String,
        prefix: String,
        scale: Float32,
        shift: Float32,
        ctx: DeviceContext,
    ) raises -> LdmVaeEncoder[Self.LH, Self.LW, Self.LATENT_CH]:
        """Load LDM-format encoder keys nested under `prefix` (e.g.
        "first_stage_model.encoder" for SD3 embedded checkpoints), WITHOUT a
        quant_conv. Mirrors the decoder's load_prefixed_no_pqc: the SD3.5
        Medium/Large embedded VAEs ship LDM encoder spellings and no quant_conv.
        Weights are already BF16 on disk; _bf16 is a no-op clone in that case."""
        var st = ShardedSafeTensors.open(dir_or_file)
        var p = prefix

        # No quant_conv: 1-elem BF16 dummies (never applied, has_quant=False).
        var d = List[Float32]()
        d.append(0.0)
        var ds = List[Int]()
        ds.append(1)
        var quant_w = Tensor.from_host(d.copy(), ds.copy(), STDtype.BF16, ctx)
        var quant_b = Tensor.from_host(d, ds^, STDtype.BF16, ctx)

        return LdmVaeEncoder[Self.LH, Self.LW, Self.LATENT_CH](
            scale,
            shift,
            False,
            quant_w^,
            quant_b^,
            _bf16(_load_conv_weight_rscf(st, p + ".conv_in.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".conv_in.bias", ctx), ctx),
            # down.0 (128->128, downsample).
            _bf16_resnet[1, Self.IH, Self.IW, ENC_CH0, ENC_CH0](
                _load_resnet_ldm[1, Self.IH, Self.IW, ENC_CH0, ENC_CH0](
                    st, p + ".down.0.block.0", ctx
                ), ctx
            ),
            _bf16_resnet[1, Self.IH, Self.IW, ENC_CH0, ENC_CH0](
                _load_resnet_ldm[1, Self.IH, Self.IW, ENC_CH0, ENC_CH0](
                    st, p + ".down.0.block.1", ctx
                ), ctx
            ),
            _bf16(_load_conv_weight_rscf(st, p + ".down.0.downsample.conv.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".down.0.downsample.conv.bias", ctx), ctx),
            # down.1 (128->256, r0 nin_shortcut, downsample).
            _bf16_resnet[1, Self.H2, Self.W2, ENC_CH0, ENC_CH1](
                _load_resnet_ldm[1, Self.H2, Self.W2, ENC_CH0, ENC_CH1](
                    st, p + ".down.1.block.0", ctx
                ), ctx
            ),
            _bf16_resnet[1, Self.H2, Self.W2, ENC_CH1, ENC_CH1](
                _load_resnet_ldm[1, Self.H2, Self.W2, ENC_CH1, ENC_CH1](
                    st, p + ".down.1.block.1", ctx
                ), ctx
            ),
            _bf16(_load_conv_weight_rscf(st, p + ".down.1.downsample.conv.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".down.1.downsample.conv.bias", ctx), ctx),
            # down.2 (256->512, r0 nin_shortcut, downsample).
            _bf16_resnet[1, Self.H4, Self.W4, ENC_CH1, ENC_CH2](
                _load_resnet_ldm[1, Self.H4, Self.W4, ENC_CH1, ENC_CH2](
                    st, p + ".down.2.block.0", ctx
                ), ctx
            ),
            _bf16_resnet[1, Self.H4, Self.W4, ENC_CH2, ENC_CH2](
                _load_resnet_ldm[1, Self.H4, Self.W4, ENC_CH2, ENC_CH2](
                    st, p + ".down.2.block.1", ctx
                ), ctx
            ),
            _bf16(_load_conv_weight_rscf(st, p + ".down.2.downsample.conv.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".down.2.downsample.conv.bias", ctx), ctx),
            # down.3 (512->512, NO downsample).
            _bf16_resnet[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                _load_resnet_ldm[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                    st, p + ".down.3.block.0", ctx
                ), ctx
            ),
            _bf16_resnet[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                _load_resnet_ldm[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                    st, p + ".down.3.block.1", ctx
                ), ctx
            ),
            # mid.
            _bf16_resnet[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                _load_resnet_ldm[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                    st, p + ".mid.block_1", ctx
                ), ctx
            ),
            _bf16_attn[1, Self.H8, Self.W8, ENC_CH2](
                _load_attn_ldm[1, Self.H8, Self.W8, ENC_CH2](
                    st, p + ".mid.attn_1", ctx
                ), ctx
            ),
            _bf16_resnet[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                _load_resnet_ldm[1, Self.H8, Self.W8, ENC_CH2, ENC_CH2](
                    st, p + ".mid.block_2", ctx
                ), ctx
            ),
            # head.
            _bf16(_load_weight(st, p + ".norm_out.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".norm_out.bias", ctx), ctx),
            _bf16(_load_conv_weight_rscf(st, p + ".conv_out.weight", ctx), ctx),
            _bf16(_load_weight(st, p + ".conv_out.bias", ctx), ctx),
        )

    def encode_moments(
        self, image_nchw: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """[1,3,8*LH,8*LW] F32/BF16 -> NHWC moments [1,LH,LW,2*LATENT_CH]."""
        var sh = image_nchw.shape()
        if len(sh) != 4 or sh[1] != 3 or sh[2] != Self.IH or sh[3] != Self.IW:
            raise Error("LdmVaeEncoder.encode_moments: expected [1,3,8*LH,8*LW]")
        if (
            image_nchw.dtype() != STDtype.F32
            and image_nchw.dtype() != STDtype.BF16
        ):
            raise Error("LdmVaeEncoder.encode_moments: expected F32 or BF16 input")

        comptime ZC2 = 2 * Self.LATENT_CH

        # NCHW -> NHWC once.
        var h = nchw_to_nhwc(image_nchw, ctx)
        # conv_in: 3 -> 128.
        h = conv2d[1, Self.IH, Self.IW, 3, 3, 3, ENC_CH0, 1, 1, 1, 1](
            h, clone(self.conv_in_w, ctx),
            Optional[Tensor](clone(self.conv_in_b, ctx)), ctx
        )
        # down.0 (128->128) + downsample -> H2.
        h = self.d0_r0.forward(h, ctx)
        h = self.d0_r1.forward(h, ctx)
        h = _pad_rb_nhwc[1, Self.IH, Self.IW, ENC_CH0](h, ctx)
        h = conv2d[1, Self.IH + 1, Self.IW + 1, ENC_CH0, 3, 3, ENC_CH0, 2, 2, 0, 0](
            h, clone(self.d0_ds_w, ctx),
            Optional[Tensor](clone(self.d0_ds_b, ctx)), ctx
        )
        # down.1 (128->256) + downsample -> H4.
        h = self.d1_r0.forward(h, ctx)
        h = self.d1_r1.forward(h, ctx)
        h = _pad_rb_nhwc[1, Self.H2, Self.W2, ENC_CH1](h, ctx)
        h = conv2d[1, Self.H2 + 1, Self.W2 + 1, ENC_CH1, 3, 3, ENC_CH1, 2, 2, 0, 0](
            h, clone(self.d1_ds_w, ctx),
            Optional[Tensor](clone(self.d1_ds_b, ctx)), ctx
        )
        # down.2 (256->512) + downsample -> H8.
        h = self.d2_r0.forward(h, ctx)
        h = self.d2_r1.forward(h, ctx)
        h = _pad_rb_nhwc[1, Self.H4, Self.W4, ENC_CH2](h, ctx)
        h = conv2d[1, Self.H4 + 1, Self.W4 + 1, ENC_CH2, 3, 3, ENC_CH2, 2, 2, 0, 0](
            h, clone(self.d2_ds_w, ctx),
            Optional[Tensor](clone(self.d2_ds_b, ctx)), ctx
        )
        # down.3 (512->512, no downsample).
        h = self.d3_r0.forward(h, ctx)
        h = self.d3_r1.forward(h, ctx)
        # mid.
        h = self.mid_res0.forward(h, ctx)
        h = self.mid_attn.forward(h, ctx)
        h = self.mid_res1.forward(h, ctx)
        # head: GroupNorm -> silu -> conv_out (512 -> 2*latent_ch).
        h = group_norm(h, self.norm_out_w, self.norm_out_b, GN_GROUPS, GN_EPS, ctx)
        h = silu(h, ctx)
        h = conv2d[1, Self.H8, Self.W8, ENC_CH2, 3, 3, ZC2, 1, 1, 1, 1](
            h, clone(self.conv_out_w, ctx),
            Optional[Tensor](clone(self.conv_out_b, ctx)), ctx
        )
        # quant_conv (1x1, 2*latent_ch -> 2*latent_ch) BEFORE channel split.
        if self.has_quant:
            h = conv2d[1, Self.H8, Self.W8, ZC2, 1, 1, ZC2, 1, 1, 0, 0](
                h, clone(self.quant_w, ctx),
                Optional[Tensor](clone(self.quant_b, ctx)), ctx
            )
        return h^

    def encode_mean(
        self, image_nchw: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """Deterministic mean latent NCHW [1,LATENT_CH,LH,LW]
        (DiagonalGaussianDistribution.mode() == first LATENT_CH channels)."""
        var moments = self.encode_moments(image_nchw, ctx)
        var mu_nhwc = slice(moments, 3, 0, Self.LATENT_CH, ctx)
        return nhwc_to_nchw(mu_nhwc, ctx)

    def encode(
        self, image_nchw: Tensor, eps_seed: UInt64, ctx: DeviceContext
    ) raises -> Tensor:
        """Sampled latent NCHW [1,LATENT_CH,LH,LW] = mu + exp(0.5*logvar)*eps
        (DiagonalGaussianDistribution.sample())."""
        var moments = self.encode_moments(image_nchw, ctx)
        var mu_nhwc = slice(moments, 3, 0, Self.LATENT_CH, ctx)
        var lv_nhwc = slice(moments, 3, Self.LATENT_CH, Self.LATENT_CH, ctx)
        # Moments are BF16 (faithful BF16 forward); the reparam kernel is F32-only,
        # so upcast mu/logvar to F32 for sampling. Rust's deterministic encode()
        # returns the mean and never samples, so this path is parity-irrelevant.
        var mu = cast_tensor(nhwc_to_nchw(mu_nhwc, ctx), STDtype.F32, ctx)
        var lv = cast_tensor(nhwc_to_nchw(lv_nhwc, ctx), STDtype.F32, ctx)
        var eps = randn(mu.shape(), eps_seed, STDtype.F32, ctx)
        return diag_gaussian_sample(mu, lv, eps, ctx)


# ── Thin per-model factories ──────────────────────────────────────────────────
# Bind the LDM encoder config so call sites stay declarative. Mirror the
# decoder's load_sdxl_ldm_decoder / load_sd15_ldm_decoder factories.

# SDXL: latent_ch 4, scale 0.13025, shift 0.0, WITH quant_conv.
def load_sdxl_ldm_encoder[
    LH: Int, LW: Int
](dir_or_file: String, ctx: DeviceContext) raises -> LdmVaeEncoder[LH, LW, 4]:
    return LdmVaeEncoder[LH, LW, 4].load(
        dir_or_file, SDXL_ENC_SCALING, SDXL_ENC_SHIFT, True, ctx
    )


# SD1.5: latent_ch 4, scale 0.18215, shift 0.0, WITH quant_conv.
def load_sd15_ldm_encoder[
    LH: Int, LW: Int
](dir_or_file: String, ctx: DeviceContext) raises -> LdmVaeEncoder[LH, LW, 4]:
    return LdmVaeEncoder[LH, LW, 4].load(
        dir_or_file, SD15_ENC_SCALING, SD15_ENC_SHIFT, True, ctx
    )


# SD3.5 embedded checkpoints: latent_ch 16, scale 1.5305, shift 0.0609,
# embedded encoder keys (first_stage_model.encoder.*), NO quant_conv (verified
# absent in SD3.5 Medium/Large). Matching factory to the decoder's
# load_sd3_embedded_ldm_decoder.
def load_sd3_embedded_ldm_encoder[
    LH: Int, LW: Int
](dir_or_file: String, ctx: DeviceContext) raises -> LdmVaeEncoder[LH, LW, SD3_ENC_LATENT_CH]:
    return LdmVaeEncoder[LH, LW, SD3_ENC_LATENT_CH].load_prefixed_no_quant(
        dir_or_file,
        String("first_stage_model.encoder"),
        SD3_ENC_SCALING,
        SD3_ENC_SHIFT,
        ctx,
    )
