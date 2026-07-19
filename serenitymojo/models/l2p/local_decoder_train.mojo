# models/l2p/local_decoder_train.mojo — FROZEN MicroDiffusionModel U-Net head
# with a TRAINING forward (saves activations) + backward that produces ONLY
# d_feat (the gradient w.r.t. the DiT feature map). The decoder weights are
# FROZEN (ai-toolkit get_model_has_grad()==False + EDv2 train_l2p.rs: local_decoder
# excluded from LoRA, requires_grad false) so we do NOT accumulate weight grads —
# we only need to backprop THROUGH the conv U-Net to the bottleneck feat input.
#
# Reference (read FULL):
#   ai-toolkit  extensions_built_in/diffusion_models/z_image/z_image_l2p_model.py
#               MicroDiffusionModel.forward (enc1..4 + pool, bottleneck fuses
#               feat, up4..1 + dec4..1, out_conv).
#   EDv2        crates/eridiffusion-core/src/models/l2p/local_decoder.rs
#   forward gate (inference, BF16, no acts): models/dit/zimage_l2p_local_decoder.mojo
#
# Math (ai-toolkit, NCHW logically; we run NHWC for the conv/pool/upsample
# kernels which are NHWC-only):
#   e1 = silu(conv3x3(x,   3   -> 64 ));  p1 = maxpool2x2(e1)
#   e2 = silu(conv3x3(p1,  64  -> 128));  p2 = maxpool2x2(e2)
#   e3 = silu(conv3x3(p2,  128 -> 256));  p3 = maxpool2x2(e3)
#   e4 = silu(conv3x3(p3,  256 -> 512));  p4 = maxpool2x2(e4)
#   b  = silu(conv1x1(cat[p4, feat],  512+3840 -> 512))   # bottleneck
#        (ai-toolkit interpolates feat to p4 spatial; at 512² p4 is 2×2 and
#         feat is [1,3840,32,32]→ here feat ALREADY arrives at p4 spatial via the
#         caller's reshape, so NO interpolate. The 512² training bucket gives
#         feat grid 32×32; p4 grid = 512/16/2/2/2/2 = 2×2. THE CALLER MUST PASS
#         feat AT p4 SPATIAL [1,3840,2,2] — see note in train_l2p_real. NOTE: the
#         inference gate passes feat at H/16 = 32×32 and the bottleneck asserts
#         feat == p4 spatial; that path uses a DIFFERENT image size. Here we
#         follow ai-toolkit EXACTLY: F.interpolate(feat -> p4 spatial, nearest).)
#   d4 = silu(conv3x3(cat[ up4 = conv3x3(upsample2x(b)),  e4],  512+512 -> 256))
#   d3 = silu(conv3x3(cat[ up3 = conv3x3(upsample2x(d4)), e3],  256+256 -> 128))
#   d2 = silu(conv3x3(cat[ up2 = conv3x3(upsample2x(d3)), e2],  128+128 -> 64 ))
#   d1 = silu(conv3x3(cat[ up1 = conv3x3(upsample2x(d2)), e1],  64+64   -> 64 ))
#   out= conv1x1(d1, 64 -> 3)
#
# NOTE on ai-toolkit's bottleneck: ai-toolkit's forward computes p4 by pooling
# FOUR times (e1..e4 each followed by a pool), so at 512² p4 is 32/2^? — actually
# 512 -> enc convs keep spatial; ONLY the 4 maxpools halve: 512→256→128→64→32.
# So p4 grid = 32×32 == feat grid. THE INTERPOLATE IS THEN A NO-OP. (The L2P
# x_embedder patch=16 makes feat grid = H/16 = 32; the 4 pools also reach 32.
# They coincide BY DESIGN.) We assert feat grid == p4 grid and skip interpolate.
#
# DTYPE: F32 throughout (the maxpool/upsample/silu/conv backward kernels are
# F32-only; ai-toolkit runs the decoder in the model dtype but the L2P loss is
# computed in F32 and the decoder is small — F32 is the faithful + numerically
# safe choice for the training head).
#
# LAYOUT: all internal tensors are NHWC (conv/pool/upsample kernels are NHWC).
# Caller passes noisy as NCHW [1,3,H,W] and feat as NCHW [1,3840,gh,gw]; we
# permute to NHWC on entry and return d_feat in NCHW [1,3840,gh,gw].
#
# Mojo 1.0.0b1: def not fn; Tensor move-only; .clone(ctx) to duplicate.

from std.gpu.host import DeviceContext
from std.collections import Optional

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.conv2d_backward import conv2d_backward
from serenitymojo.ops.activations import silu
from serenitymojo.ops.activation_backward import silu_backward
# NOTE: maxpool2d_backward is NOT imported — the encoder branch grad is dropped
# (the encoder consumes only `noisy`, a detached training input). Only the
# UPSAMPLE backward is needed on the decoder path.
from serenitymojo.ops.pool_backward import upsample_nearest2d_backward
from serenitymojo.ops.shape_backward import cat_backward
from serenitymojo.ops.tensor_algebra import concat, permute
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc
from serenitymojo.models.dit.zimage_l2p_local_decoder import (
    ZImageL2PLocalDecoderGate, maxpool2x2_nhwc,
)
from serenitymojo.models.dit.zimage_l2p_contract import (
    ZIMAGE_L2P_HIDDEN, ZIMAGE_L2P_LD_C1, ZIMAGE_L2P_LD_C2,
    ZIMAGE_L2P_LD_C3, ZIMAGE_L2P_LD_C4, ZIMAGE_L2P_PIXEL_CHANNELS,
)


# ── F32 weight bundle (decoder convs, RSCF filter [kh,kw,cin,cout] + bias) ────
# Built once from the inference gate (which loads BF16 RSCF) by casting to F32.
struct L2PDecoderF32(Movable):
    var enc1_w: Tensor; var enc1_b: Tensor
    var enc2_w: Tensor; var enc2_b: Tensor
    var enc3_w: Tensor; var enc3_b: Tensor
    var enc4_w: Tensor; var enc4_b: Tensor
    var bot_w: Tensor;  var bot_b: Tensor
    var up4_w: Tensor;  var up4_b: Tensor
    var up3_w: Tensor;  var up3_b: Tensor
    var up2_w: Tensor;  var up2_b: Tensor
    var up1_w: Tensor;  var up1_b: Tensor
    var dec4_w: Tensor; var dec4_b: Tensor
    var dec3_w: Tensor; var dec3_b: Tensor
    var dec2_w: Tensor; var dec2_b: Tensor
    var dec1_w: Tensor; var dec1_b: Tensor
    var out_w: Tensor;  var out_b: Tensor

    def __init__(
        out self,
        var enc1_w: Tensor, var enc1_b: Tensor,
        var enc2_w: Tensor, var enc2_b: Tensor,
        var enc3_w: Tensor, var enc3_b: Tensor,
        var enc4_w: Tensor, var enc4_b: Tensor,
        var bot_w: Tensor,  var bot_b: Tensor,
        var up4_w: Tensor,  var up4_b: Tensor,
        var up3_w: Tensor,  var up3_b: Tensor,
        var up2_w: Tensor,  var up2_b: Tensor,
        var up1_w: Tensor,  var up1_b: Tensor,
        var dec4_w: Tensor, var dec4_b: Tensor,
        var dec3_w: Tensor, var dec3_b: Tensor,
        var dec2_w: Tensor, var dec2_b: Tensor,
        var dec1_w: Tensor, var dec1_b: Tensor,
        var out_w: Tensor,  var out_b: Tensor,
    ):
        self.enc1_w = enc1_w^; self.enc1_b = enc1_b^
        self.enc2_w = enc2_w^; self.enc2_b = enc2_b^
        self.enc3_w = enc3_w^; self.enc3_b = enc3_b^
        self.enc4_w = enc4_w^; self.enc4_b = enc4_b^
        self.bot_w = bot_w^;   self.bot_b = bot_b^
        self.up4_w = up4_w^;   self.up4_b = up4_b^
        self.up3_w = up3_w^;   self.up3_b = up3_b^
        self.up2_w = up2_w^;   self.up2_b = up2_b^
        self.up1_w = up1_w^;   self.up1_b = up1_b^
        self.dec4_w = dec4_w^; self.dec4_b = dec4_b^
        self.dec3_w = dec3_w^; self.dec3_b = dec3_b^
        self.dec2_w = dec2_w^; self.dec2_b = dec2_b^
        self.dec1_w = dec1_w^; self.dec1_b = dec1_b^
        self.out_w = out_w^;   self.out_b = out_b^


def _f32(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    if t.dtype() == STDtype.F32:
        return t.clone(ctx)
    return cast_tensor(t, STDtype.F32, ctx)


def l2p_decoder_f32_from_gate(
    g: ZImageL2PLocalDecoderGate, ctx: DeviceContext
) raises -> L2PDecoderF32:
    """Cast the inference gate's BF16 RSCF conv weights to F32 once. The gate
    already converts OIHW->RSCF and loads bias, so we only cast dtype."""
    return L2PDecoderF32(
        _f32(g.enc1_w, ctx), _f32(g.enc1_b, ctx),
        _f32(g.enc2_w, ctx), _f32(g.enc2_b, ctx),
        _f32(g.enc3_w, ctx), _f32(g.enc3_b, ctx),
        _f32(g.enc4_w, ctx), _f32(g.enc4_b, ctx),
        _f32(g.bottleneck_w, ctx), _f32(g.bottleneck_b, ctx),
        _f32(g.up4_w, ctx), _f32(g.up4_b, ctx),
        _f32(g.up3_w, ctx), _f32(g.up3_b, ctx),
        _f32(g.up2_w, ctx), _f32(g.up2_b, ctx),
        _f32(g.up1_w, ctx), _f32(g.up1_b, ctx),
        _f32(g.dec4_w, ctx), _f32(g.dec4_b, ctx),
        _f32(g.dec3_w, ctx), _f32(g.dec3_b, ctx),
        _f32(g.dec2_w, ctx), _f32(g.dec2_b, ctx),
        _f32(g.dec1_w, ctx), _f32(g.dec1_b, ctx),
        _f32(g.out_w, ctx), _f32(g.out_b, ctx),
    )


# ── saved activations for the backward pass ───────────────────────────────────
# Every tensor a backward op consumes. silu_backward needs the PRE-silu conv
# output; conv2d_backward needs the conv INPUT (NHWC); maxpool2d_backward needs
# the pool INPUT; cat_backward needs only sizes (no tensor).
struct L2PDecoderActs(Movable):
    # enc conv inputs (NHWC) + pre-silu conv outputs
    var x_nhwc: Tensor          # [1,H,W,3]      enc1 conv input
    var enc1_pre: Tensor        # [1,H,W,64]     enc1 conv out (pre-silu)
    var e1: Tensor              # [1,H,W,64]     enc1 post-silu (= maxpool1 input & skip)
    var p1: Tensor              # [1,H/2,W/2,64] enc2 conv input
    var enc2_pre: Tensor
    var e2: Tensor
    var p2: Tensor
    var enc3_pre: Tensor
    var e3: Tensor
    var p3: Tensor
    var enc4_pre: Tensor
    var e4: Tensor
    var p4: Tensor              # [1,gh,gw,512]  bottleneck cat input (channels-first half)
    # bottleneck
    var bot_cat: Tensor         # [1,gh,gw,512+3840]  bottleneck conv input
    var bot_pre: Tensor         # [1,gh,gw,512]       bottleneck conv out (pre-silu)
    var b: Tensor               # [1,gh,gw,512]       post-silu bottleneck
    # up/dec stage 4
    var up4_in: Tensor          # upsample(b)              up4 conv input [1,2gh,2gw,512]
    var dec4_cat: Tensor        # cat[up4, e4]             dec4 conv input
    var dec4_pre: Tensor
    var d4: Tensor
    # stage 3
    var up3_in: Tensor
    var dec3_cat: Tensor
    var dec3_pre: Tensor
    var d3: Tensor
    # stage 2
    var up2_in: Tensor
    var dec2_cat: Tensor
    var dec2_pre: Tensor
    var d2: Tensor
    # stage 1
    var up1_in: Tensor
    var dec1_cat: Tensor
    var dec1_pre: Tensor
    var d1: Tensor              # out_conv input

    def __init__(
        out self,
        var x_nhwc: Tensor,
        var enc1_pre: Tensor, var e1: Tensor, var p1: Tensor,
        var enc2_pre: Tensor, var e2: Tensor, var p2: Tensor,
        var enc3_pre: Tensor, var e3: Tensor, var p3: Tensor,
        var enc4_pre: Tensor, var e4: Tensor, var p4: Tensor,
        var bot_cat: Tensor, var bot_pre: Tensor, var b: Tensor,
        var up4_in: Tensor, var dec4_cat: Tensor, var dec4_pre: Tensor, var d4: Tensor,
        var up3_in: Tensor, var dec3_cat: Tensor, var dec3_pre: Tensor, var d3: Tensor,
        var up2_in: Tensor, var dec2_cat: Tensor, var dec2_pre: Tensor, var d2: Tensor,
        var up1_in: Tensor, var dec1_cat: Tensor, var dec1_pre: Tensor, var d1: Tensor,
    ):
        self.x_nhwc = x_nhwc^
        self.enc1_pre = enc1_pre^; self.e1 = e1^; self.p1 = p1^
        self.enc2_pre = enc2_pre^; self.e2 = e2^; self.p2 = p2^
        self.enc3_pre = enc3_pre^; self.e3 = e3^; self.p3 = p3^
        self.enc4_pre = enc4_pre^; self.e4 = e4^; self.p4 = p4^
        self.bot_cat = bot_cat^; self.bot_pre = bot_pre^; self.b = b^
        self.up4_in = up4_in^; self.dec4_cat = dec4_cat^; self.dec4_pre = dec4_pre^; self.d4 = d4^
        self.up3_in = up3_in^; self.dec3_cat = dec3_cat^; self.dec3_pre = dec3_pre^; self.d3 = d3^
        self.up2_in = up2_in^; self.dec2_cat = dec2_cat^; self.dec2_pre = dec2_pre^; self.d2 = d2^
        self.up1_in = up1_in^; self.dec1_cat = dec1_cat^; self.dec1_pre = dec1_pre^; self.d1 = d1^


struct L2PDecoderFwd(Movable):
    var pred_nchw: Tensor   # [1,3,H,W]  pixel prediction
    var acts: L2PDecoderActs

    def __init__(out self, var pred_nchw: Tensor, var acts: L2PDecoderActs):
        self.pred_nchw = pred_nchw^
        self.acts = acts^


def _to_nhwc(x_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
    var p = List[Int](); p.append(0); p.append(2); p.append(3); p.append(1)
    return permute(x_nchw, p^, ctx)


def _to_nchw(x_nhwc: Tensor, ctx: DeviceContext) raises -> Tensor:
    var p = List[Int](); p.append(0); p.append(3); p.append(1); p.append(2)
    return permute(x_nhwc, p^, ctx)


# ── FORWARD (F32, saves all activations) ──────────────────────────────────────
# noisy_nchw: [1,3,H,W] F32 ;  feat_nchw: [1,3840,gh,gw] F32 where gh=H/16.
# Returns pred [1,3,H,W] F32 + acts. H,W,gh,gw are comptime (static conv shapes).
def l2p_decoder_forward[H: Int, W: Int, GH: Int, GW: Int](
    dec: L2PDecoderF32, noisy_nchw: Tensor, feat_nchw: Tensor, ctx: DeviceContext
) raises -> L2PDecoderFwd:
    comptime assert H % 16 == 0 and W % 16 == 0, "L2P decoder needs H/W divisible by 16"
    comptime assert GH == H // 16 and GW == W // 16, "feat grid must be H/16 x W/16"
    # The 4 maxpools reach H/16 == GH exactly, so feat aligns with p4 (NO interpolate).
    comptime H2 = H // 2; comptime W2 = W // 2
    comptime H4 = H // 4; comptime W4 = W // 4
    comptime H8 = H // 8; comptime W8 = W // 8

    var x = _to_nhwc(_f32(noisy_nchw, ctx), ctx)      # [1,H,W,3]

    # enc1
    var enc1_pre = conv2d[1, H, W, ZIMAGE_L2P_PIXEL_CHANNELS, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
        x.clone(ctx), dec.enc1_w.clone(ctx), Optional[Tensor](dec.enc1_b.clone(ctx)), ctx)
    var e1 = silu(enc1_pre.clone(ctx), ctx)
    var p1 = maxpool2x2_nhwc(e1.clone(ctx), ctx)      # [1,H2,W2,64]
    # enc2
    var enc2_pre = conv2d[1, H2, W2, ZIMAGE_L2P_LD_C1, 3, 3, ZIMAGE_L2P_LD_C2, 1, 1, 1, 1](
        p1.clone(ctx), dec.enc2_w.clone(ctx), Optional[Tensor](dec.enc2_b.clone(ctx)), ctx)
    var e2 = silu(enc2_pre.clone(ctx), ctx)
    var p2 = maxpool2x2_nhwc(e2.clone(ctx), ctx)      # [1,H4,W4,128]
    # enc3
    var enc3_pre = conv2d[1, H4, W4, ZIMAGE_L2P_LD_C2, 3, 3, ZIMAGE_L2P_LD_C3, 1, 1, 1, 1](
        p2.clone(ctx), dec.enc3_w.clone(ctx), Optional[Tensor](dec.enc3_b.clone(ctx)), ctx)
    var e3 = silu(enc3_pre.clone(ctx), ctx)
    var p3 = maxpool2x2_nhwc(e3.clone(ctx), ctx)      # [1,H8,W8,256]
    # enc4
    var enc4_pre = conv2d[1, H8, W8, ZIMAGE_L2P_LD_C3, 3, 3, ZIMAGE_L2P_LD_C4, 1, 1, 1, 1](
        p3.clone(ctx), dec.enc4_w.clone(ctx), Optional[Tensor](dec.enc4_b.clone(ctx)), ctx)
    var e4 = silu(enc4_pre.clone(ctx), ctx)
    var p4 = maxpool2x2_nhwc(e4.clone(ctx), ctx)      # [1,GH,GW,512]

    # bottleneck: cat[p4, feat_nhwc] -> conv1x1 -> silu.  feat is at GH×GW already.
    var feat_nhwc = _to_nhwc(_f32(feat_nchw, ctx), ctx)   # [1,GH,GW,3840]
    var bot_cat = concat(3, ctx, p4.clone(ctx), feat_nhwc)   # [1,GH,GW,512+3840]
    var bot_pre = conv2d[1, GH, GW, ZIMAGE_L2P_LD_C4 + ZIMAGE_L2P_HIDDEN, 1, 1, ZIMAGE_L2P_LD_C4, 1, 1, 0, 0](
        bot_cat.clone(ctx), dec.bot_w.clone(ctx), Optional[Tensor](dec.bot_b.clone(ctx)), ctx)
    var b = silu(bot_pre.clone(ctx), ctx)            # [1,GH,GW,512]

    # dec4: up4 = conv3x3(upsample2x(b)); cat[up4,e4]; conv3x3 -> 256; silu
    var up4_in = upsample_nearest2x_nhwc(b.clone(ctx), ctx)   # [1,H8,W8,512]
    var up4 = conv2d[1, H8, W8, ZIMAGE_L2P_LD_C4, 3, 3, ZIMAGE_L2P_LD_C4, 1, 1, 1, 1](
        up4_in.clone(ctx), dec.up4_w.clone(ctx), Optional[Tensor](dec.up4_b.clone(ctx)), ctx)
    var dec4_cat = concat(3, ctx, up4, e4.clone(ctx))         # [1,H8,W8,512+512]
    var dec4_pre = conv2d[1, H8, W8, ZIMAGE_L2P_LD_C4 + ZIMAGE_L2P_LD_C4, 3, 3, ZIMAGE_L2P_LD_C3, 1, 1, 1, 1](
        dec4_cat.clone(ctx), dec.dec4_w.clone(ctx), Optional[Tensor](dec.dec4_b.clone(ctx)), ctx)
    var d4 = silu(dec4_pre.clone(ctx), ctx)          # [1,H8,W8,256]

    # dec3
    var up3_in = upsample_nearest2x_nhwc(d4.clone(ctx), ctx)  # [1,H4,W4,256]
    var up3 = conv2d[1, H4, W4, ZIMAGE_L2P_LD_C3, 3, 3, ZIMAGE_L2P_LD_C3, 1, 1, 1, 1](
        up3_in.clone(ctx), dec.up3_w.clone(ctx), Optional[Tensor](dec.up3_b.clone(ctx)), ctx)
    var dec3_cat = concat(3, ctx, up3, e3.clone(ctx))
    var dec3_pre = conv2d[1, H4, W4, ZIMAGE_L2P_LD_C3 + ZIMAGE_L2P_LD_C3, 3, 3, ZIMAGE_L2P_LD_C2, 1, 1, 1, 1](
        dec3_cat.clone(ctx), dec.dec3_w.clone(ctx), Optional[Tensor](dec.dec3_b.clone(ctx)), ctx)
    var d3 = silu(dec3_pre.clone(ctx), ctx)          # [1,H4,W4,128]

    # dec2
    var up2_in = upsample_nearest2x_nhwc(d3.clone(ctx), ctx)  # [1,H2,W2,128]
    var up2 = conv2d[1, H2, W2, ZIMAGE_L2P_LD_C2, 3, 3, ZIMAGE_L2P_LD_C2, 1, 1, 1, 1](
        up2_in.clone(ctx), dec.up2_w.clone(ctx), Optional[Tensor](dec.up2_b.clone(ctx)), ctx)
    var dec2_cat = concat(3, ctx, up2, e2.clone(ctx))
    var dec2_pre = conv2d[1, H2, W2, ZIMAGE_L2P_LD_C2 + ZIMAGE_L2P_LD_C2, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
        dec2_cat.clone(ctx), dec.dec2_w.clone(ctx), Optional[Tensor](dec.dec2_b.clone(ctx)), ctx)
    var d2 = silu(dec2_pre.clone(ctx), ctx)          # [1,H2,W2,64]

    # dec1
    var up1_in = upsample_nearest2x_nhwc(d2.clone(ctx), ctx)  # [1,H,W,64]
    var up1 = conv2d[1, H, W, ZIMAGE_L2P_LD_C1, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
        up1_in.clone(ctx), dec.up1_w.clone(ctx), Optional[Tensor](dec.up1_b.clone(ctx)), ctx)
    var dec1_cat = concat(3, ctx, up1, e1.clone(ctx))
    var dec1_pre = conv2d[1, H, W, ZIMAGE_L2P_LD_C1 + ZIMAGE_L2P_LD_C1, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
        dec1_cat.clone(ctx), dec.dec1_w.clone(ctx), Optional[Tensor](dec.dec1_b.clone(ctx)), ctx)
    var d1 = silu(dec1_pre.clone(ctx), ctx)          # [1,H,W,64]

    # out_conv 1x1 -> 3
    var out_nhwc = conv2d[1, H, W, ZIMAGE_L2P_LD_C1, 1, 1, ZIMAGE_L2P_PIXEL_CHANNELS, 1, 1, 0, 0](
        d1.clone(ctx), dec.out_w.clone(ctx), Optional[Tensor](dec.out_b.clone(ctx)), ctx)
    var pred_nchw = _to_nchw(out_nhwc, ctx)          # [1,3,H,W]

    var acts = L2PDecoderActs(
        x^, enc1_pre^, e1^, p1^, enc2_pre^, e2^, p2^,
        enc3_pre^, e3^, p3^, enc4_pre^, e4^, p4^,
        bot_cat^, bot_pre^, b^,
        up4_in^, dec4_cat^, dec4_pre^, d4^,
        up3_in^, dec3_cat^, dec3_pre^, d3^,
        up2_in^, dec2_cat^, dec2_pre^, d2^,
        up1_in^, dec1_cat^, dec1_pre^, d1^,
    )
    return L2PDecoderFwd(pred_nchw^, acts^)


# ── BACKWARD: d_pred[1,3,H,W] -> d_feat[1,3840,GH,GW] (decoder FROZEN) ─────────
# Mirrors the forward in reverse. conv2d_backward returns d_x/d_w/d_b; we use
# only d_x (weights frozen). cat_backward splits the skip-concat grads. The
# encoder skip grads (to e4/e3/e2/e1) are summed into the encoder backward path.
def l2p_decoder_backward[H: Int, W: Int, GH: Int, GW: Int](
    dec: L2PDecoderF32, acts: L2PDecoderActs, d_pred_nchw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    comptime H2 = H // 2; comptime W2 = W // 2
    comptime H4 = H // 4; comptime W4 = W // 4
    comptime H8 = H // 8; comptime W8 = W // 8

    var d_out_nhwc = _to_nhwc(_f32(d_pred_nchw, ctx), ctx)   # [1,H,W,3]

    # out_conv 1x1 backward -> d_d1
    var go = conv2d_backward[1, H, W, ZIMAGE_L2P_LD_C1, 1, 1, ZIMAGE_L2P_PIXEL_CHANNELS, 1, 1, 0, 0](
        acts.d1.clone(ctx), dec.out_w.clone(ctx), d_out_nhwc, ctx)
    var d_d1 = go.d_x.clone(ctx)                              # [1,H,W,64]

    # dec1: silu -> conv3x3 -> cat[up1,e1]
    var d_dec1_pre = silu_backward(d_d1, acts.dec1_pre.clone(ctx), ctx)
    var gdec1 = conv2d_backward[1, H, W, ZIMAGE_L2P_LD_C1 + ZIMAGE_L2P_LD_C1, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
        acts.dec1_cat.clone(ctx), dec.dec1_w.clone(ctx), d_dec1_pre, ctx)
    var cg1 = cat_backward(gdec1.d_x.clone(ctx), ZIMAGE_L2P_LD_C1, ZIMAGE_L2P_LD_C1, 3, ctx)
    var d_up1 = cg1.d_0.clone(ctx)        # -> up1 conv
    var d_e1_skip = cg1.d_1.clone(ctx)    # -> encoder e1
    # up1 conv3x3 backward -> d_up1_in ; then upsample backward -> d_d2
    var gup1 = conv2d_backward[1, H, W, ZIMAGE_L2P_LD_C1, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
        acts.up1_in.clone(ctx), dec.up1_w.clone(ctx), d_up1, ctx)
    var d_d2 = upsample_nearest2d_backward[1, H2, W2, ZIMAGE_L2P_LD_C1, 2](gup1.d_x.clone(ctx), ctx)

    # dec2
    var d_dec2_pre = silu_backward(d_d2, acts.dec2_pre.clone(ctx), ctx)
    var gdec2 = conv2d_backward[1, H2, W2, ZIMAGE_L2P_LD_C2 + ZIMAGE_L2P_LD_C2, 3, 3, ZIMAGE_L2P_LD_C1, 1, 1, 1, 1](
        acts.dec2_cat.clone(ctx), dec.dec2_w.clone(ctx), d_dec2_pre, ctx)
    var cg2 = cat_backward(gdec2.d_x.clone(ctx), ZIMAGE_L2P_LD_C2, ZIMAGE_L2P_LD_C2, 3, ctx)
    var d_up2 = cg2.d_0.clone(ctx)
    var d_e2_skip = cg2.d_1.clone(ctx)
    var gup2 = conv2d_backward[1, H2, W2, ZIMAGE_L2P_LD_C2, 3, 3, ZIMAGE_L2P_LD_C2, 1, 1, 1, 1](
        acts.up2_in.clone(ctx), dec.up2_w.clone(ctx), d_up2, ctx)
    var d_d3 = upsample_nearest2d_backward[1, H4, W4, ZIMAGE_L2P_LD_C2, 2](gup2.d_x.clone(ctx), ctx)

    # dec3
    var d_dec3_pre = silu_backward(d_d3, acts.dec3_pre.clone(ctx), ctx)
    var gdec3 = conv2d_backward[1, H4, W4, ZIMAGE_L2P_LD_C3 + ZIMAGE_L2P_LD_C3, 3, 3, ZIMAGE_L2P_LD_C2, 1, 1, 1, 1](
        acts.dec3_cat.clone(ctx), dec.dec3_w.clone(ctx), d_dec3_pre, ctx)
    var cg3 = cat_backward(gdec3.d_x.clone(ctx), ZIMAGE_L2P_LD_C3, ZIMAGE_L2P_LD_C3, 3, ctx)
    var d_up3 = cg3.d_0.clone(ctx)
    var d_e3_skip = cg3.d_1.clone(ctx)
    var gup3 = conv2d_backward[1, H4, W4, ZIMAGE_L2P_LD_C3, 3, 3, ZIMAGE_L2P_LD_C3, 1, 1, 1, 1](
        acts.up3_in.clone(ctx), dec.up3_w.clone(ctx), d_up3, ctx)
    var d_d4 = upsample_nearest2d_backward[1, H8, W8, ZIMAGE_L2P_LD_C3, 2](gup3.d_x.clone(ctx), ctx)

    # dec4
    var d_dec4_pre = silu_backward(d_d4, acts.dec4_pre.clone(ctx), ctx)
    var gdec4 = conv2d_backward[1, H8, W8, ZIMAGE_L2P_LD_C4 + ZIMAGE_L2P_LD_C4, 3, 3, ZIMAGE_L2P_LD_C3, 1, 1, 1, 1](
        acts.dec4_cat.clone(ctx), dec.dec4_w.clone(ctx), d_dec4_pre, ctx)
    var cg4 = cat_backward(gdec4.d_x.clone(ctx), ZIMAGE_L2P_LD_C4, ZIMAGE_L2P_LD_C4, 3, ctx)
    var d_up4 = cg4.d_0.clone(ctx)
    var d_e4_skip = cg4.d_1.clone(ctx)
    var gup4 = conv2d_backward[1, H8, W8, ZIMAGE_L2P_LD_C4, 3, 3, ZIMAGE_L2P_LD_C4, 1, 1, 1, 1](
        acts.up4_in.clone(ctx), dec.up4_w.clone(ctx), d_up4, ctx)
    var d_b = upsample_nearest2d_backward[1, GH, GW, ZIMAGE_L2P_LD_C4, 2](gup4.d_x.clone(ctx), ctx)

    # bottleneck: silu -> conv1x1(cat[p4,feat])  ; split cat grads -> d_feat
    var d_bot_pre = silu_backward(d_b, acts.bot_pre.clone(ctx), ctx)
    var gbot = conv2d_backward[1, GH, GW, ZIMAGE_L2P_LD_C4 + ZIMAGE_L2P_HIDDEN, 1, 1, ZIMAGE_L2P_LD_C4, 1, 1, 0, 0](
        acts.bot_cat.clone(ctx), dec.bot_w.clone(ctx), d_bot_pre, ctx)
    var cgb = cat_backward(gbot.d_x.clone(ctx), ZIMAGE_L2P_LD_C4, ZIMAGE_L2P_HIDDEN, 3, ctx)
    var d_p4 = cgb.d_0.clone(ctx)        # -> encoder p4 (maxpool of e4) — NOT needed for d_feat
    var d_feat_nhwc = cgb.d_1.clone(ctx) # [1,GH,GW,3840]  the feature-map gradient

    # The encoder branch (d_p4 + the e4/e3/e2/e1 skip grads) does NOT contribute
    # to d_feat: the encoder consumes ONLY `noisy` (a detached training input,
    # requires_grad=False), so its gradient terminates at the noisy image and is
    # discarded. ai-toolkit's autograd does the same — noisy is built under
    # no_grad / detached, so d_noisy is never propagated. We therefore drop
    # d_p4 and the d_e*_skip terms. (They are computed above only because
    # cat_backward returns both halves; keeping the reads documents intent.)
    _ = d_p4
    _ = d_e1_skip
    _ = d_e2_skip
    _ = d_e3_skip
    _ = d_e4_skip

    return _to_nchw(d_feat_nhwc, ctx)    # [1,3840,GH,GW]
