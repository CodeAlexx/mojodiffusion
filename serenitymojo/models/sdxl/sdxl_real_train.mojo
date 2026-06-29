# models/sdxl/sdxl_real_train.mojo — REAL-DIMS SDXL conv-UNet, trainable.
#
# B1+B2+B3 (the real-run blocker): an activation-saving FORWARD + a hand-chained
# BACKWARD at the REAL SDXL topology (MC=320, ch_mult (1,2,4), num_res_blocks=2,
# ctx_dim 2048, adm 2816, head_dim 64) with LoRA on every SpatialTransformer's
# 10 projections. COMPOSES the parity-gated units (resblock/spatial_transformer-
# LoRA/sampling/embed fwd+bwd + conv2d_backward + group_norm/silu/cat backward) at
# real comptime dims — the reduced stack (sdxl_unet_stack.mojo) is the line-by-line
# backward template; only the comptime dims + the exact real topology change.
# NO new ops/ primitive (Tenet 1).
#
# TOPOLOGY (verified line-by-line vs models/dit/sdxl_unet.mojo::forward, the
# real-dims inference reference). Latent spatial H0=W0=L (comptime); H1=L/2; H2=L/4.
#   conv_in(4->320)@H0                                            -> skip[0]
#   in1 Res(320->320)@H0                                          -> skip[1]
#   in2 Res(320->320)@H0                                          -> skip[2]
#   in3 Down(320)@H0->H1                                          -> skip[3]
#   in4 Res(320->640)+ST(640,h10,d2)@H1                           -> skip[4]
#   in5 Res(640->640)+ST(640,h10,d2)@H1                           -> skip[5]
#   in6 Down(640)@H1->H2                                          -> skip[6]
#   in7 Res(640->1280)+ST(1280,h20,d10)@H2                        -> skip[7]
#   in8 Res(1280->1280)+ST(1280,h20,d10)@H2                       -> skip[8]
#   mid: Res(1280)+ST(1280,h20,d10)+Res(1280)@H2
#   out0 cat(h,skip[8])=2560 Res->1280 +ST(1280,d10)@H2
#   out1 cat(h,skip[7])=2560 Res->1280 +ST(1280,d10)@H2
#   out2 cat(h,skip[6]=640)=1920 Res->1280 +ST(1280,d10) +Up@H2->H1
#   out3 cat(h,skip[5]=640)=1920 Res->640  +ST(640,d2)@H1
#   out4 cat(h,skip[4]=640)=1280 Res->640  +ST(640,d2)@H1
#   out5 cat(h,skip[3]=320)=960  Res->640  +ST(640,d2) +Up@H1->H0
#   out6 cat(h,skip[2]=320)=960  Res->320@H0
#   out7 cat(h,skip[1]=320)=640  Res->320@H0
#   out8 cat(h,skip[0]=320)=640  Res->320@H0
#   final: GN32(eps1e-5)->SiLU->conv_out(320->4)@H0
#
# 11 SpatialTransformers (LoRA targets), indexed 0..10 in the order they run:
#   0 in4(640,h10,d2)  1 in5(640,h10,d2)  2 in7(1280,h20,d10)  3 in8(1280,h20,d10)
#   4 mid(1280,h20,d10) 5 out0(1280,d10)  6 out1(1280,d10)     7 out2(1280,d10)
#   8 out3(640,d2)      9 out4(640,d2)    10 out5(640,d2)
#
# LoRA carrier = a List[SdxlLoraSet], one per ST (each ST has its own C/Cff/depth).
# The 3 distinct ST configs are dispatched to their comptime sdxl_st_lora_{fwd,bwd}.
#
# All NHWC, F32 interior (matches the gated units + the Rust FP32 residual stream).
# Latent enters NCHW [1,4,L,L]; converted to NHWC once at entry, NCHW at exit.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.collections import List, Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.conv2d_backward import conv2d_backward
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.norm_backward import group_norm_backward
from serenitymojo.ops.activations import silu
from serenitymojo.ops.activation_backward import silu_backward
from serenitymojo.ops.tensor_algebra import add, concat
from serenitymojo.ops.shape_backward import cat_backward
from serenitymojo.models.vae.decoder2d import nchw_to_nhwc, nhwc_to_nchw

from serenitymojo.models.sdxl.block import (
    resblock_forward, resblock_backward, ResBlockActs,
)
from serenitymojo.models.sdxl.weights import ResBlockWeights
from serenitymojo.models.sdxl.config import GN_EPS_RES
from serenitymojo.models.sdxl.embed import (
    embed_forward, embed_backward, EmbWeights, EmbActs,
)
from serenitymojo.models.sdxl.sampling import (
    downsample_forward, downsample_backward, upsample_forward, upsample_backward,
)
from serenitymojo.models.sdxl.spatial_transformer import SpatialTransformerWeights
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import (
    SdxlLoraSet, SdxlStLoraActs, SdxlStLoraFwd, SdxlStLoraGrads,
    sdxl_st_lora_forward, sdxl_st_lora_backward,
)
from serenitymojo.models.sdxl.lora_block import SDXL_SLOTS

comptime TArc = ArcPointer[Tensor]

# ── fixed SDXL arch comptimes ─────────────────────────────────────────────────
comptime MC = 320
comptime C1 = 640
comptime C2 = 1280
comptime ADM = 2816
comptime TEMB = 1280
comptime SDIM = 320       # sinusoidal dim = model_channels
comptime NKV = 77
comptime CCTX = 2048
comptime HD = 64
comptime G = 32
comptime H10 = 10         # 640/64
comptime H20 = 20         # 1280/64
comptime CFF1 = 2560      # ST C=640  -> GEGLU inner half (ff.net.0.proj out = 2*Cff = 5120)
comptime CFF2 = 5120      # ST C=1280 -> inner half (proj out = 10240)
comptime D2 = 2           # transformer depth at level 1
comptime D10 = 10         # transformer depth at level 2
comptime EPS_F = Float32(1e-6)  # final GN uses 1e-5 (GN_EPS_RES); ST uses 1e-6 internally


# ── tiny shape helper ─────────────────────────────────────────────────────────
def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


def _clone_rb_acts(a: ResBlockActs, ctx: DeviceContext) raises -> ResBlockActs:
    return ResBlockActs(
        a.x.clone(ctx), a.h1.clone(ctx), a.s1.clone(ctx),
        a.emb_in.clone(ctx), a.e.clone(ctx), a.h2.clone(ctx),
        a.h3.clone(ctx), a.s2.clone(ctx),
    )


def _clone_emb_acts(a: EmbActs, ctx: DeviceContext) raises -> EmbActs:
    return EmbActs(
        a.ts.clone(ctx), a.t0lin.clone(ctx), a.t0silu.clone(ctx),
        a.y.clone(ctx), a.l0lin.clone(ctx), a.l0silu.clone(ctx),
    )


# Widen every saved embedding activation to F32 so the (discarded) embed backward
# runs against the F32-activation/BF16-weight contract (mixed_base) the rest of
# the real-weight backward uses. embed_forward itself stays in the stored (BF16)
# dtype internally; only the saved acts + the shared `emb` output are widened.
def _f32(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    if t.dtype() == STDtype.F32:
        return t.clone(ctx)
    return cast_tensor(t, STDtype.F32, ctx)


def _clone_emb_acts_f32(a: EmbActs, ctx: DeviceContext) raises -> EmbActs:
    return EmbActs(
        _f32(a.ts, ctx), _f32(a.t0lin, ctx), _f32(a.t0silu, ctx),
        _f32(a.y, ctx), _f32(a.l0lin, ctx), _f32(a.l0silu, ctx),
    )


# ── conv_in / conv_out 3x3 pad1 stride1 ───────────────────────────────────────
def _conv3x3_fwd[
    H: Int, W: Int, Cin: Int, Cout: Int,
](x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return conv2d[1, H, W, Cin, 3, 3, Cout, 1, 1, 1, 1](
        x, w.clone(ctx), Optional[Tensor](b.clone(ctx)), ctx)


struct ConvGrads(Movable):
    var d_x: Tensor; var d_w: Tensor; var d_b: Tensor
    def __init__(out self, var d_x: Tensor, var d_w: Tensor, var d_b: Tensor):
        self.d_x = d_x^; self.d_w = d_w^; self.d_b = d_b^


def _conv3x3_bwd[
    H: Int, W: Int, Cin: Int, Cout: Int,
](go: Tensor, x: Tensor, w: Tensor, ctx: DeviceContext) raises -> ConvGrads:
    var g = conv2d_backward[1, H, W, Cin, 3, 3, Cout, 1, 1, 1, 1](x, w, go, ctx)
    return ConvGrads(g.d_x.clone(ctx), g.d_w.clone(ctx), g.d_b.clone(ctx))


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT BUNDLE — every base weight the real UNet uses (frozen for LoRA training).
# Conv-in/out + final-GN are loose Tensors; ResBlocks + STs are their gated structs.
# ResBlockWeights is move-only; ST weights Copyable. 14 ResBlocks + 11 STs.
# ═══════════════════════════════════════════════════════════════════════════════
struct SdxlRealWeights(Movable):
    var emb: EmbWeights
    var conv_in_w: Tensor; var conv_in_b: Tensor
    var out_gn_w: Tensor; var out_gn_b: Tensor
    var conv_out_w: Tensor; var conv_out_b: Tensor
    # ResBlocks in run order:
    #  in1,in2 (H0,320), in4,in5 (H1,640), in7,in8 (H2,1280),
    #  mid0,mid2 (H2,1280), out0,out1,out2 (H2->1280), out3,out4,out5 (H1->640),
    #  out6,out7,out8 (H0->320)   -> 17 ResBlocks total
    var res: List[ArcPointer[ResBlockWeights]]   # move-only -> Arc
    # downsamples in3 (320), in6 (640) ; upsamples out2 (1280), out5 (640)
    var down_w: List[TArc]; var down_b: List[TArc]
    var up_w: List[TArc]; var up_b: List[TArc]
    # 11 STs in run order (see header indexing)
    var st: List[SpatialTransformerWeights]

    def __init__(
        out self, var emb: EmbWeights,
        var conv_in_w: Tensor, var conv_in_b: Tensor,
        var out_gn_w: Tensor, var out_gn_b: Tensor,
        var conv_out_w: Tensor, var conv_out_b: Tensor,
        var res: List[ArcPointer[ResBlockWeights]],
        var down_w: List[TArc], var down_b: List[TArc],
        var up_w: List[TArc], var up_b: List[TArc],
        var st: List[SpatialTransformerWeights],
    ):
        self.emb = emb^
        self.conv_in_w = conv_in_w^; self.conv_in_b = conv_in_b^
        self.out_gn_w = out_gn_w^; self.out_gn_b = out_gn_b^
        self.conv_out_w = conv_out_w^; self.conv_out_b = conv_out_b^
        self.res = res^; self.down_w = down_w^; self.down_b = down_b^
        self.up_w = up_w^; self.up_b = up_b^; self.st = st^


# ResBlock index names (positions in `res`)
comptime R_IN1 = 0
comptime R_IN2 = 1
comptime R_IN4 = 2
comptime R_IN5 = 3
comptime R_IN7 = 4
comptime R_IN8 = 5
comptime R_MID0 = 6
comptime R_MID2 = 7
comptime R_OUT0 = 8
comptime R_OUT1 = 9
comptime R_OUT2 = 10
comptime R_OUT3 = 11
comptime R_OUT4 = 12
comptime R_OUT5 = 13
comptime R_OUT6 = 14
comptime R_OUT7 = 15
comptime R_OUT8 = 16
comptime N_RES = 17

# ST index names (positions in `st` and in the LoRA set list)
comptime ST_IN4 = 0
comptime ST_IN5 = 1
comptime ST_IN7 = 2
comptime ST_IN8 = 3
comptime ST_MID = 4
comptime ST_OUT0 = 5
comptime ST_OUT1 = 6
comptime ST_OUT2 = 7
comptime ST_OUT3 = 8
comptime ST_OUT4 = 9
comptime ST_OUT5 = 10
comptime N_ST = 11


# ═══════════════════════════════════════════════════════════════════════════════
# SAVED ACTS. ResBlock + ST acts kept in run-order lists; skips + conv/final
# inputs in TArc lists. Indexing mirrors the weight lists.
# ═══════════════════════════════════════════════════════════════════════════════
struct SdxlRealActs(Movable):
    var emb_acts: EmbActs
    var emb: Tensor                            # [1,1280] shared
    var conv_in_x: Tensor                      # conv_in input NHWC [1,L,L,4]
    var res_acts: List[ArcPointer[ResBlockActs]]    # N_RES, run order (move-only -> Arc)
    var st_acts: List[ArcPointer[SdxlStLoraActs]]   # N_ST, run order (move-only -> Arc)
    var down_x: List[TArc]                     # 2 downsample inputs (in3,in6)
    var up_up: List[TArc]                      # 2 saved upsampled conv-inputs (out2,out5)
    var final_gn_in: Tensor
    var final_silu_in: Tensor
    var conv_out_in: Tensor

    def __init__(
        out self, var emb_acts: EmbActs, var emb: Tensor, var conv_in_x: Tensor,
        var res_acts: List[ArcPointer[ResBlockActs]], var st_acts: List[ArcPointer[SdxlStLoraActs]],
        var down_x: List[TArc], var up_up: List[TArc],
        var final_gn_in: Tensor, var final_silu_in: Tensor, var conv_out_in: Tensor,
    ):
        self.emb_acts = emb_acts^; self.emb = emb^; self.conv_in_x = conv_in_x^
        self.res_acts = res_acts^; self.st_acts = st_acts^
        self.down_x = down_x^; self.up_up = up_up^
        self.final_gn_in = final_gn_in^; self.final_silu_in = final_silu_in^
        self.conv_out_in = conv_out_in^


struct SdxlRealFwd(Movable):
    var out: Tensor       # eps NHWC [1,L,L,4]
    var acts: SdxlRealActs
    def __init__(out self, var out: Tensor, var acts: SdxlRealActs):
        self.out = out^; self.acts = acts^


# collected LoRA grads (flat per-ST) + load-bearing input grads.
struct SdxlRealGrads(Movable):
    var d_a: List[List[List[Float32]]]   # [ST][slot] flat
    var d_b: List[List[List[Float32]]]
    var d_x: Tensor                      # grad wrt latent NHWC [1,L,L,4]
    var nonfinite: Int

    def __init__(
        out self, var d_a: List[List[List[Float32]]], var d_b: List[List[List[Float32]]],
        var d_x: Tensor, nonfinite: Int,
    ):
        self.d_a = d_a^; self.d_b = d_b^; self.d_x = d_x^; self.nonfinite = nonfinite


# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD (real dims). x_nhwc latent [1,L,L,4]; t [1] timestep; y [1,2816] ADM;
# context [1,77,2048]. comptime L = latent spatial.
# ═══════════════════════════════════════════════════════════════════════════════
def sdxl_real_forward[H: Int, W: Int](
    x_nhwc: Tensor, t: Tensor, y: Tensor, context_in: Tensor,
    w: SdxlRealWeights, lora: List[SdxlLoraSet], ctx: DeviceContext,
) raises -> SdxlRealFwd:
    # Independent H,W threaded through every stage. The conv-UNet downsamples by
    # 2 three times; H{0,1,2} = H, H/2, H/4 ; W{0,1,2} = W, W/2, W/4. (Square = H==W.)
    comptime H0 = H
    comptime H1 = H // 2
    comptime H2 = H // 4
    comptime W0 = W
    comptime W1 = W // 2
    comptime W2 = W // 4

    # Real-weight contract: F32 activation stream + frozen BF16 stored weights
    # (mixed_base). Widen the latent + the shared `emb` + context to F32 once at
    # entry; every weight-bearing op (linear/conv2d/group_norm) consumes F32 acts
    # against BF16 weights and returns F32. embed_forward stays BF16 internally.
    var x_f = _f32(x_nhwc, ctx)
    var context = _f32(context_in, ctx)

    var y_emb = cast_tensor(y, w.emb.l0_w.dtype(), ctx)
    var ef = embed_forward[1, SDIM, TEMB, ADM](t, y_emb, w.emb, ctx)
    var a_emb = _clone_emb_acts_f32(ef.acts, ctx)
    var emb = _f32(ef.emb, ctx)

    var res_acts = List[ArcPointer[ResBlockActs]]()
    var st_acts = List[ArcPointer[SdxlStLoraActs]]()
    var down_x = List[TArc]()
    var up_up = List[TArc]()
    var skips = List[TArc]()    # LIFO skip stack

    # conv_in (4->320)
    var conv_in_x = x_f.clone(ctx)
    var h = _conv3x3_fwd[H0, W0, 4, MC](x_f, w.conv_in_w, w.conv_in_b, ctx)
    skips.append(TArc(h.clone(ctx)))                                   # skip[0]

    # in1 Res(320->320)
    var rb1 = resblock_forward[1, H0, W0, MC, MC, TEMB, G](h, emb, w.res[R_IN1][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(rb1.acts, ctx))); h = rb1.out.clone(ctx)
    skips.append(TArc(h.clone(ctx)))                                   # skip[1]
    # in2 Res(320->320)
    var rb2 = resblock_forward[1, H0, W0, MC, MC, TEMB, G](h, emb, w.res[R_IN2][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(rb2.acts, ctx))); h = rb2.out.clone(ctx)
    skips.append(TArc(h.clone(ctx)))                                   # skip[2]
    # in3 Down(320) H0->H1
    down_x.append(TArc(h.clone(ctx)))
    h = downsample_forward[1, H0, W0, MC](h, w.down_w[0][], w.down_b[0][], ctx)
    skips.append(TArc(h.clone(ctx)))                                   # skip[3]

    # in4 Res(320->640)+ST(640,h10,d2)
    var rb4 = resblock_forward[1, H1, W1, MC, C1, TEMB, G](h, emb, w.res[R_IN4][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(rb4.acts, ctx))); h = rb4.out.clone(ctx)
    var s4 = sdxl_st_lora_forward[1, H1, W1, C1, NKV, CCTX, H10, HD, CFF1, G, D2](
        h, context.clone(ctx), w.st[ST_IN4], lora[ST_IN4], ctx)
    st_acts.append(ArcPointer(s4.acts.copy())); h = s4.out.clone(ctx)
    skips.append(TArc(h.clone(ctx)))                                   # skip[4]
    # in5 Res(640->640)+ST
    var rb5 = resblock_forward[1, H1, W1, C1, C1, TEMB, G](h, emb, w.res[R_IN5][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(rb5.acts, ctx))); h = rb5.out.clone(ctx)
    var s5 = sdxl_st_lora_forward[1, H1, W1, C1, NKV, CCTX, H10, HD, CFF1, G, D2](
        h, context.clone(ctx), w.st[ST_IN5], lora[ST_IN5], ctx)
    st_acts.append(ArcPointer(s5.acts.copy())); h = s5.out.clone(ctx)
    skips.append(TArc(h.clone(ctx)))                                   # skip[5]
    # in6 Down(640) H1->H2
    down_x.append(TArc(h.clone(ctx)))
    h = downsample_forward[1, H1, W1, C1](h, w.down_w[1][], w.down_b[1][], ctx)
    skips.append(TArc(h.clone(ctx)))                                   # skip[6]

    # in7 Res(640->1280)+ST(1280,h20,d10)
    var rb7 = resblock_forward[1, H2, W2, C1, C2, TEMB, G](h, emb, w.res[R_IN7][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(rb7.acts, ctx))); h = rb7.out.clone(ctx)
    var s7 = sdxl_st_lora_forward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        h, context.clone(ctx), w.st[ST_IN7], lora[ST_IN7], ctx)
    st_acts.append(ArcPointer(s7.acts.copy())); h = s7.out.clone(ctx)
    skips.append(TArc(h.clone(ctx)))                                   # skip[7]
    # in8 Res(1280->1280)+ST
    var rb8 = resblock_forward[1, H2, W2, C2, C2, TEMB, G](h, emb, w.res[R_IN8][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(rb8.acts, ctx))); h = rb8.out.clone(ctx)
    var s8 = sdxl_st_lora_forward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        h, context.clone(ctx), w.st[ST_IN8], lora[ST_IN8], ctx)
    st_acts.append(ArcPointer(s8.acts.copy())); h = s8.out.clone(ctx)
    skips.append(TArc(h.clone(ctx)))                                   # skip[8]

    # middle: Res + ST + Res
    var rbm0 = resblock_forward[1, H2, W2, C2, C2, TEMB, G](h, emb, w.res[R_MID0][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(rbm0.acts, ctx))); h = rbm0.out.clone(ctx)
    var sm = sdxl_st_lora_forward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        h, context.clone(ctx), w.st[ST_MID], lora[ST_MID], ctx)
    st_acts.append(ArcPointer(sm.acts.copy())); h = sm.out.clone(ctx)
    var rbm2 = resblock_forward[1, H2, W2, C2, C2, TEMB, G](h, emb, w.res[R_MID2][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(rbm2.acts, ctx))); h = rbm2.out.clone(ctx)

    # decoder. cat order = [h, skip] (h first).
    # out0 cat(1280,skip[8]=1280)=2560 -> Res->1280 + ST10
    h = concat(3, ctx, h, skips[8][])
    var ro0 = resblock_forward[1, H2, W2, 2560, C2, TEMB, G](h, emb, w.res[R_OUT0][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(ro0.acts, ctx))); h = ro0.out.clone(ctx)
    var so0 = sdxl_st_lora_forward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        h, context.clone(ctx), w.st[ST_OUT0], lora[ST_OUT0], ctx)
    st_acts.append(ArcPointer(so0.acts.copy())); h = so0.out.clone(ctx)
    # out1 cat(1280,skip[7]=1280)=2560 -> Res->1280 + ST10
    h = concat(3, ctx, h, skips[7][])
    var ro1 = resblock_forward[1, H2, W2, 2560, C2, TEMB, G](h, emb, w.res[R_OUT1][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(ro1.acts, ctx))); h = ro1.out.clone(ctx)
    var so1 = sdxl_st_lora_forward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        h, context.clone(ctx), w.st[ST_OUT1], lora[ST_OUT1], ctx)
    st_acts.append(ArcPointer(so1.acts.copy())); h = so1.out.clone(ctx)
    # out2 cat(1280,skip[6]=640)=1920 -> Res->1280 + ST10 + Up H2->H1
    h = concat(3, ctx, h, skips[6][])
    var ro2 = resblock_forward[1, H2, W2, 1920, C2, TEMB, G](h, emb, w.res[R_OUT2][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(ro2.acts, ctx))); h = ro2.out.clone(ctx)
    var so2 = sdxl_st_lora_forward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        h, context.clone(ctx), w.st[ST_OUT2], lora[ST_OUT2], ctx)
    st_acts.append(ArcPointer(so2.acts.copy())); h = so2.out.clone(ctx)
    var u2 = upsample_forward[1, H2, W2, C2](h, w.up_w[0][], w.up_b[0][], ctx)
    up_up.append(TArc(u2.up.clone(ctx))); h = u2.out.clone(ctx)

    # out3 cat(1280,skip[5]=640)=1920 -> Res->640 + ST2 @H1
    h = concat(3, ctx, h, skips[5][])
    var ro3 = resblock_forward[1, H1, W1, 1920, C1, TEMB, G](h, emb, w.res[R_OUT3][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(ro3.acts, ctx))); h = ro3.out.clone(ctx)
    var so3 = sdxl_st_lora_forward[1, H1, W1, C1, NKV, CCTX, H10, HD, CFF1, G, D2](
        h, context.clone(ctx), w.st[ST_OUT3], lora[ST_OUT3], ctx)
    st_acts.append(ArcPointer(so3.acts.copy())); h = so3.out.clone(ctx)
    # out4 cat(640,skip[4]=640)=1280 -> Res->640 + ST2 @H1
    h = concat(3, ctx, h, skips[4][])
    var ro4 = resblock_forward[1, H1, W1, 1280, C1, TEMB, G](h, emb, w.res[R_OUT4][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(ro4.acts, ctx))); h = ro4.out.clone(ctx)
    var so4 = sdxl_st_lora_forward[1, H1, W1, C1, NKV, CCTX, H10, HD, CFF1, G, D2](
        h, context.clone(ctx), w.st[ST_OUT4], lora[ST_OUT4], ctx)
    st_acts.append(ArcPointer(so4.acts.copy())); h = so4.out.clone(ctx)
    # out5 cat(640,skip[3]=320)=960 -> Res->640 + ST2 + Up H1->H0
    h = concat(3, ctx, h, skips[3][])
    var ro5 = resblock_forward[1, H1, W1, 960, C1, TEMB, G](h, emb, w.res[R_OUT5][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(ro5.acts, ctx))); h = ro5.out.clone(ctx)
    var so5 = sdxl_st_lora_forward[1, H1, W1, C1, NKV, CCTX, H10, HD, CFF1, G, D2](
        h, context.clone(ctx), w.st[ST_OUT5], lora[ST_OUT5], ctx)
    st_acts.append(ArcPointer(so5.acts.copy())); h = so5.out.clone(ctx)
    var u5 = upsample_forward[1, H1, W1, C1](h, w.up_w[1][], w.up_b[1][], ctx)
    up_up.append(TArc(u5.up.clone(ctx))); h = u5.out.clone(ctx)

    # out6 cat(640,skip[2]=320)=960 -> Res->320 @H0
    h = concat(3, ctx, h, skips[2][])
    var ro6 = resblock_forward[1, H0, W0, 960, MC, TEMB, G](h, emb, w.res[R_OUT6][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(ro6.acts, ctx))); h = ro6.out.clone(ctx)
    # out7 cat(320,skip[1]=320)=640 -> Res->320 @H0
    h = concat(3, ctx, h, skips[1][])
    var ro7 = resblock_forward[1, H0, W0, 640, MC, TEMB, G](h, emb, w.res[R_OUT7][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(ro7.acts, ctx))); h = ro7.out.clone(ctx)
    # out8 cat(320,skip[0]=320)=640 -> Res->320 @H0
    h = concat(3, ctx, h, skips[0][])
    var ro8 = resblock_forward[1, H0, W0, 640, MC, TEMB, G](h, emb, w.res[R_OUT8][], ctx)
    res_acts.append(ArcPointer(_clone_rb_acts(ro8.acts, ctx))); h = ro8.out.clone(ctx)

    # final GN(eps1e-5) -> SiLU -> conv_out(320->4)
    var final_gn_in = h.clone(ctx)
    var gnf = group_norm(h, w.out_gn_w.clone(ctx), w.out_gn_b.clone(ctx), G, GN_EPS_RES, ctx)
    var final_silu_in = gnf.clone(ctx)
    var sf = silu(gnf, ctx)
    var conv_out_in = sf.clone(ctx)
    var out = _conv3x3_fwd[H0, W0, MC, 4](sf, w.conv_out_w, w.conv_out_b, ctx)

    var acts = SdxlRealActs(
        a_emb^, emb^, conv_in_x^, res_acts^, st_acts^, down_x^, up_up^,
        final_gn_in^, final_silu_in^, conv_out_in^,
    )
    return SdxlRealFwd(out^, acts^)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD (real dims). go = dL/dout NHWC [1,L,L,4]. Threads d_h in reverse; splits
# each decoder concat into (carry, skip) and ADDS the skip slab into the matching
# encoder block's output grad when the reverse walk reaches it.
# ═══════════════════════════════════════════════════════════════════════════════
def sdxl_real_backward[H: Int, W: Int](
    go_in: Tensor, acts: SdxlRealActs, w: SdxlRealWeights, lora: List[SdxlLoraSet],
    ctx: DeviceContext,
) raises -> SdxlRealGrads:
    comptime H0 = H
    comptime H1 = H // 2
    comptime H2 = H // 4
    comptime W0 = W
    comptime W1 = W // 2
    comptime W2 = W // 4

    # F32-grad / F32-activation / frozen-BF16-weight (mixed) backward — the forward
    # saved F32 acts; widen the incoming output grad to F32 so every
    # conv2d_backward/group_norm_backward/linear_backward hits its mixed path.
    var go = _f32(go_in, ctx)

    var d_a = List[List[List[Float32]]]()
    var d_b = List[List[List[Float32]]]()
    for _ in range(N_ST):
        d_a.append(List[List[Float32]]())
        d_b.append(List[List[Float32]]())
    var nonfinite = 0

    # emb grad accumulates across every ResBlock.
    var d_emb = _zeros_emb(ctx)

    # ── final: conv_out bwd -> SiLU bwd -> GN bwd ──
    var gco = _conv3x3_bwd[H0, W0, MC, 4](go, acts.conv_out_in, w.conv_out_w, ctx)
    var d_silu = silu_backward(gco.d_x, acts.final_silu_in, ctx)
    var ggn = group_norm_backward(d_silu, acts.final_gn_in, w.out_gn_w, G, GN_EPS_RES, ctx)
    var d_h = ggn.d_x.clone(ctx)         # grad into out8 output [1,H0,W0,320]

    # skip slabs (assigned as decoder concats unwind; consumed in encoder reverse)
    var d_skip = List[TArc]()
    for _ in range(9):
        d_skip.append(TArc(_zeros4(1, 1, 1, 1, ctx)))   # placeholders; overwritten below

    # ── out8 Res(640->320) bwd ; split cat(h[320],skip[0]=320) ──
    var go8 = resblock_backward[1, H0, W0, 640, MC, TEMB, G](d_h, acts.res_acts[R_OUT8][], w.res[R_OUT8][], ctx)
    d_emb = add(d_emb, go8.d_emb_in, ctx)
    var c8 = cat_backward(go8.d_x, MC, MC, 3, ctx)        # [320 | 320]
    d_h = c8.d_0.clone(ctx)
    d_skip[0] = TArc(c8.d_1.clone(ctx))
    # ── out7 Res(640->320) bwd ; split cat(h[320],skip[1]=320) ──
    var go7 = resblock_backward[1, H0, W0, 640, MC, TEMB, G](d_h, acts.res_acts[R_OUT7][], w.res[R_OUT7][], ctx)
    d_emb = add(d_emb, go7.d_emb_in, ctx)
    var c7 = cat_backward(go7.d_x, MC, MC, 3, ctx)
    d_h = c7.d_0.clone(ctx)
    d_skip[1] = TArc(c7.d_1.clone(ctx))
    # ── out6 Res(960->320) bwd ; split cat(h[640],skip[2]=320) ──
    var go6 = resblock_backward[1, H0, W0, 960, MC, TEMB, G](d_h, acts.res_acts[R_OUT6][], w.res[R_OUT6][], ctx)
    d_emb = add(d_emb, go6.d_emb_in, ctx)
    var c6 = cat_backward(go6.d_x, C1, MC, 3, ctx)        # [640 | 320]
    d_h = c6.d_0.clone(ctx)
    d_skip[2] = TArc(c6.d_1.clone(ctx))

    # ── out5 Up bwd H0->H1 -> ST2 bwd -> Res(960->640) bwd ; split cat(h[640],skip[3]=320) ──
    var gu5 = upsample_backward[1, H1, W1, C1](d_h, acts.up_up[1][], w.up_w[1][], ctx)
    var gso5 = sdxl_st_lora_backward[1, H1, W1, C1, NKV, CCTX, H10, HD, CFF1, G, D2](
        gu5.d_x, acts.st_acts[ST_OUT5][], w.st[ST_OUT5], lora[ST_OUT5], ctx)
    d_a[ST_OUT5] = gso5.d_a.copy(); d_b[ST_OUT5] = gso5.d_b.copy(); nonfinite += gso5.nonfinite_lora_grads
    var dso5x = _host_to_dev4(gso5.d_x, 1, H1, W1, C1, ctx)
    var go5 = resblock_backward[1, H1, W1, 960, C1, TEMB, G](dso5x, acts.res_acts[R_OUT5][], w.res[R_OUT5][], ctx)
    d_emb = add(d_emb, go5.d_emb_in, ctx)
    var c5 = cat_backward(go5.d_x, C1, MC, 3, ctx)        # [640 | 320]
    d_h = c5.d_0.clone(ctx)
    d_skip[3] = TArc(c5.d_1.clone(ctx))
    # ── out4 ST2 bwd -> Res(1280->640) bwd ; split cat(h[640],skip[4]=640) ──
    var gso4 = sdxl_st_lora_backward[1, H1, W1, C1, NKV, CCTX, H10, HD, CFF1, G, D2](
        d_h, acts.st_acts[ST_OUT4][], w.st[ST_OUT4], lora[ST_OUT4], ctx)
    d_a[ST_OUT4] = gso4.d_a.copy(); d_b[ST_OUT4] = gso4.d_b.copy(); nonfinite += gso4.nonfinite_lora_grads
    var dso4x = _host_to_dev4(gso4.d_x, 1, H1, W1, C1, ctx)
    var go4 = resblock_backward[1, H1, W1, 1280, C1, TEMB, G](dso4x, acts.res_acts[R_OUT4][], w.res[R_OUT4][], ctx)
    d_emb = add(d_emb, go4.d_emb_in, ctx)
    var c4 = cat_backward(go4.d_x, C1, C1, 3, ctx)        # [640 | 640]
    d_h = c4.d_0.clone(ctx)
    d_skip[4] = TArc(c4.d_1.clone(ctx))
    # ── out3 ST2 bwd -> Res(1920->640) bwd ; split cat(h[1280],skip[5]=640) ──
    var gso3 = sdxl_st_lora_backward[1, H1, W1, C1, NKV, CCTX, H10, HD, CFF1, G, D2](
        d_h, acts.st_acts[ST_OUT3][], w.st[ST_OUT3], lora[ST_OUT3], ctx)
    d_a[ST_OUT3] = gso3.d_a.copy(); d_b[ST_OUT3] = gso3.d_b.copy(); nonfinite += gso3.nonfinite_lora_grads
    var dso3x = _host_to_dev4(gso3.d_x, 1, H1, W1, C1, ctx)
    var go3 = resblock_backward[1, H1, W1, 1920, C1, TEMB, G](dso3x, acts.res_acts[R_OUT3][], w.res[R_OUT3][], ctx)
    d_emb = add(d_emb, go3.d_emb_in, ctx)
    var c3 = cat_backward(go3.d_x, C2, C1, 3, ctx)        # [1280 | 640]
    d_h = c3.d_0.clone(ctx)
    d_skip[5] = TArc(c3.d_1.clone(ctx))

    # ── out2 Up bwd H1->H2 -> ST10 bwd -> Res(1920->1280) bwd ; split cat(h[1280],skip[6]=640) ──
    var gu2 = upsample_backward[1, H2, W2, C2](d_h, acts.up_up[0][], w.up_w[0][], ctx)
    var gso2 = sdxl_st_lora_backward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        gu2.d_x, acts.st_acts[ST_OUT2][], w.st[ST_OUT2], lora[ST_OUT2], ctx)
    d_a[ST_OUT2] = gso2.d_a.copy(); d_b[ST_OUT2] = gso2.d_b.copy(); nonfinite += gso2.nonfinite_lora_grads
    var dso2x = _host_to_dev4(gso2.d_x, 1, H2, W2, C2, ctx)
    var go2 = resblock_backward[1, H2, W2, 1920, C2, TEMB, G](dso2x, acts.res_acts[R_OUT2][], w.res[R_OUT2][], ctx)
    d_emb = add(d_emb, go2.d_emb_in, ctx)
    var c2 = cat_backward(go2.d_x, C2, C1, 3, ctx)        # [1280 | 640]
    d_h = c2.d_0.clone(ctx)
    d_skip[6] = TArc(c2.d_1.clone(ctx))
    # ── out1 ST10 bwd -> Res(2560->1280) bwd ; split cat(h[1280],skip[7]=1280) ──
    var gso1 = sdxl_st_lora_backward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        d_h, acts.st_acts[ST_OUT1][], w.st[ST_OUT1], lora[ST_OUT1], ctx)
    d_a[ST_OUT1] = gso1.d_a.copy(); d_b[ST_OUT1] = gso1.d_b.copy(); nonfinite += gso1.nonfinite_lora_grads
    var dso1x = _host_to_dev4(gso1.d_x, 1, H2, W2, C2, ctx)
    var go1 = resblock_backward[1, H2, W2, 2560, C2, TEMB, G](dso1x, acts.res_acts[R_OUT1][], w.res[R_OUT1][], ctx)
    d_emb = add(d_emb, go1.d_emb_in, ctx)
    var c1 = cat_backward(go1.d_x, C2, C2, 3, ctx)        # [1280 | 1280]
    d_h = c1.d_0.clone(ctx)
    d_skip[7] = TArc(c1.d_1.clone(ctx))
    # ── out0 ST10 bwd -> Res(2560->1280) bwd ; split cat(h[1280],skip[8]=1280) ──
    var gso0 = sdxl_st_lora_backward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        d_h, acts.st_acts[ST_OUT0][], w.st[ST_OUT0], lora[ST_OUT0], ctx)
    d_a[ST_OUT0] = gso0.d_a.copy(); d_b[ST_OUT0] = gso0.d_b.copy(); nonfinite += gso0.nonfinite_lora_grads
    var dso0x = _host_to_dev4(gso0.d_x, 1, H2, W2, C2, ctx)
    var go0 = resblock_backward[1, H2, W2, 2560, C2, TEMB, G](dso0x, acts.res_acts[R_OUT0][], w.res[R_OUT0][], ctx)
    d_emb = add(d_emb, go0.d_emb_in, ctx)
    var c0 = cat_backward(go0.d_x, C2, C2, 3, ctx)        # [1280 | 1280]
    var d_mid_out = c0.d_0.clone(ctx)
    d_skip[8] = TArc(c0.d_1.clone(ctx))

    # ── middle bwd: Res2 -> ST10 -> Res0 ──
    var gm2 = resblock_backward[1, H2, W2, C2, C2, TEMB, G](d_mid_out, acts.res_acts[R_MID2][], w.res[R_MID2][], ctx)
    d_emb = add(d_emb, gm2.d_emb_in, ctx)
    var gsm = sdxl_st_lora_backward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        gm2.d_x, acts.st_acts[ST_MID][], w.st[ST_MID], lora[ST_MID], ctx)
    d_a[ST_MID] = gsm.d_a.copy(); d_b[ST_MID] = gsm.d_b.copy(); nonfinite += gsm.nonfinite_lora_grads
    var gsmx = _host_to_dev4(gsm.d_x, 1, H2, W2, C2, ctx)
    var gm0 = resblock_backward[1, H2, W2, C2, C2, TEMB, G](gsmx, acts.res_acts[R_MID0][], w.res[R_MID0][], ctx)
    d_emb = add(d_emb, gm0.d_emb_in, ctx)
    var d_in8_out = gm0.d_x.clone(ctx)

    # ── in8 output grad = (from middle) + skip slab[8] ──
    d_in8_out = add(d_in8_out, d_skip[8][], ctx)
    var gs8 = sdxl_st_lora_backward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        d_in8_out, acts.st_acts[ST_IN8][], w.st[ST_IN8], lora[ST_IN8], ctx)
    d_a[ST_IN8] = gs8.d_a.copy(); d_b[ST_IN8] = gs8.d_b.copy(); nonfinite += gs8.nonfinite_lora_grads
    var gs8x = _host_to_dev4(gs8.d_x, 1, H2, W2, C2, ctx)
    var gin8 = resblock_backward[1, H2, W2, C2, C2, TEMB, G](gs8x, acts.res_acts[R_IN8][], w.res[R_IN8][], ctx)
    d_emb = add(d_emb, gin8.d_emb_in, ctx)
    var d_in7_out = gin8.d_x.clone(ctx)
    # ── in7 output grad = (from in8) + skip slab[7] ──
    d_in7_out = add(d_in7_out, d_skip[7][], ctx)
    var gs7 = sdxl_st_lora_backward[1, H2, W2, C2, NKV, CCTX, H20, HD, CFF2, G, D10](
        d_in7_out, acts.st_acts[ST_IN7][], w.st[ST_IN7], lora[ST_IN7], ctx)
    d_a[ST_IN7] = gs7.d_a.copy(); d_b[ST_IN7] = gs7.d_b.copy(); nonfinite += gs7.nonfinite_lora_grads
    var gs7x = _host_to_dev4(gs7.d_x, 1, H2, W2, C2, ctx)
    var gin7 = resblock_backward[1, H2, W2, C1, C2, TEMB, G](gs7x, acts.res_acts[R_IN7][], w.res[R_IN7][], ctx)
    d_emb = add(d_emb, gin7.d_emb_in, ctx)
    var d_in6_out = gin7.d_x.clone(ctx)          # grad into in6 (down) output [1,H2,W2,640]

    # ── in6 output grad = (from in7) + skip slab[6] ; Down bwd H1->H2 ──
    d_in6_out = add(d_in6_out, d_skip[6][], ctx)
    var gin6 = downsample_backward[1, H1, W1, C1](d_in6_out, acts.down_x[1][], w.down_w[1][], ctx)
    var d_in5_out = gin6.d_x.clone(ctx)          # grad into in5 output [1,H1,W1,640]

    # ── in5 output grad = (from in6) + skip slab[5] ; ST2 -> Res(640->640) ──
    d_in5_out = add(d_in5_out, d_skip[5][], ctx)
    var gs5 = sdxl_st_lora_backward[1, H1, W1, C1, NKV, CCTX, H10, HD, CFF1, G, D2](
        d_in5_out, acts.st_acts[ST_IN5][], w.st[ST_IN5], lora[ST_IN5], ctx)
    d_a[ST_IN5] = gs5.d_a.copy(); d_b[ST_IN5] = gs5.d_b.copy(); nonfinite += gs5.nonfinite_lora_grads
    var gs5x = _host_to_dev4(gs5.d_x, 1, H1, W1, C1, ctx)
    var gin5 = resblock_backward[1, H1, W1, C1, C1, TEMB, G](gs5x, acts.res_acts[R_IN5][], w.res[R_IN5][], ctx)
    d_emb = add(d_emb, gin5.d_emb_in, ctx)
    var d_in4_out = gin5.d_x.clone(ctx)
    # ── in4 output grad = (from in5) + skip slab[4] ; ST2 -> Res(320->640) ──
    d_in4_out = add(d_in4_out, d_skip[4][], ctx)
    var gs4 = sdxl_st_lora_backward[1, H1, W1, C1, NKV, CCTX, H10, HD, CFF1, G, D2](
        d_in4_out, acts.st_acts[ST_IN4][], w.st[ST_IN4], lora[ST_IN4], ctx)
    d_a[ST_IN4] = gs4.d_a.copy(); d_b[ST_IN4] = gs4.d_b.copy(); nonfinite += gs4.nonfinite_lora_grads
    var gs4x = _host_to_dev4(gs4.d_x, 1, H1, W1, C1, ctx)
    var gin4 = resblock_backward[1, H1, W1, MC, C1, TEMB, G](gs4x, acts.res_acts[R_IN4][], w.res[R_IN4][], ctx)
    d_emb = add(d_emb, gin4.d_emb_in, ctx)
    var d_in3_out = gin4.d_x.clone(ctx)          # grad into in3 (down) output [1,H1,W1,320]

    # ── in3 output grad = (from in4) + skip slab[3] ; Down bwd H0->H1 ──
    d_in3_out = add(d_in3_out, d_skip[3][], ctx)
    var gin3 = downsample_backward[1, H0, W0, MC](d_in3_out, acts.down_x[0][], w.down_w[0][], ctx)
    var d_in2_out = gin3.d_x.clone(ctx)

    # ── in2 output grad = (from in3) + skip slab[2] ; Res(320->320) ──
    d_in2_out = add(d_in2_out, d_skip[2][], ctx)
    var gin2 = resblock_backward[1, H0, W0, MC, MC, TEMB, G](d_in2_out, acts.res_acts[R_IN2][], w.res[R_IN2][], ctx)
    d_emb = add(d_emb, gin2.d_emb_in, ctx)
    var d_in1_out = gin2.d_x.clone(ctx)
    # ── in1 output grad = (from in2) + skip slab[1] ; Res(320->320) ──
    d_in1_out = add(d_in1_out, d_skip[1][], ctx)
    var gin1 = resblock_backward[1, H0, W0, MC, MC, TEMB, G](d_in1_out, acts.res_acts[R_IN1][], w.res[R_IN1][], ctx)
    d_emb = add(d_emb, gin1.d_emb_in, ctx)
    var d_in0_out = gin1.d_x.clone(ctx)
    # ── conv_in output grad = (from in1) + skip slab[0] ; conv_in bwd ──
    d_in0_out = add(d_in0_out, d_skip[0][], ctx)
    var gci = _conv3x3_bwd[H0, W0, 4, MC](d_in0_out, acts.conv_in_x, w.conv_in_w, ctx)
    var d_x = gci.d_x.clone(ctx)                 # grad wrt latent input [1,L,L,4]

    # (embed backward computed but discarded for LoRA training — base weights frozen.)
    var _geb = embed_backward[1, SDIM, TEMB, ADM](d_emb, acts.emb_acts, w.emb, ctx)

    return SdxlRealGrads(d_a^, d_b^, d_x^, nonfinite)


def _zeros_emb(ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for _ in range(TEMB):
        h.append(0.0)
    var s = List[Int](); s.append(1); s.append(TEMB)
    return Tensor.from_host(h, s^, STDtype.F32, ctx)


def _zeros4(a: Int, b: Int, c: Int, d: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for _ in range(a * b * c * d):
        h.append(0.0)
    return Tensor.from_host(h, _sh4(a, b, c, d), STDtype.F32, ctx)


# host-list [B*H*W*C] -> device NHWC [B,H,W,C]
def _host_to_dev4(v: List[Float32], B: Int, H: Int, W: Int, C: Int, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(v.copy(), _sh4(B, H, W, C), STDtype.F32, ctx)
