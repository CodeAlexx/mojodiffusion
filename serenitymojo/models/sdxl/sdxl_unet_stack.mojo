# models/sdxl/sdxl_unet_stack.mojo — SDXL full conv-UNet fwd+bwd, COMPOSING the
# parity-gated block units (ResBlock, SpatialTransformer, Down/Up sample, embed,
# conv_in/conv_out, final GN) into the complete encoder→skip→decoder topology.
#
# This file COMPOSES; it rebuilds NOTHING (Tenet 1). Every backward arm it calls
# already has its own ops/parity gate:
#   * models/sdxl/block.mojo          resblock_forward / resblock_backward      (15/15)
#   * models/sdxl/spatial_transformer.mojo  spatial_transformer_{forward,backward} (29/29, 48/48)
#   * models/sdxl/sampling.mojo       downsample_/upsample_{forward,backward}
#   * models/sdxl/embed.mojo          embed_{forward,backward}                   (time+label)
#   * ops/conv.mojo + conv2d_backward.mojo   conv_in / conv_out (3x3 pad1)
#   * ops/norm.mojo + norm_backward.mojo     final GroupNorm (eps=1e-5)
#   * ops/activations.mojo + activation_backward.mojo  final SiLU
#   * ops/tensor_algebra.mojo concat + ops/shape_backward.mojo cat_backward  (skip-concat)
#
# ── FORWARD GRAPH (verified line-by-line vs inference-flame sdxl_unet.rs::forward
#    851-995 + build_block_descriptors 318-373) ─────────────────────────────────
#   emb = time_embed(t) + label_embed(y)                          [B,Temb]  (shared)
#   h = conv_in(x)                                                [B,H,W,mc]
#   # ENCODER: each block runs then PUSHES its output onto the skip stack `hs`.
#   for input_block:
#       ConvIn:   h = conv_in(x);                       hs.push(h)
#       Res(+ST): h = resblock(h,emb); if td>0: h=ST(h,ctx);  hs.push(h)
#       Down:     h = downsample(h);                    hs.push(h)
#   # MIDDLE: Res -> ST -> Res
#   h = resblock(h,emb); h = ST(h,ctx); h = resblock(h,emb)
#   # DECODER: each block POPS the skip stack (LIFO), concats on the channel axis,
#   # runs resblock(+ST)(+Up). cat order = [h, skip] (h FIRST) — matches rs:936
#   # `Tensor::cat(&[&h, &skip], 1)` (NCHW dim1 == NHWC channel axis).
#   for output_block:
#       skip = hs.pop(); h = concat(channel, h, skip)
#       h = resblock(h,emb); if td>0: h=ST(h,ctx); if up: h=upsample(h)
#   # FINAL: GroupNorm(32,eps=1e-5) -> SiLU -> conv_out(3x3 pad1)
#   h = group_norm(h); h = silu(h); out = conv_out(h)             [B,H,W,out_ch]
#
# ── SKIP-CONNECTION BOOKKEEPING (the #1 correctness risk) ─────────────────────
# The encoder pushes N_in activations in forward order [0..N_in-1]; the decoder
# pops them LIFO so output_block k consumes the skip from input_block
# (N_in-1-k). The concat is ALWAYS [current_h | skip] on the channel axis, so the
# decoder ResBlock's Cin = (decoder-carry channels) + (that skip's channels).
# In backward, the concat's grad splits with size0 = current_h channels (FIRST)
# and size1 = skip channels (SECOND); the size1 slab is the grad routed back to
# the matching encoder block's output, ADDED to whatever grad already flows into
# that encoder block from the (deeper) decoder path. For SDXL the encoder's own
# forward chain feeds each block's output ONLY into (a) the next encoder block
# and (b) exactly one decoder concat — so the encoder block's total output grad
# is (grad from the next encoder block, threaded in the encoder-reverse walk) +
# (the skip slab from its matching decoder concat). This file threads both.
#
# Because conv2d/group_norm take COMPTIME shape params, the stack is written as an
# explicit per-block sequence specialized to one config. Two instantiations are
# provided: `reduced` (structurally-complete tiny UNet for the parity + finite-diff
# gate) and `real` (320ch full-depth, small spatial dims, finite smoke). They share
# the SAME block helpers — only the comptime dims differ.
#
# All NHWC, F32 interior (matches the gated units + the Rust FP32 residual stream).

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.conv2d_backward import conv2d_backward
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.norm_backward import group_norm_backward
from serenitymojo.ops.activations import silu
from serenitymojo.ops.activation_backward import silu_backward
from serenitymojo.ops.tensor_algebra import add, concat
from serenitymojo.ops.shape_backward import cat_backward, CatGrads2

from serenitymojo.models.sdxl.block import (
    resblock_forward, resblock_backward, ResBlockFwd, ResBlockActs, ResBlockGrads,
)
from serenitymojo.models.sdxl.spatial_transformer import (
    spatial_transformer_forward, spatial_transformer_backward,
    SpatialTransformerWeights, SpatialTransformerActs, SpatialTransformerGrads,
)
from serenitymojo.models.sdxl.sampling import (
    downsample_forward, downsample_backward,
    upsample_forward, upsample_backward, UpsampleFwd, SampleGrads,
)
from serenitymojo.models.sdxl.weights import ResBlockWeights
from serenitymojo.models.sdxl.config import GN_EPS_RES
from serenitymojo.models.sdxl.embed import (
    embed_forward, embed_backward, EmbWeights, EmbActs, EmbFwd, EmbGrads,
)

comptime TArc = ArcPointer[Tensor]


# ── tiny shape helpers ─────────────────────────────────────────────────────────
def _sh1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^
def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


# ═══════════════════════════════════════════════════════════════════════════════
# conv_in / conv_out helpers (3x3 pad-1 stride-1; RSCF filter [3,3,Cin,Cout]).
# ═══════════════════════════════════════════════════════════════════════════════
def _conv3x3_fwd[
    N: Int, H: Int, W: Int, Cin: Int, Cout: Int,
](x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return conv2d[N, H, W, Cin, 3, 3, Cout, 1, 1, 1, 1](
        x, w.clone(ctx), Optional[Tensor](b.clone(ctx)), ctx
    )


struct ConvGrads(Movable):
    var d_x: Tensor
    var d_w: Tensor
    var d_b: Tensor
    def __init__(out self, var d_x: Tensor, var d_w: Tensor, var d_b: Tensor):
        self.d_x = d_x^; self.d_w = d_w^; self.d_b = d_b^


def _conv3x3_bwd[
    N: Int, H: Int, W: Int, Cin: Int, Cout: Int,
](go: Tensor, x: Tensor, w: Tensor, ctx: DeviceContext) raises -> ConvGrads:
    var g = conv2d_backward[N, H, W, Cin, 3, 3, Cout, 1, 1, 1, 1](x, w, go, ctx)
    return ConvGrads(g.d_x.clone(ctx), g.d_w.clone(ctx), g.d_b.clone(ctx))


# ── final GroupNorm -> SiLU -> conv_out backward bundle is inline in the stack. ─


# Clone a ResBlockActs by borrowing each field — leaves the source Fwd struct
# fully intact (so it auto-drops normally; avoids Mojo's destroy-out-of-middle
# partial-move error). Mirrors the klein `.copy()` idiom for move-only acts.
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


# ═══════════════════════════════════════════════════════════════════════════════
# REDUCED-CONFIG STACK (structurally complete: conv_in, 3 levels w/ downsample,
# skip stack, middle Res+ST+Res, output concat+Res(+ST)(+Up), final GN+SiLU+conv).
# Comptime dims locked to unet_stack_oracle.py:
#   B=1 H0=8 W0=8 IN_CH=4 MC=16 OUT_CH=4 TEMB=32 SDIM=16 ADM=24 NKV=7 CCTX=16
#   HEAD_DIM=8 G=16. Levels: mc=16/32/64. Spatial: 8 ->down 4 ->down 2 (middle).
# ═══════════════════════════════════════════════════════════════════════════════
comptime RB = 1
comptime RH0 = 8
comptime RW0 = 8
comptime RIN = 4
comptime RMC = 16
comptime ROUT = 4
comptime RTEMB = 32
comptime RSDIM = 16
comptime RADM = 24
comptime RNKV = 7
comptime RCCTX = 16
comptime RHD = 8        # head_dim
comptime RG = 16        # groupnorm groups
# per-level channels
comptime C1 = 16
comptime C2 = 32
comptime C3 = 64
# spatial resolutions
comptime S0 = 8         # level 0
comptime S1 = 4         # level 1 (after in2 down)
comptime S2 = 2         # level 2 (after in4 down) == middle
# attention head counts (C/HEAD_DIM)
comptime HH2 = C2 // RHD    # 4
comptime HH3 = C3 // RHD    # 8
# GEGLU inner-half dim (the `Cff` param geglu_forward[M,C,Cff] expects). The
# oracle's ff.net.0.proj projects C -> 2*C and GEGLU splits into two C-wide halves,
# so the inner half = C. (ff_proj_w is [2*C, C], ff_out_w is [C, C].)
comptime CFF2 = C2          # ST at C=32 -> inner half 32
comptime CFF3 = C3          # ST at C=64 -> inner half 64
comptime CFF_O0 = C3
comptime CFF_O1 = C3


# Per-block ST weight prefixes used at: in3 (C32 d1), in5 (C64 d1), mid (C64 d1),
# out0 (C64 d1), out1 (C64 d1). All depth 1. Self-attn head count = C/HEAD_DIM.

# ── weights bundle (named per-block; ResBlockWeights move-only, ST Copyable) ───
struct SdxlStackWeightsReduced(Movable):
    var emb: EmbWeights
    var conv_in_w: Tensor; var conv_in_b: Tensor
    var out_gn_w: Tensor; var out_gn_b: Tensor
    var conv_out_w: Tensor; var conv_out_b: Tensor
    # encoder res
    var in1: ResBlockWeights
    var in3: ResBlockWeights; var in3_st: SpatialTransformerWeights
    var in5: ResBlockWeights; var in5_st: SpatialTransformerWeights
    # downsamples
    var in2_op_w: Tensor; var in2_op_b: Tensor
    var in4_op_w: Tensor; var in4_op_b: Tensor
    # middle
    var mid0: ResBlockWeights; var mid_st: SpatialTransformerWeights; var mid2: ResBlockWeights
    # decoder res
    var out0: ResBlockWeights; var out0_st: SpatialTransformerWeights
    var out1: ResBlockWeights; var out1_st: SpatialTransformerWeights
    var out1_up_w: Tensor; var out1_up_b: Tensor
    var out2: ResBlockWeights
    var out3: ResBlockWeights
    var out3_up_w: Tensor; var out3_up_b: Tensor
    var out4: ResBlockWeights
    var out5: ResBlockWeights

    def __init__(
        out self, var emb: EmbWeights,
        var conv_in_w: Tensor, var conv_in_b: Tensor,
        var out_gn_w: Tensor, var out_gn_b: Tensor,
        var conv_out_w: Tensor, var conv_out_b: Tensor,
        var in1: ResBlockWeights,
        var in3: ResBlockWeights, var in3_st: SpatialTransformerWeights,
        var in5: ResBlockWeights, var in5_st: SpatialTransformerWeights,
        var in2_op_w: Tensor, var in2_op_b: Tensor,
        var in4_op_w: Tensor, var in4_op_b: Tensor,
        var mid0: ResBlockWeights, var mid_st: SpatialTransformerWeights, var mid2: ResBlockWeights,
        var out0: ResBlockWeights, var out0_st: SpatialTransformerWeights,
        var out1: ResBlockWeights, var out1_st: SpatialTransformerWeights,
        var out1_up_w: Tensor, var out1_up_b: Tensor,
        var out2: ResBlockWeights,
        var out3: ResBlockWeights,
        var out3_up_w: Tensor, var out3_up_b: Tensor,
        var out4: ResBlockWeights,
        var out5: ResBlockWeights,
    ):
        self.emb = emb^
        self.conv_in_w = conv_in_w^; self.conv_in_b = conv_in_b^
        self.out_gn_w = out_gn_w^; self.out_gn_b = out_gn_b^
        self.conv_out_w = conv_out_w^; self.conv_out_b = conv_out_b^
        self.in1 = in1^
        self.in3 = in3^; self.in3_st = in3_st^
        self.in5 = in5^; self.in5_st = in5_st^
        self.in2_op_w = in2_op_w^; self.in2_op_b = in2_op_b^
        self.in4_op_w = in4_op_w^; self.in4_op_b = in4_op_b^
        self.mid0 = mid0^; self.mid_st = mid_st^; self.mid2 = mid2^
        self.out0 = out0^; self.out0_st = out0_st^
        self.out1 = out1^; self.out1_st = out1_st^
        self.out1_up_w = out1_up_w^; self.out1_up_b = out1_up_b^
        self.out2 = out2^
        self.out3 = out3^
        self.out3_up_w = out3_up_w^; self.out3_up_b = out3_up_b^
        self.out4 = out4^
        self.out5 = out5^


# ── forward saved acts (everything backward needs) ───────────────────────────
# We retain per-block acts (small at reduced dims). ResBlockActs/STActs are the
# per-unit saved-activation bundles. conv_in/conv_out/final-GN need their inputs.
struct SdxlStackActsReduced(Movable):
    var emb_acts: EmbActs
    var emb: Tensor             # [B,TEMB] shared (input to every resblock emb)
    var conv_in_x: Tensor       # conv_in input  [B,8,8,4]
    var in1_a: ResBlockActs
    var in2_x: Tensor           # downsample input [B,8,8,16]
    var in3_a: ResBlockActs; var in3_st_a: SpatialTransformerActs
    var in4_x: Tensor           # downsample input [B,4,4,32]
    var in5_a: ResBlockActs; var in5_st_a: SpatialTransformerActs
    var mid0_a: ResBlockActs; var mid_st_a: SpatialTransformerActs; var mid2_a: ResBlockActs
    var out0_a: ResBlockActs; var out0_st_a: SpatialTransformerActs
    var out1_a: ResBlockActs; var out1_st_a: SpatialTransformerActs
    var out1_up_up: Tensor      # saved upsampled conv-input [B,4,4,64]
    var out2_a: ResBlockActs
    var out3_a: ResBlockActs
    var out3_up_up: Tensor      # saved upsampled conv-input [B,4,4,32]
    var out4_a: ResBlockActs
    var out5_a: ResBlockActs
    var final_gn_in: Tensor     # final GN input [B,8,8,16]
    var final_silu_in: Tensor   # final SiLU input (== GN out) [B,8,8,16]
    var conv_out_in: Tensor     # conv_out input (== SiLU out) [B,8,8,16]

    def __init__(
        out self, var emb_acts: EmbActs, var emb: Tensor,
        var conv_in_x: Tensor,
        var in1_a: ResBlockActs, var in2_x: Tensor,
        var in3_a: ResBlockActs, var in3_st_a: SpatialTransformerActs,
        var in4_x: Tensor,
        var in5_a: ResBlockActs, var in5_st_a: SpatialTransformerActs,
        var mid0_a: ResBlockActs, var mid_st_a: SpatialTransformerActs, var mid2_a: ResBlockActs,
        var out0_a: ResBlockActs, var out0_st_a: SpatialTransformerActs,
        var out1_a: ResBlockActs, var out1_st_a: SpatialTransformerActs, var out1_up_up: Tensor,
        var out2_a: ResBlockActs,
        var out3_a: ResBlockActs, var out3_up_up: Tensor,
        var out4_a: ResBlockActs,
        var out5_a: ResBlockActs,
        var final_gn_in: Tensor, var final_silu_in: Tensor, var conv_out_in: Tensor,
    ):
        self.emb_acts = emb_acts^; self.emb = emb^
        self.conv_in_x = conv_in_x^
        self.in1_a = in1_a^; self.in2_x = in2_x^
        self.in3_a = in3_a^; self.in3_st_a = in3_st_a^
        self.in4_x = in4_x^
        self.in5_a = in5_a^; self.in5_st_a = in5_st_a^
        self.mid0_a = mid0_a^; self.mid_st_a = mid_st_a^; self.mid2_a = mid2_a^
        self.out0_a = out0_a^; self.out0_st_a = out0_st_a^
        self.out1_a = out1_a^; self.out1_st_a = out1_st_a^; self.out1_up_up = out1_up_up^
        self.out2_a = out2_a^
        self.out3_a = out3_a^; self.out3_up_up = out3_up_up^
        self.out4_a = out4_a^
        self.out5_a = out5_a^
        self.final_gn_in = final_gn_in^; self.final_silu_in = final_silu_in^
        self.conv_out_in = conv_out_in^


struct SdxlStackFwdReduced(Movable):
    var out: Tensor
    var acts: SdxlStackActsReduced
    def __init__(out self, var out: Tensor, var acts: SdxlStackActsReduced):
        self.out = out^; self.acts = acts^


# ── backward grads: the load-bearing input grads + representative weight grads ─
# (full per-weight grads exist inside the per-block grad structs; here we surface
# the ones the gate checks: d_x, d_context, d_y, plus representative weight grads.)
struct SdxlStackGradsReduced(Movable):
    var d_x: Tensor             # [B,8,8,4]   grad wrt latent input
    var d_context: Tensor       # [B,NKV,CCTX] summed over every ST
    var d_y: Tensor             # [B,ADM]     grad wrt ADM vector
    var d_conv_in_w: Tensor
    var d_conv_out_w: Tensor
    var d_mid0_conv1_w: Tensor
    var d_in5_st_q2: Tensor     # in5 ST block0 attn2 to_q grad
    var d_out0_conv2_w: Tensor
    var d_t0_w: Tensor
    var d_l0_w: Tensor

    def __init__(
        out self, var d_x: Tensor, var d_context: Tensor, var d_y: Tensor,
        var d_conv_in_w: Tensor, var d_conv_out_w: Tensor,
        var d_mid0_conv1_w: Tensor, var d_in5_st_q2: Tensor,
        var d_out0_conv2_w: Tensor, var d_t0_w: Tensor, var d_l0_w: Tensor,
    ):
        self.d_x = d_x^; self.d_context = d_context^; self.d_y = d_y^
        self.d_conv_in_w = d_conv_in_w^; self.d_conv_out_w = d_conv_out_w^
        self.d_mid0_conv1_w = d_mid0_conv1_w^; self.d_in5_st_q2 = d_in5_st_q2^
        self.d_out0_conv2_w = d_out0_conv2_w^; self.d_t0_w = d_t0_w^; self.d_l0_w = d_l0_w^


# ── FULL FORWARD (reduced config) ────────────────────────────────────────────
# x: [B,8,8,4] NHWC latent.  t: [B] timesteps.  y: [B,ADM] ADM vector.
# context: [B,NKV,CCTX].  Returns out [B,8,8,4] + saved acts.
def sdxl_unet_stack_forward_reduced(
    x: Tensor, t: Tensor, y: Tensor, context: Tensor,
    w: SdxlStackWeightsReduced, ctx: DeviceContext,
) raises -> SdxlStackFwdReduced:
    # ── embeddings (shared across every resblock) ──
    var ef = embed_forward[RB, RSDIM, RTEMB, RADM](t, y, w.emb, ctx)
    var a_emb = _clone_emb_acts(ef.acts, ctx)
    var emb = ef.emb.clone(ctx)

    # ── conv_in (in0) -> push ──
    var conv_in_x = x.clone(ctx)
    var h = _conv3x3_fwd[RB, S0, S0, RIN, RMC](x, w.conv_in_w, w.conv_in_b, ctx)   # [B,8,8,16]
    var skip_in0 = h.clone(ctx)

    # ── in1 Res(16->16) -> push ──
    # NOTE: extract `.acts`/`.up` into standalone vars IMMEDIATELY after each
    # forward (out already cloned into h) so the Fwd struct is fully consumed at
    # its call site — Mojo's destroy-whole-value rule forbids moving one field of
    # a live struct out across a long scope (the klein_stack idiom).
    var rb1 = resblock_forward[RB, S0, S0, C1, C1, RTEMB, RG](h, emb, w.in1, ctx)
    var a_in1 = _clone_rb_acts(rb1.acts, ctx)
    h = rb1.out.clone(ctx)
    var skip_in1 = h.clone(ctx)

    # ── in2 Down(16) [8->4] -> push ──
    var in2_x = h.clone(ctx)
    h = downsample_forward[RB, S0, S0, C1](h, w.in2_op_w, w.in2_op_b, ctx)         # [B,4,4,16]
    var skip_in2 = h.clone(ctx)

    # ── in3 Res(16->32) + ST -> push ──
    var rb3 = resblock_forward[RB, S1, S1, C1, C2, RTEMB, RG](h, emb, w.in3, ctx)  # [B,4,4,32]
    var a_in3 = _clone_rb_acts(rb3.acts, ctx)
    h = rb3.out.clone(ctx)
    var st3 = spatial_transformer_forward[RB, S1, S1, C2, RNKV, RCCTX, HH2, RHD, CFF2, RG, 1](
        h, context.clone(ctx), w.in3_st, ctx)
    var a_in3_st = st3.acts.copy()
    h = st3.out.clone(ctx)
    var skip_in3 = h.clone(ctx)

    # ── in4 Down(32) [4->2] -> push ──
    var in4_x = h.clone(ctx)
    h = downsample_forward[RB, S1, S1, C2](h, w.in4_op_w, w.in4_op_b, ctx)         # [B,2,2,32]
    var skip_in4 = h.clone(ctx)

    # ── in5 Res(32->64) + ST -> push ──
    var rb5 = resblock_forward[RB, S2, S2, C2, C3, RTEMB, RG](h, emb, w.in5, ctx)  # [B,2,2,64]
    var a_in5 = _clone_rb_acts(rb5.acts, ctx)
    h = rb5.out.clone(ctx)
    var st5 = spatial_transformer_forward[RB, S2, S2, C3, RNKV, RCCTX, HH3, RHD, CFF3, RG, 1](
        h, context.clone(ctx), w.in5_st, ctx)
    var a_in5_st = st5.acts.copy()
    h = st5.out.clone(ctx)
    var skip_in5 = h.clone(ctx)

    # ── middle: Res(64->64) + ST + Res(64->64) ──
    var rbm0 = resblock_forward[RB, S2, S2, C3, C3, RTEMB, RG](h, emb, w.mid0, ctx)
    var a_mid0 = _clone_rb_acts(rbm0.acts, ctx)
    h = rbm0.out.clone(ctx)
    var stm = spatial_transformer_forward[RB, S2, S2, C3, RNKV, RCCTX, HH3, RHD, CFF3, RG, 1](
        h, context.clone(ctx), w.mid_st, ctx)
    var a_mid_st = stm.acts.copy()
    h = stm.out.clone(ctx)
    var rbm2 = resblock_forward[RB, S2, S2, C3, C3, RTEMB, RG](h, emb, w.mid2, ctx)
    var a_mid2 = _clone_rb_acts(rbm2.acts, ctx)
    h = rbm2.out.clone(ctx)

    # ── decoder ──
    # out0: cat(h[64], skip_in5[64]) = 128 @ 2x2 -> Res(128->64) + ST
    h = concat(3, ctx, h, skip_in5)                                                # [B,2,2,128]
    var rbo0 = resblock_forward[RB, S2, S2, 128, C3, RTEMB, RG](h, emb, w.out0, ctx)
    var a_out0 = _clone_rb_acts(rbo0.acts, ctx)
    h = rbo0.out.clone(ctx)
    var sto0 = spatial_transformer_forward[RB, S2, S2, C3, RNKV, RCCTX, HH3, RHD, CFF_O0, RG, 1](
        h, context.clone(ctx), w.out0_st, ctx)
    var a_out0_st = sto0.acts.copy()
    h = sto0.out.clone(ctx)

    # out1: cat(h[64], skip_in4[32]) = 96 @ 2x2 -> Res(96->64) + ST + Up[2->4]
    h = concat(3, ctx, h, skip_in4)                                                # [B,2,2,96]
    var rbo1 = resblock_forward[RB, S2, S2, 96, C3, RTEMB, RG](h, emb, w.out1, ctx)
    var a_out1 = _clone_rb_acts(rbo1.acts, ctx)
    h = rbo1.out.clone(ctx)
    var sto1 = spatial_transformer_forward[RB, S2, S2, C3, RNKV, RCCTX, HH3, RHD, CFF_O1, RG, 1](
        h, context.clone(ctx), w.out1_st, ctx)
    var a_out1_st = sto1.acts.copy()
    h = sto1.out.clone(ctx)
    var up1 = upsample_forward[RB, S2, S2, C3](h, w.out1_up_w, w.out1_up_b, ctx)    # [B,4,4,64]
    var a_out1_up = up1.up.clone(ctx)
    h = up1.out.clone(ctx)

    # out2: cat(h[64], skip_in3[32]) = 96 @ 4x4 -> Res(96->32)
    h = concat(3, ctx, h, skip_in3)                                                # [B,4,4,96]
    var rbo2 = resblock_forward[RB, S1, S1, 96, C2, RTEMB, RG](h, emb, w.out2, ctx)
    var a_out2 = _clone_rb_acts(rbo2.acts, ctx)
    h = rbo2.out.clone(ctx)

    # out3: cat(h[32], skip_in2[16]) = 48 @ 4x4 -> Res(48->32) + Up[4->8]
    h = concat(3, ctx, h, skip_in2)                                                # [B,4,4,48]
    var rbo3 = resblock_forward[RB, S1, S1, 48, C2, RTEMB, RG](h, emb, w.out3, ctx)
    var a_out3 = _clone_rb_acts(rbo3.acts, ctx)
    h = rbo3.out.clone(ctx)
    var up3 = upsample_forward[RB, S1, S1, C2](h, w.out3_up_w, w.out3_up_b, ctx)    # [B,8,8,32]
    var a_out3_up = up3.up.clone(ctx)
    h = up3.out.clone(ctx)

    # out4: cat(h[32], skip_in1[16]) = 48 @ 8x8 -> Res(48->16)
    h = concat(3, ctx, h, skip_in1)                                                # [B,8,8,48]
    var rbo4 = resblock_forward[RB, S0, S0, 48, C1, RTEMB, RG](h, emb, w.out4, ctx)
    var a_out4 = _clone_rb_acts(rbo4.acts, ctx)
    h = rbo4.out.clone(ctx)

    # out5: cat(h[16], skip_in0[16]) = 32 @ 8x8 -> Res(32->16)
    h = concat(3, ctx, h, skip_in0)                                                # [B,8,8,32]
    var rbo5 = resblock_forward[RB, S0, S0, 32, C1, RTEMB, RG](h, emb, w.out5, ctx)
    var a_out5 = _clone_rb_acts(rbo5.acts, ctx)
    h = rbo5.out.clone(ctx)

    # ── final GN -> SiLU -> conv_out ──
    var final_gn_in = h.clone(ctx)
    var gnf = group_norm(h, w.out_gn_w.clone(ctx), w.out_gn_b.clone(ctx), RG, GN_EPS_RES, ctx)
    var final_silu_in = gnf.clone(ctx)
    var sf = silu(gnf, ctx)
    var conv_out_in = sf.clone(ctx)
    var out = _conv3x3_fwd[RB, S0, S0, RMC, ROUT](sf, w.conv_out_w, w.conv_out_b, ctx)  # [B,8,8,4]

    var acts = SdxlStackActsReduced(
        a_emb^, emb^, conv_in_x^,
        a_in1^, in2_x^,
        a_in3^, a_in3_st^, in4_x^,
        a_in5^, a_in5_st^,
        a_mid0^, a_mid_st^, a_mid2^,
        a_out0^, a_out0_st^,
        a_out1^, a_out1_st^, a_out1_up^,
        a_out2^,
        a_out3^, a_out3_up^,
        a_out4^,
        a_out5^,
        final_gn_in^, final_silu_in^, conv_out_in^,
    )
    return SdxlStackFwdReduced(out^, acts^)


# ── FULL BACKWARD (reduced config) ───────────────────────────────────────────
# go: dL/dout [B,8,8,4]. Threads d_h in reverse; splits each decoder concat back
# into (carry, skip) and ADDS the skip slab into the matching encoder block's
# output grad at the moment the reverse walk reaches that block.
def sdxl_unet_stack_backward_reduced(
    go: Tensor, x: Tensor, t: Tensor, y: Tensor, context: Tensor,
    acts: SdxlStackActsReduced, w: SdxlStackWeightsReduced, ctx: DeviceContext,
) raises -> SdxlStackGradsReduced:
    # accumulate d_context across every ST + d_emb across every resblock.
    var d_context = _zeros_ctx(RB, RNKV, RCCTX, ctx)
    var d_emb = _zeros_emb(RB, RTEMB, ctx)

    # ── final: conv_out bwd -> SiLU bwd -> GN bwd ──
    var gco = _conv3x3_bwd[RB, S0, S0, RMC, ROUT](go, acts.conv_out_in, w.conv_out_w, ctx)
    var d_conv_out_w = gco.d_w.clone(ctx)
    var d_silu = silu_backward(gco.d_x, acts.final_silu_in, ctx)
    var ggn = group_norm_backward(d_silu, acts.final_gn_in, w.out_gn_w, RG, GN_EPS_RES, ctx)
    var d_h = ggn.d_x.clone(ctx)    # grad into out5 output [B,8,8,16]

    # ── out5 Res(32->16) bwd ; split cat(h[16],skip_in0[16]) ──
    var go5 = resblock_backward[RB, S0, S0, 32, C1, RTEMB, RG](d_h, acts.out5_a, w.out5, ctx)
    d_emb = add(d_emb, go5.d_emb_in, ctx)
    var c5 = cat_backward(go5.d_x, C1, C1, 3, ctx)   # [16 | 16]
    var d_h_carry = c5.d_0.clone(ctx)                # -> out4 carry [B,8,8,16]
    var d_skip_in0 = c5.d_1.clone(ctx)               # skip slab -> in0 (conv_in out)

    # ── out4 Res(48->16) bwd ; split cat(h[32],skip_in1[16]) ──
    var go4 = resblock_backward[RB, S0, S0, 48, C1, RTEMB, RG](d_h_carry, acts.out4_a, w.out4, ctx)
    d_emb = add(d_emb, go4.d_emb_in, ctx)
    var c4 = cat_backward(go4.d_x, C2, C1, 3, ctx)   # [32 | 16]
    d_h_carry = c4.d_0.clone(ctx)                    # -> out3-up carry [B,8,8,32]
    var d_skip_in1 = c4.d_1.clone(ctx)               # skip slab -> in1 (Res16->16 out)

    # ── out3 Up bwd [8->4] then Res(48->32) bwd ; split cat(h[32],skip_in2[16]) ──
    var gu3 = upsample_backward[RB, S1, S1, C2](d_h_carry, acts.out3_up_up, w.out3_up_w, ctx)
    var d_out3_up_w = gu3.d_w.clone(ctx)
    var go3 = resblock_backward[RB, S1, S1, 48, C2, RTEMB, RG](gu3.d_x, acts.out3_a, w.out3, ctx)
    d_emb = add(d_emb, go3.d_emb_in, ctx)
    var c3 = cat_backward(go3.d_x, C2, C1, 3, ctx)   # [32 | 16]
    d_h_carry = c3.d_0.clone(ctx)                    # -> out2 carry [B,4,4,32]
    var d_skip_in2 = c3.d_1.clone(ctx)               # skip slab -> in2 (down16 out)

    # ── out2 Res(96->32) bwd ; split cat(h[64],skip_in3[32]) ──
    var go2 = resblock_backward[RB, S1, S1, 96, C2, RTEMB, RG](d_h_carry, acts.out2_a, w.out2, ctx)
    d_emb = add(d_emb, go2.d_emb_in, ctx)
    var c2 = cat_backward(go2.d_x, C3, C2, 3, ctx)   # [64 | 32]
    d_h_carry = c2.d_0.clone(ctx)                    # -> out1-up carry [B,4,4,64]
    var d_skip_in3 = c2.d_1.clone(ctx)               # skip slab -> in3 (Res+ST out)

    # ── out1 Up bwd [4->2] -> ST bwd -> Res(96->64) bwd ; split cat(h[64],skip_in4[32]) ──
    var gu1 = upsample_backward[RB, S2, S2, C3](d_h_carry, acts.out1_up_up, w.out1_up_w, ctx)
    var d_out1_up_w = gu1.d_w.clone(ctx)
    var gsto1 = spatial_transformer_backward[RB, S2, S2, C3, RNKV, RCCTX, HH3, RHD, CFF_O1, RG, 1](
        gu1.d_x, acts.out1_st_a, w.out1_st, ctx)
    d_context = add(d_context, gsto1.d_context, ctx)
    var go1 = resblock_backward[RB, S2, S2, 96, C3, RTEMB, RG](gsto1.d_x, acts.out1_a, w.out1, ctx)
    d_emb = add(d_emb, go1.d_emb_in, ctx)
    var c1 = cat_backward(go1.d_x, C3, C2, 3, ctx)   # [64 | 32]
    d_h_carry = c1.d_0.clone(ctx)                    # -> out0 carry [B,2,2,64]
    var d_skip_in4 = c1.d_1.clone(ctx)               # skip slab -> in4 (down32 out)

    # ── out0 ST bwd -> Res(128->64) bwd ; split cat(h[64],skip_in5[64]) ──
    var gsto0 = spatial_transformer_backward[RB, S2, S2, C3, RNKV, RCCTX, HH3, RHD, CFF_O0, RG, 1](
        d_h_carry, acts.out0_st_a, w.out0_st, ctx)
    d_context = add(d_context, gsto0.d_context, ctx)
    var go0 = resblock_backward[RB, S2, S2, 128, C3, RTEMB, RG](gsto0.d_x, acts.out0_a, w.out0, ctx)
    d_emb = add(d_emb, go0.d_emb_in, ctx)
    var c0 = cat_backward(go0.d_x, C3, C3, 3, ctx)   # [64 | 64]
    var d_mid_out = c0.d_0.clone(ctx)                # -> middle output grad [B,2,2,64]
    var d_skip_in5 = c0.d_1.clone(ctx)               # skip slab -> in5 (Res+ST out)

    # ── middle bwd: Res2 -> ST -> Res0 ──
    var grbm2 = resblock_backward[RB, S2, S2, C3, C3, RTEMB, RG](d_mid_out, acts.mid2_a, w.mid2, ctx)
    d_emb = add(d_emb, grbm2.d_emb_in, ctx)
    var gstm = spatial_transformer_backward[RB, S2, S2, C3, RNKV, RCCTX, HH3, RHD, CFF3, RG, 1](
        grbm2.d_x, acts.mid_st_a, w.mid_st, ctx)
    d_context = add(d_context, gstm.d_context, ctx)
    var grbm0 = resblock_backward[RB, S2, S2, C3, C3, RTEMB, RG](gstm.d_x, acts.mid0_a, w.mid0, ctx)
    d_emb = add(d_emb, grbm0.d_emb_in, ctx)
    var d_in5_out = grbm0.d_x.clone(ctx)             # grad into in5's output (from middle)

    # ── in5 output grad = (from middle) + (skip slab from out0 concat) ──
    d_in5_out = add(d_in5_out, d_skip_in5, ctx)
    # in5 ST bwd -> in5 Res(32->64) bwd
    var gst5 = spatial_transformer_backward[RB, S2, S2, C3, RNKV, RCCTX, HH3, RHD, CFF3, RG, 1](
        d_in5_out, acts.in5_st_a, w.in5_st, ctx)
    d_context = add(d_context, gst5.d_context, ctx)
    var gin5 = resblock_backward[RB, S2, S2, C2, C3, RTEMB, RG](gst5.d_x, acts.in5_a, w.in5, ctx)
    d_emb = add(d_emb, gin5.d_emb_in, ctx)
    var d_in4_out = gin5.d_x.clone(ctx)              # grad into in4's output [B,2,2,32]
    # capture in5 ST attn2 to_q grad (representative)
    var d_in5_st_q2 = gst5.block_grads[0].a2.d_to_q_w[].clone(ctx)

    # ── in4 output grad = (from in5) + (skip slab from out1 concat) ──
    d_in4_out = add(d_in4_out, d_skip_in4, ctx)
    var gin4 = downsample_backward[RB, S1, S1, C2](d_in4_out, acts.in4_x, w.in4_op_w, ctx)
    var d_in3_out = gin4.d_x.clone(ctx)              # grad into in3's output [B,4,4,32]

    # ── in3 output grad = (from in4) + (skip slab from out2 concat) ──
    d_in3_out = add(d_in3_out, d_skip_in3, ctx)
    var gst3 = spatial_transformer_backward[RB, S1, S1, C2, RNKV, RCCTX, HH2, RHD, CFF2, RG, 1](
        d_in3_out, acts.in3_st_a, w.in3_st, ctx)
    d_context = add(d_context, gst3.d_context, ctx)
    var gin3 = resblock_backward[RB, S1, S1, C1, C2, RTEMB, RG](gst3.d_x, acts.in3_a, w.in3, ctx)
    d_emb = add(d_emb, gin3.d_emb_in, ctx)
    var d_in2_out = gin3.d_x.clone(ctx)              # grad into in2's output [B,4,4,16]

    # ── in2 output grad = (from in3) + (skip slab from out3 concat) ──
    d_in2_out = add(d_in2_out, d_skip_in2, ctx)
    var gin2 = downsample_backward[RB, S0, S0, C1](d_in2_out, acts.in2_x, w.in2_op_w, ctx)
    var d_in1_out = gin2.d_x.clone(ctx)              # grad into in1's output [B,8,8,16]

    # ── in1 output grad = (from in2) + (skip slab from out4 concat) ──
    d_in1_out = add(d_in1_out, d_skip_in1, ctx)
    var gin1 = resblock_backward[RB, S0, S0, C1, C1, RTEMB, RG](d_in1_out, acts.in1_a, w.in1, ctx)
    d_emb = add(d_emb, gin1.d_emb_in, ctx)
    var d_in0_out = gin1.d_x.clone(ctx)              # grad into in0 (conv_in) output [B,8,8,16]

    # ── in0 (conv_in) output grad = (from in1) + (skip slab from out5 concat) ──
    d_in0_out = add(d_in0_out, d_skip_in0, ctx)
    var gci = _conv3x3_bwd[RB, S0, S0, RIN, RMC](d_in0_out, acts.conv_in_x, w.conv_in_w, ctx)
    var d_conv_in_w = gci.d_w.clone(ctx)
    var d_x = gci.d_x.clone(ctx)                     # grad wrt latent input [B,8,8,4]

    # ── embedding backward (d_emb accumulated from every resblock) ──
    var geb = embed_backward[RB, RSDIM, RTEMB, RADM](d_emb, acts.emb_acts, w.emb, ctx)
    var d_y = geb.d_y.clone(ctx)
    var d_t0_w = geb.dt0_w.clone(ctx)
    var d_l0_w = geb.dl0_w.clone(ctx)

    # representative weight grads for the gate
    var d_mid0_conv1_w = grbm0.d_conv1_w.clone(ctx)
    var d_out0_conv2_w = go0.d_conv2_w.clone(ctx)

    return SdxlStackGradsReduced(
        d_x^, d_context^, d_y^,
        d_conv_in_w^, d_conv_out_w^,
        d_mid0_conv1_w^, d_in5_st_q2^,
        d_out0_conv2_w^, d_t0_w^, d_l0_w^,
    )


def _zeros_ctx(B: Int, Nkv: Int, Cctx: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for _ in range(B * Nkv * Cctx):
        h.append(0.0)
    var s = List[Int](); s.append(B); s.append(Nkv); s.append(Cctx)
    return Tensor.from_host(h, s^, STDtype.F32, ctx)


def _zeros_emb(B: Int, Temb: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for _ in range(B * Temb):
        h.append(0.0)
    var s = List[Int](); s.append(B); s.append(Temb)
    return Tensor.from_host(h, s^, STDtype.F32, ctx)
