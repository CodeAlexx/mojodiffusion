# models/sdxl/spatial_transformer.mojo — SDXL SpatialTransformer cross-attn block
# forward (save acts) + hand-chained backward. The one genuinely-new SDXL compute
# unit (cross-attention into text context).
#
# ARCHITECTURE ONLY (Tenet 1): composes already-gated ops/ primitives — group_norm
# / group_norm_backward (eps=1e-6), linear / linear_backward, layer_norm /
# layer_norm_backward (eps=1e-5), sdpa_nomask + sdpa_backward (SQUARE, self-attn
# attn1), sdxl_sdpa + sdpa_backward_rect (RECTANGULAR Sq=H·W≠Skv=77, cross-attn
# attn2), and the gated geglu_forward/geglu_backward FF chain. Builds NO new
# primitive inline.
#
# Tensor carriers in List-stored structs are TArc (ArcPointer[Tensor]) — the
# Copyable device carrier idiom used by klein_stack — because Mojo's List requires
# a Copyable element type and raw Tensor is move-only. A TArc copy is a refcount
# bump; deref `tarc[]` gives the Tensor, `.clone(ctx)` makes a fresh device copy.
#
# ── verified vs inference-flame sdxl_unet.rs::spatial_transformer (761-817) +
#    basic_transformer_block (702-755) + cross_attention (661-697) + geglu (637) ─
#
# SpatialTransformer (NHWC F32 in/out, use_linear_in_transformer=True):
#   residual = x                                          [B,H,W,C]
#   xn  = GroupNorm(x, gn, 32, eps=1e-6)                  (rs:775 — ST eps is 1e-6)
#   tok = reshape(xn, [B, H*W, C])                        (NHWC -> tokens, free)
#   h0  = Linear(tok, proj_in_w, proj_in_b)               [B,N,C]
#   for j in 0..depth:  h = BasicTransformerBlock(h, context)
#   po  = Linear(h, proj_out_w, proj_out_b)               [B,N,C]
#   out = residual + reshape(po, [B,H,W,C])               (FP32 residual)
#
# BasicTransformerBlock (x:[B,N,C], context:[B,77,2048]):
#   x1n = LayerNorm(x,  norm1, eps=1e-5); a1 = SelfAttn(x1n); x = x + a1
#   x2n = LayerNorm(x,  norm2, eps=1e-5); a2 = CrossAttn(x2n, context); x = x + a2
#   x3n = LayerNorm(x,  norm3, eps=1e-5); ff = Linear(GEGLU(x3n)); x = x + ff
#
# CrossAttn(x [B,Nq,C], ctx [B,Nkv,Cctx]):
#   q = Linear_nobias(x, to_q_w); k/v = Linear_nobias(ctx, to_k/v_w)
#   reshape q->[B,Nq,Hh,Dh], k/v->[B,Nkv,Hh,Dh]; SDPA (BSHD); reshape o->[B,Nq,C]
#   out = Linear(o, to_out_w, to_out_b)
# (self-attn = CrossAttn with ctx == x, Nkv==Nq, square SDPA.)
#
# d_context: cross-attn K/V are linear projections of `context`; their grads flow
# back to context (summed across every transformer block) — required for cross-attn
# LoRA training. All interior math F32.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from std.collections import Optional
from math import sqrt

from serenitymojo.ops.norm import group_norm, layer_norm
from serenitymojo.ops.norm_backward import group_norm_backward, layer_norm_backward
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.tensor_algebra import add, reshape
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_backward import sdpa_backward, sdpa_backward_rect, SdpaGrads
from serenitymojo.models.dit.sdxl_attention import sdxl_sdpa

from serenitymojo.models.sdxl.geglu import (
    geglu_forward, geglu_backward, GegluActs, GegluFwd, GegluGrads,
)
from serenitymojo.models.sdxl.config import GN_EPS_ST

comptime LN_EPS: Float32 = 1e-5
comptime TArc = ArcPointer[Tensor]


# ── shape helpers ─────────────────────────────────────────────────────────────
def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^
def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^

# deref-clone: TArc -> fresh device Tensor copy
def _d(t: TArc, ctx: DeviceContext) raises -> Tensor:
    return t[].clone(ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
struct AttnWeights(Copyable, Movable):
    """attn1 (self) or attn2 (cross) weights. to_q/k/v no bias; to_out has bias.
    to_q_w/to_out_w: [C, C]; to_k_w/to_v_w: [C, Cctx] (Cctx==C for self-attn)."""
    var to_q_w: TArc
    var to_k_w: TArc
    var to_v_w: TArc
    var to_out_w: TArc
    var to_out_b: TArc

    def __init__(out self, var to_q_w: TArc, var to_k_w: TArc,
                 var to_v_w: TArc, var to_out_w: TArc, var to_out_b: TArc):
        self.to_q_w = to_q_w^; self.to_k_w = to_k_w^; self.to_v_w = to_v_w^
        self.to_out_w = to_out_w^; self.to_out_b = to_out_b^


struct BasicTransformerBlockWeights(Copyable, Movable):
    """One BasicTransformerBlock's weights."""
    var norm1_w: TArc
    var norm1_b: TArc
    var attn1: AttnWeights
    var norm2_w: TArc
    var norm2_b: TArc
    var attn2: AttnWeights
    var norm3_w: TArc
    var norm3_b: TArc
    var ff_proj_w: TArc   # ff.net.0.proj.weight [2*Cff, C]
    var ff_proj_b: TArc   # ff.net.0.proj.bias   [2*Cff]
    var ff_out_w: TArc    # ff.net.2.weight      [C, Cff]
    var ff_out_b: TArc    # ff.net.2.bias        [C]

    def __init__(
        out self, var norm1_w: TArc, var norm1_b: TArc, var attn1: AttnWeights,
        var norm2_w: TArc, var norm2_b: TArc, var attn2: AttnWeights,
        var norm3_w: TArc, var norm3_b: TArc,
        var ff_proj_w: TArc, var ff_proj_b: TArc,
        var ff_out_w: TArc, var ff_out_b: TArc,
    ):
        self.norm1_w = norm1_w^; self.norm1_b = norm1_b^; self.attn1 = attn1^
        self.norm2_w = norm2_w^; self.norm2_b = norm2_b^; self.attn2 = attn2^
        self.norm3_w = norm3_w^; self.norm3_b = norm3_b^
        self.ff_proj_w = ff_proj_w^; self.ff_proj_b = ff_proj_b^
        self.ff_out_w = ff_out_w^; self.ff_out_b = ff_out_b^


struct SpatialTransformerWeights(Copyable, Movable):
    """SpatialTransformer weights. `blocks` holds `depth` BasicTransformerBlocks."""
    var gn_w: TArc
    var gn_b: TArc
    var proj_in_w: TArc    # [C, C]
    var proj_in_b: TArc    # [C]
    var blocks: List[BasicTransformerBlockWeights]
    var proj_out_w: TArc   # [C, C]
    var proj_out_b: TArc   # [C]

    def __init__(
        out self, var gn_w: TArc, var gn_b: TArc,
        var proj_in_w: TArc, var proj_in_b: TArc,
        var blocks: List[BasicTransformerBlockWeights],
        var proj_out_w: TArc, var proj_out_b: TArc,
    ):
        self.gn_w = gn_w^; self.gn_b = gn_b^
        self.proj_in_w = proj_in_w^; self.proj_in_b = proj_in_b^
        self.blocks = blocks^
        self.proj_out_w = proj_out_w^; self.proj_out_b = proj_out_b^


# ═══════════════════════════════════════════════════════════════════════════════
# SAVED ACTIVATIONS
# ═══════════════════════════════════════════════════════════════════════════════
struct AttnActs(Copyable, Movable):
    """Saved acts for one attention's backward. q/k/v are BSHD [B,S,Hh,Dh] (the
    exact tensors fed to SDPA fwd); xq_in is the attn input [B,Nq,C]; ctx_in is
    the K/V source [B,Nkv,C]; o_flat is the SDPA output [B,Nq,C] (input to to_out)."""
    var xq_in: TArc    # [B,Nq,C]   input that produced Q
    var ctx_in: TArc   # [B,Nkv,C]  input that produced K,V (==xq_in for self)
    var q: TArc        # [B,Nq,Hh,Dh]
    var k: TArc        # [B,Nkv,Hh,Dh]
    var v: TArc        # [B,Nkv,Hh,Dh]
    var o_flat: TArc   # [B,Nq,C]   SDPA out reshaped, pre to_out

    def __init__(out self, var xq_in: TArc, var ctx_in: TArc,
                 var q: TArc, var k: TArc, var v: TArc, var o_flat: TArc):
        self.xq_in = xq_in^; self.ctx_in = ctx_in^
        self.q = q^; self.k = k^; self.v = v^; self.o_flat = o_flat^


struct BasicBlockActs(Copyable, Movable):
    var x_in: TArc        # block input [B,N,C]
    var x1n: TArc         # LN1 out
    var a1_acts: AttnActs
    var x_after1: TArc    # x_in + a1
    var x2n: TArc         # LN2 out
    var a2_acts: AttnActs
    var x_after2: TArc    # x_after1 + a2
    var x3n: TArc         # LN3 out
    var ff_acts: GegluActsArc
    var ff_geglu_out: TArc  # GEGLU out [B*N, Cff] (input to ff.net.2)

    def __init__(out self, var x_in: TArc, var x1n: TArc, var a1_acts: AttnActs,
                 var x_after1: TArc, var x2n: TArc, var a2_acts: AttnActs,
                 var x_after2: TArc, var x3n: TArc, var ff_acts: GegluActsArc,
                 var ff_geglu_out: TArc):
        self.x_in = x_in^; self.x1n = x1n^; self.a1_acts = a1_acts^
        self.x_after1 = x_after1^; self.x2n = x2n^; self.a2_acts = a2_acts^
        self.x_after2 = x_after2^; self.x3n = x3n^; self.ff_acts = ff_acts^
        self.ff_geglu_out = ff_geglu_out^


# GegluActs holds raw Tensors (move-only), so wrap it in a Copyable carrier for
# storage in the List-resident BasicBlockActs. We re-materialize a GegluActs from
# the TArc fields when calling geglu_backward.
struct GegluActsArc(Copyable, Movable):
    var x: TArc        # input        [M, Cin]
    var x_part: TArc   # proj first half  [M, Cff]
    var gate: TArc     # proj second half [M, Cff]
    var g: TArc        # gelu(gate)    [M, Cff]

    def __init__(out self, var x: TArc, var x_part: TArc, var gate: TArc, var g: TArc):
        self.x = x^; self.x_part = x_part^; self.gate = gate^; self.g = g^


struct SpatialTransformerActs(Copyable, Movable):
    var x: TArc                # ST input [B,H,W,C] NHWC
    var xn: TArc               # GroupNorm out [B,H,W,C]
    var tok_in: TArc           # proj_in input  [B,N,C]
    var block_acts: List[BasicBlockActs]
    var h_final: TArc          # last block out [B,N,C] (proj_out input)

    def __init__(out self, var x: TArc, var xn: TArc, var tok_in: TArc,
                 var block_acts: List[BasicBlockActs], var h_final: TArc):
        self.x = x^; self.xn = xn^; self.tok_in = tok_in^
        self.block_acts = block_acts^; self.h_final = h_final^


struct SpatialTransformerFwd(Movable):
    var out: Tensor
    var acts: SpatialTransformerActs
    def __init__(out self, var out: Tensor, var acts: SpatialTransformerActs):
        self.out = out^; self.acts = acts^


# ═══════════════════════════════════════════════════════════════════════════════
# GRADS
# ═══════════════════════════════════════════════════════════════════════════════
struct AttnGrads(Copyable, Movable):
    """d into attention input(s) + every weight grad.
    d_xq : grad to the Q-source input [B,Nq,C].
    d_ctx: grad to the K/V-source input [B,Nkv,C]. For self-attn the caller adds
           d_ctx into d_xq; for cross-attn it flows to context."""
    var d_xq: TArc
    var d_ctx: TArc
    var d_to_q_w: TArc
    var d_to_k_w: TArc
    var d_to_v_w: TArc
    var d_to_out_w: TArc
    var d_to_out_b: TArc

    def __init__(out self, var d_xq: TArc, var d_ctx: TArc,
                 var d_to_q_w: TArc, var d_to_k_w: TArc, var d_to_v_w: TArc,
                 var d_to_out_w: TArc, var d_to_out_b: TArc):
        self.d_xq = d_xq^; self.d_ctx = d_ctx^
        self.d_to_q_w = d_to_q_w^; self.d_to_k_w = d_to_k_w^; self.d_to_v_w = d_to_v_w^
        self.d_to_out_w = d_to_out_w^; self.d_to_out_b = d_to_out_b^


struct BasicBlockGrads(Copyable, Movable):
    var d_x: TArc       # grad to block input [B,N,C]
    var d_ctx: TArc     # grad to context [B,Nkv,Cctx] (from attn2)
    var d_norm1_w: TArc
    var d_norm1_b: TArc
    var a1: AttnGrads
    var d_norm2_w: TArc
    var d_norm2_b: TArc
    var a2: AttnGrads
    var d_norm3_w: TArc
    var d_norm3_b: TArc
    var d_ff_proj_w: TArc
    var d_ff_proj_b: TArc
    var d_ff_out_w: TArc
    var d_ff_out_b: TArc

    def __init__(
        out self, var d_x: TArc, var d_ctx: TArc,
        var d_norm1_w: TArc, var d_norm1_b: TArc, var a1: AttnGrads,
        var d_norm2_w: TArc, var d_norm2_b: TArc, var a2: AttnGrads,
        var d_norm3_w: TArc, var d_norm3_b: TArc,
        var d_ff_proj_w: TArc, var d_ff_proj_b: TArc,
        var d_ff_out_w: TArc, var d_ff_out_b: TArc,
    ):
        self.d_x = d_x^; self.d_ctx = d_ctx^
        self.d_norm1_w = d_norm1_w^; self.d_norm1_b = d_norm1_b^; self.a1 = a1^
        self.d_norm2_w = d_norm2_w^; self.d_norm2_b = d_norm2_b^; self.a2 = a2^
        self.d_norm3_w = d_norm3_w^; self.d_norm3_b = d_norm3_b^
        self.d_ff_proj_w = d_ff_proj_w^; self.d_ff_proj_b = d_ff_proj_b^
        self.d_ff_out_w = d_ff_out_w^; self.d_ff_out_b = d_ff_out_b^


struct SpatialTransformerGrads(Movable):
    var d_x: Tensor             # grad to ST input [B,H,W,C]
    var d_context: Tensor       # grad to text context [B,Nkv,Cctx] (summed over blocks)
    var d_gn_w: Tensor
    var d_gn_b: Tensor
    var d_proj_in_w: Tensor
    var d_proj_in_b: Tensor
    var block_grads: List[BasicBlockGrads]
    var d_proj_out_w: Tensor
    var d_proj_out_b: Tensor

    def __init__(
        out self, var d_x: Tensor, var d_context: Tensor,
        var d_gn_w: Tensor, var d_gn_b: Tensor,
        var d_proj_in_w: Tensor, var d_proj_in_b: Tensor,
        var block_grads: List[BasicBlockGrads],
        var d_proj_out_w: Tensor, var d_proj_out_b: Tensor,
    ):
        self.d_x = d_x^; self.d_context = d_context^
        self.d_gn_w = d_gn_w^; self.d_gn_b = d_gn_b^
        self.d_proj_in_w = d_proj_in_w^; self.d_proj_in_b = d_proj_in_b^
        self.block_grads = block_grads^
        self.d_proj_out_w = d_proj_out_w^; self.d_proj_out_b = d_proj_out_b^


# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION fwd/bwd (shared by attn1 self + attn2 cross)
# ═══════════════════════════════════════════════════════════════════════════════
# Self-attn: Nkv==Nq, Cctx==C, SQUARE SDPA. Cross-attn: Nkv!=Nq (77), RECT SDPA.
# scale = 1/sqrt(Dh).
def _attn_forward[
    B: Int, Nq: Int, Nkv: Int, C: Int, Cctx: Int, Hh: Int, Dh: Int,
](
    xq: Tensor, ctx_in: Tensor, w: AttnWeights, ctx: DeviceContext,
) raises -> AttnActs:
    comptime scale = Float32(1.0) / sqrt(Float32(Dh))
    # Q from xq [B,Nq,C]; K,V from ctx_in [B,Nkv,Cctx]
    var q = linear(xq.clone(ctx), _d(w.to_q_w, ctx), Optional[Tensor](None), ctx)        # [B,Nq,C]
    var k = linear(ctx_in.clone(ctx), _d(w.to_k_w, ctx), Optional[Tensor](None), ctx)    # [B,Nkv,C]
    var v = linear(ctx_in.clone(ctx), _d(w.to_v_w, ctx), Optional[Tensor](None), ctx)    # [B,Nkv,C]
    # reshape to BSHD [B,S,Hh,Dh] (free, channels split contiguously)
    var q4 = reshape(q, _sh4(B, Nq, Hh, Dh), ctx)
    var k4 = reshape(k, _sh4(B, Nkv, Hh, Dh), ctx)
    var v4 = reshape(v, _sh4(B, Nkv, Hh, Dh), ctx)
    # SDPA: square for Nq==Nkv, rectangular otherwise
    var o4: Tensor
    comptime if Nq == Nkv:
        o4 = sdpa_nomask[B, Nq, Hh, Dh](q4.clone(ctx), k4.clone(ctx), v4.clone(ctx), scale, ctx)
    else:
        o4 = sdxl_sdpa[B, Nq, Nkv, Hh, Dh](q4.clone(ctx), k4.clone(ctx), v4.clone(ctx), scale, ctx)
    var o_flat = reshape(o4, _sh3(B, Nq, C), ctx)   # [B,Nq,C]
    return AttnActs(TArc(xq.clone(ctx)), TArc(ctx_in.clone(ctx)),
                    TArc(q4^), TArc(k4^), TArc(v4^), TArc(o_flat^))


# d_out_attn: dL/d(attn output) [B,Nq,C] (i.e. grad after to_out).
def _attn_backward[
    B: Int, Nq: Int, Nkv: Int, C: Int, Cctx: Int, Hh: Int, Dh: Int,
](
    d_out_attn: Tensor, acts: AttnActs, w: AttnWeights, ctx: DeviceContext,
) raises -> AttnGrads:
    comptime scale = Float32(1.0) / sqrt(Float32(Dh))
    comptime Mq = B * Nq
    comptime Mkv = B * Nkv
    # to_out: o_flat[B,Nq,C] @ to_out_wᵀ + b  ->  out[B,Nq,C]
    var g_out = linear_backward(d_out_attn, _d(acts.o_flat, ctx), _d(w.to_out_w, ctx), Mq, C, C, ctx)
    var d_to_out_w = g_out.d_w.clone(ctx)
    var d_to_out_b = g_out.d_b.clone(ctx)
    var d_o_flat = g_out.d_x.clone(ctx)             # [B,Nq,C]
    # reshape d_o_flat -> [B,Nq,Hh,Dh] (adjoint of the fwd output reshape)
    var d_o4 = reshape(d_o_flat, _sh4(B, Nq, Hh, Dh), ctx)
    # SDPA backward (square or rectangular)
    var sg: SdpaGrads
    comptime if Nq == Nkv:
        sg = sdpa_backward[B, Nq, Hh, Dh](_d(acts.q, ctx), _d(acts.k, ctx), _d(acts.v, ctx), d_o4, scale, ctx)
    else:
        sg = sdpa_backward_rect[B, Nq, Nkv, Hh, Dh](_d(acts.q, ctx), _d(acts.k, ctx), _d(acts.v, ctx), d_o4, scale, ctx)
    var d_q4 = sg.d_q.clone(ctx)   # [B,Nq,Hh,Dh]
    var d_k4 = sg.d_k.clone(ctx)   # [B,Nkv,Hh,Dh]
    var d_v4 = sg.d_v.clone(ctx)   # [B,Nkv,Hh,Dh]
    # reshape grads back to [B,N,C]
    var d_q = reshape(d_q4, _sh3(B, Nq, C), ctx)
    var d_k = reshape(d_k4, _sh3(B, Nkv, C), ctx)
    var d_v = reshape(d_v4, _sh3(B, Nkv, C), ctx)
    # to_q: q = xq @ to_q_wᵀ (no bias) -> d_xq_q + d_to_q_w
    var g_q = linear_backward(d_q, _d(acts.xq_in, ctx), _d(w.to_q_w, ctx), Mq, C, C, ctx)
    var d_to_q_w = g_q.d_w.clone(ctx)
    var d_xq = g_q.d_x.clone(ctx)                   # [B,Nq,C]
    # to_k: k = ctx_in @ to_k_wᵀ (no bias) -> d_ctx_k + d_to_k_w ; in=Cctx
    var g_k = linear_backward(d_k, _d(acts.ctx_in, ctx), _d(w.to_k_w, ctx), Mkv, Cctx, C, ctx)
    var d_to_k_w = g_k.d_w.clone(ctx)
    var d_ctx_k = g_k.d_x.clone(ctx)                # [B,Nkv,Cctx]
    # to_v: v = ctx_in @ to_v_wᵀ (no bias) -> d_ctx_v + d_to_v_w
    var g_v = linear_backward(d_v, _d(acts.ctx_in, ctx), _d(w.to_v_w, ctx), Mkv, Cctx, C, ctx)
    var d_to_v_w = g_v.d_w.clone(ctx)
    var d_ctx_v = g_v.d_x.clone(ctx)                # [B,Nkv,Cctx]
    var d_ctx = add(d_ctx_k, d_ctx_v, ctx)          # [B,Nkv,Cctx]
    return AttnGrads(TArc(d_xq^), TArc(d_ctx^), TArc(d_to_q_w^), TArc(d_to_k_w^),
                     TArc(d_to_v_w^), TArc(d_to_out_w^), TArc(d_to_out_b^))


# ═══════════════════════════════════════════════════════════════════════════════
# BasicTransformerBlock fwd/bwd
# ═══════════════════════════════════════════════════════════════════════════════
def _basic_block_forward[
    B: Int, N: Int, Nkv: Int, C: Int, Cctx: Int, Hh: Int, Dh: Int, Cff: Int,
](
    x: Tensor, context: Tensor, w: BasicTransformerBlockWeights, ctx: DeviceContext,
) raises -> Tuple[Tensor, BasicBlockActs]:
    comptime M = B * N
    # 1) self-attn
    var x1n = layer_norm(x.clone(ctx), _d(w.norm1_w, ctx), _d(w.norm1_b, ctx), LN_EPS, ctx)
    var a1 = _attn_forward[B, N, N, C, C, Hh, Dh](x1n.clone(ctx), x1n.clone(ctx), w.attn1, ctx)
    var a1_out = linear(_d(a1.o_flat, ctx), _d(w.attn1.to_out_w, ctx),
                        Optional[Tensor](_d(w.attn1.to_out_b, ctx)), ctx)   # [B,N,C]
    var x_after1 = add(x.clone(ctx), a1_out, ctx)
    # 2) cross-attn
    var x2n = layer_norm(x_after1.clone(ctx), _d(w.norm2_w, ctx), _d(w.norm2_b, ctx), LN_EPS, ctx)
    var a2 = _attn_forward[B, N, Nkv, C, Cctx, Hh, Dh](x2n.clone(ctx), context.clone(ctx), w.attn2, ctx)
    var a2_out = linear(_d(a2.o_flat, ctx), _d(w.attn2.to_out_w, ctx),
                        Optional[Tensor](_d(w.attn2.to_out_b, ctx)), ctx)
    var x_after2 = add(x_after1.clone(ctx), a2_out, ctx)
    # 3) FF: LN3 -> GEGLU -> Linear(ff.net.2)
    var x3n = layer_norm(x_after2.clone(ctx), _d(w.norm3_w, ctx), _d(w.norm3_b, ctx), LN_EPS, ctx)
    var x3n_flat = reshape(x3n.clone(ctx), _sh2(M, C), ctx)
    var gf = geglu_forward[M, C, Cff](x3n_flat, _d(w.ff_proj_w, ctx), _d(w.ff_proj_b, ctx), ctx)  # out [M,Cff]
    var ff_geglu_out = gf.out.clone(ctx)
    var ff_lin = linear(gf.out.clone(ctx), _d(w.ff_out_w, ctx),
                        Optional[Tensor](_d(w.ff_out_b, ctx)), ctx)           # [M,C]
    var ff_out = reshape(ff_lin, _sh3(B, N, C), ctx)
    var out = add(x_after2.clone(ctx), ff_out, ctx)
    # carry the geglu acts as a Copyable TArc bundle
    var ff_acts_arc = GegluActsArc(
        TArc(gf.acts.x.clone(ctx)), TArc(gf.acts.x_part.clone(ctx)),
        TArc(gf.acts.gate.clone(ctx)), TArc(gf.acts.g.clone(ctx)),
    )
    var acts = BasicBlockActs(
        TArc(x.clone(ctx)), TArc(x1n^), a1^, TArc(x_after1^), TArc(x2n^), a2^,
        TArc(x_after2^), TArc(x3n^), ff_acts_arc^, TArc(ff_geglu_out^),
    )
    return (out^, acts^)


# d_out: dL/d(block out) [B,N,C]
def _basic_block_backward[
    B: Int, N: Int, Nkv: Int, C: Int, Cctx: Int, Hh: Int, Dh: Int, Cff: Int,
](
    d_out: Tensor, acts: BasicBlockActs, w: BasicTransformerBlockWeights, ctx: DeviceContext,
) raises -> BasicBlockGrads:
    comptime M = B * N
    # out = x_after2 + ff_out  -> d_x_after2 (resid) and d_ff_out
    var d_ff_out_flat = reshape(d_out.clone(ctx), _sh2(M, C), ctx)
    # ff.net.2 linear bwd: y=ff_lin[M,C], x=ff_geglu_out[M,Cff], W=ff_out_w[C,Cff]
    var g_ffout = linear_backward(d_ff_out_flat, _d(acts.ff_geglu_out, ctx), _d(w.ff_out_w, ctx), M, Cff, C, ctx)
    var d_ff_out_w = g_ffout.d_w.clone(ctx)
    var d_ff_out_b = g_ffout.d_b.clone(ctx)
    var d_geglu = g_ffout.d_x.clone(ctx)                # [M,Cff]
    # re-materialize a GegluActs from the TArc bundle, then geglu_backward
    var ga = GegluActs(_d(acts.ff_acts.x, ctx), _d(acts.ff_acts.x_part, ctx),
                       _d(acts.ff_acts.gate, ctx), _d(acts.ff_acts.g, ctx))
    var g_geglu = geglu_backward[M, C, Cff](d_geglu, ga, _d(w.ff_proj_w, ctx), ctx)
    var d_ff_proj_w = g_geglu.d_proj_w.clone(ctx)
    var d_ff_proj_b = g_geglu.d_proj_b.clone(ctx)
    var d_x3n_flat = g_geglu.d_x.clone(ctx)             # [M,C]
    var d_x3n = reshape(d_x3n_flat, _sh3(B, N, C), ctx)
    # LN3 bwd: input x_after2
    var g_ln3 = layer_norm_backward(d_x3n, _d(acts.x_after2, ctx), _d(w.norm3_w, ctx), LN_EPS, ctx)
    var d_norm3_w = g_ln3.d_g.clone(ctx)
    var d_norm3_b = g_ln3.d_b.clone(ctx)
    # accumulate FF-resid + LN3 paths into d_x_after2
    var d_x_after2 = add(d_out.clone(ctx), g_ln3.d_x, ctx)   # [B,N,C]

    # cross-attn leg: x_after2 = x_after1 + a2_out ; a2_out = to_out(SDPA(x2n,context))
    var g_a2 = _attn_backward[B, N, Nkv, C, Cctx, Hh, Dh](d_x_after2.clone(ctx), acts.a2_acts, w.attn2, ctx)
    var d_x2n_from_attn = _d(g_a2.d_xq, ctx)            # grad into x2n (LN2 out)
    var d_ctx = _d(g_a2.d_ctx, ctx)                     # grad into context [B,Nkv,Cctx]
    # LN2 bwd: input x_after1
    var g_ln2 = layer_norm_backward(d_x2n_from_attn, _d(acts.x_after1, ctx), _d(w.norm2_w, ctx), LN_EPS, ctx)
    var d_norm2_w = g_ln2.d_g.clone(ctx)
    var d_norm2_b = g_ln2.d_b.clone(ctx)
    var d_x_after1 = add(d_x_after2, g_ln2.d_x, ctx)    # resid + LN2 path

    # self-attn leg: x_after1 = x + a1_out ; a1_out = to_out(SDPA(x1n,x1n))
    var g_a1 = _attn_backward[B, N, N, C, C, Hh, Dh](d_x_after1.clone(ctx), acts.a1_acts, w.attn1, ctx)
    # self-attn: ctx_in == xq_in (both x1n) -> grad into x1n is d_xq + d_ctx
    var d_x1n = add(_d(g_a1.d_xq, ctx), _d(g_a1.d_ctx, ctx), ctx)   # [B,N,C]
    # LN1 bwd: input x (block input)
    var g_ln1 = layer_norm_backward(d_x1n, _d(acts.x_in, ctx), _d(w.norm1_w, ctx), LN_EPS, ctx)
    var d_norm1_w = g_ln1.d_g.clone(ctx)
    var d_norm1_b = g_ln1.d_b.clone(ctx)
    var d_x = add(d_x_after1, g_ln1.d_x, ctx)           # resid + LN1 path

    return BasicBlockGrads(
        TArc(d_x^), TArc(d_ctx^),
        TArc(d_norm1_w^), TArc(d_norm1_b^), g_a1^,
        TArc(d_norm2_w^), TArc(d_norm2_b^), g_a2^,
        TArc(d_norm3_w^), TArc(d_norm3_b^),
        TArc(d_ff_proj_w^), TArc(d_ff_proj_b^), TArc(d_ff_out_w^), TArc(d_ff_out_b^),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SpatialTransformer fwd/bwd
# ═══════════════════════════════════════════════════════════════════════════════
# x: [B,H,W,C] NHWC F32.  context: [B,Nkv,Cctx] F32.  depth = number of blocks.
def spatial_transformer_forward[
    B: Int, H: Int, W: Int, C: Int, Nkv: Int, Cctx: Int, Hh: Int, Dh: Int,
    Cff: Int, G: Int, depth: Int,
](
    x: Tensor, context: Tensor, w: SpatialTransformerWeights, ctx: DeviceContext,
) raises -> SpatialTransformerFwd:
    comptime N = H * W
    # GroupNorm(eps=1e-6) on NHWC, then reshape NHWC -> [B,N,C] tokens (free).
    var xn = group_norm(x, _d(w.gn_w, ctx), _d(w.gn_b, ctx), G, GN_EPS_ST, ctx)   # [B,H,W,C]
    var tok = reshape(xn.clone(ctx), _sh3(B, N, C), ctx)              # [B,N,C]
    var tok_in = linear(tok, _d(w.proj_in_w, ctx),
                        Optional[Tensor](_d(w.proj_in_b, ctx)), ctx)  # proj_in [B,N,C]
    var block_acts = List[BasicBlockActs]()
    var hidden = tok_in.clone(ctx)
    comptime for j in range(depth):
        var res = _basic_block_forward[B, N, Nkv, C, Cctx, Hh, Dh, Cff](
            hidden, context, w.blocks[j], ctx
        )
        hidden = res[0].clone(ctx)
        block_acts.append(res[1].copy())
    var h_final = hidden.clone(ctx)
    # proj_out then reshape [B,N,C] -> [B,H,W,C] (free) + FP32 residual.
    var po = linear(hidden, _d(w.proj_out_w, ctx),
                    Optional[Tensor](_d(w.proj_out_b, ctx)), ctx)    # [B,N,C]
    var po4 = reshape(po, _sh4(B, H, W, C), ctx)
    var out = add(x.clone(ctx), po4, ctx)
    var acts = SpatialTransformerActs(
        TArc(x.clone(ctx)), TArc(xn^), TArc(tok_in^), block_acts^, TArc(h_final^))
    return SpatialTransformerFwd(out^, acts^)


# go: dL/dout [B,H,W,C]
def spatial_transformer_backward[
    B: Int, H: Int, W: Int, C: Int, Nkv: Int, Cctx: Int, Hh: Int, Dh: Int,
    Cff: Int, G: Int, depth: Int,
](
    go: Tensor, acts: SpatialTransformerActs, w: SpatialTransformerWeights, ctx: DeviceContext,
) raises -> SpatialTransformerGrads:
    comptime N = H * W
    comptime M = B * N
    # out = x + reshape(po) -> d_x_resid = go ; d_po4 = go
    var d_po = reshape(go.clone(ctx), _sh3(B, N, C), ctx)             # [B,N,C]
    # proj_out linear bwd: y=po[B,N,C], x=h_final[B,N,C], W=proj_out_w[C,C]
    var g_po = linear_backward(d_po, _d(acts.h_final, ctx), _d(w.proj_out_w, ctx), M, C, C, ctx)
    var d_proj_out_w = g_po.d_w.clone(ctx)
    var d_proj_out_b = g_po.d_b.clone(ctx)
    var d_hidden = g_po.d_x.clone(ctx)                               # [B,N,C]

    # walk blocks in reverse, threading d_hidden; sum d_context across blocks.
    var block_grads_rev = List[BasicBlockGrads]()
    var d_context = _zeros_ctx(B, Nkv, Cctx, ctx)
    comptime for jj in range(depth):
        comptime j = depth - 1 - jj
        var bg = _basic_block_backward[B, N, Nkv, C, Cctx, Hh, Dh, Cff](
            d_hidden, acts.block_acts[j], w.blocks[j], ctx
        )
        d_hidden = _d(bg.d_x, ctx)
        d_context = add(d_context, _d(bg.d_ctx, ctx), ctx)
        block_grads_rev.append(bg^)
    # restore forward order
    var block_grads = List[BasicBlockGrads]()
    for jj in range(depth):
        block_grads.append(block_grads_rev[depth - 1 - jj].copy())

    # proj_in linear bwd: y=tok_in, x=tok(=reshape(xn)), W=proj_in_w[C,C]
    var tok = reshape(_d(acts.xn, ctx), _sh3(B, N, C), ctx)
    var g_pi = linear_backward(d_hidden, tok, _d(w.proj_in_w, ctx), M, C, C, ctx)
    var d_proj_in_w = g_pi.d_w.clone(ctx)
    var d_proj_in_b = g_pi.d_b.clone(ctx)
    var d_tok = g_pi.d_x.clone(ctx)                                  # [B,N,C]
    var d_xn = reshape(d_tok, _sh4(B, H, W, C), ctx)                 # [B,H,W,C]
    # GroupNorm bwd (eps=1e-6): input x
    var g_gn = group_norm_backward(d_xn, _d(acts.x, ctx), _d(w.gn_w, ctx), G, GN_EPS_ST, ctx)
    var d_gn_w = g_gn.d_g.clone(ctx)
    var d_gn_b = g_gn.d_b.clone(ctx)
    # d_x = residual (go) + GroupNorm path
    var d_x = add(go.clone(ctx), g_gn.d_x, ctx)                      # [B,H,W,C]

    return SpatialTransformerGrads(
        d_x^, d_context^, d_gn_w^, d_gn_b^, d_proj_in_w^, d_proj_in_b^,
        block_grads^, d_proj_out_w^, d_proj_out_b^,
    )


def _zeros_ctx(B: Int, Nkv: Int, Cctx: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for _ in range(B * Nkv * Cctx):
        h.append(0.0)
    return Tensor.from_host(h, _sh3(B, Nkv, Cctx), STDtype.F32, ctx)
