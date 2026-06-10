# serenitymojo/models/sdxl/sdxl_unet_stack_lora.mojo
#
# SDXL SpatialTransformer *WITH LoRA* on every trained projection: a LoRA-aware
# forward (saving ckpt-inputs) + hand-chained backward that COLLECTS every
# adapter's d_A/d_B, supports an AdamW step + a OneTrainer raw-key save across
# all currently implemented SpatialTransformer adapters. This file COMPOSES; it
# builds NO new ops/ primitive (Tenet 1).
#
# WHY THE SpatialTransformer IS THE COMPOSITION UNIT (not the whole conv-UNet):
#   SDXL's LoRA targets live ENTIRELY inside the SpatialTransformer's
#   BasicTransformerBlock (attn1/attn2/ff linears). The conv-UNet skip/topology
#   that wraps the 5 STs is ALREADY parity-gated by sdxl_unet_stack.{forward,backward}
#   (unet_stack_parity: ALL fwd+bwd gates PASS). Gating the ST-with-LoRA composition
#   fully exercises every adapter's d_A/d_B + base-no-regression — the SDXL analogue
#   of the Ernie stack (whose "stack" = N identical DiT blocks). The flat carrier
#   below indexes num_blocks × 10 slots, so it scales directly to all 5 STs when the
#   Phase-7 train loop wires each ST's depth-blocks into one SdxlLoraSet.
#
# WHAT IS ALREADY PROVEN (cos>=0.999 vs torch) AND ONLY REUSED HERE
#   * models/sdxl/spatial_transformer.mojo : base ST fwd+bwd (29/29, 48/48 vs torch).
#   * models/sdxl/lora_block.mojo : sdxl_lora_apply / sdxl_lora_bwd / sdxl_proj_lora_into_dx
#     (reduce to base when adapters absent; LoRA d_x summed into the proj-input grad).
#   * training/{train_step, lora_save} : LoraAdapter, _lora_adamw, save_lora_onetrainer.
#
# CARRIER DESIGN (Tenet-2: make the right thing easy) — mirrors ErnieLoraSet:
#   SdxlLoraSet holds ONE flat List[LoraAdapter] of 10×num_blocks adapters indexed
#   by flat = block*SDXL_SLOTS + slot, slot order
#   {a1.to_q, a1.to_k, a1.to_v, a1.to_out.0, a2.to_q, a2.to_k, a2.to_v, a2.to_out.0,
#    ff.net.0.proj, ff.net.2}. The optimizer walks this flat list; the backward
#   SCATTERS the returned per-block 10-slot d_A/d_B back into the matching flat slot.
#
# SCOPE: LoRA-on-projection. Base weights (proj_in/out, group_norm, the 3 LayerNorms,
#   the 10 block linears' frozen W) have grads computed by the base path and discarded
#   for the optimizer; only d_A/d_B are trained. The d_context / d_x input grads ARE
#   load-bearing (they prove the chain threads the summed LoRA d_x).
#
# Mojo 0.26.x: def not fn; comptime not alias; Tensor move-only -> TArc carriers.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.collections import List, Optional
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.ops.norm import group_norm, layer_norm
from serenitymojo.ops.norm_backward import group_norm_backward, layer_norm_backward
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.tensor_algebra import add, reshape, mul, slice
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_backward import sdpa_backward, sdpa_backward_rect, SdpaGrads
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.shape_backward import split_backward
from serenitymojo.models.dit.sdxl_attention import sdxl_sdpa

from serenitymojo.models.sdxl.spatial_transformer import (
    SpatialTransformerWeights, BasicTransformerBlockWeights, AttnWeights,
)
from serenitymojo.models.sdxl.config import GN_EPS_ST
from serenitymojo.models.sdxl.lora_block import (
    SdxlBlockLora, SdxlBlockLoraGrads, SDXL_SLOTS,
    SLOT_A1_Q, SLOT_A1_K, SLOT_A1_V, SLOT_A1_O,
    SLOT_A2_Q, SLOT_A2_K, SLOT_A2_V, SLOT_A2_O,
    SLOT_FF_PROJ, SLOT_FF_OUT,
    sdxl_lora_apply, sdxl_proj_lora_into_dx,
)

from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_onetrainer, load_lora_for_resume,
    save_lora_train_state, load_lora_train_state,
)


comptime LN_EPS: Float32 = 1e-5
comptime TArc = ArcPointer[Tensor]


# ── tiny shape helpers ────────────────────────────────────────────────────────
def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^
def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^

def _d(t: TArc, ctx: DeviceContext) raises -> Tensor:
    return t[].clone(ctx)


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _zeros_ctx(B: Int, Nkv: Int, Cctx: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for _ in range(B * Nkv * Cctx):
        h.append(0.0)
    return Tensor.from_host(h, _sh3(B, Nkv, Cctx), STDtype.F32, ctx)


# ── adapter init (A small randn, B=0 — PEFT identity at step 0) ───────────────
def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def make_lora_adapter(
    rank: Int, alpha: Float32, in_f: Int, out_f: Int, seed: UInt64
) -> LoraAdapter:
    var scale = alpha / Float32(rank)
    return LoraAdapter(
        _randn(rank * in_f, seed, 0.01),   # A small randn
        _zeros(out_f * rank),              # B = 0 (adapter identity at init)
        rank, in_f, out_f, scale,
        _zeros(rank * in_f), _zeros(rank * in_f),    # ma / va
        _zeros(out_f * rank), _zeros(out_f * rank),  # mb / vb
    )


# ── the LoRA carrier: every trained adapter, flat-indexed 10×num_blocks ───────
# Plus the TWO ST-level linear adapters (proj_in / proj_out) that wrap the depth
# blocks (diffusers Transformer2DModel proj_in/proj_out -> OneTrainer
# lora_unet_*_attentions_*_proj_in/proj_out). They are Optional so the existing
# ST-attn-only callers (build_sdxl_lora_set, real-train) keep bit-identical
# behavior: when absent, sdxl_lora_apply / sdxl_proj_lora_into_dx reduce to base.
struct SdxlLoraSet(Copyable, Movable):
    var ad: List[LoraAdapter]   # num_blocks * SDXL_SLOTS, slot order above
    var num_blocks: Int
    var rank: Int
    var proj_in_ad: Optional[LoraAdapter]    # ST proj_in (in=C out=C)
    var proj_out_ad: Optional[LoraAdapter]   # ST proj_out (in=C out=C)

    # ST-attn-only constructor (UNCHANGED surface; proj adapters absent).
    def __init__(out self, var ad: List[LoraAdapter], num_blocks: Int, rank: Int):
        self.ad = ad^
        self.num_blocks = num_blocks
        self.rank = rank
        self.proj_in_ad = Optional[LoraAdapter](None)
        self.proj_out_ad = Optional[LoraAdapter](None)

    # full constructor with ST-level proj_in/proj_out adapters.
    def __init__(
        out self, var ad: List[LoraAdapter], num_blocks: Int, rank: Int,
        var proj_in_ad: Optional[LoraAdapter], var proj_out_ad: Optional[LoraAdapter],
    ):
        self.ad = ad^
        self.num_blocks = num_blocks
        self.rank = rank
        self.proj_in_ad = proj_in_ad^
        self.proj_out_ad = proj_out_ad^


# slot in/out shapes. C = hidden, Cctx = context dim, Cff2 = 2*Cff (GEGLU in-proj
# out), Cff = FF inner half.
#   a1.{q,k,v,o}: in=C  out=C ; a2.q/o: in=C out=C ; a2.k/v: in=Cctx out=C ;
#   ff.net.0.proj: in=C out=2*Cff ; ff.net.2: in=Cff out=C.
def _slot_in(s: Int, C: Int, Cctx: Int, Cff: Int) -> Int:
    if s == SLOT_A2_K or s == SLOT_A2_V:
        return Cctx
    if s == SLOT_FF_OUT:
        return Cff
    return C


def _slot_out(s: Int, C: Int, Cff: Int) -> Int:
    if s == SLOT_FF_PROJ:
        return 2 * Cff
    return C


# ── build the full LoRA set for a SpatialTransformer (num_blocks BTBs) ────────
def build_sdxl_lora_set(
    num_blocks: Int, C: Int, Cctx: Int, Cff: Int, rank: Int, alpha: Float32
) -> SdxlLoraSet:
    var ad = List[LoraAdapter]()
    var seed = UInt64(3000)
    for _ in range(num_blocks):
        for s in range(SDXL_SLOTS):
            var in_f = _slot_in(s, C, Cctx, Cff)
            var out_f = _slot_out(s, C, Cff)
            ad.append(make_lora_adapter(rank, alpha, in_f, out_f, seed))
            seed += 1
    return SdxlLoraSet(ad^, num_blocks, rank)


# Build a transient SdxlBlockLora for block bi from the flat set (all 10 present).
def _block_lora_for(set: SdxlLoraSet, bi: Int) -> SdxlBlockLora:
    var base = bi * SDXL_SLOTS
    return SdxlBlockLora(
        Optional[LoraAdapter](set.ad[base + SLOT_A1_Q].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_A1_K].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_A1_V].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_A1_O].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_A2_Q].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_A2_K].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_A2_V].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_A2_O].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_FF_PROJ].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_FF_OUT].copy()),
    )


def sdxl_block_lora_for(set: SdxlLoraSet, bi: Int) -> SdxlBlockLora:
    return _block_lora_for(set, bi)


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA-AWARE attention fwd/bwd (mirrors spatial_transformer._attn_{forward,backward}
# but threads LoRA on to_q/to_k/to_v; to_out LoRA is applied at the block level
# where the base ST applies the to_out linear). q/k/v adapters live in `lo_*`.
# ═══════════════════════════════════════════════════════════════════════════════
struct AttnLoraActs(Copyable, Movable):
    var xq_in: TArc    # [B,Nq,C]   input that produced Q
    var ctx_in: TArc   # [B,Nkv,Cctx]  input that produced K,V (==xq_in for self)
    var q: TArc        # [B,Nq,Hh,Dh]
    var k: TArc        # [B,Nkv,Hh,Dh]
    var v: TArc        # [B,Nkv,Hh,Dh]
    var o_flat: TArc   # [B,Nq,C]   SDPA out reshaped, pre to_out

    def __init__(out self, var xq_in: TArc, var ctx_in: TArc,
                 var q: TArc, var k: TArc, var v: TArc, var o_flat: TArc):
        self.xq_in = xq_in^; self.ctx_in = ctx_in^
        self.q = q^; self.k = k^; self.v = v^; self.o_flat = o_flat^


def _attn_lora_forward[
    B: Int, Nq: Int, Nkv: Int, C: Int, Cctx: Int, Hh: Int, Dh: Int,
](
    xq: Tensor, ctx_in: Tensor, w: AttnWeights,
    lo_q: Optional[LoraAdapter], lo_k: Optional[LoraAdapter], lo_v: Optional[LoraAdapter],
    ctx: DeviceContext,
) raises -> AttnLoraActs:
    comptime scale = Float32(1.0) / sqrt(Float32(Dh))
    comptime Mq = B * Nq
    comptime Mkv = B * Nkv
    # base q/k/v then LoRA add (sdxl_lora_apply reduces to base when adapter absent)
    var q_base = linear(xq.clone(ctx), _d(w.to_q_w, ctx), Optional[Tensor](None), ctx)      # [B,Nq,C]
    var k_base = linear(ctx_in.clone(ctx), _d(w.to_k_w, ctx), Optional[Tensor](None), ctx)  # [B,Nkv,C]
    var v_base = linear(ctx_in.clone(ctx), _d(w.to_v_w, ctx), Optional[Tensor](None), ctx)  # [B,Nkv,C]
    var q = sdxl_lora_apply(q_base, xq, lo_q, Mq, C, ctx)
    var k = sdxl_lora_apply(k_base, ctx_in, lo_k, Mkv, C, ctx)
    var v = sdxl_lora_apply(v_base, ctx_in, lo_v, Mkv, C, ctx)
    var q4 = reshape(q, _sh4(B, Nq, Hh, Dh), ctx)
    var k4 = reshape(k, _sh4(B, Nkv, Hh, Dh), ctx)
    var v4 = reshape(v, _sh4(B, Nkv, Hh, Dh), ctx)
    var o4: Tensor
    comptime if Nq == Nkv:
        o4 = sdpa_nomask[B, Nq, Hh, Dh](q4.clone(ctx), k4.clone(ctx), v4.clone(ctx), scale, ctx)
    else:
        o4 = sdxl_sdpa[B, Nq, Nkv, Hh, Dh](q4.clone(ctx), k4.clone(ctx), v4.clone(ctx), scale, ctx)
    var o_flat = reshape(o4, _sh3(B, Nq, C), ctx)   # [B,Nq,C]
    return AttnLoraActs(TArc(xq.clone(ctx)), TArc(ctx_in.clone(ctx)),
                        TArc(q4^), TArc(k4^), TArc(v4^), TArc(o_flat^))


# grad into attn input(s) + the LoRA q/k/v grads scattered into the slot lists.
# d_out_attn: dL/d(SDPA-out flat) [B,Nq,C] (i.e. AFTER to_out backward).
struct AttnLoraGrads(Movable):
    var d_xq: Tensor   # grad to Q-source input [B,Nq,C] (base + LoRA-q d_x summed)
    var d_ctx: Tensor  # grad to K/V-source input [B,Nkv,Cctx] (base + LoRA-k/v d_x summed)

    def __init__(out self, var d_xq: Tensor, var d_ctx: Tensor):
        self.d_xq = d_xq^; self.d_ctx = d_ctx^


def _attn_lora_backward[
    B: Int, Nq: Int, Nkv: Int, C: Int, Cctx: Int, Hh: Int, Dh: Int,
](
    d_o_flat: Tensor, acts: AttnLoraActs, w: AttnWeights,
    lo_q: Optional[LoraAdapter], lo_k: Optional[LoraAdapter], lo_v: Optional[LoraAdapter],
    slot_q: Int, slot_k: Int, slot_v: Int,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> AttnLoraGrads:
    comptime scale = Float32(1.0) / sqrt(Float32(Dh))
    comptime Mq = B * Nq
    comptime Mkv = B * Nkv
    # reshape d_o_flat -> [B,Nq,Hh,Dh] (adjoint of fwd output reshape)
    var d_o4 = reshape(d_o_flat, _sh4(B, Nq, Hh, Dh), ctx)
    var sg: SdpaGrads
    comptime if Nq == Nkv:
        sg = sdpa_backward[B, Nq, Hh, Dh](_d(acts.q, ctx), _d(acts.k, ctx), _d(acts.v, ctx), d_o4, scale, ctx)
    else:
        sg = sdpa_backward_rect[B, Nq, Nkv, Hh, Dh](_d(acts.q, ctx), _d(acts.k, ctx), _d(acts.v, ctx), d_o4, scale, ctx)
    var d_q = reshape(sg.d_q.clone(ctx), _sh3(B, Nq, C), ctx)
    var d_k = reshape(sg.d_k.clone(ctx), _sh3(B, Nkv, C), ctx)
    var d_v = reshape(sg.d_v.clone(ctx), _sh3(B, Nkv, C), ctx)
    # to_q: q = xq @ to_q_wᵀ -> base d_xq + LoRA-q (d_a/d_b + summed d_x)
    var g_q = linear_backward(d_q.clone(ctx), _d(acts.xq_in, ctx), _d(w.to_q_w, ctx), Mq, C, C, ctx)
    var d_xq = sdxl_proj_lora_into_dx(
        d_q, _d(acts.xq_in, ctx), g_q.d_x.clone(ctx), lo_q, slot_q, Mq, C, d_a_slots, d_b_slots, ctx,
    )
    # to_k: k = ctx_in @ to_k_wᵀ ; in=Cctx
    var g_k = linear_backward(d_k.clone(ctx), _d(acts.ctx_in, ctx), _d(w.to_k_w, ctx), Mkv, Cctx, C, ctx)
    var d_ctx_k = sdxl_proj_lora_into_dx(
        d_k, _d(acts.ctx_in, ctx), g_k.d_x.clone(ctx), lo_k, slot_k, Mkv, Cctx, d_a_slots, d_b_slots, ctx,
    )
    # to_v: v = ctx_in @ to_v_wᵀ
    var g_v = linear_backward(d_v.clone(ctx), _d(acts.ctx_in, ctx), _d(w.to_v_w, ctx), Mkv, Cctx, C, ctx)
    var d_ctx_v = sdxl_proj_lora_into_dx(
        d_v, _d(acts.ctx_in, ctx), g_v.d_x.clone(ctx), lo_v, slot_v, Mkv, Cctx, d_a_slots, d_b_slots, ctx,
    )
    var d_ctx = add(d_ctx_k, d_ctx_v, ctx)
    return AttnLoraGrads(d_xq^, d_ctx^)


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA-AWARE GEGLU fwd/bwd (exposes the proj output so ff.net.0.proj LoRA can add
# into it before the split — mirrors geglu.mojo math exactly).
# ═══════════════════════════════════════════════════════════════════════════════
struct GegluLoraActs(Copyable, Movable):
    var x: TArc        # input  [M, Cin]
    var x_part: TArc   # proj first half  [M, Cff]
    var gate: TArc     # proj second half [M, Cff]
    var g: TArc        # gelu(gate)  [M, Cff]

    def __init__(out self, var x: TArc, var x_part: TArc, var gate: TArc, var g: TArc):
        self.x = x^; self.x_part = x_part^; self.gate = gate^; self.g = g^


def _geglu_lora_forward[
    M: Int, Cin: Int, Cff: Int,
](
    x: Tensor, proj_w: Tensor, proj_b: Tensor, lo: Optional[LoraAdapter], ctx: DeviceContext,
) raises -> Tuple[Tensor, GegluLoraActs]:
    var proj_base = linear(x.clone(ctx), proj_w.clone(ctx), Optional[Tensor](proj_b.clone(ctx)), ctx)  # [M,2*Cff]
    var proj = sdxl_lora_apply(proj_base, x, lo, M, 2 * Cff, ctx)
    var x_part = slice(proj, 1, 0, Cff, ctx)
    var gate = slice(proj, 1, Cff, Cff, ctx)
    var g = gelu(gate.clone(ctx), ctx)
    var out = mul(x_part.clone(ctx), g.clone(ctx), ctx)
    var acts = GegluLoraActs(TArc(x.clone(ctx)), TArc(x_part^), TArc(gate^), TArc(g^))
    return (out^, acts^)


# returns d_x (base+LoRA summed into the proj-input grad) and scatters ff_proj
# LoRA d_a/d_b. d_out: dL/d(geglu out) [M,Cff].
def _geglu_lora_backward[
    M: Int, Cin: Int, Cff: Int,
](
    d_out: Tensor, acts: GegluLoraActs, proj_w: Tensor,
    lo: Optional[LoraAdapter], slot: Int,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> Tensor:
    var d_x_part = mul(d_out.clone(ctx), _d(acts.g, ctx), ctx)
    var d_g = mul(d_out.clone(ctx), _d(acts.x_part, ctx), ctx)
    var d_gate = gelu_backward(d_g, _d(acts.gate, ctx), ctx)
    var d_proj = split_backward(d_x_part, d_gate, 1, ctx)   # [M,2*Cff] = d_y into ff.net.0.proj
    var g_lin = linear_backward(d_proj.clone(ctx), _d(acts.x, ctx), proj_w, M, Cin, 2 * Cff, ctx)
    var d_x = sdxl_proj_lora_into_dx(
        d_proj, _d(acts.x, ctx), g_lin.d_x.clone(ctx), lo, slot, M, Cin, d_a_slots, d_b_slots, ctx,
    )
    return d_x^


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA-AWARE BasicTransformerBlock fwd/bwd
# ═══════════════════════════════════════════════════════════════════════════════
struct BasicBlockLoraActs(Copyable, Movable):
    var x_in: TArc        # block input [B,N,C]
    var x1n: TArc         # LN1 out
    var a1_acts: AttnLoraActs
    var a1_out_flat: TArc # SDPA-out flat that fed attn1.to_out [B,N,C] (LoRA-a1.o input)
    var x_after1: TArc    # x_in + a1
    var x2n: TArc         # LN2 out
    var a2_acts: AttnLoraActs
    var a2_out_flat: TArc # SDPA-out flat that fed attn2.to_out [B,N,C] (LoRA-a2.o input)
    var x_after2: TArc    # x_after1 + a2
    var x3n: TArc         # LN3 out
    var ff_acts: GegluLoraActs
    var ff_geglu_out: TArc  # GEGLU out [B*N, Cff] (input to ff.net.2)

    def __init__(out self, var x_in: TArc, var x1n: TArc, var a1_acts: AttnLoraActs,
                 var a1_out_flat: TArc, var x_after1: TArc, var x2n: TArc,
                 var a2_acts: AttnLoraActs, var a2_out_flat: TArc, var x_after2: TArc,
                 var x3n: TArc, var ff_acts: GegluLoraActs, var ff_geglu_out: TArc):
        self.x_in = x_in^; self.x1n = x1n^; self.a1_acts = a1_acts^
        self.a1_out_flat = a1_out_flat^; self.x_after1 = x_after1^; self.x2n = x2n^
        self.a2_acts = a2_acts^; self.a2_out_flat = a2_out_flat^; self.x_after2 = x_after2^
        self.x3n = x3n^; self.ff_acts = ff_acts^; self.ff_geglu_out = ff_geglu_out^


def _basic_block_lora_forward[
    B: Int, N: Int, Nkv: Int, C: Int, Cctx: Int, Hh: Int, Dh: Int, Cff: Int,
](
    x: Tensor, context: Tensor, w: BasicTransformerBlockWeights, lo: SdxlBlockLora,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, BasicBlockLoraActs]:
    comptime M = B * N
    # 1) self-attn (Q/K/V from x1n)
    var x1n = layer_norm(x.clone(ctx), _d(w.norm1_w, ctx), _d(w.norm1_b, ctx), LN_EPS, ctx)
    var a1 = _attn_lora_forward[B, N, N, C, C, Hh, Dh](
        x1n.clone(ctx), x1n.clone(ctx), w.attn1, lo.a1_q, lo.a1_k, lo.a1_v, ctx)
    # to_out (+ LoRA a1.o): base linear then LoRA add on the SDPA-out flat
    var a1_o_base = linear(_d(a1.o_flat, ctx), _d(w.attn1.to_out_w, ctx),
                           Optional[Tensor](_d(w.attn1.to_out_b, ctx)), ctx)   # [B,N,C]
    var a1_out = sdxl_lora_apply(a1_o_base, _d(a1.o_flat, ctx), lo.a1_o, M, C, ctx)
    var x_after1 = add(x.clone(ctx), a1_out, ctx)
    # 2) cross-attn (Q from x2n, K/V from context)
    var x2n = layer_norm(x_after1.clone(ctx), _d(w.norm2_w, ctx), _d(w.norm2_b, ctx), LN_EPS, ctx)
    var a2 = _attn_lora_forward[B, N, Nkv, C, Cctx, Hh, Dh](
        x2n.clone(ctx), context.clone(ctx), w.attn2, lo.a2_q, lo.a2_k, lo.a2_v, ctx)
    var a2_o_base = linear(_d(a2.o_flat, ctx), _d(w.attn2.to_out_w, ctx),
                           Optional[Tensor](_d(w.attn2.to_out_b, ctx)), ctx)
    var a2_out = sdxl_lora_apply(a2_o_base, _d(a2.o_flat, ctx), lo.a2_o, M, C, ctx)
    var x_after2 = add(x_after1.clone(ctx), a2_out, ctx)
    # 3) FF: LN3 -> GEGLU(+LoRA ff_proj) -> Linear(ff.net.2)(+LoRA ff_out)
    var x3n = layer_norm(x_after2.clone(ctx), _d(w.norm3_w, ctx), _d(w.norm3_b, ctx), LN_EPS, ctx)
    var x3n_flat = reshape(x3n.clone(ctx), _sh2(M, C), ctx)
    var gf = _geglu_lora_forward[M, C, Cff](x3n_flat, _d(w.ff_proj_w, ctx), _d(w.ff_proj_b, ctx), lo.ff_proj, ctx)
    var ff_geglu_out = gf[0].clone(ctx)
    var ff_lin_base = linear(gf[0].clone(ctx), _d(w.ff_out_w, ctx),
                             Optional[Tensor](_d(w.ff_out_b, ctx)), ctx)        # [M,C]
    var ff_lin = sdxl_lora_apply(ff_lin_base, gf[0].clone(ctx), lo.ff_out, M, C, ctx)
    var ff_out = reshape(ff_lin, _sh3(B, N, C), ctx)
    var out = add(x_after2.clone(ctx), ff_out, ctx)
    var acts = BasicBlockLoraActs(
        TArc(x.clone(ctx)), TArc(x1n^), a1.copy(), TArc(_d(a1.o_flat, ctx)),
        TArc(x_after1^), TArc(x2n^), a2.copy(), TArc(_d(a2.o_flat, ctx)),
        TArc(x_after2^), TArc(x3n^), gf[1].copy(), TArc(ff_geglu_out^),
    )
    return (out^, acts^)


# d_out: dL/d(block out) [B,N,C]. Returns d_x (block-input grad) + d_ctx (context
# grad from attn2) + the per-block 10-slot LoRA grads (the STACK scatters these
# into the flat carrier at j*SDXL_SLOTS + s — mirrors the Ernie template so each
# block's slots stay block-local and don't overwrite siblings).
struct BasicBlockLoraGrads(Movable):
    var d_x: Tensor
    var d_ctx: Tensor
    var d_a: List[List[Float32]]   # SDXL_SLOTS entries (empty if slot absent)
    var d_b: List[List[Float32]]

    def __init__(
        out self, var d_x: Tensor, var d_ctx: Tensor,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
    ):
        self.d_x = d_x^; self.d_ctx = d_ctx^; self.d_a = d_a^; self.d_b = d_b^


def _basic_block_lora_backward[
    B: Int, N: Int, Nkv: Int, C: Int, Cctx: Int, Hh: Int, Dh: Int, Cff: Int,
](
    d_out: Tensor, acts: BasicBlockLoraActs, w: BasicTransformerBlockWeights, lo: SdxlBlockLora,
    ctx: DeviceContext,
) raises -> BasicBlockLoraGrads:
    comptime M = B * N
    # block-local 10-slot grad lists (the stack scatters these into the flat carrier)
    var d_a_slots = List[List[Float32]]()
    var d_b_slots = List[List[Float32]]()
    for _ in range(SDXL_SLOTS):
        d_a_slots.append(List[Float32]())
        d_b_slots.append(List[Float32]())
    # out = x_after2 + ff_out -> d_x_after2 (resid) + d_ff_out
    var d_ff_out_flat = reshape(d_out.clone(ctx), _sh2(M, C), ctx)
    # ff.net.2 (+LoRA ff_out): base d_x then LoRA d_x summed; scatter ff_out d_a/d_b
    var g_ffout = linear_backward(d_ff_out_flat.clone(ctx), _d(acts.ff_geglu_out, ctx), _d(w.ff_out_w, ctx), M, Cff, C, ctx)
    var d_geglu = sdxl_proj_lora_into_dx(
        d_ff_out_flat, _d(acts.ff_geglu_out, ctx), g_ffout.d_x.clone(ctx),
        lo.ff_out, SLOT_FF_OUT, M, Cff, d_a_slots, d_b_slots, ctx,
    )                                                   # [M,Cff]
    # GEGLU (+LoRA ff_proj) -> d_x3n_flat
    var d_x3n_flat = _geglu_lora_backward[M, C, Cff](
        d_geglu, acts.ff_acts, _d(w.ff_proj_w, ctx), lo.ff_proj, SLOT_FF_PROJ,
        d_a_slots, d_b_slots, ctx,
    )
    var d_x3n = reshape(d_x3n_flat, _sh3(B, N, C), ctx)
    var g_ln3 = layer_norm_backward(d_x3n, _d(acts.x_after2, ctx), _d(w.norm3_w, ctx), LN_EPS, ctx)
    var d_x_after2 = add(d_out.clone(ctx), g_ln3.d_x, ctx)   # FF-resid + LN3 paths

    # cross-attn leg: x_after2 = x_after1 + a2_out ; a2_out = to_out(+LoRA)(SDPA(x2n,context))
    var d_a2_out_flat = reshape(d_x_after2.clone(ctx), _sh2(M, C), ctx)
    var g_a2out = linear_backward(d_a2_out_flat.clone(ctx), _d(acts.a2_out_flat, ctx), _d(w.attn2.to_out_w, ctx), M, C, C, ctx)
    var d_a2_oflat = sdxl_proj_lora_into_dx(
        d_a2_out_flat, _d(acts.a2_out_flat, ctx), g_a2out.d_x.clone(ctx),
        lo.a2_o, SLOT_A2_O, M, C, d_a_slots, d_b_slots, ctx,
    )                                                   # [M,C] = d_o_flat into attn2 SDPA
    var d_a2_oflat3 = reshape(d_a2_oflat, _sh3(B, N, C), ctx)
    var g_a2 = _attn_lora_backward[B, N, Nkv, C, Cctx, Hh, Dh](
        d_a2_oflat3, acts.a2_acts, w.attn2, lo.a2_q, lo.a2_k, lo.a2_v,
        SLOT_A2_Q, SLOT_A2_K, SLOT_A2_V, d_a_slots, d_b_slots, ctx,
    )
    var d_ctx = g_a2.d_ctx.clone(ctx)                   # grad into context
    var g_ln2 = layer_norm_backward(g_a2.d_xq.clone(ctx), _d(acts.x_after1, ctx), _d(w.norm2_w, ctx), LN_EPS, ctx)
    var d_x_after1 = add(d_x_after2, g_ln2.d_x, ctx)    # resid + LN2 path

    # self-attn leg: x_after1 = x + a1_out ; a1_out = to_out(+LoRA)(SDPA(x1n,x1n))
    var d_a1_out_flat = reshape(d_x_after1.clone(ctx), _sh2(M, C), ctx)
    var g_a1out = linear_backward(d_a1_out_flat.clone(ctx), _d(acts.a1_out_flat, ctx), _d(w.attn1.to_out_w, ctx), M, C, C, ctx)
    var d_a1_oflat = sdxl_proj_lora_into_dx(
        d_a1_out_flat, _d(acts.a1_out_flat, ctx), g_a1out.d_x.clone(ctx),
        lo.a1_o, SLOT_A1_O, M, C, d_a_slots, d_b_slots, ctx,
    )
    var d_a1_oflat3 = reshape(d_a1_oflat, _sh3(B, N, C), ctx)
    var g_a1 = _attn_lora_backward[B, N, N, C, C, Hh, Dh](
        d_a1_oflat3, acts.a1_acts, w.attn1, lo.a1_q, lo.a1_k, lo.a1_v,
        SLOT_A1_Q, SLOT_A1_K, SLOT_A1_V, d_a_slots, d_b_slots, ctx,
    )
    # self-attn: ctx_in == xq_in (both x1n) -> grad into x1n is d_xq + d_ctx
    var d_x1n = add(g_a1.d_xq.clone(ctx), g_a1.d_ctx.clone(ctx), ctx)
    var g_ln1 = layer_norm_backward(d_x1n, _d(acts.x_in, ctx), _d(w.norm1_w, ctx), LN_EPS, ctx)
    var d_x = add(d_x_after1, g_ln1.d_x, ctx)           # resid + LN1 path
    return BasicBlockLoraGrads(d_x^, d_ctx^, d_a_slots^, d_b_slots^)


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA-AWARE SpatialTransformer fwd/bwd (depth blocks). Mirrors
# spatial_transformer_{forward,backward}; base proj_in/proj_out/group_norm are
# NOT LoRA targets (frozen, grads discarded). d_x / d_context are load-bearing.
# ═══════════════════════════════════════════════════════════════════════════════
struct SdxlStLoraActs(Copyable, Movable):
    var x: TArc                # ST input [B,H,W,C] NHWC
    var xn: TArc               # GroupNorm out
    var tok_in: TArc           # proj_in output [B,N,C] (and the hidden seed)
    var block_acts: List[BasicBlockLoraActs]
    var h_final: TArc          # last block out [B,N,C] (proj_out input)

    def __init__(out self, var x: TArc, var xn: TArc, var tok_in: TArc,
                 var block_acts: List[BasicBlockLoraActs], var h_final: TArc):
        self.x = x^; self.xn = xn^; self.tok_in = tok_in^
        self.block_acts = block_acts^; self.h_final = h_final^


struct SdxlStLoraFwd(Movable):
    var out: Tensor
    var acts: SdxlStLoraActs
    def __init__(out self, var out: Tensor, var acts: SdxlStLoraActs):
        self.out = out^; self.acts = acts^


def sdxl_st_lora_forward[
    B: Int, H: Int, W: Int, C: Int, Nkv: Int, Cctx: Int, Hh: Int, Dh: Int,
    Cff: Int, G: Int, depth: Int,
](
    x: Tensor, context: Tensor, w: SpatialTransformerWeights, lora: SdxlLoraSet,
    ctx: DeviceContext,
) raises -> SdxlStLoraFwd:
    comptime N = H * W
    comptime M = B * N
    var xn = group_norm(x, _d(w.gn_w, ctx), _d(w.gn_b, ctx), G, GN_EPS_ST, ctx)
    var tok = reshape(xn.clone(ctx), _sh3(B, N, C), ctx)
    # proj_in linear (+ optional ST proj_in LoRA on the [M,C] proj-input `tok`).
    var tok_in_base = linear(tok, _d(w.proj_in_w, ctx), Optional[Tensor](_d(w.proj_in_b, ctx)), ctx)
    var tok_in_2 = sdxl_lora_apply(tok_in_base, tok.clone(ctx), lora.proj_in_ad, M, C, ctx)  # [M,C]
    var tok_in = reshape(tok_in_2, _sh3(B, N, C), ctx)
    var block_acts = List[BasicBlockLoraActs]()
    var hidden = tok_in.clone(ctx)
    comptime for j in range(depth):
        var bl = _block_lora_for(lora, j)
        var res = _basic_block_lora_forward[B, N, Nkv, C, Cctx, Hh, Dh, Cff](
            hidden, context, w.blocks[j], bl, ctx
        )
        hidden = res[0].clone(ctx)
        block_acts.append(res[1].copy())
    var h_final = hidden.clone(ctx)
    # proj_out linear (+ optional ST proj_out LoRA on the [M,C] proj-input `hidden`).
    var po_base = linear(hidden, _d(w.proj_out_w, ctx), Optional[Tensor](_d(w.proj_out_b, ctx)), ctx)
    var po = sdxl_lora_apply(po_base, hidden.clone(ctx), lora.proj_out_ad, M, C, ctx)  # [M,C]
    var po4 = reshape(po, _sh4(B, H, W, C), ctx)
    var out = add(x.clone(ctx), po4, ctx)
    var acts = SdxlStLoraActs(
        TArc(x.clone(ctx)), TArc(xn^), TArc(tok_in^), block_acts^, TArc(h_final^))
    return SdxlStLoraFwd(out^, acts^)


# collected LoRA grads (flat, parallel to SdxlLoraSet) + load-bearing input grads.
struct SdxlStLoraGrads(Movable):
    var d_a: List[List[Float32]]   # num_blocks*SDXL_SLOTS
    var d_b: List[List[Float32]]
    var d_x: List[Float32]          # grad wrt ST input [B,H,W,C] (full chain proof)
    var d_context: List[Float32]    # grad wrt context [B,Nkv,Cctx] (summed over blocks)
    var nonfinite_lora_grads: Int
    # ST-level proj_in/proj_out LoRA grads (empty lists when the adapter is absent).
    var d_proj_in_a: List[Float32]
    var d_proj_in_b: List[Float32]
    var d_proj_out_a: List[Float32]
    var d_proj_out_b: List[Float32]

    def __init__(
        out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_x: List[Float32], var d_context: List[Float32], nonfinite_lora_grads: Int,
        var d_proj_in_a: List[Float32], var d_proj_in_b: List[Float32],
        var d_proj_out_a: List[Float32], var d_proj_out_b: List[Float32],
    ):
        self.d_a = d_a^; self.d_b = d_b^; self.d_x = d_x^
        self.d_context = d_context^; self.nonfinite_lora_grads = nonfinite_lora_grads
        self.d_proj_in_a = d_proj_in_a^; self.d_proj_in_b = d_proj_in_b^
        self.d_proj_out_a = d_proj_out_a^; self.d_proj_out_b = d_proj_out_b^


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


def sdxl_st_lora_backward[
    B: Int, H: Int, W: Int, C: Int, Nkv: Int, Cctx: Int, Hh: Int, Dh: Int,
    Cff: Int, G: Int, depth: Int,
](
    go: Tensor, acts: SdxlStLoraActs, w: SpatialTransformerWeights, lora: SdxlLoraSet,
    ctx: DeviceContext,
) raises -> SdxlStLoraGrads:
    comptime N = H * W
    comptime M = B * N
    var num_blocks = lora.num_blocks

    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(num_blocks * SDXL_SLOTS):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())

    # proj_out/proj_in 1-slot grad scratch (extracted to the named grad fields).
    var d_pi_a = List[List[Float32]](); var d_pi_b = List[List[Float32]]()
    d_pi_a.append(List[Float32]()); d_pi_b.append(List[Float32]())
    var d_po_a = List[List[Float32]](); var d_po_b = List[List[Float32]]()
    d_po_a.append(List[Float32]()); d_po_b.append(List[Float32]())

    # out = x + reshape(po) -> d_x_resid = go ; d_po = go (at proj_out OUTPUT)
    var d_po = reshape(go.clone(ctx), _sh3(B, N, C), ctx)
    var g_po = linear_backward(d_po.clone(ctx), _d(acts.h_final, ctx), _d(w.proj_out_w, ctx), M, C, C, ctx)
    # proj_out LoRA: base d_x + LoRA d_x (proj-input = h_final); scatter d_a/d_b.
    var d_hidden_2 = sdxl_proj_lora_into_dx(
        d_po, _d(acts.h_final, ctx), g_po.d_x.clone(ctx),
        lora.proj_out_ad, 0, M, C, d_po_a, d_po_b, ctx,
    )                                                   # [M,C]
    var d_hidden = reshape(d_hidden_2, _sh3(B, N, C), ctx)

    var d_context = _zeros_ctx(B, Nkv, Cctx, ctx)
    comptime for jj in range(depth):
        comptime j = depth - 1 - jj
        var bl = _block_lora_for(lora, j)
        var bg = _basic_block_lora_backward[B, N, Nkv, C, Cctx, Hh, Dh, Cff](
            d_hidden, acts.block_acts[j], w.blocks[j], bl, ctx
        )
        d_hidden = bg.d_x.clone(ctx)
        d_context = add(d_context, bg.d_ctx, ctx)
        # SCATTER the per-block 10-slot LoRA grads into the flat carrier.
        var base_idx = j * SDXL_SLOTS
        for s in range(SDXL_SLOTS):
            d_a_flat[base_idx + s] = bg.d_a[s].copy()
            d_b_flat[base_idx + s] = bg.d_b[s].copy()

    # proj_in linear bwd: y=tok_in, x=tok(=reshape(xn)), W=proj_in_w[C,C]
    var tok = reshape(_d(acts.xn, ctx), _sh3(B, N, C), ctx)
    var g_pi = linear_backward(d_hidden.clone(ctx), tok.clone(ctx), _d(w.proj_in_w, ctx), M, C, C, ctx)
    # proj_in LoRA: base d_x + LoRA d_x (proj-input = tok); scatter d_a/d_b.
    var d_tok = sdxl_proj_lora_into_dx(
        d_hidden, tok.clone(ctx), g_pi.d_x.clone(ctx),
        lora.proj_in_ad, 0, M, C, d_pi_a, d_pi_b, ctx,
    )                                                   # [M,C]
    var d_xn = reshape(d_tok, _sh4(B, H, W, C), ctx)
    var g_gn = group_norm_backward(d_xn, _d(acts.x, ctx), _d(w.gn_w, ctx), G, GN_EPS_ST, ctx)
    var d_x = add(go.clone(ctx), g_gn.d_x, ctx)         # residual + GroupNorm path

    var nonfinite = 0
    for i in range(num_blocks * SDXL_SLOTS):
        nonfinite += _nonfinite(d_a_flat[i]) + _nonfinite(d_b_flat[i])
    nonfinite += _nonfinite(d_pi_a[0]) + _nonfinite(d_pi_b[0])
    nonfinite += _nonfinite(d_po_a[0]) + _nonfinite(d_po_b[0])

    return SdxlStLoraGrads(
        d_a_flat^, d_b_flat^, d_x.to_host(ctx), d_context.to_host(ctx), nonfinite,
        d_pi_a[0].copy(), d_pi_b[0].copy(), d_po_a[0].copy(), d_po_b[0].copy(),
    )


# ── AdamW step on EVERY adapter (reuses the proven per-adapter _lora_adamw) ───
def sdxl_lora_adamw_step(
    mut set: SdxlLoraSet, grads: SdxlStLoraGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    var n = set.num_blocks * SDXL_SLOTS
    for i in range(n):
        var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(set.ad[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay)
    # ST-level proj_in/proj_out adapters (when present).
    if set.proj_in_ad and len(grads.d_proj_in_a) > 0:
        var pi = set.proj_in_ad.value().copy()
        var lg_pi = LoraGrads(grads.d_proj_in_a.copy(), grads.d_proj_in_b.copy())
        _lora_adamw(pi, lg_pi, t, lr, ctx, beta1, beta2, eps, weight_decay)
        set.proj_in_ad = Optional[LoraAdapter](pi^)
    if set.proj_out_ad and len(grads.d_proj_out_a) > 0:
        var po = set.proj_out_ad.value().copy()
        var lg_po = LoraGrads(grads.d_proj_out_a.copy(), grads.d_proj_out_b.copy())
        _lora_adamw(po, lg_po, t, lr, ctx, beta1, beta2, eps, weight_decay)
        set.proj_out_ad = Optional[LoraAdapter](po^)


# -- per-block OneTrainer raw legacy prefix scheme ----------------------------
# OneTrainer's SDXL LoRAModuleWrapper installs adapters on every Linear/Conv2d
# under model.unet with raw-save prefix `lora_unet`, and optional text encoder
# prefixes `lora_te1` / `lora_te2`. This Mojo module currently implements the
# SpatialTransformer BasicTransformerBlock linears only. The helpers below make
# that bounded surface explicit: implemented ST linears save as OT raw keys,
# while the still-missing conv/resnet/add-embedding/proj_in/proj_out/TE targets
# are listed and fail before an incomplete TE-enabled file can be written.
#
# The trainer still passes local LDM SpatialTransformer prefixes
# ("input_blocks.4.1", "middle_block.1", ...). The product LoRA save exposes the
# matching OneTrainer raw module prefixes, e.g.
#   lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q
# and save_lora_onetrainer appends .alpha/.lora_down.weight/.lora_up.weight.
def _sdxl_lora_st_prefix(st_prefix: String) -> String:
    if st_prefix == "input_blocks.4.1":
        return "lora_unet_down_blocks_1_attentions_0"
    elif st_prefix == "input_blocks.5.1":
        return "lora_unet_down_blocks_1_attentions_1"
    elif st_prefix == "input_blocks.7.1":
        return "lora_unet_down_blocks_2_attentions_0"
    elif st_prefix == "input_blocks.8.1":
        return "lora_unet_down_blocks_2_attentions_1"
    elif st_prefix == "middle_block.1":
        return "lora_unet_mid_block_attentions_0"
    elif st_prefix == "output_blocks.0.1":
        return "lora_unet_up_blocks_0_attentions_0"
    elif st_prefix == "output_blocks.1.1":
        return "lora_unet_up_blocks_0_attentions_1"
    elif st_prefix == "output_blocks.2.1":
        return "lora_unet_up_blocks_0_attentions_2"
    elif st_prefix == "output_blocks.3.1":
        return "lora_unet_up_blocks_1_attentions_0"
    elif st_prefix == "output_blocks.4.1":
        return "lora_unet_up_blocks_1_attentions_1"
    elif st_prefix == "output_blocks.5.1":
        return "lora_unet_up_blocks_1_attentions_2"
    return st_prefix


def _slot_suffix(slot: Int) -> String:
    if slot == SLOT_A1_Q:
        return "attn1_to_q"
    elif slot == SLOT_A1_K:
        return "attn1_to_k"
    elif slot == SLOT_A1_V:
        return "attn1_to_v"
    elif slot == SLOT_A1_O:
        return "attn1_to_out_0"
    elif slot == SLOT_A2_Q:
        return "attn2_to_q"
    elif slot == SLOT_A2_K:
        return "attn2_to_k"
    elif slot == SLOT_A2_V:
        return "attn2_to_v"
    elif slot == SLOT_A2_O:
        return "attn2_to_out_0"
    elif slot == SLOT_FF_PROJ:
        return "ff_net_0_proj"
    return "ff_net_2"


def _sdxl_lora_prefix(st_prefix: String, block_idx: Int, slot: Int) -> String:
    return (
        _sdxl_lora_st_prefix(st_prefix) + "_transformer_blocks_"
        + String(block_idx) + "_" + _slot_suffix(slot)
    )


def sdxl_lora_prefixes(st_prefix: String, num_blocks: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_blocks):
        for s in range(SDXL_SLOTS):
            out.append(_sdxl_lora_prefix(st_prefix, bi, s))
    return out^


def sdxl_lora_supported_unet_prefixes(st_prefix: String, num_blocks: Int) -> List[String]:
    """Implemented OneTrainer SDXL UNet LoRA module prefixes.

    This is intentionally identical to `sdxl_lora_prefixes`, but named for the
    contract checker/smokes so the supported OT save surface is source-visible.
    """
    return sdxl_lora_prefixes(st_prefix, num_blocks)


def sdxl_lora_unsupported_onetrainer_targets() -> List[String]:
    """OneTrainer SDXL LoRA targets not yet implemented by this Mojo carrier."""
    var out = List[String]()
    out.append("lora_unet_conv_in")
    out.append("lora_unet_time_embedding_linear_1")
    out.append("lora_unet_time_embedding_linear_2")
    out.append("lora_unet_add_embedding_linear_1")
    out.append("lora_unet_add_embedding_linear_2")
    out.append("lora_unet_*_resnets_*_time_emb_proj")
    out.append("lora_unet_*_resnets_*_conv1")
    out.append("lora_unet_*_resnets_*_conv2")
    out.append("lora_unet_*_resnets_*_conv_shortcut")
    out.append("lora_unet_*_samplers_*_conv")
    # NOTE: lora_unet_*_attentions_*_proj_in / proj_out are now IMPLEMENTED as
    # ST-level linear LoRA in SdxlLoraSet.proj_in_ad/proj_out_ad (math gated by
    # lora_stack_parity.mojo). They are no longer listed here.
    out.append("lora_unet_conv_norm_out")
    out.append("lora_unet_conv_out")
    out.append("lora_te1")
    out.append("lora_te2")
    return out^


def sdxl_lora_requires_text_encoder_surface(enable_te1: Bool, enable_te2: Bool) raises:
    if enable_te1 or enable_te2:
        raise Error(
            "SDXL OneTrainer LoRA save requested text-encoder adapters "
            + "(lora_te1/lora_te2), but the Mojo SDXL LoRA carrier currently "
            + "implements only lora_unet SpatialTransformer linears"
        )


def _sdxl_named_loras(set: SdxlLoraSet, st_prefix: String) -> List[NamedLora]:
    var named = List[NamedLora]()
    for bi in range(set.num_blocks):
        for s in range(SDXL_SLOTS):
            named.append(NamedLora(
                _sdxl_lora_prefix(st_prefix, bi, s),
                set.ad[bi * SDXL_SLOTS + s].copy(),
            ))
    return named^


# -- SAVE every adapter as a OneTrainer raw-keyed safetensors -----------------
def save_sdxl_lora(
    set: SdxlLoraSet, st_prefix: String, path: String, ctx: DeviceContext
) raises -> Int:
    return save_lora_onetrainer(_sdxl_named_loras(set, st_prefix), path, ctx)


def save_sdxl_lora_with_text_encoder_flags(
    set: SdxlLoraSet, st_prefix: String, path: String, ctx: DeviceContext,
    enable_te1: Bool, enable_te2: Bool,
) raises -> Int:
    sdxl_lora_requires_text_encoder_surface(enable_te1, enable_te2)
    return save_sdxl_lora(set, st_prefix, path, ctx)


def save_sdxl_lora_state(
    set: SdxlLoraSet, st_prefix: String, path: String, ctx: DeviceContext
) raises -> Int:
    return save_lora_train_state(_sdxl_named_loras(set, st_prefix), path, ctx)


# -- RESUME: load raw or PEFT A/B back through the shared resume helper --------
def load_sdxl_lora_resume(
    st_prefix: String, num_blocks: Int, rank: Int, alpha: Float32,
    C: Int, Cctx: Int, Cff: Int,
    path: String, ctx: DeviceContext,
) raises -> SdxlLoraSet:
    var prefixes = sdxl_lora_prefixes(st_prefix, num_blocks)
    var scale = alpha / Float32(rank)
    var named = load_lora_for_resume(prefixes, scale, path, ctx)   # flat order
    var ad = List[LoraAdapter]()
    for i in range(num_blocks * SDXL_SLOTS):
        ad.append(named[i].adapter.copy())
    return SdxlLoraSet(ad^, num_blocks, rank)


def load_sdxl_lora_state(
    st_prefix: String, num_blocks: Int, rank: Int, alpha: Float32,
    C: Int, Cctx: Int, Cff: Int,
    path: String, ctx: DeviceContext,
) raises -> SdxlLoraSet:
    var prefixes = sdxl_lora_prefixes(st_prefix, num_blocks)
    var scale = alpha / Float32(rank)
    var named = load_lora_train_state(prefixes, scale, path, ctx)
    var ad = List[LoraAdapter]()
    for i in range(num_blocks * SDXL_SLOTS):
        ad.append(named[i].adapter.copy())
    return SdxlLoraSet(ad^, num_blocks, rank)
