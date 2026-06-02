# serenitymojo/models/flux/lora_block.mojo
#
# LoRA-ON-PROJECTION for the Flux (flux1-dev) DOUBLE + SINGLE blocks. Mirrors the
# PROVEN Ernie/Klein LoRA template (models/ernie/lora_block.mojo,
# models/klein/lora_block.mojo) specialized to FLUX's OneTrainer target set
# (verified line-by-line against /home/alex/OneTrainer/modules/util/convert/lora/
# convert_flux_lora.py:6-41):
#
#   DOUBLE block, per stream s in {img, txt} — 6 trained projections each:
#     img_attn.qkv.0/.1/.2  -> to_q / to_k / to_v   (the 3 D-slices of wqkv [3D,D])
#     img_attn.proj         -> proj                  (wproj [D,D])
#     img_mlp.0             -> mlp0                  (wmlp0 [Fmlp,D])
#     img_mlp.2             -> mlp2                  (wmlp2 [D,Fmlp])
#     (txt_* mirror; img_mod.lin/txt_mod.lin are STACK-level base linears, wired
#      at the stack layer, NOT here — same scope as Klein/Ernie which keep the
#      modulation/embedder linears frozen.)
#   SINGLE block — 5 trained projections:
#     linear1.0/.1/.2  -> to_q / to_k / to_v   (the 3 D-slices of w1's first 3D rows)
#     linear1.3        -> proj_mlp              (the Fmlp-slice of w1's rows)
#     linear2          -> linear2               (w2 [D, D+Fmlp])
#     (modulation.lin is STACK-level, wired at the stack layer.)
#
# WHY 3 SEPARATE q/k/v ADAPTERS (not one fused qkv adapter like Klein): OT/
# diffusers train to_q/to_k/to_v as SEPARATE Linears (convert map emits distinct
# keys img_attn.qkv.0/.1/.2). A rank-r adapter on a fused [3D,D] weight is NOT
# the same low-rank family as three independent rank-r adapters on the 3 D-slices.
# Modelling them separately is the OT-faithful recipe AND makes the saved keys
# round-trip with OT-trained / ai-toolkit LoRAs.
#
# WHY THE TRAINER MATH IS THE AUTHORITY (identical to Ernie/Klein lora_block):
#   For a projection y = linear(x, W) (W [out,in]), LoRA-adapted output is
#       y' = linear(x, W) + scale*((x @ Aᵀ) @ Bᵀ)
#   A=[rank,in], B=[out,rank], scale = alpha/rank.
#   BACKWARD (given d_y' at the projection output, x the saved projection input):
#       d_dy = scale*d_y' ; d_B = d_dyᵀ@t (t=x@Aᵀ) ; d_t = d_dy@B ;
#       d_A = d_tᵀ@x ; d_x = d_t@A  (the LoRA branch's contribution to the
#       projection INPUT grad, SUMMED into the base d_x).
#   The two helpers below are byte-identical to train_step._lora_fwd / _lora_bwd
#   plus the d_x term that file drops.
#
# NO NEW ops/ PRIMITIVE: forward = two linear()s; backward = two linear_backward()s
# plus the existing slice/concat the base block already uses. Tenet 1 honored.
#
# Bit-exact base when adapters absent: each flux_lora_apply returns base_y
# unchanged when the Optional is empty, so the LoRA forward reduces to the
# verified base forward (saved activations are the LoRA-modified ones, so the
# backward recompute regenerates them identically — same checkpoint contract).
#
# Mojo 1.0.0b1: def not fn; Tensor move-only crosses API as host List[Float32].

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward

# REUSE the trainer's LoRA structs (the target authority).
from serenitymojo.training.train_step import LoraAdapter, LoraGrads

# Forward + backward ops shared with the base block (Tenet 1: nothing new here).
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, reshape_in_place, slice, concat, add,
)
from serenitymojo.ops.norm_backward import rms_norm_backward, layer_norm_backward
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import gate_residual_backward, rope_backward
from serenitymojo.ops.shape_backward import cat_backward

from serenitymojo.models.flux.block import (
    ModVecs, SingleModVecs,
    StreamWeights, DoubleBlockWeights, StreamSaved, DoubleBlockSaved,
    DoubleBlockForward, StreamGrads, DoubleBlockGrads,
    SingleBlockWeights, SingleBlockSaved, SingleBlockForward, SingleBlockGrads,
)


comptime TArc = ArcPointer[Tensor]


# ── host helpers ─────────────────────────────────────────────────────────────
def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _ones(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(1.0)
    return o^


def _zeros(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(0.0)
    return o^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


# ── LoRA fwd/bwd (host list; byte-identical to train_step._lora_fwd/_lora_bwd) ─
def flux_lora_fwd(
    x_h: List[Float32], lo: LoraAdapter, M: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var nb1 = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        nb1^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var nb2 = Optional[Tensor](None)
    var dy = linear(
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(lo.b.copy(), [lo.out_f, lo.rank], STDtype.F32, ctx),
        nb2^, ctx,
    ).to_host(ctx)                                   # [M,out]
    var out = List[Float32]()
    for i in range(len(dy)):
        out.append(lo.scale * dy[i])
    return out^


# base_y + LoRA(x) if present; else base_y unchanged (bit-exact base no-regress).
def flux_lora_apply(
    base_y: List[Float32], x_h: List[Float32], lo: Optional[LoraAdapter],
    M: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    if not lo:
        return base_y.copy()
    var contrib = flux_lora_fwd(x_h, lo.value(), M, ctx)
    var out = List[Float32]()
    for i in range(len(base_y)):
        out.append(base_y[i] + contrib[i])
    return out^


struct FluxLoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: List[Float32]   # LoRA contribution to the projection INPUT grad [M,in]

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: List[Float32]
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def flux_lora_bwd(
    d_contrib_h: List[Float32], x_h: List[Float32], lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> FluxLoraGrads:
    var nb_t = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        nb_t^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var d_dy = List[Float32]()
    for i in range(len(d_contrib_h)):
        d_dy.append(lo.scale * d_contrib_h[i])       # [M,out]
    var lbB = linear_backward(
        Tensor.from_host(d_dy^, [M, lo.out_f], STDtype.F32, ctx),
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(lo.b.copy(), [lo.out_f, lo.rank], STDtype.F32, ctx),
        M, lo.rank, lo.out_f, ctx,
    )
    var d_t = lbB.d_x.to_host(ctx)                   # [M,rank]
    var d_b = lbB.d_w.to_host(ctx)                   # [out_f,rank]
    var lbA = linear_backward(
        Tensor.from_host(d_t^, [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        M, lo.in_f, lo.rank, ctx,
    )
    var d_x_lo = lbA.d_x.to_host(ctx)                # [M,in_f]
    var d_a = lbA.d_w.to_host(ctx)                   # [rank,in_f]
    return FluxLoraGrads(d_a^, d_b^, d_x_lo^)


# take cols [c0,c0+w) from a [rows,total] row-major host list -> [rows,w].
def _take_cols(src: List[Float32], rows: Int, total: Int, c0: Int, w: Int) -> List[Float32]:
    var o = List[Float32]()
    for r in range(rows):
        var base = r * total
        for c in range(w):
            o.append(src[base + c0 + c])
    return o^


# add a [rows,w] delta into cols [c0,c0+w) of a [rows,total] buffer (others kept).
def _add_into_cols(
    dst: List[Float32], delta: List[Float32], rows: Int, total: Int, c0: Int, w: Int
) -> List[Float32]:
    var o = dst.copy()
    for r in range(rows):
        var base = r * total
        for c in range(w):
            o[base + c0 + c] = o[base + c0 + c] + delta[r * w + c]
    return o^


# ═══════════════════════════════════════════════════════════════════════════
# Per-block LoRA carriers (Optional slots; canonical slot order below).
# ═══════════════════════════════════════════════════════════════════════════
# Double-stream slot order (per stream): to_q, to_k, to_v, proj, mlp0, mlp2.
comptime DBL_STREAM_SLOTS = 6
comptime D_SQ = 0    # to_q
comptime D_SK = 1    # to_k
comptime D_SV = 2    # to_v
comptime D_PROJ = 3  # img/txt_attn.proj
comptime D_MLP0 = 4  # img/txt_mlp.0
comptime D_MLP2 = 5  # img/txt_mlp.2

# Single-block slot order: to_q, to_k, to_v, proj_mlp, linear2.
comptime SGL_SLOTS = 5
comptime S_SQ = 0
comptime S_SK = 1
comptime S_SV = 2
comptime S_PMLP = 3   # linear1.3 (proj_mlp)
comptime S_L2 = 4     # linear2


struct StreamLora(Copyable, Movable):
    var to_q: Optional[LoraAdapter]
    var to_k: Optional[LoraAdapter]
    var to_v: Optional[LoraAdapter]
    var proj: Optional[LoraAdapter]
    var mlp0: Optional[LoraAdapter]
    var mlp2: Optional[LoraAdapter]

    def __init__(
        out self,
        var to_q: Optional[LoraAdapter], var to_k: Optional[LoraAdapter],
        var to_v: Optional[LoraAdapter], var proj: Optional[LoraAdapter],
        var mlp0: Optional[LoraAdapter], var mlp2: Optional[LoraAdapter],
    ):
        self.to_q = to_q^
        self.to_k = to_k^
        self.to_v = to_v^
        self.proj = proj^
        self.mlp0 = mlp0^
        self.mlp2 = mlp2^


struct DoubleBlockLora(Copyable, Movable):
    var img: StreamLora
    var txt: StreamLora

    def __init__(out self, var img: StreamLora, var txt: StreamLora):
        self.img = img^
        self.txt = txt^


struct SingleBlockLora(Copyable, Movable):
    var to_q: Optional[LoraAdapter]
    var to_k: Optional[LoraAdapter]
    var to_v: Optional[LoraAdapter]
    var proj_mlp: Optional[LoraAdapter]
    var linear2: Optional[LoraAdapter]

    def __init__(
        out self,
        var to_q: Optional[LoraAdapter], var to_k: Optional[LoraAdapter],
        var to_v: Optional[LoraAdapter], var proj_mlp: Optional[LoraAdapter],
        var linear2: Optional[LoraAdapter],
    ):
        self.to_q = to_q^
        self.to_k = to_k^
        self.to_v = to_v^
        self.proj_mlp = proj_mlp^
        self.linear2 = linear2^


# ── per-stream / per-block LoRA grads (parallel to slots; empty if absent) ────
struct StreamLoraGrads(Copyable, Movable):
    var d_a: List[List[Float32]]   # DBL_STREAM_SLOTS entries
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


struct DoubleBlockLoraGrads(Copyable, Movable):
    var img: StreamLoraGrads
    var txt: StreamLoraGrads

    def __init__(out self, var img: StreamLoraGrads, var txt: StreamLoraGrads):
        self.img = img^
        self.txt = txt^


struct SingleBlockLoraGrads(Copyable, Movable):
    var d_a: List[List[Float32]]   # SGL_SLOTS entries
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


struct DoubleBlockLoraBackward(Movable):
    var base: DoubleBlockGrads
    var lora: DoubleBlockLoraGrads

    def __init__(out self, var base: DoubleBlockGrads, var lora: DoubleBlockLoraGrads):
        self.base = base^
        self.lora = lora^


struct SingleBlockLoraBackward(Movable):
    var base: SingleBlockGrads
    var lora: SingleBlockLoraGrads

    def __init__(out self, var base: SingleBlockGrads, var lora: SingleBlockLoraGrads):
        self.base = base^
        self.lora = lora^


# proj-backward helper: base linear_backward d_x then add the LoRA branch's d_x
# (if present), collecting d_a/d_b into the slot lists. Returns the SUMMED d_x.
struct _ProjBwd(Movable):
    var d_x: Tensor
    var d_w: Tensor
    var d_b: Tensor

    def __init__(out self, var d_x: Tensor, var d_w: Tensor, var d_b: Tensor):
        self.d_x = d_x^
        self.d_w = d_w^
        self.d_b = d_b^


def _proj_bwd_with_lora(
    d_y: Tensor, x_in: Tensor, w: Tensor, x_in_h: List[Float32],
    lo: Optional[LoraAdapter], slot: Int,
    M: Int, in_f: Int, out_f: Int,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> _ProjBwd:
    var lb = linear_backward(d_y, x_in, w, M, in_f, out_f, ctx)
    var d_w = lb.d_w.clone(ctx)
    var d_b = lb.d_b.clone(ctx)
    if lo:
        var d_y_h = d_y.to_host(ctx)
        var lg = flux_lora_bwd(d_y_h, x_in_h, lo.value(), M, ctx)
        d_a_slots[slot] = lg.d_a.copy()
        d_b_slots[slot] = lg.d_b.copy()
        var base_dx = lb.d_x.to_host(ctx)
        var summed = _add_lists(base_dx, lg.d_x)
        return _ProjBwd(_t(summed, [M, in_f], ctx), d_w^, d_b^)
    var d_x = lb.d_x.clone(ctx)
    return _ProjBwd(d_x^, d_w^, d_b^)


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block LoRA forward — mirrors double_block_forward, injecting LoRA on
# each stream's q/k/v slices + proj + mlp0 + mlp2 BEFORE the downstream op.
# ═══════════════════════════════════════════════════════════════════════════
struct _StreamPreH(Movable):
    var q_rms: Tensor
    var k_rms: Tensor
    var v: Tensor
    var ln1_h: List[Float32]
    var norm_h: List[Float32]
    var q_pre_h: List[Float32]
    var k_pre_h: List[Float32]

    def __init__(
        out self, var q_rms: Tensor, var k_rms: Tensor, var v: Tensor,
        var ln1_h: List[Float32], var norm_h: List[Float32],
        var q_pre_h: List[Float32], var k_pre_h: List[Float32],
    ):
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^
        self.ln1_h = ln1_h^
        self.norm_h = norm_h^
        self.q_pre_h = q_pre_h^
        self.k_pre_h = k_pre_h^


def _stream_pre_lora[
    H: Int, Dh: Int
](
    x: Tensor, w: StreamWeights, mv: ModVecs, lo: StreamLora,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPreH:
    var ln1 = layer_norm(x, ones, zeros, eps, ctx)
    var norm = modulate(ln1, _t(mv.scale1.copy(), [D], ctx), _t(mv.shift1.copy(), [D], ctx), ctx)
    var norm_h = norm.to_host(ctx)                          # [N,D] LoRA input for q/k/v
    var b = Optional[Tensor](w.bqkv[].clone(ctx))
    var qkv = linear(norm, w.wqkv[], b, ctx)                # [N,3D]
    # base q/k/v slices
    var q_base = slice(qkv, 1, 0, D, ctx).to_host(ctx)
    var k_base = slice(qkv, 1, D, D, ctx).to_host(ctx)
    var v_base = slice(qkv, 1, 2 * D, D, ctx).to_host(ctx)
    # LoRA on each (separate to_q/to_k/to_v adapters; shared norm input)
    var q_h = flux_lora_apply(q_base, norm_h, lo.to_q, N, ctx)
    var k_h = flux_lora_apply(k_base, norm_h, lo.to_k, N, ctx)
    var v_h = flux_lora_apply(v_base, norm_h, lo.to_v, N, ctx)
    var q_pre = reshape_owned(_t(q_h, [N, D], ctx)^, [1, N, H, Dh])
    var k_pre = reshape_owned(_t(k_h, [N, D], ctx)^, [1, N, H, Dh])
    var v = reshape_owned(_t(v_h, [N, D], ctx)^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPreH(
        q_rms^, k_rms^, v^, ln1.to_host(ctx), norm_h.copy(),
        q_pre.to_host(ctx), k_pre.to_host(ctx),
    )


struct _StreamPostH(Movable):
    var out: Tensor
    var attn_res_h: List[Float32]
    var ln2_h: List[Float32]
    var mlp_in_h: List[Float32]
    var mlp_pre_h: List[Float32]
    var mlp_h_h: List[Float32]

    def __init__(
        out self, var out: Tensor, var attn_res_h: List[Float32],
        var ln2_h: List[Float32], var mlp_in_h: List[Float32],
        var mlp_pre_h: List[Float32], var mlp_h_h: List[Float32],
    ):
        self.out = out^
        self.attn_res_h = attn_res_h^
        self.ln2_h = ln2_h^
        self.mlp_in_h = mlp_in_h^
        self.mlp_pre_h = mlp_pre_h^
        self.mlp_h_h = mlp_h_h^


def _stream_post_lora(
    x: Tensor, att: Tensor, att_h: List[Float32],
    w: StreamWeights, mv: ModVecs, lo: StreamLora,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor, zeros: Tensor,
    ctx: DeviceContext,
) raises -> _StreamPostH:
    var bp = Optional[Tensor](w.bproj[].clone(ctx))
    var out_base = linear(att, w.wproj[], bp, ctx).to_host(ctx)   # [N,D]
    var out_h = flux_lora_apply(out_base, att_h, lo.proj, N, ctx)
    var out = _t(out_h, [N, D], ctx)
    var attn_res = residual_gate(x, _t(mv.gate1.copy(), [D], ctx), out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(ln2, _t(mv.scale2.copy(), [D], ctx), _t(mv.shift2.copy(), [D], ctx), ctx)
    var mlp_in_h = mlp_in.to_host(ctx)                           # [N,D] LoRA input mlp0
    var b0 = Optional[Tensor](w.bmlp0[].clone(ctx))
    var mlp_pre_base = linear(mlp_in, w.wmlp0[], b0, ctx).to_host(ctx)   # [N,Fmlp]
    var mlp_pre_h = flux_lora_apply(mlp_pre_base, mlp_in_h, lo.mlp0, N, ctx)
    var mlp_pre = _t(mlp_pre_h, [N, Fmlp], ctx)
    var mlp_h = gelu(mlp_pre, ctx)                               # [N,Fmlp]
    var mlp_h_h = mlp_h.to_host(ctx)                             # LoRA input mlp2
    var b2 = Optional[Tensor](w.bmlp2[].clone(ctx))
    var mlp_base = linear(mlp_h, w.wmlp2[], b2, ctx).to_host(ctx)   # [N,D]
    var mlp_out_h = flux_lora_apply(mlp_base, mlp_h_h, lo.mlp2, N, ctx)
    var mlp = _t(mlp_out_h, [N, D], ctx)
    var final = residual_gate(attn_res, _t(mv.gate2.copy(), [D], ctx), mlp, ctx)
    return _StreamPostH(
        final^, attn_res.to_host(ctx), ln2.to_host(ctx),
        mlp_in_h^, mlp_pre_h^, mlp_h_h^,
    )


def double_block_lora_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img: List[Float32], txt: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs, lora: DoubleBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)

    var img_x = _t(img, [N_IMG, D], ctx)
    var txt_x = _t(txt, [N_TXT, D], ctx)

    var ip = _stream_pre_lora[H, Dh](img_x, w.img, img_mod, lora.img, N_IMG, D, eps, ones_t, zeros_t, ctx)
    var tp = _stream_pre_lora[H, Dh](txt_x, w.txt, txt_mod, lora.txt, N_TXT, D, eps, ones_t, zeros_t, ctx)

    var q = concat(1, ctx, tp.q_rms, ip.q_rms)   # [1,S,H,Dh] txt FIRST
    var k = concat(1, ctx, tp.k_rms, ip.k_rms)
    var v = concat(1, ctx, tp.v, ip.v)

    var q_rope = rope_interleaved(q, cos, sin, ctx)
    var k_rope = rope_interleaved(k, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = reshape_owned(txt_att_4d^, [N_TXT, D])
    var img_att = reshape_owned(img_att_4d^, [N_IMG, D])
    var img_att_h = img_att.to_host(ctx)
    var txt_att_h = txt_att.to_host(ctx)

    var ipost = _stream_post_lora(img_x, img_att, img_att_h, w.img, img_mod, lora.img, N_IMG, D, Fmlp, eps, ones_t, zeros_t, ctx)
    var tpost = _stream_post_lora(txt_x, txt_att, txt_att_h, w.txt, txt_mod, lora.txt, N_TXT, D, Fmlp, eps, ones_t, zeros_t, ctx)

    var img_saved = StreamSaved(
        img.copy(), ip.ln1_h.copy(), ip.norm_h.copy(),
        ip.q_pre_h.copy(), ip.k_pre_h.copy(),
        img_att_h.copy(), ipost.attn_res_h.copy(),
        ipost.ln2_h.copy(), ipost.mlp_in_h.copy(),
        ipost.mlp_pre_h.copy(), ipost.mlp_h_h.copy(),
    )
    var txt_saved = StreamSaved(
        txt.copy(), tp.ln1_h.copy(), tp.norm_h.copy(),
        tp.q_pre_h.copy(), tp.k_pre_h.copy(),
        txt_att_h.copy(), tpost.attn_res_h.copy(),
        tpost.ln2_h.copy(), tpost.mlp_in_h.copy(),
        tpost.mlp_pre_h.copy(), tpost.mlp_h_h.copy(),
    )
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^,
        q_rope.to_host(ctx), k_rope.to_host(ctx), v.to_host(ctx),
    )

    var img_out = ipost.out.to_host(ctx)
    var txt_out = tpost.out.to_host(ctx)
    return DoubleBlockForward(img_out^, txt_out^, saved^)


# ── per-stream post backward (LoRA-aware: proj, mlp0, mlp2) ──────────────────
struct _StreamPostBackL(Movable):
    var d_x: List[Float32]
    var d_att: List[Float32]
    var d_wproj: List[Float32]
    var d_bproj: List[Float32]
    var d_wmlp0: List[Float32]
    var d_bmlp0: List[Float32]
    var d_wmlp2: List[Float32]
    var d_bmlp2: List[Float32]
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]

    def __init__(
        out self, var d_x: List[Float32], var d_att: List[Float32],
        var d_wproj: List[Float32], var d_bproj: List[Float32],
        var d_wmlp0: List[Float32], var d_bmlp0: List[Float32],
        var d_wmlp2: List[Float32], var d_bmlp2: List[Float32],
        var d_gate1: List[Float32],
        var d_shift2: List[Float32], var d_scale2: List[Float32], var d_gate2: List[Float32],
    ):
        self.d_x = d_x^
        self.d_att = d_att^
        self.d_wproj = d_wproj^
        self.d_bproj = d_bproj^
        self.d_wmlp0 = d_wmlp0^
        self.d_bmlp0 = d_bmlp0^
        self.d_wmlp2 = d_wmlp2^
        self.d_bmlp2 = d_bmlp2^
        self.d_gate1 = d_gate1^
        self.d_shift2 = d_shift2^
        self.d_scale2 = d_scale2^
        self.d_gate2 = d_gate2^


def _stream_post_backward_lora(
    d_out: Tensor, x: Tensor, att: Tensor,
    w: StreamWeights, mv: ModVecs, sv: StreamSaved, lo: StreamLora,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> _StreamPostBackL:
    var attn_res_t = _t(sv.attn_res.copy(), [N, D], ctx)
    var mlp_h_t = _t(sv.mlp_h.copy(), [N, Fmlp], ctx)
    # recompute mlp output WITH LoRA(mlp2) so gate_residual_backward y matches fwd.
    var b2 = Optional[Tensor](w.bmlp2[].clone(ctx))
    var mlp_base = linear(mlp_h_t, w.wmlp2[], b2, ctx).to_host(ctx)
    var mlp_y_h = flux_lora_apply(mlp_base, sv.mlp_h.copy(), lo.mlp2, N, ctx)
    var mlp_y = _t(mlp_y_h, [N, D], ctx)
    var grg2 = gate_residual_backward(d_out, attn_res_t, _t(mv.gate2.copy(), [D], ctx), mlp_y, ctx)
    var d_gate2 = grg2.d_g.to_host(ctx)

    # mlp = linear(mlp_h, Wmlp2)[+LoRA(mlp2)]  W [D, Fmlp]
    var pm2 = _proj_bwd_with_lora(
        grg2.d_y, mlp_h_t, w.wmlp2[], sv.mlp_h.copy(), lo.mlp2, D_MLP2, N, Fmlp, D,
        d_a_slots, d_b_slots, ctx,
    )
    var d_wmlp2 = pm2.d_w.to_host(ctx)
    var d_bmlp2 = pm2.d_b.to_host(ctx)

    # mlp_h = gelu(mlp_pre)
    var mlp_pre_t = _t(sv.mlp_pre.copy(), [N, Fmlp], ctx)
    var d_mlp_pre = gelu_backward(pm2.d_x, mlp_pre_t, ctx)

    # mlp_pre = linear(mlp_in, Wmlp0)[+LoRA(mlp0)]  W [Fmlp, D]
    var mlp_in_t = _t(sv.mlp_in.copy(), [N, D], ctx)
    var pm0 = _proj_bwd_with_lora(
        d_mlp_pre, mlp_in_t, w.wmlp0[], sv.mlp_in.copy(), lo.mlp0, D_MLP0, N, D, Fmlp,
        d_a_slots, d_b_slots, ctx,
    )
    var d_wmlp0 = pm0.d_w.to_host(ctx)
    var d_bmlp0 = pm0.d_b.to_host(ctx)

    # mlp_in = modulate(ln2, scale2, shift2)
    var ln2_t = _t(sv.ln2.copy(), [N, D], ctx)
    var mb2 = modulate_backward(pm0.d_x, ln2_t, _t(mv.scale2.copy(), [D], ctx), ctx)
    var d_scale2 = mb2.d_scale.to_host(ctx)
    var d_shift2 = mb2.d_shift.to_host(ctx)

    var lnb2 = layer_norm_backward(mb2.d_x, attn_res_t, ones, eps, ctx)
    var d_attn_res_total = add(grg2.d_x, lnb2.d_x, ctx)

    # attn_res = residual_gate(x, gate1, proj_out): recompute proj WITH LoRA(proj)
    var bp = Optional[Tensor](w.bproj[].clone(ctx))
    var proj_base = linear(att, w.wproj[], bp, ctx).to_host(ctx)
    var att_h = att.to_host(ctx)
    var proj_y_h = flux_lora_apply(proj_base, att_h, lo.proj, N, ctx)
    var proj_out = _t(proj_y_h, [N, D], ctx)
    var grg1 = gate_residual_backward(d_attn_res_total, x, _t(mv.gate1.copy(), [D], ctx), proj_out, ctx)
    var d_gate1 = grg1.d_g.to_host(ctx)
    var d_x_res = grg1.d_x.to_host(ctx)

    # proj_out = linear(att, Wproj)[+LoRA(proj)]  W [D, D]
    var pproj = _proj_bwd_with_lora(
        grg1.d_y, att, w.wproj[], att_h, lo.proj, D_PROJ, N, D, D,
        d_a_slots, d_b_slots, ctx,
    )
    var d_wproj = pproj.d_w.to_host(ctx)
    var d_bproj = pproj.d_b.to_host(ctx)
    var d_att = pproj.d_x.to_host(ctx)

    return _StreamPostBackL(
        d_x_res^, d_att^, d_wproj^, d_bproj^,
        d_wmlp0^, d_bmlp0^, d_wmlp2^, d_bmlp2^,
        d_gate1^, d_shift2^, d_scale2^, d_gate2^,
    )


# ── per-stream pre backward (LoRA-aware: to_q, to_k, to_v) ───────────────────
struct _StreamPreBackL(Movable):
    var d_x: List[Float32]
    var d_wqkv: List[Float32]
    var d_bqkv: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]

    def __init__(
        out self, var d_x: List[Float32],
        var d_wqkv: List[Float32], var d_bqkv: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_shift1: List[Float32], var d_scale1: List[Float32],
    ):
        self.d_x = d_x^
        self.d_wqkv = d_wqkv^
        self.d_bqkv = d_bqkv^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_shift1 = d_shift1^
        self.d_scale1 = d_scale1^


def _stream_pre_backward_lora[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: ModVecs, sv: StreamSaved, lo: StreamLora,
    N: Int, D: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> _StreamPreBackL:
    var q_pre_t = _t(sv.q_pre.copy(), [1, N, H, Dh], ctx)
    var k_pre_t = _t(sv.k_pre.copy(), [1, N, H, Dh], ctx)
    var rb_q = rms_norm_backward(d_q_rms, q_pre_t, w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, k_pre_t, w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [N, D])
    reshape_in_place(rb_k.d_x, [N, D])
    var d_v_flat = reshape(d_v, [N, D], ctx)

    # The fused qkv linear's base d_w/d_b come from the joined d_qkv [N,3D]; the
    # LoRA on to_q/to_k/to_v consumes the per-slice d_y (rb_q.d_x / rb_k.d_x /
    # d_v_flat) against the SHARED input `norm`. d_x_lo from all three slices SUMS
    # into the base norm grad (LoRA contribution to the projection input).
    var norm_t = _t(sv.norm.copy(), [N, D], ctx)
    var d_qkv = concat(1, ctx, rb_q.d_x, rb_k.d_x, d_v_flat)   # [N,3D]
    var lb_qkv = linear_backward(d_qkv, norm_t, w.wqkv[], N, D, 3 * D, ctx)
    var d_wqkv = lb_qkv.d_w.to_host(ctx)
    var d_bqkv = lb_qkv.d_b.to_host(ctx)
    var d_norm = lb_qkv.d_x.to_host(ctx)   # base norm grad [N,D]

    # to_q / to_k / to_v LoRA: each consumes its own d_y slice, input = norm.
    var d_q_h = rb_q.d_x.to_host(ctx)
    var d_k_h = rb_k.d_x.to_host(ctx)
    var d_v_h = d_v_flat.to_host(ctx)
    if lo.to_q:
        var lg = flux_lora_bwd(d_q_h, sv.norm.copy(), lo.to_q.value(), N, ctx)
        d_a_slots[D_SQ] = lg.d_a.copy()
        d_b_slots[D_SQ] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)
    if lo.to_k:
        var lg = flux_lora_bwd(d_k_h, sv.norm.copy(), lo.to_k.value(), N, ctx)
        d_a_slots[D_SK] = lg.d_a.copy()
        d_b_slots[D_SK] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)
    if lo.to_v:
        var lg = flux_lora_bwd(d_v_h, sv.norm.copy(), lo.to_v.value(), N, ctx)
        d_a_slots[D_SV] = lg.d_a.copy()
        d_b_slots[D_SV] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)

    # norm = modulate(ln1, scale1, shift1)
    var ln1_t = _t(sv.ln1.copy(), [N, D], ctx)
    var mb1 = modulate_backward(_t(d_norm, [N, D], ctx), ln1_t, _t(mv.scale1.copy(), [D], ctx), ctx)
    var d_scale1 = mb1.d_scale.to_host(ctx)
    var d_shift1 = mb1.d_shift.to_host(ctx)

    var x_t = _t(sv.x.copy(), [N, D], ctx)
    var lnb1 = layer_norm_backward(mb1.d_x, x_t, ones, eps, ctx)
    var d_x_norm = lnb1.d_x.to_host(ctx)
    return _StreamPreBackL(d_x_norm^, d_wqkv^, d_bqkv^, d_q_norm^, d_k_norm^, d_shift1^, d_scale1^)


def double_block_lora_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_img_out: List[Float32], d_txt_out: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs, lora: DoubleBlockLora,
    saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockLoraBackward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)

    var d_io_t = _t(d_img_out, [N_IMG, D], ctx)
    var d_to_t = _t(d_txt_out, [N_TXT, D], ctx)

    var img_x = _t(saved.img.x.copy(), [N_IMG, D], ctx)
    var txt_x = _t(saved.txt.x.copy(), [N_TXT, D], ctx)
    var img_att = _t(saved.img.att.copy(), [N_IMG, D], ctx)
    var txt_att = _t(saved.txt.att.copy(), [N_TXT, D], ctx)

    # slot lists per stream
    var ia = List[List[Float32]]()
    var ib = List[List[Float32]]()
    var ta = List[List[Float32]]()
    var tb = List[List[Float32]]()
    for _ in range(DBL_STREAM_SLOTS):
        ia.append(List[Float32]()); ib.append(List[Float32]())
        ta.append(List[Float32]()); tb.append(List[Float32]())

    var ipb = _stream_post_backward_lora(
        d_io_t, img_x, img_att, w.img, img_mod, saved.img, lora.img,
        N_IMG, D, Fmlp, eps, ones_t, ia, ib, ctx,
    )
    var tpb = _stream_post_backward_lora(
        d_to_t, txt_x, txt_att, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT, D, Fmlp, eps, ones_t, ta, tb, ctx,
    )

    # join per-stream attention-slice grads into joint d_att (txt FIRST)
    var d_tatt_4d = _t(tpb.d_att.copy(), [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = _t(ipb.d_att.copy(), [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)   # [1,S,H,Dh]

    var q_rope_t = _t(saved.q_rope.copy(), [1, S, H, Dh], ctx)
    var k_rope_t = _t(saved.k_rope.copy(), [1, S, H, Dh], ctx)
    var v_joint_t = _t(saved.v_joint.copy(), [1, S, H, Dh], ctx)
    var sb = sdpa_backward[1, S, H, Dh](q_rope_t, k_rope_t, v_joint_t, d_att_joint, scale, ctx)

    var d_q_joint = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_joint = rope_backward(sb.d_k, cos, sin, True, ctx)

    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(sb.d_v, N_TXT, N_IMG, 1, ctx)

    var iprb = _stream_pre_backward_lora[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, saved.img, lora.img,
        N_IMG, D, eps, ones_t, ia, ib, ctx,
    )
    var tprb = _stream_pre_backward_lora[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT, D, eps, ones_t, ta, tb, ctx,
    )

    var d_img_x = _add_lists(ipb.d_x, iprb.d_x)
    var d_txt_x = _add_lists(tpb.d_x, tprb.d_x)

    var img_grads = StreamGrads(
        d_img_x^,
        iprb.d_wqkv.copy(), iprb.d_bqkv.copy(),
        ipb.d_wproj.copy(), ipb.d_bproj.copy(),
        ipb.d_wmlp0.copy(), ipb.d_bmlp0.copy(),
        ipb.d_wmlp2.copy(), ipb.d_bmlp2.copy(),
        iprb.d_q_norm.copy(), iprb.d_k_norm.copy(),
        iprb.d_shift1.copy(), iprb.d_scale1.copy(), ipb.d_gate1.copy(),
        ipb.d_shift2.copy(), ipb.d_scale2.copy(), ipb.d_gate2.copy(),
    )
    var txt_grads = StreamGrads(
        d_txt_x^,
        tprb.d_wqkv.copy(), tprb.d_bqkv.copy(),
        tpb.d_wproj.copy(), tpb.d_bproj.copy(),
        tpb.d_wmlp0.copy(), tpb.d_bmlp0.copy(),
        tpb.d_wmlp2.copy(), tpb.d_bmlp2.copy(),
        tprb.d_q_norm.copy(), tprb.d_k_norm.copy(),
        tprb.d_shift1.copy(), tprb.d_scale1.copy(), tpb.d_gate1.copy(),
        tpb.d_shift2.copy(), tpb.d_scale2.copy(), tpb.d_gate2.copy(),
    )
    return DoubleBlockLoraBackward(
        DoubleBlockGrads(img_grads^, txt_grads^),
        DoubleBlockLoraGrads(StreamLoraGrads(ia^, ib^), StreamLoraGrads(ta^, tb^)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE block LoRA forward/backward.
#   fused = linear(norm, W1, b1)   [S, 3D+Fmlp]
#   qkv = fused[:, :3D] ; mlp_in = fused[:, 3D:3D+Fmlp]
#   LoRA on to_q/to_k/to_v (3 D-slices of qkv) + proj_mlp (the Fmlp slice).
#   out = linear(out_in, W2, b2) ; LoRA on linear2 (input = out_in [S,D+Fmlp]).
# ═══════════════════════════════════════════════════════════════════════════
def single_block_lora_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs, lora: SingleBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)

    var x_t = _t(x, [S, D], ctx)
    var ln_t = layer_norm(x_t, ones_t, zeros_t, eps, ctx)
    var norm_t = modulate(ln_t, _t(mv.scale.copy(), [D], ctx), _t(mv.shift.copy(), [D], ctx), ctx)
    var norm_h = norm_t.to_host(ctx)                       # [S,D] LoRA input

    var b1 = Optional[Tensor](w.b1[].clone(ctx))
    var fused = linear(norm_t, w.w1[], b1, ctx)            # [S, 3D+Fmlp]

    var q_base = slice(fused, 1, 0, D, ctx).to_host(ctx)
    var k_base = slice(fused, 1, D, D, ctx).to_host(ctx)
    var v_base = slice(fused, 1, 2 * D, D, ctx).to_host(ctx)
    var mlp_base = slice(fused, 1, 3 * D, Fmlp, ctx).to_host(ctx)   # [S,Fmlp]

    var q_h = flux_lora_apply(q_base, norm_h, lora.to_q, S, ctx)
    var k_h = flux_lora_apply(k_base, norm_h, lora.to_k, S, ctx)
    var v_h = flux_lora_apply(v_base, norm_h, lora.to_v, S, ctx)
    var mlp_in_h = flux_lora_apply(mlp_base, norm_h, lora.proj_mlp, S, ctx)

    var q_pre = reshape_owned(_t(q_h, [S, D], ctx)^, [1, S, H, Dh])
    var k_pre = reshape_owned(_t(k_h, [S, D], ctx)^, [1, S, H, Dh])
    var v = reshape_owned(_t(v_h, [S, D], ctx)^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var mlp_in = _t(mlp_in_h, [S, Fmlp], ctx)
    var mlp_h = gelu(mlp_in, ctx)                          # [S,Fmlp]

    var out_in = concat(1, ctx, att_flat, mlp_h)           # [S, D+Fmlp]
    var out_in_h = out_in.to_host(ctx)                     # LoRA input for linear2

    var b2 = Optional[Tensor](w.b2[].clone(ctx))
    var out_base = linear(out_in, w.w2[], b2, ctx).to_host(ctx)   # [S,D]
    var out_h = flux_lora_apply(out_base, out_in_h, lora.linear2, S, ctx)
    var out_proj = _t(out_h, [S, D], ctx)

    var result = residual_gate(x_t, _t(mv.gate.copy(), [D], ctx), out_proj, ctx)

    var saved = SingleBlockSaved(
        x.copy(), ln_t.to_host(ctx), norm_h.copy(),
        q_pre.to_host(ctx), k_pre.to_host(ctx),
        q_rope.to_host(ctx), k_rope.to_host(ctx), v.to_host(ctx),
        att_flat.to_host(ctx),
        mlp_in_h.copy(), mlp_h.to_host(ctx), out_in_h.copy(),
    )
    return SingleBlockForward(result.to_host(ctx), saved^)


def single_block_lora_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs, lora: SingleBlockLora, saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockLoraBackward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var scale_t = _t(mv.scale.copy(), [D], ctx)
    var gate_t = _t(mv.gate.copy(), [D], ctx)

    var d_out_t = _t(d_out, [S, D], ctx)
    var x_t = _t(saved.x.copy(), [S, D], ctx)
    var out_in_t = _t(saved.out_in.copy(), [S, D + Fmlp], ctx)

    var d_a_slots = List[List[Float32]]()
    var d_b_slots = List[List[Float32]]()
    for _ in range(SGL_SLOTS):
        d_a_slots.append(List[Float32]()); d_b_slots.append(List[Float32]())

    # result = residual_gate(x, gate, out): recompute out WITH LoRA(linear2)
    var b2 = Optional[Tensor](w.b2[].clone(ctx))
    var out_base = linear(out_in_t, w.w2[], b2, ctx).to_host(ctx)
    var out_y_h = flux_lora_apply(out_base, saved.out_in.copy(), lora.linear2, S, ctx)
    var out_y = _t(out_y_h, [S, D], ctx)
    var grg = gate_residual_backward(d_out_t, x_t, gate_t, out_y, ctx)
    var d_gate = grg.d_g.to_host(ctx)

    # out = linear(out_in, W2)[+LoRA(linear2)]  W2 [D, D+Fmlp]
    var pl2 = _proj_bwd_with_lora(
        grg.d_y, out_in_t, w.w2[], saved.out_in.copy(), lora.linear2, S_L2,
        S, D + Fmlp, D, d_a_slots, d_b_slots, ctx,
    )
    var d_w2 = pl2.d_w.to_host(ctx)
    var d_b2 = pl2.d_b.to_host(ctx)

    # out_in = concat(att_flat, mlp_h) on channel axis (sizes D, Fmlp)
    reshape_in_place(pl2.d_x, [1, S, D + Fmlp])
    var cb = cat_backward(pl2.d_x, D, Fmlp, 2, ctx)
    reshape_in_place(cb.d_0, [1, S, H, Dh])
    reshape_in_place(cb.d_1, [S, Fmlp])

    # mlp_h = gelu(mlp_in)
    var mlp_in_t = _t(saved.mlp_in.copy(), [S, Fmlp], ctx)
    var d_mlp_in = gelu_backward(cb.d_1, mlp_in_t, ctx)   # [S,Fmlp]

    # attention branch
    var q_rope_t = _t(saved.q_rope.copy(), [1, S, H, Dh], ctx)
    var k_rope_t = _t(saved.k_rope.copy(), [1, S, H, Dh], ctx)
    var v_t = _t(saved.v.copy(), [1, S, H, Dh], ctx)
    var sb = sdpa_backward[1, S, H, Dh](q_rope_t, k_rope_t, v_t, cb.d_0, scale, ctx)

    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var q_pre_t = _t(saved.q_pre.copy(), [1, S, H, Dh], ctx)
    var k_pre_t = _t(saved.k_pre.copy(), [1, S, H, Dh], ctx)
    var rb_q = rms_norm_backward(d_q_rms, q_pre_t, w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, k_pre_t, w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    # join the per-slice d_y into d_fused [S, 3D+Fmlp]
    var d_qkv = concat(1, ctx, rb_q.d_x, rb_k.d_x, sb.d_v)   # [S,3D]
    var d_fused = concat(1, ctx, d_qkv, d_mlp_in)            # [S,3D+Fmlp]

    # fused = linear(norm, W1, b1)
    var norm_t = _t(saved.norm.copy(), [S, D], ctx)
    var lb_w1 = linear_backward(d_fused, norm_t, w.w1[], S, D, 3 * D + Fmlp, ctx)
    var d_w1 = lb_w1.d_w.to_host(ctx)
    var d_b1 = lb_w1.d_b.to_host(ctx)
    var d_norm = lb_w1.d_x.to_host(ctx)   # base norm grad [S,D]

    # LoRA on to_q/to_k/to_v (input = norm, d_y = per-slice grads) + proj_mlp.
    var d_q_h = rb_q.d_x.to_host(ctx)
    var d_k_h = rb_k.d_x.to_host(ctx)
    var d_v_h = sb.d_v.to_host(ctx)
    var d_mlp_in_h = d_mlp_in.to_host(ctx)
    if lora.to_q:
        var lg = flux_lora_bwd(d_q_h, saved.norm.copy(), lora.to_q.value(), S, ctx)
        d_a_slots[S_SQ] = lg.d_a.copy(); d_b_slots[S_SQ] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)
    if lora.to_k:
        var lg = flux_lora_bwd(d_k_h, saved.norm.copy(), lora.to_k.value(), S, ctx)
        d_a_slots[S_SK] = lg.d_a.copy(); d_b_slots[S_SK] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)
    if lora.to_v:
        var lg = flux_lora_bwd(d_v_h, saved.norm.copy(), lora.to_v.value(), S, ctx)
        d_a_slots[S_SV] = lg.d_a.copy(); d_b_slots[S_SV] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)
    if lora.proj_mlp:
        var lg = flux_lora_bwd(d_mlp_in_h, saved.norm.copy(), lora.proj_mlp.value(), S, ctx)
        d_a_slots[S_PMLP] = lg.d_a.copy(); d_b_slots[S_PMLP] = lg.d_b.copy()
        d_norm = _add_lists(d_norm, lg.d_x)

    # norm = modulate(ln, scale, shift)
    var ln_t = _t(saved.ln.copy(), [S, D], ctx)
    var mb = modulate_backward(_t(d_norm, [S, D], ctx), ln_t, scale_t, ctx)
    var d_scale = mb.d_scale.to_host(ctx)
    var d_shift = mb.d_shift.to_host(ctx)

    var lnb = layer_norm_backward(mb.d_x, x_t, ones_t, eps, ctx)
    var d_x_res = grg.d_x.to_host(ctx)
    var d_x_norm = lnb.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    var base = SingleBlockGrads(
        d_x^, d_w1^, d_b1^, d_w2^, d_b2^, d_q_norm^, d_k_norm^,
        d_shift^, d_scale^, d_gate^,
    )
    return SingleBlockLoraBackward(base^, SingleBlockLoraGrads(d_a_slots^, d_b_slots^))
