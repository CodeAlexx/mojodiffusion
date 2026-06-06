# serenitymojo/models/flux/flux_stack.mojo
#
# Flux (flux1-dev) FULL DiT STACK: forward (saving acts) + full-depth backward
# (training), COMPOSING the parity-verified Flux double/single blocks
# (models/flux/block.mojo, Phase-1 GREEN cos>=0.99999) into the complete model.
# This file COMPOSES; it rebuilds NO block math.
#
# WHAT IS ALREADY PROVEN (cos>=0.99999 vs torch) AND ONLY REUSED HERE
#   * models/flux/block.mojo : double_block_forward/backward (36 arms),
#       single_block_forward/backward (15 arms). The d_img/d_txt -> d_y
#       inter-block handoff and the joint-attention coupling are THAT file's.
#   * models/klein/klein_stack.mojo : the PROVEN stack-assembly contract (block
#       i's d_x IS block i-1's d_y, chained in REVERSE; the double->single
#       concat/slice seam; the final-layer backward). This file mirrors that
#       contract verbatim and ADDS the Flux-specific embed/modulation chain.
#
# ── HOW FLUX DIFFERS FROM KLEIN AT THE STACK LEVEL ──────────────────────────
# Klein passes FROZEN, SHARED modulation vectors into every block; their grads
# are accumulated but NOT backpropped further. Flux is different (measured from
# inference-flame/src/models/flux1_dit.rs and models/dit/flux1_dit.mojo, the
# composition oracle):
#
#   vec = time_in(t) + guidance_in(g) + vector_in(clip_pooled)      # [1, D]
#     time_in / guidance_in : t_embedder(timestep_embedding(t*1000)) MLP
#     vector_in             : linear -> silu -> linear over CLIP pooled (768->D)
#
#   PER DOUBLE block bi:  silu(vec) -> img_mod.lin / txt_mod.lin -> [1,6D] each
#                         -> chunk into (shift1,scale1,gate1,shift2,scale2,gate2)
#   PER SINGLE block bi:  silu(vec) -> modulation.lin -> [1,3D]
#                         -> chunk into (shift,scale,gate)
#   FINAL layer:          silu(vec) -> adaLN_modulation.1 -> [1,2D]
#                         -> (shift,scale); modulate_pre(img_out); linear -> out
#
# So in Flux the modulation vectors are NOT inputs; they are PRODUCED from `vec`
# by per-block mod.lin linears. The BACKWARD therefore threads the per-block
# [6D]/[3D] modvec grads (which the block already returns as d_shift1.. etc.)
# back through mod.lin (linear_backward) -> silu_backward(vec) -> d_vec, summed
# across EVERY block AND the final layer, then through the three embed MLPs to
# d_timestep / d_guidance / d_vector. Those three grads are the load-bearing
# arms the parity gate asserts (alongside out, d_img/d_txt-tokens, and a deep
# double + deep single block's weight grads), exactly per the Phase-2 brief.
#
# modulate_pre(x, shift, scale) = (1+scale)*LayerNorm(x,1e-6) + shift. The block
# already does layer_norm internally; the modvecs entering the block are [D].
# vec is [1,D] so a mod.lin output [1,6D] chunks to [1,D] == [D] after flatten.
#
# ── MEMORY / RESIDENCY (Flux-dev is ~12B) ───────────────────────────────────
# Full F32 residency of 19 double + 38 single blocks at D=3072 will NOT fit
# 24GB; like Ernie this needs streaming/checkpoint at real depth (noted in the
# real-smoke gate). The PARITY gate runs at REDUCED-but-structurally-complete
# depth (L=2-3 double + 2-3 single, REAL H/Dh/D) which fits and proves the
# composition. The forward here retains per-block `saved` (as Klein's small-depth
# stack gate does) — at real depth the same per-block-recompute checkpoint as
# klein_stack would be layered on; that is a Phase-3/runtime concern.
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor move-only; host List[Float32] carriers;
# bias linear = linear(x, w, Optional[Tensor](b), ctx).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.activations import silu, gelu
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.embeddings import t_embedder

from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import layer_norm_backward, LayerNormBackward
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.activation_backward import silu_backward

from serenitymojo.models.flux.block import (
    DoubleBlockWeights, StreamWeights, ModVecs, DoubleBlockSaved, DoubleBlockGrads,
    StreamGrads, double_block_forward, double_block_backward,
    SingleBlockWeights, SingleModVecs, SingleBlockSaved, SingleBlockGrads,
    single_block_forward, single_block_backward,
)


comptime TArc = ArcPointer[Tensor]


# ── host helpers ─────────────────────────────────────────────────────────────
def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _zeros(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(0.0)
    return o^


def _ones(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(1.0)
    return o^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


def _concat_seq(txt: List[Float32], img: List[Float32]) -> List[Float32]:
    # [N_TXT,D] then [N_IMG,D] -> [S,D] (txt FIRST), row-major append.
    var o = List[Float32]()
    for i in range(len(txt)):
        o.append(txt[i])
    for i in range(len(img)):
        o.append(img[i])
    return o^


def _split_seq(x: List[Float32], n_txt: Int, n_img: Int, d: Int) -> List[List[Float32]]:
    var txt = List[Float32]()
    var img = List[Float32]()
    var cut = n_txt * d
    for i in range(cut):
        txt.append(x[i])
    for i in range(cut, (n_txt + n_img) * d):
        img.append(x[i])
    var o = List[List[Float32]]()
    o.append(txt^)
    o.append(img^)
    return o^


# slice a [chunk_count * D] flat list into chunk #idx (length D).
def _chunk(x: List[Float32], idx: Int, d: Int) -> List[Float32]:
    var o = List[Float32]()
    var base = idx * d
    for i in range(d):
        o.append(x[base + i])
    return o^


# ── modvec packing (mirror block ModVecs chunk order) ────────────────────────
def _modvecs_from_flat(flat: List[Float32], d: Int) raises -> ModVecs:
    # flat is [6D]: shift1,scale1,gate1,shift2,scale2,gate2 (flux1_dit.rs:786-801)
    return ModVecs(
        _chunk(flat, 0, d), _chunk(flat, 1, d), _chunk(flat, 2, d),
        _chunk(flat, 3, d), _chunk(flat, 4, d), _chunk(flat, 5, d),
    )


def _single_modvecs_from_flat(flat: List[Float32], d: Int) raises -> SingleModVecs:
    # flat is [3D]: shift,scale,gate (flux1_dit.rs:944-948)
    return SingleModVecs(_chunk(flat, 0, d), _chunk(flat, 1, d), _chunk(flat, 2, d))


def _modvec6(g: StreamGrads) -> List[Float32]:
    # pack a double-block stream's 6 modvec grads into [6D] (block chunk order).
    var o = List[Float32]()
    for i in range(len(g.d_shift1)):
        o.append(g.d_shift1[i])
    for i in range(len(g.d_scale1)):
        o.append(g.d_scale1[i])
    for i in range(len(g.d_gate1)):
        o.append(g.d_gate1[i])
    for i in range(len(g.d_shift2)):
        o.append(g.d_shift2[i])
    for i in range(len(g.d_scale2)):
        o.append(g.d_scale2[i])
    for i in range(len(g.d_gate2)):
        o.append(g.d_gate2[i])
    return o^


def _single_modvec3(g: SingleBlockGrads) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(g.d_shift)):
        o.append(g.d_shift[i])
    for i in range(len(g.d_scale)):
        o.append(g.d_scale[i])
    for i in range(len(g.d_gate)):
        o.append(g.d_gate[i])
    return o^


# ═══════════════════════════════════════════════════════════════════════════
# Per-block modulation projection weights (Flux: img_mod.lin/txt_mod.lin per
# double block; modulation.lin per single block). vec [1,D] -> [1, chunk].
#   img_mod.lin / txt_mod.lin : w [6D, D], b [6D]
#   modulation.lin            : w [3D, D], b [3D]
# Device-resident (uploaded once). Used by forward AND backward.
# ═══════════════════════════════════════════════════════════════════════════
struct ModLin(Copyable, Movable):
    var w: TArc      # [chunk, D]
    var b: TArc      # [chunk]

    def __init__(out self, var w: List[Float32], var b: List[Float32], chunk: Int, D: Int, ctx: DeviceContext) raises:
        self.w = TArc(Tensor.from_host(w^, [chunk, D], STDtype.BF16, ctx))
        self.b = TArc(Tensor.from_host(b^, [chunk], STDtype.BF16, ctx))

    def __init__(out self, var w: TArc, var b: TArc):
        self.w = w^
        self.b = b^


struct DoubleModLin(Copyable, Movable):
    var img: ModLin
    var txt: ModLin

    def __init__(out self, var img: ModLin, var txt: ModLin):
        self.img = img^
        self.txt = txt^


# ── embed MLP weights (time_in / guidance_in / vector_in) ────────────────────
# Each is an MLPEmbedder: in_layer -> silu -> out_layer (both with bias).
#   time_in/guidance_in : in_layer [D, T_DIM], out_layer [D, D]   (T_DIM=256)
#       fed by the sinusoidal timestep_embedding(t*1000) -> [1, T_DIM].
#   vector_in           : in_layer [D, VEC_DIM], out_layer [D, D]  (VEC_DIM=768)
#       fed directly by CLIP-pooled [1, VEC_DIM] (NOT a sinusoid).
struct EmbedMlp(Copyable, Movable):
    var in_w: TArc
    var in_b: TArc
    var out_w: TArc
    var out_b: TArc

    def __init__(
        out self,
        var in_w: List[Float32], var in_b: List[Float32],
        var out_w: List[Float32], var out_b: List[Float32],
        in_dim: Int, D: Int, ctx: DeviceContext,
    ) raises:
        self.in_w = TArc(Tensor.from_host(in_w^, [D, in_dim], STDtype.BF16, ctx))
        self.in_b = TArc(Tensor.from_host(in_b^, [D], STDtype.BF16, ctx))
        self.out_w = TArc(Tensor.from_host(out_w^, [D, D], STDtype.BF16, ctx))
        self.out_b = TArc(Tensor.from_host(out_b^, [D], STDtype.BF16, ctx))

    def __init__(
        out self,
        var in_w: TArc, var in_b: TArc,
        var out_w: TArc, var out_b: TArc,
    ):
        self.in_w = in_w^
        self.in_b = in_b^
        self.out_w = out_w^
        self.out_b = out_b^


# ── stack-level frozen/shared base (embeds + per-block mod.lin + final layer) ─
struct FluxStackBase(Copyable, Movable):
    var img_in: TArc          # [D, in_ch]
    var img_in_b: TArc        # [D]
    var txt_in: TArc          # [D, txt_ch]
    var txt_in_b: TArc        # [D]
    var time_in: EmbedMlp
    var has_guidance: Bool
    var guidance_in: EmbedMlp
    var vector_in: EmbedMlp
    var dbl_mod: List[DoubleModLin]   # per double block
    var sgl_mod: List[ModLin]         # per single block
    var final_adaln_w: TArc   # [2D, D]   adaLN_modulation.1
    var final_adaln_b: TArc   # [2D]
    var final_lin: TArc       # [out_ch, D]
    var final_lin_b: TArc     # [out_ch]

    def __init__(
        out self,
        var img_in: List[Float32], var img_in_b: List[Float32],
        var txt_in: List[Float32], var txt_in_b: List[Float32],
        var time_in: EmbedMlp, has_guidance: Bool, var guidance_in: EmbedMlp,
        var vector_in: EmbedMlp,
        var dbl_mod: List[DoubleModLin], var sgl_mod: List[ModLin],
        var final_adaln_w: List[Float32], var final_adaln_b: List[Float32],
        var final_lin: List[Float32], var final_lin_b: List[Float32],
        D: Int, in_ch: Int, txt_ch: Int, out_ch: Int, ctx: DeviceContext,
    ) raises:
        self.img_in = TArc(Tensor.from_host(img_in^, [D, in_ch], STDtype.BF16, ctx))
        self.img_in_b = TArc(Tensor.from_host(img_in_b^, [D], STDtype.BF16, ctx))
        self.txt_in = TArc(Tensor.from_host(txt_in^, [D, txt_ch], STDtype.BF16, ctx))
        self.txt_in_b = TArc(Tensor.from_host(txt_in_b^, [D], STDtype.BF16, ctx))
        self.time_in = time_in^
        self.has_guidance = has_guidance
        self.guidance_in = guidance_in^
        self.vector_in = vector_in^
        self.dbl_mod = dbl_mod^
        self.sgl_mod = sgl_mod^
        self.final_adaln_w = TArc(Tensor.from_host(final_adaln_w^, [2 * D, D], STDtype.BF16, ctx))
        self.final_adaln_b = TArc(Tensor.from_host(final_adaln_b^, [2 * D], STDtype.BF16, ctx))
        self.final_lin = TArc(Tensor.from_host(final_lin^, [out_ch, D], STDtype.BF16, ctx))
        self.final_lin_b = TArc(Tensor.from_host(final_lin_b^, [out_ch], STDtype.BF16, ctx))

    def __init__(
        out self,
        var img_in: TArc, var img_in_b: TArc,
        var txt_in: TArc, var txt_in_b: TArc,
        var time_in: EmbedMlp, has_guidance: Bool, var guidance_in: EmbedMlp,
        var vector_in: EmbedMlp,
        var dbl_mod: List[DoubleModLin], var sgl_mod: List[ModLin],
        var final_adaln_w: TArc, var final_adaln_b: TArc,
        var final_lin: TArc, var final_lin_b: TArc,
    ):
        self.img_in = img_in^
        self.img_in_b = img_in_b^
        self.txt_in = txt_in^
        self.txt_in_b = txt_in_b^
        self.time_in = time_in^
        self.has_guidance = has_guidance
        self.guidance_in = guidance_in^
        self.vector_in = vector_in^
        self.dbl_mod = dbl_mod^
        self.sgl_mod = sgl_mod^
        self.final_adaln_w = final_adaln_w^
        self.final_adaln_b = final_adaln_b^
        self.final_lin = final_lin^
        self.final_lin_b = final_lin_b^


# ── forward result ───────────────────────────────────────────────────────────
struct FluxStackForward(Copyable, Movable):
    var out: List[Float32]            # [N_IMG, out_ch]
    var vec: List[Float32]            # [1, D]   the summed embed vector
    var vec_silu: List[Float32]       # [1, D]   silu(vec)  (mod.lin / final input)
    # per-block modvec FLATS actually used (recomputed for backward chunking)
    var dbl_img_mod: List[List[Float32]]   # num_double x [6D]
    var dbl_txt_mod: List[List[Float32]]
    var sgl_mod_flat: List[List[Float32]]  # num_single x [3D]
    var dbl_saved: List[DoubleBlockSaved]
    var sgl_saved: List[SingleBlockSaved]
    var img_out: TArc                 # [N_IMG, D]  slice before final layer
    var ln_img_out: TArc              # [N_IMG, D]  layer_norm(img_out)
    var final_shift: List[Float32]    # [D]
    var final_scale: List[Float32]    # [D]
    # embed-MLP intermediate acts (for the embed backward)
    var t_emb: List[Float32]          # [1, T_DIM]  sinusoid (time)
    var t_hid: List[Float32]          # [1, D]      in_layer(t_emb)
    var g_emb: List[Float32]          # [1, T_DIM]
    var g_hid: List[Float32]          # [1, D]
    var v_hid: List[Float32]          # [1, D]      vector_in.in_layer(clip)

    def __init__(
        out self,
        var out: List[Float32], var vec: List[Float32], var vec_silu: List[Float32],
        var dbl_img_mod: List[List[Float32]], var dbl_txt_mod: List[List[Float32]],
        var sgl_mod_flat: List[List[Float32]],
        var dbl_saved: List[DoubleBlockSaved], var sgl_saved: List[SingleBlockSaved],
        var img_out: TArc, var ln_img_out: TArc,
        var final_shift: List[Float32], var final_scale: List[Float32],
        var t_emb: List[Float32], var t_hid: List[Float32],
        var g_emb: List[Float32], var g_hid: List[Float32], var v_hid: List[Float32],
    ):
        self.out = out^
        self.vec = vec^
        self.vec_silu = vec_silu^
        self.dbl_img_mod = dbl_img_mod^
        self.dbl_txt_mod = dbl_txt_mod^
        self.sgl_mod_flat = sgl_mod_flat^
        self.dbl_saved = dbl_saved^
        self.sgl_saved = sgl_saved^
        self.img_out = img_out^
        self.ln_img_out = ln_img_out^
        self.final_shift = final_shift^
        self.final_scale = final_scale^
        self.t_emb = t_emb^
        self.t_hid = t_hid^
        self.g_emb = g_emb^
        self.g_hid = g_hid^
        self.v_hid = v_hid^


# ── backward result ──────────────────────────────────────────────────────────
struct FluxStackGrads(Copyable, Movable):
    var d_img_tokens: List[Float32]   # [N_IMG, in_ch]
    var d_txt_tokens: List[Float32]   # [N_TXT, txt_ch]
    var d_vec: List[Float32]          # [1, D]   (accumulated into the embeds)
    var d_timestep: List[Float32]     # [1]      grad wrt scaled-t input
    var d_guidance: List[Float32]     # [1]      grad wrt scaled-g input
    var d_vector: List[Float32]       # [1, VEC_DIM]  grad wrt CLIP-pooled
    var dbl_grads: List[DoubleBlockGrads]
    var sgl_grads: List[SingleBlockGrads]

    def __init__(
        out self,
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
        var d_vec: List[Float32],
        var d_timestep: List[Float32], var d_guidance: List[Float32], var d_vector: List[Float32],
        var dbl_grads: List[DoubleBlockGrads], var sgl_grads: List[SingleBlockGrads],
    ):
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^
        self.d_vec = d_vec^
        self.d_timestep = d_timestep^
        self.d_guidance = d_guidance^
        self.d_vector = d_vector^
        self.dbl_grads = dbl_grads^
        self.sgl_grads = sgl_grads^


# ── embed-vec forward: vec = time + guidance + vector ────────────────────────
# Returns (vec [1,D], intermediates for backward).
struct _VecFwd(Copyable, Movable):
    var vec: List[Float32]
    var t_emb: List[Float32]
    var t_hid: List[Float32]
    var g_emb: List[Float32]
    var g_hid: List[Float32]
    var v_hid: List[Float32]

    def __init__(
        out self, var vec: List[Float32], var t_emb: List[Float32], var t_hid: List[Float32],
        var g_emb: List[Float32], var g_hid: List[Float32], var v_hid: List[Float32],
    ):
        self.vec = vec^
        self.t_emb = t_emb^
        self.t_hid = t_hid^
        self.g_emb = g_emb^
        self.g_hid = g_hid^
        self.v_hid = v_hid^


# An MLPEmbedder forward that exposes the sinusoid + in_layer hidden for backward.
# emb_in [1, in_dim] (already sinusoid OR raw clip) -> in_layer -> silu -> out_layer.
def _mlp_embed_fwd(
    emb_in: List[Float32], mlp: EmbedMlp, in_dim: Int, D: Int, ctx: DeviceContext,
) raises -> List[List[Float32]]:
    var x = _t(emb_in, [1, in_dim], ctx)
    var b_in = Optional[Tensor](mlp.in_b[].clone(ctx))
    var hid = linear(x, mlp.in_w[], b_in, ctx)        # [1, D]
    var hid_h = hid.to_host(ctx)
    var act = silu(_t(hid_h.copy(), [1, D], ctx), ctx)
    var b_out = Optional[Tensor](mlp.out_b[].clone(ctx))
    var out = linear(act, mlp.out_w[], b_out, ctx)    # [1, D]
    var o = List[List[Float32]]()
    o.append(out.to_host(ctx))   # [0] = mlp output
    o.append(hid_h^)             # [1] = in_layer hidden (silu input)
    return o^


# T_DIM: sinusoid dim for time/guidance (flux1-dev timestep_dim = 256).
def _embed_vec_forward(
    timestep: List[Float32], guidance: Optional[List[Float32]], vector: List[Float32],
    base: FluxStackBase,
    D: Int, T_DIM: Int, VEC_DIM: Int, ctx: DeviceContext,
) raises -> _VecFwd:
    # time: timestep_embedding(t*1000) is done by the CALLER (t already scaled),
    # matching models/dit/flux1_dit.mojo (caller pre-scales t*1000). The sinusoid
    # itself is produced here via t_embedder? No — t_embedder fuses sinusoid+MLP;
    # we need the intermediate, so we build the sinusoid + MLP explicitly.
    from serenitymojo.ops.embeddings import timestep_embedding
    var t_emb = timestep_embedding(
        _t(timestep, [1], ctx), T_DIM, ctx, Float32(10000.0), STDtype.F32
    ).to_host(ctx)  # legacy host MLP path
    var tr = _mlp_embed_fwd(t_emb.copy(), base.time_in, T_DIM, D, ctx)
    var vec = tr[0].copy()
    var t_hid = tr[1].copy()

    var g_emb = _zeros(T_DIM)
    var g_hid = _zeros(D)
    if base.has_guidance and guidance:
        g_emb = timestep_embedding(
            _t(guidance.value().copy(), [1], ctx),
            T_DIM,
            ctx,
            Float32(10000.0),
            STDtype.F32,
        ).to_host(ctx)
        var gr = _mlp_embed_fwd(g_emb.copy(), base.guidance_in, T_DIM, D, ctx)
        vec = _add_lists(vec, gr[0])
        g_hid = gr[1].copy()

    # vector_in: raw CLIP pooled [1,VEC_DIM] through the SAME MLPEmbedder shape.
    var vr = _mlp_embed_fwd(vector, base.vector_in, VEC_DIM, D, ctx)
    vec = _add_lists(vec, vr[0])
    var v_hid = vr[1].copy()

    return _VecFwd(vec^, t_emb^, t_hid^, g_emb^, g_hid^, v_hid^)


# silu(vec) -> mod.lin -> [chunk] flat (host).
def _mod_proj(vec_silu: List[Float32], ml: ModLin, chunk: Int, D: Int, ctx: DeviceContext) raises -> List[Float32]:
    var b = Optional[Tensor](ml.b[].clone(ctx))
    return linear(_t(vec_silu, [1, D], ctx), ml.w[], b, ctx).to_host(ctx)


# ── FULL FORWARD ─────────────────────────────────────────────────────────────
def flux_stack_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    timestep: List[Float32], guidance: Optional[List[Float32]], vector: List[Float32],
    base: FluxStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxStackForward:
    var num_double = len(dbw)
    var num_single = len(sbw)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    # ── embeds -> vec ──
    var vf = _embed_vec_forward(timestep, guidance, vector, base, D, T_DIM, VEC_DIM, ctx)
    var vec = vf.vec.copy()
    var vec_silu = silu(_t(vec.copy(), [1, D], ctx), ctx).to_host(ctx)   # [1,D]

    # ── input projections (with bias) ──
    var bi_img = Optional[Tensor](base.img_in_b[].clone(ctx))
    var img = linear(_t(img_tokens.copy(), [N_IMG, in_ch], ctx), base.img_in[], bi_img, ctx).to_host(ctx)
    var bi_txt = Optional[Tensor](base.txt_in_b[].clone(ctx))
    var txt = linear(_t(txt_tokens.copy(), [N_TXT, txt_ch], ctx), base.txt_in[], bi_txt, ctx).to_host(ctx)

    # ── double-stream stack ──
    var dbl_img_mod = List[List[Float32]]()
    var dbl_txt_mod = List[List[Float32]]()
    var dbl_saved = List[DoubleBlockSaved]()
    for bi in range(num_double):
        var im_flat = _mod_proj(vec_silu.copy(), base.dbl_mod[bi].img, 6 * D, D, ctx)
        var tm_flat = _mod_proj(vec_silu.copy(), base.dbl_mod[bi].txt, 6 * D, D, ctx)
        var im = _modvecs_from_flat(im_flat, D)
        var tm = _modvecs_from_flat(tm_flat, D)
        var fwd = double_block_forward[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt.copy(), dbw[bi], im, tm, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        dbl_saved.append(fwd.saved.copy())
        dbl_img_mod.append(im_flat^)
        dbl_txt_mod.append(tm_flat^)
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()

    # double->single transition: x = concat(1, txt, img) -> [S,D]  (txt FIRST)
    var x = _concat_seq(txt, img)

    # ── single-stream stack ──
    var sgl_mod_flat = List[List[Float32]]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        var sm_flat = _mod_proj(vec_silu.copy(), base.sgl_mod[bi], 3 * D, D, ctx)
        var sm = _single_modvecs_from_flat(sm_flat, D)
        var fwd = single_block_forward[H, Dh, S](
            x.copy(), sbw[bi], sm, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        sgl_saved.append(fwd.saved.copy())
        sgl_mod_flat.append(sm_flat^)
        x = fwd.out.copy()

    # img_out = slice(x, 1, N_TXT, N_IMG) -> [N_IMG,D]
    var parts = _split_seq(x, N_TXT, N_IMG, D)
    var img_out = parts[1].copy()

    # ── final layer: silu(vec) -> adaLN.1 -> (shift,scale); modulate_pre; linear ──
    var fb = Optional[Tensor](base.final_adaln_b[].clone(ctx))
    var fmods = linear(_t(vec_silu.copy(), [1, D], ctx), base.final_adaln_w[], fb, ctx).to_host(ctx)  # [1,2D]
    var final_shift = _chunk(fmods, 0, D)
    var final_scale = _chunk(fmods, 1, D)

    var ln_img_out = layer_norm(
        _t(img_out.copy(), [N_IMG, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), eps, ctx,
    ).to_host(ctx)
    var normed = modulate(
        _t(ln_img_out.copy(), [N_IMG, D], ctx),
        _t(final_scale.copy(), [D], ctx), _t(final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var flb = Optional[Tensor](base.final_lin_b[].clone(ctx))
    var out = linear(_t(normed, [N_IMG, D], ctx), base.final_lin[], flb, ctx).to_host(ctx)

    return FluxStackForward(
        out^, vec^, vec_silu^,
        dbl_img_mod^, dbl_txt_mod^, sgl_mod_flat^,
        dbl_saved^, sgl_saved^,
        TArc(_t(img_out^, [N_IMG, D], ctx)), TArc(_t(ln_img_out^, [N_IMG, D], ctx)),
        final_shift^, final_scale^,
        vf.t_emb.copy(), vf.t_hid.copy(), vf.g_emb.copy(), vf.g_hid.copy(), vf.v_hid.copy(),
    )


# ── embed backward: d_vec -> d_timestep, d_guidance, d_vector ────────────────
struct _EmbedBack(Copyable, Movable):
    var d_timestep: List[Float32]
    var d_guidance: List[Float32]
    var d_vector: List[Float32]

    def __init__(out self, var d_timestep: List[Float32], var d_guidance: List[Float32], var d_vector: List[Float32]):
        self.d_timestep = d_timestep^
        self.d_guidance = d_guidance^
        self.d_vector = d_vector^


# d_out_mlp [1,D] is the upstream grad of one MLPEmbedder's output. Returns the
# grad wrt its (sinusoid OR raw) input [1, in_dim].
def _mlp_embed_backward(
    d_out_mlp: List[Float32], emb_in: List[Float32], hid: List[Float32],
    mlp: EmbedMlp, in_dim: Int, D: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    # out = linear(act, out_w, out_b)
    var act = silu(_t(hid.copy(), [1, D], ctx), ctx).to_host(ctx)
    var lb_out = linear_backward(
        _t(d_out_mlp, [1, D], ctx), _t(act, [1, D], ctx), mlp.out_w[], 1, D, D, ctx,
    )
    # act = silu(hid)
    var d_hid = silu_backward(lb_out.d_x, _t(hid.copy(), [1, D], ctx), ctx)
    # hid = linear(emb_in, in_w, in_b)
    var lb_in = linear_backward(d_hid, _t(emb_in.copy(), [1, in_dim], ctx), mlp.in_w[], 1, in_dim, D, ctx)
    return lb_in.d_x.to_host(ctx)


# d_timestep: the sinusoid timestep_embedding is non-learnable; we backprop
# through it numerically-equivalent via its analytic jacobian? The gate only
# needs d wrt the SCALED-t scalar. timestep_embedding(t) -> [cos(t*f), sin(t*f)];
# d/dt = sum_i d_emb_cos_i * (-f_i sin) + d_emb_sin_i * (f_i cos). We compute it
# from the sinusoid grad and the saved sinusoid. Same for guidance.
def _sinusoid_t_grad(d_emb: List[Float32], emb: List[Float32], T_DIM: Int, max_period: Float32) -> Float32:
    from std.math import log as flog, exp as fexp
    var half = T_DIM // 2
    var neg_ln = -flog(max_period)
    var acc = Float32(0.0)
    # emb layout (cos-first): emb[i]=cos(angle_i) for i in [0,half); emb[half+i]=sin.
    # angle_i = t * f_i ; d angle/dt = f_i. d cos/dt = -f_i sin = -f_i*emb[half+i];
    # d sin/dt =  f_i cos =  f_i*emb[i].
    for i in range(half):
        var f = fexp(neg_ln * (Float32(i) / Float32(half)))
        acc += d_emb[i] * (-f) * emb[half + i]
        acc += d_emb[half + i] * (f) * emb[i]
    return acc


def _embed_vec_backward(
    d_vec: List[Float32], saved: FluxStackForward, base: FluxStackBase,
    has_guidance: Bool, D: Int, T_DIM: Int, VEC_DIM: Int, max_period: Float32, ctx: DeviceContext,
) raises -> _EmbedBack:
    # vec = t_out + g_out + v_out, each is one MLPEmbedder output. The same d_vec
    # flows into each branch's output (sum rule).
    # time branch
    var d_t_emb = _mlp_embed_backward(d_vec.copy(), saved.t_emb, saved.t_hid, base.time_in, T_DIM, D, ctx)
    var d_timestep = List[Float32]()
    d_timestep.append(_sinusoid_t_grad(d_t_emb, saved.t_emb, T_DIM, max_period))

    var d_guidance = List[Float32]()
    d_guidance.append(0.0)
    if has_guidance:
        var d_g_emb = _mlp_embed_backward(d_vec.copy(), saved.g_emb, saved.g_hid, base.guidance_in, T_DIM, D, ctx)
        d_guidance[0] = _sinusoid_t_grad(d_g_emb, saved.g_emb, T_DIM, max_period)

    # vector branch: input is the raw CLIP pooled [1,VEC_DIM]; d_vector is the
    # grad wrt that raw input (load-bearing arm).
    # need the raw clip input to recompute act — we saved v_hid (= in_layer(clip));
    # the in_layer backward needs emb_in only for d_w, but d_x needs only out_w/hid.
    # _mlp_embed_backward returns d wrt emb_in directly; pass a zero emb_in (only
    # used for d_w which we discard) — but linear_backward needs the real emb_in to
    # form d_w; d_x is independent of emb_in, so a placeholder of right length is fine.
    var d_vector = _mlp_embed_backward(d_vec.copy(), _zeros(VEC_DIM), saved.v_hid, base.vector_in, VEC_DIM, D, ctx)

    return _EmbedBack(d_timestep^, d_guidance^, d_vector^)


# ── FULL BACKWARD ────────────────────────────────────────────────────────────
def flux_stack_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: FluxStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    cos: List[Float32], sin: List[Float32],
    saved: FluxStackForward,
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, max_period: Float32, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxStackGrads:
    var num_double = len(dbw)
    var num_single = len(sbw)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    # accumulator: d_vec_silu (grad into silu(vec)). Every mod.lin + final adaLN
    # reads silu(vec); we accumulate their d_vec_silu, then silu_backward once.
    var d_vec_silu = _zeros(D)

    # ── final layer backward ──
    var normed = modulate(
        saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), _t(saved.final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var lbf = linear_backward(
        _t(d_out, [N_IMG, out_ch], ctx), _t(normed, [N_IMG, D], ctx), base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)
    # normed = modulate(ln_img_out, final_scale, final_shift)
    var mbf = modulate_backward(
        _t(d_normed, [N_IMG, D], ctx), saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)
    var d_final_scale = mbf.d_scale.to_host(ctx)   # [D]
    var d_final_shift = mbf.d_shift.to_host(ctx)   # [D]
    # fmods = linear(vec_silu, adaLN_w, adaLN_b); fmods chunks = (shift,scale)
    # d_fmods = [d_shift | d_scale]  (order: chunk0=shift, chunk1=scale)
    var d_fmods = _concat_seq(d_final_shift, d_final_scale)   # [2D]
    var lb_fmods = linear_backward(
        _t(d_fmods, [1, 2 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
        base.final_adaln_w[], 1, D, 2 * D, ctx,
    )
    d_vec_silu = _add_lists(d_vec_silu, lb_fmods.d_x.to_host(ctx))

    # ln_img_out = layer_norm(img_out)
    var lnbf = layer_norm_backward(
        _t(d_ln_img_out, [N_IMG, D], ctx), saved.img_out[], _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_img_out = lnbf.d_x.to_host(ctx)   # [N_IMG,D]

    # single stack seed: img rows = d_img_out, txt rows = 0.
    var d_x = _concat_seq(_zeros(N_TXT * D), d_img_out)

    # ── single-stream backward (REVERSE) ──
    var sgl_grads_rev = List[SingleBlockGrads]()
    var bi = num_single - 1
    while bi >= 0:
        var sm = _single_modvecs_from_flat(saved.sgl_mod_flat[bi].copy(), D)
        var bg = single_block_backward[H, Dh, S](
            d_x.copy(), sbw[bi], sm, saved.sgl_saved[bi], cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_x = bg.d_x.copy()
        # this block's modvec grad [3D] -> mod.lin backward -> d_vec_silu
        var d_sm = _single_modvec3(bg)
        var lb_sm = linear_backward(
            _t(d_sm, [1, 3 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
            base.sgl_mod[bi].w[], 1, D, 3 * D, ctx,
        )
        d_vec_silu = _add_lists(d_vec_silu, lb_sm.d_x.to_host(ctx))
        sgl_grads_rev.append(bg^)
        bi -= 1
    var sgl_grads = List[SingleBlockGrads]()
    var j = len(sgl_grads_rev) - 1
    while j >= 0:
        sgl_grads.append(sgl_grads_rev[j].copy())
        j -= 1

    # double->single seam: split d_x [S,D] -> d_txt_out, d_img_out (txt FIRST)
    var seam = _split_seq(d_x, N_TXT, N_IMG, D)
    var d_to = seam[0].copy()   # [N_TXT,D]
    var d_io = seam[1].copy()   # [N_IMG,D]

    # ── double-stream backward (REVERSE) ──
    var dbl_grads_rev = List[DoubleBlockGrads]()
    var di = num_double - 1
    while di >= 0:
        var im = _modvecs_from_flat(saved.dbl_img_mod[di].copy(), D)
        var tm = _modvecs_from_flat(saved.dbl_txt_mod[di].copy(), D)
        var bg = double_block_backward[H, Dh, N_IMG, N_TXT, S](
            d_io.copy(), d_to.copy(), dbw[di], im, tm, saved.dbl_saved[di],
            cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_io = bg.img.d_x.copy()
        d_to = bg.txt.d_x.copy()
        # img_mod / txt_mod [6D] grads -> their mod.lin backward -> d_vec_silu
        var d_im = _modvec6(bg.img)
        var d_tm = _modvec6(bg.txt)
        var lb_im = linear_backward(
            _t(d_im, [1, 6 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
            base.dbl_mod[di].img.w[], 1, D, 6 * D, ctx,
        )
        var lb_tm = linear_backward(
            _t(d_tm, [1, 6 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
            base.dbl_mod[di].txt.w[], 1, D, 6 * D, ctx,
        )
        d_vec_silu = _add_lists(d_vec_silu, lb_im.d_x.to_host(ctx))
        d_vec_silu = _add_lists(d_vec_silu, lb_tm.d_x.to_host(ctx))
        dbl_grads_rev.append(bg^)
        di -= 1
    var dbl_grads = List[DoubleBlockGrads]()
    var k = len(dbl_grads_rev) - 1
    while k >= 0:
        dbl_grads.append(dbl_grads_rev[k].copy())
        k -= 1

    # ── input-projection backward ──
    var lbi = linear_backward(
        _t(d_io, [N_IMG, D], ctx), _t(img_tokens, [N_IMG, in_ch], ctx), base.img_in[],
        N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var lbt = linear_backward(
        _t(d_to, [N_TXT, D], ctx), _t(txt_tokens, [N_TXT, txt_ch], ctx), base.txt_in[],
        N_TXT, txt_ch, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)

    # ── vec backward: silu(vec) was the input to every mod.lin + final adaLN ──
    # d_vec = silu_backward(d_vec_silu, vec)
    var d_vec = silu_backward(_t(d_vec_silu.copy(), [1, D], ctx), _t(saved.vec.copy(), [1, D], ctx), ctx).to_host(ctx)

    # ── embed backward: d_vec -> d_timestep, d_guidance, d_vector ──
    var eb = _embed_vec_backward(d_vec.copy(), saved, base, base.has_guidance, D, T_DIM, VEC_DIM, max_period, ctx)
    var d_timestep = eb.d_timestep.copy()
    var d_guidance = eb.d_guidance.copy()
    var d_vector = eb.d_vector.copy()

    return FluxStackGrads(
        d_img_tokens^, d_txt_tokens^, d_vec^,
        d_timestep^, d_guidance^, d_vector^,
        dbl_grads^, sgl_grads^,
    )
