# serenitymojo/models/anima/anima_stack.mojo
#
# ANIMA (Cosmos-Predict2 MiniTrainDIT) FULL 28-BLOCK STACK: forward (saving
# ckpt-inputs) + full-depth backward (training), COMPOSING the already-parity-
# verified Anima block (models/anima/block.mojo, 23/23 cos>=0.99999999 vs torch,
# skeptic-confirmed P1b) into the complete DiT backbone. This file COMPOSES; it
# rebuilds NOTHING per-block. Mirrors models/ernie/ernie_stack.mojo (the PROVEN
# uniform-block stack pattern: per-block recompute in backward to bound memory,
# d_x -> d_y inter-block handoff chained in REVERSE).
#
# ANIMA vs ERNIE/KLEIN — the genuinely-new composition detail:
#   * PER-BLOCK modulation (NOT shared). Each block i has its OWN
#     adaln_modulation_{self_attn,cross_attn,mlp}.{1,2} weights and computes its
#     own (shift,scale,gate) from the SHARED t_silu + base_adaln (anima.rs:474-503,
#     net.blocks.{i}.adaln_modulation_*). So — unlike Klein/Ernie shared-AdaLN —
#     the 6 mod-weight grads are PER-BLOCK (returned per block, NOT summed). What
#     IS shared across all 28 blocks is t_silu = silu(t_cond) and base_adaln, so
#     d_t_silu and d_base_adaln SUM across all 28 blocks + the final layer.
#     (Verified: each block.backward returns d_t_silu; the final layer also
#     re-uses silu(t_cond) and base_adaln[:4096].)
#   * SELF + CROSS + MLP per block (Ernie is self+MLP only). cross-attn context is
#     the FROZEN cached text adapter output — a stack INPUT, not a leaf; its grad
#     path stops at the cross-attn k/v Linear d_input (discarded by block.backward).
#   * Final layer: final_adaln_modulation = silu(t_cond) @ fl_mod1[256,2048] @
#     fl_mod2[4096,256] + base_adaln[:4096] -> chunk2 (shift,scale); apply_adaln
#     (LayerNorm-no-affine + (1+scale)*ln+shift); linear(2048->64). unpatchify is
#     reshape/permute only (no learnable weights, no-op grads) — the gate compares
#     the [B,S_img,64] patches directly (like Ernie compares patches).
#
# SCOPE (mirrors Ernie/Klein stack): BASE forward+backward, NO LoRA wiring (E).
#   t_silu (=silu(t_cond)) + base_adaln are passed in PRECOMPUTED (the t_embedder
#   MLP / t_embedding_norm backprop link is the train-loop phase E, explicitly
#   deferred — exactly as ernie_stack defers the time-MLP link). The summed
#   d_t_silu + d_base_adaln ARE produced and RETURNED so E can run the t_embedder
#   backward at step end. x_embedder + final-layer weight grads are computed and
#   returned for completeness.
#
# Boundary contract: inter-block stream carried as a HOST List[Float32] [B,S,D]
#   (matches block.mojo's host out / host d_out). Per-block recompute keeps peak
#   device memory at ~one block's activation footprint + resident weights + rope.
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor move-only; host List[Float32] carriers;
# no-bias linear = linear(x, w, Optional[Tensor](None), ctx).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import reshape, slice, add, mul, concat
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import layer_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward

from serenitymojo.models.anima.weights import (
    AnimaBlockWeights, AnimaStackBase,
)
from serenitymojo.models.anima.block import (
    AnimaBlockSaved, AnimaBlockGrads, AnimaBlockForward,
    anima_block_forward, anima_block_backward,
)
from serenitymojo.models.dit.anima_contract import ANIMA_HIDDEN  # 2048


comptime TArc = ArcPointer[Tensor]


# ── host helpers (boundary only) ─────────────────────────────────────────────
def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _ones(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(1.0)
    return o^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


# linear with a RESIDENT device weight (borrowed), no bias. x_h is [rows, kin].
def _linear_wdev(
    x_h: List[Float32], w_t: Tensor, rows: Int, kin: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    return linear(_t(x_h, [rows, kin], ctx), w_t, Optional[Tensor](None), ctx).to_host(ctx)


# ── FORWARD RESULT: out patches + checkpoint inputs (per-block input) + final
#    layer activations needed for the final-layer backward. ────────────────────
struct AnimaStackForward(Movable):
    var out: List[Float32]        # [B*S_img, 64]  final patches (host)
    var x_emb: List[Float32]      # [B*S_img, D]   patch_embed(patches) — block-0 input
    var blk_x_in: List[TArc]      # num_blocks x [B,S_img,D] checkpoint inputs
    var x_final: TArc             # [B,S_img,D] last-block output (final-layer input)
    var fl_ln: TArc               # [B,S_img,D] layer_norm(x_final) (no affine)
    var fl_mod_h: TArc            # [B,256]   final_layer mod1 hidden
    var fl_shift: TArc            # [B,D]
    var fl_scale: TArc            # [B,D]
    var fl_xmod: TArc             # [B,S_img,D] apply_adaln output (final_linear input)
    # SAVED-ACTIVATION fast path (device-resident only): each block's full
    # AnimaBlockSaved retained so backward READS it instead of recomputing the
    # block forward. Empty for the recompute paths (streamed / small-depth parity).
    var blk_saved: List[AnimaBlockSaved]

    def __init__(
        out self,
        var out: List[Float32], var x_emb: List[Float32],
        var blk_x_in: List[TArc], var x_final: TArc, var fl_ln: TArc,
        var fl_mod_h: TArc, var fl_shift: TArc, var fl_scale: TArc, var fl_xmod: TArc,
        var blk_saved: List[AnimaBlockSaved] = List[AnimaBlockSaved](),
    ):
        self.out = out^
        self.x_emb = x_emb^
        self.blk_x_in = blk_x_in^
        self.x_final = x_final^
        self.fl_ln = fl_ln^
        self.fl_mod_h = fl_mod_h^
        self.fl_shift = fl_shift^
        self.fl_scale = fl_scale^
        self.fl_xmod = fl_xmod^
        self.blk_saved = blk_saved^


# ── BACKWARD RESULT: input-patch grad + per-block grads + summed shared grads ─
struct AnimaStackGrads(Movable):
    var d_patches: List[Float32]      # [B*S_img, 68]  grad into the patchify input
    var blk_grads: List[AnimaBlockGrads]
    # SUMMED-across-(28 blocks + final layer) shared grads (composition detail):
    var d_t_silu: List[Float32]       # [B, 2048]  grad into silu(t_cond)
    var d_base_adaln: List[Float32]   # [B, 6144]  grad into base_adaln
    # base-weight grads (for completeness / full-FT):
    var d_x_embed: List[Float32]      # [D, 68]   x_embedder.proj.1.weight
    var d_fl_lin: List[Float32]       # [64, D]   final_layer.linear.weight
    var d_fl_mod1: List[Float32]      # [256, D]  final_layer.adaln_modulation.1.weight
    var d_fl_mod2: List[Float32]      # [4096,256] final_layer.adaln_modulation.2.weight

    def __init__(
        out self,
        var d_patches: List[Float32], var blk_grads: List[AnimaBlockGrads],
        var d_t_silu: List[Float32], var d_base_adaln: List[Float32],
        var d_x_embed: List[Float32], var d_fl_lin: List[Float32],
        var d_fl_mod1: List[Float32], var d_fl_mod2: List[Float32],
    ):
        self.d_patches = d_patches^
        self.blk_grads = blk_grads^
        self.d_t_silu = d_t_silu^
        self.d_base_adaln = d_base_adaln^
        self.d_x_embed = d_x_embed^
        self.d_fl_lin = d_fl_lin^
        self.d_fl_mod1 = d_fl_mod1^
        self.d_fl_mod2 = d_fl_mod2^


# ══════════════════════════════════════════════════════════════════════════════
# FULL FORWARD (checkpoint inputs only retained)
# ══════════════════════════════════════════════════════════════════════════════
# patches:    [B*S_img, 68]   patchify output (the +1 mask channel is already in
#                              the 68 — this is the x_embedder.proj input).
# t_cond:     [B, 2048]       RMSNorm(sinusoidal) timestep cond (shared across all
#                              blocks + final layer). The block internally computes
#                              silu(t_cond) (matching anima.rs adaln_modulation /
#                              final_adaln_modulation, both of which call t_cond.silu()).
#                              The stack therefore passes t_cond RAW to each block and
#                              applies silu(t_cond) once in the final layer. The block
#                              returns d_t_silu (grad at the silu output); the stack
#                              SUMS those — the natural handoff to the deferred
#                              t_embedder link (one silu_backward at step end).
# base_adaln: [B, 6144]       base modulation (shared)
# context:    [B, S_txt, 1024] FROZEN cached text adapter output
# blocks:     num_blocks AnimaBlockWeights (F32 resident)
# cos/sin:    [B*S_img*H, Dh/2] F32 (3D-RoPE half-split tables, resident)
def anima_stack_forward[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    patches: List[Float32],
    t_cond: List[Float32], base_adaln: List[Float32], context: List[Float32],
    base: AnimaStackBase, blocks: List[AnimaBlockWeights],
    cos: Tensor, sin: Tensor,
    B: Int, D: Int, JOINT: Int, F: Int, IN_PATCH: Int, OUT_PATCH: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaStackForward:
    var num_blocks = len(blocks)

    # ── patch embed: x_emb = linear(patches, x_embedder.proj.1.weight)  [BS,D] ──
    var x_emb = _linear_wdev(patches, base.x_embed[], B * S_IMG, IN_PATCH, ctx)

    # ── 28-block stack (per-block recompute contract: save only block inputs) ──
    # The block takes t_cond RAW and silus it internally (verified block API).
    var x = x_emb.copy()                    # host [BS, D] == [B,S_img,D]
    var blk_x_in = List[TArc]()
    for bi in range(num_blocks):
        blk_x_in.append(TArc(_t(x.copy(), [B, S_IMG, D], ctx)))
        var fwd = anima_block_forward[H, Dh, S_IMG, S_TXT](
            x.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
            blocks[bi], cos, sin, B, D, JOINT, F, eps, ctx,
        )
        x = fwd.out.copy()                  # [B,S_img,D] (block returns F32 host)

    # ── final layer ──
    var x_final_t = _t(x.copy(), [B, S_IMG, D], ctx)
    # final_adaln_modulation: silu(t_cond) @ fl_mod1 @ fl_mod2 + base_adaln[:4096]
    # (anima.rs:321 final_adaln_modulation calls t_cond.silu()).
    var tc_t = _t(t_cond.copy(), [B, ANIMA_HIDDEN], ctx)
    var ts_t = silu(tc_t, ctx)
    var fl_h = linear(ts_t, base.fl_mod1[], Optional[Tensor](None), ctx)        # [B,256]
    var fl_modout = linear(fl_h, base.fl_mod2[], Optional[Tensor](None), ctx)   # [B,4096]
    var base_t = _t(base_adaln.copy(), [B, 3 * D], ctx)
    var base_half = slice(base_t, 1, 0, 2 * D, ctx)                             # [B,4096]
    var fl_added = add(fl_modout, base_half, ctx)
    var fl_shift = slice(fl_added, 1, 0, D, ctx)
    var fl_scale = slice(fl_added, 1, D, D, ctx)
    # apply_adaln: (1+scale)*LayerNorm_noaffine(x_final) + shift
    var ln_ones = _t(_ones(D), [D], ctx)
    var ln_zeros = _t(_zeros(D), [D], ctx)
    var fl_ln = layer_norm(x_final_t, ln_ones, ln_zeros, eps, ctx)              # [B,S,D]
    var s3 = List[Int](); s3.append(B); s3.append(1); s3.append(D)
    var scale3 = reshape(fl_scale, s3.copy(), ctx)
    var shift3 = reshape(fl_shift, s3.copy(), ctx)
    var one = _t(_ones(B * D), [B, 1, D], ctx)
    var factor = add(scale3, one, ctx)
    var fl_scaled = mul(fl_ln, factor, ctx)
    var fl_xmod = add(fl_scaled, shift3, ctx)                                   # [B,S,D]
    # final linear (2048 -> 64, no bias) -> patches [BS, 64]
    var fl_xmod_2d = reshape(fl_xmod, _row2(B * S_IMG, D), ctx)
    var out = linear(fl_xmod_2d, base.fl_lin[], Optional[Tensor](None), ctx).to_host(ctx)

    return AnimaStackForward(
        out^, x_emb.copy(), blk_x_in^,
        TArc(x_final_t^), TArc(fl_ln^),
        TArc(fl_h^), TArc(fl_shift^), TArc(fl_scale^), TArc(fl_xmod^),
    )


def _row2(a: Int, b: Int) -> List[Int]:
    var o = List[Int](); o.append(a); o.append(b); return o^


# ══════════════════════════════════════════════════════════════════════════════
# FULL BACKWARD (full-depth; per-block recompute to bound memory)
# ══════════════════════════════════════════════════════════════════════════════
# d_out: upstream grad of the stack output patches [B*S_img, 64] (host).
def anima_stack_backward[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    d_out: List[Float32],
    patches: List[Float32],
    t_cond: List[Float32], base_adaln: List[Float32], context: List[Float32],
    base: AnimaStackBase, blocks: List[AnimaBlockWeights],
    cos: Tensor, sin: Tensor,
    saved: AnimaStackForward,
    B: Int, D: Int, JOINT: Int, F: Int, IN_PATCH: Int, OUT_PATCH: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaStackGrads:
    var num_blocks = len(blocks)
    var ln_ones = _t(_ones(D), [D], ctx)

    # accumulators (shared across all blocks + final layer)
    var d_t_silu_acc = _zeros(B * ANIMA_HIDDEN)
    var d_base_adaln_acc = _zeros(B * 3 * D)

    # ── final-layer backward ──
    # out = linear(fl_xmod, fl_lin)  [BS,64] ; fl_xmod [BS,D]
    var fl_xmod_2d = reshape(saved.fl_xmod[], _row2(B * S_IMG, D), ctx)
    var lbf = linear_backward(
        _t(d_out, [B * S_IMG, OUT_PATCH], ctx), fl_xmod_2d, base.fl_lin[],
        B * S_IMG, D, OUT_PATCH, ctx,
    )
    var d_fl_lin = lbf.d_w.to_host(ctx)
    # fl_xmod = (1+scale)*LN(x_final) + shift  -> modulate_backward
    var d_fl_xmod_3d = reshape(lbf.d_x, _sh3(B, S_IMG, D), ctx)
    var mbf = modulate_backward(d_fl_xmod_3d, saved.fl_ln[], saved.fl_scale[], ctx)
    # mbf.d_x = d_ln_x ; mbf.d_scale = d_fl_scale [B,D] ; mbf.d_shift = d_fl_shift [B,D]
    var d_x_final = layer_norm_backward_dx(mbf.d_x, saved.x_final[], ln_ones, eps, ctx)
    var d_x = d_x_final.to_host(ctx)        # [B,S_img,D] grad of last block output

    # final-layer modulation chain backward:
    #   fl_added = fl_modout + base_adaln[:4096]; chunk2 -> (shift,scale)
    #   d_fl_added = concat(d_shift, d_scale) [B,4096]
    var d_fl_added = concat(1, ctx, mbf.d_shift, mbf.d_scale)   # [B,4096]
    # base_adaln[:4096] grad: scatter d_fl_added into the first 4096 of [B,6144]
    var d_base_fl = _scatter_first(d_fl_added.to_host(ctx), B, 2 * D, 3 * D)
    d_base_adaln_acc = _add_lists(d_base_adaln_acc, d_base_fl)
    # fl_modout = linear(fl_h, fl_mod2);  fl_h = linear(silu(t_cond), fl_mod1).
    # The final-layer mod1 input activation is silu(t_cond) (forward applies silu
    # internally — anima.rs:321). fl_lb1.d_x is the grad at the silu OUTPUT, which
    # is exactly the final layer's contribution to d_t_silu (the oracle's retained
    # t_silu.grad). The block returns the same d_t_silu convention, so they SUM.
    var fl_ts = silu(_t(t_cond.copy(), [B, ANIMA_HIDDEN], ctx), ctx)
    var fl_lb2 = linear_backward(d_fl_added, saved.fl_mod_h[], base.fl_mod2[], B, 256, 2 * D, ctx)
    var d_fl_mod2 = fl_lb2.d_w.to_host(ctx)
    var fl_lb1 = linear_backward(fl_lb2.d_x, fl_ts,
                                 base.fl_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    var d_fl_mod1 = fl_lb1.d_w.to_host(ctx)
    d_t_silu_acc = _add_lists(d_t_silu_acc, fl_lb1.d_x.to_host(ctx))

    # ── 28-block backward (REVERSE; per-block recompute) ──
    var blk_grads_rev = List[AnimaBlockGrads]()
    var bi = num_blocks - 1
    while bi >= 0:
        var refwd = anima_block_forward[H, Dh, S_IMG, S_TXT](
            saved.blk_x_in[bi][].to_host(ctx),
            t_cond.copy(), base_adaln.copy(), context.copy(),
            blocks[bi], cos, sin, B, D, JOINT, F, eps, ctx,
        )
        var bg = anima_block_backward[H, Dh, S_IMG, S_TXT](
            d_x.copy(), refwd.saved, blocks[bi], cos, sin, B, D, JOINT, F, eps, ctx,
        )
        d_x = bg.d_x.copy()                                 # INTER-BLOCK HANDOFF
        d_t_silu_acc = _add_lists(d_t_silu_acc, bg.d_t_silu)
        # base_adaln grad: each block adds base_adaln into ALL 3 mod chains; the
        # block's d_t_silu is the only shared-grad it returns. The base_adaln grad
        # equals (d_shift+d_scale+d_gate) of each sub-block, which the block folds
        # into its mod-chain. The block does NOT separately return d_base_adaln; for
        # the LoRA-DiT path base_adaln is a frozen function of t_cond (the E-link),
        # so we accumulate ONLY d_t_silu here (the genuinely-shared trained quantity).
        blk_grads_rev.append(bg^)
        bi -= 1
    # un-reverse into forward order
    var blk_grads = List[AnimaBlockGrads]()
    var j = len(blk_grads_rev) - 1
    while j >= 0:
        blk_grads.append(blk_grads_rev[j].copy())
        j -= 1

    # ── patch-embed backward: x_emb = linear(patches, x_embed) ──
    var lbpe = linear_backward(
        _t(d_x, [B * S_IMG, D], ctx), _t(patches.copy(), [B * S_IMG, IN_PATCH], ctx),
        base.x_embed[], B * S_IMG, IN_PATCH, D, ctx,
    )
    var d_patches = lbpe.d_x.to_host(ctx)
    var d_x_embed = lbpe.d_w.to_host(ctx)

    return AnimaStackGrads(
        d_patches^, blk_grads^,
        d_t_silu_acc^, d_base_adaln_acc^,
        d_x_embed^, d_fl_lin^, d_fl_mod1^, d_fl_mod2^,
    )


def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var o = List[Int](); o.append(a); o.append(b); o.append(c); return o^


# scatter a [B, lo:hi-width] grad into the first (hi-lo)-wide slot of a [B, full]
# zero vector (the base_adaln[:4096] -> [B,6144] embedding for the final layer).
def _scatter_first(d: List[Float32], B: Int, width: Int, full: Int) -> List[Float32]:
    var o = List[Float32]()
    for b in range(B):
        for i in range(full):
            if i < width:
                o.append(d[b * width + i])
            else:
                o.append(0.0)
    return o^
