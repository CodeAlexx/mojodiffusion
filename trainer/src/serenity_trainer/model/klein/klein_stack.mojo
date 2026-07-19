# serenitymojo/models/klein/klein_stack.mojo
#
# Klein (FLUX.2) FULL DiT STACK: forward (saving acts) + full-depth backward
# (training), COMPOSING the already-parity-verified double-stream and single-stream
# blocks into the complete Klein model. This file COMPOSES; it rebuilds NOTHING.
#
# WHAT IS ALREADY PROVEN (cos>=0.999 vs torch) AND ONLY REUSED HERE
#   * models/klein/double_block.mojo : double_block_forward / double_block_backward
#       (28/28 grads vs torch). The d_x -> d_y inter-block handoff is THAT file's.
#   * models/klein/single_block.mojo : single_block_forward / single_block_backward.
#   * training/dit_block.mojo + training/parity/stack_train_parity.mojo : the
#       PROVEN multi-block stacking contract — block i's d_x IS block i-1's d_y,
#       chained in REVERSE. This file mirrors that contract verbatim across the
#       double→single transition.
#   * training/checkpoint_block.mojo : recompute-in-backward (gradient checkpoint).
#       Here the same idea is applied PER BLOCK by NOT retaining per-block saved
#       activations across the whole forward; instead the backward RE-RUNS each
#       block's forward (cheap, one block) to regenerate its `saved`, then runs
#       that block's verified backward. That keeps peak memory at ~one block's
#       activation footprint + the resident inter-block stream tensors, which is
#       what fits 8+24 blocks at real dims in 24 GB.
#
# FORWARD GRAPH (mirrors models/dit/klein_dit.mojo `forward_full`, lines 451-519):
#   img = linear(img_tokens, img_in_w)        # [N_IMG, D]
#   txt = linear(txt_tokens, txt_in_w)        # [N_TXT, D]
#   (modulation is computed by the CALLER and passed in as ModVecs/SingleModVecs;
#    it is FROZEN base weights — see SCOPE below. img_mod/txt_mod/single_mod are
#    SHARED across every block of their kind, exactly as forward_full reuses the
#    same img_mod/txt_mod/single_mod for all blocks.)
#   for bi in range(num_double):
#       x = double_block(dbw[bi], img, txt, img_mod, txt_mod, cos, sin)
#       txt = slice(x, 1, 0, N_TXT) ; img = slice(x, 1, N_TXT, N_IMG)
#   x = concat(1, txt, img)                   # [S, D]   (txt FIRST, then img)
#   for bi in range(num_single):
#       x = single_block(sbw[bi], x, single_mod, cos, sin)
#   img_out = slice(x, 1, N_TXT, N_IMG)       # [N_IMG, D]
#   normed = modulate(layer_norm(img_out), final_scale, final_shift)
#   out = linear(normed, final_lin_w)         # [N_IMG, out_ch]
#
# In the double loop, the block returns one concat [S,D] = [txt_final | img_final];
# the next iteration's img/txt are re-sliced from it. So the inter-DOUBLE-block
# handoff carries the FULL [S,D] stream (we keep both txt and img). The
# double→single transition is exactly `concat(1, txt, img)` -> the single-block
# input x [S,D]. The single loop carries [S,D] forward unchanged.
#
# BACKWARD (reverse, the proven handoff at every seam):
#   out=linear(normed,Wf) ; normed=modulate(layer_norm(img_out),scale,shift)
#     -> final-layer backward yields d_img_out [N_IMG,D].
#   The single stack's input grad to the *last single block output* is, for the
#   img rows, d_img_out, and for the txt rows, 0 (txt_out is not read by the
#   final layer — exactly forward_full, which slices ONLY img_out). So the single
#   stack's d_x seed is concat(1, zeros[N_TXT,D], d_img_out) = [S,D].
#   Then chain single_block_backward in REVERSE: each block's d_x IS the next
#   (deeper) block's d_y — the stack_train_parity contract.
#   At the double→single seam: the single stack's final d_x [S,D] splits back into
#   d_txt [N_TXT,D] and d_img [N_IMG,D] (the inverse of concat(1,txt,img)).
#   Then chain double_block_backward in REVERSE. Each double block consumes
#   (d_img_out, d_txt_out) and returns (d_img_x, d_txt_x) — those ARE the
#   (d_img_out, d_txt_out) of the previous (deeper) double block. After the last
#   (deepest, bi=0) double block, d_img_x / d_txt_x are the grads wrt `img`/`txt`,
#   i.e. the input-projection outputs; back through img_in/txt_in linears gives
#   the input-token grads (load-bearing, asserted in the parity gate).
#
# SCOPE (per the task): BASE forward+backward, NO LoRA wiring (deferred to the
#   next phase). The modulation MLP and the input/output projections are FROZEN
#   base weights. Backward STILL flows d_x through them (img_in/txt_in/final
#   linear backward run), and their weight grads are RETURNED (computed; optional
#   to consume). The modulation-vector grads are produced per block and SUMMED
#   across blocks (the modvecs are shared) but are NOT backpropped into the
#   modulation MLP — that link is the LoRA/finetune phase, explicitly deferred.
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor move-only (return Movable structs, never
# store Tensor in a collection); host List[Float32] are the Copyable carriers;
# no-bias linear = linear(x, w, Optional[Tensor](None), ctx).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm


# TArc = the Copyable device carrier (ArcPointer[Tensor]); a copy is a refcount
# bump of the SAME device buffer (no D2D, no sync). Mirrors autograd.mojo:50.
comptime TArc = ArcPointer[Tensor]
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import layer_norm_backward, layer_norm_backward_dx, LayerNormBackward
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward

from serenity_trainer.model.klein.double_block import (
    DoubleBlockWeights, ModVecs, DoubleBlockSaved, DoubleBlockGrads,
    StreamGrads, double_block_forward, double_block_backward,
)
from serenity_trainer.model.klein.single_block import (
    SingleBlockWeights, SingleModVecs, SingleBlockSaved, SingleBlockGrads,
    single_block_forward, single_block_backward,
)


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
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


def _t_dtype(
    vals: List[Float32], var shape: List[Int], dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    return Tensor.from_host(vals, shape^, dtype, ctx)


def _linear_fwd(
    x_h: List[Float32], w_h: List[Float32],
    rows: Int, kin: Int, nout: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var no_bias = Optional[Tensor](None)
    return linear(
        _t(x_h, [rows, kin], ctx), _t(w_h, [nout, kin], ctx), no_bias^, ctx
    ).to_host(ctx)


# A2: linear with a RESIDENT device weight (borrowed, uploaded once at load).
# x still enters host (the inter-block stream contract is host List); only the
# WEIGHT upload is eliminated vs `_linear_fwd`.
def _linear_fwd_wdev(
    x_h: List[Float32], w_t: Tensor,
    rows: Int, kin: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var no_bias = Optional[Tensor](None)
    return linear(_t(x_h, [rows, kin], ctx), w_t, no_bias^, ctx).to_host(ctx)


# concat(axis=1) of [N_TXT,D] then [N_IMG,D] -> [S,D] (txt FIRST). Row-major
# [rows,D] lists concatenate along the sequence (row) axis by plain append.
def _concat_seq(txt: List[Float32], img: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(txt)):
        o.append(txt[i])
    for i in range(len(img)):
        o.append(img[i])
    return o^


# inverse: split [S,D] back into [N_TXT,D] (first) and [N_IMG,D] (rest).
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


# ── frozen base weights for the input projections + final layer ──────────────
# A2 PERF (2026-05-31): DEVICE-RESIDENT via TArc, uploaded EXACTLY ONCE at
# construction. These are FROZEN base weights read on every step's forward AND
# backward (img_in/txt_in/final_lin are the big input/output projections; the
# final adaLN shift/scale are small [D] but frozen for the whole run). Previously
# every use re-uploaded them via `_t(base.field, ...)` (from_host). Now use-sites
# pass `base.field[]` — a borrow of the SAME resident buffer.
struct KleinStackBase(Copyable, Movable):
    var img_in: TArc      # [D, in_ch]        img_in.weight
    var txt_in: TArc      # [D, txt_ch]       txt_in.weight
    var final_lin: TArc   # [out_ch, D]       final_layer.linear.weight
    var final_shift: TArc # [D]               final adaLN chunk 0
    var final_scale: TArc # [D]               final adaLN chunk 1

    def __init__(
        out self,
        var img_in: List[Float32], var txt_in: List[Float32],
        var final_lin: List[Float32],
        var final_shift: List[Float32], var final_scale: List[Float32],
        D: Int, in_ch: Int, txt_ch: Int, out_ch: Int, ctx: DeviceContext,
    ) raises:
        self.img_in = TArc(Tensor.from_host(img_in^, [D, in_ch], STDtype.F32, ctx))
        self.txt_in = TArc(Tensor.from_host(txt_in^, [D, txt_ch], STDtype.F32, ctx))
        self.final_lin = TArc(Tensor.from_host(final_lin^, [out_ch, D], STDtype.F32, ctx))
        self.final_shift = TArc(Tensor.from_host(final_shift^, [D], STDtype.F32, ctx))
        self.final_scale = TArc(Tensor.from_host(final_scale^, [D], STDtype.F32, ctx))

    def __init__(
        out self,
        var img_in: TArc, var txt_in: TArc, var final_lin: TArc,
        var final_shift: TArc, var final_scale: TArc,
    ):
        self.img_in = img_in^
        self.txt_in = txt_in^
        self.final_lin = final_lin^
        self.final_shift = final_shift^
        self.final_scale = final_scale^


# ── forward result: out + everything the backward needs to recompute/chain ───
# To keep memory bounded we DO NOT retain per-block saved activations. We retain
# only the inter-block stream tensors at each seam (the block INPUTS), which is
# the checkpoint contract: backward RE-RUNS each block's forward from its saved
# input to regenerate `saved`, then runs the verified per-block backward.
struct KleinStackForward(Copyable, Movable):
    var out: List[Float32]          # [N_IMG, out_ch]
    var img_in_act: TArc            # [N_IMG, D]   img = linear(img_tokens, img_in)
    var txt_in_act: TArc            # [N_TXT, D]   txt = linear(txt_tokens, txt_in)
    # per-double-block inputs (img,txt) — checkpoint inputs, one [.,D] pair each.
    var dbl_img_in: List[TArc]      # num_double x [N_IMG,D]
    var dbl_txt_in: List[TArc]      # num_double x [N_TXT,D]
    # per-single-block input x [S,D] — checkpoint inputs.
    var sgl_x_in: List[TArc]        # num_single x [S,D]
    var dbl_saved: List[DoubleBlockSaved]
    var sgl_saved: List[SingleBlockSaved]
    var img_out: TArc               # [N_IMG, D]   slice before final layer
    var ln_img_out: TArc            # [N_IMG, D]   layer_norm(img_out)

    def __init__(
        out self,
        var out: List[Float32],
        var img_in_act: TArc, var txt_in_act: TArc,
        var dbl_img_in: List[TArc], var dbl_txt_in: List[TArc],
        var sgl_x_in: List[TArc],
        var dbl_saved: List[DoubleBlockSaved], var sgl_saved: List[SingleBlockSaved],
        var img_out: TArc, var ln_img_out: TArc,
    ):
        self.out = out^
        self.img_in_act = img_in_act^
        self.txt_in_act = txt_in_act^
        self.dbl_img_in = dbl_img_in^
        self.dbl_txt_in = dbl_txt_in^
        self.sgl_x_in = sgl_x_in^
        self.dbl_saved = dbl_saved^
        self.sgl_saved = sgl_saved^
        self.img_out = img_out^
        self.ln_img_out = ln_img_out^


# ── backward result: input-token grads (load-bearing) + per-block grads ──────
# d_img_tokens / d_txt_tokens are dL wrt the raw token inputs (asserted by the
# parity gate — they prove the input-projection backward + the whole chain).
# dbl_grads[bi] / sgl_grads[bi] carry every per-block weight grad (and modvec
# grads, summed externally). Base-weight grads are computed and returned too.
struct KleinStackGrads(Copyable, Movable):
    var d_img_tokens: List[Float32]   # [N_IMG, in_ch]
    var d_txt_tokens: List[Float32]   # [N_TXT, txt_ch]
    var dbl_grads: List[DoubleBlockGrads]
    var sgl_grads: List[SingleBlockGrads]
    # base-weight grads (optional to consume; computed for completeness)
    var d_img_in: List[Float32]       # [D, in_ch]
    var d_txt_in: List[Float32]       # [D, txt_ch]
    var d_final_lin: List[Float32]    # [out_ch, D]
    var d_final_shift: List[Float32]  # [D]
    var d_final_scale: List[Float32]  # [D]
    # accumulated (summed-across-blocks) shared modvec grads (NOT backpropped
    # into the modulation MLP — that is the deferred LoRA/finetune link).
    var d_img_mod: List[Float32]      # [6D]  shift1,scale1,gate1,shift2,scale2,gate2
    var d_txt_mod: List[Float32]      # [6D]
    var d_single_mod: List[Float32]   # [3D]  shift,scale,gate

    def __init__(
        out self,
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
        var dbl_grads: List[DoubleBlockGrads], var sgl_grads: List[SingleBlockGrads],
        var d_img_in: List[Float32], var d_txt_in: List[Float32],
        var d_final_lin: List[Float32],
        var d_final_shift: List[Float32], var d_final_scale: List[Float32],
        var d_img_mod: List[Float32], var d_txt_mod: List[Float32],
        var d_single_mod: List[Float32],
    ):
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^
        self.dbl_grads = dbl_grads^
        self.sgl_grads = sgl_grads^
        self.d_img_in = d_img_in^
        self.d_txt_in = d_txt_in^
        self.d_final_lin = d_final_lin^
        self.d_final_shift = d_final_shift^
        self.d_final_scale = d_final_scale^
        self.d_img_mod = d_img_mod^
        self.d_txt_mod = d_txt_mod^
        self.d_single_mod = d_single_mod^


# ── FULL FORWARD (checkpoint inputs only retained) ───────────────────────────
# img_tokens: [N_IMG, in_ch] ; txt_tokens: [N_TXT, txt_ch].
# dbw: num_double DoubleBlockWeights ; sbw: num_single SingleBlockWeights.
# img_mod/txt_mod (ModVecs) and single_mod (SingleModVecs) are SHARED across
# every block of their kind. cos/sin: rope tables for the joint sequence.
# in_ch / txt_ch / out_ch: the raw token & final output channel counts.
def klein_stack_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    img_mod: ModVecs, txt_mod: ModVecs, single_mod: SingleModVecs,
    cos: List[Float32], sin: List[Float32],
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> KleinStackForward:
    var num_double = len(dbw)
    var num_single = len(sbw)

    # Resident rope tables: upload ONCE for the whole stack pass (borrowed by
    # every block instead of per-block cos.copy()/sin.copy() → from_host).
    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # input projections
    var img = _linear_fwd_wdev(img_tokens, base.img_in[], N_IMG, in_ch, ctx)   # [N_IMG,D]
    var txt = _linear_fwd_wdev(txt_tokens, base.txt_in[], N_TXT, txt_ch, ctx)  # [N_TXT,D]
    var img_in_act = TArc(_t(img.copy(), [N_IMG, D], ctx))
    var txt_in_act = TArc(_t(txt.copy(), [N_TXT, D], ctx))

    # ── double-stream stack ──
    var dbl_img_in = List[TArc]()
    var dbl_txt_in = List[TArc]()
    var dbl_saved = List[DoubleBlockSaved]()
    for bi in range(num_double):
        dbl_img_in.append(TArc(_t(img.copy(), [N_IMG, D], ctx)))
        dbl_txt_in.append(TArc(_t(txt.copy(), [N_TXT, D], ctx)))
        var fwd = double_block_forward[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt.copy(), dbw[bi], img_mod, txt_mod,
            cos_t, sin_t, D, F, eps, ctx,
        )
        dbl_saved.append(fwd.saved.copy())
        # next iteration's img/txt are the block's two stream outputs (the block
        # already returns them separated — forward_full re-slices the concat; we
        # take them directly, which is byte-identical).
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()

    # double→single transition: x = concat(1, txt, img) -> [S,D]
    var x = _concat_seq(txt, img)

    # ── single-stream stack ──
    var sgl_x_in = List[TArc]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        sgl_x_in.append(TArc(_t(x.copy(), [S, D], ctx)))
        var fwd = single_block_forward[H, Dh, S](
            x.copy(), sbw[bi], single_mod, cos_t, sin_t, D, F, eps, ctx,
        )
        sgl_saved.append(fwd.saved.copy())
        x = fwd.out.copy()

    # img_out = slice(x, 1, N_TXT, N_IMG) -> [N_IMG,D]
    var parts = _split_seq(x, N_TXT, N_IMG, D)
    var img_out = parts[1].copy()

    # final layer: normed = modulate(layer_norm(img_out), scale, shift); out = linear(normed, Wf)
    var ln_img_out = layer_norm(
        _t(img_out, [N_IMG, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), eps, ctx,
    ).to_host(ctx)
    var normed = modulate(
        _t(ln_img_out, [N_IMG, D], ctx),
        base.final_scale[], base.final_shift[], ctx,
    ).to_host(ctx)
    var out = _linear_fwd_wdev(normed, base.final_lin[], N_IMG, D, ctx)   # [N_IMG,out_ch]

    return KleinStackForward(
        out^, img_in_act^, txt_in_act^,
        dbl_img_in^, dbl_txt_in^, sgl_x_in^,
        dbl_saved^, sgl_saved^,
        TArc(_t(img_out^, [N_IMG, D], ctx)), TArc(_t(ln_img_out^, [N_IMG, D], ctx)),
    )


# ── FULL BACKWARD (full-depth; per-block recompute to bound memory) ──────────
# d_out: upstream grad of the stack output [N_IMG, out_ch].
def klein_stack_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights],
    img_mod: ModVecs, txt_mod: ModVecs, single_mod: SingleModVecs,
    cos: List[Float32], sin: List[Float32],
    saved: KleinStackForward,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> KleinStackGrads:
    var num_double = len(dbw)
    var num_single = len(sbw)

    # Resident rope tables: upload ONCE (recompute fwd + block bwd both borrow).
    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # ── final layer backward ──
    # out = linear(normed, Wf):  d_normed, d_final_lin
    var normed = modulate(
        saved.ln_img_out[],
        base.final_scale[], base.final_shift[], ctx,
    ).to_host(ctx)
    var lbf = linear_backward(
        _t(d_out, [N_IMG, out_ch], ctx), _t(normed, [N_IMG, D], ctx),
        base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)
    var d_final_lin = lbf.d_w.to_host(ctx)

    # normed = modulate(ln_img_out, final_scale, final_shift)
    var mbf = modulate_backward(
        _t(d_normed, [N_IMG, D], ctx), saved.ln_img_out[],
        base.final_scale[], ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)
    var d_final_scale = mbf.d_scale.to_host(ctx)
    var d_final_shift = mbf.d_shift.to_host(ctx)

    # ln_img_out = layer_norm(img_out, 1, 0); weight is frozen ones -> d_x-only
    var lnbf_dx = layer_norm_backward_dx(
        _t(d_ln_img_out, [N_IMG, D], ctx), saved.img_out[],
        _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_img_out = lnbf_dx.to_host(ctx)   # [N_IMG,D]

    # single stack seed: img_out feeds the img rows of the LAST single block's
    # output; txt rows are not read by the final layer -> zero. d_x = [S,D].
    var d_x = _concat_seq(_zeros(N_TXT * D), d_img_out)

    # ── single-stream backward (REVERSE; per-block recompute) ──
    var sgl_grads_rev = List[SingleBlockGrads]()
    var d_single_mod = _zeros(3 * D)
    var bi = num_single - 1
    while bi >= 0:
        var bg = single_block_backward[H, Dh, S](
            d_x.copy(), sbw[bi], single_mod, saved.sgl_saved[bi], cos_t, sin_t,
            D, F, eps, ctx,
        )
        d_x = bg.d_x.copy()                       # INTER-BLOCK HANDOFF: d_x -> d_y
        # accumulate the shared single modvec grad (shift,scale,gate)
        d_single_mod = _add_lists(
            d_single_mod,
            _concat3(bg.d_shift, bg.d_scale, bg.d_gate),
        )
        sgl_grads_rev.append(bg^)
        bi -= 1
    # un-reverse into forward order
    var sgl_grads = List[SingleBlockGrads]()
    var j = len(sgl_grads_rev) - 1
    while j >= 0:
        sgl_grads.append(sgl_grads_rev[j].copy())
        j -= 1

    # double→single seam: split d_x [S,D] back into d_txt_out, d_img_out (the
    # grads of the LAST double block's two stream outputs, txt FIRST).
    var seam = _split_seq(d_x, N_TXT, N_IMG, D)
    var d_txt_out = seam[0].copy()   # [N_TXT,D]
    var d_img_out2 = seam[1].copy()  # [N_IMG,D]

    # ── double-stream backward (REVERSE; per-block recompute) ──
    var dbl_grads_rev = List[DoubleBlockGrads]()
    var d_img_mod = _zeros(6 * D)
    var d_txt_mod = _zeros(6 * D)
    var di = num_double - 1
    var d_io = d_img_out2.copy()
    var d_to = d_txt_out.copy()
    while di >= 0:
        var bg = double_block_backward[H, Dh, N_IMG, N_TXT, S](
            d_io.copy(), d_to.copy(), dbw[di], img_mod, txt_mod, saved.dbl_saved[di],
            cos_t, sin_t, D, F, eps, ctx,
        )
        # the block's stream input grads ARE the previous (deeper) block's
        # stream OUTPUT grads — the proven inter-block handoff.
        d_io = bg.img.d_x.copy()
        d_to = bg.txt.d_x.copy()
        # accumulate shared double modvec grads (6 chunks each)
        d_img_mod = _add_lists(d_img_mod, _modvec6(bg.img))
        d_txt_mod = _add_lists(d_txt_mod, _modvec6(bg.txt))
        dbl_grads_rev.append(bg^)
        di -= 1
    var dbl_grads = List[DoubleBlockGrads]()
    var k = len(dbl_grads_rev) - 1
    while k >= 0:
        dbl_grads.append(dbl_grads_rev[k].copy())
        k -= 1

    # ── input-projection backward ──
    # img = linear(img_tokens, img_in):  d_img_tokens, d_img_in
    var lbi = linear_backward(
        _t(d_io, [N_IMG, D], ctx), _t(img_tokens, [N_IMG, in_ch], ctx),
        base.img_in[],
        N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var d_img_in = lbi.d_w.to_host(ctx)

    var lbt = linear_backward(
        _t(d_to, [N_TXT, D], ctx), _t(txt_tokens, [N_TXT, txt_ch], ctx),
        base.txt_in[],
        N_TXT, txt_ch, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)
    var d_txt_in = lbt.d_w.to_host(ctx)

    return KleinStackGrads(
        d_img_tokens^, d_txt_tokens^, dbl_grads^, sgl_grads^,
        d_img_in^, d_txt_in^, d_final_lin^, d_final_shift^, d_final_scale^,
        d_img_mod^, d_txt_mod^, d_single_mod^,
    )


# concat three [D] lists into one [3D] (single modvec grad packing).
def _concat3(a: List[Float32], b: List[Float32], c: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i])
    for i in range(len(b)):
        o.append(b[i])
    for i in range(len(c)):
        o.append(c[i])
    return o^


# pack one stream's 6 modvec grads into [6D] in the chunk order
# shift1,scale1,gate1,shift2,scale2,gate2 (matches _chunk_last in klein_dit).
def _modvec6(g: StreamGrads) -> List[Float32]:
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
