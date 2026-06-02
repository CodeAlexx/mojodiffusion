# serenitymojo/models/ernie/ernie_stack.mojo
#
# ERNIE-Image FULL DiT STACK: forward (saving ckpt-inputs) + full-depth backward
# (training), COMPOSING the already-parity-verified single-stream ERNIE block
# (models/ernie/block.mojo, 19/19 cos>=0.99999 vs torch) into the complete model.
# This file COMPOSES; it rebuilds NOTHING. Mirrors models/klein/klein_stack.mojo
# (the PROVEN stack pattern: per-block recompute in backward to bound memory; the
# d_x -> d_y inter-block handoff chained in REVERSE).
#
# ERNIE differs from Klein in ways that matter to the composition:
#   * SINGLE stream only (no double-stream). One block kind, looped num_single×.
#   * IMAGE-FIRST concat: x = concat(1, img, txt). The final layer reads the
#     FIRST n_img rows (narrow(1,0,n_img)) — Klein reads the img rows AFTER the
#     txt rows (txt-first). So the single-stack backward seed puts d_img_out on
#     the FIRST n_img rows and zeros on the trailing text rows.
#   * SHARED AdaLN: one modulation (6 chunks [D]) computed once and broadcast to
#     EVERY block. => the 6 mod-vec grads SUM across all num_single blocks. This
#     is the genuinely-new composition detail (gated explicitly). Klein has the
#     same shared-modulation contract (its single_mod is shared across all single
#     blocks); ERNIE's 6-chunk msa/mlp packing is what we accumulate here.
#   * FINAL layer modulation is computed from `c` (the timestep embedding) via
#     final_norm.linear -> chunk(2) = [f_scale, f_shift] (ernie_image.rs:549-555).
#     x_out = layer_norm(x) * (1 + f_scale) + f_shift = modulate(layer_norm(x),
#     f_scale, f_shift). layer_norm here is NON-learnable (weight 1, bias 0,
#     eps 1e-6). Then final_linear (+ bias) -> patches; narrow img rows; unpatchify.
#
# SCOPE (mirrors Klein stack): BASE forward+backward, NO LoRA wiring (E4). The
#   shared-AdaLN vectors + final f_scale/f_shift are passed in PRECOMPUTED (the
#   AdaLN MLP / final_norm.linear / time MLP backprop link is the train-loop phase
#   E5, explicitly deferred — exactly as klein_stack defers the modulation-MLP
#   link). The 6 shared mod-vec grads ARE produced and SUMMED across blocks and
#   RETURNED (so E5 can run one linear_backward into adaLN_modulation.1 at step
#   end). Input-projection (patch/text) + final-layer weight grads are computed
#   and returned for completeness.
#
# NOTE: the stack carries the inter-block stream as a HOST List[Float32] [S,D]
#   (the same boundary contract block.mojo's forward/backward use — out is host,
#   d_out is host). Per-block recompute keeps peak device memory at ~one block's
#   activation footprint + the resident weights + rope tables, which is what fits
#   36 blocks at real dims (D=4096, S=4352) in 24 GB.
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
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import layer_norm_backward, LayerNormBackward
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.ernie.weights import (
    ErnieBlockWeights, ErnieStackBase, load_ernie_block_weights,
)
from serenitymojo.models.ernie.block import (
    ErnieModVecs, ErnieBlockSaved, ErnieBlockGrads,
    ernie_block_forward, ernie_block_backward,
)


comptime TArc = ArcPointer[Tensor]


# ── host helpers (boundary only) ─────────────────────────────────────────────
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


# linear with a RESIDENT device weight (borrowed), no bias.
def _linear_wdev(
    x_h: List[Float32], w_t: Tensor,
    rows: Int, kin: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var no_bias = Optional[Tensor](None)
    return linear(_t(x_h, [rows, kin], ctx), w_t, no_bias^, ctx).to_host(ctx)


# linear with a RESIDENT device weight + RESIDENT bias (both borrowed). The bias
# is cloned into a fresh Tensor for the Optional (Tensor is move-only; the linear
# consumes the Optional). Small bias ([D]/[out_ch]) so the clone is cheap.
def _linear_wdev_bias(
    x_h: List[Float32], w_t: Tensor, b_t: Tensor,
    rows: Int, kin: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var bias = Optional[Tensor](b_t.clone(ctx))
    return linear(_t(x_h, [rows, kin], ctx), w_t, bias^, ctx).to_host(ctx)


# concat(axis=1) of [N_IMG,D] then [N_TXT,D] -> [S,D] (IMAGE FIRST). Row-major
# [rows,D] lists concatenate along the sequence (row) axis by plain append.
def _concat_img_txt(img: List[Float32], txt: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(img)):
        o.append(img[i])
    for i in range(len(txt)):
        o.append(txt[i])
    return o^


# inverse: split [S,D] back into [N_IMG,D] (FIRST) and [N_TXT,D] (rest).
def _split_img_txt(x: List[Float32], n_img: Int, n_txt: Int, d: Int) -> List[List[Float32]]:
    var img = List[Float32]()
    var txt = List[Float32]()
    var cut = n_img * d
    for i in range(cut):
        img.append(x[i])
    for i in range(cut, (n_img + n_txt) * d):
        txt.append(x[i])
    var o = List[List[Float32]]()
    o.append(img^)
    o.append(txt^)
    return o^


# pack one block's 6 shared-AdaLN mod-vec grads into [6D] in chunk order
# shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp (matches the AdaLN
# chunk order in ernie_image.rs:530-535 and _adaln_chunk in ernie_image.mojo).
def _modvec6(g: ErnieBlockGrads) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(g.d_shift_msa)):
        o.append(g.d_shift_msa[i])
    for i in range(len(g.d_scale_msa)):
        o.append(g.d_scale_msa[i])
    for i in range(len(g.d_gate_msa)):
        o.append(g.d_gate_msa[i])
    for i in range(len(g.d_shift_mlp)):
        o.append(g.d_shift_mlp[i])
    for i in range(len(g.d_scale_mlp)):
        o.append(g.d_scale_mlp[i])
    for i in range(len(g.d_gate_mlp)):
        o.append(g.d_gate_mlp[i])
    return o^


# ── forward result: out + checkpoint inputs (block inputs) + final-layer acts ─
# Per the checkpoint contract we retain ONLY each block's input [S,D] (not its
# full saved activations); backward RE-RUNS each block's forward to regenerate
# `saved`, then runs the verified per-block backward.
struct ErnieStackForward(Movable):
    var out: List[Float32]          # [N_IMG, out_ch] (host)
    var img_in_act: List[Float32]   # [N_IMG, D]   img = patch_embed(img_tokens)
    var txt_in_act: List[Float32]   # [N_TXT, D]   txt = text_proj(txt_tokens)
    var blk_x_in: List[TArc]        # num_layers x [S,D] (checkpoint inputs)
    var x_final: TArc               # [S,D] output of the last block (pre final layer)
    var ln_x: TArc                  # [S,D] layer_norm(x_final) (non-learnable)

    def __init__(
        out self,
        var out: List[Float32],
        var img_in_act: List[Float32], var txt_in_act: List[Float32],
        var blk_x_in: List[TArc],
        var x_final: TArc, var ln_x: TArc,
    ):
        self.out = out^
        self.img_in_act = img_in_act^
        self.txt_in_act = txt_in_act^
        self.blk_x_in = blk_x_in^
        self.x_final = x_final^
        self.ln_x = ln_x^


# ── backward result: token grads + per-block grads + summed shared mod grads ──
struct ErnieStackGrads(Movable):
    var d_img_tokens: List[Float32]   # [N_IMG, in_ch]
    var d_txt_tokens: List[Float32]   # [N_TXT, text_in_dim]
    var blk_grads: List[ErnieBlockGrads]
    # base-weight grads (optional to consume; computed for completeness)
    var d_patch_w: List[Float32]      # [D, in_ch]
    var d_text_proj: List[Float32]    # [D, text_in_dim]
    var d_final_lin: List[Float32]    # [out_ch, D]
    # final-layer modulation grads (wrt the PRECOMPUTED f_scale/f_shift [D])
    var d_f_scale: List[Float32]      # [D]
    var d_f_shift: List[Float32]      # [D]
    # SUMMED-across-blocks shared-AdaLN mod-vec grads [6D] (the composition
    # detail). E5 backprops these into adaLN_modulation.1 with one linear_backward.
    var d_shared_mod: List[Float32]   # [6D]

    def __init__(
        out self,
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
        var blk_grads: List[ErnieBlockGrads],
        var d_patch_w: List[Float32], var d_text_proj: List[Float32],
        var d_final_lin: List[Float32],
        var d_f_scale: List[Float32], var d_f_shift: List[Float32],
        var d_shared_mod: List[Float32],
    ):
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^
        self.blk_grads = blk_grads^
        self.d_patch_w = d_patch_w^
        self.d_text_proj = d_text_proj^
        self.d_final_lin = d_final_lin^
        self.d_f_scale = d_f_scale^
        self.d_f_shift = d_f_shift^
        self.d_shared_mod = d_shared_mod^


# ── FULL FORWARD (checkpoint inputs only retained) ───────────────────────────
# img_tokens: [N_IMG, in_ch] ; txt_tokens: [N_TXT, text_in_dim] (the Mistral
#   hidden states; text_proj maps text_in_dim -> D).
# blocks: num_single ErnieBlockWeights. mv: the SHARED modulation (6 chunks [D]),
#   computed once by the caller (E5 builds it from silu(c)@adaLN_modulation). cos/sin:
#   FULL-WIDTH [S*H, Dh] half-split rope tables (resident, REAL 3-axis layout).
# f_scale/f_shift: the final-layer modulation [D] (from c@final_norm.linear chunk2).
def ernie_stack_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase,
    blocks: List[ErnieBlockWeights], mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieStackForward:
    var num_layers = len(blocks)

    # input projections (resident weights borrowed; biases borrowed):
    #   img = patch_embed(img_tokens)  = linear(img_tokens, patch_w) + patch_b
    #   txt = text_proj(txt_tokens)    = linear(txt_tokens, text_proj)  (no bias)
    var img = _linear_wdev_bias(
        img_tokens, base.patch_w[], base.patch_b[], N_IMG, in_ch, ctx
    )                                                            # [N_IMG, D]
    var txt = _linear_wdev(
        txt_tokens, base.text_proj[], N_TXT, text_in, ctx
    )                                                            # [N_TXT, D]

    # concat IMAGE FIRST -> [S,D]
    var x = _concat_img_txt(img, txt)

    # ── single-stream stack (shared modulation broadcast to every block) ──
    var blk_x_in = List[TArc]()
    for bi in range(num_layers):
        blk_x_in.append(TArc(_t(x.copy(), [S, D], ctx)))
        var fwd = ernie_block_forward[H, Dh, S](
            x.copy(), blocks[bi], mv, cos, sin, D, F, eps, ctx,
        )
        x = fwd.out.copy()

    # ── final layer ──
    # ln_x = layer_norm(x) (NON-learnable: weight 1, bias 0)
    var ln_x = layer_norm(
        _t(x.copy(), [S, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), eps, ctx,
    ).to_host(ctx)
    # x_out = modulate(ln_x, f_scale, f_shift) = (1 + f_scale)*ln_x + f_shift
    var x_out = modulate(
        _t(ln_x.copy(), [S, D], ctx),
        _t(f_scale.copy(), [D], ctx), _t(f_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    # patches = linear(x_out, final_linear) + final_lin_b  -> [S, out_ch]
    var patches = _linear_wdev_bias(
        x_out, base.final_lin_w[], base.final_lin_b[], S, D, ctx
    )                                                            # [S, out_ch]
    # img patches = first N_IMG rows (narrow(1,0,n_img)); unpatchify is identity
    # at patch_size=1 (the gate compares the [N_IMG,out_ch] patches directly).
    var parts = _split_img_txt(patches, N_IMG, N_TXT, out_ch)
    var out = parts[0].copy()                                    # [N_IMG, out_ch]

    return ErnieStackForward(
        out^, img.copy(), txt.copy(), blk_x_in^,
        TArc(_t(x^, [S, D], ctx)), TArc(_t(ln_x^, [S, D], ctx)),
    )


# ── FULL BACKWARD (full-depth; per-block recompute to bound memory) ──────────
# d_out: upstream grad of the stack output [N_IMG, out_ch].
def ernie_stack_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase,
    blocks: List[ErnieBlockWeights], mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    saved: ErnieStackForward,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieStackGrads:
    var num_layers = len(blocks)

    # ── final-layer backward ──
    # out is the FIRST N_IMG rows of `patches`; the text rows of patches are not
    # read by the loss -> d_patches has d_out on the first N_IMG rows, 0 on text.
    var d_patches = _concat_img_txt(d_out, _zeros(N_TXT * out_ch))   # [S, out_ch]

    # patches = linear(x_out, final_linear)   W [out_ch, D]
    var lbf = linear_backward(
        _t(d_patches, [S, out_ch], ctx), _t(saved_x_out(saved, f_scale, f_shift, D, S, ctx), [S, D], ctx),
        base.final_lin_w[],
        S, D, out_ch, ctx,
    )
    var d_x_out = lbf.d_x.to_host(ctx)
    var d_final_lin = lbf.d_w.to_host(ctx)

    # x_out = modulate(ln_x, f_scale, f_shift)
    var mbf = modulate_backward(
        _t(d_x_out, [S, D], ctx), saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_x = mbf.d_x.to_host(ctx)
    var d_f_scale = mbf.d_scale.to_host(ctx)
    var d_f_shift = mbf.d_shift.to_host(ctx)

    # ln_x = layer_norm(x_final, 1, 0) (non-learnable)
    var lnbf = layer_norm_backward(
        _t(d_ln_x, [S, D], ctx), saved.x_final[], _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_x = lnbf.d_x.to_host(ctx)   # [S,D] grad of the last block's output

    # ── single-stream backward (REVERSE; per-block recompute) ──
    var blk_grads_rev = List[ErnieBlockGrads]()
    var d_shared_mod = _zeros(6 * D)
    var bi = num_layers - 1
    while bi >= 0:
        # recompute this block's forward from its saved input to regenerate `saved`
        var refwd = ernie_block_forward[H, Dh, S](
            saved.blk_x_in[bi][].to_host(ctx), blocks[bi], mv, cos, sin, D, F, eps, ctx,
        )
        var bg = ernie_block_backward[H, Dh, S](
            d_x.copy(), blocks[bi], mv, refwd.saved, cos, sin, D, F, eps, ctx,
        )
        d_x = bg.d_x.copy()                       # INTER-BLOCK HANDOFF: d_x -> d_y
        # accumulate the SHARED mod-vec grads (6 chunks) across all blocks
        d_shared_mod = _add_lists(d_shared_mod, _modvec6(bg))
        blk_grads_rev.append(bg^)
        bi -= 1
    # un-reverse into forward order
    var blk_grads = List[ErnieBlockGrads]()
    var j = len(blk_grads_rev) - 1
    while j >= 0:
        blk_grads.append(blk_grads_rev[j].copy())
        j -= 1

    # double->input seam: split d_x [S,D] back into d_img, d_txt (IMAGE FIRST).
    var seam = _split_img_txt(d_x, N_IMG, N_TXT, D)
    var d_img = seam[0].copy()   # [N_IMG,D]
    var d_txt = seam[1].copy()   # [N_TXT,D]

    # ── input-projection backward ──
    # img = linear(img_tokens, patch_w):  d_img_tokens, d_patch_w
    var lbi = linear_backward(
        _t(d_img, [N_IMG, D], ctx), _t(img_tokens, [N_IMG, in_ch], ctx),
        base.patch_w[],
        N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var d_patch_w = lbi.d_w.to_host(ctx)

    # txt = linear(txt_tokens, text_proj):  d_txt_tokens, d_text_proj
    var lbt = linear_backward(
        _t(d_txt, [N_TXT, D], ctx), _t(txt_tokens, [N_TXT, text_in], ctx),
        base.text_proj[],
        N_TXT, text_in, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)
    var d_text_proj = lbt.d_w.to_host(ctx)

    return ErnieStackGrads(
        d_img_tokens^, d_txt_tokens^, blk_grads^,
        d_patch_w^, d_text_proj^, d_final_lin^,
        d_f_scale^, d_f_shift^, d_shared_mod^,
    )


# recompute x_out (the final_linear input) from saved.ln_x for the final-layer
# linear backward (the activation the GEMM-weight grad needs).
def saved_x_out(
    saved: ErnieStackForward, f_scale: List[Float32], f_shift: List[Float32],
    D: Int, S: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    return modulate(
        saved.ln_x[], _t(f_scale.copy(), [D], ctx), _t(f_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMED (block-offload) STACK — for the 24GB real-depth budget.
#
# Full F32 residency of all 36 blocks is ~31 GB (4×D² + 3×F·D F32 weights per
# block × 36) and does NOT fit a 24 GB 3090. The per-block recompute bounds the
# ACTIVATION footprint but not the resident WEIGHT footprint. These streamed
# variants load each block's weights from the sharded checkpoint ON DEMAND
# (load → use → drop), bounding RESIDENT weight memory to ~one block (~0.9 GB
# F32). This mirrors the Rust ErnieImageSwapped/BlockOffloader contract (block
# weights swapped in per layer). The math is IDENTICAL to the resident stack
# (same ernie_block_forward/backward, same composition) — only WHERE the weights
# live differs. Proven equivalent by the resident composition gate (stack_parity).
#
# The streamed backward returns a LITE result (token grads + summed shared-AdaLN
# grad + base grads + the deepest/shallowest per-block grads), accumulating
# finiteness on the fly, so neither host nor device holds all 36 blocks' grads.

struct ErnieStackGradsLite(Movable):
    var d_img_tokens: List[Float32]   # [N_IMG, in_ch]
    var d_txt_tokens: List[Float32]   # [N_TXT, text_in]
    var d_patch_w: List[Float32]      # [D, in_ch]
    var d_text_proj: List[Float32]    # [D, text_in]
    var d_final_lin: List[Float32]    # [out_ch, D]
    var d_f_scale: List[Float32]      # [D]
    var d_f_shift: List[Float32]      # [D]
    var d_shared_mod: List[Float32]   # [6D]
    # probe-block grads (deepest layer L-1 = first reversed, shallowest 0)
    var d_wq_deep: List[Float32]
    var d_wdown_deep: List[Float32]
    var d_wq_shallow: List[Float32]
    var d_wdown_shallow: List[Float32]
    # non-finite counter accumulated across ALL per-block grads of every block
    var nonfinite_block_grads: Int

    def __init__(
        out self,
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
        var d_patch_w: List[Float32], var d_text_proj: List[Float32],
        var d_final_lin: List[Float32],
        var d_f_scale: List[Float32], var d_f_shift: List[Float32],
        var d_shared_mod: List[Float32],
        var d_wq_deep: List[Float32], var d_wdown_deep: List[Float32],
        var d_wq_shallow: List[Float32], var d_wdown_shallow: List[Float32],
        nonfinite_block_grads: Int,
    ):
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^
        self.d_patch_w = d_patch_w^
        self.d_text_proj = d_text_proj^
        self.d_final_lin = d_final_lin^
        self.d_f_scale = d_f_scale^
        self.d_f_shift = d_f_shift^
        self.d_shared_mod = d_shared_mod^
        self.d_wq_deep = d_wq_deep^
        self.d_wdown_deep = d_wdown_deep^
        self.d_wq_shallow = d_wq_shallow^
        self.d_wdown_shallow = d_wdown_shallow^
        self.nonfinite_block_grads = nonfinite_block_grads


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        # NaN: x != x ; Inf: x*0 != 0 (or magnitude beyond finite range)
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# STREAMED forward: loads each block from `st` on demand. Identical math to
# ernie_stack_forward; weights never all-resident. num_layers from the caller.
def ernie_stack_forward_streamed[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, st: ShardedSafeTensors,
    num_layers: Int, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieStackForward:
    var img = _linear_wdev_bias(
        img_tokens, base.patch_w[], base.patch_b[], N_IMG, in_ch, ctx
    )
    var txt = _linear_wdev(txt_tokens, base.text_proj[], N_TXT, text_in, ctx)
    var x = _concat_img_txt(img, txt)

    var blk_x_in = List[TArc]()
    for bi in range(num_layers):
        blk_x_in.append(TArc(_t(x.copy(), [S, D], ctx)))
        var w = load_ernie_block_weights(st, bi, ctx)   # swap IN
        var fwd = ernie_block_forward[H, Dh, S](
            x.copy(), w, mv, cos, sin, D, F, eps, ctx,
        )
        x = fwd.out.copy()
        # `w` (the block's resident weights) drops here -> swap OUT before next.

    var ln_x = layer_norm(
        _t(x.copy(), [S, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), eps, ctx,
    ).to_host(ctx)
    var x_out = modulate(
        _t(ln_x.copy(), [S, D], ctx),
        _t(f_scale.copy(), [D], ctx), _t(f_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var patches = _linear_wdev_bias(
        x_out, base.final_lin_w[], base.final_lin_b[], S, D, ctx
    )
    var parts = _split_img_txt(patches, N_IMG, N_TXT, out_ch)
    var out = parts[0].copy()

    return ErnieStackForward(
        out^, img.copy(), txt.copy(), blk_x_in^,
        TArc(_t(x^, [S, D], ctx)), TArc(_t(ln_x^, [S, D], ctx)),
    )


# STREAMED backward: reloads each block from `st` on demand (recompute fwd +
# verified bwd), accumulating finiteness. Returns the LITE result.
def ernie_stack_backward_streamed[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, st: ShardedSafeTensors,
    num_layers: Int, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    saved: ErnieStackForward,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieStackGradsLite:
    # ── final-layer backward (identical to the resident path) ──
    var d_patches = _concat_img_txt(d_out, _zeros(N_TXT * out_ch))
    var x_out = saved_x_out(saved, f_scale, f_shift, D, S, ctx)
    var lbf = linear_backward(
        _t(d_patches, [S, out_ch], ctx), _t(x_out, [S, D], ctx),
        base.final_lin_w[], S, D, out_ch, ctx,
    )
    var d_x_out = lbf.d_x.to_host(ctx)
    var d_final_lin = lbf.d_w.to_host(ctx)
    var mbf = modulate_backward(
        _t(d_x_out, [S, D], ctx), saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_x = mbf.d_x.to_host(ctx)
    var d_f_scale = mbf.d_scale.to_host(ctx)
    var d_f_shift = mbf.d_shift.to_host(ctx)
    var lnbf = layer_norm_backward(
        _t(d_ln_x, [S, D], ctx), saved.x_final[], _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_x = lnbf.d_x.to_host(ctx)

    # ── single-stream backward (REVERSE; per-block recompute + on-demand load) ──
    var d_shared_mod = _zeros(6 * D)
    var nonfinite_blk = 0
    var d_wq_deep = List[Float32]()
    var d_wdown_deep = List[Float32]()
    var d_wq_shallow = List[Float32]()
    var d_wdown_shallow = List[Float32]()
    var bi = num_layers - 1
    while bi >= 0:
        var w = load_ernie_block_weights(st, bi, ctx)        # swap IN
        var refwd = ernie_block_forward[H, Dh, S](
            saved.blk_x_in[bi][].to_host(ctx), w, mv, cos, sin, D, F, eps, ctx,
        )
        var bg = ernie_block_backward[H, Dh, S](
            d_x.copy(), w, mv, refwd.saved, cos, sin, D, F, eps, ctx,
        )
        d_x = bg.d_x.copy()
        d_shared_mod = _add_lists(d_shared_mod, _modvec6(bg))
        # accumulate finiteness across this block's full grad set
        nonfinite_blk += _nonfinite(bg.d_wq) + _nonfinite(bg.d_wk)
        nonfinite_blk += _nonfinite(bg.d_wv) + _nonfinite(bg.d_wo)
        nonfinite_blk += _nonfinite(bg.d_wgate) + _nonfinite(bg.d_wup)
        nonfinite_blk += _nonfinite(bg.d_wdown)
        nonfinite_blk += _nonfinite(bg.d_q_norm) + _nonfinite(bg.d_k_norm)
        nonfinite_blk += _nonfinite(bg.d_sa_norm) + _nonfinite(bg.d_mlp_norm)
        if bi == num_layers - 1:
            d_wq_deep = bg.d_wq.copy()
            d_wdown_deep = bg.d_wdown.copy()
        if bi == 0:
            d_wq_shallow = bg.d_wq.copy()
            d_wdown_shallow = bg.d_wdown.copy()
        bi -= 1
        # `w` + `bg` drop here -> resident weight + per-block grads freed.

    var seam = _split_img_txt(d_x, N_IMG, N_TXT, D)
    var d_img = seam[0].copy()
    var d_txt = seam[1].copy()

    var lbi = linear_backward(
        _t(d_img, [N_IMG, D], ctx), _t(img_tokens, [N_IMG, in_ch], ctx),
        base.patch_w[], N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var d_patch_w = lbi.d_w.to_host(ctx)
    var lbt = linear_backward(
        _t(d_txt, [N_TXT, D], ctx), _t(txt_tokens, [N_TXT, text_in], ctx),
        base.text_proj[], N_TXT, text_in, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)
    var d_text_proj = lbt.d_w.to_host(ctx)

    return ErnieStackGradsLite(
        d_img_tokens^, d_txt_tokens^, d_patch_w^, d_text_proj^, d_final_lin^,
        d_f_scale^, d_f_shift^, d_shared_mod^,
        d_wq_deep^, d_wdown_deep^, d_wq_shallow^, d_wdown_shallow^,
        nonfinite_blk,
    )
