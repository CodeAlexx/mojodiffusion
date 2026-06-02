# serenitymojo/models/zimage/zimage_stack.mojo
#
# Z-IMAGE (NextDiT) FULL DiT STACK: forward (saving ckpt-inputs) + full-depth
# backward (training), COMPOSING the parity-verified Z-Image blocks
# (models/zimage/block.mojo):
#   * zimage_block_forward/_backward   — MODULATED block (main layers + noise
#     refiners), 19/19 cos>=0.99999 vs torch (Phase 1).
#   * zimage_refiner_forward/_backward — UNMODULATED block (context refiners),
#     15/15 cos>=0.99999 vs torch (Phase 2 refiner gate).
# This file COMPOSES; it rebuilds NOTHING. Mirrors models/ernie/ernie_stack.mojo
# (the PROVEN stack pattern: per-block recompute in backward to bound memory; the
# d_x -> d_y inter-block handoff chained in REVERSE).
#
# TOPOLOGY (verified against models/dit/zimage_dit.mojo `_forward_impl` — the
# parity-verified inference oracle — and inference-flame zimage_nextdit.rs
# `forward_inner`):
#   adaln    = t_embedder(timestep)                      # [1, t_dim]  (precomputed)
#   x_seq    = pad(x_embedder(patchify(x)))              # [N_IMG, D]   (precomputed input)
#   cap_seq  = pad(cap_embedder(cap_feats))              # [N_TXT, D]   (precomputed input)
#   for i in noise_refiner:   x_seq   = block(x_seq, x_rope,  mod_nr[i])   # MODULATED
#   for i in context_refiner: cap_seq = refiner(cap_seq, cap_rope)         # UNMODULATED
#   unified  = concat([x_seq, cap_seq], dim=row)         # [x, cap] order (zimage_dit.mojo:730)
#   for i in main_layers:     unified = block(unified, uni_rope, mod_main[i])  # MODULATED
#   # final layer: LayerNorm(no-affine, 1e-6) -> (1 + f_scale)*ln -> Linear
#   ln_u     = layer_norm(unified, 1, 0, 1e-6)
#   x_out    = modulate(ln_u, f_scale, 0)                # scale-only (no shift)
#   patches  = linear(x_out, final_lin) + final_lin_b
#   out      = patches[:N_IMG]                           # image rows (zimage_dit.mojo:750)
#   # (unpatchify is identity at the parity scale; the gate compares patches)
#
# SCOPE (mirrors the Ernie stack scope): BASE forward+backward, NO LoRA wiring
# (Phase 3). The embedders / t_embedder / per-block adaLN_modulation.0 MLP /
# final-layer adaLN MLP backprop links are the TRAIN-LOOP phase (deferred,
# exactly as ernie_stack defers them). This stack is passed:
#   * x_seq / cap_seq tokens PRECOMPUTED (post-embedder) — and returns the grads
#     into them (the full-chain proof; the embedder linear_backward is one
#     step the train loop runs at the boundary).
#   * per-block RAW modulation vectors PRECOMPUTED (noise refiners + main layers,
#     each its OWN 4 chunks) — and returns the per-block RAW mod-vec grads
#     (each block backprops into its own adaLN_modulation.0 at step end; Z-Image
#     mod is PER-BLOCK, unlike Ernie's shared mod, so NO summation across blocks).
#   * final-layer f_scale PRECOMPUTED (scale-only) — returns d_f_scale.
#
# Per-block recompute keeps peak device memory at ~one block's activation
# footprint + resident weights + rope tables (the Ernie/Klein contract for 24GB).
#
# Mojo 1.0.0b1: `def` not `fn`; host List[Float32] carriers; no-bias linear =
# linear(x, w, Optional[Tensor](None), ctx).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.tensor_algebra import add
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import layer_norm_backward, LayerNormBackward
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward

from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import (
    ZImageModVecs,
    ZImageBlockSaved, ZImageBlockGrads, zimage_block_forward, zimage_block_backward,
    ZImageRefinerSaved, ZImageRefinerGrads, zimage_refiner_forward, zimage_refiner_backward,
)


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
        o.append(Float32(0.0))
    return o^


def _ones(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(1.0))
    return o^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


# concat [N_IMG,D] then [N_TXT,D] -> [S,D]  (IMAGE FIRST = zimage_dit.mojo [x,cap]).
def _concat_img_cap(img: List[Float32], cap: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(img)):
        o.append(img[i])
    for i in range(len(cap)):
        o.append(cap[i])
    return o^


# inverse: split [S,D] -> [N_IMG,D] (first) and [N_TXT,D] (rest).
def _split_img_cap(x: List[Float32], n_img: Int, n_txt: Int, d: Int) -> List[List[Float32]]:
    var img = List[Float32]()
    var cap = List[Float32]()
    var cut = n_img * d
    for i in range(cut):
        img.append(x[i])
    for i in range(cut, (n_img + n_txt) * d):
        cap.append(x[i])
    var o = List[List[Float32]]()
    o.append(img^)
    o.append(cap^)
    return o^


# pack one block's 4 RAW mod-vec grads into [4D]: scale_msa|gate_msa|scale_mlp|gate_mlp
# (matches adaLN_modulation.0 chunk order in zimage_dit.mojo:428-431).
def _modvec4(g: ZImageBlockGrads) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(g.d_scale_msa)):
        o.append(g.d_scale_msa[i])
    for i in range(len(g.d_gate_msa)):
        o.append(g.d_gate_msa[i])
    for i in range(len(g.d_scale_mlp)):
        o.append(g.d_scale_mlp[i])
    for i in range(len(g.d_gate_mlp)):
        o.append(g.d_gate_mlp[i])
    return o^


# ── forward result: out + checkpoint inputs for each stream + final-layer acts ─
struct ZImageStackForward(Movable):
    var out: List[Float32]              # [N_IMG, out_ch] (host) = patches[:N_IMG]
    var x_seq_in: List[Float32]         # [N_IMG, D] (precomputed embedder output)
    var cap_seq_in: List[Float32]       # [N_TXT, D]
    # checkpoint inputs (each block's [rows,D] input)
    var nr_x_in: List[TArc]             # num_noise_refiner x [N_IMG,D]
    var cr_x_in: List[TArc]             # num_context_refiner x [N_TXT,D]
    var main_x_in: List[TArc]           # num_main x [S,D]
    var x_final: TArc                   # [S,D] output of last main block (pre final layer)
    var ln_x: TArc                      # [S,D] layer_norm(x_final) (non-learnable)

    def __init__(
        out self,
        var out: List[Float32],
        var x_seq_in: List[Float32], var cap_seq_in: List[Float32],
        var nr_x_in: List[TArc], var cr_x_in: List[TArc], var main_x_in: List[TArc],
        var x_final: TArc, var ln_x: TArc,
    ):
        self.out = out^
        self.x_seq_in = x_seq_in^
        self.cap_seq_in = cap_seq_in^
        self.nr_x_in = nr_x_in^
        self.cr_x_in = cr_x_in^
        self.main_x_in = main_x_in^
        self.x_final = x_final^
        self.ln_x = ln_x^


# ── backward result: token grads + per-block grads + final-layer grads ────────
struct ZImageStackGrads(Movable):
    var d_x_seq: List[Float32]          # [N_IMG, D]  grad into noise-refiner input (embedder out)
    var d_cap_seq: List[Float32]        # [N_TXT, D]  grad into context-refiner input (embedder out)
    var nr_grads: List[ZImageBlockGrads]    # noise refiners (modulated)
    var cr_grads: List[ZImageRefinerGrads]  # context refiners (unmodulated)
    var main_grads: List[ZImageBlockGrads]  # main layers (modulated)
    var d_final_lin: List[Float32]      # [out_ch, D]
    var d_f_scale: List[Float32]        # [D] final-layer scale-only modulation grad

    def __init__(
        out self,
        var d_x_seq: List[Float32], var d_cap_seq: List[Float32],
        var nr_grads: List[ZImageBlockGrads], var cr_grads: List[ZImageRefinerGrads],
        var main_grads: List[ZImageBlockGrads],
        var d_final_lin: List[Float32], var d_f_scale: List[Float32],
    ):
        self.d_x_seq = d_x_seq^
        self.d_cap_seq = d_cap_seq^
        self.nr_grads = nr_grads^
        self.cr_grads = cr_grads^
        self.main_grads = main_grads^
        self.d_final_lin = d_final_lin^
        self.d_f_scale = d_f_scale^


# linear with a RESIDENT device weight + RESIDENT bias (both borrowed).
def _linear_wdev_bias(
    x_h: List[Float32], w_t: Tensor, b_t: Tensor,
    rows: Int, kin: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var bias = Optional[Tensor](b_t.clone(ctx))
    return linear(_t(x_h, [rows, kin], ctx), w_t, bias^, ctx).to_host(ctx)


# ── FULL FORWARD (checkpoint inputs only retained) ───────────────────────────
# x_seq / cap_seq: PRECOMPUTED embedder outputs ([N_IMG,D] / [N_TXT,D]).
# nr_blocks / nr_mod: noise refiner weights + per-block RAW mod (MODULATED).
# cr_blocks: context refiner weights (UNMODULATED).
# main_blocks / main_mod: main-layer weights + per-block RAW mod (MODULATED).
# x_rope/cap_rope/uni_rope: HALF-WIDTH [rows*H, Dh/2] interleaved rope tables.
# f_scale: final-layer scale-only modulation [D]. final_lin_w [out_ch,D] + bias [out_ch].
def zimage_stack_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    x_seq: List[Float32], cap_seq: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    f_scale: List[Float32],
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageStackForward:
    var num_nr = len(nr_blocks)
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)

    # ── noise refiner (MODULATED) on x_seq [N_IMG,D] ──
    var nr_x_in = List[TArc]()
    var xs = x_seq.copy()
    for i in range(num_nr):
        nr_x_in.append(TArc(_t(xs.copy(), [N_IMG, D], ctx)))
        var fwd = zimage_block_forward[H, Dh, N_IMG](
            xs.copy(), nr_blocks[i], nr_mod[i], x_cos, x_sin, D, F, eps, ctx,
        )
        xs = fwd.out.copy()

    # ── context refiner (UNMODULATED) on cap_seq [N_TXT,D] ──
    var cr_x_in = List[TArc]()
    var cs = cap_seq.copy()
    for i in range(num_cr):
        cr_x_in.append(TArc(_t(cs.copy(), [N_TXT, D], ctx)))
        var fwd = zimage_refiner_forward[H, Dh, N_TXT](
            cs.copy(), cr_blocks[i], cap_cos, cap_sin, D, F, eps, ctx,
        )
        cs = fwd.out.copy()

    # ── unified = concat([x, cap]) -> [S,D] ──
    var x = _concat_img_cap(xs, cs)

    # ── main layers (MODULATED) ──
    var main_x_in = List[TArc]()
    for i in range(num_main):
        main_x_in.append(TArc(_t(x.copy(), [S, D], ctx)))
        var fwd = zimage_block_forward[H, Dh, S](
            x.copy(), main_blocks[i], main_mod[i], uni_cos, uni_sin, D, F, eps, ctx,
        )
        x = fwd.out.copy()

    # ── final layer ──
    var ln_x = layer_norm(
        _t(x.copy(), [S, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), final_eps, ctx,
    ).to_host(ctx)
    var x_out = modulate(
        _t(ln_x.copy(), [S, D], ctx),
        _t(f_scale.copy(), [D], ctx), _t(_zeros(D), [D], ctx), ctx,
    ).to_host(ctx)
    var patches = _linear_wdev_bias(x_out, final_lin_w, final_lin_b, S, D, ctx)  # [S,out_ch]
    var parts = _split_img_cap(patches, N_IMG, N_TXT, out_ch)
    var out = parts[0].copy()                                    # [N_IMG, out_ch]

    return ZImageStackForward(
        out^, x_seq.copy(), cap_seq.copy(),
        nr_x_in^, cr_x_in^, main_x_in^,
        TArc(_t(x^, [S, D], ctx)), TArc(_t(ln_x^, [S, D], ctx)),
    )


# recompute x_out (final_linear input) from saved.ln_x for the final-layer
# linear backward.
def _saved_x_out(
    saved: ZImageStackForward, f_scale: List[Float32], D: Int, S: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    return modulate(
        saved.ln_x[], _t(f_scale.copy(), [D], ctx), _t(_zeros(D), [D], ctx), ctx,
    ).to_host(ctx)


# ── FULL BACKWARD (full-depth; per-block recompute to bound memory) ──────────
def zimage_stack_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    f_scale: List[Float32],
    final_lin_w: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    saved: ZImageStackForward,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageStackGrads:
    var num_nr = len(nr_blocks)
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)

    # ── final-layer backward ──
    # out = patches[:N_IMG]; the cap rows of patches are not read -> d_patches has
    # d_out on the first N_IMG rows, 0 on cap rows.
    var d_patches = _concat_img_cap(d_out, _zeros(N_TXT * out_ch))   # [S, out_ch]
    var x_out = _saved_x_out(saved, f_scale, D, S, ctx)
    # patches = linear(x_out, final_lin)  W [out_ch, D]
    var lbf = linear_backward(
        _t(d_patches, [S, out_ch], ctx), _t(x_out, [S, D], ctx),
        final_lin_w, S, D, out_ch, ctx,
    )
    var d_x_out = lbf.d_x.to_host(ctx)
    var d_final_lin = lbf.d_w.to_host(ctx)

    # x_out = modulate(ln_x, f_scale, 0) (scale-only): d_ln_x + d_f_scale (d_shift discarded)
    var mbf = modulate_backward(
        _t(d_x_out, [S, D], ctx), saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_x = mbf.d_x.to_host(ctx)
    var d_f_scale = mbf.d_scale.to_host(ctx)

    # ln_x = layer_norm(x_final, 1, 0) (non-learnable)
    var lnbf = layer_norm_backward(
        _t(d_ln_x, [S, D], ctx), saved.x_final[], _t(_ones(D), [D], ctx), final_eps, ctx,
    )
    var d_x = lnbf.d_x.to_host(ctx)   # [S,D] grad of last main block's output

    # ── main layers backward (REVERSE; per-block recompute) ──
    var main_grads_rev = List[ZImageBlockGrads]()
    var bi = num_main - 1
    while bi >= 0:
        var refwd = zimage_block_forward[H, Dh, S](
            saved.main_x_in[bi][].to_host(ctx), main_blocks[bi], main_mod[bi],
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        var bg = zimage_block_backward[H, Dh, S](
            d_x.copy(), main_blocks[bi], main_mod[bi], refwd.saved,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        d_x = bg.d_x.copy()                       # inter-block handoff
        main_grads_rev.append(bg^)
        bi -= 1
    var main_grads = List[ZImageBlockGrads]()
    var jm = len(main_grads_rev) - 1
    while jm >= 0:
        main_grads.append(main_grads_rev[jm].copy())
        jm -= 1

    # ── unified seam: split d_x [S,D] -> d_x_img (first), d_cap (rest) ──
    var seam = _split_img_cap(d_x, N_IMG, N_TXT, D)
    var d_xs = seam[0].copy()   # [N_IMG,D] -> back through noise refiners
    var d_cs = seam[1].copy()   # [N_TXT,D] -> back through context refiners

    # ── context refiner backward (UNMODULATED; REVERSE) ──
    var cr_grads_rev = List[ZImageRefinerGrads]()
    var ci = num_cr - 1
    while ci >= 0:
        var refwd = zimage_refiner_forward[H, Dh, N_TXT](
            saved.cr_x_in[ci][].to_host(ctx), cr_blocks[ci], cap_cos, cap_sin, D, F, eps, ctx,
        )
        var bg = zimage_refiner_backward[H, Dh, N_TXT](
            d_cs.copy(), cr_blocks[ci], refwd.saved, cap_cos, cap_sin, D, F, eps, ctx,
        )
        d_cs = bg.d_x.copy()
        cr_grads_rev.append(bg^)
        ci -= 1
    var cr_grads = List[ZImageRefinerGrads]()
    var jc = len(cr_grads_rev) - 1
    while jc >= 0:
        cr_grads.append(cr_grads_rev[jc].copy())
        jc -= 1

    # ── noise refiner backward (MODULATED; REVERSE) ──
    var nr_grads_rev = List[ZImageBlockGrads]()
    var ni = num_nr - 1
    while ni >= 0:
        var refwd = zimage_block_forward[H, Dh, N_IMG](
            saved.nr_x_in[ni][].to_host(ctx), nr_blocks[ni], nr_mod[ni], x_cos, x_sin, D, F, eps, ctx,
        )
        var bg = zimage_block_backward[H, Dh, N_IMG](
            d_xs.copy(), nr_blocks[ni], nr_mod[ni], refwd.saved, x_cos, x_sin, D, F, eps, ctx,
        )
        d_xs = bg.d_x.copy()
        nr_grads_rev.append(bg^)
        ni -= 1
    var nr_grads = List[ZImageBlockGrads]()
    var jn = len(nr_grads_rev) - 1
    while jn >= 0:
        nr_grads.append(nr_grads_rev[jn].copy())
        jn -= 1

    return ZImageStackGrads(
        d_xs^, d_cs^, nr_grads^, cr_grads^, main_grads^,
        d_final_lin^, d_f_scale^,
    )
