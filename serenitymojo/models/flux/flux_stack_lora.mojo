# serenitymojo/models/flux/flux_stack_lora.mojo
#
# Flux (flux1-dev) FULL DiT STACK *WITH LoRA* on every trained block projection:
# forward (saving ckpt-inputs) + full-depth backward (training) that uses the
# parity-verified per-block LoRA variants (models/flux/lora_block.mojo), COLLECTS
# every adapter's d_A/d_B, and supports an AdamW step + a PEFT/ai-toolkit save
# across all adapters. This file COMPOSES; it rebuilds NOTHING.
#
# WHAT IS ALREADY PROVEN (cos>=0.999 vs torch) AND ONLY REUSED HERE
#   * models/flux/block.mojo : base double/single block fwd+bwd (Phase-1 GREEN).
#   * models/flux/flux_stack.mojo : the BASE full-stack composition (Phase-2
#     GREEN). THIS FILE is that file with the base per-block calls swapped for the
#     LoRA variants + LoRA-grad collection. The embed/vec chain + per-block
#     modulation projections + final layer are reused VERBATIM (frozen base).
#   * models/flux/lora_block.mojo : double/single_block_lora_forward/backward
#     (reduce to base when adapters absent; LoRA d_x summed into the proj-input
#     grad). Slot order per the OneTrainer convert_flux_lora.py target set.
#   * training/{lora_save, train_step} : LoraAdapter, _lora_adamw, save_lora_peft.
#
# CARRIER DESIGN (Tenet-2: make the right thing easy) — mirrors KleinLoraSet /
#   ErnieLoraSet. FluxLoraSet holds ONE flat List[LoraAdapter]:
#     doubles first: block bi (0..num_double-1), 12 slots/block
#       flat = bi*DBL_SLOTS_PER_BLOCK + (stream*6 + slot)
#       stream 0=img 1=txt ; slot order {to_q,to_k,to_v,proj,mlp0,mlp2}
#     singles next: block bi (0..num_single-1), 5 slots/block
#       flat = num_double*DBL_SLOTS_PER_BLOCK + bi*SGL_SLOTS + slot
#       slot order {to_q,to_k,to_v,proj_mlp,linear2}
#   The optimizer walks this flat list; the backward SCATTERS the returned per-
#   block slot d_A/d_B into the matching flat FluxLoraGrads.
#
# SCOPE: LoRA-on-block-projection training. Base weights (input/text proj, the
#   embed MLPs, per-block modulation linears, the block linears, final layer) are
#   FROZEN — their grads are computed by the base path and discarded for the
#   optimizer; only block-projection d_A/d_B are trained. This matches Klein/Ernie
#   scope. OT also LoRAs the modulation/embedder/final linears; those are
#   stack-level base linears and are the NEXT increment (noted in the plan).
#
# RESIDENCY: full F32 residency of 19+38 real-depth blocks is ~34GB (Phase-2
#   finding) and does NOT fit a 3090 — real-depth runs need block-swap offload +
#   per-block recompute (Ernie/Klein-9B strategy). The PARITY gate runs at REDUCED
#   depth (fits); composition is proven there. A streamed variant mirrors the
#   ernie_stack_lora_*_streamed path and is deferred to the runtime increment.
#
# Mojo 1.0.0b1: def not fn; Tensor move-only; host List[Float32] carriers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.models.flux.block import (
    ModVecs, SingleModVecs,
    StreamWeights,
    DoubleBlockWeights, SingleBlockWeights,
    DoubleBlockSaved, SingleBlockSaved, DoubleBlockGrads, SingleBlockGrads,
    StreamGrads,
)
from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.flux.lora_block import (
    StreamLora, DoubleBlockLora, SingleBlockLora,
    DoubleBlockLoraGrads, SingleBlockLoraGrads, StreamLoraGrads,
    DBL_STREAM_SLOTS, SGL_SLOTS,
    D_SQ, D_SK, D_SV, D_PROJ, D_MLP0, D_MLP2,
    S_SQ, S_SK, S_SV, S_PMLP, S_L2,
    double_block_lora_forward, double_block_lora_backward,
    single_block_lora_forward, single_block_lora_backward,
)
from serenitymojo.models.flux.flux_stack import (
    FluxStackBase, FluxStackForward, FluxStackGrads,
    _add_lists, _zeros, _ones, _t, _concat_seq, _split_seq, _chunk,
    _modvecs_from_flat, _single_modvecs_from_flat, _modvec6, _single_modvec3,
    _embed_vec_forward, _mod_proj, _embed_vec_backward,
)
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm_backward import layer_norm_backward
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.activation_backward import silu_backward

from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_peft, load_lora_for_resume,
)


comptime TArc = ArcPointer[Tensor]

# 6 slots per stream x 2 streams = 12 slots per double block.
comptime DBL_SLOTS_PER_BLOCK = 2 * DBL_STREAM_SLOTS


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


# ── the LoRA carrier: every trained adapter, flat-indexed ─────────────────────
struct FluxLoraSet(Copyable, Movable):
    var ad: List[LoraAdapter]
    var num_double: Int
    var num_single: Int
    var rank: Int

    def __init__(out self, var ad: List[LoraAdapter], num_double: Int, num_single: Int, rank: Int):
        self.ad = ad^
        self.num_double = num_double
        self.num_single = num_single
        self.rank = rank


def _dbl_base(bi: Int) -> Int:
    return bi * DBL_SLOTS_PER_BLOCK


def _sgl_base(set: FluxLoraSet, bi: Int) -> Int:
    return set.num_double * DBL_SLOTS_PER_BLOCK + bi * SGL_SLOTS


# Build the full LoRA set for a Flux stack. Per-projection in/out shapes:
#   double to_q/to_k/to_v : in=D out=D ; proj : in=D out=D ;
#     mlp0 : in=D out=Fmlp ; mlp2 : in=Fmlp out=D.
#   single to_q/to_k/to_v : in=D out=D ; proj_mlp : in=D out=Fmlp ;
#     linear2 : in=D+Fmlp out=D.
def build_flux_lora_set(
    num_double: Int, num_single: Int, D: Int, Fmlp: Int, rank: Int, alpha: Float32
) -> FluxLoraSet:
    var ad = List[LoraAdapter]()
    var seed = UInt64(3000)
    for _ in range(num_double):
        for _stream in range(2):   # img then txt
            ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1     # to_q
            ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1     # to_k
            ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1     # to_v
            ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1     # proj
            ad.append(make_lora_adapter(rank, alpha, D, Fmlp, seed)); seed += 1  # mlp0
            ad.append(make_lora_adapter(rank, alpha, Fmlp, D, seed)); seed += 1  # mlp2
    for _ in range(num_single):
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1         # to_q
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1         # to_k
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1         # to_v
        ad.append(make_lora_adapter(rank, alpha, D, Fmlp, seed)); seed += 1      # proj_mlp
        ad.append(make_lora_adapter(rank, alpha, D + Fmlp, D, seed)); seed += 1  # linear2
    return FluxLoraSet(ad^, num_double, num_single, rank)


def total_adapters(set: FluxLoraSet) -> Int:
    return set.num_double * DBL_SLOTS_PER_BLOCK + set.num_single * SGL_SLOTS


# accessor by flat index -> a COPY.
def flux_lora_get(set: FluxLoraSet, idx: Int) -> LoraAdapter:
    return set.ad[idx].copy()


def _opt(ad: LoraAdapter) -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](ad.copy())


def _stream_lora_for(set: FluxLoraSet, base: Int) -> StreamLora:
    return StreamLora(
        _opt(set.ad[base + D_SQ]), _opt(set.ad[base + D_SK]), _opt(set.ad[base + D_SV]),
        _opt(set.ad[base + D_PROJ]), _opt(set.ad[base + D_MLP0]), _opt(set.ad[base + D_MLP2]),
    )


def _double_lora_for(set: FluxLoraSet, bi: Int) -> DoubleBlockLora:
    var base = _dbl_base(bi)
    var img = _stream_lora_for(set, base)
    var txt = _stream_lora_for(set, base + DBL_STREAM_SLOTS)
    return DoubleBlockLora(img^, txt^)


def _single_lora_for(set: FluxLoraSet, bi: Int) -> SingleBlockLora:
    var base = _sgl_base(set, bi)
    return SingleBlockLora(
        _opt(set.ad[base + S_SQ]), _opt(set.ad[base + S_SK]), _opt(set.ad[base + S_SV]),
        _opt(set.ad[base + S_PMLP]), _opt(set.ad[base + S_L2]),
    )


# ── collected LoRA grads (flat, parallel to FluxLoraSet) ─────────────────────
struct FluxLoraGradSet(Movable):
    var d_a: List[List[Float32]]
    var d_b: List[List[Float32]]
    # load-bearing arms (prove the chain composes through the stack).
    var d_img_tokens: List[Float32]
    var d_txt_tokens: List[Float32]
    var d_vec: List[Float32]
    var d_timestep: List[Float32]
    var d_guidance: List[Float32]
    var d_vector: List[Float32]
    var nonfinite_lora_grads: Int

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
        var d_vec: List[Float32],
        var d_timestep: List[Float32], var d_guidance: List[Float32], var d_vector: List[Float32],
        nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^
        self.d_vec = d_vec^
        self.d_timestep = d_timestep^
        self.d_guidance = d_guidance^
        self.d_vector = d_vector^
        self.nonfinite_lora_grads = nonfinite_lora_grads


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# RESIDENT LoRA stack (reduced depth, for the COMPOSITION parity gate). Mirrors
# flux_stack_forward/backward exactly, swapping per-block calls for LoRA ones.
# ─────────────────────────────────────────────────────────────────────────────
def flux_stack_lora_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    timestep: List[Float32], guidance: Optional[List[Float32]], vector: List[Float32],
    base: FluxStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights], lora: FluxLoraSet,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxStackForward:
    var num_double = len(dbw)
    var num_single = len(sbw)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    var vf = _embed_vec_forward(timestep, guidance, vector, base, D, T_DIM, VEC_DIM, ctx)
    var vec = vf.vec.copy()
    var vec_silu = silu(_t(vec.copy(), [1, D], ctx), ctx).to_host(ctx)

    var bi_img = Optional[Tensor](base.img_in_b[].clone(ctx))
    var img = linear(_t(img_tokens.copy(), [N_IMG, in_ch], ctx), base.img_in[], bi_img, ctx).to_host(ctx)
    var bi_txt = Optional[Tensor](base.txt_in_b[].clone(ctx))
    var txt = linear(_t(txt_tokens.copy(), [N_TXT, txt_ch], ctx), base.txt_in[], bi_txt, ctx).to_host(ctx)

    var dbl_img_mod = List[List[Float32]]()
    var dbl_txt_mod = List[List[Float32]]()
    var dbl_saved = List[DoubleBlockSaved]()
    for bi in range(num_double):
        var im_flat = _mod_proj(vec_silu.copy(), base.dbl_mod[bi].img, 6 * D, D, ctx)
        var tm_flat = _mod_proj(vec_silu.copy(), base.dbl_mod[bi].txt, 6 * D, D, ctx)
        var im = _modvecs_from_flat(im_flat, D)
        var tm = _modvecs_from_flat(tm_flat, D)
        var bl = _double_lora_for(lora, bi)
        var fwd = double_block_lora_forward[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt.copy(), dbw[bi], im, tm, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        dbl_saved.append(fwd.saved.copy())
        dbl_img_mod.append(im_flat^)
        dbl_txt_mod.append(tm_flat^)
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()

    var x = _concat_seq(txt, img)

    var sgl_mod_flat = List[List[Float32]]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        var sm_flat = _mod_proj(vec_silu.copy(), base.sgl_mod[bi], 3 * D, D, ctx)
        var sm = _single_modvecs_from_flat(sm_flat, D)
        var bl = _single_lora_for(lora, bi)
        var fwd = single_block_lora_forward[H, Dh, S](
            x.copy(), sbw[bi], sm, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        sgl_saved.append(fwd.saved.copy())
        sgl_mod_flat.append(sm_flat^)
        x = fwd.out.copy()

    var parts = _split_seq(x, N_TXT, N_IMG, D)
    var img_out = parts[1].copy()

    var fb = Optional[Tensor](base.final_adaln_b[].clone(ctx))
    var fmods = linear(_t(vec_silu.copy(), [1, D], ctx), base.final_adaln_w[], fb, ctx).to_host(ctx)
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


def flux_stack_lora_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: FluxStackBase,
    dbw: List[DoubleBlockWeights], sbw: List[SingleBlockWeights], lora: FluxLoraSet,
    cos: List[Float32], sin: List[Float32],
    saved: FluxStackForward,
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, max_period: Float32, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxLoraGradSet:
    var num_double = len(dbw)
    var num_single = len(sbw)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    var n_adapters = total_adapters(lora)
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(n_adapters):
        d_a_flat.append(List[Float32]()); d_b_flat.append(List[Float32]())
    var nonfinite = 0

    var d_vec_silu = _zeros(D)

    # ── final layer backward (frozen base, same as flux_stack_backward) ──
    var normed = modulate(
        saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), _t(saved.final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var lbf = linear_backward(
        _t(d_out, [N_IMG, out_ch], ctx), _t(normed, [N_IMG, D], ctx), base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)
    var mbf = modulate_backward(
        _t(d_normed, [N_IMG, D], ctx), saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)
    var d_final_scale = mbf.d_scale.to_host(ctx)
    var d_final_shift = mbf.d_shift.to_host(ctx)
    var d_fmods = _concat_seq(d_final_shift, d_final_scale)
    var lb_fmods = linear_backward(
        _t(d_fmods, [1, 2 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
        base.final_adaln_w[], 1, D, 2 * D, ctx,
    )
    d_vec_silu = _add_lists(d_vec_silu, lb_fmods.d_x.to_host(ctx))

    var lnbf = layer_norm_backward(
        _t(d_ln_img_out, [N_IMG, D], ctx), saved.img_out[], _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_img_out = lnbf.d_x.to_host(ctx)

    var d_x = _concat_seq(_zeros(N_TXT * D), d_img_out)

    # ── single-stream backward (REVERSE; LoRA) ──
    var bi = num_single - 1
    while bi >= 0:
        var sm = _single_modvecs_from_flat(saved.sgl_mod_flat[bi].copy(), D)
        var bl = _single_lora_for(lora, bi)
        var bg = single_block_lora_backward[H, Dh, S](
            d_x.copy(), sbw[bi], sm, bl, saved.sgl_saved[bi], cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_x = bg.base.d_x.copy()
        # scatter single-block LoRA grads
        var sbase = _sgl_base(lora, bi)
        for s in range(SGL_SLOTS):
            d_a_flat[sbase + s] = bg.lora.d_a[s].copy()
            d_b_flat[sbase + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        # modulation.lin grad -> d_vec_silu (frozen base mod.lin, grad threaded)
        var d_sm = _single_modvec3(bg.base)
        var lb_sm = linear_backward(
            _t(d_sm, [1, 3 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
            base.sgl_mod[bi].w[], 1, D, 3 * D, ctx,
        )
        d_vec_silu = _add_lists(d_vec_silu, lb_sm.d_x.to_host(ctx))
        bi -= 1

    var seam = _split_seq(d_x, N_TXT, N_IMG, D)
    var d_to = seam[0].copy()
    var d_io = seam[1].copy()

    # ── double-stream backward (REVERSE; LoRA) ──
    var di = num_double - 1
    while di >= 0:
        var im = _modvecs_from_flat(saved.dbl_img_mod[di].copy(), D)
        var tm = _modvecs_from_flat(saved.dbl_txt_mod[di].copy(), D)
        var bl = _double_lora_for(lora, di)
        var bg = double_block_lora_backward[H, Dh, N_IMG, N_TXT, S](
            d_io.copy(), d_to.copy(), dbw[di], im, tm, bl, saved.dbl_saved[di],
            cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_io = bg.base.img.d_x.copy()
        d_to = bg.base.txt.d_x.copy()
        # scatter double-block LoRA grads (img slots then txt slots)
        var dbase = _dbl_base(di)
        for s in range(DBL_STREAM_SLOTS):
            d_a_flat[dbase + s] = bg.lora.img.d_a[s].copy()
            d_b_flat[dbase + s] = bg.lora.img.d_b[s].copy()
            d_a_flat[dbase + DBL_STREAM_SLOTS + s] = bg.lora.txt.d_a[s].copy()
            d_b_flat[dbase + DBL_STREAM_SLOTS + s] = bg.lora.txt.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.img.d_a[s]) + _nonfinite(bg.lora.img.d_b[s])
            nonfinite += _nonfinite(bg.lora.txt.d_a[s]) + _nonfinite(bg.lora.txt.d_b[s])
        # img_mod/txt_mod grads -> their mod.lin backward -> d_vec_silu
        var d_im = _modvec6(bg.base.img)
        var d_tm = _modvec6(bg.base.txt)
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
        di -= 1

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

    # ── vec backward: silu(vec) -> d_vec -> embeds ──
    var d_vec = silu_backward(_t(d_vec_silu.copy(), [1, D], ctx), _t(saved.vec.copy(), [1, D], ctx), ctx).to_host(ctx)
    var eb = _embed_vec_backward(d_vec.copy(), saved, base, base.has_guidance, D, T_DIM, VEC_DIM, max_period, ctx)

    return FluxLoraGradSet(
        d_a_flat^, d_b_flat^,
        d_img_tokens^, d_txt_tokens^, d_vec^,
        eb.d_timestep.copy(), eb.d_guidance.copy(), eb.d_vector.copy(),
        nonfinite,
    )


# ── AdamW step on EVERY adapter (reuses the proven per-adapter _lora_adamw) ───
def flux_lora_adamw_step(
    mut set: FluxLoraSet, grads: FluxLoraGradSet, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    var n = total_adapters(set)
    for i in range(n):
        var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(set.ad[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay)


# ── OT/BFL key-name scheme (convert_flux_lora.py:6-41 BFL module keys) ───────
# Saved as "<bfl_module>.lora_A.weight"/".lora_B.weight". These are the BFL
# (black-forest-labs flux) module names; OT's saver converts them to the
# diffusers names on export, but the BFL keys are the canonical in-memory target
# set and round-trip exactly through save_lora_peft/load_lora_for_resume.
def _dbl_stream_prefix(bi: Int, stream_img: Bool, slot: Int) -> String:
    var b = String("double_blocks.") + String(bi) + "."
    var s = "img" if stream_img else "txt"
    if slot == D_SQ:
        return b + s + "_attn.qkv.0"
    elif slot == D_SK:
        return b + s + "_attn.qkv.1"
    elif slot == D_SV:
        return b + s + "_attn.qkv.2"
    elif slot == D_PROJ:
        return b + s + "_attn.proj"
    elif slot == D_MLP0:
        return b + s + "_mlp.0"
    return b + s + "_mlp.2"


def _sgl_prefix(bi: Int, slot: Int) -> String:
    var b = String("single_blocks.") + String(bi) + "."
    if slot == S_SQ:
        return b + "linear1.0"
    elif slot == S_SK:
        return b + "linear1.1"
    elif slot == S_SV:
        return b + "linear1.2"
    elif slot == S_PMLP:
        return b + "linear1.3"
    return b + "linear2"


def flux_lora_prefixes(num_double: Int, num_single: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_double):
        for s in range(DBL_STREAM_SLOTS):   # img stream
            out.append(_dbl_stream_prefix(bi, True, s))
        for s in range(DBL_STREAM_SLOTS):   # txt stream
            out.append(_dbl_stream_prefix(bi, False, s))
    for bi in range(num_single):
        for s in range(SGL_SLOTS):
            out.append(_sgl_prefix(bi, s))
    return out^


def save_flux_lora(set: FluxLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var prefixes = flux_lora_prefixes(set.num_double, set.num_single)
    var named = List[NamedLora]()
    var n = total_adapters(set)
    for i in range(n):
        named.append(NamedLora(prefixes[i], set.ad[i].copy()))
    return save_lora_peft(named, path, ctx)


def load_flux_lora_resume(
    num_double: Int, num_single: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> FluxLoraSet:
    var prefixes = flux_lora_prefixes(num_double, num_single)
    var scale = alpha / Float32(rank)
    var named = load_lora_for_resume(prefixes, scale, path, ctx)   # flat order
    var ad = List[LoraAdapter]()
    for i in range(len(named)):
        ad.append(named[i].adapter.copy())
    return FluxLoraSet(ad^, num_double, num_single, rank)


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK-SWAP OFFLOAD LoRA stack — streams ONE transformer block at a time.
#
# WHY THIS PATH EXISTS (RESIDENCY note above): full F32 residency of 19+38
# real-depth blocks is ~34GB and does NOT fit a 24GB GPU. The RESIDENT stack
# (flux_stack_lora_forward/backward) holds every DoubleBlockWeights/
# SingleBlockWeights at once. The offload stack mirrors it EXACTLY — same math,
# same checkpoint contract, same LoRA-grad collection — but pulls each block's
# base weights from a `TurboPlannedLoader` (block-swap), builds the block weight
# struct, runs the ALREADY-VERIFIED per-block LoRA fwd/bwd (lora_block.mojo),
# then DROPS the block weights before the next block. The LoRA adapters + the
# checkpoint activations (host List[Float32], cheap) stay resident.
#
# CONTRACT (mirrors klein_stack_lora's offload_turbo path, lines 699-786 +
# 1457-1629): the streamed block plan is build_flux1_dev_block_plan() (19 double
# then 38 single), so block index bi (0..num_double-1) maps to double_blocks.bi
# and num_double+bi maps to single_blocks.bi — the SAME flat order the resident
# stack and the LoRA carrier use. The forward keeps the FULL FluxStackForward
# tape (per-block host saved activations) so the backward is recompute-FREE for
# the block math (it only re-streams the block WEIGHTS). This matches the
# resident stack's no-recompute backward: flux block forward already saves every
# activation it needs, so the offload backward reuses saved.dbl_saved[bi] /
# saved.sgl_saved[bi] verbatim, just re-streaming the weights to run the bwd
# arms. No double_block_lora_forward recompute is needed (unlike Klein, whose
# device-resident blocks discard most saved tensors).
#
# Tenet 1: NO new block math. The block weight struct is built from streamed
# device tensors by copying ArcPointer handles, preserving checkpoint dtype and
# avoiding D2H/H2D round-trips. Tenet 2: the offload entry points take the same
# args as the resident ones with `dbw`/`sbw` replaced by `mut loader`.
# ═════════════════════════════════════════════════════════════════════════════

# Borrow/copy one streamed block tensor as a device-resident Arc. The copied Arc
# keeps the tensor alive after the loader's active block handle is released.
def _block_tensor(block: Block, key: String) raises -> TArc:
    if not (key in block):
        raise Error(String("Flux offload block missing tensor: ") + key)
    return block[key].copy()


def _stream_weights_from_block(
    block: Block, dp: String, stream: String,
    D: Int, Fmlp: Int, Dh: Int, ctx: DeviceContext,
) raises -> StreamWeights:
    var ap = dp + String(".") + stream + String("_attn")
    var mp = dp + String(".") + stream + String("_mlp")
    return StreamWeights(
        _block_tensor(block, ap + String(".qkv.weight")),
        _block_tensor(block, ap + String(".qkv.bias")),
        _block_tensor(block, ap + String(".proj.weight")),
        _block_tensor(block, ap + String(".proj.bias")),
        _block_tensor(block, mp + String(".0.weight")),
        _block_tensor(block, mp + String(".0.bias")),
        _block_tensor(block, mp + String(".2.weight")),
        _block_tensor(block, mp + String(".2.bias")),
        _block_tensor(block, ap + String(".norm.query_norm.scale")),
        _block_tensor(block, ap + String(".norm.key_norm.scale")),
    )


def _double_weights_from_block(
    block: Block, dp: String, D: Int, Fmlp: Int, Dh: Int, ctx: DeviceContext,
) raises -> DoubleBlockWeights:
    return DoubleBlockWeights(
        _stream_weights_from_block(block, dp, String("img"), D, Fmlp, Dh, ctx),
        _stream_weights_from_block(block, dp, String("txt"), D, Fmlp, Dh, ctx),
    )


def _single_weights_from_block(
    block: Block, sp: String, D: Int, Fmlp: Int, Dh: Int, ctx: DeviceContext,
) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _block_tensor(block, sp + String(".linear1.weight")),
        _block_tensor(block, sp + String(".linear1.bias")),
        _block_tensor(block, sp + String(".linear2.weight")),
        _block_tensor(block, sp + String(".linear2.bias")),
        _block_tensor(block, sp + String(".norm.query_norm.scale")),
        _block_tensor(block, sp + String(".norm.key_norm.scale")),
    )


# ── FULL FORWARD WITH LoRA, BLOCK-SWAP OFFLOAD ───────────────────────────────
# Identical to flux_stack_lora_forward except double/single block weights are
# streamed per-block via `loader` (built from build_flux1_dev_block_plan) instead
# of held in resident `dbw`/`sbw` lists. Dh is needed to build the streamed
# weight structs (resident path derives it from the passed structs).
def flux_stack_lora_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    timestep: List[Float32], guidance: Optional[List[Float32]], vector: List[Float32],
    base: FluxStackBase,
    mut loader: TurboPlannedLoader, lora: FluxLoraSet,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxStackForward:
    var num_double = lora.num_double
    var num_single = lora.num_single

    loader.prefetch_with_ctx(0, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    var vf = _embed_vec_forward(timestep, guidance, vector, base, D, T_DIM, VEC_DIM, ctx)
    var vec = vf.vec.copy()
    var vec_silu = silu(_t(vec.copy(), [1, D], ctx), ctx).to_host(ctx)

    var bi_img = Optional[Tensor](base.img_in_b[].clone(ctx))
    var img = linear(_t(img_tokens.copy(), [N_IMG, in_ch], ctx), base.img_in[], bi_img, ctx).to_host(ctx)
    var bi_txt = Optional[Tensor](base.txt_in_b[].clone(ctx))
    var txt = linear(_t(txt_tokens.copy(), [N_TXT, txt_ch], ctx), base.txt_in[], bi_txt, ctx).to_host(ctx)

    var dbl_img_mod = List[List[Float32]]()
    var dbl_txt_mod = List[List[Float32]]()
    var dbl_saved = List[DoubleBlockSaved]()
    for bi in range(num_double):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, D, Fmlp, Dh, ctx)
        var im_flat = _mod_proj(vec_silu.copy(), base.dbl_mod[bi].img, 6 * D, D, ctx)
        var tm_flat = _mod_proj(vec_silu.copy(), base.dbl_mod[bi].txt, 6 * D, D, ctx)
        var im = _modvecs_from_flat(im_flat, D)
        var tm = _modvecs_from_flat(tm_flat, D)
        var bl = _double_lora_for(lora, bi)
        var fwd = double_block_lora_forward[H, Dh, N_IMG, N_TXT, S](
            img.copy(), txt.copy(), w, im, tm, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        dbl_saved.append(fwd.saved.copy())
        dbl_img_mod.append(im_flat^)
        dbl_txt_mod.append(tm_flat^)
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        loader.mark_active_block_done(ctx)

    var x = _concat_seq(txt, img)

    var sgl_mod_flat = List[List[Float32]]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        loader.prefetch_next_with_ctx(block_idx, ctx)
        var w = _single_weights_from_block(handle.block, handle.prefix, D, Fmlp, Dh, ctx)
        var sm_flat = _mod_proj(vec_silu.copy(), base.sgl_mod[bi], 3 * D, D, ctx)
        var sm = _single_modvecs_from_flat(sm_flat, D)
        var bl = _single_lora_for(lora, bi)
        var fwd = single_block_lora_forward[H, Dh, S](
            x.copy(), w, sm, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        sgl_saved.append(fwd.saved.copy())
        sgl_mod_flat.append(sm_flat^)
        x = fwd.out.copy()
        loader.mark_active_block_done(ctx)

    var parts = _split_seq(x, N_TXT, N_IMG, D)
    var img_out = parts[1].copy()

    var fb = Optional[Tensor](base.final_adaln_b[].clone(ctx))
    var fmods = linear(_t(vec_silu.copy(), [1, D], ctx), base.final_adaln_w[], fb, ctx).to_host(ctx)
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


# ── FULL BACKWARD WITH LoRA, BLOCK-SWAP OFFLOAD ──────────────────────────────
# Identical to flux_stack_lora_backward except block weights are streamed per
# block in REVERSE (singles last..0, then doubles last..0). Saved activations are
# reused from the forward tape (recompute-free block math); only the WEIGHTS are
# re-streamed. The plan is streamed back-to-front: prefetch the last block, then
# step down.
def flux_stack_lora_backward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: FluxStackBase,
    mut loader: TurboPlannedLoader, lora: FluxLoraSet,
    cos: List[Float32], sin: List[Float32],
    saved: FluxStackForward,
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, max_period: Float32, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxLoraGradSet:
    var num_double = lora.num_double
    var num_single = lora.num_single

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.BF16, ctx)

    var n_adapters = total_adapters(lora)
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(n_adapters):
        d_a_flat.append(List[Float32]()); d_b_flat.append(List[Float32]())
    var nonfinite = 0

    var d_vec_silu = _zeros(D)

    # ── final layer backward (frozen base, same as flux_stack_backward) ──
    var normed = modulate(
        saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), _t(saved.final_shift.copy(), [D], ctx), ctx,
    ).to_host(ctx)
    var lbf = linear_backward(
        _t(d_out, [N_IMG, out_ch], ctx), _t(normed, [N_IMG, D], ctx), base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )
    var d_normed = lbf.d_x.to_host(ctx)
    var mbf = modulate_backward(
        _t(d_normed, [N_IMG, D], ctx), saved.ln_img_out[],
        _t(saved.final_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_img_out = mbf.d_x.to_host(ctx)
    var d_final_scale = mbf.d_scale.to_host(ctx)
    var d_final_shift = mbf.d_shift.to_host(ctx)
    var d_fmods = _concat_seq(d_final_shift, d_final_scale)
    var lb_fmods = linear_backward(
        _t(d_fmods, [1, 2 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
        base.final_adaln_w[], 1, D, 2 * D, ctx,
    )
    d_vec_silu = _add_lists(d_vec_silu, lb_fmods.d_x.to_host(ctx))

    var lnbf = layer_norm_backward(
        _t(d_ln_img_out, [N_IMG, D], ctx), saved.img_out[], _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_img_out = lnbf.d_x.to_host(ctx)

    var d_x = _concat_seq(_zeros(N_TXT * D), d_img_out)

    # ── single-stream backward (REVERSE; LoRA; streamed weights) ──
    var bi = num_single - 1
    while bi >= 0:
        var block_idx = num_double + bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)
        var w = _single_weights_from_block(handle.block, handle.prefix, D, Fmlp, Dh, ctx)
        var sm = _single_modvecs_from_flat(saved.sgl_mod_flat[bi].copy(), D)
        var bl = _single_lora_for(lora, bi)
        var bg = single_block_lora_backward[H, Dh, S](
            d_x.copy(), w, sm, bl, saved.sgl_saved[bi], cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_x = bg.base.d_x.copy()
        var sbase = _sgl_base(lora, bi)
        for s in range(SGL_SLOTS):
            d_a_flat[sbase + s] = bg.lora.d_a[s].copy()
            d_b_flat[sbase + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        var d_sm = _single_modvec3(bg.base)
        var lb_sm = linear_backward(
            _t(d_sm, [1, 3 * D], ctx), _t(saved.vec_silu.copy(), [1, D], ctx),
            base.sgl_mod[bi].w[], 1, D, 3 * D, ctx,
        )
        d_vec_silu = _add_lists(d_vec_silu, lb_sm.d_x.to_host(ctx))
        loader.mark_active_block_done(ctx)
        bi -= 1

    var seam = _split_seq(d_x, N_TXT, N_IMG, D)
    var d_to = seam[0].copy()
    var d_io = seam[1].copy()

    # ── double-stream backward (REVERSE; LoRA; streamed weights) ──
    var di = num_double - 1
    while di >= 0:
        var handle = loader.await_block(di, ctx)
        if di > 0:
            loader.prefetch_with_ctx(di - 1, ctx)
        var w = _double_weights_from_block(handle.block, handle.prefix, D, Fmlp, Dh, ctx)
        var im = _modvecs_from_flat(saved.dbl_img_mod[di].copy(), D)
        var tm = _modvecs_from_flat(saved.dbl_txt_mod[di].copy(), D)
        var bl = _double_lora_for(lora, di)
        var bg = double_block_lora_backward[H, Dh, N_IMG, N_TXT, S](
            d_io.copy(), d_to.copy(), w, im, tm, bl, saved.dbl_saved[di],
            cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        d_io = bg.base.img.d_x.copy()
        d_to = bg.base.txt.d_x.copy()
        var dbase = _dbl_base(di)
        for s in range(DBL_STREAM_SLOTS):
            d_a_flat[dbase + s] = bg.lora.img.d_a[s].copy()
            d_b_flat[dbase + s] = bg.lora.img.d_b[s].copy()
            d_a_flat[dbase + DBL_STREAM_SLOTS + s] = bg.lora.txt.d_a[s].copy()
            d_b_flat[dbase + DBL_STREAM_SLOTS + s] = bg.lora.txt.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.img.d_a[s]) + _nonfinite(bg.lora.img.d_b[s])
            nonfinite += _nonfinite(bg.lora.txt.d_a[s]) + _nonfinite(bg.lora.txt.d_b[s])
        var d_im = _modvec6(bg.base.img)
        var d_tm = _modvec6(bg.base.txt)
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
        loader.mark_active_block_done(ctx)
        di -= 1

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

    # ── vec backward: silu(vec) -> d_vec -> embeds ──
    var d_vec = silu_backward(_t(d_vec_silu.copy(), [1, D], ctx), _t(saved.vec.copy(), [1, D], ctx), ctx).to_host(ctx)
    var eb = _embed_vec_backward(d_vec.copy(), saved, base, base.has_guidance, D, T_DIM, VEC_DIM, max_period, ctx)

    return FluxLoraGradSet(
        d_a_flat^, d_b_flat^,
        d_img_tokens^, d_txt_tokens^, d_vec^,
        eb.d_timestep.copy(), eb.d_guidance.copy(), eb.d_vector.copy(),
        nonfinite,
    )
