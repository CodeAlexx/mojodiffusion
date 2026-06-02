# serenitymojo/models/ernie/ernie_stack_lora.mojo
#
# ERNIE-Image FULL DiT STACK *WITH LoRA* on every trained projection: forward
# (saving ckpt-inputs) + full-depth backward (training) that uses the parity-
# verified per-block LoRA variants (models/ernie/lora_block.mojo), COLLECTS every
# adapter's d_A/d_B, and supports an AdamW step + a PEFT/ai-toolkit save across
# all 7×num_layers adapters. This file COMPOSES; it rebuilds NOTHING.
#
# WHAT IS ALREADY PROVEN (cos>=0.999 vs torch) AND ONLY REUSED HERE
#   * models/ernie/block.mojo : base block fwd+bwd (19/19 cos>=0.99999 vs torch).
#   * models/ernie/ernie_stack.mojo : the BASE full-stack composition (streamed
#     block-offload path proven finite + no-OOM at 36 layers). THIS FILE is that
#     file with the base per-block calls swapped for the LoRA variants + LoRA-grad
#     collection.
#   * models/ernie/lora_block.mojo : ernie_block_lora_forward/backward (reduces to
#     base when adapters absent; LoRA d_x summed into the projection-input grad).
#   * training/{lora_save, train_step, optim} : LoraAdapter, _lora_adamw, save_lora_peft.
#
# CARRIER DESIGN (Tenet-2: make the right thing easy) — mirrors KleinLoraSet:
#   ErnieLoraSet holds ONE flat List[LoraAdapter] of 7×num_layers adapters indexed
#   by a deterministic scheme: flat = block*ERNIE_SLOTS + slot, slot order
#   {to_q, to_k, to_v, to_out.0, gate_proj, up_proj, linear_fc2}. The optimizer
#   walks this flat list; the backward SCATTERS the returned per-block 7-slot
#   d_A/d_B back into the matching flat ErnieLoraGrads.
#
# SCOPE: LoRA-on-projection training. Base weights (input/text proj, the 7 block
#   linears, AdaLN/final MLP) are FROZEN — their grads are computed by the base
#   path and discarded for the optimizer; only d_A/d_B are trained. The shared
#   mod-vec grads are still summed-across-blocks and returned (NOT backpropped into
#   the AdaLN MLP — that link is the E5 train-loop phase).
#
# USES THE STREAMED PATH (load->use->drop per layer): full F32 residency of all 36
#   blocks is ~31 GB and does NOT fit a 24 GB 3090 (E2 memory finding). LoRA
#   adapters stay resident (host List masters, small); base block weights stream.
#
# Mojo 1.0.0b1: def not fn; Tensor move-only; host List[Float32] carriers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm_backward import layer_norm_backward
from serenitymojo.ops.elementwise_backward import modulate_backward

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.ernie.weights import (
    ErnieBlockWeights, ErnieStackBase, load_ernie_block_weights,
)
from serenitymojo.models.ernie.block import (
    ErnieModVecs, ErnieBlockSaved, ErnieBlockGrads, ernie_block_forward,
)
from serenitymojo.models.ernie.lora_block import (
    ErnieBlockLora, ErnieBlockLoraGrads, ERNIE_SLOTS,
    SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_GATE, SLOT_UP, SLOT_DOWN,
    ernie_block_lora_forward, ernie_block_lora_backward,
)
from serenitymojo.models.ernie.ernie_stack import (
    ErnieStackForward, _zeros, _ones, _t, _linear_wdev, _linear_wdev_bias,
    _concat_img_txt, _split_img_txt, _add_lists, saved_x_out,
)

from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_peft, load_lora_for_resume,
)


comptime TArc = ArcPointer[Tensor]


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


# ── the LoRA carrier: every trained adapter, flat-indexed 7×num_layers ────────
struct ErnieLoraSet(Copyable, Movable):
    var ad: List[LoraAdapter]   # num_layers * ERNIE_SLOTS, slot order Q,K,V,O,gate,up,down
    var num_layers: Int
    var rank: Int

    def __init__(out self, var ad: List[LoraAdapter], num_layers: Int, rank: Int):
        self.ad = ad^
        self.num_layers = num_layers
        self.rank = rank


# Accessor by (block_idx, slot) -> a COPY of the adapter.
def ernie_lora_get(set: ErnieLoraSet, block_idx: Int, slot: Int) -> LoraAdapter:
    return set.ad[block_idx * ERNIE_SLOTS + slot].copy()


# ── build the full LoRA set for an ERNIE stack ────────────────────────────────
# Per-block projection in/out shapes (D = hidden, F = ffn):
#   to_q/to_k/to_v/to_out.0 : in=D out=D ; gate_proj/up_proj : in=D out=F ;
#   linear_fc2 : in=F out=D.
def build_ernie_lora_set(
    num_layers: Int, D: Int, F: Int, rank: Int, alpha: Float32
) -> ErnieLoraSet:
    var ad = List[LoraAdapter]()
    var seed = UInt64(2000)
    for _ in range(num_layers):
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1  # to_q
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1  # to_k
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1  # to_v
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1  # to_out.0
        ad.append(make_lora_adapter(rank, alpha, D, F, seed)); seed += 1  # gate_proj
        ad.append(make_lora_adapter(rank, alpha, D, F, seed)); seed += 1  # up_proj
        ad.append(make_lora_adapter(rank, alpha, F, D, seed)); seed += 1  # linear_fc2
    return ErnieLoraSet(ad^, num_layers, rank)


# Build a transient ErnieBlockLora for block bi from the flat set (all 7 present).
def _block_lora_for(set: ErnieLoraSet, bi: Int) -> ErnieBlockLora:
    var base = bi * ERNIE_SLOTS
    return ErnieBlockLora(
        Optional[LoraAdapter](set.ad[base + SLOT_Q].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_K].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_V].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_O].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_GATE].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_UP].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_DOWN].copy()),
    )


def ernie_block_lora_for(set: ErnieLoraSet, bi: Int) -> ErnieBlockLora:
    return _block_lora_for(set, bi)


# ── collected LoRA grads (flat, parallel to ErnieLoraSet) ────────────────────
struct ErnieLoraGrads(Movable):
    var d_a: List[List[Float32]]   # num_layers*ERNIE_SLOTS
    var d_b: List[List[Float32]]
    # load-bearing input-token grads + summed shared mod grads (prove the chain).
    var d_img_tokens: List[Float32]
    var d_txt_tokens: List[Float32]
    var d_shared_mod: List[Float32]   # [6D]
    var d_f_scale: List[Float32]      # [D]
    var d_f_shift: List[Float32]      # [D]
    var d_final_lin: List[Float32]    # [out_ch, D] (base, discarded by AdamW)
    var nonfinite_lora_grads: Int

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_img_tokens: List[Float32], var d_txt_tokens: List[Float32],
        var d_shared_mod: List[Float32],
        var d_f_scale: List[Float32], var d_f_shift: List[Float32],
        var d_final_lin: List[Float32],
        nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_img_tokens = d_img_tokens^
        self.d_txt_tokens = d_txt_tokens^
        self.d_shared_mod = d_shared_mod^
        self.d_f_scale = d_f_scale^
        self.d_f_shift = d_f_shift^
        self.d_final_lin = d_final_lin^
        self.nonfinite_lora_grads = nonfinite_lora_grads


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


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# RESIDENT LoRA stack (small depth, for the COMPOSITION parity gate). Mirrors
# ernie_stack_forward/backward exactly, swapping per-block calls for LoRA ones.
# Blocks are passed resident (the gate uses L=2-3 so they fit).
# ─────────────────────────────────────────────────────────────────────────────
def ernie_stack_lora_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase,
    blocks: List[ErnieBlockWeights], lora: ErnieLoraSet, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieStackForward:
    var num_layers = len(blocks)
    var img = _linear_wdev_bias(
        img_tokens, base.patch_w[], base.patch_b[], N_IMG, in_ch, ctx
    )
    var txt = _linear_wdev(txt_tokens, base.text_proj[], N_TXT, text_in, ctx)
    var x = _concat_img_txt(img, txt)

    var blk_x_in = List[TArc]()
    for bi in range(num_layers):
        blk_x_in.append(TArc(_t(x.copy(), [S, D], ctx)))
        var bl = _block_lora_for(lora, bi)
        var fwd = ernie_block_lora_forward[H, Dh, S](
            x.copy(), blocks[bi], mv, bl, cos, sin, D, F, eps, ctx,
        )
        x = fwd.out.copy()

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


def ernie_stack_lora_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase,
    blocks: List[ErnieBlockWeights], lora: ErnieLoraSet, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    saved: ErnieStackForward,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieLoraGrads:
    var num_layers = len(blocks)

    # ── final-layer backward (identical to base stack) ──
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

    # ── single-stream backward (REVERSE; per-block recompute) ──
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(num_layers * ERNIE_SLOTS):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())
    var d_shared_mod = _zeros(6 * D)
    var nonfinite = 0
    var bi = num_layers - 1
    while bi >= 0:
        var bl = _block_lora_for(lora, bi)
        var refwd = ernie_block_lora_forward[H, Dh, S](
            saved.blk_x_in[bi][].to_host(ctx), blocks[bi], mv, bl, cos, sin, D, F, eps, ctx,
        )
        var bg = ernie_block_lora_backward[H, Dh, S](
            d_x.copy(), blocks[bi], mv, bl, refwd.saved, cos, sin, D, F, eps, ctx,
        )
        d_x = bg.base.d_x.copy()
        d_shared_mod = _add_lists(d_shared_mod, _modvec6(bg.base))
        # SCATTER the 7-slot LoRA grads into the flat lists + accumulate finiteness.
        var base_idx = bi * ERNIE_SLOTS
        for s in range(ERNIE_SLOTS):
            d_a_flat[base_idx + s] = bg.lora.d_a[s].copy()
            d_b_flat[base_idx + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        bi -= 1

    var seam = _split_img_txt(d_x, N_IMG, N_TXT, D)
    var d_img = seam[0].copy()
    var d_txt = seam[1].copy()
    var lbi = linear_backward(
        _t(d_img, [N_IMG, D], ctx), _t(img_tokens, [N_IMG, in_ch], ctx),
        base.patch_w[], N_IMG, in_ch, D, ctx,
    )
    var d_img_tokens = lbi.d_x.to_host(ctx)
    var lbt = linear_backward(
        _t(d_txt, [N_TXT, D], ctx), _t(txt_tokens, [N_TXT, text_in], ctx),
        base.text_proj[], N_TXT, text_in, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)

    return ErnieLoraGrads(
        d_a_flat^, d_b_flat^,
        d_img_tokens^, d_txt_tokens^,
        d_shared_mod^, d_f_scale^, d_f_shift^, d_final_lin^,
        nonfinite,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STREAMED LoRA stack (real depth; the E5 train-loop path). Loads each block from
# the sharded checkpoint on demand (load->use->drop), bounding resident weight
# memory to ~one block. LoRA adapters stay host-resident (small). Math identical
# to the resident path; proven equivalent by the composition gate.
# ─────────────────────────────────────────────────────────────────────────────
def ernie_stack_lora_forward_streamed[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, st: ShardedSafeTensors,
    lora: ErnieLoraSet, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieStackForward:
    var num_layers = lora.num_layers
    var img = _linear_wdev_bias(
        img_tokens, base.patch_w[], base.patch_b[], N_IMG, in_ch, ctx
    )
    var txt = _linear_wdev(txt_tokens, base.text_proj[], N_TXT, text_in, ctx)
    var x = _concat_img_txt(img, txt)

    var blk_x_in = List[TArc]()
    for bi in range(num_layers):
        blk_x_in.append(TArc(_t(x.copy(), [S, D], ctx)))
        var w = load_ernie_block_weights(st, bi, ctx)        # swap IN
        var bl = _block_lora_for(lora, bi)
        var fwd = ernie_block_lora_forward[H, Dh, S](
            x.copy(), w, mv, bl, cos, sin, D, F, eps, ctx,
        )
        x = fwd.out.copy()
        # `w` drops here -> swap OUT before next block.

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


def ernie_stack_lora_backward_streamed[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, st: ShardedSafeTensors,
    lora: ErnieLoraSet, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    saved: ErnieStackForward,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieLoraGrads:
    var num_layers = lora.num_layers

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

    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(num_layers * ERNIE_SLOTS):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())
    var d_shared_mod = _zeros(6 * D)
    var nonfinite = 0
    var bi = num_layers - 1
    while bi >= 0:
        var w = load_ernie_block_weights(st, bi, ctx)        # swap IN
        var bl = _block_lora_for(lora, bi)
        var refwd = ernie_block_lora_forward[H, Dh, S](
            saved.blk_x_in[bi][].to_host(ctx), w, mv, bl, cos, sin, D, F, eps, ctx,
        )
        var bg = ernie_block_lora_backward[H, Dh, S](
            d_x.copy(), w, mv, bl, refwd.saved, cos, sin, D, F, eps, ctx,
        )
        d_x = bg.base.d_x.copy()
        d_shared_mod = _add_lists(d_shared_mod, _modvec6(bg.base))
        var base_idx = bi * ERNIE_SLOTS
        for s in range(ERNIE_SLOTS):
            d_a_flat[base_idx + s] = bg.lora.d_a[s].copy()
            d_b_flat[base_idx + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
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
    var lbt = linear_backward(
        _t(d_txt, [N_TXT, D], ctx), _t(txt_tokens, [N_TXT, text_in], ctx),
        base.text_proj[], N_TXT, text_in, D, ctx,
    )
    var d_txt_tokens = lbt.d_x.to_host(ctx)

    return ErnieLoraGrads(
        d_a_flat^, d_b_flat^,
        d_img_tokens^, d_txt_tokens^,
        d_shared_mod^, d_f_scale^, d_f_shift^, d_final_lin^,
        nonfinite,
    )


# ── AdamW step on EVERY adapter (reuses the proven per-adapter _lora_adamw) ───
def ernie_lora_adamw_step(
    mut set: ErnieLoraSet, grads: ErnieLoraGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    var n = set.num_layers * ERNIE_SLOTS
    for i in range(n):
        var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(set.ad[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay)


# ── per-block PEFT/kohya prefix scheme (the INVERSE of the inference target map) ─
# inference-flame ernie_image.rs lora.apply uses these module keys:
#   layers.<i>.self_attention.{to_q, to_k, to_v, to_out.0}
#   layers.<i>.mlp.{gate_proj, up_proj, linear_fc2}
# save_lora_peft appends .lora_A.weight / .lora_B.weight, so the saved file is
# byte-exact loadable by the inference LoRA path and ai-toolkit/PEFT.
def _ernie_lora_prefix(block_idx: Int, slot: Int) -> String:
    var b = String("layers.") + String(block_idx)
    if slot == SLOT_Q:
        return b + ".self_attention.to_q"
    elif slot == SLOT_K:
        return b + ".self_attention.to_k"
    elif slot == SLOT_V:
        return b + ".self_attention.to_v"
    elif slot == SLOT_O:
        return b + ".self_attention.to_out.0"
    elif slot == SLOT_GATE:
        return b + ".mlp.gate_proj"
    elif slot == SLOT_UP:
        return b + ".mlp.up_proj"
    return b + ".mlp.linear_fc2"


def ernie_lora_prefixes(num_layers: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_layers):
        for s in range(ERNIE_SLOTS):
            out.append(_ernie_lora_prefix(bi, s))
    return out^


# ── SAVE every adapter as a PEFT/ai-toolkit safetensors ──────────────────────
def save_ernie_lora(set: ErnieLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLora]()
    for bi in range(set.num_layers):
        for s in range(ERNIE_SLOTS):
            named.append(NamedLora(
                _ernie_lora_prefix(bi, s),
                set.ad[bi * ERNIE_SLOTS + s].copy(),
            ))
    return save_lora_peft(named, path, ctx)


# ── RESUME: load the adapter A/B back from a save_ernie_lora file ────────────
# AdamW moments are ZEROED (resume them from a loop TrainState checkpoint). The
# returned set carries the SAME flat order build_ernie_lora_set produces.
def load_ernie_lora_resume(
    num_layers: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> ErnieLoraSet:
    var prefixes = ernie_lora_prefixes(num_layers)
    var scale = alpha / Float32(rank)
    var named = load_lora_for_resume(prefixes, scale, path, ctx)   # flat order
    var ad = List[LoraAdapter]()
    for i in range(num_layers * ERNIE_SLOTS):
        ad.append(named[i].adapter.copy())
    return ErnieLoraSet(ad^, num_layers, rank)
