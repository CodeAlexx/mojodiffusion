# serenitymojo/models/anima/anima_stack_lora.mojo
#
# ANIMA (Cosmos-Predict2 MiniTrainDIT) FULL 28-block STACK *WITH LoRA* on every
# trained projection: forward (saving ckpt-inputs) + full-depth backward (training)
# that uses the parity-verified per-block LoRA variants (models/anima/lora_block.mojo),
# COLLECTS every adapter's d_A/d_B, and supports an AdamW step + a PEFT/ai-toolkit
# save across all 10×num_blocks adapters. This file COMPOSES; it rebuilds NOTHING.
#
# WHAT IS ALREADY PROVEN (cos>=0.999 vs torch) AND ONLY REUSED HERE
#   * models/anima/block.mojo : base block fwd+bwd (23/23 cos>=0.99999999 vs torch).
#   * models/anima/anima_stack.mojo : the BASE full-stack composition (per-block
#     recompute backward, PER-BLOCK modulation, shared t_silu/base_adaln). THIS FILE
#     is that file with the base per-block calls swapped for the LoRA variants +
#     LoRA-grad collection.
#   * models/anima/lora_block.mojo : anima_block_lora_forward/backward (reduces to
#     base when adapters absent; LoRA d_x summed into the trained-stream projection
#     input grad; ca k/v LoRA d_x discarded since input is frozen context).
#   * training/{lora_save, train_step} : LoraAdapter, _lora_adamw, save_lora_peft.
#
# CARRIER DESIGN (Tenet-2: make the right thing easy) — mirrors ErnieLoraSet:
#   AnimaLoraSet holds ONE flat List[LoraAdapter] of 10×num_blocks adapters indexed
#   by flat = block*ANIMA_SLOTS + slot, slot order {sa_q, sa_k, sa_v, sa_out,
#   ca_q, ca_k, ca_v, ca_out, mlp1, mlp2}. The optimizer walks this flat list; the
#   backward SCATTERS the returned per-block 10-slot d_A/d_B into the flat grads.
#
# SCOPE: LoRA-on-projection training. Base weights (x_embedder, the 10 block linears,
#   AdaLN-LoRA mod, final layer) are FROZEN — their grads are computed by the base
#   path and discarded for the optimizer; only d_A/d_B are trained. The summed
#   d_t_silu (the genuinely-shared trained quantity) is still produced + returned
#   (the t_embedder backprop link is the E5 train-loop phase, deferred).
#
# Mojo 1.0.0b1: def not fn; Tensor move-only; host List[Float32] carriers.

from std.gpu.host import DeviceContext, HostBuffer
from std.collections import List, Optional
from std.memory import ArcPointer
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import reshape, slice, add, mul, concat
from serenitymojo.ops.linalg_backward import (
    linear_backward, linear_backward_dx, linear_backward_dw,
)
from serenitymojo.ops.norm_backward import layer_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.anima.weights import (
    AnimaBlockWeights, AnimaStackBase, load_anima_block_weights_f32,
)
from serenitymojo.models.anima.block import AnimaBlockSaved, AnimaBlockGrads
from serenitymojo.models.anima.lora_block import (
    AnimaBlockLora, AnimaBlockLoraGrads, ANIMA_SLOTS,
    SLOT_SA_Q, SLOT_SA_K, SLOT_SA_V, SLOT_SA_O,
    SLOT_CA_Q, SLOT_CA_K, SLOT_CA_V, SLOT_CA_O, SLOT_MLP1, SLOT_MLP2,
    anima_block_lora_forward, anima_block_lora_backward,
    AnimaLoraAdapterDevice, AnimaBlockLoraDevice, anima_lora_adapter_to_device,
    anima_block_lora_forward_device_tensor, anima_block_lora_backward_device_tensors,
)
from serenitymojo.models.anima.anima_stack import (
    AnimaStackForward, _add_lists, _zeros, _ones, _t, _linear_wdev, _row2, _sh3,
    _scatter_first,
)
from serenitymojo.models.dit.anima_contract import ANIMA_HIDDEN  # 2048

from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_peft, load_lora_for_resume,
    save_lora_train_state, load_lora_train_state,
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


# ── the LoRA carrier: every trained adapter, flat-indexed 10×num_blocks ───────
struct AnimaLoraSet(Copyable, Movable):
    var ad: List[LoraAdapter]   # num_blocks * ANIMA_SLOTS, slot order above
    var num_blocks: Int
    var rank: Int

    def __init__(out self, var ad: List[LoraAdapter], num_blocks: Int, rank: Int):
        self.ad = ad^
        self.num_blocks = num_blocks
        self.rank = rank


def anima_lora_get(set: AnimaLoraSet, block_idx: Int, slot: Int) -> LoraAdapter:
    return set.ad[block_idx * ANIMA_SLOTS + slot].copy()


# ── build the full LoRA set for an Anima stack ────────────────────────────────
# Per-block projection in/out shapes (D = hidden 2048, F = mlp hidden, JOINT = 1024):
#   sa_q/sa_k/sa_v/sa_out : in=D out=D
#   ca_q/ca_out           : in=D out=D
#   ca_k/ca_v             : in=JOINT out=D
#   mlp1 : in=D out=F     mlp2 : in=F out=D
def build_anima_lora_set(
    num_blocks: Int, D: Int, JOINT: Int, F: Int, rank: Int, alpha: Float32
) -> AnimaLoraSet:
    var ad = List[LoraAdapter]()
    var seed = UInt64(3000)
    for _ in range(num_blocks):
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1      # sa_q
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1      # sa_k
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1      # sa_v
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1      # sa_out
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1      # ca_q
        ad.append(make_lora_adapter(rank, alpha, JOINT, D, seed)); seed += 1  # ca_k
        ad.append(make_lora_adapter(rank, alpha, JOINT, D, seed)); seed += 1  # ca_v
        ad.append(make_lora_adapter(rank, alpha, D, D, seed)); seed += 1      # ca_out
        ad.append(make_lora_adapter(rank, alpha, D, F, seed)); seed += 1      # mlp1
        ad.append(make_lora_adapter(rank, alpha, F, D, seed)); seed += 1      # mlp2
    return AnimaLoraSet(ad^, num_blocks, rank)


# Build a transient AnimaBlockLora for block bi from the flat set (all 10 present).
def _block_lora_for(set: AnimaLoraSet, bi: Int) -> AnimaBlockLora:
    var base = bi * ANIMA_SLOTS
    return AnimaBlockLora(
        Optional[LoraAdapter](set.ad[base + SLOT_SA_Q].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_SA_K].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_SA_V].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_SA_O].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_CA_Q].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_CA_K].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_CA_V].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_CA_O].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_MLP1].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_MLP2].copy()),
    )


def anima_block_lora_for(set: AnimaLoraSet, bi: Int) -> AnimaBlockLora:
    return _block_lora_for(set, bi)


# ── collected LoRA grads (flat, parallel to AnimaLoraSet) ─────────────────────
struct AnimaLoraGrads(Movable):
    var d_a: List[List[Float32]]   # num_blocks*ANIMA_SLOTS
    var d_b: List[List[Float32]]
    # load-bearing input grads + summed shared grad (prove the chain).
    var d_patches: List[Float32]
    var d_t_silu: List[Float32]       # [B, 2048]
    var d_base_adaln: List[Float32]   # [B, 6144]
    var d_x_embed: List[Float32]      # base, discarded by AdamW
    var d_fl_lin: List[Float32]
    var d_fl_mod1: List[Float32]
    var d_fl_mod2: List[Float32]
    var nonfinite_lora_grads: Int

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_patches: List[Float32], var d_t_silu: List[Float32],
        var d_base_adaln: List[Float32],
        var d_x_embed: List[Float32], var d_fl_lin: List[Float32],
        var d_fl_mod1: List[Float32], var d_fl_mod2: List[Float32],
        nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_patches = d_patches^
        self.d_t_silu = d_t_silu^
        self.d_base_adaln = d_base_adaln^
        self.d_x_embed = d_x_embed^
        self.d_fl_lin = d_fl_lin^
        self.d_fl_mod1 = d_fl_mod1^
        self.d_fl_mod2 = d_fl_mod2^
        self.nonfinite_lora_grads = nonfinite_lora_grads


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# RESIDENT LoRA stack (small depth, for the COMPOSITION parity gate). Mirrors
# anima_stack_forward/backward exactly, swapping per-block calls for LoRA ones.
# Blocks are passed resident (the gate uses L=3 so they fit).
# ─────────────────────────────────────────────────────────────────────────────
def anima_stack_lora_forward[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    patches: List[Float32],
    t_cond: List[Float32], base_adaln: List[Float32], context: List[Float32],
    base: AnimaStackBase, blocks: List[AnimaBlockWeights], lora: AnimaLoraSet,
    cos: Tensor, sin: Tensor,
    B: Int, D: Int, JOINT: Int, F: Int, IN_PATCH: Int, OUT_PATCH: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaStackForward:
    var num_blocks = len(blocks)

    var x_emb = _linear_wdev(patches, base.x_embed[], B * S_IMG, IN_PATCH, ctx)

    var x = x_emb.copy()
    var blk_x_in = List[TArc]()
    for bi in range(num_blocks):
        blk_x_in.append(TArc(_t(x.copy(), [B, S_IMG, D], ctx)))
        var bl = _block_lora_for(lora, bi)
        var fwd = anima_block_lora_forward[H, Dh, S_IMG, S_TXT](
            x.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
            blocks[bi], bl, cos, sin, B, D, JOINT, F, eps, ctx,
        )
        x = fwd.out.copy()

    # ── final layer (identical to base stack) ──
    var x_final_t = _t(x.copy(), [B, S_IMG, D], ctx)
    var tc_t = _t(t_cond.copy(), [B, ANIMA_HIDDEN], ctx)
    var ts_t = silu(tc_t, ctx)
    var fl_h = linear(ts_t, base.fl_mod1[], Optional[Tensor](None), ctx)
    var fl_modout = linear(fl_h, base.fl_mod2[], Optional[Tensor](None), ctx)
    var base_t = _t(base_adaln.copy(), [B, 3 * D], ctx)
    var base_half = slice(base_t, 1, 0, 2 * D, ctx)
    var fl_added = add(fl_modout, base_half, ctx)
    var fl_shift = slice(fl_added, 1, 0, D, ctx)
    var fl_scale = slice(fl_added, 1, D, D, ctx)
    var ln_ones = _t(_ones(D), [D], ctx)
    var ln_zeros = _t(_zeros(D), [D], ctx)
    var fl_ln = layer_norm(x_final_t, ln_ones, ln_zeros, eps, ctx)
    var s3 = List[Int](); s3.append(B); s3.append(1); s3.append(D)
    var scale3 = reshape(fl_scale, s3.copy(), ctx)
    var shift3 = reshape(fl_shift, s3.copy(), ctx)
    var one = _t(_ones(B * D), [B, 1, D], ctx)
    var factor = add(scale3, one, ctx)
    var fl_scaled = mul(fl_ln, factor, ctx)
    var fl_xmod = add(fl_scaled, shift3, ctx)
    var fl_xmod_2d = reshape(fl_xmod, _row2(B * S_IMG, D), ctx)
    var out = linear(fl_xmod_2d, base.fl_lin[], Optional[Tensor](None), ctx).to_host(ctx)

    return AnimaStackForward(
        out^, x_emb.copy(), blk_x_in^,
        TArc(x_final_t^), TArc(fl_ln^),
        TArc(fl_h^), TArc(fl_shift^), TArc(fl_scale^), TArc(fl_xmod^),
    )


def anima_stack_lora_backward[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    d_out: List[Float32],
    patches: List[Float32],
    t_cond: List[Float32], base_adaln: List[Float32], context: List[Float32],
    base: AnimaStackBase, blocks: List[AnimaBlockWeights], lora: AnimaLoraSet,
    cos: Tensor, sin: Tensor,
    saved: AnimaStackForward,
    B: Int, D: Int, JOINT: Int, F: Int, IN_PATCH: Int, OUT_PATCH: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaLoraGrads:
    var num_blocks = len(blocks)
    var ln_ones = _t(_ones(D), [D], ctx)

    var d_t_silu_acc = _zeros(B * ANIMA_HIDDEN)
    var d_base_adaln_acc = _zeros(B * 3 * D)

    # ── final-layer backward (identical to base stack) ──
    var fl_xmod_2d = reshape(saved.fl_xmod[], _row2(B * S_IMG, D), ctx)
    var lbf = linear_backward(
        _t(d_out, [B * S_IMG, OUT_PATCH], ctx), fl_xmod_2d, base.fl_lin[],
        B * S_IMG, D, OUT_PATCH, ctx,
    )
    var d_fl_lin = lbf.d_w.to_host(ctx)
    var d_fl_xmod_3d = reshape(lbf.d_x, _sh3(B, S_IMG, D), ctx)
    var mbf = modulate_backward(d_fl_xmod_3d, saved.fl_ln[], saved.fl_scale[], ctx)
    var d_x_final = layer_norm_backward_dx(mbf.d_x, saved.x_final[], ln_ones, eps, ctx)
    var d_x = d_x_final.to_host(ctx)

    var d_fl_added = concat(1, ctx, mbf.d_shift, mbf.d_scale)
    var d_base_fl = _scatter_first(d_fl_added.to_host(ctx), B, 2 * D, 3 * D)
    d_base_adaln_acc = _add_lists(d_base_adaln_acc, d_base_fl)
    var fl_ts = silu(_t(t_cond.copy(), [B, ANIMA_HIDDEN], ctx), ctx)
    var fl_lb2 = linear_backward(d_fl_added, saved.fl_mod_h[], base.fl_mod2[], B, 256, 2 * D, ctx)
    var d_fl_mod2 = fl_lb2.d_w.to_host(ctx)
    var fl_lb1 = linear_backward(fl_lb2.d_x, fl_ts, base.fl_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    var d_fl_mod1 = fl_lb1.d_w.to_host(ctx)
    d_t_silu_acc = _add_lists(d_t_silu_acc, fl_lb1.d_x.to_host(ctx))

    # ── block backward (REVERSE; per-block recompute) ──
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(num_blocks * ANIMA_SLOTS):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())
    var nonfinite = 0
    var bi = num_blocks - 1
    while bi >= 0:
        var bl = _block_lora_for(lora, bi)
        var refwd = anima_block_lora_forward[H, Dh, S_IMG, S_TXT](
            saved.blk_x_in[bi][].to_host(ctx),
            t_cond.copy(), base_adaln.copy(), context.copy(),
            blocks[bi], bl, cos, sin, B, D, JOINT, F, eps, ctx,
        )
        var bg = anima_block_lora_backward[H, Dh, S_IMG, S_TXT](
            d_x.copy(), refwd.saved, blocks[bi], bl, cos, sin, B, D, JOINT, F, eps, ctx,
        )
        d_x = bg.base.d_x.copy()                                 # INTER-BLOCK HANDOFF
        d_t_silu_acc = _add_lists(d_t_silu_acc, bg.base.d_t_silu)
        var base_idx = bi * ANIMA_SLOTS
        for s in range(ANIMA_SLOTS):
            d_a_flat[base_idx + s] = bg.lora.d_a[s].copy()
            d_b_flat[base_idx + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        bi -= 1

    # ── patch-embed backward (base, discarded by AdamW; d_patches load-bearing) ──
    var lbpe = linear_backward(
        _t(d_x, [B * S_IMG, D], ctx), _t(patches.copy(), [B * S_IMG, IN_PATCH], ctx),
        base.x_embed[], B * S_IMG, IN_PATCH, D, ctx,
    )
    var d_patches = lbpe.d_x.to_host(ctx)
    var d_x_embed = lbpe.d_w.to_host(ctx)

    return AnimaLoraGrads(
        d_a_flat^, d_b_flat^,
        d_patches^, d_t_silu_acc^, d_base_adaln_acc^,
        d_x_embed^, d_fl_lin^, d_fl_mod1^, d_fl_mod2^,
        nonfinite,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STREAMED LoRA stack (real depth; the E5 train-loop path). Loads each block from
# the real safetensors on demand (load->use->drop). LoRA adapters stay host-resident.
# Math identical to the resident path; proven equivalent by the composition gate.
# ─────────────────────────────────────────────────────────────────────────────
def anima_stack_lora_forward_streamed[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    patches: List[Float32],
    t_cond: List[Float32], base_adaln: List[Float32], context: List[Float32],
    base: AnimaStackBase, st: SafeTensors, lora: AnimaLoraSet,
    cos: Tensor, sin: Tensor,
    B: Int, D: Int, JOINT: Int, F: Int, IN_PATCH: Int, OUT_PATCH: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaStackForward:
    var num_blocks = lora.num_blocks
    var x_emb = _linear_wdev(patches, base.x_embed[], B * S_IMG, IN_PATCH, ctx)

    var x = x_emb.copy()
    var blk_x_in = List[TArc]()
    for bi in range(num_blocks):
        blk_x_in.append(TArc(_t(x.copy(), [B, S_IMG, D], ctx)))
        var w = load_anima_block_weights_f32(st, bi, ctx)        # swap IN
        var bl = _block_lora_for(lora, bi)
        var fwd = anima_block_lora_forward[H, Dh, S_IMG, S_TXT](
            x.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
            w, bl, cos, sin, B, D, JOINT, F, eps, ctx,
        )
        x = fwd.out.copy()
        # `w` drops here -> swap OUT before next block.

    var x_final_t = _t(x.copy(), [B, S_IMG, D], ctx)
    var tc_t = _t(t_cond.copy(), [B, ANIMA_HIDDEN], ctx)
    var ts_t = silu(tc_t, ctx)
    var fl_h = linear(ts_t, base.fl_mod1[], Optional[Tensor](None), ctx)
    var fl_modout = linear(fl_h, base.fl_mod2[], Optional[Tensor](None), ctx)
    var base_t = _t(base_adaln.copy(), [B, 3 * D], ctx)
    var base_half = slice(base_t, 1, 0, 2 * D, ctx)
    var fl_added = add(fl_modout, base_half, ctx)
    var fl_shift = slice(fl_added, 1, 0, D, ctx)
    var fl_scale = slice(fl_added, 1, D, D, ctx)
    var ln_ones = _t(_ones(D), [D], ctx)
    var ln_zeros = _t(_zeros(D), [D], ctx)
    var fl_ln = layer_norm(x_final_t, ln_ones, ln_zeros, eps, ctx)
    var s3 = List[Int](); s3.append(B); s3.append(1); s3.append(D)
    var scale3 = reshape(fl_scale, s3.copy(), ctx)
    var shift3 = reshape(fl_shift, s3.copy(), ctx)
    var one = _t(_ones(B * D), [B, 1, D], ctx)
    var factor = add(scale3, one, ctx)
    var fl_scaled = mul(fl_ln, factor, ctx)
    var fl_xmod = add(fl_scaled, shift3, ctx)
    var fl_xmod_2d = reshape(fl_xmod, _row2(B * S_IMG, D), ctx)
    var out = linear(fl_xmod_2d, base.fl_lin[], Optional[Tensor](None), ctx).to_host(ctx)

    return AnimaStackForward(
        out^, x_emb.copy(), blk_x_in^,
        TArc(x_final_t^), TArc(fl_ln^),
        TArc(fl_h^), TArc(fl_shift^), TArc(fl_scale^), TArc(fl_xmod^),
    )


def anima_stack_lora_backward_streamed[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    d_out: List[Float32],
    patches: List[Float32],
    t_cond: List[Float32], base_adaln: List[Float32], context: List[Float32],
    base: AnimaStackBase, st: SafeTensors, lora: AnimaLoraSet,
    cos: Tensor, sin: Tensor,
    saved: AnimaStackForward,
    B: Int, D: Int, JOINT: Int, F: Int, IN_PATCH: Int, OUT_PATCH: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaLoraGrads:
    var num_blocks = lora.num_blocks
    var ln_ones = _t(_ones(D), [D], ctx)

    var d_t_silu_acc = _zeros(B * ANIMA_HIDDEN)
    var d_base_adaln_acc = _zeros(B * 3 * D)

    var fl_xmod_2d = reshape(saved.fl_xmod[], _row2(B * S_IMG, D), ctx)
    var lbf = linear_backward(
        _t(d_out, [B * S_IMG, OUT_PATCH], ctx), fl_xmod_2d, base.fl_lin[],
        B * S_IMG, D, OUT_PATCH, ctx,
    )
    var d_fl_lin = lbf.d_w.to_host(ctx)
    var d_fl_xmod_3d = reshape(lbf.d_x, _sh3(B, S_IMG, D), ctx)
    var mbf = modulate_backward(d_fl_xmod_3d, saved.fl_ln[], saved.fl_scale[], ctx)
    var d_x_final = layer_norm_backward_dx(mbf.d_x, saved.x_final[], ln_ones, eps, ctx)
    var d_x = d_x_final.to_host(ctx)

    var d_fl_added = concat(1, ctx, mbf.d_shift, mbf.d_scale)
    var d_base_fl = _scatter_first(d_fl_added.to_host(ctx), B, 2 * D, 3 * D)
    d_base_adaln_acc = _add_lists(d_base_adaln_acc, d_base_fl)
    var fl_ts = silu(_t(t_cond.copy(), [B, ANIMA_HIDDEN], ctx), ctx)
    var fl_lb2 = linear_backward(d_fl_added, saved.fl_mod_h[], base.fl_mod2[], B, 256, 2 * D, ctx)
    var d_fl_mod2 = fl_lb2.d_w.to_host(ctx)
    var fl_lb1 = linear_backward(fl_lb2.d_x, fl_ts, base.fl_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    var d_fl_mod1 = fl_lb1.d_w.to_host(ctx)
    d_t_silu_acc = _add_lists(d_t_silu_acc, fl_lb1.d_x.to_host(ctx))

    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(num_blocks * ANIMA_SLOTS):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())
    var nonfinite = 0
    var bi = num_blocks - 1
    while bi >= 0:
        var w = load_anima_block_weights_f32(st, bi, ctx)        # swap IN
        var bl = _block_lora_for(lora, bi)
        var refwd = anima_block_lora_forward[H, Dh, S_IMG, S_TXT](
            saved.blk_x_in[bi][].to_host(ctx),
            t_cond.copy(), base_adaln.copy(), context.copy(),
            w, bl, cos, sin, B, D, JOINT, F, eps, ctx,
        )
        var bg = anima_block_lora_backward[H, Dh, S_IMG, S_TXT](
            d_x.copy(), refwd.saved, w, bl, cos, sin, B, D, JOINT, F, eps, ctx,
        )
        d_x = bg.base.d_x.copy()
        d_t_silu_acc = _add_lists(d_t_silu_acc, bg.base.d_t_silu)
        var base_idx = bi * ANIMA_SLOTS
        for s in range(ANIMA_SLOTS):
            d_a_flat[base_idx + s] = bg.lora.d_a[s].copy()
            d_b_flat[base_idx + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        bi -= 1
        # `w` + `bg` drop here -> resident weight + per-block grads freed.

    var lbpe = linear_backward(
        _t(d_x, [B * S_IMG, D], ctx), _t(patches.copy(), [B * S_IMG, IN_PATCH], ctx),
        base.x_embed[], B * S_IMG, IN_PATCH, D, ctx,
    )
    var d_patches = lbpe.d_x.to_host(ctx)
    var d_x_embed = lbpe.d_w.to_host(ctx)

    return AnimaLoraGrads(
        d_a_flat^, d_b_flat^,
        d_patches^, d_t_silu_acc^, d_base_adaln_acc^,
        d_x_embed^, d_fl_lin^, d_fl_mod1^, d_fl_mod2^,
        nonfinite,
    )


# ═════════════════════════════════════════════════════════════════════════════
# DEVICE-RESIDENT LoRA STACK (the fast hot path; mirrors Z-Image's
# zimage_stack_lora_forward_main_device / _backward_main_device). All 28 blocks +
# saved block-input activations + LoRA A/B stay DEVICE tensors across forward AND
# backward recompute. NO to_host()/from_host() inside the block loop. The LoRA set
# is uploaded to device ONCE per step by the trainer (anima_lora_set_to_device).
# Base block weights are passed resident (BF16 production / F32 parity); the F32
# residual stream is preserved via linear's mixed_base path.
# ═════════════════════════════════════════════════════════════════════════════
struct AnimaLoraDeviceSet(Copyable, Movable):
    var ad: List[AnimaLoraAdapterDevice]   # num_blocks * ANIMA_SLOTS
    var num_blocks: Int
    var rank: Int

    def __init__(
        out self, var ad: List[AnimaLoraAdapterDevice], num_blocks: Int, rank: Int
    ):
        self.ad = ad^
        self.num_blocks = num_blocks
        self.rank = rank


def anima_lora_set_to_device(
    set: AnimaLoraSet, dt: STDtype, ctx: DeviceContext
) raises -> AnimaLoraDeviceSet:
    var ad = List[AnimaLoraAdapterDevice]()
    var n = set.num_blocks * ANIMA_SLOTS
    for i in range(n):
        ad.append(anima_lora_adapter_to_device(set.ad[i], dt, ctx))
    return AnimaLoraDeviceSet(ad^, set.num_blocks, set.rank)


def _block_lora_dev_for(set: AnimaLoraDeviceSet, bi: Int) -> AnimaBlockLoraDevice:
    var base = bi * ANIMA_SLOTS
    return AnimaBlockLoraDevice(
        set.ad[base + SLOT_SA_Q].copy(), set.ad[base + SLOT_SA_K].copy(),
        set.ad[base + SLOT_SA_V].copy(), set.ad[base + SLOT_SA_O].copy(),
        set.ad[base + SLOT_CA_Q].copy(), set.ad[base + SLOT_CA_K].copy(),
        set.ad[base + SLOT_CA_V].copy(), set.ad[base + SLOT_CA_O].copy(),
        set.ad[base + SLOT_MLP1].copy(), set.ad[base + SLOT_MLP2].copy(),
    )


# Bulk D2H of the flat d_a/d_b device tensors into the host List[List[Float32]]
# grad layout (parallel to AnimaLoraSet). ONE host buffer + ONE sync (mirrors
# zimage _zimage_tensor_grads_to_host).
struct _AnimaHostGradLists(Movable):
    var d_a: List[List[Float32]]
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


def _host_grad_slice(host: HostBuffer[DType.uint8], offset: Int, numel: Int) -> List[Float32]:
    var out = List[Float32]()
    var fp = (host.unsafe_ptr() + offset).bitcast[Float32]()
    for i in range(numel):
        out.append(fp[i])
    return out^


def _grad_arc_f32(t: TArc, ctx: DeviceContext) raises -> TArc:
    if t[].dtype() == STDtype.F32:
        return t.copy()
    # Host AdamW stores master params and moments as F32; device grads may be BF16.
    var t32 = cast_tensor(t[], STDtype.F32, ctx)
    return TArc(t32^)


def _anima_tensor_grads_to_host(
    indices: List[Int], d_a_t: List[TArc], d_b_t: List[TArc],
    total_slots: Int, ctx: DeviceContext,
) raises -> _AnimaHostGradLists:
    var a_f32 = List[TArc]()
    var b_f32 = List[TArc]()
    for i in range(len(d_a_t)):
        a_f32.append(_grad_arc_f32(d_a_t[i], ctx))
    for i in range(len(d_b_t)):
        b_f32.append(_grad_arc_f32(d_b_t[i], ctx))

    var total_bytes = 0
    for i in range(len(a_f32)):
        total_bytes += a_f32[i][].nbytes()
    for i in range(len(b_f32)):
        total_bytes += b_f32[i][].nbytes()

    var host = ctx.enqueue_create_host_buffer[DType.uint8](total_bytes)
    var a_off = List[Int](); var a_num = List[Int]()
    var b_off = List[Int](); var b_num = List[Int]()
    var cursor = 0
    for i in range(len(a_f32)):
        a_off.append(cursor); a_num.append(a_f32[i][].numel())
        var dst = host.create_sub_buffer[DType.uint8](cursor, a_f32[i][].nbytes())
        ctx.enqueue_copy(dst_buf=dst, src_buf=a_f32[i][].buf)
        cursor += a_f32[i][].nbytes()
    for i in range(len(b_f32)):
        b_off.append(cursor); b_num.append(b_f32[i][].numel())
        var dst = host.create_sub_buffer[DType.uint8](cursor, b_f32[i][].nbytes())
        ctx.enqueue_copy(dst_buf=dst, src_buf=b_f32[i][].buf)
        cursor += b_f32[i][].nbytes()
    ctx.synchronize()

    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(total_slots):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())
    for i in range(len(indices)):
        var flat = indices[i]
        d_a_flat[flat] = _host_grad_slice(host, a_off[i], a_num[i])
        d_b_flat[flat] = _host_grad_slice(host, b_off[i], b_num[i])
    return _AnimaHostGradLists(d_a_flat^, d_b_flat^)


# Save the t_silu(t_cond) precompute on device ONCE: silu of the host t_cond.
def anima_stack_lora_forward_device_resident[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    patches: List[Float32],
    t_cond: List[Float32], base_adaln: List[Float32], context: List[Float32],
    base: AnimaStackBase, blocks: List[AnimaBlockWeights], lora: AnimaLoraDeviceSet,
    cos: Tensor, sin: Tensor,
    B: Int, D: Int, JOINT: Int, F: Int, IN_PATCH: Int, OUT_PATCH: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaStackForward:
    var num_blocks = len(blocks)

    # patch embed (base, F32 weight) -> [B*S_img, D]
    var x_emb = _linear_wdev(patches, base.x_embed[], B * S_IMG, IN_PATCH, ctx)

    # upload the shared per-step conditioning ONCE (device tensors).
    var t_silu_dev = silu(_t(t_cond.copy(), [B, ANIMA_HIDDEN], ctx), ctx)
    var t_silu_arc = TArc(t_silu_dev^)
    var base_t_arc = TArc(_t(base_adaln.copy(), [B, 3 * D], ctx))
    var ctx_t_arc = TArc(_t(context.copy(), [B, S_TXT, JOINT], ctx))

    var x_arc = TArc(_t(x_emb.copy(), [B, S_IMG, D], ctx))
    var blk_x_in = List[TArc]()
    # SAVED-ACTIVATION fast path: retain each block's full AnimaBlockSaved so the
    # backward READS it instead of recomputing the block forward (the OT-parity
    # speed lever). Device-resident; ~17 GiB headroom at 512px/S_IMG=1024 BF16.
    var blk_saved = List[AnimaBlockSaved]()
    for bi in range(num_blocks):
        blk_x_in.append(x_arc.copy())   # device block-input checkpoint
        var bl = _block_lora_dev_for(lora, bi)
        var fwd = anima_block_lora_forward_device_tensor[H, Dh, S_IMG, S_TXT](
            x_arc.copy(), t_silu_arc.copy(), base_t_arc.copy(), ctx_t_arc.copy(),
            blocks[bi], bl, cos, sin, B, D, JOINT, F, eps, ctx,
        )
        x_arc = fwd.out.copy()
        blk_saved.append(fwd.saved.copy())   # retain for backward (no recompute)

    # ── final layer (identical to base/streamed stack; device-resident) ──
    var fl_h = linear(t_silu_arc[], base.fl_mod1[], Optional[Tensor](None), ctx)
    var fl_modout = linear(fl_h, base.fl_mod2[], Optional[Tensor](None), ctx)
    var base_half = slice(base_t_arc[], 1, 0, 2 * D, ctx)
    var fl_added = add(fl_modout, base_half, ctx)
    var fl_shift = slice(fl_added, 1, 0, D, ctx)
    var fl_scale = slice(fl_added, 1, D, D, ctx)
    var ln_ones = _t(_ones(D), [D], ctx)
    var ln_zeros = _t(_zeros(D), [D], ctx)
    var fl_ln = layer_norm(x_arc[], ln_ones, ln_zeros, eps, ctx)
    var scale3 = reshape(fl_scale, _sh3(B, 1, D), ctx)
    var shift3 = reshape(fl_shift, _sh3(B, 1, D), ctx)
    var one = _t(_ones(B * D), [B, 1, D], ctx)
    var factor = add(scale3, one, ctx)
    var fl_scaled = mul(fl_ln, factor, ctx)
    var fl_xmod = add(fl_scaled, shift3, ctx)
    var fl_xmod_2d = reshape(fl_xmod, _row2(B * S_IMG, D), ctx)
    var out = linear(fl_xmod_2d, base.fl_lin[], Optional[Tensor](None), ctx).to_host(ctx)

    return AnimaStackForward(
        out^, x_emb.copy(), blk_x_in^,
        x_arc.copy(), TArc(fl_ln^),
        TArc(fl_h^), TArc(fl_shift^), TArc(fl_scale^), TArc(fl_xmod^),
        blk_saved^,
    )


# ═════════════════════════════════════════════════════════════════════════════
# NO-SAVE device-resident forward (INFERENCE/SAMPLING only). Identical math to
# anima_stack_lora_forward_device_resident, but DISCARDS each block's saved
# activations immediately (only one block's activations live at a time) and
# returns ONLY the host output velocity [B*S_IMG, OUT_PATCH]. This is the
# 24GB-safe 1024 sampling path: all 28 blocks stay resident (BF16, ~3.7 GiB) and
# there is NO per-block disk reload (unlike the streamed path) and NO 28× saved-
# activation retention (unlike the backward/training resident path). The LoRA
# overlay is applied additively per block (scale·B·A, NEVER fused). Used by the
# standalone anima_sample_cli.
# ═════════════════════════════════════════════════════════════════════════════
def anima_stack_lora_forward_device_resident_nosave[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    patches: List[Float32],
    t_cond: List[Float32], base_adaln: List[Float32], context: List[Float32],
    base: AnimaStackBase, blocks: List[AnimaBlockWeights], lora: AnimaLoraDeviceSet,
    cos: Tensor, sin: Tensor,
    B: Int, D: Int, JOINT: Int, F: Int, IN_PATCH: Int, OUT_PATCH: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> List[Float32]:
    var num_blocks = len(blocks)

    # patch embed (base, F32 weight) -> [B*S_img, D]
    var x_emb = _linear_wdev(patches, base.x_embed[], B * S_IMG, IN_PATCH, ctx)

    # upload the shared per-step conditioning ONCE (device tensors).
    var t_silu_dev = silu(_t(t_cond.copy(), [B, ANIMA_HIDDEN], ctx), ctx)
    var t_silu_arc = TArc(t_silu_dev^)
    var base_t_arc = TArc(_t(base_adaln.copy(), [B, 3 * D], ctx))
    var ctx_t_arc = TArc(_t(context.copy(), [B, S_TXT, JOINT], ctx))

    var x_arc = TArc(_t(x_emb.copy(), [B, S_IMG, D], ctx))
    for bi in range(num_blocks):
        var bl = _block_lora_dev_for(lora, bi)
        var fwd = anima_block_lora_forward_device_tensor[H, Dh, S_IMG, S_TXT](
            x_arc.copy(), t_silu_arc.copy(), base_t_arc.copy(), ctx_t_arc.copy(),
            blocks[bi], bl, cos, sin, B, D, JOINT, F, eps, ctx,
        )
        x_arc = fwd.out.copy()

    # ── final layer (identical to the resident/streamed stack) ──
    var fl_h = linear(t_silu_arc[], base.fl_mod1[], Optional[Tensor](None), ctx)
    var fl_modout = linear(fl_h, base.fl_mod2[], Optional[Tensor](None), ctx)
    var base_half = slice(base_t_arc[], 1, 0, 2 * D, ctx)
    var fl_added = add(fl_modout, base_half, ctx)
    var fl_shift = slice(fl_added, 1, 0, D, ctx)
    var fl_scale = slice(fl_added, 1, D, D, ctx)
    var ln_ones = _t(_ones(D), [D], ctx)
    var ln_zeros = _t(_zeros(D), [D], ctx)
    var fl_ln = layer_norm(x_arc[], ln_ones, ln_zeros, eps, ctx)
    var scale3 = reshape(fl_scale, _sh3(B, 1, D), ctx)
    var shift3 = reshape(fl_shift, _sh3(B, 1, D), ctx)
    var one = _t(_ones(B * D), [B, 1, D], ctx)
    var factor = add(scale3, one, ctx)
    var fl_scaled = mul(fl_ln, factor, ctx)
    var fl_xmod = add(fl_scaled, shift3, ctx)
    var fl_xmod_2d = reshape(fl_xmod, _row2(B * S_IMG, D), ctx)
    var out = linear(fl_xmod_2d, base.fl_lin[], Optional[Tensor](None), ctx).to_host(ctx)
    return out^


# Public INFERENCE entrypoint name (mirrors zimage_block_lora_predict_device_tensor):
# a resident, NO-SAVE forward used by anima_sample_cli. Blocks are loaded ONCE
# (BF16, resident) and the LoRA set is uploaded ONCE to device; this is called per
# CFG branch per denoise step with NO disk reload and NO saved-for-backward
# activations. Thin alias over anima_stack_lora_forward_device_resident_nosave.
def anima_stack_lora_predict_device_resident[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    patches: List[Float32],
    t_cond: List[Float32], base_adaln: List[Float32], context: List[Float32],
    base: AnimaStackBase, blocks: List[AnimaBlockWeights], lora: AnimaLoraDeviceSet,
    cos: Tensor, sin: Tensor,
    B: Int, D: Int, JOINT: Int, F: Int, IN_PATCH: Int, OUT_PATCH: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> List[Float32]:
    return anima_stack_lora_forward_device_resident_nosave[H, Dh, S_IMG, S_TXT](
        patches, t_cond, base_adaln, context,
        base, blocks, lora, cos, sin,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, eps, ctx,
    )


def anima_stack_lora_backward_device_resident[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    d_out: List[Float32],
    patches: List[Float32],
    t_cond: List[Float32], base_adaln: List[Float32], context: List[Float32],
    base: AnimaStackBase, blocks: List[AnimaBlockWeights], lora: AnimaLoraDeviceSet,
    cos: Tensor, sin: Tensor,
    saved: AnimaStackForward,
    B: Int, D: Int, JOINT: Int, F: Int, IN_PATCH: Int, OUT_PATCH: Int, eps: Float32,
    ctx: DeviceContext,
    trace: Bool = False,
) raises -> AnimaLoraGrads:
    var num_blocks = len(blocks)
    var ln_ones = _t(_ones(D), [D], ctx)

    # SAVED-ACTIVATION fast path: the forward retained every block's AnimaBlockSaved,
    # so the backward READS it (no per-block recompute forward, no re-upload of the
    # shared conditioning for the block loop). This removes the recompute that
    # dominated the backward time.
    var have_saved = len(saved.blk_saved) == num_blocks

    var d_t_silu_acc = _t(_zeros(B * ANIMA_HIDDEN), [B, ANIMA_HIDDEN], ctx)

    # ── final-layer backward (frozen base; dx-only) ──
    var fl_xmod_2d = reshape(saved.fl_xmod[], _row2(B * S_IMG, D), ctx)
    var d_fl_xmod_2d = linear_backward_dx(
        _t(d_out, [B * S_IMG, OUT_PATCH], ctx), base.fl_lin[], B * S_IMG, D, OUT_PATCH, ctx,
    )
    var d_fl_xmod_3d = reshape(d_fl_xmod_2d, _sh3(B, S_IMG, D), ctx)
    var mbf = modulate_backward(d_fl_xmod_3d, saved.fl_ln[], saved.fl_scale[], ctx)
    var d_x_final = layer_norm_backward_dx(mbf.d_x, saved.x_final[], ln_ones, eps, ctx)
    var d_x = TArc(d_x_final^)

    var d_fl_added = concat(1, ctx, mbf.d_shift, mbf.d_scale)
    var d_base_fl = _scatter_first(d_fl_added.to_host(ctx), B, 2 * D, 3 * D)
    var fl_ts = silu(_t(t_cond.copy(), [B, ANIMA_HIDDEN], ctx), ctx)
    var d_fl_modh = linear_backward_dx(d_fl_added, base.fl_mod2[], B, 256, 2 * D, ctx)
    var d_fl_tsilu = linear_backward_dx(d_fl_modh, base.fl_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    d_t_silu_acc = add(d_t_silu_acc, d_fl_tsilu, ctx)

    # shared conditioning is only needed for the recompute FALLBACK (have_saved=False).
    var t_silu_arc = TArc(_t(_zeros(1), [1], ctx))
    var base_t_arc = TArc(_t(_zeros(1), [1], ctx))
    var ctx_t_arc = TArc(_t(_zeros(1), [1], ctx))
    if not have_saved:
        t_silu_arc = TArc(silu(_t(t_cond.copy(), [B, ANIMA_HIDDEN], ctx), ctx))
        base_t_arc = TArc(_t(base_adaln.copy(), [B, 3 * D], ctx))
        ctx_t_arc = TArc(_t(context.copy(), [B, S_TXT, JOINT], ctx))

    # ── block backward (REVERSE; READ saved activations, no recompute) ──
    var grad_indices = List[Int]()
    var d_a_t = List[TArc]()
    var d_b_t = List[TArc]()
    var bi = num_blocks - 1
    while bi >= 0:
        var bl = _block_lora_dev_for(lora, bi)
        var do_trace = trace and (bi == num_blocks - 1)
        if have_saved:
            # READ the retained activations directly (no recompute forward).
            var bg = anima_block_lora_backward_device_tensors[H, Dh, S_IMG, S_TXT](
                d_x[], saved.blk_saved[bi], blocks[bi], bl, cos, sin,
                B, D, JOINT, F, eps, ctx, do_trace,
            )
            d_x = bg.d_x.copy()
            d_t_silu_acc = add(d_t_silu_acc, bg.d_t_silu[], ctx)
            var base_idx = bi * ANIMA_SLOTS
            for s in range(ANIMA_SLOTS):
                grad_indices.append(base_idx + s)
                d_a_t.append(bg.d_a[s].copy())
                d_b_t.append(bg.d_b[s].copy())
        else:
            # Recompute fallback (empty blk_saved path).
            var refwd = anima_block_lora_forward_device_tensor[H, Dh, S_IMG, S_TXT](
                saved.blk_x_in[bi].copy(), t_silu_arc.copy(), base_t_arc.copy(), ctx_t_arc.copy(),
                blocks[bi], bl, cos, sin, B, D, JOINT, F, eps, ctx,
            )
            var bg = anima_block_lora_backward_device_tensors[H, Dh, S_IMG, S_TXT](
                d_x[], refwd.saved, blocks[bi], bl, cos, sin, B, D, JOINT, F, eps, ctx, do_trace,
            )
            d_x = bg.d_x.copy()
            d_t_silu_acc = add(d_t_silu_acc, bg.d_t_silu[], ctx)
            var base_idx = bi * ANIMA_SLOTS
            for s in range(ANIMA_SLOTS):
                grad_indices.append(base_idx + s)
                d_a_t.append(bg.d_a[s].copy())
                d_b_t.append(bg.d_b[s].copy())
        bi -= 1

    # ── patch-embed backward (frozen base; d_patches load-bearing) ──
    var d_x_2d = reshape(d_x[], _row2(B * S_IMG, D), ctx)
    var d_patches_t = linear_backward_dx(d_x_2d, base.x_embed[], B * S_IMG, IN_PATCH, D, ctx)
    var d_patches = d_patches_t.to_host(ctx)
    var d_x_embed = List[Float32]()

    # ── ONE bulk D2H of all LoRA grads ──
    var host_grads = _anima_tensor_grads_to_host(
        grad_indices, d_a_t, d_b_t, num_blocks * ANIMA_SLOTS, ctx,
    )
    var nonfinite = 0
    for i in range(len(grad_indices)):
        var idx = grad_indices[i]
        nonfinite += _nonfinite(host_grads.d_a[idx]) + _nonfinite(host_grads.d_b[idx])

    var d_t_silu_host = d_t_silu_acc.to_host(ctx)
    var empty_fl1 = List[Float32]()
    var empty_fl2 = List[Float32]()
    var empty_fll = List[Float32]()
    return AnimaLoraGrads(
        host_grads.d_a.copy(), host_grads.d_b.copy(),
        d_patches^, d_t_silu_host^, d_base_fl^,
        d_x_embed^, empty_fll^, empty_fl1^, empty_fl2^,
        nonfinite,
    )


# ── AdamW step on EVERY adapter (reuses the proven per-adapter _lora_adamw) ───
def anima_lora_adamw_step(
    mut set: AnimaLoraSet, grads: AnimaLoraGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    var n = set.num_blocks * ANIMA_SLOTS
    for i in range(n):
        var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(set.ad[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay)


# ── per-block kohya/inference prefix scheme (the INVERSE of the inference map) ─
# inference-flame anima.rs linear_no_bias chokepoint uses these base weight keys
# (LoraStack::apply matches against the weight_key minus the LoRA suffix). For the
# DiffusionModel LoRA format the saved prefix is the base key WITHOUT `.weight`:
#   net.blocks.<i>.self_attn.{q_proj,k_proj,v_proj,output_proj}    (anima.rs:365-390)
#   net.blocks.<i>.cross_attn.{q_proj,k_proj,v_proj,output_proj}   (anima.rs:411-440)
#   net.blocks.<i>.mlp.{layer1,layer2}                             (anima.rs:449-451)
# save_lora_peft appends .lora_A.weight / .lora_B.weight, so the file is byte-exact
# loadable by the inference LoRA path (FMT_DIFFUSION_MODEL) and ai-toolkit/PEFT.
def _anima_lora_prefix(block_idx: Int, slot: Int) -> String:
    var b = String("net.blocks.") + String(block_idx)
    if slot == SLOT_SA_Q:
        return b + ".self_attn.q_proj"
    elif slot == SLOT_SA_K:
        return b + ".self_attn.k_proj"
    elif slot == SLOT_SA_V:
        return b + ".self_attn.v_proj"
    elif slot == SLOT_SA_O:
        return b + ".self_attn.output_proj"
    elif slot == SLOT_CA_Q:
        return b + ".cross_attn.q_proj"
    elif slot == SLOT_CA_K:
        return b + ".cross_attn.k_proj"
    elif slot == SLOT_CA_V:
        return b + ".cross_attn.v_proj"
    elif slot == SLOT_CA_O:
        return b + ".cross_attn.output_proj"
    elif slot == SLOT_MLP1:
        return b + ".mlp.layer1"
    return b + ".mlp.layer2"


def anima_lora_prefixes(num_blocks: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_blocks):
        for s in range(ANIMA_SLOTS):
            out.append(_anima_lora_prefix(bi, s))
    return out^


# ── SAVE every adapter as a PEFT/ai-toolkit safetensors ──────────────────────
def save_anima_lora(set: AnimaLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLora]()
    for bi in range(set.num_blocks):
        for s in range(ANIMA_SLOTS):
            named.append(NamedLora(
                _anima_lora_prefix(bi, s),
                set.ad[bi * ANIMA_SLOTS + s].copy(),
            ))
    return save_lora_peft(named, path, ctx)


# ── OneTrainer diffusers-state-dict key naming (delta 5) ─────────────────────
# Mirrors what AnimaLoRASaver dumps: raw transformer_lora.state_dict() of an OT
# LoRAModuleWrapper(prefix="transformer", filter=attn1,attn2,ff). Each LoRAModule's
# prefix = "transformer.<diffusers module name>." and state_dict emits per module:
#   <prefix>.lora_down.weight   [rank, in]   (== LoraAdapter.a)   LoRAModule.lora_down
#   <prefix>.lora_up.weight     [out, rank]  (== LoraAdapter.b)   LoRAModule.lora_up
#   <prefix>.alpha              scalar       (register_buffer "alpha", LoRAModule.py:538)
# The diffusers module names come from AnimaModel.diffusers_to_original() (the exact
# inverse rename of the saved Cosmos checkpoint):
#   self_attn   -> attn1.{to_q,to_k,to_v,to_out.0}     (AnimaModel.py:44-47)
#   cross_attn  -> attn2.{to_q,to_k,to_v,to_out.0}     (AnimaModel.py:52-55)
#   mlp.layer1  -> ff.net.0.proj   mlp.layer2 -> ff.net.2  (AnimaModel.py:58-59)
# LoRAModuleWrapper.__create_modules names children via orig_module.named_modules()
# (LoRAModule.py:897) -> "transformer_blocks.{i}.{attn1.to_q...}"; prefix prepended
# with "transformer." (AnimaLoRASetup.py:62). So the full OT key is:
#   transformer.transformer_blocks.{i}.{module}.lora_down.weight  etc.
def _anima_ot_module(slot: Int) -> String:
    if slot == SLOT_SA_Q:
        return String("attn1.to_q")
    elif slot == SLOT_SA_K:
        return String("attn1.to_k")
    elif slot == SLOT_SA_V:
        return String("attn1.to_v")
    elif slot == SLOT_SA_O:
        return String("attn1.to_out.0")
    elif slot == SLOT_CA_Q:
        return String("attn2.to_q")
    elif slot == SLOT_CA_K:
        return String("attn2.to_k")
    elif slot == SLOT_CA_V:
        return String("attn2.to_v")
    elif slot == SLOT_CA_O:
        return String("attn2.to_out.0")
    elif slot == SLOT_MLP1:
        return String("ff.net.0.proj")
    return String("ff.net.2")


def _anima_ot_prefix(block_idx: Int, slot: Int) -> String:
    return (String("transformer.transformer_blocks.") + String(block_idx)
            + String(".") + _anima_ot_module(slot))


def anima_ot_prefixes(num_blocks: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_blocks):
        for s in range(ANIMA_SLOTS):
            out.append(_anima_ot_prefix(bi, s))
    return out^


# ── SAVE every adapter in OneTrainer diffusers key naming (OT-loadable) ───────
# Writes lora_down.weight [rank,in] / lora_up.weight [out,rank] / alpha (scalar)
# per module, byte-exact to AnimaLoRASaver's transformer_lora.state_dict() layout.
def save_anima_lora_ot(
    set: AnimaLoraSet, alpha: Float32, path: String, ctx: DeviceContext
) raises -> Int:
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    for bi in range(set.num_blocks):
        for s in range(ANIMA_SLOTS):
            var ad = set.ad[bi * ANIMA_SLOTS + s].copy()
            var pfx = _anima_ot_prefix(bi, s)
            if len(ad.a) != ad.rank * ad.in_f:
                raise Error(String("save_anima_lora_ot: A numel mismatch for ") + pfx)
            if len(ad.b) != ad.out_f * ad.rank:
                raise Error(String("save_anima_lora_ot: B numel mismatch for ") + pfx)
            var a_sh = List[Int](); a_sh.append(ad.rank); a_sh.append(ad.in_f)
            var b_sh = List[Int](); b_sh.append(ad.out_f); b_sh.append(ad.rank)
            var al_sh = List[Int](); al_sh.append(1)
            var al_v = List[Float32](); al_v.append(alpha)
            names.append(pfx + String(".lora_down.weight"))
            tensors.append(ArcPointer(Tensor.from_host_bf16(ad.a.copy(), a_sh^, ctx)))
            names.append(pfx + String(".lora_up.weight"))
            tensors.append(ArcPointer(Tensor.from_host_bf16(ad.b.copy(), b_sh^, ctx)))
            names.append(pfx + String(".alpha"))
            tensors.append(ArcPointer(Tensor.from_host(al_v^, al_sh^, STDtype.F32, ctx)))
    save_safetensors(names, tensors, path, ctx)
    return set.num_blocks * ANIMA_SLOTS


# ── RESUME: load the adapter A/B back from a save_anima_lora file ────────────
# AdamW moments are ZEROED (resume them from a loop TrainState checkpoint). The
# returned set carries the SAME flat order build_anima_lora_set produces.
def load_anima_lora_resume(
    num_blocks: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> AnimaLoraSet:
    var prefixes = anima_lora_prefixes(num_blocks)
    var scale = alpha / Float32(rank)
    var named = load_lora_for_resume(prefixes, scale, path, ctx)   # flat order
    var ad = List[LoraAdapter]()
    for i in range(num_blocks * ANIMA_SLOTS):
        ad.append(named[i].adapter.copy())
    return AnimaLoraSet(ad^, num_blocks, rank)


# ── FAITHFUL-RESUME state sidecar: A/B + AdamW moments (ma/va/mb/vb) ──────────
# Mirrors save_klein_lora_state / load_klein_lora_state. The PEFT file
# (save_anima_lora) stays plain external-compatible LoRA; this `.opt`/state file
# additionally persists the AdamW moments so a resumed run does NOT zero momentum.
# The AdamW bias-correction step `t` is reconstructed by the trainer from
# start_step (it passes t = step+1, continuous across the resume boundary), so it
# is not stored here — only the load-bearing m/v carry across.
def save_anima_lora_state(set: AnimaLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLora]()
    for bi in range(set.num_blocks):
        for s in range(ANIMA_SLOTS):
            named.append(NamedLora(
                _anima_lora_prefix(bi, s),
                set.ad[bi * ANIMA_SLOTS + s].copy(),
            ))
    return save_lora_train_state(named, path, ctx)


def load_anima_lora_state(
    num_blocks: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> AnimaLoraSet:
    var prefixes = anima_lora_prefixes(num_blocks)
    var scale = alpha / Float32(rank)
    var named = load_lora_train_state(prefixes, scale, path, ctx)   # A/B + m/v
    var ad = List[LoraAdapter]()
    for i in range(num_blocks * ANIMA_SLOTS):
        ad.append(named[i].adapter.copy())
    return AnimaLoraSet(ad^, num_blocks, rank)
