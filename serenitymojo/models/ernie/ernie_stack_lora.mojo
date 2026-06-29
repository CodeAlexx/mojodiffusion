# serenitymojo/models/ernie/ernie_stack_lora.mojo
#
# ERNIE-Image FULL DiT STACK *WITH LoRA* on every trained projection: forward
# (saving ckpt-inputs) + full-depth backward (training) that uses the parity-
# verified per-block LoRA variants (models/ernie/lora_block.mojo), COLLECTS every
# adapter's d_A/d_B, and supports an AdamW step + a OneTrainer raw LoRA save
# across all 7×num_layers adapters. This file COMPOSES; it rebuilds NOTHING.
#
# WHAT IS ALREADY PROVEN (cos>=0.999 vs torch) AND ONLY REUSED HERE
#   * models/ernie/block.mojo : base block fwd+bwd (19/19 cos>=0.99999 vs torch).
#   * models/ernie/ernie_stack.mojo : the BASE full-stack composition (streamed
#     block-offload path proven finite + no-OOM at 36 layers). THIS FILE is that
#     file with the base per-block calls swapped for the LoRA variants + LoRA-grad
#     collection.
#   * models/ernie/lora_block.mojo : ernie_block_lora_forward/backward (reduces to
#     base when adapters absent; LoRA d_x summed into the projection-input grad).
#   * training/{lora_save, train_step, optim} : LoraAdapter, _lora_adamw, save_lora_onetrainer.
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

from std.gpu.host import DeviceContext, HostBuffer
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx
from serenitymojo.ops.norm_backward import layer_norm_backward, layer_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.tensor_algebra import concat, slice

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.ernie.weights import (
    ErnieBlockWeights, ErnieStackBase, load_ernie_block_weights,
    load_ernie_block_weights_bf16_normf32,
)
from serenitymojo.models.ernie.block import (
    ErnieModVecs, ErnieBlockSaved, ErnieBlockGrads, ernie_block_forward,
)
from serenitymojo.models.ernie.lora_block import (
    ErnieLoraAdapterDevice, ErnieBlockLora, ErnieBlockLoraDevice,
    ErnieBlockLoraGrads, ERNIE_SLOTS,
    SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_GATE, SLOT_UP, SLOT_DOWN,
    ernie_lora_adapter_to_device,
    ernie_block_lora_forward, ernie_block_lora_backward,
    ernie_block_lora_forward_device_tensor, ernie_block_lora_backward_device_tensors,
    ernie_block_direct_lycoris_forward_device_tensor,
    ernie_block_direct_lycoris_backward_device_tensors,
    build_ernie_text_pad_mask_fwd, build_ernie_text_pad_mask_bwd,
)
from serenitymojo.models.zimage.lora_block import (
    ZImageBlockDirectLycoris, ZImageDirectProjectionGrad,
    ZImageBlockDirectGrads, ZIMAGE_DIRECT_ALGO_DORA,
    ZIMAGE_DIRECT_ALGO_OFT,
)
from serenitymojo.models.ernie.ernie_stack import (
    ErnieStackForward, _zeros, _ones, _t, _linear_wdev, _linear_wdev_bias,
    _concat_img_txt, _split_img_txt, _add_lists, saved_x_out,
)

from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.dora_adapter import DoRAGrads
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectDoRAGrads,
    FlatDirectOFTSet, FlatDirectOFTGrads,
    empty_flat_direct_dora_set, empty_flat_direct_oft_set,
)
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_onetrainer, load_lora_for_resume,
    save_lora_train_state, load_lora_train_state,
)


comptime TArc = ArcPointer[Tensor]


def _t_bf16(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


def _t_as(
    vals: List[Float32], var shape: List[Int], dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    return Tensor.from_host(vals, shape^, dtype, ctx)


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


struct ErnieLoraDeviceSet(Copyable, Movable):
    var ad: List[ErnieLoraAdapterDevice]
    var num_layers: Int
    var rank: Int

    def __init__(
        out self, var ad: List[ErnieLoraAdapterDevice], num_layers: Int, rank: Int
    ):
        self.ad = ad^
        self.num_layers = num_layers
        self.rank = rank


def ernie_lora_set_to_device(
    set: ErnieLoraSet, dtype: STDtype, ctx: DeviceContext
) raises -> ErnieLoraDeviceSet:
    var ad = List[ErnieLoraAdapterDevice]()
    var n = set.num_layers * ERNIE_SLOTS
    for i in range(n):
        ad.append(ernie_lora_adapter_to_device(set.ad[i], dtype, ctx))
    return ErnieLoraDeviceSet(ad^, set.num_layers, set.rank)


def _block_lora_dev_for(set: ErnieLoraDeviceSet, bi: Int) -> ErnieBlockLoraDevice:
    var base = bi * ERNIE_SLOTS
    return ErnieBlockLoraDevice(
        set.ad[base + SLOT_Q].copy(),
        set.ad[base + SLOT_K].copy(),
        set.ad[base + SLOT_V].copy(),
        set.ad[base + SLOT_O].copy(),
        set.ad[base + SLOT_GATE].copy(),
        set.ad[base + SLOT_UP].copy(),
        set.ad[base + SLOT_DOWN].copy(),
    )


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


struct _ErnieHostGradLists(Movable):
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


def _ernie_tensor_grads_to_host(
    indices: List[Int], d_a_t: List[TArc], d_b_t: List[TArc],
    total_slots: Int, ctx: DeviceContext,
) raises -> _ErnieHostGradLists:
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
    var a_off = List[Int]()
    var a_num = List[Int]()
    var b_off = List[Int]()
    var b_num = List[Int]()
    var cursor = 0
    for i in range(len(a_f32)):
        a_off.append(cursor)
        a_num.append(a_f32[i][].numel())
        var dst = host.create_sub_buffer[DType.uint8](cursor, a_f32[i][].nbytes())
        ctx.enqueue_copy(dst_buf=dst, src_buf=a_f32[i][].buf)
        cursor += a_f32[i][].nbytes()
    for i in range(len(b_f32)):
        b_off.append(cursor)
        b_num.append(b_f32[i][].numel())
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
    return _ErnieHostGradLists(d_a_flat^, d_b_flat^)


def _ernie_direct_slots_per_block(targets: Int) raises -> Int:
    if targets == 1:
        return 4
    if targets == 2:
        return ERNIE_SLOTS
    raise Error("ERNIE direct LyCORIS targets must be 1(attn) or 2(all)")


def _ernie_direct_dora_for(
    dora: FlatDirectDoRASet, bi: Int, targets: Int,
) raises -> ZImageBlockDirectLycoris:
    return ZImageBlockDirectLycoris(
        ZIMAGE_DIRECT_ALGO_DORA, dora.copy(), empty_flat_direct_oft_set(),
        bi * _ernie_direct_slots_per_block(targets), targets,
    )


def _ernie_direct_oft_for(
    oft: FlatDirectOFTSet, bi: Int, targets: Int,
) raises -> ZImageBlockDirectLycoris:
    return ZImageBlockDirectLycoris(
        ZIMAGE_DIRECT_ALGO_OFT, empty_flat_direct_dora_set(), oft.copy(),
        bi * _ernie_direct_slots_per_block(targets), targets,
    )


def _ernie_empty_dora_grads_for(set: FlatDirectDoRASet, slot: Int) -> DoRAGrads:
    ref ad = set.ad[slot]
    return DoRAGrads(
        _zeros(len(ad.a)), _zeros(len(ad.b)), _zeros(len(ad.m)), List[Float32](),
    )


def _ernie_direct_dora_zero_grads(set: FlatDirectDoRASet) -> FlatDirectDoRAGrads:
    var out = List[DoRAGrads]()
    for i in range(len(set.ad)):
        out.append(_ernie_empty_dora_grads_for(set, i))
    return FlatDirectDoRAGrads(out^)


def _ernie_direct_oft_zero_grads(set: FlatDirectOFTSet) -> FlatDirectOFTGrads:
    var out = List[List[Float32]]()
    for i in range(len(set.ad)):
        out.append(_zeros(len(set.ad[i].vec)))
    return FlatDirectOFTGrads(out^)


def _ernie_nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


def _ernie_scatter_dora_slot(
    mut grads: FlatDirectDoRAGrads, slot: Int, g: ZImageDirectProjectionGrad,
) -> Int:
    if slot < 0:
        return 0
    grads.g[slot] = DoRAGrads(
        g.d_a.copy(), g.d_b.copy(), g.d_m.copy(), List[Float32](),
    )
    return (
        _ernie_nonfinite(g.d_a)
        + _ernie_nonfinite(g.d_b)
        + _ernie_nonfinite(g.d_m)
    )


def _ernie_scatter_oft_slot(
    mut grads: FlatDirectOFTGrads, slot: Int, g: ZImageDirectProjectionGrad,
) -> Int:
    if slot < 0:
        return 0
    grads.d_vec[slot] = g.d_vec.copy()
    return _ernie_nonfinite(g.d_vec)


def _ernie_scatter_dora_block(
    mut grads: FlatDirectDoRAGrads, direct: ZImageBlockDirectLycoris,
    bg: ZImageBlockDirectGrads,
) -> Int:
    var bad = 0
    bad += _ernie_scatter_dora_slot(grads, direct.q_slot, bg.q)
    bad += _ernie_scatter_dora_slot(grads, direct.k_slot, bg.k)
    bad += _ernie_scatter_dora_slot(grads, direct.v_slot, bg.v)
    bad += _ernie_scatter_dora_slot(grads, direct.o_slot, bg.out_proj)
    bad += _ernie_scatter_dora_slot(grads, direct.w1_slot, bg.w1)
    bad += _ernie_scatter_dora_slot(grads, direct.w3_slot, bg.w3)
    bad += _ernie_scatter_dora_slot(grads, direct.w2_slot, bg.w2)
    return bad


def _ernie_scatter_oft_block(
    mut grads: FlatDirectOFTGrads, direct: ZImageBlockDirectLycoris,
    bg: ZImageBlockDirectGrads,
) -> Int:
    var bad = 0
    bad += _ernie_scatter_oft_slot(grads, direct.q_slot, bg.q)
    bad += _ernie_scatter_oft_slot(grads, direct.k_slot, bg.k)
    bad += _ernie_scatter_oft_slot(grads, direct.v_slot, bg.v)
    bad += _ernie_scatter_oft_slot(grads, direct.o_slot, bg.out_proj)
    bad += _ernie_scatter_oft_slot(grads, direct.w1_slot, bg.w1)
    bad += _ernie_scatter_oft_slot(grads, direct.w3_slot, bg.w3)
    bad += _ernie_scatter_oft_slot(grads, direct.w2_slot, bg.w2)
    return bad


struct ErnieDirectDoRABackward(Movable):
    var grads: FlatDirectDoRAGrads
    var nonfinite_grads: Int

    def __init__(out self, var grads: FlatDirectDoRAGrads, nonfinite_grads: Int):
        self.grads = grads^
        self.nonfinite_grads = nonfinite_grads


struct ErnieDirectOFTBackward(Movable):
    var grads: FlatDirectOFTGrads
    var nonfinite_grads: Int

    def __init__(out self, var grads: FlatDirectOFTGrads, nonfinite_grads: Int):
        self.grads = grads^
        self.nonfinite_grads = nonfinite_grads


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
    # Streamed LoRA training freezes the base stack. Only d_x_out is needed to
    # continue the chain; final_lin d_W would be computed/read then discarded.
    var d_x_out = linear_backward_dx(
        _t(d_patches, [S, out_ch], ctx), base.final_lin_w[],
        S, D, out_ch, ctx,
    ).to_host(ctx)
    var d_final_lin = List[Float32]()
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

    # Input token gradients are not consumed by the LoRA optimizer in the real
    # streamed trainer. Keep placeholders so the result shape stays compatible.
    var d_img_tokens = List[Float32]()
    var d_txt_tokens = List[Float32]()

    return ErnieLoraGrads(
        d_a_flat^, d_b_flat^,
        d_img_tokens^, d_txt_tokens^,
        d_shared_mod^, d_f_scale^, d_f_shift^, d_final_lin^,
        nonfinite,
    )


def ernie_stack_lora_forward_streamed_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, st: ShardedSafeTensors,
    lora: ErnieLoraDeviceSet, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieStackForward:
    var num_layers = lora.num_layers

    var img_bias = Optional[Tensor](base.patch_b[].clone(ctx))
    var img = linear(
        _t_bf16(img_tokens, [N_IMG, in_ch], ctx), base.patch_w[], img_bias^, ctx
    )
    var no_txt_bias = Optional[Tensor](None)
    var txt = linear(_t_bf16(txt_tokens, [N_TXT, text_in], ctx), base.text_proj[], no_txt_bias^, ctx)
    var x = concat(0, ctx, img, txt)
    var x_arc = TArc(x^)

    var blk_x_in = List[TArc]()
    for bi in range(num_layers):
        blk_x_in.append(x_arc.copy())
        var w = load_ernie_block_weights_bf16_normf32(st, bi, ctx)
        var bl = _block_lora_dev_for(lora, bi)
        var fwd = ernie_block_lora_forward_device_tensor[H, Dh, S](
            x_arc.copy(), w, mv, bl, cos, sin, D, F, eps, ctx,
        )
        x_arc = fwd.out.copy()

    var ln_x = layer_norm(
        x_arc[], _t_bf16(_ones(D), [D], ctx), _t_bf16(_zeros(D), [D], ctx), eps, ctx,
    )
    var x_out = modulate(
        ln_x, _t_bf16(f_scale.copy(), [D], ctx), _t_bf16(f_shift.copy(), [D], ctx), ctx,
    )
    var final_bias = Optional[Tensor](base.final_lin_b[].clone(ctx))
    var patches = linear(x_out, base.final_lin_w[], final_bias^, ctx)
    var out_img = slice(patches, 0, 0, N_IMG, ctx).to_host(ctx)

    return ErnieStackForward(
        out_img^, List[Float32](), List[Float32](), blk_x_in^,
        x_arc.copy(), TArc(ln_x^),
    )


def ernie_stack_lora_predict_streamed_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, st: ShardedSafeTensors,
    lora: ErnieLoraDeviceSet, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> List[Float32]:
    var num_layers = lora.num_layers

    var img_bias = Optional[Tensor](base.patch_b[].clone(ctx))
    var img = linear(
        _t_bf16(img_tokens, [N_IMG, in_ch], ctx), base.patch_w[], img_bias^, ctx
    )
    var no_txt_bias = Optional[Tensor](None)
    var txt = linear(_t_bf16(txt_tokens, [N_TXT, text_in], ctx), base.text_proj[], no_txt_bias^, ctx)
    var x = concat(0, ctx, img, txt)
    var x_arc = TArc(x^)

    for bi in range(num_layers):
        var w = load_ernie_block_weights_bf16_normf32(st, bi, ctx)
        var bl = _block_lora_dev_for(lora, bi)
        var fwd = ernie_block_lora_forward_device_tensor[H, Dh, S](
            x_arc.copy(), w, mv, bl, cos, sin, D, F, eps, ctx,
        )
        x_arc = fwd.out.copy()

    var ln_x = layer_norm(
        x_arc[], _t_bf16(_ones(D), [D], ctx), _t_bf16(_zeros(D), [D], ctx), eps, ctx,
    )
    var x_out = modulate(
        ln_x, _t_bf16(f_scale.copy(), [D], ctx), _t_bf16(f_shift.copy(), [D], ctx), ctx,
    )
    var final_bias = Optional[Tensor](base.final_lin_b[].clone(ctx))
    var patches = linear(x_out, base.final_lin_w[], final_bias^, ctx)
    var out_img = slice(patches, 0, 0, N_IMG, ctx).to_host(ctx)
    return out_img^


def ernie_stack_lora_backward_streamed_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, st: ShardedSafeTensors,
    lora: ErnieLoraDeviceSet, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    saved: ErnieStackForward,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ErnieLoraGrads:
    var num_layers = lora.num_layers

    var d_img = _t_bf16(d_out, [N_IMG, out_ch], ctx)
    var d_txt = _t_bf16(_zeros(N_TXT * out_ch), [N_TXT, out_ch], ctx)
    var d_patches = concat(0, ctx, d_img, d_txt)

    var d_x_out = linear_backward_dx(d_patches, base.final_lin_w[], S, D, out_ch, ctx)
    var mbf = modulate_backward(
        d_x_out, saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_f_scale = mbf.d_scale.to_host(ctx)
    var d_f_shift = mbf.d_shift.to_host(ctx)
    var d_x_final = layer_norm_backward_dx(
        mbf.d_x, saved.x_final[], _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_x_arc = TArc(d_x_final^)

    var d_shared_mod = _zeros(6 * D)
    var grad_indices = List[Int]()
    var d_a_t = List[TArc]()
    var d_b_t = List[TArc]()

    var bi = num_layers - 1
    while bi >= 0:
        var w = load_ernie_block_weights_bf16_normf32(st, bi, ctx)
        var bl = _block_lora_dev_for(lora, bi)
        var refwd = ernie_block_lora_forward_device_tensor[H, Dh, S](
            saved.blk_x_in[bi].copy(), w, mv, bl, cos, sin, D, F, eps, ctx,
        )
        var bg = ernie_block_lora_backward_device_tensors[H, Dh, S](
            d_x_arc[], w, mv, bl, refwd.saved, cos, sin, D, F, eps, ctx,
        )
        d_x_arc = bg.d_x.copy()
        d_shared_mod = _add_lists(d_shared_mod, bg.d_shared_mod)
        var base_idx = bi * ERNIE_SLOTS
        for s in range(ERNIE_SLOTS):
            grad_indices.append(base_idx + s)
            d_a_t.append(bg.d_a[s].copy())
            d_b_t.append(bg.d_b[s].copy())
        bi -= 1

    var host_grads = _ernie_tensor_grads_to_host(
        grad_indices, d_a_t, d_b_t, num_layers * ERNIE_SLOTS, ctx,
    )
    var nonfinite = 0
    for i in range(len(grad_indices)):
        var idx = grad_indices[i]
        nonfinite += _nonfinite(host_grads.d_a[idx]) + _nonfinite(host_grads.d_b[idx])

    return ErnieLoraGrads(
        host_grads.d_a.copy(), host_grads.d_b.copy(),
        List[Float32](), List[Float32](),
        d_shared_mod^, d_f_scale^, d_f_shift^, List[Float32](),
        nonfinite,
    )


def ernie_stack_lora_forward_resident_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, blocks: List[ErnieBlockWeights],
    lora: ErnieLoraDeviceSet, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    text_real_len: Int = -1,  # real text token count; <0 or >=N_TXT = no text-pad mask
) raises -> ErnieStackForward:
    var num_layers = len(blocks)

    var img_bias = Optional[Tensor](base.patch_b[].clone(ctx))
    var img = linear(
        _t_bf16(img_tokens, [N_IMG, in_ch], ctx), base.patch_w[], img_bias^, ctx
    )
    var no_txt_bias = Optional[Tensor](None)
    var txt = linear(_t_bf16(txt_tokens, [N_TXT, text_in], ctx), base.text_proj[], no_txt_bias^, ctx)
    var x = concat(0, ctx, img, txt)
    var x_arc = TArc(x^)

    # Build the text-pad attention mask ONCE (constant across the 36 blocks).
    # OneTrainer masks padded text keys [N_IMG+real_len, S); built in x's compute
    # dtype so `sdpa` accepts it. Skipped when text_real_len covers all of N_TXT.
    # Held as an ArcPointer so the one allocation is shared (copied) across blocks.
    var fwd_mask = Optional[TArc](None)
    if text_real_len >= 0 and text_real_len < N_TXT:
        fwd_mask = Optional[TArc](TArc(
            build_ernie_text_pad_mask_fwd(H, S, N_IMG, text_real_len, x_arc[].dtype(), ctx)
        ))

    var blk_x_in = List[TArc]()
    for bi in range(num_layers):
        blk_x_in.append(x_arc.copy())
        var bl = _block_lora_dev_for(lora, bi)
        var fwd = ernie_block_lora_forward_device_tensor[H, Dh, S](
            x_arc.copy(), blocks[bi], mv, bl, cos, sin, D, F, eps, ctx,
            fwd_mask.copy(),
        )
        x_arc = fwd.out.copy()

    var ln_x = layer_norm(
        x_arc[], _t_bf16(_ones(D), [D], ctx), _t_bf16(_zeros(D), [D], ctx), eps, ctx,
    )
    var x_out = modulate(
        ln_x, _t_bf16(f_scale.copy(), [D], ctx), _t_bf16(f_shift.copy(), [D], ctx), ctx,
    )
    var final_bias = Optional[Tensor](base.final_lin_b[].clone(ctx))
    var patches = linear(x_out, base.final_lin_w[], final_bias^, ctx)
    var out_img = slice(patches, 0, 0, N_IMG, ctx).to_host(ctx)

    return ErnieStackForward(
        out_img^, List[Float32](), List[Float32](), blk_x_in^,
        x_arc.copy(), TArc(ln_x^),
    )


def ernie_stack_lora_predict_resident_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, blocks: List[ErnieBlockWeights],
    lora: ErnieLoraDeviceSet, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> List[Float32]:
    var num_layers = len(blocks)

    var img_bias = Optional[Tensor](base.patch_b[].clone(ctx))
    var img = linear(
        _t_bf16(img_tokens, [N_IMG, in_ch], ctx), base.patch_w[], img_bias^, ctx
    )
    var no_txt_bias = Optional[Tensor](None)
    var txt = linear(_t_bf16(txt_tokens, [N_TXT, text_in], ctx), base.text_proj[], no_txt_bias^, ctx)
    var x = concat(0, ctx, img, txt)
    var x_arc = TArc(x^)

    for bi in range(num_layers):
        var bl = _block_lora_dev_for(lora, bi)
        var fwd = ernie_block_lora_forward_device_tensor[H, Dh, S](
            x_arc.copy(), blocks[bi], mv, bl, cos, sin, D, F, eps, ctx,
        )
        x_arc = fwd.out.copy()

    var ln_x = layer_norm(
        x_arc[], _t_bf16(_ones(D), [D], ctx), _t_bf16(_zeros(D), [D], ctx), eps, ctx,
    )
    var x_out = modulate(
        ln_x, _t_bf16(f_scale.copy(), [D], ctx), _t_bf16(f_shift.copy(), [D], ctx), ctx,
    )
    var final_bias = Optional[Tensor](base.final_lin_b[].clone(ctx))
    var patches = linear(x_out, base.final_lin_w[], final_bias^, ctx)
    var out_img = slice(patches, 0, 0, N_IMG, ctx).to_host(ctx)
    return out_img^


def ernie_stack_lora_backward_resident_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, blocks: List[ErnieBlockWeights],
    lora: ErnieLoraDeviceSet, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    saved: ErnieStackForward,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    text_real_len: Int = -1,  # real text token count; <0 or >=N_TXT = no text-pad mask
) raises -> ErnieLoraGrads:
    var num_layers = len(blocks)

    # Build BOTH text-pad masks once (constant across the 36 blocks). The
    # per-block recompute reruns the FORWARD (needs the [1,H,S,S] q-dtype mask);
    # the per-block backward needs the [H*S,S] F32 mask for sdpa_backward_masked.
    # Held as ArcPointers so each one allocation is shared (copied) across blocks.
    var has_mask = text_real_len >= 0 and text_real_len < N_TXT
    var fwd_mask = Optional[TArc](None)
    var bwd_mask = Optional[TArc](None)
    if has_mask:
        bwd_mask = Optional[TArc](TArc(
            build_ernie_text_pad_mask_bwd(H, S, N_IMG, text_real_len, ctx)
        ))

    var d_img = _t_bf16(d_out, [N_IMG, out_ch], ctx)
    var d_txt = _t_bf16(_zeros(N_TXT * out_ch), [N_TXT, out_ch], ctx)
    var d_patches = concat(0, ctx, d_img, d_txt)

    var d_x_out = linear_backward_dx(d_patches, base.final_lin_w[], S, D, out_ch, ctx)
    var mbf = modulate_backward(
        d_x_out, saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_f_scale = mbf.d_scale.to_host(ctx)
    var d_f_shift = mbf.d_shift.to_host(ctx)
    var d_x_final = layer_norm_backward_dx(
        mbf.d_x, saved.x_final[], _t(_ones(D), [D], ctx), eps, ctx,
    )
    var d_x_arc = TArc(d_x_final^)

    var d_shared_mod = _zeros(6 * D)
    var grad_indices = List[Int]()
    var d_a_t = List[TArc]()
    var d_b_t = List[TArc]()

    # Forward-recompute mask in the recompute input's compute dtype (matches the
    # original forward's fwd_mask). blk_x_in carries the per-block recompute input.
    if has_mask and num_layers > 0:
        fwd_mask = Optional[TArc](TArc(
            build_ernie_text_pad_mask_fwd(
                H, S, N_IMG, text_real_len, saved.blk_x_in[0][].dtype(), ctx
            )
        ))

    var bi = num_layers - 1
    while bi >= 0:
        var bl = _block_lora_dev_for(lora, bi)
        var refwd = ernie_block_lora_forward_device_tensor[H, Dh, S](
            saved.blk_x_in[bi].copy(), blocks[bi], mv, bl, cos, sin, D, F, eps, ctx,
            fwd_mask.copy(),
        )
        var bg = ernie_block_lora_backward_device_tensors[H, Dh, S](
            d_x_arc[], blocks[bi], mv, bl, refwd.saved, cos, sin, D, F, eps, ctx,
            bwd_mask.copy(),
        )
        d_x_arc = bg.d_x.copy()
        d_shared_mod = _add_lists(d_shared_mod, bg.d_shared_mod)
        var base_idx = bi * ERNIE_SLOTS
        for s in range(ERNIE_SLOTS):
            grad_indices.append(base_idx + s)
            d_a_t.append(bg.d_a[s].copy())
            d_b_t.append(bg.d_b[s].copy())
        bi -= 1

    var host_grads = _ernie_tensor_grads_to_host(
        grad_indices, d_a_t, d_b_t, num_layers * ERNIE_SLOTS, ctx,
    )
    var nonfinite = 0
    for i in range(len(grad_indices)):
        var idx = grad_indices[i]
        nonfinite += _nonfinite(host_grads.d_a[idx]) + _nonfinite(host_grads.d_b[idx])

    return ErnieLoraGrads(
        host_grads.d_a.copy(), host_grads.d_b.copy(),
        List[Float32](), List[Float32](),
        d_shared_mod^, d_f_scale^, d_f_shift^, List[Float32](),
        nonfinite,
    )


def ernie_stack_direct_dora_forward_resident_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, blocks: List[ErnieBlockWeights],
    dora: FlatDirectDoRASet, targets: Int, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    text_real_len: Int = -1,
) raises -> ErnieStackForward:
    var num_layers = len(blocks)

    var img_bias = Optional[Tensor](base.patch_b[].clone(ctx))
    var img = linear(
        _t_bf16(img_tokens, [N_IMG, in_ch], ctx), base.patch_w[], img_bias^, ctx
    )
    var no_txt_bias = Optional[Tensor](None)
    var txt = linear(
        _t_bf16(txt_tokens, [N_TXT, text_in], ctx), base.text_proj[],
        no_txt_bias^, ctx,
    )
    var x = concat(0, ctx, img, txt)
    var x_arc = TArc(x^)

    var fwd_mask = Optional[TArc](None)
    if text_real_len >= 0 and text_real_len < N_TXT:
        fwd_mask = Optional[TArc](TArc(
            build_ernie_text_pad_mask_fwd(H, S, N_IMG, text_real_len, x_arc[].dtype(), ctx)
        ))

    var blk_x_in = List[TArc]()
    for bi in range(num_layers):
        blk_x_in.append(x_arc.copy())
        var direct = _ernie_direct_dora_for(dora, bi, targets)
        var fwd = ernie_block_direct_lycoris_forward_device_tensor[H, Dh, S](
            x_arc.copy(), blocks[bi], mv, direct, cos, sin, D, F, eps, ctx,
            fwd_mask.copy(),
        )
        x_arc = fwd.out.copy()

    var ln_x = layer_norm(
        x_arc[], _t_bf16(_ones(D), [D], ctx), _t_bf16(_zeros(D), [D], ctx), eps, ctx,
    )
    var x_out = modulate(
        ln_x, _t_bf16(f_scale.copy(), [D], ctx),
        _t_bf16(f_shift.copy(), [D], ctx), ctx,
    )
    var final_bias = Optional[Tensor](base.final_lin_b[].clone(ctx))
    var patches = linear(x_out, base.final_lin_w[], final_bias^, ctx)
    var out_img = slice(patches, 0, 0, N_IMG, ctx).to_host(ctx)

    return ErnieStackForward(
        out_img^, List[Float32](), List[Float32](), blk_x_in^,
        x_arc.copy(), TArc(ln_x^),
    )


def ernie_stack_direct_oft_forward_resident_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, blocks: List[ErnieBlockWeights],
    oft: FlatDirectOFTSet, targets: Int, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    text_real_len: Int = -1,
) raises -> ErnieStackForward:
    var num_layers = len(blocks)

    var img_bias = Optional[Tensor](base.patch_b[].clone(ctx))
    var img = linear(
        _t_bf16(img_tokens, [N_IMG, in_ch], ctx), base.patch_w[], img_bias^, ctx
    )
    var no_txt_bias = Optional[Tensor](None)
    var txt = linear(
        _t_bf16(txt_tokens, [N_TXT, text_in], ctx), base.text_proj[],
        no_txt_bias^, ctx,
    )
    var x = concat(0, ctx, img, txt)
    var x_arc = TArc(x^)

    var fwd_mask = Optional[TArc](None)
    if text_real_len >= 0 and text_real_len < N_TXT:
        fwd_mask = Optional[TArc](TArc(
            build_ernie_text_pad_mask_fwd(H, S, N_IMG, text_real_len, x_arc[].dtype(), ctx)
        ))

    var blk_x_in = List[TArc]()
    for bi in range(num_layers):
        blk_x_in.append(x_arc.copy())
        var direct = _ernie_direct_oft_for(oft, bi, targets)
        var fwd = ernie_block_direct_lycoris_forward_device_tensor[H, Dh, S](
            x_arc.copy(), blocks[bi], mv, direct, cos, sin, D, F, eps, ctx,
            fwd_mask.copy(),
        )
        x_arc = fwd.out.copy()

    var ln_x = layer_norm(
        x_arc[], _t_bf16(_ones(D), [D], ctx), _t_bf16(_zeros(D), [D], ctx), eps, ctx,
    )
    var x_out = modulate(
        ln_x, _t_bf16(f_scale.copy(), [D], ctx),
        _t_bf16(f_shift.copy(), [D], ctx), ctx,
    )
    var final_bias = Optional[Tensor](base.final_lin_b[].clone(ctx))
    var patches = linear(x_out, base.final_lin_w[], final_bias^, ctx)
    var out_img = slice(patches, 0, 0, N_IMG, ctx).to_host(ctx)

    return ErnieStackForward(
        out_img^, List[Float32](), List[Float32](), blk_x_in^,
        x_arc.copy(), TArc(ln_x^),
    )


def ernie_stack_direct_dora_backward_resident_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, blocks: List[ErnieBlockWeights],
    dora: FlatDirectDoRASet, targets: Int, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    saved: ErnieStackForward,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    text_real_len: Int = -1,
) raises -> ErnieDirectDoRABackward:
    var num_layers = len(blocks)

    var has_mask = text_real_len >= 0 and text_real_len < N_TXT
    var fwd_mask = Optional[TArc](None)
    var bwd_mask = Optional[TArc](None)
    if has_mask:
        bwd_mask = Optional[TArc](TArc(
            build_ernie_text_pad_mask_bwd(H, S, N_IMG, text_real_len, ctx)
        ))

    var d_img = _t_bf16(d_out, [N_IMG, out_ch], ctx)
    var d_txt = _t_bf16(_zeros(N_TXT * out_ch), [N_TXT, out_ch], ctx)
    var d_patches = concat(0, ctx, d_img, d_txt)

    var d_x_out = linear_backward_dx(d_patches, base.final_lin_w[], S, D, out_ch, ctx)
    var mbf = modulate_backward(
        cast_tensor(d_x_out, saved.ln_x[].dtype(), ctx, False),
        saved.ln_x[], _t_as(f_scale.copy(), [D], saved.ln_x[].dtype(), ctx), ctx,
    )
    var d_x_final = layer_norm_backward_dx(
        mbf.d_x, saved.x_final[], _t_as(_ones(D), [D], saved.x_final[].dtype(), ctx), eps, ctx,
    )
    var d_x_arc = TArc(d_x_final^)

    var direct_grads = _ernie_direct_dora_zero_grads(dora)
    var nonfinite = 0

    if has_mask and num_layers > 0:
        fwd_mask = Optional[TArc](TArc(
            build_ernie_text_pad_mask_fwd(
                H, S, N_IMG, text_real_len, saved.blk_x_in[0][].dtype(), ctx
            )
        ))

    var bi = num_layers - 1
    while bi >= 0:
        var direct = _ernie_direct_dora_for(dora, bi, targets)
        var refwd = ernie_block_direct_lycoris_forward_device_tensor[H, Dh, S](
            saved.blk_x_in[bi].copy(), blocks[bi], mv, direct,
            cos, sin, D, F, eps, ctx, fwd_mask.copy(),
        )
        var bg = ernie_block_direct_lycoris_backward_device_tensors[H, Dh, S](
            d_x_arc[], blocks[bi], mv, direct, refwd.saved,
            cos, sin, D, F, eps, ctx, bwd_mask.copy(),
        )
        d_x_arc = bg.d_x.copy()
        nonfinite += _ernie_scatter_dora_block(direct_grads, direct, bg.grads)
        bi -= 1

    return ErnieDirectDoRABackward(direct_grads^, nonfinite)


def ernie_stack_direct_oft_backward_resident_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    img_tokens: List[Float32], txt_tokens: List[Float32],
    base: ErnieStackBase, blocks: List[ErnieBlockWeights],
    oft: FlatDirectOFTSet, targets: Int, mv: ErnieModVecs,
    f_scale: List[Float32], f_shift: List[Float32],
    cos: Tensor, sin: Tensor,
    saved: ErnieStackForward,
    D: Int, F: Int, in_ch: Int, text_in: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
    text_real_len: Int = -1,
) raises -> ErnieDirectOFTBackward:
    var num_layers = len(blocks)

    var has_mask = text_real_len >= 0 and text_real_len < N_TXT
    var fwd_mask = Optional[TArc](None)
    var bwd_mask = Optional[TArc](None)
    if has_mask:
        bwd_mask = Optional[TArc](TArc(
            build_ernie_text_pad_mask_bwd(H, S, N_IMG, text_real_len, ctx)
        ))

    var d_img = _t_bf16(d_out, [N_IMG, out_ch], ctx)
    var d_txt = _t_bf16(_zeros(N_TXT * out_ch), [N_TXT, out_ch], ctx)
    var d_patches = concat(0, ctx, d_img, d_txt)

    var d_x_out = linear_backward_dx(d_patches, base.final_lin_w[], S, D, out_ch, ctx)
    var mbf = modulate_backward(
        cast_tensor(d_x_out, saved.ln_x[].dtype(), ctx, False),
        saved.ln_x[], _t_as(f_scale.copy(), [D], saved.ln_x[].dtype(), ctx), ctx,
    )
    var d_x_final = layer_norm_backward_dx(
        mbf.d_x, saved.x_final[], _t_as(_ones(D), [D], saved.x_final[].dtype(), ctx), eps, ctx,
    )
    var d_x_arc = TArc(d_x_final^)

    var direct_grads = _ernie_direct_oft_zero_grads(oft)
    var nonfinite = 0

    if has_mask and num_layers > 0:
        fwd_mask = Optional[TArc](TArc(
            build_ernie_text_pad_mask_fwd(
                H, S, N_IMG, text_real_len, saved.blk_x_in[0][].dtype(), ctx
            )
        ))

    var bi = num_layers - 1
    while bi >= 0:
        var direct = _ernie_direct_oft_for(oft, bi, targets)
        var refwd = ernie_block_direct_lycoris_forward_device_tensor[H, Dh, S](
            saved.blk_x_in[bi].copy(), blocks[bi], mv, direct,
            cos, sin, D, F, eps, ctx, fwd_mask.copy(),
        )
        var bg = ernie_block_direct_lycoris_backward_device_tensors[H, Dh, S](
            d_x_arc[], blocks[bi], mv, direct, refwd.saved,
            cos, sin, D, F, eps, ctx, bwd_mask.copy(),
        )
        d_x_arc = bg.d_x.copy()
        nonfinite += _ernie_scatter_oft_block(direct_grads, direct, bg.grads)
        bi -= 1

    return ErnieDirectOFTBackward(direct_grads^, nonfinite)


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


# ── per-block OneTrainer raw prefix scheme ──────────────────────────────────
# OneTrainer ErnieLoRASetup wraps model.transformer with prefix "transformer".
# LoRAModuleWrapper then prefixes child module names from the diffusers
# transformer, yielding:
#   transformer.layers.<i>.self_attention.{to_q, to_k, to_v, to_out.0}
#   transformer.layers.<i>.mlp.{gate_proj, up_proj, linear_fc2}
# save_lora_onetrainer appends .alpha / .lora_down.weight / .lora_up.weight.
def _ernie_lora_prefix(block_idx: Int, slot: Int) -> String:
    var b = String("transformer.layers.") + String(block_idx)
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


def _ernie_named_loras(set: ErnieLoraSet) -> List[NamedLora]:
    var named = List[NamedLora]()
    for bi in range(set.num_layers):
        for s in range(ERNIE_SLOTS):
            named.append(NamedLora(
                _ernie_lora_prefix(bi, s),
                set.ad[bi * ERNIE_SLOTS + s].copy(),
            ))
    return named^


# ── SAVE every adapter as a OneTrainer raw LoRA safetensors ──────────────────
def save_ernie_lora(set: ErnieLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = _ernie_named_loras(set)
    return save_lora_onetrainer(named, path, ctx)


def save_ernie_lora_state(
    set: ErnieLoraSet, path: String, ctx: DeviceContext
) raises -> Int:
    var named = _ernie_named_loras(set)
    return save_lora_train_state(named, path, ctx)


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


def load_ernie_lora_state(
    num_layers: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> ErnieLoraSet:
    var prefixes = ernie_lora_prefixes(num_layers)
    var scale = alpha / Float32(rank)
    var named = load_lora_train_state(prefixes, scale, path, ctx)
    var ad = List[LoraAdapter]()
    for i in range(num_layers * ERNIE_SLOTS):
        ad.append(named[i].adapter.copy())
    return ErnieLoraSet(ad^, num_layers, rank)
