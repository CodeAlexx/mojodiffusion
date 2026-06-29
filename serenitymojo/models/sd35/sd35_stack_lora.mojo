# models/sd35/sd35_stack_lora.mojo
#
# SD3.5-Large FULL JOINT DiT STACK *WITH LoRA*, BLOCK-SWAP OFFLOAD.
#
# MIRRORS chroma_stack_lora.mojo's structure faithfully:
#   - SD35StackBase    : frozen resident base (embedders, final layer)
#   - SD35LoraSet      : all trainable LoRA adapters across all 38 joint blocks
#   - SD35LoraGradSet  : per-step gradients
#   - sd35_stack_lora_forward_offload  : full-depth forward, streaming blocks
#   - sd35_stack_lora_backward_offload : full-depth backward, REVERSE block stream
#
# SD3.5-LARGE SPECIFICS (vs chroma):
#   (1) MODULATION: per-block adaLN. Each block stream has its OWN modulation linear
#       (adaLN_modulation.1 [6D,D]) stored inside the streamed block. Conditioning
#       c = t_embed(sigma) + y_embed(pooled_clip) [D]. No frozen approximator table.
#   (2) JOINT BLOCKS ONLY: 38 joint blocks (context + x streams), no single-stream.
#       SD3.5 Large has NO dual attention blocks. SD3.5 Medium does (not ours).
#   (3) PATCHIFY: x_embedder.proj is a Conv2d(in=16,out=D,k=2,s=2) = linear map on
#       [N_IMG, 64] (64 = 16ch * 2*2 patch). We run it as a linear forward.
#   (4) FINAL LAYER: LayerNorm(no affine) -> silu(c)->final_ada_linear->chunk(shift,scale)
#       -> modulate -> final linear -> [N_IMG, 64].
#   (5) LoRA TARGETS: per joint block x 8 adapters:
#         ctx: attn.qkv (in=D,out=3D), attn.proj (in=D,out=D), mlp.fc1 (in=D,out=MLP), mlp.fc2 (in=MLP,out=D)
#         x:   same 4
#       Total: 38 * 8 = 304 adapters.
#   (6) MODULATION VECS: computed on the fly each block from the streamed adaLN weights;
#       saved for backward (like chroma saves dbl_img_mod / dbl_txt_mod).
#
# DTYPE STATUS: this training stack still uses host List[Float32] carriers around
# sd35_block.mojo and streamed weights. LoRA adapter storage remains BF16, but the
# SD3.5 block/stack path is not dtype-clean until the carriers become tensors.
#
# Mojo 0.26.x+: def not fn; Tensor move-only; host List[Float32] carriers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.embeddings import t_embedder
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm_backward import layer_norm_backward
from serenitymojo.ops.elementwise_backward import modulate_backward

from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

from serenitymojo.models.sd35.sd35_block import (
    JointBlockWeights, StreamWeights, ModVecs, JointBlockForward,
    StreamSaved, sd35_joint_block_forward, sd35_joint_block_backward,
    SD35BlockDirectLycoris, SD35DirectProjectionGrad,
    SD35JointBlockDirectGrads,
    SD35_DIRECT_ALGO_DORA, SD35_DIRECT_ALGO_OFT,
    sd35_joint_block_direct_lycoris_forward,
    sd35_joint_block_direct_lycoris_backward,
    Attn2Weights, DualBlockForward, DualXSaved,
    sd35_dual_joint_block_forward, sd35_dual_joint_block_backward,
    CtxPreForward, CtxPreSaved,
    sd35_context_preonly_forward, sd35_context_preonly_backward,
    sd35_context_preonly_direct_lycoris_forward,
    sd35_context_preonly_direct_lycoris_backward,
)
from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.dora_adapter import DoRAGrads
from serenitymojo.training.dora_save import NamedDoRA, save_dora_onetrainer
from serenitymojo.training.oft_onetrainer import OFTOTGrads
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectDoRAGrads, FlatDirectOFTSet, FlatDirectOFTGrads,
    empty_flat_direct_dora_set, empty_flat_direct_oft_set,
    flat_direct_dora_append_from_weight, flat_direct_oft_append,
    flat_direct_dora_grad_norm, flat_direct_dora_clip_grads,
    flat_direct_dora_adamw_step, flat_direct_dora_zero_leg_l1,
    flat_direct_dora_trainable_bytes,
    flat_direct_oft_grad_norm, flat_direct_oft_clip_grads,
    flat_direct_oft_adamw_step, flat_direct_oft_vec_l1,
    flat_direct_oft_trainable_bytes,
)
from serenitymojo.training.lora_save import (
    NamedLora,
    save_lora_peft,
    save_lora_train_state,
)


comptime TArc = ArcPointer[Tensor]


# ── Frozen stack-level resident base ─────────────────────────────────────────
# Stack-level embedder/final-layer weights stay resident as checkpoint-dtype
# tensors. The block stream below still has a separate host-F32 carrier blocker.
struct SD35StackBase(Copyable, Movable):
    # x_embedder: Conv2d [D,16,2,2] flattened to linear [D,64]
    var xe_w: TArc   # [D,64]
    var xe_b: TArc   # [D]
    # context_embedder: [D, 4096]
    var ce_w: TArc   # [D,4096]
    var ce_b: TArc   # [D]
    # t_embedder MLP: 256 -> D -> D
    var t_w0: TArc   # [D,256]
    var t_b0: TArc   # [D]
    var t_w2: TArc   # [D,D]
    var t_b2: TArc   # [D]
    # y_embedder (pooled CLIP): 2048 -> D -> D
    var y_w0: TArc   # [D,2048]
    var y_b0: TArc   # [D]
    var y_w2: TArc   # [D,D]
    var y_b2: TArc   # [D]
    # final_layer: adaLN [2D,D] + linear [64,D]
    var fl_ada_w: TArc  # [2D,D]
    var fl_ada_b: TArc  # [2D]
    var fl_lin_w: TArc  # [64,D]
    var fl_lin_b: TArc  # [64]
    # learned positional embedding [1, POS_MAX*POS_MAX, D] (center-cropped per res)
    var pos_embed: TArc

    def __init__(
        out self,
        var xe_w: TArc, var xe_b: TArc,
        var ce_w: TArc, var ce_b: TArc,
        var t_w0: TArc, var t_b0: TArc,
        var t_w2: TArc, var t_b2: TArc,
        var y_w0: TArc, var y_b0: TArc,
        var y_w2: TArc, var y_b2: TArc,
        var fl_ada_w: TArc, var fl_ada_b: TArc,
        var fl_lin_w: TArc, var fl_lin_b: TArc,
        var pos_embed: TArc,
    ):
        self.xe_w = xe_w^
        self.xe_b = xe_b^
        self.ce_w = ce_w^
        self.ce_b = ce_b^
        self.t_w0 = t_w0^
        self.t_b0 = t_b0^
        self.t_w2 = t_w2^
        self.t_b2 = t_b2^
        self.y_w0 = y_w0^
        self.y_b0 = y_b0^
        self.y_w2 = y_w2^
        self.y_b2 = y_b2^
        self.fl_ada_w = fl_ada_w^
        self.fl_ada_b = fl_ada_b^
        self.fl_lin_w = fl_lin_w^
        self.fl_lin_b = fl_lin_b^
        self.pos_embed = pos_embed^


# ── LoRA slot indices (8 adapters per joint block) ────────────────────────────
# Slots per block: [ctx_qkv=0, ctx_proj=1, ctx_fc1=2, ctx_fc2=3,
#                   x_qkv=4, x_proj=5, x_fc1=6, x_fc2=7]
comptime SLOTS_PER_BLOCK = 8
comptime SLOT_CTX_QKV  = 0
comptime SLOT_CTX_PROJ = 1
comptime SLOT_CTX_FC1  = 2
comptime SLOT_CTX_FC2  = 3
comptime SLOT_X_QKV    = 4
comptime SLOT_X_PROJ   = 5
comptime SLOT_X_FC1    = 6
comptime SLOT_X_FC2    = 7


def _block_base(bi: Int) -> Int:
    return bi * SLOTS_PER_BLOCK


# ── LoRA carrier ─────────────────────────────────────────────────────────────
struct SD35LoraSet(Copyable, Movable):
    var ad: List[LoraAdapter]
    var depth: Int
    var rank: Int

    def __init__(out self, var ad: List[LoraAdapter], depth: Int, rank: Int):
        self.ad = ad^
        self.depth = depth
        self.rank = rank


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _randn_lora(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * Float32(0.02))
    return out^


def _make_lora(rank: Int, in_f: Int, out_f: Int, alpha: Float32, seed: UInt64) -> LoraAdapter:
    var scale = alpha / Float32(rank)
    return LoraAdapter(
        _randn_lora(rank * in_f, seed),
        _zeros(out_f * rank),
        rank, in_f, out_f, scale,
        _zeros(rank * in_f), _zeros(rank * in_f),
        _zeros(out_f * rank), _zeros(out_f * rank),
    )


def build_sd35_lora_set(
    depth: Int, D: Int, MLP: Int, rank: Int, alpha: Float32
) -> SD35LoraSet:
    """Build the full LoRA set: 8 adapters per joint block (4 ctx + 4 x)."""
    var ad = List[LoraAdapter]()
    var seed = UInt64(9000)
    for _ in range(depth):
        ad.append(_make_lora(rank, D, 3 * D, alpha, seed)); seed += 1  # ctx qkv
        ad.append(_make_lora(rank, D, D, alpha, seed)); seed += 1      # ctx proj
        ad.append(_make_lora(rank, D, MLP, alpha, seed)); seed += 1    # ctx fc1
        ad.append(_make_lora(rank, MLP, D, alpha, seed)); seed += 1    # ctx fc2
        ad.append(_make_lora(rank, D, 3 * D, alpha, seed)); seed += 1  # x qkv
        ad.append(_make_lora(rank, D, D, alpha, seed)); seed += 1      # x proj
        ad.append(_make_lora(rank, D, MLP, alpha, seed)); seed += 1    # x fc1
        ad.append(_make_lora(rank, MLP, D, alpha, seed)); seed += 1    # x fc2
    return SD35LoraSet(ad^, depth, rank)


def total_adapters(lora: SD35LoraSet) -> Int:
    return len(lora.ad)


# ── LoRA gradient set ─────────────────────────────────────────────────────────
struct SD35LoraGradSet(Movable):
    var d_a: List[List[Float32]]
    var d_b: List[List[Float32]]
    var nonfinite: Int

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]], nonfinite: Int):
        self.d_a = d_a^
        self.d_b = d_b^
        self.nonfinite = nonfinite


def _nonfinite_count(v: List[Float32]) -> Int:
    var n = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            n += 1
    return n


# ── AdamW step for all LoRA adapters ─────────────────────────────────────────
def sd35_lora_adamw_step(
    mut lora: SD35LoraSet, grads: SD35LoraGradSet,
    step: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = 0.9, beta2: Float32 = 0.999, eps: Float32 = 1.0e-8,
    weight_decay: Float32 = 0.01,
) raises:
    for i in range(len(lora.ad)):
        var g = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(lora.ad[i], g, step, lr, ctx, beta1, beta2, eps, weight_decay)


# ── Save LoRA (PEFT-compatible safetensors) ───────────────────────────────────
def _sd35_named_loras(lora: SD35LoraSet) -> List[NamedLora]:
    var named = List[NamedLora]()
    for bi in range(lora.depth):
        var base = _block_base(bi)
        var bp = String("transformer.joint_blocks.") + String(bi)
        named.append(NamedLora(bp + String(".context_block.attn.qkv"),
            lora.ad[base + SLOT_CTX_QKV].copy()))
        named.append(NamedLora(bp + String(".context_block.attn.proj"),
            lora.ad[base + SLOT_CTX_PROJ].copy()))
        named.append(NamedLora(bp + String(".context_block.mlp.fc1"),
            lora.ad[base + SLOT_CTX_FC1].copy()))
        named.append(NamedLora(bp + String(".context_block.mlp.fc2"),
            lora.ad[base + SLOT_CTX_FC2].copy()))
        named.append(NamedLora(bp + String(".x_block.attn.qkv"),
            lora.ad[base + SLOT_X_QKV].copy()))
        named.append(NamedLora(bp + String(".x_block.attn.proj"),
            lora.ad[base + SLOT_X_PROJ].copy()))
        named.append(NamedLora(bp + String(".x_block.mlp.fc1"),
            lora.ad[base + SLOT_X_FC1].copy()))
        named.append(NamedLora(bp + String(".x_block.mlp.fc2"),
            lora.ad[base + SLOT_X_FC2].copy()))
    return named^


def save_sd35_lora(lora: SD35LoraSet, path: String, ctx: DeviceContext) raises -> Int:
    return save_lora_peft(_sd35_named_loras(lora), path, ctx)


def save_sd35_lora_state(lora: SD35LoraSet, path: String, ctx: DeviceContext) raises -> Int:
    return save_lora_train_state(_sd35_named_loras(lora), path, ctx)


# ── Direct DoRA/OFT metadata and optimizer/save helpers ─────────────────────
comptime SD35_DIRECT_24_GIB = 24 * 1024 * 1024 * 1024


struct SD35DirectDoRAGradSet(Movable):
    var grads: FlatDirectDoRAGrads
    var nonfinite_grads: Int

    def __init__(out self, var grads: FlatDirectDoRAGrads, nonfinite_grads: Int):
        self.grads = grads^
        self.nonfinite_grads = nonfinite_grads


struct SD35DirectOFTGradSet(Movable):
    var grads: FlatDirectOFTGrads
    var nonfinite_grads: Int

    def __init__(out self, var grads: FlatDirectOFTGrads, nonfinite_grads: Int):
        self.grads = grads^
        self.nonfinite_grads = nonfinite_grads


def empty_sd35_direct_dora_set() -> FlatDirectDoRASet:
    return empty_flat_direct_dora_set()


def empty_sd35_direct_oft_set() -> FlatDirectOFTSet:
    return empty_flat_direct_oft_set()


def _sd35_direct_slot_is_attn(slot: Int) -> Bool:
    var s = slot % SLOTS_PER_BLOCK
    return (
        s == SLOT_CTX_QKV or s == SLOT_CTX_PROJ
        or s == SLOT_X_QKV or s == SLOT_X_PROJ
    )


def _sd35_direct_slot_targeted(slot: Int, targets: Int) raises -> Bool:
    if targets < 1 or targets > 3:
        raise Error("SD35 direct LyCORIS: targets must be 1(attn)|2(all)|3(all)")
    if _sd35_direct_slot_is_attn(slot):
        return targets >= 1
    return targets >= 2


def _sd35_direct_slot_exists_in_block(bi: Int, depth: Int, slot: Int) -> Bool:
    # SD3.5 Large's final joint block is context_pre_only: context qkv exists,
    # but context proj/fc1/fc2 do not exist in the checkpoint or forward graph.
    if bi != depth - 1:
        return True
    var s = slot % SLOTS_PER_BLOCK
    return (
        s != SLOT_CTX_PROJ
        and s != SLOT_CTX_FC1
        and s != SLOT_CTX_FC2
    )


def _sd35_direct_slot_targeted_in_block(
    bi: Int, depth: Int, slot: Int, targets: Int,
) raises -> Bool:
    if not _sd35_direct_slot_exists_in_block(bi, depth, slot):
        return False
    return _sd35_direct_slot_targeted(slot, targets)


def _sd35_direct_slot_dims(slot: Int, D: Int, MLP: Int) raises -> Tuple[Int, Int]:
    var s = slot % SLOTS_PER_BLOCK
    if s == SLOT_CTX_QKV or s == SLOT_X_QKV:
        return (D, 3 * D)
    if s == SLOT_CTX_FC1 or s == SLOT_X_FC1:
        return (D, MLP)
    if s == SLOT_CTX_FC2 or s == SLOT_X_FC2:
        return (MLP, D)
    return (D, D)


def _sd35_direct_prefix(bi: Int, slot: Int) -> String:
    var bp = String("transformer.joint_blocks.") + String(bi)
    if slot == SLOT_CTX_QKV:
        return bp + String(".context_block.attn.qkv")
    if slot == SLOT_CTX_PROJ:
        return bp + String(".context_block.attn.proj")
    if slot == SLOT_CTX_FC1:
        return bp + String(".context_block.mlp.fc1")
    if slot == SLOT_CTX_FC2:
        return bp + String(".context_block.mlp.fc2")
    if slot == SLOT_X_QKV:
        return bp + String(".x_block.attn.qkv")
    if slot == SLOT_X_PROJ:
        return bp + String(".x_block.attn.proj")
    if slot == SLOT_X_FC1:
        return bp + String(".x_block.mlp.fc1")
    return bp + String(".x_block.mlp.fc2")


def _sd35_direct_slots_per_block(bi: Int, depth: Int, targets: Int) raises -> Int:
    var n = 0
    for slot in range(SLOTS_PER_BLOCK):
        if _sd35_direct_slot_targeted_in_block(bi, depth, slot, targets):
            n += 1
    return n


def _sd35_direct_expected_slots(depth: Int, targets: Int) raises -> Int:
    var n = 0
    for bi in range(depth):
        n += _sd35_direct_slots_per_block(bi, depth, targets)
    return n


def _sd35_direct_block_base(bi: Int, targets: Int) raises -> Int:
    raise Error("_sd35_direct_block_base: use depth-aware overload")


def _sd35_direct_block_base(bi: Int, depth: Int, targets: Int) raises -> Int:
    var n = 0
    for bj in range(bi):
        n += _sd35_direct_slots_per_block(bj, depth, targets)
    return n


def sd35_direct_dense_carrier_bytes(
    depth: Int, D: Int, MLP: Int, targets: Int,
) raises -> Int:
    var elems = 0
    for bi in range(depth):
        for slot in range(SLOTS_PER_BLOCK):
            if not _sd35_direct_slot_targeted_in_block(bi, depth, slot, targets):
                continue
            var dims = _sd35_direct_slot_dims(slot, D, MLP)
            elems += dims[0] * dims[0] + dims[1] * dims[0]
    return elems * 2


def sd35_direct_dora_trainable_bytes_estimate(
    depth: Int, D: Int, MLP: Int, rank: Int, targets: Int,
    wd_on_out: Bool = False,
) raises -> Int:
    var total = 0
    for bi in range(depth):
        for slot in range(SLOTS_PER_BLOCK):
            if not _sd35_direct_slot_targeted_in_block(bi, depth, slot, targets):
                continue
            var dims = _sd35_direct_slot_dims(slot, D, MLP)
            var in_f = dims[0]
            var out_f = dims[1]
            var mlen = out_f if wd_on_out else in_f
            var bf16_elems = rank * in_f + out_f * rank
            var f32_elems = mlen + (2 * rank * in_f) + (2 * out_f * rank) + (2 * mlen)
            total += bf16_elems * 2 + f32_elems * 4
    return total


def sd35_direct_oft_trainable_bytes_estimate(
    depth: Int, D: Int, MLP: Int, block_size: Int, targets: Int,
) raises -> Int:
    var total = 0
    for bi in range(depth):
        for slot in range(SLOTS_PER_BLOCK):
            if not _sd35_direct_slot_targeted_in_block(bi, depth, slot, targets):
                continue
            var dims = _sd35_direct_slot_dims(slot, D, MLP)
            var in_f = dims[0]
            if in_f % block_size != 0:
                raise Error("sd35_direct_oft_trainable_bytes_estimate: in_f not divisible by block_size")
            var r = in_f // block_size
            var ne = block_size * (block_size - 1) // 2
            total += 3 * r * ne * 4
    return total


def sd35_direct_dora_preflight(
    depth: Int, D: Int, MLP: Int, rank: Int, targets: Int,
    budget_bytes: Int, wd_on_out: Bool = False,
) raises -> Int:
    var direct = sd35_direct_dora_trainable_bytes_estimate(
        depth, D, MLP, rank, targets, wd_on_out,
    )
    if direct > budget_bytes:
        raise Error(
            String("SD3.5 direct DoRA trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


def sd35_direct_oft_preflight(
    depth: Int, D: Int, MLP: Int, block_size: Int, targets: Int,
    budget_bytes: Int,
) raises -> Int:
    var direct = sd35_direct_oft_trainable_bytes_estimate(
        depth, D, MLP, block_size, targets,
    )
    if direct > budget_bytes:
        raise Error(
            String("SD3.5 direct OFT trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


def _sd35_direct_block_for_dora(
    dora: FlatDirectDoRASet, bi: Int, depth: Int, targets: Int,
) raises -> SD35BlockDirectLycoris:
    return SD35BlockDirectLycoris(
        SD35_DIRECT_ALGO_DORA, dora.copy(), empty_flat_direct_oft_set(),
        _sd35_direct_block_base(bi, depth, targets), targets, bi == depth - 1,
    )


def _sd35_direct_block_for_oft(
    oft: FlatDirectOFTSet, bi: Int, depth: Int, targets: Int,
) raises -> SD35BlockDirectLycoris:
    return SD35BlockDirectLycoris(
        SD35_DIRECT_ALGO_OFT, empty_flat_direct_dora_set(), oft.copy(),
        _sd35_direct_block_base(bi, depth, targets), targets, bi == depth - 1,
    )


def _append_sd35_direct_dora_stream(
    mut set: FlatDirectDoRASet, w: StreamWeights, bi: Int, slot_base: Int,
    depth: Int, D: Int, MLP: Int, rank: Int, alpha: Float32, targets: Int,
    seed: UInt64, wd_on_out: Bool,
) raises:
    if _sd35_direct_slot_targeted_in_block(bi, depth, slot_base + SLOT_CTX_QKV, targets):
        flat_direct_dora_append_from_weight(
            set, w.wqkv.copy(), D, 3 * D, rank, alpha,
            _sd35_direct_prefix(bi, slot_base + SLOT_CTX_QKV), seed + UInt64(slot_base + SLOT_CTX_QKV), wd_on_out,
        )
    if _sd35_direct_slot_targeted_in_block(bi, depth, slot_base + SLOT_CTX_PROJ, targets):
        flat_direct_dora_append_from_weight(
            set, w.wproj.copy(), D, D, rank, alpha,
            _sd35_direct_prefix(bi, slot_base + SLOT_CTX_PROJ), seed + UInt64(slot_base + SLOT_CTX_PROJ), wd_on_out,
        )
    if _sd35_direct_slot_targeted_in_block(bi, depth, slot_base + SLOT_CTX_FC1, targets):
        flat_direct_dora_append_from_weight(
            set, w.wfc1.copy(), D, MLP, rank, alpha,
            _sd35_direct_prefix(bi, slot_base + SLOT_CTX_FC1), seed + UInt64(slot_base + SLOT_CTX_FC1), wd_on_out,
        )
    if _sd35_direct_slot_targeted_in_block(bi, depth, slot_base + SLOT_CTX_FC2, targets):
        flat_direct_dora_append_from_weight(
            set, w.wfc2.copy(), MLP, D, rank, alpha,
            _sd35_direct_prefix(bi, slot_base + SLOT_CTX_FC2), seed + UInt64(slot_base + SLOT_CTX_FC2), wd_on_out,
        )


def build_sd35_direct_dora_set_from_offload(
    mut loader: TurboPlannedLoader,
    depth: Int, D: Int, MLP: Int, Dh: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool, ctx: DeviceContext,
) raises -> FlatDirectDoRASet:
    if loader.block_count() < depth:
        raise Error("build_sd35_direct_dora_set_from_offload: loader depth too small")
    var set = empty_flat_direct_dora_set()
    if depth > 0:
        loader.prefetch_with_ctx(0, ctx)
    for bi in range(depth):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var pfx = handle.prefix + String(".")
        var block_seed = seed + UInt64(bi * SLOTS_PER_BLOCK)
        if bi == depth - 1:
            var cqw = _ctx_preonly_qkv_from_block(handle.block, pfx + String("context_block."), ctx)
            if _sd35_direct_slot_targeted_in_block(bi, depth, SLOT_CTX_QKV, targets):
                flat_direct_dora_append_from_weight(
                    set, cqw[0].copy(), D, 3 * D, rank, alpha,
                    _sd35_direct_prefix(bi, SLOT_CTX_QKV),
                    block_seed + UInt64(SLOT_CTX_QKV), wd_on_out,
                )
            var xw = _stream_weights_from_block(handle.block, pfx + String("x_block."), D, MLP, Dh, ctx)
            _append_sd35_direct_dora_stream(
                set, xw, bi, SLOT_X_QKV, depth, D, MLP, rank, alpha, targets, block_seed, wd_on_out,
            )
        else:
            var bwr = _joint_weights_from_block(handle.block, pfx, D, MLP, Dh, ctx)
            _append_sd35_direct_dora_stream(
                set, bwr.w.ctxw, bi, 0, depth, D, MLP, rank, alpha, targets, block_seed, wd_on_out,
            )
            _append_sd35_direct_dora_stream(
                set, bwr.w.xw, bi, SLOT_X_QKV, depth, D, MLP, rank, alpha, targets, block_seed, wd_on_out,
            )
        loader.mark_active_block_done(ctx)
    if len(set.ad) != _sd35_direct_expected_slots(depth, targets):
        raise Error("build_sd35_direct_dora_set_from_offload: direct slot count mismatch")
    return set^


def build_sd35_direct_oft_set_for_stack(
    depth: Int, D: Int, MLP: Int, block_size: Int, targets: Int,
) raises -> FlatDirectOFTSet:
    var set = empty_flat_direct_oft_set()
    for bi in range(depth):
        for slot in range(SLOTS_PER_BLOCK):
            if not _sd35_direct_slot_targeted_in_block(bi, depth, slot, targets):
                continue
            var dims = _sd35_direct_slot_dims(slot, D, MLP)
            flat_direct_oft_append(
                set, dims[0], dims[1], block_size,
                _sd35_direct_prefix(bi, slot),
            )
    if len(set.ad) != _sd35_direct_expected_slots(depth, targets):
        raise Error("build_sd35_direct_oft_set_for_stack: direct slot count mismatch")
    return set^


def _sd35_direct_dora_zero_grads(set: FlatDirectDoRASet) -> FlatDirectDoRAGrads:
    var out = List[DoRAGrads]()
    for i in range(len(set.ad)):
        ref d = set.ad[i]
        out.append(DoRAGrads(
            _zeros(len(d.a)), _zeros(len(d.b)), _zeros(len(d.m)), List[Float32](),
        ))
    return FlatDirectDoRAGrads(out^)


def _sd35_direct_oft_zero_grads(set: FlatDirectOFTSet) -> FlatDirectOFTGrads:
    var out = List[List[Float32]]()
    for i in range(len(set.ad)):
        out.append(_zeros(len(set.ad[i].vec)))
    return FlatDirectOFTGrads(out^)


def _sd35_scatter_dora(
    mut grads: FlatDirectDoRAGrads, slot: Int, g: SD35DirectProjectionGrad,
) raises:
    if slot < 0:
        return
    if slot >= len(grads.g):
        raise Error("_sd35_scatter_dora: slot out of range")
    grads.g[slot] = DoRAGrads(g.d_a.copy(), g.d_b.copy(), g.d_m.copy(), List[Float32]())


def _sd35_scatter_oft(
    mut grads: FlatDirectOFTGrads, slot: Int, g: SD35DirectProjectionGrad,
) raises:
    if slot < 0:
        return
    if slot >= len(grads.d_vec):
        raise Error("_sd35_scatter_oft: slot out of range")
    grads.d_vec[slot] = g.d_vec.copy()


def _sd35_nonfinite_direct(g: SD35DirectProjectionGrad) -> Int:
    return (
        _nonfinite_count(g.d_a) + _nonfinite_count(g.d_b)
        + _nonfinite_count(g.d_m) + _nonfinite_count(g.d_vec)
    )


def _sd35_nonfinite_block(g: SD35JointBlockDirectGrads) -> Int:
    return (
        _sd35_nonfinite_direct(g.ctx_g.qkv)
        + _sd35_nonfinite_direct(g.ctx_g.proj)
        + _sd35_nonfinite_direct(g.ctx_g.fc1)
        + _sd35_nonfinite_direct(g.ctx_g.fc2)
        + _sd35_nonfinite_direct(g.x_g.qkv)
        + _sd35_nonfinite_direct(g.x_g.proj)
        + _sd35_nonfinite_direct(g.x_g.fc1)
        + _sd35_nonfinite_direct(g.x_g.fc2)
    )


def sd35_direct_dora_grad_norm(g: FlatDirectDoRAGrads) -> Float64:
    return flat_direct_dora_grad_norm(g)


def sd35_direct_dora_clip_grads(mut g: FlatDirectDoRAGrads, clip_scale: Float32):
    flat_direct_dora_clip_grads(g, clip_scale)


def sd35_direct_dora_adamw_step(
    mut set: FlatDirectDoRASet, g: FlatDirectDoRAGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_dora_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def sd35_direct_dora_zero_leg_l1(set: FlatDirectDoRASet) -> Float64:
    return flat_direct_dora_zero_leg_l1(set)


def sd35_direct_dora_trainable_bytes(set: FlatDirectDoRASet) -> Int:
    return flat_direct_dora_trainable_bytes(set)


def sd35_direct_oft_grad_norm(g: FlatDirectOFTGrads) -> Float64:
    return flat_direct_oft_grad_norm(g)


def sd35_direct_oft_clip_grads(mut g: FlatDirectOFTGrads, clip_scale: Float32):
    flat_direct_oft_clip_grads(g, clip_scale)


def sd35_direct_oft_adamw_step(
    mut set: FlatDirectOFTSet, g: FlatDirectOFTGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_oft_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def sd35_direct_oft_vec_l1(set: FlatDirectOFTSet) -> Float64:
    return flat_direct_oft_vec_l1(set)


def sd35_direct_oft_trainable_bytes(set: FlatDirectOFTSet) -> Int:
    return flat_direct_oft_trainable_bytes(set)


def _sd35_f32_2d(var values: List[Float32], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


def save_sd35_direct_dora(
    set: FlatDirectDoRASet, path: String, ctx: DeviceContext,
) raises -> Int:
    var named = List[NamedDoRA]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedDoRA(set.prefix[i].copy(), set.ad[i].copy()))
    return save_dora_onetrainer(named, path, ctx)


def save_sd35_direct_oft(
    set: FlatDirectOFTSet, path: String, ctx: DeviceContext,
) raises -> Int:
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    var nmods = 0
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref sl = set.ad[i]
        var ne = sl.b * (sl.b - 1) // 2
        names.append(set.prefix[i].copy() + String(".oft_R.weight"))
        tensors.append(ArcPointer(_sd35_f32_2d(sl.vec.copy(), sl.r, ne, ctx)))
        nmods += 1
    if nmods == 0:
        raise Error("save_sd35_direct_oft: refusing to write an empty OFT file")
    save_safetensors(names, tensors, path, ctx)
    return nmods


# ── Host utility helpers (mirrors chroma_stack_lora / flux_stack) ─────────────
def _t(v: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(v, shape^, STDtype.F32, ctx)


def _t_as(v: List[Float32], var shape: List[Int], dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(v, shape^, dtype, ctx)


def _tb(v: List[BFloat16], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host_bf16(v, shape^, ctx)


def _zeros_list(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _ones_list(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(1.0))
    return o^


# ── Conditioning: t_embed(sigma) + y_embed(pooled_clip) -> [1,D] ─────────────
def _build_conditioning(
    base: SD35StackBase, sigma: Float32, pooled_h: List[Float32],
    D: Int, TIMESTEP_DIM: Int, POOLED_DIM: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    # t_embedder: sinusoidal(sigma*1000) -> mlp0 -> silu -> mlp2  -> [1, D]
    var sigma_scaled = sigma * Float32(1000.0)
    var t_in_vals = List[Float32]()
    t_in_vals.append(sigma_scaled)
    var t_in = _t(t_in_vals^, [1], ctx)
    var t_out = t_embedder(
        t_in,
        TIMESTEP_DIM,
        base.t_w0[],
        Optional[Tensor](base.t_b0[].clone(ctx)),
        base.t_w2[],
        Optional[Tensor](base.t_b2[].clone(ctx)),
        ctx,
        Float32(10000.0),
    )  # [1, D]

    # y_embedder: linear -> silu -> linear
    var y_dtype = base.y_w0[].dtype()
    var y0 = linear(
        _t_as(pooled_h.copy(), [1, POOLED_DIM], y_dtype, ctx),
        base.y_w0[],
        Optional[Tensor](base.y_b0[].clone(ctx)),
        ctx,
    )
    y0 = silu(y0, ctx)
    var y_out = linear(
        y0,
        base.y_w2[],
        Optional[Tensor](base.y_b2[].clone(ctx)),
        ctx,
    )  # [1, D]

    # c = t_embed + y_embed (elementwise)
    var t_h = t_out.to_host(ctx)
    var y_h = y_out.to_host(ctx)
    var c = List[Float32]()
    for i in range(D):
        c.append(t_h[i] + y_h[i])
    return c^


# ── adaLN modulation: silu(c) @ W.T + b -> [6D] -> chunk 6 x [D] ModVecs ─────
def _block_modvecs(
    c: List[Float32], ada_w: List[Float32], ada_b: List[Float32],
    D: Int, ctx: DeviceContext,
) raises -> ModVecs:
    var c_silu = silu(_t(c.copy(), [1, D], ctx), ctx)
    var raw = linear(
        c_silu,
        _t(ada_w.copy(), [6 * D, D], ctx),
        Optional[Tensor](_t(ada_b.copy(), [6 * D], ctx)),
        ctx,
    ).to_host(ctx)  # [6D]
    return _chunk6(raw^, D)


def _chunk6(v: List[Float32], D: Int) -> ModVecs:
    """Split flat [6D] into ModVecs (shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp)."""
    var shift_msa = List[Float32]()
    var scale_msa = List[Float32]()
    var gate_msa = List[Float32]()
    var shift_mlp = List[Float32]()
    var scale_mlp = List[Float32]()
    var gate_mlp = List[Float32]()
    for c in range(D):
        shift_msa.append(v[0 * D + c])
        scale_msa.append(v[1 * D + c])
        gate_msa.append(v[2 * D + c])
        shift_mlp.append(v[3 * D + c])
        scale_mlp.append(v[4 * D + c])
        gate_mlp.append(v[5 * D + c])
    return ModVecs(shift_msa^, scale_msa^, gate_msa^, shift_mlp^, scale_mlp^, gate_mlp^)


struct _StreamModResult(Movable):
    var flat: List[Float32]   # [6D]
    var mv: ModVecs

    def __init__(out self, var flat: List[Float32], var mv: ModVecs):
        self.flat = flat^
        self.mv = mv^


# ── DUAL adaLN: silu(c) @ W.T + b -> [9D] -> ModVecs(6) + msa2 triple(3) ──────
# chunk-9 order VERIFIED vs diffusers single_file_utils.py (adaLN_modulation.1 ->
# norm1.linear is an identity map) + SD35AdaLayerNormZeroX.forward:
#   [shift_msa,scale_msa,gate_msa, shift_mlp,scale_mlp,gate_mlp, shift_msa2,scale_msa2,gate_msa2]
struct _DualModResult(Movable):
    var flat: List[Float32]        # [9D] (saved for backward)
    var mv: ModVecs                # first 6 chunks
    var shift_msa2: List[Float32]
    var scale_msa2: List[Float32]
    var gate_msa2: List[Float32]

    def __init__(
        out self, var flat: List[Float32], var mv: ModVecs,
        var shift_msa2: List[Float32], var scale_msa2: List[Float32], var gate_msa2: List[Float32],
    ):
        self.flat = flat^
        self.mv = mv^
        self.shift_msa2 = shift_msa2^
        self.scale_msa2 = scale_msa2^
        self.gate_msa2 = gate_msa2^


def _chunk9_msa2(v: List[Float32], D: Int) -> List[List[Float32]]:
    """From flat [9D] extract the msa2 triple (chunks 6,7,8)."""
    var s2 = List[Float32](); var sc2 = List[Float32](); var g2 = List[Float32]()
    for c in range(D):
        s2.append(v[6 * D + c])
        sc2.append(v[7 * D + c])
        g2.append(v[8 * D + c])
    var out = List[List[Float32]]()
    out.append(s2^); out.append(sc2^); out.append(g2^)
    return out^


def _compute_dual_modvecs(
    c: List[Float32], ada_w: List[Float32], ada_b: List[Float32],
    D: Int, ctx: DeviceContext,
) raises -> _DualModResult:
    var sh1 = List[Int](); sh1.append(1); sh1.append(D)
    var c_silu = silu(_t(c.copy(), sh1^, ctx), ctx)
    var sh2 = List[Int](); sh2.append(9 * D); sh2.append(D)
    var sh3 = List[Int](); sh3.append(9 * D)
    var raw = linear(
        c_silu, _t(ada_w.copy(), sh2^, ctx),
        Optional[Tensor](_t(ada_b.copy(), sh3^, ctx)), ctx,
    ).to_host(ctx)
    var mv = _chunk6(raw.copy(), D)        # first 6 chunks
    var m2 = _chunk9_msa2(raw.copy(), D)   # chunks 6,7,8
    return _DualModResult(raw^, mv^, m2[0].copy(), m2[1].copy(), m2[2].copy())


def _attn2_weights_from_block(
    block: Block, x_prefix: String, ctx: DeviceContext
) raises -> Attn2Weights:
    var bp = x_prefix + String("attn2.")
    return Attn2Weights(
        _block_host_f32(block, bp + String("qkv.weight"), ctx),
        _block_host_f32(block, bp + String("qkv.bias"), ctx),
        _block_host_f32(block, bp + String("proj.weight"), ctx),
        _block_host_f32(block, bp + String("proj.bias"), ctx),
        _block_host_f32(block, bp + String("ln_q.weight"), ctx),
        _block_host_f32(block, bp + String("ln_k.weight"), ctx),
    )


def _empty_stream_saved() -> StreamSaved:
    var e = List[Float32]()
    return StreamSaved(
        e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy(),
        e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy(),
    )


def _empty_joint_fwd() -> JointBlockForward:
    var e = List[Float32]()
    return JointBlockForward(e.copy(), e.copy(), _empty_stream_saved(), _empty_stream_saved())


def _empty_dual_xsaved() -> DualXSaved:
    var e = List[Float32]()
    return DualXSaved(
        e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy(),
        e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy(),
        e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy(),
    )


def _empty_dual_fwd() -> DualBlockForward:
    var e = List[Float32]()
    return DualBlockForward(e.copy(), e.copy(), _empty_stream_saved(), _empty_dual_xsaved())


def _empty_ctxpre_fwd() -> CtxPreForward:
    var e = List[Float32]()
    return CtxPreForward(e.copy(), CtxPreSaved(e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), _empty_stream_saved()))


# ── ctx-pre-only (final) block: context has ONLY attn.qkv + ln_q/ln_k ──────────
def _ctx_preonly_qkv_from_block(
    block: Block, ctx_prefix: String, ctx: DeviceContext
) raises -> List[List[Float32]]:
    var bp = ctx_prefix + String("attn.")
    var out = List[List[Float32]]()
    out.append(_block_host_f32(block, bp + String("qkv.weight"), ctx))
    out.append(_block_host_f32(block, bp + String("qkv.bias"), ctx))
    out.append(_block_host_f32(block, bp + String("ln_q.weight"), ctx))
    out.append(_block_host_f32(block, bp + String("ln_k.weight"), ctx))
    return out^


# AdaLayerNormContinuous: silu(c)@W.T+b -> [2D] raw order [shift,scale].
# Returns [flat_2D, scale, shift].
def _compute_ctx_continuous_mod(
    c: List[Float32], ada_w: List[Float32], ada_b: List[Float32],
    D: Int, ctx: DeviceContext,
) raises -> List[List[Float32]]:
    var sh1 = List[Int](); sh1.append(1); sh1.append(D)
    var c_silu = silu(_t(c.copy(), sh1^, ctx), ctx)
    var sh2 = List[Int](); sh2.append(2 * D); sh2.append(D)
    var sh3 = List[Int](); sh3.append(2 * D)
    var raw = linear(
        c_silu, _t(ada_w.copy(), sh2^, ctx),
        Optional[Tensor](_t(ada_b.copy(), sh3^, ctx)), ctx,
    ).to_host(ctx)
    var shift = List[Float32](); var scale = List[Float32]()
    for cc in range(D):
        shift.append(raw[cc])          # chunk 0 = shift
        scale.append(raw[D + cc])      # chunk 1 = scale
    var out = List[List[Float32]]()
    out.append(raw.copy()); out.append(scale^); out.append(shift^)
    return out^


def _compute_stream_modvecs(
    c: List[Float32], ada_w: List[Float32], ada_b: List[Float32],
    D: Int, ctx: DeviceContext,
) raises -> _StreamModResult:
    """Returns _StreamModResult(flat_6D, ModVecs) for one stream's adaLN."""
    var sh1 = List[Int](); sh1.append(1); sh1.append(D)
    var c_silu = silu(_t(c.copy(), sh1^, ctx), ctx)
    var sh2 = List[Int](); sh2.append(6 * D); sh2.append(D)
    var sh3 = List[Int](); sh3.append(6 * D)
    var raw = linear(
        c_silu,
        _t(ada_w.copy(), sh2^, ctx),
        Optional[Tensor](_t(ada_b.copy(), sh3^, ctx)),
        ctx,
    ).to_host(ctx)
    var mv = _chunk6(raw.copy(), D)
    return _StreamModResult(raw^, mv^)


# ── Saved per-block state (for backward) ─────────────────────────────────────
struct BlockSaved(Copyable, Movable):
    var is_dual: Bool
    var is_ctxpre: Bool                # context_pre_only FINAL block
    var fwd: JointBlockForward         # valid when standard (not dual/ctxpre)
    var dual_fwd: DualBlockForward     # valid when is_dual
    var ctxpre_fwd: CtxPreForward      # valid when is_ctxpre
    var ctx_ada_flat: List[Float32]    # [6D] std / [2D] ctxpre — ctx stream
    var x_ada_flat: List[Float32]      # [6D] std / [9D] dual — x stream

    def __init__(
        out self,
        is_dual: Bool, is_ctxpre: Bool,
        var fwd: JointBlockForward,
        var dual_fwd: DualBlockForward,
        var ctxpre_fwd: CtxPreForward,
        var ctx_ada_flat: List[Float32],
        var x_ada_flat: List[Float32],
    ):
        self.is_dual = is_dual
        self.is_ctxpre = is_ctxpre
        self.fwd = fwd^
        self.dual_fwd = dual_fwd^
        self.ctxpre_fwd = ctxpre_fwd^
        self.ctx_ada_flat = ctx_ada_flat^
        self.x_ada_flat = x_ada_flat^


# ── Full forward saved tape ────────────────────────────────────────────────────
struct SD35StackForward(Movable):
    var out: List[Float32]             # [N_IMG, 64] final output
    var blocks: List[BlockSaved]       # per-block saved activations
    var c: List[Float32]               # conditioning [D]
    var x_proj: List[Float32]          # x after x_embedder [N_IMG, D]
    var ctx_proj: List[Float32]        # context after ctx_embedder [N_CTX, D]
    var final_ada_flat: List[Float32]  # [2D] final adaLN raw output
    var pre_final_x: List[Float32]     # [N_IMG, D] x before final layer

    def __init__(
        out self,
        var out: List[Float32],
        var blocks: List[BlockSaved],
        var c: List[Float32],
        var x_proj: List[Float32],
        var ctx_proj: List[Float32],
        var final_ada_flat: List[Float32],
        var pre_final_x: List[Float32],
    ):
        self.out = out^
        self.blocks = blocks^
        self.c = c^
        self.x_proj = x_proj^
        self.ctx_proj = ctx_proj^
        self.final_ada_flat = final_ada_flat^
        self.pre_final_x = pre_final_x^


# ── Block weight helpers (from streamed block) ─────────────────────────────────
def _block_host_f32(block: Block, key: String, ctx: DeviceContext) raises -> List[Float32]:
    if not (key in block):
        raise Error(String("SD35 block missing: ") + key)
    return cast_tensor(block[key][], STDtype.F32, ctx).to_host(ctx)


def _stream_weights_from_block(
    block: Block, prefix: String, D: Int, MLP: Int, Dh: Int,
    ctx: DeviceContext,
) raises -> StreamWeights:
    var bp = prefix
    return StreamWeights(
        _block_host_f32(block, bp + String("attn.qkv.weight"), ctx),
        _block_host_f32(block, bp + String("attn.qkv.bias"), ctx),
        _block_host_f32(block, bp + String("attn.proj.weight"), ctx),
        _block_host_f32(block, bp + String("attn.proj.bias"), ctx),
        _block_host_f32(block, bp + String("mlp.fc1.weight"), ctx),
        _block_host_f32(block, bp + String("mlp.fc1.bias"), ctx),
        _block_host_f32(block, bp + String("mlp.fc2.weight"), ctx),
        _block_host_f32(block, bp + String("mlp.fc2.bias"), ctx),
        _block_host_f32(block, bp + String("attn.ln_q.weight"), ctx),
        _block_host_f32(block, bp + String("attn.ln_k.weight"), ctx),
    )


struct _BlockWeightsResult(Movable):
    var w: JointBlockWeights
    var ctx_ada_w: List[Float32]
    var ctx_ada_b: List[Float32]
    var x_ada_w: List[Float32]
    var x_ada_b: List[Float32]

    def __init__(
        out self,
        var w: JointBlockWeights,
        var ctx_ada_w: List[Float32], var ctx_ada_b: List[Float32],
        var x_ada_w: List[Float32], var x_ada_b: List[Float32],
    ):
        self.w = w^
        self.ctx_ada_w = ctx_ada_w^
        self.ctx_ada_b = ctx_ada_b^
        self.x_ada_w = x_ada_w^
        self.x_ada_b = x_ada_b^


def _joint_weights_from_block(
    block: Block, block_prefix: String, D: Int, MLP: Int, Dh: Int,
    ctx: DeviceContext,
) raises -> _BlockWeightsResult:
    """Returns _BlockWeightsResult with (weights, ctx_ada_w, ctx_ada_b, x_ada_w, x_ada_b)."""
    var ctx_bp = block_prefix + String("context_block.")
    var x_bp = block_prefix + String("x_block.")
    var ctxw = _stream_weights_from_block(block, ctx_bp, D, MLP, Dh, ctx)
    var xw = _stream_weights_from_block(block, x_bp, D, MLP, Dh, ctx)
    var ctx_ada_w = _block_host_f32(block, ctx_bp + String("adaLN_modulation.1.weight"), ctx)
    var ctx_ada_b = _block_host_f32(block, ctx_bp + String("adaLN_modulation.1.bias"), ctx)
    var x_ada_w = _block_host_f32(block, x_bp + String("adaLN_modulation.1.weight"), ctx)
    var x_ada_b = _block_host_f32(block, x_bp + String("adaLN_modulation.1.bias"), ctx)
    return _BlockWeightsResult(
        JointBlockWeights(ctxw^, xw^),
        ctx_ada_w^, ctx_ada_b^, x_ada_w^, x_ada_b^,
    )


# ── patchify: [N_IMG, 64] x_embedder linear ──────────────────────────────────
def _x_embed(
    noisy_h: List[Float32], base: SD35StackBase,
    N_IMG: Int, IN_CH: Int, D: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    # x_embedder.proj is Conv2d(16,D,2,2); the checkpoint stores it rank-4
    # [D,16,2,2]. With stride=kernel it is a linear on patch vectors [N_IMG,64]:
    # flatten the weight to [D, IN_CH] (row-major (c,kh,kw) == the patch layout).
    var xe_dtype = base.xe_w[].dtype()
    var xe_w2 = reshape(base.xe_w[], [D, IN_CH], ctx)
    return linear(
        _t_as(noisy_h, [N_IMG, IN_CH], xe_dtype, ctx),
        xe_w2,
        Optional[Tensor](base.xe_b[].clone(ctx)),
        ctx,
    ).to_host(ctx)


def _isqrt(n: Int) -> Int:
    var r = 0
    while (r + 1) * (r + 1) <= n:
        r += 1
    return r


# Add the center-cropped learned pos_embed to the image tokens (SD3 PatchEmbed):
# reshape pos_embed [POS_MAX*POS_MAX, D], crop center [HT,WT] (top/left=(MAX-H)//2),
# flatten row-major (h,w) — matches _pack_latents' (ih,iw) token order — and add.
def _add_pos_embed(
    var x_proj: List[Float32], base: SD35StackBase, N_IMG: Int, D: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var pos = cast_tensor(base.pos_embed[], STDtype.F32, ctx).to_host(ctx)  # [MAX*MAX*D]
    var pos_max = _isqrt(len(pos) // D)
    var ht = _isqrt(N_IMG)
    var wt = N_IMG // ht
    var top = (pos_max - ht) // 2
    var left = (pos_max - wt) // 2
    for ih in range(ht):
        for iw in range(wt):
            var src = ((top + ih) * pos_max + (left + iw)) * D
            var dst = (ih * wt + iw) * D
            for d in range(D):
                x_proj[dst + d] = x_proj[dst + d] + pos[src + d]
    return x_proj^


def _ctx_embed(
    ctx_tokens_h: List[Float32], base: SD35StackBase,
    N_CTX: Int, CTX_CH: Int, D: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var ce_dtype = base.ce_w[].dtype()
    return linear(
        _t_as(ctx_tokens_h, [N_CTX, CTX_CH], ce_dtype, ctx),
        base.ce_w[],
        Optional[Tensor](base.ce_b[].clone(ctx)),
        ctx,
    ).to_host(ctx)


struct _FinalLayerResult(Movable):
    var out: List[Float32]
    var ada_flat: List[Float32]
    var pre_x: List[Float32]

    def __init__(out self, var out: List[Float32], var ada_flat: List[Float32], var pre_x: List[Float32]):
        self.out = out^
        self.ada_flat = ada_flat^
        self.pre_x = pre_x^


# ── final layer ───────────────────────────────────────────────────────────────
def _final_layer(
    x_h: List[Float32], c_h: List[Float32], base: SD35StackBase,
    N_IMG: Int, D: Int, OUT_CH: Int, eps: Float32, ctx: DeviceContext,
) raises -> _FinalLayerResult:
    """Returns _FinalLayerResult(out, final_ada_flat [2D], pre_final_x [N_IMG,D])."""
    var ada_dtype = base.fl_ada_w[].dtype()
    var lin_dtype = base.fl_lin_w[].dtype()
    # LayerNorm (no affine)
    var ln_x = layer_norm(
        _t_as(x_h.copy(), [N_IMG, D], ada_dtype, ctx),
        _t_as(_ones_list(D), [D], ada_dtype, ctx),
        _t_as(_zeros_list(D), [D], ada_dtype, ctx),
        eps, ctx,
    ).to_host(ctx)

    # silu(c) -> linear -> [2D] -> chunk shift, scale
    var c_silu = silu(_t_as(c_h.copy(), [1, D], ada_dtype, ctx), ctx)
    var ada_raw = linear(
        c_silu,
        base.fl_ada_w[],
        Optional[Tensor](base.fl_ada_b[].clone(ctx)),
        ctx,
    ).to_host(ctx)  # [2D]

    # shift = ada_raw[:D], scale = ada_raw[D:]
    var shift = List[Float32]()
    var scale = List[Float32]()
    for i in range(D):
        shift.append(ada_raw[i])
    for i in range(D):
        scale.append(ada_raw[D + i])

    # modulate: o = (1+scale)*x + shift
    var x_mod = List[Float32]()
    for r in range(N_IMG):
        for c_idx in range(D):
            x_mod.append((1.0 + scale[c_idx]) * ln_x[r * D + c_idx] + shift[c_idx])

    # final linear -> [N_IMG, OUT_CH]
    var out = linear(
        _t_as(x_mod, [N_IMG, D], lin_dtype, ctx),
        base.fl_lin_w[],
        Optional[Tensor](base.fl_lin_b[].clone(ctx)),
        ctx,
    ).to_host(ctx)

    return _FinalLayerResult(out^, ada_raw^, x_h.copy())


# ═══════════════════════════════════════════════════════════════════════════════
# FULL FORWARD WITH LoRA, BLOCK-SWAP OFFLOAD (38 joint blocks, SD3.5 Large).
#
# Key flow:
#   1. Compute conditioning c = t_embed + y_embed [D]
#   2. x = x_embedder(noisy) [N_IMG, D]; ctx = ctx_embedder(text) [N_CTX, D]
#   3. For each joint block bi (0..38):
#      a. Load block from offload; extract weights + adaLN params
#      b. Compute per-stream modvecs via adaLN(silu(c))
#      c. Compute LoRA deltas on qkv input for both streams
#      d. Run sd35_joint_block_forward; save activations
#      e. Prefetch next block; mark done
#   4. Final layer: LayerNorm -> adaLN(c) -> modulate -> linear -> [N_IMG, 64]
# ═══════════════════════════════════════════════════════════════════════════════
def sd35_stack_lora_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_CTX: Int, S: Int
](
    noisy_h: List[Float32],      # [N_IMG, IN_CH] patchified + packed
    text_h: List[Float32],       # [N_CTX, CTX_CH]
    pooled_h: List[Float32],     # [1, POOLED_DIM] -> [POOLED_DIM]
    sigma: Float32,
    base: SD35StackBase,
    mut loader: TurboPlannedLoader,
    lora: SD35LoraSet,
    D: Int, MLP: Int, IN_CH: Int, CTX_CH: Int, OUT_CH: Int,
    TIMESTEP_DIM: Int, POOLED_DIM: Int,
    eps: Float32, qk_eps: Float32,
    ctx: DeviceContext,
    num_dual: Int = 0,   # blocks [0, num_dual) are DUAL-attention (sd3.5-medium=13)
    last_ctx_preonly: Bool = False,   # final block is context_pre_only (SD3.5)
) raises -> SD35StackForward:
    var depth = lora.depth

    loader.prefetch_with_ctx(0, ctx)

    # ── 1. conditioning ──
    var c = _build_conditioning(base, sigma, pooled_h, D, TIMESTEP_DIM, POOLED_DIM, ctx)

    # ── 2. input projections (+ learned pos_embed on image tokens) ──
    var x_proj = _x_embed(noisy_h.copy(), base, N_IMG, IN_CH, D, ctx)
    x_proj = _add_pos_embed(x_proj^, base, N_IMG, D, ctx)
    var ctx_proj = _ctx_embed(text_h.copy(), base, N_CTX, CTX_CH, D, ctx)

    # ── 3. joint blocks ──
    var x = x_proj.copy()
    var context = ctx_proj.copy()
    var blocks = List[BlockSaved]()
    var scale = Float32(1.0) / Float32(8.0)   # 1/sqrt(64)

    for bi in range(depth):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)

        var pfx = handle.prefix + String(".")
        var base_lora = _block_base(bi)

        if last_ctx_preonly and bi == depth - 1:
            # ── CONTEXT_PRE_ONLY final block: ctx has qkv only (no proj/mlp) ──
            var xw = _stream_weights_from_block(handle.block, pfx + String("x_block."), D, MLP, Dh, ctx)
            var x_ada_w = _block_host_f32(handle.block, pfx + String("x_block.adaLN_modulation.1.weight"), ctx)
            var x_ada_b = _block_host_f32(handle.block, pfx + String("x_block.adaLN_modulation.1.bias"), ctx)
            var x_smr = _compute_stream_modvecs(c.copy(), x_ada_w^, x_ada_b^, D, ctx)
            var cqw = _ctx_preonly_qkv_from_block(handle.block, pfx + String("context_block."), ctx)
            var c_ada_w = _block_host_f32(handle.block, pfx + String("context_block.adaLN_modulation.1.weight"), ctx)
            var c_ada_b = _block_host_f32(handle.block, pfx + String("context_block.adaLN_modulation.1.bias"), ctx)
            var cmod = _compute_ctx_continuous_mod(c.copy(), c_ada_w^, c_ada_b^, D, ctx)  # [flat2D, scale, shift]
            var cpfwd = sd35_context_preonly_forward[1, S, H, Dh](
                context.copy(), x.copy(), cqw[0].copy(), cqw[1].copy(), cqw[2].copy(), cqw[3].copy(),
                cmod[1].copy(), cmod[2].copy(), xw, x_smr.mv.copy(),
                N_CTX, N_IMG, D, MLP, eps, qk_eps, scale, ctx,
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_CTX_QKV].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_QKV].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_PROJ].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_FC1].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_FC2].copy()),
            )
            x = cpfwd.x_out.copy()   # context unchanged (discarded downstream)
            blocks.append(BlockSaved(False, True, _empty_joint_fwd(), _empty_dual_fwd(), cpfwd^, cmod[0].copy(), x_smr.flat.copy()))
            loader.mark_active_block_done(ctx)
            continue

        var bwr = _joint_weights_from_block(handle.block, pfx, D, MLP, Dh, ctx)
        var w = bwr.w.copy()
        var ctx_ada_w = bwr.ctx_ada_w.copy()
        var ctx_ada_b = bwr.ctx_ada_b.copy()
        var x_ada_w = bwr.x_ada_w.copy()
        var x_ada_b = bwr.x_ada_b.copy()

        # context stream modvecs (always 6-chunk AdaLayerNormZero)
        var ctx_smr = _compute_stream_modvecs(c.copy(), ctx_ada_w.copy(), ctx_ada_b.copy(), D, ctx)
        var ctx_ada_flat = ctx_smr.flat.copy()
        var ctx_mv = ctx_smr.mv.copy()

        if bi < num_dual:
            # ── DUAL-attention block (SD35AdaLayerNormZeroX 9-chunk + attn2) ──
            var dmr = _compute_dual_modvecs(c.copy(), x_ada_w.copy(), x_ada_b.copy(), D, ctx)
            var x_ada_flat = dmr.flat.copy()   # [9D] saved
            var a2w = _attn2_weights_from_block(handle.block, pfx + String("x_block."), ctx)
            var dfwd = sd35_dual_joint_block_forward[1, S, N_IMG, H, Dh](
                context.copy(), x.copy(), w.ctxw, w.xw, a2w, ctx_mv.copy(), dmr.mv.copy(),
                dmr.shift_msa2.copy(), dmr.scale_msa2.copy(), dmr.gate_msa2.copy(),
                N_CTX, N_IMG, D, MLP, eps, qk_eps, scale, ctx,
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_QKV].copy()),
                None,   # attn2 LoRA not yet in the slot set (base attn2, faithful fwd)
            )
            context = dfwd.ctx_out.copy()
            x = dfwd.x_out.copy()
            blocks.append(BlockSaved(True, False, _empty_joint_fwd(), dfwd^, _empty_ctxpre_fwd(), ctx_ada_flat^, x_ada_flat^))
        else:
            # ── STANDARD joint block (6-chunk) ──
            var x_smr = _compute_stream_modvecs(c.copy(), x_ada_w.copy(), x_ada_b.copy(), D, ctx)
            var x_ada_flat = x_smr.flat.copy()
            var fwd = sd35_joint_block_forward[1, S, H, Dh](
                context.copy(), x.copy(), w, ctx_mv.copy(), x_smr.mv.copy(),
                N_CTX, N_IMG, D, MLP, eps, qk_eps, scale, ctx,
                Optional[List[Float32]](None),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_CTX_QKV].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_CTX_PROJ].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_CTX_FC1].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_CTX_FC2].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_QKV].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_PROJ].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_FC1].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_FC2].copy()),
            )
            context = fwd.ctx_out.copy()
            x = fwd.x_out.copy()
            blocks.append(BlockSaved(False, False, fwd^, _empty_dual_fwd(), _empty_ctxpre_fwd(), ctx_ada_flat^, x_ada_flat^))
        loader.mark_active_block_done(ctx)

    # ── 4. final layer ──
    var fl = _final_layer(x, c.copy(), base, N_IMG, D, OUT_CH, eps, ctx)
    var out = fl.out.copy()
    var final_ada_flat = fl.ada_flat.copy()
    var pre_final_x = fl.pre_x.copy()

    return SD35StackForward(out^, blocks^, c^, x_proj^, ctx_proj^, final_ada_flat^, pre_final_x^)


def _sd35_stack_direct_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_CTX: Int, S: Int
](
    noisy_h: List[Float32],
    text_h: List[Float32],
    pooled_h: List[Float32],
    sigma: Float32,
    base: SD35StackBase,
    mut loader: TurboPlannedLoader,
    dora: FlatDirectDoRASet, oft: FlatDirectOFTSet, algo: Int,
    depth: Int, targets: Int,
    D: Int, MLP: Int, IN_CH: Int, CTX_CH: Int, OUT_CH: Int,
    TIMESTEP_DIM: Int, POOLED_DIM: Int,
    eps: Float32, qk_eps: Float32,
    ctx: DeviceContext,
) raises -> SD35StackForward:
    loader.prefetch_with_ctx(0, ctx)

    var c = _build_conditioning(base, sigma, pooled_h, D, TIMESTEP_DIM, POOLED_DIM, ctx)
    var x_proj = _x_embed(noisy_h.copy(), base, N_IMG, IN_CH, D, ctx)
    x_proj = _add_pos_embed(x_proj^, base, N_IMG, D, ctx)
    var ctx_proj = _ctx_embed(text_h.copy(), base, N_CTX, CTX_CH, D, ctx)

    var x = x_proj.copy()
    var context = ctx_proj.copy()
    var blocks = List[BlockSaved]()
    var scale = Float32(1.0) / Float32(8.0)

    for bi in range(depth):
        var handle = loader.await_block(bi, ctx)
        loader.prefetch_next_with_ctx(bi, ctx)
        var pfx = handle.prefix + String(".")
        var direct = _sd35_direct_block_for_dora(dora, bi, depth, targets) if algo == SD35_DIRECT_ALGO_DORA else _sd35_direct_block_for_oft(oft, bi, depth, targets)
        if bi == depth - 1:
            var xw = _stream_weights_from_block(handle.block, pfx + String("x_block."), D, MLP, Dh, ctx)
            var x_ada_w = _block_host_f32(handle.block, pfx + String("x_block.adaLN_modulation.1.weight"), ctx)
            var x_ada_b = _block_host_f32(handle.block, pfx + String("x_block.adaLN_modulation.1.bias"), ctx)
            var x_smr = _compute_stream_modvecs(c.copy(), x_ada_w^, x_ada_b^, D, ctx)
            var cqw = _ctx_preonly_qkv_from_block(handle.block, pfx + String("context_block."), ctx)
            var c_ada_w = _block_host_f32(handle.block, pfx + String("context_block.adaLN_modulation.1.weight"), ctx)
            var c_ada_b = _block_host_f32(handle.block, pfx + String("context_block.adaLN_modulation.1.bias"), ctx)
            var cmod = _compute_ctx_continuous_mod(c.copy(), c_ada_w^, c_ada_b^, D, ctx)
            var cpfwd = sd35_context_preonly_direct_lycoris_forward[1, S, H, Dh](
                context.copy(), x.copy(), cqw[0].copy(), cqw[1].copy(), cqw[2].copy(), cqw[3].copy(),
                cmod[1].copy(), cmod[2].copy(), xw, x_smr.mv.copy(), direct,
                N_CTX, N_IMG, D, MLP, eps, qk_eps, scale, ctx,
            )
            x = cpfwd.x_out.copy()
            blocks.append(BlockSaved(False, True, _empty_joint_fwd(), _empty_dual_fwd(), cpfwd^, cmod[0].copy(), x_smr.flat.copy()))
            loader.mark_active_block_done(ctx)
            continue
        var bwr = _joint_weights_from_block(handle.block, pfx, D, MLP, Dh, ctx)
        var w = bwr.w.copy()
        var ctx_smr = _compute_stream_modvecs(c.copy(), bwr.ctx_ada_w.copy(), bwr.ctx_ada_b.copy(), D, ctx)
        var x_smr = _compute_stream_modvecs(c.copy(), bwr.x_ada_w.copy(), bwr.x_ada_b.copy(), D, ctx)
        var fwd = sd35_joint_block_direct_lycoris_forward[1, S, H, Dh](
            context.copy(), x.copy(), w, ctx_smr.mv.copy(), x_smr.mv.copy(),
            direct, N_CTX, N_IMG, D, MLP, eps, qk_eps, scale, ctx,
        )
        context = fwd.ctx_out.copy()
        x = fwd.x_out.copy()
        blocks.append(BlockSaved(False, False, fwd^, _empty_dual_fwd(), _empty_ctxpre_fwd(), ctx_smr.flat.copy(), x_smr.flat.copy()))
        loader.mark_active_block_done(ctx)

    var fl = _final_layer(x, c.copy(), base, N_IMG, D, OUT_CH, eps, ctx)
    var out = fl.out.copy()
    var final_ada_flat = fl.ada_flat.copy()
    var pre_final_x = fl.pre_x.copy()

    return SD35StackForward(out^, blocks^, c^, x_proj^, ctx_proj^, final_ada_flat^, pre_final_x^)


def sd35_stack_direct_dora_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_CTX: Int, S: Int
](
    noisy_h: List[Float32],
    text_h: List[Float32],
    pooled_h: List[Float32],
    sigma: Float32,
    base: SD35StackBase,
    mut loader: TurboPlannedLoader,
    dora: FlatDirectDoRASet,
    depth: Int, targets: Int,
    D: Int, MLP: Int, IN_CH: Int, CTX_CH: Int, OUT_CH: Int,
    TIMESTEP_DIM: Int, POOLED_DIM: Int,
    eps: Float32, qk_eps: Float32,
    ctx: DeviceContext,
) raises -> SD35StackForward:
    if len(dora.ad) != _sd35_direct_expected_slots(depth, targets):
        raise Error("sd35_stack_direct_dora_forward_offload: direct slot count mismatch")
    return _sd35_stack_direct_forward_offload[H, Dh, N_IMG, N_CTX, S](
        noisy_h, text_h, pooled_h, sigma, base, loader,
        dora, empty_flat_direct_oft_set(), SD35_DIRECT_ALGO_DORA,
        depth, targets, D, MLP, IN_CH, CTX_CH, OUT_CH, TIMESTEP_DIM,
        POOLED_DIM, eps, qk_eps, ctx,
    )


def sd35_stack_direct_oft_forward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_CTX: Int, S: Int
](
    noisy_h: List[Float32],
    text_h: List[Float32],
    pooled_h: List[Float32],
    sigma: Float32,
    base: SD35StackBase,
    mut loader: TurboPlannedLoader,
    oft: FlatDirectOFTSet,
    depth: Int, targets: Int,
    D: Int, MLP: Int, IN_CH: Int, CTX_CH: Int, OUT_CH: Int,
    TIMESTEP_DIM: Int, POOLED_DIM: Int,
    eps: Float32, qk_eps: Float32,
    ctx: DeviceContext,
) raises -> SD35StackForward:
    if len(oft.ad) != _sd35_direct_expected_slots(depth, targets):
        raise Error("sd35_stack_direct_oft_forward_offload: direct slot count mismatch")
    return _sd35_stack_direct_forward_offload[H, Dh, N_IMG, N_CTX, S](
        noisy_h, text_h, pooled_h, sigma, base, loader,
        empty_flat_direct_dora_set(), oft, SD35_DIRECT_ALGO_OFT,
        depth, targets, D, MLP, IN_CH, CTX_CH, OUT_CH, TIMESTEP_DIM,
        POOLED_DIM, eps, qk_eps, ctx,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FULL BACKWARD WITH LoRA, BLOCK-SWAP OFFLOAD (REVERSE block stream).
#   Collects LoRA d_A/d_B from all 8 slots per block.
#   Per-block modulation linear grads are DISCARDED (frozen base).
# ═══════════════════════════════════════════════════════════════════════════════
def sd35_stack_lora_backward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_CTX: Int, S: Int
](
    d_out: List[Float32],         # [N_IMG, OUT_CH] loss grad
    noisy_h: List[Float32],       # [N_IMG, IN_CH] — used for input-proj backward
    text_h: List[Float32],        # [N_CTX, CTX_CH]
    base: SD35StackBase,
    mut loader: TurboPlannedLoader,
    lora: SD35LoraSet,
    saved: SD35StackForward,
    D: Int, MLP: Int, IN_CH: Int, CTX_CH: Int, OUT_CH: Int,
    TIMESTEP_DIM: Int, POOLED_DIM: Int,
    eps: Float32, qk_eps: Float32,
    ctx: DeviceContext,
    num_dual: Int = 0,   # blocks [0, num_dual) are DUAL-attention
    last_ctx_preonly: Bool = False,
) raises -> SD35LoraGradSet:
    var depth = lora.depth

    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)

    var n_adapters = total_adapters(lora)
    # zero-init every adapter grad to its correct length: dual blocks only fill
    # SLOT_X_QKV, so the other slots must stay valid zeros (AdamW no-op), not empty.
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for ai in range(n_adapters):
        d_a_flat.append(_zeros(len(lora.ad[ai].a)))
        d_b_flat.append(_zeros(len(lora.ad[ai].b)))
    var nonfinite = 0

    # ── final layer backward ──
    # out = linear(modulate(layernorm(x), scale, shift), fl_lin_w)
    # d_out -> d_x_mod -> d_layernorm -> d_x
    var final_dtype = base.fl_lin_w[].dtype()
    var d_x_mod = linear_backward_dx(
        _t_as(d_out, [N_IMG, OUT_CH], final_dtype, ctx),
        base.fl_lin_w[],
        N_IMG, D, OUT_CH, ctx,
    ).to_host(ctx)

    # modulate backward: o = (1+scale)*ln_x + shift; saved final_ada_flat[D:2D]=scale
    var scale_final = List[Float32]()
    var shift_final = List[Float32]()
    for i in range(D):
        shift_final.append(saved.final_ada_flat[i])
    for i in range(D):
        scale_final.append(saved.final_ada_flat[D + i])
    var mb_fl = modulate_backward(
        _t_as(d_x_mod, [N_IMG, D], final_dtype, ctx),
        _t_as(_layer_norm_x(saved.pre_final_x.copy(), N_IMG, D, eps, final_dtype, ctx), [N_IMG, D], final_dtype, ctx),
        _t_as(scale_final^, [D], final_dtype, ctx), ctx,
    )
    var d_ln_x = mb_fl.d_x^.to_host(ctx)
    # layer_norm backward
    var lnb_fl = layer_norm_backward(
        _t_as(d_ln_x, [N_IMG, D], final_dtype, ctx),
        _t_as(saved.pre_final_x.copy(), [N_IMG, D], final_dtype, ctx),
        _t_as(_ones_list(D), [D], final_dtype, ctx),
        eps, ctx,
    )
    var d_x = lnb_fl.d_x^.to_host(ctx)
    var d_ctx = _zeros_list(N_CTX * D)

    # ── joint block backward (REVERSE) ──
    var bi = depth - 1
    while bi >= 0:
        var block_idx = bi
        var handle = loader.await_block(block_idx, ctx)
        if block_idx > 0:
            loader.prefetch_with_ctx(block_idx - 1, ctx)

        var pfx = handle.prefix + String(".")
        var scale = Float32(1.0) / Float32(8.0)
        var base_lora = _block_base(bi)

        if last_ctx_preonly and bi == depth - 1:
            # ── CONTEXT_PRE_ONLY final block backward ──
            var xw = _stream_weights_from_block(handle.block, pfx + String("x_block."), D, MLP, Dh, ctx)
            var x_mv = _chunk6(saved.blocks[bi].x_ada_flat.copy(), D)
            var cqw = _ctx_preonly_qkv_from_block(handle.block, pfx + String("context_block."), ctx)
            # ctx_scale = chunk1 of the saved [2D] continuous-mod flat
            var ctx_scale = List[Float32]()
            for cc in range(D):
                ctx_scale.append(saved.blocks[bi].ctx_ada_flat[D + cc])
            var cg = sd35_context_preonly_backward[1, S, H, Dh](
                d_x.copy(), cqw[0].copy(), cqw[2].copy(), cqw[3].copy(), ctx_scale^,
                xw, x_mv.copy(), saved.blocks[bi].ctxpre_fwd,
                N_CTX, N_IMG, D, MLP, eps, qk_eps, scale, ctx,
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_CTX_QKV].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_QKV].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_PROJ].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_FC1].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_FC2].copy()),
            )
            d_x = cg.d_x.copy()
            d_ctx = cg.d_ctx.copy()
            d_a_flat[base_lora + SLOT_CTX_QKV] = cg.ctx_qkv_lora_d_a.copy()
            d_b_flat[base_lora + SLOT_CTX_QKV] = cg.ctx_qkv_lora_d_b.copy()
            d_a_flat[base_lora + SLOT_X_QKV] = cg.x_lora.qkv_d_a.copy()
            d_b_flat[base_lora + SLOT_X_QKV] = cg.x_lora.qkv_d_b.copy()
            d_a_flat[base_lora + SLOT_X_PROJ] = cg.x_lora.proj_d_a.copy()
            d_b_flat[base_lora + SLOT_X_PROJ] = cg.x_lora.proj_d_b.copy()
            d_a_flat[base_lora + SLOT_X_FC1] = cg.x_lora.fc1_d_a.copy()
            d_b_flat[base_lora + SLOT_X_FC1] = cg.x_lora.fc1_d_b.copy()
            d_a_flat[base_lora + SLOT_X_FC2] = cg.x_lora.fc2_d_a.copy()
            d_b_flat[base_lora + SLOT_X_FC2] = cg.x_lora.fc2_d_b.copy()
            for s in range(SLOTS_PER_BLOCK):
                nonfinite += _nonfinite_count(d_a_flat[base_lora + s])
                nonfinite += _nonfinite_count(d_b_flat[base_lora + s])
            loader.mark_active_block_done(ctx)
            bi -= 1
            continue

        var bwr = _joint_weights_from_block(handle.block, pfx, D, MLP, Dh, ctx)
        var w = bwr.w.copy()
        # adaLN weights unused in backward (modvecs rebuilt from saved flat)

        var ctx_mv = _chunk6(saved.blocks[bi].ctx_ada_flat.copy(), D)

        if bi < num_dual:
            # ── DUAL block backward (x adaLN was 9-chunk; rebuild mv + msa2) ──
            var x_mv = _chunk6(saved.blocks[bi].x_ada_flat.copy(), D)
            var m2 = _chunk9_msa2(saved.blocks[bi].x_ada_flat.copy(), D)
            var a2w = _attn2_weights_from_block(handle.block, pfx + String("x_block."), ctx)
            var dg = sd35_dual_joint_block_backward[1, S, N_IMG, H, Dh](
                d_ctx.copy(), d_x.copy(), w.ctxw, w.xw, a2w, ctx_mv.copy(), x_mv.copy(),
                m2[1].copy(), m2[2].copy(), saved.blocks[bi].dual_fwd,
                N_CTX, N_IMG, D, MLP, eps, qk_eps, scale, ctx,
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_QKV].copy()),
                None,
            )
            d_x = dg.d_x.copy()
            d_ctx = dg.d_ctx.copy()
            # only SLOT_X_QKV LoRA is active on dual blocks; rest stay zero (no-op)
            d_a_flat[base_lora + SLOT_X_QKV] = dg.x_qkv_lora_d_a.copy()
            d_b_flat[base_lora + SLOT_X_QKV] = dg.x_qkv_lora_d_b.copy()
        else:
            var x_mv = _chunk6(saved.blocks[bi].x_ada_flat.copy(), D)
            var bg = sd35_joint_block_backward[1, S, H, Dh](
                d_ctx.copy(), d_x.copy(),
                w, ctx_mv.copy(), x_mv.copy(),
                saved.blocks[bi].fwd,
                N_CTX, N_IMG, D, MLP, eps, qk_eps, scale, ctx,
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_CTX_QKV].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_CTX_PROJ].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_CTX_FC1].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_CTX_FC2].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_QKV].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_PROJ].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_FC1].copy()),
                Optional[LoraAdapter](lora.ad[base_lora + SLOT_X_FC2].copy()),
            )
            d_x = bg.d_x.copy()
            d_ctx = bg.d_ctx.copy()
            d_a_flat[base_lora + SLOT_CTX_QKV] = bg.ctx_lora.qkv_d_a.copy()
            d_b_flat[base_lora + SLOT_CTX_QKV] = bg.ctx_lora.qkv_d_b.copy()
            d_a_flat[base_lora + SLOT_CTX_PROJ] = bg.ctx_lora.proj_d_a.copy()
            d_b_flat[base_lora + SLOT_CTX_PROJ] = bg.ctx_lora.proj_d_b.copy()
            d_a_flat[base_lora + SLOT_CTX_FC1] = bg.ctx_lora.fc1_d_a.copy()
            d_b_flat[base_lora + SLOT_CTX_FC1] = bg.ctx_lora.fc1_d_b.copy()
            d_a_flat[base_lora + SLOT_CTX_FC2] = bg.ctx_lora.fc2_d_a.copy()
            d_b_flat[base_lora + SLOT_CTX_FC2] = bg.ctx_lora.fc2_d_b.copy()
            d_a_flat[base_lora + SLOT_X_QKV] = bg.x_lora.qkv_d_a.copy()
            d_b_flat[base_lora + SLOT_X_QKV] = bg.x_lora.qkv_d_b.copy()
            d_a_flat[base_lora + SLOT_X_PROJ] = bg.x_lora.proj_d_a.copy()
            d_b_flat[base_lora + SLOT_X_PROJ] = bg.x_lora.proj_d_b.copy()
            d_a_flat[base_lora + SLOT_X_FC1] = bg.x_lora.fc1_d_a.copy()
            d_b_flat[base_lora + SLOT_X_FC1] = bg.x_lora.fc1_d_b.copy()
            d_a_flat[base_lora + SLOT_X_FC2] = bg.x_lora.fc2_d_a.copy()
            d_b_flat[base_lora + SLOT_X_FC2] = bg.x_lora.fc2_d_b.copy()

        for s in range(SLOTS_PER_BLOCK):
            nonfinite += _nonfinite_count(d_a_flat[base_lora + s])
            nonfinite += _nonfinite_count(d_b_flat[base_lora + s])

        loader.mark_active_block_done(ctx)
        bi -= 1

    return SD35LoraGradSet(d_a_flat^, d_b_flat^, nonfinite)


def _sd35_scatter_dora_block(
    mut grads: FlatDirectDoRAGrads, direct: SD35BlockDirectLycoris,
    g: SD35JointBlockDirectGrads,
) raises:
    _sd35_scatter_dora(grads, direct.ctx_qkv_slot, g.ctx_g.qkv)
    _sd35_scatter_dora(grads, direct.ctx_proj_slot, g.ctx_g.proj)
    _sd35_scatter_dora(grads, direct.ctx_fc1_slot, g.ctx_g.fc1)
    _sd35_scatter_dora(grads, direct.ctx_fc2_slot, g.ctx_g.fc2)
    _sd35_scatter_dora(grads, direct.x_qkv_slot, g.x_g.qkv)
    _sd35_scatter_dora(grads, direct.x_proj_slot, g.x_g.proj)
    _sd35_scatter_dora(grads, direct.x_fc1_slot, g.x_g.fc1)
    _sd35_scatter_dora(grads, direct.x_fc2_slot, g.x_g.fc2)


def _sd35_scatter_oft_block(
    mut grads: FlatDirectOFTGrads, direct: SD35BlockDirectLycoris,
    g: SD35JointBlockDirectGrads,
) raises:
    _sd35_scatter_oft(grads, direct.ctx_qkv_slot, g.ctx_g.qkv)
    _sd35_scatter_oft(grads, direct.ctx_proj_slot, g.ctx_g.proj)
    _sd35_scatter_oft(grads, direct.ctx_fc1_slot, g.ctx_g.fc1)
    _sd35_scatter_oft(grads, direct.ctx_fc2_slot, g.ctx_g.fc2)
    _sd35_scatter_oft(grads, direct.x_qkv_slot, g.x_g.qkv)
    _sd35_scatter_oft(grads, direct.x_proj_slot, g.x_g.proj)
    _sd35_scatter_oft(grads, direct.x_fc1_slot, g.x_g.fc1)
    _sd35_scatter_oft(grads, direct.x_fc2_slot, g.x_g.fc2)


def sd35_stack_direct_dora_backward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_CTX: Int, S: Int
](
    d_out: List[Float32],
    noisy_h: List[Float32],
    text_h: List[Float32],
    base: SD35StackBase,
    mut loader: TurboPlannedLoader,
    dora: FlatDirectDoRASet,
    saved: SD35StackForward,
    depth: Int, targets: Int,
    D: Int, MLP: Int, IN_CH: Int, CTX_CH: Int, OUT_CH: Int,
    TIMESTEP_DIM: Int, POOLED_DIM: Int,
    eps: Float32, qk_eps: Float32,
    ctx: DeviceContext,
) raises -> SD35DirectDoRAGradSet:
    if len(dora.ad) != _sd35_direct_expected_slots(depth, targets):
        raise Error("sd35_stack_direct_dora_backward_offload: direct slot count mismatch")
    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)
    var dora_grads = _sd35_direct_dora_zero_grads(dora)
    var nonfinite = 0

    var final_dtype = base.fl_lin_w[].dtype()
    var d_x_mod = linear_backward_dx(
        _t_as(d_out, [N_IMG, OUT_CH], final_dtype, ctx),
        base.fl_lin_w[],
        N_IMG, D, OUT_CH, ctx,
    ).to_host(ctx)

    var scale_final = List[Float32]()
    for i in range(D):
        scale_final.append(saved.final_ada_flat[D + i])
    var mb_fl = modulate_backward(
        _t_as(d_x_mod, [N_IMG, D], final_dtype, ctx),
        _t_as(_layer_norm_x(saved.pre_final_x.copy(), N_IMG, D, eps, final_dtype, ctx), [N_IMG, D], final_dtype, ctx),
        _t_as(scale_final^, [D], final_dtype, ctx), ctx,
    )
    var d_ln_x = mb_fl.d_x^.to_host(ctx)
    var lnb_fl = layer_norm_backward(
        _t_as(d_ln_x, [N_IMG, D], final_dtype, ctx),
        _t_as(saved.pre_final_x.copy(), [N_IMG, D], final_dtype, ctx),
        _t_as(_ones_list(D), [D], final_dtype, ctx),
        eps, ctx,
    )
    var d_x = lnb_fl.d_x^.to_host(ctx)
    var d_ctx = _zeros_list(N_CTX * D)

    var bi = depth - 1
    while bi >= 0:
        var handle = loader.await_block(bi, ctx)
        if bi > 0:
            loader.prefetch_with_ctx(bi - 1, ctx)
        var pfx = handle.prefix + String(".")
        var direct = _sd35_direct_block_for_dora(dora, bi, depth, targets)
        if bi == depth - 1:
            var xw = _stream_weights_from_block(handle.block, pfx + String("x_block."), D, MLP, Dh, ctx)
            var x_mv = _chunk6(saved.blocks[bi].x_ada_flat.copy(), D)
            var cqw = _ctx_preonly_qkv_from_block(handle.block, pfx + String("context_block."), ctx)
            var ctx_scale = List[Float32]()
            for cc in range(D):
                ctx_scale.append(saved.blocks[bi].ctx_ada_flat[D + cc])
            var bg = sd35_context_preonly_direct_lycoris_backward[1, S, H, Dh](
                d_x.copy(), cqw[0].copy(), cqw[2].copy(), cqw[3].copy(), ctx_scale^,
                xw, x_mv.copy(), direct, saved.blocks[bi].ctxpre_fwd,
                N_CTX, N_IMG, D, MLP, eps, qk_eps, Float32(1.0) / Float32(8.0), ctx,
            )
            d_x = bg.d_x.copy()
            d_ctx = bg.d_ctx.copy()
            _sd35_scatter_dora_block(dora_grads, direct, bg)
            nonfinite += _sd35_nonfinite_block(bg)
            loader.mark_active_block_done(ctx)
            bi -= 1
            continue
        var bwr = _joint_weights_from_block(handle.block, pfx, D, MLP, Dh, ctx)
        var w = bwr.w.copy()
        var ctx_mv = _chunk6(saved.blocks[bi].ctx_ada_flat.copy(), D)
        var x_mv = _chunk6(saved.blocks[bi].x_ada_flat.copy(), D)
        var bg = sd35_joint_block_direct_lycoris_backward[1, S, H, Dh](
            d_ctx.copy(), d_x.copy(), w, ctx_mv.copy(), x_mv.copy(),
            direct, saved.blocks[bi].fwd,
            N_CTX, N_IMG, D, MLP, eps, qk_eps, Float32(1.0) / Float32(8.0), ctx,
        )
        d_x = bg.d_x.copy()
        d_ctx = bg.d_ctx.copy()
        _sd35_scatter_dora_block(dora_grads, direct, bg)
        nonfinite += _sd35_nonfinite_block(bg)
        loader.mark_active_block_done(ctx)
        bi -= 1

    # Exercise frozen input-projection backwards; grads are intentionally discarded.
    _ = linear_backward_dx(
        _t_as(d_x, [N_IMG, D], base.xe_w[].dtype(), ctx),
        reshape(base.xe_w[], [D, IN_CH], ctx),
        N_IMG, IN_CH, D, ctx,
    )
    _ = linear_backward_dx(
        _t_as(d_ctx, [N_CTX, D], base.ce_w[].dtype(), ctx),
        base.ce_w[],
        N_CTX, CTX_CH, D, ctx,
    )

    return SD35DirectDoRAGradSet(dora_grads^, nonfinite)


def sd35_stack_direct_oft_backward_offload[
    H: Int, Dh: Int, N_IMG: Int, N_CTX: Int, S: Int
](
    d_out: List[Float32],
    noisy_h: List[Float32],
    text_h: List[Float32],
    base: SD35StackBase,
    mut loader: TurboPlannedLoader,
    oft: FlatDirectOFTSet,
    saved: SD35StackForward,
    depth: Int, targets: Int,
    D: Int, MLP: Int, IN_CH: Int, CTX_CH: Int, OUT_CH: Int,
    TIMESTEP_DIM: Int, POOLED_DIM: Int,
    eps: Float32, qk_eps: Float32,
    ctx: DeviceContext,
) raises -> SD35DirectOFTGradSet:
    if len(oft.ad) != _sd35_direct_expected_slots(depth, targets):
        raise Error("sd35_stack_direct_oft_backward_offload: direct slot count mismatch")
    if loader.block_count() > 0:
        loader.prefetch_with_ctx(loader.block_count() - 1, ctx)
    var oft_grads = _sd35_direct_oft_zero_grads(oft)
    var nonfinite = 0

    var final_dtype = base.fl_lin_w[].dtype()
    var d_x_mod = linear_backward_dx(
        _t_as(d_out, [N_IMG, OUT_CH], final_dtype, ctx),
        base.fl_lin_w[],
        N_IMG, D, OUT_CH, ctx,
    ).to_host(ctx)

    var scale_final = List[Float32]()
    for i in range(D):
        scale_final.append(saved.final_ada_flat[D + i])
    var mb_fl = modulate_backward(
        _t_as(d_x_mod, [N_IMG, D], final_dtype, ctx),
        _t_as(_layer_norm_x(saved.pre_final_x.copy(), N_IMG, D, eps, final_dtype, ctx), [N_IMG, D], final_dtype, ctx),
        _t_as(scale_final^, [D], final_dtype, ctx), ctx,
    )
    var d_ln_x = mb_fl.d_x^.to_host(ctx)
    var lnb_fl = layer_norm_backward(
        _t_as(d_ln_x, [N_IMG, D], final_dtype, ctx),
        _t_as(saved.pre_final_x.copy(), [N_IMG, D], final_dtype, ctx),
        _t_as(_ones_list(D), [D], final_dtype, ctx),
        eps, ctx,
    )
    var d_x = lnb_fl.d_x^.to_host(ctx)
    var d_ctx = _zeros_list(N_CTX * D)

    var bi = depth - 1
    while bi >= 0:
        var handle = loader.await_block(bi, ctx)
        if bi > 0:
            loader.prefetch_with_ctx(bi - 1, ctx)
        var pfx = handle.prefix + String(".")
        var direct = _sd35_direct_block_for_oft(oft, bi, depth, targets)
        if bi == depth - 1:
            var xw = _stream_weights_from_block(handle.block, pfx + String("x_block."), D, MLP, Dh, ctx)
            var x_mv = _chunk6(saved.blocks[bi].x_ada_flat.copy(), D)
            var cqw = _ctx_preonly_qkv_from_block(handle.block, pfx + String("context_block."), ctx)
            var ctx_scale = List[Float32]()
            for cc in range(D):
                ctx_scale.append(saved.blocks[bi].ctx_ada_flat[D + cc])
            var bg = sd35_context_preonly_direct_lycoris_backward[1, S, H, Dh](
                d_x.copy(), cqw[0].copy(), cqw[2].copy(), cqw[3].copy(), ctx_scale^,
                xw, x_mv.copy(), direct, saved.blocks[bi].ctxpre_fwd,
                N_CTX, N_IMG, D, MLP, eps, qk_eps, Float32(1.0) / Float32(8.0), ctx,
            )
            d_x = bg.d_x.copy()
            d_ctx = bg.d_ctx.copy()
            _sd35_scatter_oft_block(oft_grads, direct, bg)
            nonfinite += _sd35_nonfinite_block(bg)
            loader.mark_active_block_done(ctx)
            bi -= 1
            continue
        var bwr = _joint_weights_from_block(handle.block, pfx, D, MLP, Dh, ctx)
        var w = bwr.w.copy()
        var ctx_mv = _chunk6(saved.blocks[bi].ctx_ada_flat.copy(), D)
        var x_mv = _chunk6(saved.blocks[bi].x_ada_flat.copy(), D)
        var bg = sd35_joint_block_direct_lycoris_backward[1, S, H, Dh](
            d_ctx.copy(), d_x.copy(), w, ctx_mv.copy(), x_mv.copy(),
            direct, saved.blocks[bi].fwd,
            N_CTX, N_IMG, D, MLP, eps, qk_eps, Float32(1.0) / Float32(8.0), ctx,
        )
        d_x = bg.d_x.copy()
        d_ctx = bg.d_ctx.copy()
        _sd35_scatter_oft_block(oft_grads, direct, bg)
        nonfinite += _sd35_nonfinite_block(bg)
        loader.mark_active_block_done(ctx)
        bi -= 1

    _ = linear_backward_dx(
        _t_as(d_x, [N_IMG, D], base.xe_w[].dtype(), ctx),
        reshape(base.xe_w[], [D, IN_CH], ctx),
        N_IMG, IN_CH, D, ctx,
    )
    _ = linear_backward_dx(
        _t_as(d_ctx, [N_CTX, D], base.ce_w[].dtype(), ctx),
        base.ce_w[],
        N_CTX, CTX_CH, D, ctx,
    )

    return SD35DirectOFTGradSet(oft_grads^, nonfinite)


# ── Helpers for backward ──────────────────────────────────────────────────────
def _layer_norm_x(
    x_h: List[Float32], N: Int, D: Int, eps: Float32, dtype: STDtype, ctx: DeviceContext,
) raises -> List[Float32]:
    return layer_norm(
        _t_as(x_h, [N, D], dtype, ctx),
        _t_as(_ones_list(D), [D], dtype, ctx),
        _t_as(_zeros_list(D), [D], dtype, ctx),
        eps, ctx,
    ).to_host(ctx)
