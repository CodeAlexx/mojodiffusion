# models/qwenimage/qwenimage_direct_lycoris_stack.mojo -- Qwen-Image direct DoRA/OFT slots.
#
# Qwen-Image has 60 double-stream blocks with 12 trained projections per block.
# This module owns the model-specific direct DoRA/OFT slot map, byte preflight,
# projection wrappers, optimizer helpers, and save names used before lowering the
# direct W_eff path into qwenimage_block/qwenimage_stack_lora.

from std.collections import List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.dora_save import NamedDoRA, save_dora_onetrainer
from serenitymojo.training.dora_adapter import DoRAGrads
from serenitymojo.training.oft_onetrainer import OFTOTGrads
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectDoRAGrads, FlatDirectOFTSet, FlatDirectOFTGrads,
    empty_flat_direct_dora_set, empty_flat_direct_oft_set,
    flat_direct_dora_append_from_weight, flat_direct_oft_append,
    flat_direct_dora_forward_slot, flat_direct_oft_forward_slot,
    flat_direct_dora_backward_slot, flat_direct_oft_backward_slot,
    flat_direct_dora_grad_norm, flat_direct_dora_clip_grads,
    flat_direct_dora_adamw_step, flat_direct_dora_zero_leg_l1,
    flat_direct_dora_trainable_bytes,
    flat_direct_oft_grad_norm, flat_direct_oft_clip_grads,
    flat_direct_oft_adamw_step, flat_direct_oft_vec_l1,
    flat_direct_oft_trainable_bytes,
)
from serenitymojo.models.qwenimage.qwenimage_stack_lora import DBL_SLOTS
from serenitymojo.models.qwenimage.qwenimage_lycoris_stack import (
    QWEN_LYCORIS_TGT_ATTN, QWEN_LYCORIS_TGT_ALL,
    qwen_lycoris_slot_dims,
)


comptime QWEN_DIRECT_24_GIB = 24 * 1024 * 1024 * 1024
comptime QD_IMG_Q = 0
comptime QD_IMG_K = 1
comptime QD_IMG_V = 2
comptime QD_IMG_OUT = 3
comptime QD_IMG_FF_UP = 4
comptime QD_IMG_FF_DOWN = 5
comptime QD_TXT_Q = 6
comptime QD_TXT_K = 7
comptime QD_TXT_V = 8
comptime QD_TXT_OUT = 9
comptime QD_TXT_FF_UP = 10
comptime QD_TXT_FF_DOWN = 11


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _f32_2d(var values: List[Float32], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


def _validate_targets(targets: Int) raises:
    if targets < QWEN_LYCORIS_TGT_ATTN or targets > QWEN_LYCORIS_TGT_ALL:
        raise Error("qwen direct LyCORIS: targets must be 1(attn)|2(all)")


def _slot_is_attn(slot: Int) -> Bool:
    var s = slot % DBL_SLOTS
    return s <= QD_IMG_OUT or (s >= QD_TXT_Q and s <= QD_TXT_OUT)


def _slot_targeted(slot: Int, targets: Int) -> Bool:
    if _slot_is_attn(slot):
        return targets >= QWEN_LYCORIS_TGT_ATTN
    return targets >= QWEN_LYCORIS_TGT_ALL


def qwen_direct_slot_prefix(block: Int, slot: Int) raises -> String:
    var b = String("transformer.transformer_blocks.") + String(block)
    if slot == QD_IMG_Q:
        return b + String(".attn.to_q")
    if slot == QD_IMG_K:
        return b + String(".attn.to_k")
    if slot == QD_IMG_V:
        return b + String(".attn.to_v")
    if slot == QD_IMG_OUT:
        return b + String(".attn.to_out.0")
    if slot == QD_IMG_FF_UP:
        return b + String(".img_mlp.net.0.proj")
    if slot == QD_IMG_FF_DOWN:
        return b + String(".img_mlp.net.2")
    if slot == QD_TXT_Q:
        return b + String(".attn.add_q_proj")
    if slot == QD_TXT_K:
        return b + String(".attn.add_k_proj")
    if slot == QD_TXT_V:
        return b + String(".attn.add_v_proj")
    if slot == QD_TXT_OUT:
        return b + String(".attn.to_add_out")
    if slot == QD_TXT_FF_UP:
        return b + String(".txt_mlp.net.0.proj")
    if slot == QD_TXT_FF_DOWN:
        return b + String(".txt_mlp.net.2")
    raise Error(String("qwen_direct_slot_prefix: bad slot ") + String(slot))


def qwen_direct_active_slot_count(num_blocks: Int, targets: Int) raises -> Int:
    _validate_targets(targets)
    var n = 0
    for _bi in range(num_blocks):
        for slot in range(DBL_SLOTS):
            if _slot_targeted(slot, targets):
                n += 1
    return n


def empty_qwen_direct_dora_set() -> FlatDirectDoRASet:
    return empty_flat_direct_dora_set()


def empty_qwen_direct_oft_set() -> FlatDirectOFTSet:
    return empty_flat_direct_oft_set()


def qwen_direct_dense_carrier_bytes(
    num_blocks: Int, D: Int, F: Int, targets: Int,
) raises -> Int:
    _validate_targets(targets)
    var elems = 0
    for _bi in range(num_blocks):
        for slot in range(DBL_SLOTS):
            if not _slot_targeted(slot, targets):
                continue
            var dims = qwen_lycoris_slot_dims(slot, D, F)
            var in_f = dims[0]
            var out_f = dims[1]
            elems += in_f * in_f + out_f * in_f
    return elems * 2


def qwen_direct_dora_trainable_bytes_estimate(
    num_blocks: Int, D: Int, F: Int, rank: Int, targets: Int,
    wd_on_out: Bool = False,
) raises -> Int:
    _validate_targets(targets)
    var total = 0
    for _bi in range(num_blocks):
        for slot in range(DBL_SLOTS):
            if not _slot_targeted(slot, targets):
                continue
            var dims = qwen_lycoris_slot_dims(slot, D, F)
            var in_f = dims[0]
            var out_f = dims[1]
            var mlen = out_f if wd_on_out else in_f
            var bf16_elems = rank * in_f + out_f * rank
            var f32_elems = mlen + (2 * rank * in_f) + (2 * out_f * rank) + (2 * mlen)
            total += bf16_elems * 2 + f32_elems * 4
    return total


def qwen_direct_oft_trainable_bytes_estimate(
    num_blocks: Int, D: Int, F: Int, block_size: Int, targets: Int,
) raises -> Int:
    _validate_targets(targets)
    var total = 0
    for _bi in range(num_blocks):
        for slot in range(DBL_SLOTS):
            if not _slot_targeted(slot, targets):
                continue
            var dims = qwen_lycoris_slot_dims(slot, D, F)
            var in_f = dims[0]
            if in_f % block_size != 0:
                raise Error("qwen_direct_oft_trainable_bytes_estimate: in_f not divisible by block_size")
            var r = in_f // block_size
            var ne = block_size * (block_size - 1) // 2
            total += 3 * r * ne * 4
    return total


def qwen_direct_dora_preflight(
    num_blocks: Int, D: Int, F: Int, rank: Int, targets: Int,
    budget_bytes: Int, wd_on_out: Bool = False,
) raises -> Int:
    var direct = qwen_direct_dora_trainable_bytes_estimate(
        num_blocks, D, F, rank, targets, wd_on_out,
    )
    if direct > budget_bytes:
        raise Error(
            String("Qwen-Image direct DoRA trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


def qwen_direct_oft_preflight(
    num_blocks: Int, D: Int, F: Int, block_size: Int, targets: Int,
    budget_bytes: Int,
) raises -> Int:
    var direct = qwen_direct_oft_trainable_bytes_estimate(
        num_blocks, D, F, block_size, targets,
    )
    if direct > budget_bytes:
        raise Error(
            String("Qwen-Image direct OFT trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


def qwen_direct_runtime_blocker(algo_name: String) -> String:
    return (
        String("Qwen-Image trainer: network_algorithm=") + algo_name
        + String(" passed direct-state 24 GiB preflight, but production ")
        + String("Qwen DoRA/OFT still needs trainer dispatch around the ")
        + String("GPU direct W_eff block/stack lowering. Do not route through host flat_direct_* ")
        + String("substitution for real training: the 60-block Qwen projection ")
        + String("sizes make host full-matmul infeasible and would not satisfy ")
        + String("the 24 GiB runtime claim. Missing: trainer control flow ")
        + String("that builds/calls the direct qwenimage_stack_lora/offload ")
        + String("wrappers, optimizer/update/save/resume dispatch around ")
        + String("the returned direct grad sets, and a peak-VRAM runtime gate.")
    )


def _add_bias(
    var y: List[Float32], bias: List[Float32], M: Int, out_f: Int,
) raises -> List[Float32]:
    if len(y) != M * out_f:
        raise Error("_add_bias: y numel mismatch")
    if len(bias) != out_f:
        raise Error("_add_bias: bias numel mismatch")
    for m in range(M):
        for o in range(out_f):
            y[m * out_f + o] += bias[o]
    return y^


def build_qwen_direct_dora_set_from_weights(
    weights: List[List[Float32]], num_blocks: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool = False,
) raises -> FlatDirectDoRASet:
    if len(weights) != num_blocks * DBL_SLOTS:
        raise Error("build_qwen_direct_dora_set_from_weights: weight count mismatch")
    var set = empty_flat_direct_dora_set()
    for bi in range(num_blocks):
        var block_weights = List[List[Float32]]()
        for slot in range(DBL_SLOTS):
            block_weights.append(weights[bi * DBL_SLOTS + slot].copy())
        qwen_direct_dora_append_block_weights(
            set, bi, block_weights^, D, F, rank, alpha,
            targets, seed + UInt64(bi * DBL_SLOTS), wd_on_out,
        )
    return set^


def qwen_direct_dora_append_block_weights(
    mut set: FlatDirectDoRASet, block: Int, weights: List[List[Float32]],
    D: Int, F: Int, rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool = False,
) raises:
    _validate_targets(targets)
    if len(weights) != DBL_SLOTS:
        raise Error("qwen_direct_dora_append_block_weights: expected 12 block weights")
    var s = seed
    for slot in range(DBL_SLOTS):
        var dims = qwen_lycoris_slot_dims(slot, D, F)
        if _slot_targeted(slot, targets):
            if len(weights[slot]) != dims[0] * dims[1]:
                raise Error("qwen_direct_dora_append_block_weights: weight numel mismatch")
            flat_direct_dora_append_from_weight(
                set, weights[slot].copy(), dims[0], dims[1], rank, alpha,
                qwen_direct_slot_prefix(block, slot), s, wd_on_out,
            )
        s += 1


def build_qwen_direct_oft_set(
    num_blocks: Int, D: Int, F: Int, block_size: Int, targets: Int,
) raises -> FlatDirectOFTSet:
    var set = empty_flat_direct_oft_set()
    for bi in range(num_blocks):
        qwen_direct_oft_append_block(set, bi, D, F, block_size, targets)
    return set^


def qwen_direct_oft_append_block(
    mut set: FlatDirectOFTSet, block: Int,
    D: Int, F: Int, block_size: Int, targets: Int,
) raises:
    _validate_targets(targets)
    for slot in range(DBL_SLOTS):
        if not _slot_targeted(slot, targets):
            continue
        var dims = qwen_lycoris_slot_dims(slot, D, F)
        flat_direct_oft_append(
            set, dims[0], dims[1], block_size,
            qwen_direct_slot_prefix(block, slot),
        )


def qwen_direct_dora_projection_forward(
    set: FlatDirectDoRASet, slot: Int, x_h: List[Float32],
    w_orig: List[Float32], bias: List[Float32], M: Int,
) raises -> List[Float32]:
    var y = flat_direct_dora_forward_slot(set, slot, x_h, w_orig, M)
    return _add_bias(y^, bias, M, set.ad[slot].out_f)


def qwen_direct_dora_projection_backward(
    set: FlatDirectDoRASet, slot: Int, d_y_h: List[Float32],
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> DoRAGrads:
    return flat_direct_dora_backward_slot(set, slot, d_y_h, x_h, w_orig, M)


def qwen_direct_oft_projection_forward(
    set: FlatDirectOFTSet, slot: Int, x_h: List[Float32],
    w_orig: List[Float32], bias: List[Float32], M: Int,
) raises -> List[Float32]:
    var y = flat_direct_oft_forward_slot(set, slot, x_h, w_orig, M)
    return _add_bias(y^, bias, M, set.ad[slot].out_f)


def qwen_direct_oft_projection_backward(
    set: FlatDirectOFTSet, slot: Int, d_y_h: List[Float32],
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> OFTOTGrads:
    return flat_direct_oft_backward_slot(set, slot, d_y_h, x_h, w_orig, M)


def qwen_direct_dora_zero_grads(set: FlatDirectDoRASet) -> FlatDirectDoRAGrads:
    var out = List[DoRAGrads]()
    for i in range(len(set.ad)):
        ref d = set.ad[i]
        out.append(DoRAGrads(
            _zeros(len(d.a)), _zeros(len(d.b)), _zeros(len(d.m)), List[Float32](),
        ))
    return FlatDirectDoRAGrads(out^)


def qwen_direct_dora_scatter_slot_grad(
    mut grads: FlatDirectDoRAGrads, slot: Int, g: DoRAGrads,
) raises:
    if slot < 0 or slot >= len(grads.g):
        raise Error("qwen_direct_dora_scatter_slot_grad: slot out of range")
    grads.g[slot] = DoRAGrads(g.d_a.copy(), g.d_b.copy(), g.d_m.copy(), List[Float32]())


def qwen_direct_dora_grad_norm(g: FlatDirectDoRAGrads) -> Float64:
    return flat_direct_dora_grad_norm(g)


def qwen_direct_dora_clip_grads(mut g: FlatDirectDoRAGrads, clip_scale: Float32):
    flat_direct_dora_clip_grads(g, clip_scale)


def qwen_direct_dora_adamw_step(
    mut set: FlatDirectDoRASet, g: FlatDirectDoRAGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_dora_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def qwen_direct_dora_zero_leg_l1(set: FlatDirectDoRASet) -> Float64:
    return flat_direct_dora_zero_leg_l1(set)


def qwen_direct_dora_trainable_bytes(set: FlatDirectDoRASet) -> Int:
    return flat_direct_dora_trainable_bytes(set)


def qwen_direct_oft_zero_grads(set: FlatDirectOFTSet) -> FlatDirectOFTGrads:
    var out = List[List[Float32]]()
    for i in range(len(set.ad)):
        out.append(_zeros(len(set.ad[i].vec)))
    return FlatDirectOFTGrads(out^)


def qwen_direct_oft_scatter_slot_grad(
    mut grads: FlatDirectOFTGrads, slot: Int, g: OFTOTGrads,
) raises:
    if slot < 0 or slot >= len(grads.d_vec):
        raise Error("qwen_direct_oft_scatter_slot_grad: slot out of range")
    grads.d_vec[slot] = g.d_vec.copy()


def qwen_direct_oft_grad_norm(g: FlatDirectOFTGrads) -> Float64:
    return flat_direct_oft_grad_norm(g)


def qwen_direct_oft_clip_grads(mut g: FlatDirectOFTGrads, clip_scale: Float32):
    flat_direct_oft_clip_grads(g, clip_scale)


def qwen_direct_oft_adamw_step(
    mut set: FlatDirectOFTSet, g: FlatDirectOFTGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_oft_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def qwen_direct_oft_vec_l1(set: FlatDirectOFTSet) -> Float64:
    return flat_direct_oft_vec_l1(set)


def qwen_direct_oft_trainable_bytes(set: FlatDirectOFTSet) -> Int:
    return flat_direct_oft_trainable_bytes(set)


def save_qwen_direct_dora(
    set: FlatDirectDoRASet, path: String, ctx: DeviceContext,
) raises -> Int:
    var named = List[NamedDoRA]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedDoRA(set.prefix[i].copy(), set.ad[i].copy()))
    return save_dora_onetrainer(named, path, ctx)


def save_qwen_direct_oft(
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
        tensors.append(ArcPointer(_f32_2d(sl.vec.copy(), sl.r, ne, ctx)))
        nmods += 1
    if nmods == 0:
        raise Error("save_qwen_direct_oft: refusing to write an empty OFT file")
    save_safetensors(names, tensors, path, ctx)
    return nmods
