# models/flux/flux_direct_lycoris_stack.mojo -- shared Flux/Chroma direct DoRA/OFT slots.
#
# Flux and Chroma share the flat block-projection LoRA surface: double blocks
# first (img stream 6 slots, txt stream 6 slots), then single blocks (5 slots).
# This module owns direct DoRA/OFT slot metadata, byte preflight, projection
# wrappers, optimizer helpers, and save names before direct W_eff is lowered into
# the shared block/stack path.

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
from serenitymojo.models.flux.lora_block import (
    DBL_STREAM_SLOTS, SGL_SLOTS,
    D_SQ, D_SK, D_SV, D_PROJ, D_MLP0, D_MLP2,
    S_SQ, S_SK, S_SV, S_PMLP, S_L2,
)
from serenitymojo.models.flux.flux_stack_lora import DBL_SLOTS_PER_BLOCK
from serenitymojo.models.flux.flux_lycoris_stack import (
    FLUX_LYCORIS_TGT_ATTN, FLUX_LYCORIS_TGT_ALL,
    flux_lycoris_dbl_slot_dims, flux_lycoris_sgl_slot_dims,
)


comptime FLUX_DIRECT_24_GIB = 24 * 1024 * 1024 * 1024


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
    if targets < FLUX_LYCORIS_TGT_ATTN or targets > FLUX_LYCORIS_TGT_ALL:
        raise Error("flux direct LyCORIS: targets must be 1(attn)|2(all)")


def _dbl_slot_is_attn(slot: Int) -> Bool:
    var s = slot % DBL_STREAM_SLOTS
    return s == D_SQ or s == D_SK or s == D_SV or s == D_PROJ


def _sgl_slot_is_attn(slot: Int) -> Bool:
    var s = slot % SGL_SLOTS
    return s == S_SQ or s == S_SK or s == S_SV


def _slot_targeted(is_double: Bool, slot: Int, targets: Int) -> Bool:
    if is_double:
        if _dbl_slot_is_attn(slot):
            return targets >= FLUX_LYCORIS_TGT_ATTN
        return targets >= FLUX_LYCORIS_TGT_ALL
    if _sgl_slot_is_attn(slot):
        return targets >= FLUX_LYCORIS_TGT_ATTN
    return targets >= FLUX_LYCORIS_TGT_ALL


def flux_direct_total_slots(num_double: Int, num_single: Int) -> Int:
    return num_double * DBL_SLOTS_PER_BLOCK + num_single * SGL_SLOTS


def _flat_is_double(flat: Int, num_double: Int) -> Bool:
    return flat < num_double * DBL_SLOTS_PER_BLOCK


def flux_direct_flat_dims(flat: Int, num_double: Int, D: Int, F: Int) raises -> Tuple[Int, Int]:
    var dbl_count = num_double * DBL_SLOTS_PER_BLOCK
    if flat < dbl_count:
        return flux_lycoris_dbl_slot_dims(flat % DBL_STREAM_SLOTS, D, F)
    return flux_lycoris_sgl_slot_dims((flat - dbl_count) % SGL_SLOTS, D, F)


def _dbl_prefix(bi: Int, stream_img: Bool, slot: Int) -> String:
    var b = String("lora_transformer_transformer_blocks_") + String(bi) + String("_")
    if stream_img:
        if slot == D_SQ:
            return b + String("attn_to_q")
        if slot == D_SK:
            return b + String("attn_to_k")
        if slot == D_SV:
            return b + String("attn_to_v")
        if slot == D_PROJ:
            return b + String("attn_to_out_0")
        if slot == D_MLP0:
            return b + String("ff_net_0_proj")
        return b + String("ff_net_2")
    if slot == D_SQ:
        return b + String("attn_add_q_proj")
    if slot == D_SK:
        return b + String("attn_add_k_proj")
    if slot == D_SV:
        return b + String("attn_add_v_proj")
    if slot == D_PROJ:
        return b + String("attn_to_add_out")
    if slot == D_MLP0:
        return b + String("ff_context_net_0_proj")
    return b + String("ff_context_net_2")


def _sgl_prefix(bi: Int, slot: Int) -> String:
    var b = String("lora_transformer_single_transformer_blocks_") + String(bi) + String("_")
    if slot == S_SQ:
        return b + String("attn_to_q")
    if slot == S_SK:
        return b + String("attn_to_k")
    if slot == S_SV:
        return b + String("attn_to_v")
    if slot == S_PMLP:
        return b + String("proj_mlp")
    return b + String("proj_out")


def flux_direct_flat_prefix(flat: Int, num_double: Int) -> String:
    var dbl_count = num_double * DBL_SLOTS_PER_BLOCK
    if flat < dbl_count:
        var bi = flat // DBL_SLOTS_PER_BLOCK
        var s12 = flat % DBL_SLOTS_PER_BLOCK
        var stream_img = s12 < DBL_STREAM_SLOTS
        return _dbl_prefix(bi, stream_img, s12 % DBL_STREAM_SLOTS)
    var off = flat - dbl_count
    return _sgl_prefix(off // SGL_SLOTS, off % SGL_SLOTS)


def flux_direct_active_slot_count(num_double: Int, num_single: Int, targets: Int) raises -> Int:
    _validate_targets(targets)
    var n = 0
    for flat in range(flux_direct_total_slots(num_double, num_single)):
        var is_d = _flat_is_double(flat, num_double)
        var slot = flat % DBL_STREAM_SLOTS if is_d else (flat - num_double * DBL_SLOTS_PER_BLOCK) % SGL_SLOTS
        if _slot_targeted(is_d, slot, targets):
            n += 1
    return n


def empty_flux_direct_dora_set() -> FlatDirectDoRASet:
    return empty_flat_direct_dora_set()


def empty_flux_direct_oft_set() -> FlatDirectOFTSet:
    return empty_flat_direct_oft_set()


def flux_direct_dense_carrier_bytes(
    num_double: Int, num_single: Int, D: Int, F: Int, targets: Int,
) raises -> Int:
    _validate_targets(targets)
    var elems = 0
    for flat in range(flux_direct_total_slots(num_double, num_single)):
        var is_d = _flat_is_double(flat, num_double)
        var slot = flat % DBL_STREAM_SLOTS if is_d else (flat - num_double * DBL_SLOTS_PER_BLOCK) % SGL_SLOTS
        if not _slot_targeted(is_d, slot, targets):
            continue
        var dims = flux_direct_flat_dims(flat, num_double, D, F)
        elems += dims[0] * dims[0] + dims[1] * dims[0]
    return elems * 2


def flux_direct_dora_trainable_bytes_estimate(
    num_double: Int, num_single: Int, D: Int, F: Int,
    rank: Int, targets: Int, wd_on_out: Bool = False,
) raises -> Int:
    _validate_targets(targets)
    var total = 0
    for flat in range(flux_direct_total_slots(num_double, num_single)):
        var is_d = _flat_is_double(flat, num_double)
        var slot = flat % DBL_STREAM_SLOTS if is_d else (flat - num_double * DBL_SLOTS_PER_BLOCK) % SGL_SLOTS
        if not _slot_targeted(is_d, slot, targets):
            continue
        var dims = flux_direct_flat_dims(flat, num_double, D, F)
        var in_f = dims[0]
        var out_f = dims[1]
        var mlen = out_f if wd_on_out else in_f
        var bf16_elems = rank * in_f + out_f * rank
        var f32_elems = mlen + (2 * rank * in_f) + (2 * out_f * rank) + (2 * mlen)
        total += bf16_elems * 2 + f32_elems * 4
    return total


def flux_direct_oft_trainable_bytes_estimate(
    num_double: Int, num_single: Int, D: Int, F: Int,
    block_size: Int, targets: Int,
) raises -> Int:
    _validate_targets(targets)
    var total = 0
    for flat in range(flux_direct_total_slots(num_double, num_single)):
        var is_d = _flat_is_double(flat, num_double)
        var slot = flat % DBL_STREAM_SLOTS if is_d else (flat - num_double * DBL_SLOTS_PER_BLOCK) % SGL_SLOTS
        if not _slot_targeted(is_d, slot, targets):
            continue
        var dims = flux_direct_flat_dims(flat, num_double, D, F)
        var in_f = dims[0]
        if in_f % block_size != 0:
            raise Error("flux_direct_oft_trainable_bytes_estimate: in_f not divisible by block_size")
        var r = in_f // block_size
        var ne = block_size * (block_size - 1) // 2
        total += 3 * r * ne * 4
    return total


def flux_direct_dora_preflight(
    num_double: Int, num_single: Int, D: Int, F: Int,
    rank: Int, targets: Int, budget_bytes: Int, wd_on_out: Bool = False,
) raises -> Int:
    var direct = flux_direct_dora_trainable_bytes_estimate(
        num_double, num_single, D, F, rank, targets, wd_on_out,
    )
    if direct > budget_bytes:
        raise Error(
            String("Flux/Chroma direct DoRA trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


def flux_direct_oft_preflight(
    num_double: Int, num_single: Int, D: Int, F: Int,
    block_size: Int, targets: Int, budget_bytes: Int,
) raises -> Int:
    var direct = flux_direct_oft_trainable_bytes_estimate(
        num_double, num_single, D, F, block_size, targets,
    )
    if direct > budget_bytes:
        raise Error(
            String("Flux/Chroma direct OFT trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


def flux_direct_runtime_blocker(model_name: String, algo_name: String) -> String:
    return (
        model_name
        + String(" trainer: network_algorithm=") + algo_name
        + String(" passed direct-state 24 GiB preflight, and direct stack wrappers plus streamed direct-set initialization are gated, but production ")
        + model_name
        + String(" DoRA/OFT still needs live trainer dispatch around the compiled model-specific GPU direct W_eff stack wrappers. ")
        + String("Do not route through host flat_direct_* substitution for real ")
        + String("training: fused qkv/mlp projection sizes make host full-matmul ")
        + String("infeasible and would not satisfy the 24 GiB runtime claim. ")
        + String("Missing: external trainer call-site dispatch, direct DoRA/OFT ")
        + String("master update/save/resume, and a peak-VRAM runtime gate.")
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


def build_flux_direct_dora_set_from_weights(
    weights: List[List[Float32]], num_double: Int, num_single: Int,
    D: Int, F: Int, rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool = False,
) raises -> FlatDirectDoRASet:
    if len(weights) != flux_direct_total_slots(num_double, num_single):
        raise Error("build_flux_direct_dora_set_from_weights: weight count mismatch")
    var set = empty_flat_direct_dora_set()
    var s = seed
    for flat in range(len(weights)):
        var is_d = _flat_is_double(flat, num_double)
        var slot = flat % DBL_STREAM_SLOTS if is_d else (flat - num_double * DBL_SLOTS_PER_BLOCK) % SGL_SLOTS
        var dims = flux_direct_flat_dims(flat, num_double, D, F)
        if _slot_targeted(is_d, slot, targets):
            if len(weights[flat]) != dims[0] * dims[1]:
                raise Error("build_flux_direct_dora_set_from_weights: weight numel mismatch")
            flat_direct_dora_append_from_weight(
                set, weights[flat].copy(), dims[0], dims[1], rank, alpha,
                flux_direct_flat_prefix(flat, num_double), s, wd_on_out,
            )
        s += 1
    return set^


def build_flux_direct_oft_set(
    num_double: Int, num_single: Int, D: Int, F: Int,
    block_size: Int, targets: Int,
) raises -> FlatDirectOFTSet:
    var set = empty_flat_direct_oft_set()
    for flat in range(flux_direct_total_slots(num_double, num_single)):
        var is_d = _flat_is_double(flat, num_double)
        var slot = flat % DBL_STREAM_SLOTS if is_d else (flat - num_double * DBL_SLOTS_PER_BLOCK) % SGL_SLOTS
        if not _slot_targeted(is_d, slot, targets):
            continue
        var dims = flux_direct_flat_dims(flat, num_double, D, F)
        flat_direct_oft_append(
            set, dims[0], dims[1], block_size,
            flux_direct_flat_prefix(flat, num_double),
        )
    return set^


def flux_direct_dora_projection_forward(
    set: FlatDirectDoRASet, slot: Int, x_h: List[Float32],
    w_orig: List[Float32], bias: List[Float32], M: Int,
) raises -> List[Float32]:
    var y = flat_direct_dora_forward_slot(set, slot, x_h, w_orig, M)
    return _add_bias(y^, bias, M, set.ad[slot].out_f)


def flux_direct_dora_projection_backward(
    set: FlatDirectDoRASet, slot: Int, d_y_h: List[Float32],
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> DoRAGrads:
    return flat_direct_dora_backward_slot(set, slot, d_y_h, x_h, w_orig, M)


def flux_direct_oft_projection_forward(
    set: FlatDirectOFTSet, slot: Int, x_h: List[Float32],
    w_orig: List[Float32], bias: List[Float32], M: Int,
) raises -> List[Float32]:
    var y = flat_direct_oft_forward_slot(set, slot, x_h, w_orig, M)
    return _add_bias(y^, bias, M, set.ad[slot].out_f)


def flux_direct_oft_projection_backward(
    set: FlatDirectOFTSet, slot: Int, d_y_h: List[Float32],
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> OFTOTGrads:
    return flat_direct_oft_backward_slot(set, slot, d_y_h, x_h, w_orig, M)


def flux_direct_dora_zero_grads(set: FlatDirectDoRASet) -> FlatDirectDoRAGrads:
    var out = List[DoRAGrads]()
    for i in range(len(set.ad)):
        ref d = set.ad[i]
        out.append(DoRAGrads(
            _zeros(len(d.a)), _zeros(len(d.b)), _zeros(len(d.m)), List[Float32](),
        ))
    return FlatDirectDoRAGrads(out^)


def flux_direct_dora_scatter_slot_grad(
    mut grads: FlatDirectDoRAGrads, slot: Int, g: DoRAGrads,
) raises:
    if slot < 0 or slot >= len(grads.g):
        raise Error("flux_direct_dora_scatter_slot_grad: slot out of range")
    grads.g[slot] = DoRAGrads(g.d_a.copy(), g.d_b.copy(), g.d_m.copy(), List[Float32]())


def flux_direct_dora_grad_norm(g: FlatDirectDoRAGrads) -> Float64:
    return flat_direct_dora_grad_norm(g)


def flux_direct_dora_clip_grads(mut g: FlatDirectDoRAGrads, clip_scale: Float32):
    flat_direct_dora_clip_grads(g, clip_scale)


def flux_direct_dora_adamw_step(
    mut set: FlatDirectDoRASet, g: FlatDirectDoRAGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_dora_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def flux_direct_dora_zero_leg_l1(set: FlatDirectDoRASet) -> Float64:
    return flat_direct_dora_zero_leg_l1(set)


def flux_direct_dora_trainable_bytes(set: FlatDirectDoRASet) -> Int:
    return flat_direct_dora_trainable_bytes(set)


def flux_direct_oft_zero_grads(set: FlatDirectOFTSet) -> FlatDirectOFTGrads:
    var out = List[List[Float32]]()
    for i in range(len(set.ad)):
        out.append(_zeros(len(set.ad[i].vec)))
    return FlatDirectOFTGrads(out^)


def flux_direct_oft_scatter_slot_grad(
    mut grads: FlatDirectOFTGrads, slot: Int, g: OFTOTGrads,
) raises:
    if slot < 0 or slot >= len(grads.d_vec):
        raise Error("flux_direct_oft_scatter_slot_grad: slot out of range")
    grads.d_vec[slot] = g.d_vec.copy()


def flux_direct_oft_grad_norm(g: FlatDirectOFTGrads) -> Float64:
    return flat_direct_oft_grad_norm(g)


def flux_direct_oft_clip_grads(mut g: FlatDirectOFTGrads, clip_scale: Float32):
    flat_direct_oft_clip_grads(g, clip_scale)


def flux_direct_oft_adamw_step(
    mut set: FlatDirectOFTSet, g: FlatDirectOFTGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_oft_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def flux_direct_oft_vec_l1(set: FlatDirectOFTSet) -> Float64:
    return flat_direct_oft_vec_l1(set)


def flux_direct_oft_trainable_bytes(set: FlatDirectOFTSet) -> Int:
    return flat_direct_oft_trainable_bytes(set)


def save_flux_direct_dora(
    set: FlatDirectDoRASet, path: String, ctx: DeviceContext,
) raises -> Int:
    var named = List[NamedDoRA]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedDoRA(set.prefix[i].copy(), set.ad[i].copy()))
    return save_dora_onetrainer(named, path, ctx)


def save_flux_direct_oft(
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
        raise Error("save_flux_direct_oft: refusing to write an empty OFT file")
    save_safetensors(names, tensors, path, ctx)
    return nmods
