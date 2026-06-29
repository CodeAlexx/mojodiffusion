# models/wan22/wan22_direct_lycoris_stack.mojo -- Wan2.2 direct DoRA/OFT slots.
#
# This is the Wan2.2-specific lowering point for the non-carrier DoRA/OFT path:
# the adapter sets own only trainables/moments, while each projection call
# receives the streamed block's W_orig and bias. Slot order mirrors
# wan22_stack_lora.mojo: self_attn q/k/v/o, then cross_attn q/k/v/o.

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


comptime WAN22_DIRECT_SLOTS = 8
comptime W_SA_Q = 0
comptime W_SA_K = 1
comptime W_SA_V = 2
comptime W_SA_O = 3
comptime W_CA_Q = 4
comptime W_CA_K = 5
comptime W_CA_V = 6
comptime W_CA_O = 7


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


def wan22_direct_slot_count(num_blocks: Int) -> Int:
    return num_blocks * WAN22_DIRECT_SLOTS


def empty_wan22_direct_dora_set() -> FlatDirectDoRASet:
    return empty_flat_direct_dora_set()


def empty_wan22_direct_oft_set() -> FlatDirectOFTSet:
    return empty_flat_direct_oft_set()


def wan22_direct_slot_prefix(block: Int, slot: Int) raises -> String:
    var b = String("blocks.") + String(block) + String(".")
    if slot == W_SA_Q:
        return b + String("self_attn.q")
    if slot == W_SA_K:
        return b + String("self_attn.k")
    if slot == W_SA_V:
        return b + String("self_attn.v")
    if slot == W_SA_O:
        return b + String("self_attn.o")
    if slot == W_CA_Q:
        return b + String("cross_attn.q")
    if slot == W_CA_K:
        return b + String("cross_attn.k")
    if slot == W_CA_V:
        return b + String("cross_attn.v")
    if slot == W_CA_O:
        return b + String("cross_attn.o")
    raise Error(String("wan22_direct_slot_prefix: bad slot ") + String(slot))


def wan22_direct_dense_carrier_bytes(num_blocks: Int, dim: Int) -> Int:
    # Full-delta DoRA/OFT carrier: per slot a=I[dim,dim] + b=delta[dim,dim],
    # BF16 storage. This is the path Wan2.2 must avoid for 24 GB.
    return wan22_direct_slot_count(num_blocks) * (dim * dim + dim * dim) * 2


def wan22_direct_dora_trainable_bytes_estimate(
    num_blocks: Int, dim: Int, rank: Int, wd_on_out: Bool = False,
) -> Int:
    var mlen = dim
    # FlatDirectDoRASet storage per square Wan projection:
    # BF16 params: A[rank,dim] + B[dim,rank].
    # F32 values/moments: m + ma/va + mb/vb + mm/vm.
    var bf16_elems = 2 * rank * dim
    var f32_elems = mlen + (2 * rank * dim) + (2 * dim * rank) + (2 * mlen)
    return wan22_direct_slot_count(num_blocks) * (bf16_elems * 2 + f32_elems * 4)


def wan22_direct_oft_trainable_bytes_estimate(
    num_blocks: Int, dim: Int, block_size: Int,
) raises -> Int:
    if dim % block_size != 0:
        raise Error("wan22_direct_oft_trainable_bytes_estimate: dim not divisible by block_size")
    var r = dim // block_size
    var ne = block_size * (block_size - 1) // 2
    # vec plus AdamW m/v, all F32.
    return wan22_direct_slot_count(num_blocks) * (3 * r * ne * 4)


def wan22_direct_dora_preflight(
    num_blocks: Int, dim: Int, rank: Int, budget_bytes: Int,
    wd_on_out: Bool = False,
) raises -> Int:
    var direct = wan22_direct_dora_trainable_bytes_estimate(num_blocks, dim, rank, wd_on_out)
    if direct > budget_bytes:
        raise Error(
            String("Wan22 direct DoRA trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


def wan22_direct_oft_preflight(
    num_blocks: Int, dim: Int, block_size: Int, budget_bytes: Int,
) raises -> Int:
    var direct = wan22_direct_oft_trainable_bytes_estimate(num_blocks, dim, block_size)
    if direct > budget_bytes:
        raise Error(
            String("Wan22 direct OFT trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


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


def build_wan22_direct_dora_set_from_weights(
    weights: List[List[Float32]], num_blocks: Int, dim: Int,
    rank: Int, alpha: Float32, seed: UInt64, wd_on_out: Bool = False,
) raises -> FlatDirectDoRASet:
    if len(weights) != wan22_direct_slot_count(num_blocks):
        raise Error("build_wan22_direct_dora_set_from_weights: weight count mismatch")
    var set = empty_flat_direct_dora_set()
    for bi in range(num_blocks):
        var block_weights = List[List[Float32]]()
        for slot in range(WAN22_DIRECT_SLOTS):
            block_weights.append(weights[bi * WAN22_DIRECT_SLOTS + slot].copy())
        wan22_direct_dora_append_block_weights(
            set, bi, block_weights^, dim, rank, alpha,
            seed + UInt64(bi * WAN22_DIRECT_SLOTS), wd_on_out,
        )
    return set^


def wan22_direct_dora_append_block_weights(
    mut set: FlatDirectDoRASet, block: Int, weights: List[List[Float32]],
    dim: Int, rank: Int, alpha: Float32, seed: UInt64,
    wd_on_out: Bool = False,
) raises:
    if len(weights) != WAN22_DIRECT_SLOTS:
        raise Error("wan22_direct_dora_append_block_weights: expected 8 block weights")
    var s = seed
    for slot in range(WAN22_DIRECT_SLOTS):
        if len(weights[slot]) != dim * dim:
            raise Error("wan22_direct_dora_append_block_weights: weight numel mismatch")
        flat_direct_dora_append_from_weight(
            set, weights[slot].copy(), dim, dim, rank, alpha,
            wan22_direct_slot_prefix(block, slot), s, wd_on_out,
        )
        s += 1


def build_wan22_direct_oft_set(
    num_blocks: Int, dim: Int, block_size: Int,
) raises -> FlatDirectOFTSet:
    var set = empty_flat_direct_oft_set()
    for bi in range(num_blocks):
        wan22_direct_oft_append_block(set, bi, dim, block_size)
    return set^


def wan22_direct_oft_append_block(
    mut set: FlatDirectOFTSet, block: Int, dim: Int, block_size: Int,
) raises:
    for slot in range(WAN22_DIRECT_SLOTS):
        flat_direct_oft_append(
            set, dim, dim, block_size, wan22_direct_slot_prefix(block, slot),
        )


def wan22_direct_dora_projection_forward(
    set: FlatDirectDoRASet, slot: Int, x_h: List[Float32],
    w_orig: List[Float32], bias: List[Float32], M: Int,
) raises -> List[Float32]:
    var y = flat_direct_dora_forward_slot(set, slot, x_h, w_orig, M)
    return _add_bias(y^, bias, M, set.ad[slot].out_f)


def wan22_direct_dora_projection_backward(
    set: FlatDirectDoRASet, slot: Int, d_y_h: List[Float32],
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> DoRAGrads:
    return flat_direct_dora_backward_slot(set, slot, d_y_h, x_h, w_orig, M)


def wan22_direct_oft_projection_forward(
    set: FlatDirectOFTSet, slot: Int, x_h: List[Float32],
    w_orig: List[Float32], bias: List[Float32], M: Int,
) raises -> List[Float32]:
    var y = flat_direct_oft_forward_slot(set, slot, x_h, w_orig, M)
    return _add_bias(y^, bias, M, set.ad[slot].out_f)


def wan22_direct_oft_projection_backward(
    set: FlatDirectOFTSet, slot: Int, d_y_h: List[Float32],
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> OFTOTGrads:
    return flat_direct_oft_backward_slot(set, slot, d_y_h, x_h, w_orig, M)


def wan22_direct_dora_zero_grads(set: FlatDirectDoRASet) -> FlatDirectDoRAGrads:
    var out = List[DoRAGrads]()
    for i in range(len(set.ad)):
        ref d = set.ad[i]
        out.append(DoRAGrads(
            _zeros(len(d.a)), _zeros(len(d.b)), _zeros(len(d.m)), List[Float32](),
        ))
    return FlatDirectDoRAGrads(out^)


def wan22_direct_dora_scatter_slot_grad(
    mut grads: FlatDirectDoRAGrads, slot: Int, g: DoRAGrads,
) raises:
    if slot < 0 or slot >= len(grads.g):
        raise Error("wan22_direct_dora_scatter_slot_grad: slot out of range")
    grads.g[slot] = DoRAGrads(g.d_a.copy(), g.d_b.copy(), g.d_m.copy(), List[Float32]())


def wan22_direct_dora_grad_norm(g: FlatDirectDoRAGrads) -> Float64:
    return flat_direct_dora_grad_norm(g)


def wan22_direct_dora_clip_grads(mut g: FlatDirectDoRAGrads, clip_scale: Float32):
    flat_direct_dora_clip_grads(g, clip_scale)


def wan22_direct_dora_adamw_step(
    mut set: FlatDirectDoRASet, g: FlatDirectDoRAGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_dora_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def wan22_direct_dora_zero_leg_l1(set: FlatDirectDoRASet) -> Float64:
    return flat_direct_dora_zero_leg_l1(set)


def wan22_direct_dora_trainable_bytes(set: FlatDirectDoRASet) -> Int:
    return flat_direct_dora_trainable_bytes(set)


def wan22_direct_oft_zero_grads(set: FlatDirectOFTSet) -> FlatDirectOFTGrads:
    var out = List[List[Float32]]()
    for i in range(len(set.ad)):
        out.append(_zeros(len(set.ad[i].vec)))
    return FlatDirectOFTGrads(out^)


def wan22_direct_oft_scatter_slot_grad(
    mut grads: FlatDirectOFTGrads, slot: Int, g: OFTOTGrads,
) raises:
    if slot < 0 or slot >= len(grads.d_vec):
        raise Error("wan22_direct_oft_scatter_slot_grad: slot out of range")
    grads.d_vec[slot] = g.d_vec.copy()


def wan22_direct_oft_grad_norm(g: FlatDirectOFTGrads) -> Float64:
    return flat_direct_oft_grad_norm(g)


def wan22_direct_oft_clip_grads(mut g: FlatDirectOFTGrads, clip_scale: Float32):
    flat_direct_oft_clip_grads(g, clip_scale)


def wan22_direct_oft_adamw_step(
    mut set: FlatDirectOFTSet, g: FlatDirectOFTGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_oft_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def wan22_direct_oft_vec_l1(set: FlatDirectOFTSet) -> Float64:
    return flat_direct_oft_vec_l1(set)


def wan22_direct_oft_trainable_bytes(set: FlatDirectOFTSet) -> Int:
    return flat_direct_oft_trainable_bytes(set)


def save_wan22_direct_dora(
    set: FlatDirectDoRASet, path: String, ctx: DeviceContext,
) raises -> Int:
    var named = List[NamedDoRA]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedDoRA(set.prefix[i].copy(), set.ad[i].copy()))
    return save_dora_onetrainer(named, path, ctx)


def save_wan22_direct_oft(
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
        raise Error("save_wan22_direct_oft: refusing to write an empty OFT file")
    save_safetensors(names, tensors, path, ctx)
    return nmods
