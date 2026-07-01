# models/krea2/krea2_direct_lycoris_stack.mojo -- Krea2 direct DoRA/OFT slots.
#
# Krea2's live block path is device-resident plain LoRA today. This module owns
# the model-specific direct DoRA/OFT slot map, byte preflight, save names, and
# host-side projection wrappers needed before the GPU block lowering is wired.

from std.collections import List, Optional
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.dora_save import NamedDoRA, save_dora_onetrainer
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, reshape_owned

from serenitymojo.training.dora_adapter import DoRAGrads
from serenitymojo.training.dora_substitution_device import (
    DoRAAdapterDevice, DoRADeviceGrads, dora_device_from_host,
    dora_substitution_forward_device, dora_substitution_backward_device,
)
from serenitymojo.training.oft_onetrainer import OFTOTGrads
from serenitymojo.training.oft_onetrainer_device import (
    OFTOTDeviceGrads, oft_ot_rotate_b4, oft_ot_rotate_backward_b4,
)
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
from serenitymojo.models.krea2.krea2_lokr_stack import (
    KREA2_SLOTS, K2LOKR_TGT_ATTN, K2LOKR_TGT_ALL,
    krea2_lokr_slot_dims, _krea2_slot_targeted,
)


comptime KREA2_DIRECT_24_GIB = 24 * 1024 * 1024 * 1024
comptime K2D_WQ = 0
comptime K2D_WK = 1
comptime K2D_WV = 2
comptime K2D_GATE = 3
comptime K2D_WO = 4
comptime K2D_MLP_GATE = 5
comptime K2D_MLP_UP = 6
comptime K2D_MLP_DOWN = 7
comptime TArc = ArcPointer[Tensor]


struct Krea2DirectDoRADeviceSlots(Movable):
    var slots: List[DoRAAdapterDevice]

    def __init__(out self, var slots: List[DoRAAdapterDevice]):
        self.slots = slots^


struct Krea2DirectOFTDeviceSlot(Copyable, Movable):
    var vec: TArc
    var in_f: Int
    var out_f: Int
    var b: Int
    var r: Int

    def __init__(
        out self, var vec: TArc, in_f: Int, out_f: Int, b: Int, r: Int,
    ):
        self.vec = vec^
        self.in_f = in_f
        self.out_f = out_f
        self.b = b
        self.r = r


struct Krea2DirectOFTDeviceSlots(Movable):
    var slots: List[Krea2DirectOFTDeviceSlot]

    def __init__(out self, var slots: List[Krea2DirectOFTDeviceSlot]):
        self.slots = slots^


struct Krea2BlockDirectDoRA(Copyable, Movable):
    var wq: Optional[DoRAAdapterDevice]
    var wk: Optional[DoRAAdapterDevice]
    var wv: Optional[DoRAAdapterDevice]
    var gate_w: Optional[DoRAAdapterDevice]
    var wo: Optional[DoRAAdapterDevice]
    var mlp_gate_w: Optional[DoRAAdapterDevice]
    var mlp_up_w: Optional[DoRAAdapterDevice]
    var mlp_down_w: Optional[DoRAAdapterDevice]

    def __init__(
        out self,
        var wq: Optional[DoRAAdapterDevice], var wk: Optional[DoRAAdapterDevice],
        var wv: Optional[DoRAAdapterDevice], var gate_w: Optional[DoRAAdapterDevice],
        var wo: Optional[DoRAAdapterDevice],
        var mlp_gate_w: Optional[DoRAAdapterDevice],
        var mlp_up_w: Optional[DoRAAdapterDevice],
        var mlp_down_w: Optional[DoRAAdapterDevice],
    ):
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^


struct Krea2StackDirectDoRA(Movable):
    var blocks: List[Krea2BlockDirectDoRA]

    def __init__(out self, var blocks: List[Krea2BlockDirectDoRA]):
        self.blocks = blocks^


struct Krea2BlockDirectOFT(Copyable, Movable):
    var wq: Optional[Krea2DirectOFTDeviceSlot]
    var wk: Optional[Krea2DirectOFTDeviceSlot]
    var wv: Optional[Krea2DirectOFTDeviceSlot]
    var gate_w: Optional[Krea2DirectOFTDeviceSlot]
    var wo: Optional[Krea2DirectOFTDeviceSlot]
    var mlp_gate_w: Optional[Krea2DirectOFTDeviceSlot]
    var mlp_up_w: Optional[Krea2DirectOFTDeviceSlot]
    var mlp_down_w: Optional[Krea2DirectOFTDeviceSlot]

    def __init__(
        out self,
        var wq: Optional[Krea2DirectOFTDeviceSlot],
        var wk: Optional[Krea2DirectOFTDeviceSlot],
        var wv: Optional[Krea2DirectOFTDeviceSlot],
        var gate_w: Optional[Krea2DirectOFTDeviceSlot],
        var wo: Optional[Krea2DirectOFTDeviceSlot],
        var mlp_gate_w: Optional[Krea2DirectOFTDeviceSlot],
        var mlp_up_w: Optional[Krea2DirectOFTDeviceSlot],
        var mlp_down_w: Optional[Krea2DirectOFTDeviceSlot],
    ):
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^


struct Krea2StackDirectOFT(Movable):
    var blocks: List[Krea2BlockDirectOFT]

    def __init__(out self, var blocks: List[Krea2BlockDirectOFT]):
        self.blocks = blocks^


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


def _bf16_2d(var values: List[Float32], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host(values^, sh^, STDtype.BF16, ctx)


def _validate_targets(targets: Int) raises:
    if targets < K2LOKR_TGT_ATTN or targets > K2LOKR_TGT_ALL:
        raise Error("krea2 direct LyCORIS: targets must be 1(attn)|2(all)")


def krea2_direct_slot_prefix(block: Int, slot: Int) raises -> String:
    var b = String("diffusion_model.blocks.") + String(block)
    if slot == K2D_WQ:
        return b + String(".attn.wq")
    if slot == K2D_WK:
        return b + String(".attn.wk")
    if slot == K2D_WV:
        return b + String(".attn.wv")
    if slot == K2D_GATE:
        return b + String(".attn.gate")
    if slot == K2D_WO:
        return b + String(".attn.wo")
    if slot == K2D_MLP_GATE:
        return b + String(".mlp.gate")
    if slot == K2D_MLP_UP:
        return b + String(".mlp.up")
    if slot == K2D_MLP_DOWN:
        return b + String(".mlp.down")
    raise Error(String("krea2_direct_slot_prefix: bad slot ") + String(slot))


def krea2_direct_active_slot_count(num_blocks: Int, targets: Int) raises -> Int:
    _validate_targets(targets)
    var n = 0
    for _bi in range(num_blocks):
        for slot in range(KREA2_SLOTS):
            if _krea2_slot_targeted(slot, targets):
                n += 1
    return n


def empty_krea2_direct_dora_set() -> FlatDirectDoRASet:
    return empty_flat_direct_dora_set()


def empty_krea2_direct_oft_set() -> FlatDirectOFTSet:
    return empty_flat_direct_oft_set()


def krea2_direct_dense_carrier_bytes(
    num_blocks: Int, D: Int, F: Int, qdim: Int, kvdim: Int, targets: Int,
) raises -> Int:
    _validate_targets(targets)
    var elems = 0
    for _bi in range(num_blocks):
        for slot in range(KREA2_SLOTS):
            if not _krea2_slot_targeted(slot, targets):
                continue
            var dims = krea2_lokr_slot_dims(slot, D, F, qdim, kvdim)
            var in_f = dims[0]
            var out_f = dims[1]
            # Full-delta carrier: a=I_in [in,in] and b=W_eff-W [out,in].
            elems += in_f * in_f + out_f * in_f
    return elems * 2


def krea2_direct_dora_trainable_bytes_estimate(
    num_blocks: Int, D: Int, F: Int, qdim: Int, kvdim: Int,
    rank: Int, targets: Int, wd_on_out: Bool = False,
) raises -> Int:
    _validate_targets(targets)
    var total = 0
    for _bi in range(num_blocks):
        for slot in range(KREA2_SLOTS):
            if not _krea2_slot_targeted(slot, targets):
                continue
            var dims = krea2_lokr_slot_dims(slot, D, F, qdim, kvdim)
            var in_f = dims[0]
            var out_f = dims[1]
            var mlen = out_f if wd_on_out else in_f
            var bf16_elems = rank * in_f + out_f * rank
            var f32_elems = mlen + (2 * rank * in_f) + (2 * out_f * rank) + (2 * mlen)
            total += bf16_elems * 2 + f32_elems * 4
    return total


def krea2_direct_oft_trainable_bytes_estimate(
    num_blocks: Int, D: Int, F: Int, qdim: Int, kvdim: Int,
    block_size: Int, targets: Int,
) raises -> Int:
    _validate_targets(targets)
    var total = 0
    for _bi in range(num_blocks):
        for slot in range(KREA2_SLOTS):
            if not _krea2_slot_targeted(slot, targets):
                continue
            var dims = krea2_lokr_slot_dims(slot, D, F, qdim, kvdim)
            var in_f = dims[0]
            if in_f % block_size != 0:
                raise Error("krea2_direct_oft_trainable_bytes_estimate: in_f not divisible by block_size")
            var r = in_f // block_size
            var ne = block_size * (block_size - 1) // 2
            total += 3 * r * ne * 4
    return total


def krea2_direct_dora_preflight(
    num_blocks: Int, D: Int, F: Int, qdim: Int, kvdim: Int,
    rank: Int, targets: Int, budget_bytes: Int, wd_on_out: Bool = False,
) raises -> Int:
    var direct = krea2_direct_dora_trainable_bytes_estimate(
        num_blocks, D, F, qdim, kvdim, rank, targets, wd_on_out,
    )
    if direct > budget_bytes:
        raise Error(
            String("Krea2 direct DoRA trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


def krea2_direct_oft_preflight(
    num_blocks: Int, D: Int, F: Int, qdim: Int, kvdim: Int,
    block_size: Int, targets: Int, budget_bytes: Int,
) raises -> Int:
    var direct = krea2_direct_oft_trainable_bytes_estimate(
        num_blocks, D, F, qdim, kvdim, block_size, targets,
    )
    if direct > budget_bytes:
        raise Error(
            String("Krea2 direct OFT trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


def krea2_direct_runtime_blocker(algo_name: String) -> String:
    return (
        String("krea2 trainer: network_algorithm=") + algo_name
        + String(" passed direct-state 24 GiB preflight, but production ")
        + String("Krea2 DoRA/OFT needs GPU direct W_eff lowering. ")
        + String("Do not route through host flat_direct_* substitution: ")
        + String("real Krea2 projection dims make host matmul infeasible ")
        + String("and would not satisfy the 24 GiB runtime claim. Missing: ")
        + String("thread the gated resident DoRA/OFT projection slots into ")
        + String("the live block/stack path, scatter device grads into the ")
        + String("master sets, and run a peak-VRAM runtime gate.")
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


def build_krea2_direct_dora_set_from_weights(
    weights: List[List[Float32]], num_blocks: Int,
    D: Int, F: Int, qdim: Int, kvdim: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool = False,
) raises -> FlatDirectDoRASet:
    if len(weights) != num_blocks * KREA2_SLOTS:
        raise Error("build_krea2_direct_dora_set_from_weights: weight count mismatch")
    var set = empty_flat_direct_dora_set()
    for bi in range(num_blocks):
        var block_weights = List[List[Float32]]()
        for slot in range(KREA2_SLOTS):
            block_weights.append(weights[bi * KREA2_SLOTS + slot].copy())
        krea2_direct_dora_append_block_weights(
            set, bi, block_weights^, D, F, qdim, kvdim, rank, alpha,
            targets, seed + UInt64(bi * KREA2_SLOTS), wd_on_out,
        )
    return set^


def krea2_direct_dora_append_block_weights(
    mut set: FlatDirectDoRASet, block: Int, weights: List[List[Float32]],
    D: Int, F: Int, qdim: Int, kvdim: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool = False,
) raises:
    _validate_targets(targets)
    if len(weights) != KREA2_SLOTS:
        raise Error("krea2_direct_dora_append_block_weights: expected 8 block weights")
    var s = seed
    for slot in range(KREA2_SLOTS):
        var dims = krea2_lokr_slot_dims(slot, D, F, qdim, kvdim)
        if _krea2_slot_targeted(slot, targets):
            if len(weights[slot]) != dims[0] * dims[1]:
                raise Error("krea2_direct_dora_append_block_weights: weight numel mismatch")
            flat_direct_dora_append_from_weight(
                set, weights[slot].copy(), dims[0], dims[1], rank, alpha,
                krea2_direct_slot_prefix(block, slot), s, wd_on_out,
            )
        s += 1


def build_krea2_direct_oft_set(
    num_blocks: Int, D: Int, F: Int, qdim: Int, kvdim: Int,
    block_size: Int, targets: Int,
) raises -> FlatDirectOFTSet:
    var set = empty_flat_direct_oft_set()
    for bi in range(num_blocks):
        krea2_direct_oft_append_block(set, bi, D, F, qdim, kvdim, block_size, targets)
    return set^


def krea2_direct_oft_append_block(
    mut set: FlatDirectOFTSet, block: Int,
    D: Int, F: Int, qdim: Int, kvdim: Int,
    block_size: Int, targets: Int,
) raises:
    _validate_targets(targets)
    for slot in range(KREA2_SLOTS):
        if not _krea2_slot_targeted(slot, targets):
            continue
        var dims = krea2_lokr_slot_dims(slot, D, F, qdim, kvdim)
        flat_direct_oft_append(
            set, dims[0], dims[1], block_size,
            krea2_direct_slot_prefix(block, slot),
        )


def krea2_direct_dora_projection_forward(
    set: FlatDirectDoRASet, slot: Int, x_h: List[Float32],
    w_orig: List[Float32], bias: List[Float32], M: Int,
) raises -> List[Float32]:
    var y = flat_direct_dora_forward_slot(set, slot, x_h, w_orig, M)
    return _add_bias(y^, bias, M, set.ad[slot].out_f)


def krea2_direct_dora_projection_backward(
    set: FlatDirectDoRASet, slot: Int, d_y_h: List[Float32],
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> DoRAGrads:
    return flat_direct_dora_backward_slot(set, slot, d_y_h, x_h, w_orig, M)


def _krea2_check_dora_projection_tensors(
    set: FlatDirectDoRASet, slot: Int, x: Tensor, w_orig: Tensor, M: Int,
) raises:
    if slot < 0 or slot >= len(set.ad):
        raise Error("krea2_direct_dora_device: slot out of range")
    if not set.active[slot]:
        raise Error("krea2_direct_dora_device: inactive slot")
    ref sl = set.ad[slot]
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("krea2_direct_dora_device: x rank must be >= 1")
    if xshape[len(xshape) - 1] != sl.in_f:
        raise Error("krea2_direct_dora_device: x trailing dim does not match slot")
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    if rows != M:
        raise Error("krea2_direct_dora_device: M does not match x leading rows")
    var wshape = w_orig.shape()
    if len(wshape) != 2 or wshape[0] != sl.out_f or wshape[1] != sl.in_f:
        raise Error("krea2_direct_dora_device: w_orig shape does not match slot")


def krea2_direct_dora_projection_slot_to_device(
    set: FlatDirectDoRASet, slot: Int, ctx: DeviceContext,
) raises -> DoRAAdapterDevice:
    if slot < 0 or slot >= len(set.ad):
        raise Error("krea2_direct_dora_projection_slot_to_device: slot out of range")
    if not set.active[slot]:
        raise Error("krea2_direct_dora_projection_slot_to_device: inactive slot")
    return dora_device_from_host(set.ad[slot], ctx)


def krea2_direct_dora_set_to_device(
    set: FlatDirectDoRASet, ctx: DeviceContext,
) raises -> Krea2DirectDoRADeviceSlots:
    var slots = List[DoRAAdapterDevice]()
    for i in range(len(set.ad)):
        if not set.active[i]:
            raise Error("krea2_direct_dora_set_to_device: inactive compact slot")
        slots.append(dora_device_from_host(set.ad[i], ctx))
    return Krea2DirectDoRADeviceSlots(slots^)


def _krea2_check_dora_projection_resident(
    slot_dev: DoRAAdapterDevice, x: Tensor, w_orig: Tensor, M: Int,
) raises:
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("krea2_direct_dora_resident: x rank must be >= 1")
    if xshape[len(xshape) - 1] != slot_dev.in_f:
        raise Error("krea2_direct_dora_resident: x trailing dim does not match slot")
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    if rows != M:
        raise Error("krea2_direct_dora_resident: M does not match x leading rows")
    var wshape = w_orig.shape()
    if len(wshape) != 2 or wshape[0] != slot_dev.out_f or wshape[1] != slot_dev.in_f:
        raise Error("krea2_direct_dora_resident: w_orig shape does not match slot")


def krea2_direct_dora_projection_forward_resident(
    slot_dev: DoRAAdapterDevice, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
    bias: Optional[Tensor] = Optional[Tensor](None),
) raises -> Tensor:
    """Device Krea2 DoRA projection with pre-uploaded resident A/B/m."""
    _krea2_check_dora_projection_resident(slot_dev, x, w_orig, M)
    var y = dora_substitution_forward_device(x, w_orig, slot_dev, ctx)
    if bias:
        if bias.value().dtype() != y.dtype():
            var bc = cast_tensor(bias.value(), y.dtype(), ctx)
            return add(y, bc, ctx)
        return add(y, bias.value(), ctx)
    return y^


def krea2_direct_dora_projection_backward_resident(
    slot_dev: DoRAAdapterDevice, d_y: Tensor, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
) raises -> DoRADeviceGrads:
    """Device Krea2 DoRA backward with pre-uploaded resident A/B/m."""
    _krea2_check_dora_projection_resident(slot_dev, x, w_orig, M)
    return dora_substitution_backward_device(d_y, x, w_orig, slot_dev, ctx)


def krea2_direct_dora_projection_forward_device(
    set: FlatDirectDoRASet, slot: Int, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
    bias: Optional[Tensor] = Optional[Tensor](None),
) raises -> Tensor:
    """Device Krea2 DoRA projection: direct W_eff substitution on GPU."""
    _krea2_check_dora_projection_tensors(set, slot, x, w_orig, M)
    var dev = krea2_direct_dora_projection_slot_to_device(set, slot, ctx)
    return krea2_direct_dora_projection_forward_resident(dev, x, w_orig, M, ctx, bias)


def krea2_direct_dora_projection_backward_device(
    set: FlatDirectDoRASet, slot: Int, d_y: Tensor, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
) raises -> DoRADeviceGrads:
    """Device Krea2 DoRA projection backward. Base weight is frozen."""
    _krea2_check_dora_projection_tensors(set, slot, x, w_orig, M)
    var dev = krea2_direct_dora_projection_slot_to_device(set, slot, ctx)
    return krea2_direct_dora_projection_backward_resident(dev, d_y, x, w_orig, M, ctx)


def krea2_direct_oft_projection_forward(
    set: FlatDirectOFTSet, slot: Int, x_h: List[Float32],
    w_orig: List[Float32], bias: List[Float32], M: Int,
) raises -> List[Float32]:
    var y = flat_direct_oft_forward_slot(set, slot, x_h, w_orig, M)
    return _add_bias(y^, bias, M, set.ad[slot].out_f)


def krea2_direct_oft_projection_backward(
    set: FlatDirectOFTSet, slot: Int, d_y_h: List[Float32],
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> OFTOTGrads:
    return flat_direct_oft_backward_slot(set, slot, d_y_h, x_h, w_orig, M)


def _krea2_oft_vec_tensor(set: FlatDirectOFTSet, slot: Int, ctx: DeviceContext) raises -> Tensor:
    if slot < 0 or slot >= len(set.ad):
        raise Error("krea2_direct_oft_device: slot out of range")
    if not set.active[slot]:
        raise Error("krea2_direct_oft_device: inactive slot")
    ref sl = set.ad[slot]
    if sl.b != 4:
        raise Error("krea2_direct_oft_device: only block_size=4 is wired on GPU")
    return _bf16_2d(sl.vec.copy(), sl.r, 6, ctx)


def _krea2_check_oft_projection_tensors(
    set: FlatDirectOFTSet, slot: Int, x: Tensor, w_orig: Tensor, M: Int,
) raises:
    if slot < 0 or slot >= len(set.ad):
        raise Error("krea2_direct_oft_device: slot out of range")
    if not set.active[slot]:
        raise Error("krea2_direct_oft_device: inactive slot")
    ref sl = set.ad[slot]
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("krea2_direct_oft_device: x rank must be >= 1")
    if xshape[len(xshape) - 1] != sl.in_f:
        raise Error("krea2_direct_oft_device: x trailing dim does not match slot")
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    if rows != M:
        raise Error("krea2_direct_oft_device: M does not match x leading rows")
    var wshape = w_orig.shape()
    if len(wshape) != 2 or wshape[0] != sl.out_f or wshape[1] != sl.in_f:
        raise Error("krea2_direct_oft_device: w_orig shape does not match slot")


def krea2_direct_oft_projection_slot_to_device(
    set: FlatDirectOFTSet, slot: Int, ctx: DeviceContext,
) raises -> Krea2DirectOFTDeviceSlot:
    if slot < 0 or slot >= len(set.ad):
        raise Error("krea2_direct_oft_projection_slot_to_device: slot out of range")
    if not set.active[slot]:
        raise Error("krea2_direct_oft_projection_slot_to_device: inactive slot")
    ref sl = set.ad[slot]
    var vec = _krea2_oft_vec_tensor(set, slot, ctx)
    return Krea2DirectOFTDeviceSlot(TArc(vec^), sl.in_f, sl.out_f, sl.b, sl.r)


def krea2_direct_oft_set_to_device(
    set: FlatDirectOFTSet, ctx: DeviceContext,
) raises -> Krea2DirectOFTDeviceSlots:
    var slots = List[Krea2DirectOFTDeviceSlot]()
    for i in range(len(set.ad)):
        if not set.active[i]:
            raise Error("krea2_direct_oft_set_to_device: inactive compact slot")
        slots.append(krea2_direct_oft_projection_slot_to_device(set, i, ctx))
    return Krea2DirectOFTDeviceSlots(slots^)


def krea2_direct_dora_blocks_to_device(
    set: FlatDirectDoRASet, num_blocks: Int, targets: Int, ctx: DeviceContext,
) raises -> Krea2StackDirectDoRA:
    _validate_targets(targets)
    var compact = 0
    var blocks = List[Krea2BlockDirectDoRA]()
    for _bi in range(num_blocks):
        var wq = Optional[DoRAAdapterDevice](None)
        var wk = Optional[DoRAAdapterDevice](None)
        var wv = Optional[DoRAAdapterDevice](None)
        var gate_w = Optional[DoRAAdapterDevice](None)
        var wo = Optional[DoRAAdapterDevice](None)
        var mlp_gate_w = Optional[DoRAAdapterDevice](None)
        var mlp_up_w = Optional[DoRAAdapterDevice](None)
        var mlp_down_w = Optional[DoRAAdapterDevice](None)
        for slot in range(KREA2_SLOTS):
            if not _krea2_slot_targeted(slot, targets):
                continue
            if compact >= len(set.ad):
                raise Error("krea2_direct_dora_blocks_to_device: compact set too short")
            if not set.active[compact]:
                raise Error("krea2_direct_dora_blocks_to_device: inactive compact slot")
            var dev = dora_device_from_host(set.ad[compact], ctx)
            if slot == K2D_WQ:
                wq = Optional[DoRAAdapterDevice](dev^)
            elif slot == K2D_WK:
                wk = Optional[DoRAAdapterDevice](dev^)
            elif slot == K2D_WV:
                wv = Optional[DoRAAdapterDevice](dev^)
            elif slot == K2D_GATE:
                gate_w = Optional[DoRAAdapterDevice](dev^)
            elif slot == K2D_WO:
                wo = Optional[DoRAAdapterDevice](dev^)
            elif slot == K2D_MLP_GATE:
                mlp_gate_w = Optional[DoRAAdapterDevice](dev^)
            elif slot == K2D_MLP_UP:
                mlp_up_w = Optional[DoRAAdapterDevice](dev^)
            elif slot == K2D_MLP_DOWN:
                mlp_down_w = Optional[DoRAAdapterDevice](dev^)
            compact += 1
        blocks.append(Krea2BlockDirectDoRA(
            wq^, wk^, wv^, gate_w^, wo^, mlp_gate_w^, mlp_up_w^, mlp_down_w^,
        ))
    if compact != len(set.ad):
        raise Error("krea2_direct_dora_blocks_to_device: compact set has trailing slots")
    return Krea2StackDirectDoRA(blocks^)


def krea2_direct_oft_blocks_to_device(
    set: FlatDirectOFTSet, num_blocks: Int, targets: Int, ctx: DeviceContext,
) raises -> Krea2StackDirectOFT:
    _validate_targets(targets)
    var compact = 0
    var blocks = List[Krea2BlockDirectOFT]()
    for _bi in range(num_blocks):
        var wq = Optional[Krea2DirectOFTDeviceSlot](None)
        var wk = Optional[Krea2DirectOFTDeviceSlot](None)
        var wv = Optional[Krea2DirectOFTDeviceSlot](None)
        var gate_w = Optional[Krea2DirectOFTDeviceSlot](None)
        var wo = Optional[Krea2DirectOFTDeviceSlot](None)
        var mlp_gate_w = Optional[Krea2DirectOFTDeviceSlot](None)
        var mlp_up_w = Optional[Krea2DirectOFTDeviceSlot](None)
        var mlp_down_w = Optional[Krea2DirectOFTDeviceSlot](None)
        for slot in range(KREA2_SLOTS):
            if not _krea2_slot_targeted(slot, targets):
                continue
            if compact >= len(set.ad):
                raise Error("krea2_direct_oft_blocks_to_device: compact set too short")
            var dev = krea2_direct_oft_projection_slot_to_device(set, compact, ctx)
            if slot == K2D_WQ:
                wq = Optional[Krea2DirectOFTDeviceSlot](dev^)
            elif slot == K2D_WK:
                wk = Optional[Krea2DirectOFTDeviceSlot](dev^)
            elif slot == K2D_WV:
                wv = Optional[Krea2DirectOFTDeviceSlot](dev^)
            elif slot == K2D_GATE:
                gate_w = Optional[Krea2DirectOFTDeviceSlot](dev^)
            elif slot == K2D_WO:
                wo = Optional[Krea2DirectOFTDeviceSlot](dev^)
            elif slot == K2D_MLP_GATE:
                mlp_gate_w = Optional[Krea2DirectOFTDeviceSlot](dev^)
            elif slot == K2D_MLP_UP:
                mlp_up_w = Optional[Krea2DirectOFTDeviceSlot](dev^)
            elif slot == K2D_MLP_DOWN:
                mlp_down_w = Optional[Krea2DirectOFTDeviceSlot](dev^)
            compact += 1
        blocks.append(Krea2BlockDirectOFT(
            wq^, wk^, wv^, gate_w^, wo^, mlp_gate_w^, mlp_up_w^, mlp_down_w^,
        ))
    if compact != len(set.ad):
        raise Error("krea2_direct_oft_blocks_to_device: compact set has trailing slots")
    return Krea2StackDirectOFT(blocks^)


def _krea2_check_oft_projection_resident(
    slot_dev: Krea2DirectOFTDeviceSlot, x: Tensor, w_orig: Tensor, M: Int,
) raises:
    if slot_dev.b != 4:
        raise Error("krea2_direct_oft_resident: only block_size=4 is wired on GPU")
    if slot_dev.vec[].dtype() != STDtype.BF16:
        raise Error("krea2_direct_oft_resident: vec storage must be BF16")
    if slot_dev.vec[].shape() != [slot_dev.r, 6]:
        raise Error("krea2_direct_oft_resident: vec shape mismatch")
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("krea2_direct_oft_resident: x rank must be >= 1")
    if xshape[len(xshape) - 1] != slot_dev.in_f:
        raise Error("krea2_direct_oft_resident: x trailing dim does not match slot")
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    if rows != M:
        raise Error("krea2_direct_oft_resident: M does not match x leading rows")
    var wshape = w_orig.shape()
    if len(wshape) != 2 or wshape[0] != slot_dev.out_f or wshape[1] != slot_dev.in_f:
        raise Error("krea2_direct_oft_resident: w_orig shape does not match slot")


def krea2_direct_oft_projection_forward_resident(
    slot_dev: Krea2DirectOFTDeviceSlot, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
    bias: Optional[Tensor] = Optional[Tensor](None),
) raises -> Tensor:
    """Device Krea2 OFT projection with pre-uploaded resident vec."""
    _krea2_check_oft_projection_resident(slot_dev, x, w_orig, M)
    var x_rot = oft_ot_rotate_b4(x, slot_dev.vec[], ctx)
    return linear(x_rot, w_orig, bias, ctx)


def krea2_direct_oft_projection_backward_resident(
    slot_dev: Krea2DirectOFTDeviceSlot, d_y: Tensor, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
) raises -> OFTOTDeviceGrads:
    """Device Krea2 OFT backward with pre-uploaded resident vec."""
    _krea2_check_oft_projection_resident(slot_dev, x, w_orig, M)
    var d_x_rot = linear_backward_dx(d_y, w_orig, M, slot_dev.in_f, slot_dev.out_f, ctx)
    if d_x_rot.dtype() != x.dtype():
        d_x_rot = cast_tensor(d_x_rot^, x.dtype(), ctx)
    if d_x_rot.shape() != x.shape():
        d_x_rot = reshape_owned(d_x_rot^, x.shape())
    return oft_ot_rotate_backward_b4(d_x_rot^, x, slot_dev.vec[], ctx)


def krea2_direct_oft_projection_forward_device(
    set: FlatDirectOFTSet, slot: Int, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
    bias: Optional[Tensor] = Optional[Tensor](None),
) raises -> Tensor:
    """Device Krea2 OFT projection: rotate input on GPU, then frozen linear."""
    _krea2_check_oft_projection_tensors(set, slot, x, w_orig, M)
    var dev = krea2_direct_oft_projection_slot_to_device(set, slot, ctx)
    return krea2_direct_oft_projection_forward_resident(dev, x, w_orig, M, ctx, bias)


def krea2_direct_oft_projection_backward_device(
    set: FlatDirectOFTSet, slot: Int, d_y: Tensor, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
) raises -> OFTOTDeviceGrads:
    """Device Krea2 OFT projection backward. Base weight is frozen."""
    _krea2_check_oft_projection_tensors(set, slot, x, w_orig, M)
    var dev = krea2_direct_oft_projection_slot_to_device(set, slot, ctx)
    return krea2_direct_oft_projection_backward_resident(dev, d_y, x, w_orig, M, ctx)


def krea2_direct_dora_zero_grads(set: FlatDirectDoRASet) -> FlatDirectDoRAGrads:
    var out = List[DoRAGrads]()
    for i in range(len(set.ad)):
        ref d = set.ad[i]
        out.append(DoRAGrads(
            _zeros(len(d.a)), _zeros(len(d.b)), _zeros(len(d.m)), List[Float32](),
        ))
    return FlatDirectDoRAGrads(out^)


def krea2_direct_dora_scatter_slot_grad(
    mut grads: FlatDirectDoRAGrads, slot: Int, g: DoRAGrads,
) raises:
    if slot < 0 or slot >= len(grads.g):
        raise Error("krea2_direct_dora_scatter_slot_grad: slot out of range")
    grads.g[slot] = DoRAGrads(g.d_a.copy(), g.d_b.copy(), g.d_m.copy(), List[Float32]())


def krea2_direct_dora_grad_norm(g: FlatDirectDoRAGrads) -> Float64:
    return flat_direct_dora_grad_norm(g)


def krea2_direct_dora_clip_grads(mut g: FlatDirectDoRAGrads, clip_scale: Float32):
    flat_direct_dora_clip_grads(g, clip_scale)


def krea2_direct_dora_adamw_step(
    mut set: FlatDirectDoRASet, g: FlatDirectDoRAGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_dora_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def krea2_direct_dora_zero_leg_l1(set: FlatDirectDoRASet) -> Float64:
    return flat_direct_dora_zero_leg_l1(set)


def krea2_direct_dora_trainable_bytes(set: FlatDirectDoRASet) -> Int:
    return flat_direct_dora_trainable_bytes(set)


def krea2_direct_oft_zero_grads(set: FlatDirectOFTSet) -> FlatDirectOFTGrads:
    var out = List[List[Float32]]()
    for i in range(len(set.ad)):
        out.append(_zeros(len(set.ad[i].vec)))
    return FlatDirectOFTGrads(out^)


def krea2_direct_oft_scatter_slot_grad(
    mut grads: FlatDirectOFTGrads, slot: Int, g: OFTOTGrads,
) raises:
    if slot < 0 or slot >= len(grads.d_vec):
        raise Error("krea2_direct_oft_scatter_slot_grad: slot out of range")
    grads.d_vec[slot] = g.d_vec.copy()


def krea2_direct_oft_grad_norm(g: FlatDirectOFTGrads) -> Float64:
    return flat_direct_oft_grad_norm(g)


def krea2_direct_oft_clip_grads(mut g: FlatDirectOFTGrads, clip_scale: Float32):
    flat_direct_oft_clip_grads(g, clip_scale)


def krea2_direct_oft_adamw_step(
    mut set: FlatDirectOFTSet, g: FlatDirectOFTGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_oft_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def krea2_direct_oft_vec_l1(set: FlatDirectOFTSet) -> Float64:
    return flat_direct_oft_vec_l1(set)


def krea2_direct_oft_trainable_bytes(set: FlatDirectOFTSet) -> Int:
    return flat_direct_oft_trainable_bytes(set)


def save_krea2_direct_dora(
    set: FlatDirectDoRASet, path: String, ctx: DeviceContext,
) raises -> Int:
    var named = List[NamedDoRA]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedDoRA(set.prefix[i].copy(), set.ad[i].copy()))
    return save_dora_onetrainer(named, path, ctx)


def save_krea2_direct_oft(
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
        tensors.append(ArcPointer(_bf16_2d(sl.vec.copy(), sl.r, ne, ctx)))
        nmods += 1
    if nmods == 0:
        raise Error("save_krea2_direct_oft: refusing to write an empty OFT file")
    save_safetensors(names, tensors, path, ctx)
    return nmods
