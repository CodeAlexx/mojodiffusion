# models/klein/klein_direct_lycoris_stack.mojo -- Klein direct DoRA/OFT slots.
#
# DoRA/OFT cannot use the dense full-delta carrier for broader Klein target
# sets: the carrier materializes a=I_in and b=(W_eff-W) for every projection.
# This module mirrors the established Flux/Krea2/Qwen direct path: it owns the
# Klein slot map, byte preflight, checkpoint-backed direct adapter construction,
# GPU projection helpers, optimizer wrappers, and save names. The frozen W_orig
# remains supplied by the live Klein block at each projection call site.

from std.collections import List, Optional
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.tensor_algebra import add
from serenitymojo.training.dora_save import NamedDoRA, save_dora_onetrainer
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
from serenitymojo.training.lokr_stack import (
    LOKR_TGT_ATTN, LOKR_TGT_ATTN_FF, LOKR_TGT_ALL,
    _DBL_SLOTS, _SGL_SLOTS, klein_lokr_slot_dims, _slot_targeted,
    klein_lokr_prefix, klein_lokr_base_weight_f32,
)


comptime KLEIN_DIRECT_24_GIB = 24 * 1024 * 1024 * 1024
comptime TArc = ArcPointer[Tensor]


struct KleinDirectOFTDeviceSlot(Copyable, Movable):
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


struct KleinStreamDirectDoRA(Copyable, Movable):
    var q: Optional[DoRAAdapterDevice]
    var k: Optional[DoRAAdapterDevice]
    var v: Optional[DoRAAdapterDevice]
    var out: Optional[DoRAAdapterDevice]
    var ff_in: Optional[DoRAAdapterDevice]
    var ff_out: Optional[DoRAAdapterDevice]

    def __init__(
        out self,
        var q: Optional[DoRAAdapterDevice], var k: Optional[DoRAAdapterDevice],
        var v: Optional[DoRAAdapterDevice], var out: Optional[DoRAAdapterDevice],
        var ff_in: Optional[DoRAAdapterDevice], var ff_out: Optional[DoRAAdapterDevice],
    ):
        self.q = q^
        self.k = k^
        self.v = v^
        self.out = out^
        self.ff_in = ff_in^
        self.ff_out = ff_out^


struct KleinDoubleDirectDoRA(Copyable, Movable):
    var img: KleinStreamDirectDoRA
    var txt: KleinStreamDirectDoRA

    def __init__(
        out self, var img: KleinStreamDirectDoRA, var txt: KleinStreamDirectDoRA,
    ):
        self.img = img^
        self.txt = txt^


struct KleinSingleDirectDoRA(Copyable, Movable):
    var qkv: Optional[DoRAAdapterDevice]
    var out: Optional[DoRAAdapterDevice]

    def __init__(
        out self, var qkv: Optional[DoRAAdapterDevice], var out: Optional[DoRAAdapterDevice],
    ):
        self.qkv = qkv^
        self.out = out^


struct KleinStackDirectDoRA(Movable):
    var dbl: List[KleinDoubleDirectDoRA]
    var sgl: List[KleinSingleDirectDoRA]

    def __init__(
        out self, var dbl: List[KleinDoubleDirectDoRA], var sgl: List[KleinSingleDirectDoRA],
    ):
        self.dbl = dbl^
        self.sgl = sgl^


struct KleinStreamDirectOFT(Copyable, Movable):
    var q: Optional[KleinDirectOFTDeviceSlot]
    var k: Optional[KleinDirectOFTDeviceSlot]
    var v: Optional[KleinDirectOFTDeviceSlot]
    var out: Optional[KleinDirectOFTDeviceSlot]
    var ff_in: Optional[KleinDirectOFTDeviceSlot]
    var ff_out: Optional[KleinDirectOFTDeviceSlot]

    def __init__(
        out self,
        var q: Optional[KleinDirectOFTDeviceSlot],
        var k: Optional[KleinDirectOFTDeviceSlot],
        var v: Optional[KleinDirectOFTDeviceSlot],
        var out: Optional[KleinDirectOFTDeviceSlot],
        var ff_in: Optional[KleinDirectOFTDeviceSlot],
        var ff_out: Optional[KleinDirectOFTDeviceSlot],
    ):
        self.q = q^
        self.k = k^
        self.v = v^
        self.out = out^
        self.ff_in = ff_in^
        self.ff_out = ff_out^


struct KleinDoubleDirectOFT(Copyable, Movable):
    var img: KleinStreamDirectOFT
    var txt: KleinStreamDirectOFT

    def __init__(
        out self, var img: KleinStreamDirectOFT, var txt: KleinStreamDirectOFT,
    ):
        self.img = img^
        self.txt = txt^


struct KleinSingleDirectOFT(Copyable, Movable):
    var qkv: Optional[KleinDirectOFTDeviceSlot]
    var out: Optional[KleinDirectOFTDeviceSlot]

    def __init__(
        out self,
        var qkv: Optional[KleinDirectOFTDeviceSlot],
        var out: Optional[KleinDirectOFTDeviceSlot],
    ):
        self.qkv = qkv^
        self.out = out^


struct KleinStackDirectOFT(Movable):
    var dbl: List[KleinDoubleDirectOFT]
    var sgl: List[KleinSingleDirectOFT]

    def __init__(
        out self, var dbl: List[KleinDoubleDirectOFT], var sgl: List[KleinSingleDirectOFT],
    ):
        self.dbl = dbl^
        self.sgl = sgl^


struct KleinDirectDoRAGradT(Copyable, Movable):
    var d_a: Optional[TArc]
    var d_b: Optional[TArc]
    var d_m: Optional[TArc]

    def __init__(
        out self, var d_a: Optional[TArc], var d_b: Optional[TArc],
        var d_m: Optional[TArc],
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_m = d_m^


struct KleinDirectOFTGradT(Copyable, Movable):
    var d_vec: Optional[TArc]

    def __init__(out self, var d_vec: Optional[TArc]):
        self.d_vec = d_vec^


struct KleinDirectDoRALinBwdT(Movable):
    var d_x: Tensor
    var dora: KleinDirectDoRAGradT

    def __init__(out self, var d_x: Tensor, var dora: KleinDirectDoRAGradT):
        self.d_x = d_x^
        self.dora = dora^


struct KleinDirectOFTLinBwdT(Movable):
    var d_x: Tensor
    var oft: KleinDirectOFTGradT

    def __init__(out self, var d_x: Tensor, var oft: KleinDirectOFTGradT):
        self.d_x = d_x^
        self.oft = oft^


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
    if targets < LOKR_TGT_ATTN or targets > LOKR_TGT_ALL:
        raise Error("klein direct LyCORIS: targets must be 1(attn)|2(attn+ff)|3(all)")


def klein_direct_total_slots(num_double: Int, num_single: Int) -> Int:
    return num_double * _DBL_SLOTS + num_single * _SGL_SLOTS


def klein_direct_active_slot_count(num_double: Int, num_single: Int, targets: Int) raises -> Int:
    _validate_targets(targets)
    var n = 0
    for _bi in range(num_double):
        for slot in range(_DBL_SLOTS):
            if _slot_targeted(True, slot, targets):
                n += 1
    for _bi in range(num_single):
        for slot in range(_SGL_SLOTS):
            if _slot_targeted(False, slot, targets):
                n += 1
    return n


def empty_klein_direct_dora_set() -> FlatDirectDoRASet:
    return empty_flat_direct_dora_set()


def empty_klein_direct_oft_set() -> FlatDirectOFTSet:
    return empty_flat_direct_oft_set()


def klein_direct_dense_carrier_bytes(
    num_double: Int, num_single: Int, D: Int, F: Int, targets: Int,
) raises -> Int:
    _validate_targets(targets)
    var elems = 0
    for _bi in range(num_double):
        for slot in range(_DBL_SLOTS):
            if not _slot_targeted(True, slot, targets):
                continue
            var dims = klein_lokr_slot_dims(True, slot, D, F)
            elems += dims[0] * dims[0] + dims[1] * dims[0]
    for _bi in range(num_single):
        for slot in range(_SGL_SLOTS):
            if not _slot_targeted(False, slot, targets):
                continue
            var dims = klein_lokr_slot_dims(False, slot, D, F)
            elems += dims[0] * dims[0] + dims[1] * dims[0]
    return elems * 2


def klein_direct_dora_trainable_bytes_estimate(
    num_double: Int, num_single: Int, D: Int, F: Int,
    rank: Int, targets: Int, wd_on_out: Bool = False,
) raises -> Int:
    _validate_targets(targets)
    var total = 0
    for _bi in range(num_double):
        for slot in range(_DBL_SLOTS):
            if not _slot_targeted(True, slot, targets):
                continue
            var dims = klein_lokr_slot_dims(True, slot, D, F)
            var in_f = dims[0]
            var out_f = dims[1]
            var mlen = out_f if wd_on_out else in_f
            var bf16_elems = rank * in_f + out_f * rank
            var f32_elems = mlen + (2 * rank * in_f) + (2 * out_f * rank) + (2 * mlen)
            total += bf16_elems * 2 + f32_elems * 4
    for _bi in range(num_single):
        for slot in range(_SGL_SLOTS):
            if not _slot_targeted(False, slot, targets):
                continue
            var dims = klein_lokr_slot_dims(False, slot, D, F)
            var in_f = dims[0]
            var out_f = dims[1]
            var mlen = out_f if wd_on_out else in_f
            var bf16_elems = rank * in_f + out_f * rank
            var f32_elems = mlen + (2 * rank * in_f) + (2 * out_f * rank) + (2 * mlen)
            total += bf16_elems * 2 + f32_elems * 4
    return total


def klein_direct_oft_trainable_bytes_estimate(
    num_double: Int, num_single: Int, D: Int, F: Int,
    block_size: Int, targets: Int,
) raises -> Int:
    _validate_targets(targets)
    var total = 0
    for _bi in range(num_double):
        for slot in range(_DBL_SLOTS):
            if not _slot_targeted(True, slot, targets):
                continue
            var dims = klein_lokr_slot_dims(True, slot, D, F)
            var in_f = dims[0]
            if in_f % block_size != 0:
                raise Error("klein_direct_oft_trainable_bytes_estimate: in_f not divisible by block_size")
            var r = in_f // block_size
            var ne = block_size * (block_size - 1) // 2
            total += 3 * r * ne * 4
    for _bi in range(num_single):
        for slot in range(_SGL_SLOTS):
            if not _slot_targeted(False, slot, targets):
                continue
            var dims = klein_lokr_slot_dims(False, slot, D, F)
            var in_f = dims[0]
            if in_f % block_size != 0:
                raise Error("klein_direct_oft_trainable_bytes_estimate: single in_f not divisible by block_size")
            var r = in_f // block_size
            var ne = block_size * (block_size - 1) // 2
            total += 3 * r * ne * 4
    return total


def klein_direct_dora_preflight(
    num_double: Int, num_single: Int, D: Int, F: Int,
    rank: Int, targets: Int, budget_bytes: Int, wd_on_out: Bool = False,
) raises -> Int:
    var direct = klein_direct_dora_trainable_bytes_estimate(
        num_double, num_single, D, F, rank, targets, wd_on_out,
    )
    if direct > budget_bytes:
        raise Error(
            String("Klein direct DoRA trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


def klein_direct_oft_preflight(
    num_double: Int, num_single: Int, D: Int, F: Int,
    block_size: Int, targets: Int, budget_bytes: Int,
) raises -> Int:
    var direct = klein_direct_oft_trainable_bytes_estimate(
        num_double, num_single, D, F, block_size, targets,
    )
    if direct > budget_bytes:
        raise Error(
            String("Klein direct OFT trainable state needs ") + String(direct)
            + String(" bytes (> budget ") + String(budget_bytes) + String(")")
        )
    return direct


def build_klein_direct_dora_set_from_checkpoint(
    st: SafeTensors, num_double: Int, num_single: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool = False,
) raises -> FlatDirectDoRASet:
    _validate_targets(targets)
    var set = empty_flat_direct_dora_set()
    var s = seed
    for bi in range(num_double):
        for slot in range(_DBL_SLOTS):
            if _slot_targeted(True, slot, targets):
                var dims = klein_lokr_slot_dims(True, slot, D, F)
                var w = klein_lokr_base_weight_f32(st, True, bi, slot, D, F)
                if len(w) != dims[0] * dims[1]:
                    raise Error("build_klein_direct_dora_set_from_checkpoint: double weight numel mismatch")
                flat_direct_dora_append_from_weight(
                    set, w^, dims[0], dims[1], rank, alpha,
                    klein_lokr_prefix(True, bi, slot), s, wd_on_out,
                )
            s += 1
    for bi in range(num_single):
        for slot in range(_SGL_SLOTS):
            if _slot_targeted(False, slot, targets):
                var dims = klein_lokr_slot_dims(False, slot, D, F)
                var w = klein_lokr_base_weight_f32(st, False, bi, slot, D, F)
                if len(w) != dims[0] * dims[1]:
                    raise Error("build_klein_direct_dora_set_from_checkpoint: single weight numel mismatch")
                flat_direct_dora_append_from_weight(
                    set, w^, dims[0], dims[1], rank, alpha,
                    klein_lokr_prefix(False, bi, slot), s, wd_on_out,
                )
            s += 1
    return set^


def build_klein_direct_oft_set(
    num_double: Int, num_single: Int, D: Int, F: Int,
    block_size: Int, targets: Int,
) raises -> FlatDirectOFTSet:
    _validate_targets(targets)
    var set = empty_flat_direct_oft_set()
    for bi in range(num_double):
        for slot in range(_DBL_SLOTS):
            if not _slot_targeted(True, slot, targets):
                continue
            var dims = klein_lokr_slot_dims(True, slot, D, F)
            flat_direct_oft_append(
                set, dims[0], dims[1], block_size,
                klein_lokr_prefix(True, bi, slot),
            )
    for bi in range(num_single):
        for slot in range(_SGL_SLOTS):
            if not _slot_targeted(False, slot, targets):
                continue
            var dims = klein_lokr_slot_dims(False, slot, D, F)
            flat_direct_oft_append(
                set, dims[0], dims[1], block_size,
                klein_lokr_prefix(False, bi, slot),
            )
    return set^


def build_klein_direct_oft_set_from_checkpoint(
    st: SafeTensors, num_double: Int, num_single: Int, D: Int, F: Int,
    block_size: Int, targets: Int,
) raises -> FlatDirectOFTSet:
    # OFT direct state only needs shape/prefix. The checkpoint argument keeps the
    # builder's call-site parallel with DoRA and fails early if targeted base
    # weights are absent or have unexpected dimensions.
    _validate_targets(targets)
    for bi in range(num_double):
        for slot in range(_DBL_SLOTS):
            if _slot_targeted(True, slot, targets):
                var dims = klein_lokr_slot_dims(True, slot, D, F)
                var w = klein_lokr_base_weight_f32(st, True, bi, slot, D, F)
                if len(w) != dims[0] * dims[1]:
                    raise Error("build_klein_direct_oft_set_from_checkpoint: double weight numel mismatch")
    for bi in range(num_single):
        for slot in range(_SGL_SLOTS):
            if _slot_targeted(False, slot, targets):
                var dims = klein_lokr_slot_dims(False, slot, D, F)
                var w = klein_lokr_base_weight_f32(st, False, bi, slot, D, F)
                if len(w) != dims[0] * dims[1]:
                    raise Error("build_klein_direct_oft_set_from_checkpoint: single weight numel mismatch")
    return build_klein_direct_oft_set(num_double, num_single, D, F, block_size, targets)


def _check_dora_projection_resident(
    slot_dev: DoRAAdapterDevice, x: Tensor, w_orig: Tensor, M: Int,
) raises:
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("klein_direct_dora_resident: x rank must be >= 1")
    if xshape[len(xshape) - 1] != slot_dev.in_f:
        raise Error("klein_direct_dora_resident: x trailing dim does not match slot")
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    if rows != M:
        raise Error("klein_direct_dora_resident: M does not match x leading rows")
    var wshape = w_orig.shape()
    if len(wshape) != 2 or wshape[0] != slot_dev.out_f or wshape[1] != slot_dev.in_f:
        raise Error("klein_direct_dora_resident: w_orig shape does not match slot")


def klein_direct_dora_projection_forward_resident(
    slot_dev: DoRAAdapterDevice, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
    bias: Optional[Tensor] = Optional[Tensor](None),
) raises -> Tensor:
    _check_dora_projection_resident(slot_dev, x, w_orig, M)
    var y = dora_substitution_forward_device(x, w_orig, slot_dev, ctx)
    if bias:
        if bias.value().dtype() != y.dtype():
            var bc = cast_tensor(bias.value(), y.dtype(), ctx)
            return add(y, bc, ctx)
        return add(y, bias.value(), ctx)
    return y^


def klein_direct_dora_projection_backward_resident(
    slot_dev: DoRAAdapterDevice, d_y: Tensor, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
) raises -> DoRADeviceGrads:
    _check_dora_projection_resident(slot_dev, x, w_orig, M)
    return dora_substitution_backward_device(d_y, x, w_orig, slot_dev, ctx)


def _oft_vec_tensor(set: FlatDirectOFTSet, slot: Int, ctx: DeviceContext) raises -> Tensor:
    if slot < 0 or slot >= len(set.ad):
        raise Error("klein_direct_oft_device: slot out of range")
    if not set.active[slot]:
        raise Error("klein_direct_oft_device: inactive slot")
    ref sl = set.ad[slot]
    if sl.b != 4:
        raise Error("klein_direct_oft_device: only block_size=4 is wired on GPU")
    return _f32_2d(sl.vec.copy(), sl.r, 6, ctx)


def klein_direct_oft_projection_slot_to_device(
    set: FlatDirectOFTSet, slot: Int, ctx: DeviceContext,
) raises -> KleinDirectOFTDeviceSlot:
    if slot < 0 or slot >= len(set.ad):
        raise Error("klein_direct_oft_projection_slot_to_device: slot out of range")
    if not set.active[slot]:
        raise Error("klein_direct_oft_projection_slot_to_device: inactive slot")
    ref sl = set.ad[slot]
    var vec = _oft_vec_tensor(set, slot, ctx)
    return KleinDirectOFTDeviceSlot(TArc(vec^), sl.in_f, sl.out_f, sl.b, sl.r)


def _check_oft_projection_resident(
    slot_dev: KleinDirectOFTDeviceSlot, x: Tensor, w_orig: Tensor, M: Int,
) raises:
    if slot_dev.b != 4:
        raise Error("klein_direct_oft_resident: only block_size=4 is wired on GPU")
    if slot_dev.vec[].dtype() != STDtype.F32:
        raise Error("klein_direct_oft_resident: vec storage must be F32")
    if slot_dev.vec[].shape() != [slot_dev.r, 6]:
        raise Error("klein_direct_oft_resident: vec shape mismatch")
    var xshape = x.shape()
    if len(xshape) < 1:
        raise Error("klein_direct_oft_resident: x rank must be >= 1")
    if xshape[len(xshape) - 1] != slot_dev.in_f:
        raise Error("klein_direct_oft_resident: x trailing dim does not match slot")
    var rows = 1
    for i in range(len(xshape) - 1):
        rows *= xshape[i]
    if rows != M:
        raise Error("klein_direct_oft_resident: M does not match x leading rows")
    var wshape = w_orig.shape()
    if len(wshape) != 2 or wshape[0] != slot_dev.out_f or wshape[1] != slot_dev.in_f:
        raise Error("klein_direct_oft_resident: w_orig shape does not match slot")


def klein_direct_oft_projection_forward_resident(
    slot_dev: KleinDirectOFTDeviceSlot, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
    bias: Optional[Tensor] = Optional[Tensor](None),
) raises -> Tensor:
    _check_oft_projection_resident(slot_dev, x, w_orig, M)
    var x_rot = oft_ot_rotate_b4(x, slot_dev.vec[], ctx)
    return linear(x_rot, w_orig, bias, ctx)


def klein_direct_oft_projection_backward_resident(
    slot_dev: KleinDirectOFTDeviceSlot, d_y: Tensor, x: Tensor, w_orig: Tensor,
    M: Int, ctx: DeviceContext,
) raises -> OFTOTDeviceGrads:
    _check_oft_projection_resident(slot_dev, x, w_orig, M)
    var d_x_rot = linear_backward_dx(d_y, w_orig, M, slot_dev.in_f, slot_dev.out_f, ctx)
    if d_x_rot.dtype() != x.dtype():
        d_x_rot = cast_tensor(d_x_rot^, x.dtype(), ctx)
    return oft_ot_rotate_backward_b4(d_x_rot^, x, slot_dev.vec[], ctx)


def klein_direct_dora_projection_forward_optional(
    x: Tensor, w: Tensor, ad: Optional[DoRAAdapterDevice],
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    if ad:
        return klein_direct_dora_projection_forward_resident(ad.value(), x, w, M, ctx)
    return linear(x, w, Optional[Tensor](None), ctx)


def klein_direct_dora_projection_backward_optional(
    d_y: Tensor, x: Tensor, w: Tensor, ad: Optional[DoRAAdapterDevice],
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> KleinDirectDoRALinBwdT:
    if ad:
        var g = klein_direct_dora_projection_backward_resident(
            ad.value(), d_y, x, w, M, ctx,
        )
        return KleinDirectDoRALinBwdT(
            g.d_x.clone(ctx),
            KleinDirectDoRAGradT(
                Optional[TArc](TArc(g.d_a.clone(ctx))),
                Optional[TArc](TArc(g.d_b.clone(ctx))),
                Optional[TArc](TArc(g.d_m.clone(ctx))),
            ),
        )
    var d_x = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    return KleinDirectDoRALinBwdT(d_x^, KleinDirectDoRAGradT(None, None, None))


def klein_direct_oft_projection_forward_optional(
    x: Tensor, w: Tensor, ad: Optional[KleinDirectOFTDeviceSlot],
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    if ad:
        return klein_direct_oft_projection_forward_resident(ad.value(), x, w, M, ctx)
    return linear(x, w, Optional[Tensor](None), ctx)


def klein_direct_oft_projection_backward_optional(
    d_y: Tensor, x: Tensor, w: Tensor, ad: Optional[KleinDirectOFTDeviceSlot],
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> KleinDirectOFTLinBwdT:
    if ad:
        var g = klein_direct_oft_projection_backward_resident(
            ad.value(), d_y, x, w, M, ctx,
        )
        return KleinDirectOFTLinBwdT(
            g.d_x.clone(ctx),
            KleinDirectOFTGradT(Optional[TArc](TArc(g.d_vec.clone(ctx)))),
        )
    var d_x = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    return KleinDirectOFTLinBwdT(d_x^, KleinDirectOFTGradT(None))


def klein_direct_dora_set_to_device(
    set: FlatDirectDoRASet, ctx: DeviceContext,
) raises -> List[DoRAAdapterDevice]:
    var slots = List[DoRAAdapterDevice]()
    for i in range(len(set.ad)):
        if not set.active[i]:
            raise Error("klein_direct_dora_set_to_device: inactive compact slot")
        slots.append(dora_device_from_host(set.ad[i], ctx))
    return slots^


def klein_direct_oft_set_to_device(
    set: FlatDirectOFTSet, ctx: DeviceContext,
) raises -> List[KleinDirectOFTDeviceSlot]:
    var slots = List[KleinDirectOFTDeviceSlot]()
    for i in range(len(set.ad)):
        if not set.active[i]:
            raise Error("klein_direct_oft_set_to_device: inactive compact slot")
        slots.append(klein_direct_oft_projection_slot_to_device(set, i, ctx))
    return slots^


def klein_direct_dora_blocks_to_device(
    set: FlatDirectDoRASet, num_double: Int, num_single: Int,
    targets: Int, ctx: DeviceContext,
) raises -> KleinStackDirectDoRA:
    _validate_targets(targets)
    var compact = 0
    var dbl = List[KleinDoubleDirectDoRA]()
    for _bi in range(num_double):
        var img_q = Optional[DoRAAdapterDevice](None)
        var img_k = Optional[DoRAAdapterDevice](None)
        var img_v = Optional[DoRAAdapterDevice](None)
        var img_out = Optional[DoRAAdapterDevice](None)
        var img_ff_in = Optional[DoRAAdapterDevice](None)
        var img_ff_out = Optional[DoRAAdapterDevice](None)
        var txt_q = Optional[DoRAAdapterDevice](None)
        var txt_k = Optional[DoRAAdapterDevice](None)
        var txt_v = Optional[DoRAAdapterDevice](None)
        var txt_out = Optional[DoRAAdapterDevice](None)
        var txt_ff_in = Optional[DoRAAdapterDevice](None)
        var txt_ff_out = Optional[DoRAAdapterDevice](None)
        for slot in range(_DBL_SLOTS):
            if not _slot_targeted(True, slot, targets):
                continue
            if compact >= len(set.ad):
                raise Error("klein_direct_dora_blocks_to_device: compact set too short")
            var dev = dora_device_from_host(set.ad[compact], ctx)
            if slot == 0:
                img_q = Optional[DoRAAdapterDevice](dev^)
            elif slot == 1:
                img_k = Optional[DoRAAdapterDevice](dev^)
            elif slot == 2:
                img_v = Optional[DoRAAdapterDevice](dev^)
            elif slot == 3:
                img_out = Optional[DoRAAdapterDevice](dev^)
            elif slot == 4:
                img_ff_in = Optional[DoRAAdapterDevice](dev^)
            elif slot == 5:
                img_ff_out = Optional[DoRAAdapterDevice](dev^)
            elif slot == 6:
                txt_q = Optional[DoRAAdapterDevice](dev^)
            elif slot == 7:
                txt_k = Optional[DoRAAdapterDevice](dev^)
            elif slot == 8:
                txt_v = Optional[DoRAAdapterDevice](dev^)
            elif slot == 9:
                txt_out = Optional[DoRAAdapterDevice](dev^)
            elif slot == 10:
                txt_ff_in = Optional[DoRAAdapterDevice](dev^)
            else:
                txt_ff_out = Optional[DoRAAdapterDevice](dev^)
            compact += 1
        var img = KleinStreamDirectDoRA(
            img_q^, img_k^, img_v^, img_out^, img_ff_in^, img_ff_out^,
        )
        var txt = KleinStreamDirectDoRA(
            txt_q^, txt_k^, txt_v^, txt_out^, txt_ff_in^, txt_ff_out^,
        )
        dbl.append(KleinDoubleDirectDoRA(img^, txt^))
    var sgl = List[KleinSingleDirectDoRA]()
    for _bi in range(num_single):
        var qkv = Optional[DoRAAdapterDevice](None)
        var out = Optional[DoRAAdapterDevice](None)
        for slot in range(_SGL_SLOTS):
            if not _slot_targeted(False, slot, targets):
                continue
            if compact >= len(set.ad):
                raise Error("klein_direct_dora_blocks_to_device: compact set too short")
            var dev = dora_device_from_host(set.ad[compact], ctx)
            if slot == 0:
                qkv = Optional[DoRAAdapterDevice](dev^)
            else:
                out = Optional[DoRAAdapterDevice](dev^)
            compact += 1
        sgl.append(KleinSingleDirectDoRA(qkv^, out^))
    if compact != len(set.ad):
        raise Error("klein_direct_dora_blocks_to_device: compact set has trailing slots")
    return KleinStackDirectDoRA(dbl^, sgl^)


def klein_direct_oft_blocks_to_device(
    set: FlatDirectOFTSet, num_double: Int, num_single: Int,
    targets: Int, ctx: DeviceContext,
) raises -> KleinStackDirectOFT:
    _validate_targets(targets)
    var compact = 0
    var dbl = List[KleinDoubleDirectOFT]()
    for _bi in range(num_double):
        var img_q = Optional[KleinDirectOFTDeviceSlot](None)
        var img_k = Optional[KleinDirectOFTDeviceSlot](None)
        var img_v = Optional[KleinDirectOFTDeviceSlot](None)
        var img_out = Optional[KleinDirectOFTDeviceSlot](None)
        var img_ff_in = Optional[KleinDirectOFTDeviceSlot](None)
        var img_ff_out = Optional[KleinDirectOFTDeviceSlot](None)
        var txt_q = Optional[KleinDirectOFTDeviceSlot](None)
        var txt_k = Optional[KleinDirectOFTDeviceSlot](None)
        var txt_v = Optional[KleinDirectOFTDeviceSlot](None)
        var txt_out = Optional[KleinDirectOFTDeviceSlot](None)
        var txt_ff_in = Optional[KleinDirectOFTDeviceSlot](None)
        var txt_ff_out = Optional[KleinDirectOFTDeviceSlot](None)
        for slot in range(_DBL_SLOTS):
            if not _slot_targeted(True, slot, targets):
                continue
            if compact >= len(set.ad):
                raise Error("klein_direct_oft_blocks_to_device: compact set too short")
            var dev = klein_direct_oft_projection_slot_to_device(set, compact, ctx)
            if slot == 0:
                img_q = Optional[KleinDirectOFTDeviceSlot](dev^)
            elif slot == 1:
                img_k = Optional[KleinDirectOFTDeviceSlot](dev^)
            elif slot == 2:
                img_v = Optional[KleinDirectOFTDeviceSlot](dev^)
            elif slot == 3:
                img_out = Optional[KleinDirectOFTDeviceSlot](dev^)
            elif slot == 4:
                img_ff_in = Optional[KleinDirectOFTDeviceSlot](dev^)
            elif slot == 5:
                img_ff_out = Optional[KleinDirectOFTDeviceSlot](dev^)
            elif slot == 6:
                txt_q = Optional[KleinDirectOFTDeviceSlot](dev^)
            elif slot == 7:
                txt_k = Optional[KleinDirectOFTDeviceSlot](dev^)
            elif slot == 8:
                txt_v = Optional[KleinDirectOFTDeviceSlot](dev^)
            elif slot == 9:
                txt_out = Optional[KleinDirectOFTDeviceSlot](dev^)
            elif slot == 10:
                txt_ff_in = Optional[KleinDirectOFTDeviceSlot](dev^)
            else:
                txt_ff_out = Optional[KleinDirectOFTDeviceSlot](dev^)
            compact += 1
        var img = KleinStreamDirectOFT(
            img_q^, img_k^, img_v^, img_out^, img_ff_in^, img_ff_out^,
        )
        var txt = KleinStreamDirectOFT(
            txt_q^, txt_k^, txt_v^, txt_out^, txt_ff_in^, txt_ff_out^,
        )
        dbl.append(KleinDoubleDirectOFT(img^, txt^))
    var sgl = List[KleinSingleDirectOFT]()
    for _bi in range(num_single):
        var qkv = Optional[KleinDirectOFTDeviceSlot](None)
        var out = Optional[KleinDirectOFTDeviceSlot](None)
        for slot in range(_SGL_SLOTS):
            if not _slot_targeted(False, slot, targets):
                continue
            if compact >= len(set.ad):
                raise Error("klein_direct_oft_blocks_to_device: compact set too short")
            var dev = klein_direct_oft_projection_slot_to_device(set, compact, ctx)
            if slot == 0:
                qkv = Optional[KleinDirectOFTDeviceSlot](dev^)
            else:
                out = Optional[KleinDirectOFTDeviceSlot](dev^)
            compact += 1
        sgl.append(KleinSingleDirectOFT(qkv^, out^))
    if compact != len(set.ad):
        raise Error("klein_direct_oft_blocks_to_device: compact set has trailing slots")
    return KleinStackDirectOFT(dbl^, sgl^)


def klein_direct_dora_projection_forward(
    set: FlatDirectDoRASet, slot: Int, x_h: List[Float32],
    w_orig: List[Float32], bias: List[Float32], M: Int,
) raises -> List[Float32]:
    var y = flat_direct_dora_forward_slot(set, slot, x_h, w_orig, M)
    if len(bias) == 0:
        return y^
    if len(bias) != set.ad[slot].out_f:
        raise Error("klein_direct_dora_projection_forward: bias numel mismatch")
    for m in range(M):
        for o in range(set.ad[slot].out_f):
            y[m * set.ad[slot].out_f + o] += bias[o]
    return y^


def klein_direct_dora_projection_backward(
    set: FlatDirectDoRASet, slot: Int, d_y_h: List[Float32],
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> DoRAGrads:
    return flat_direct_dora_backward_slot(set, slot, d_y_h, x_h, w_orig, M)


def klein_direct_oft_projection_forward(
    set: FlatDirectOFTSet, slot: Int, x_h: List[Float32],
    w_orig: List[Float32], bias: List[Float32], M: Int,
) raises -> List[Float32]:
    var y = flat_direct_oft_forward_slot(set, slot, x_h, w_orig, M)
    if len(bias) == 0:
        return y^
    if len(bias) != set.ad[slot].out_f:
        raise Error("klein_direct_oft_projection_forward: bias numel mismatch")
    for m in range(M):
        for o in range(set.ad[slot].out_f):
            y[m * set.ad[slot].out_f + o] += bias[o]
    return y^


def klein_direct_oft_projection_backward(
    set: FlatDirectOFTSet, slot: Int, d_y_h: List[Float32],
    x_h: List[Float32], w_orig: List[Float32], M: Int,
) raises -> OFTOTGrads:
    return flat_direct_oft_backward_slot(set, slot, d_y_h, x_h, w_orig, M)


def klein_direct_dora_zero_grads(set: FlatDirectDoRASet) -> FlatDirectDoRAGrads:
    var out = List[DoRAGrads]()
    for i in range(len(set.ad)):
        ref d = set.ad[i]
        out.append(DoRAGrads(
            _zeros(len(d.a)), _zeros(len(d.b)), _zeros(len(d.m)), List[Float32](),
        ))
    return FlatDirectDoRAGrads(out^)


def klein_direct_dora_scatter_slot_grad(
    mut grads: FlatDirectDoRAGrads, slot: Int, g: DoRAGrads,
) raises:
    if slot < 0 or slot >= len(grads.g):
        raise Error("klein_direct_dora_scatter_slot_grad: slot out of range")
    grads.g[slot] = DoRAGrads(g.d_a.copy(), g.d_b.copy(), g.d_m.copy(), List[Float32]())


def klein_direct_dora_grad_norm(g: FlatDirectDoRAGrads) -> Float64:
    return flat_direct_dora_grad_norm(g)


def klein_direct_dora_clip_grads(mut g: FlatDirectDoRAGrads, clip_scale: Float32):
    flat_direct_dora_clip_grads(g, clip_scale)


def klein_direct_dora_adamw_step(
    mut set: FlatDirectDoRASet, g: FlatDirectDoRAGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_dora_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def klein_direct_dora_zero_leg_l1(set: FlatDirectDoRASet) -> Float64:
    return flat_direct_dora_zero_leg_l1(set)


def klein_direct_dora_trainable_bytes(set: FlatDirectDoRASet) -> Int:
    return flat_direct_dora_trainable_bytes(set)


def klein_direct_oft_zero_grads(set: FlatDirectOFTSet) -> FlatDirectOFTGrads:
    var out = List[List[Float32]]()
    for i in range(len(set.ad)):
        out.append(_zeros(len(set.ad[i].vec)))
    return FlatDirectOFTGrads(out^)


def klein_direct_oft_scatter_slot_grad(
    mut grads: FlatDirectOFTGrads, slot: Int, g: OFTOTGrads,
) raises:
    if slot < 0 or slot >= len(grads.d_vec):
        raise Error("klein_direct_oft_scatter_slot_grad: slot out of range")
    grads.d_vec[slot] = g.d_vec.copy()


def klein_direct_oft_grad_norm(g: FlatDirectOFTGrads) -> Float64:
    return flat_direct_oft_grad_norm(g)


def klein_direct_oft_clip_grads(mut g: FlatDirectOFTGrads, clip_scale: Float32):
    flat_direct_oft_clip_grads(g, clip_scale)


def klein_direct_oft_adamw_step(
    mut set: FlatDirectOFTSet, g: FlatDirectOFTGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    flat_direct_oft_adamw_step(set, g, t, lr, beta1, beta2, eps, weight_decay)


def klein_direct_oft_vec_l1(set: FlatDirectOFTSet) -> Float64:
    return flat_direct_oft_vec_l1(set)


def klein_direct_oft_trainable_bytes(set: FlatDirectOFTSet) -> Int:
    return flat_direct_oft_trainable_bytes(set)


def save_klein_direct_dora(
    set: FlatDirectDoRASet, path: String, ctx: DeviceContext,
) raises -> Int:
    var named = List[NamedDoRA]()
    for i in range(len(set.ad)):
        if set.active[i]:
            named.append(NamedDoRA(set.prefix[i].copy(), set.ad[i].copy()))
    return save_dora_onetrainer(named, path, ctx)


def save_klein_direct_oft(
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
        raise Error("save_klein_direct_oft: refusing to write an empty OFT file")
    save_safetensors(names, tensors, path, ctx)
    return nmods
