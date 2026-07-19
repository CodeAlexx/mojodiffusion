# models/sd35/sd35_lycoris_stack.mojo -- LoKr/LoHa carrier dispatch for SD3.5.
#
# SD3.5 trains a flat SD35LoraSet: depth * 8 adapters in sd35_stack_lora slot
# order. LoKr/LoHa can be trained through the existing stack by materializing
# plain-LoRA carriers and chaining the returned dA/dB back to LyCORIS masters.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lokr_adapter import (
    LoKrAdapter, LoKrGrads, new_lokr_adapter, lokr_adamw,
)
from serenitymojo.training.lokr_stack import (
    lokr_carrier_adapter, lokr_carrier_r_eff, lokr_chain_carrier_grads,
    _inactive_carrier, _dummy_lokr, _empty_lokr_grads,
    _grads_sqsum, _grads_scale,
)
from serenitymojo.training.loha_adapter import (
    LoHaAdapter, LoHaGrads, new_loha_adapter, loha_adamw,
)
from serenitymojo.training.loha_stack import (
    loha_carrier_adapter, loha_carrier_r_eff, loha_chain_carrier_grads,
    _dummy_loha, _empty_loha_grads, _loha_grads_sqsum, _loha_grads_scale,
)
from serenitymojo.training.lokr_save import NamedLoKr, save_lokr_peft
from serenitymojo.training.loha_save import NamedLoHa, save_loha_peft
from serenitymojo.models.sd35.sd35_stack_lora import (
    SD35LoraSet, SLOTS_PER_BLOCK,
    SLOT_CTX_QKV, SLOT_CTX_PROJ, SLOT_CTX_FC1, SLOT_CTX_FC2,
    SLOT_X_QKV, SLOT_X_PROJ, SLOT_X_FC1, SLOT_X_FC2,
)

comptime SD35_LYCORIS_TGT_ATTN = 1
comptime SD35_LYCORIS_TGT_ALL = 2


def sd35_lycoris_slot_dims(slot: Int, D: Int, F: Int) raises -> Tuple[Int, Int]:
    var s = slot % SLOTS_PER_BLOCK
    if s == SLOT_CTX_QKV or s == SLOT_X_QKV:
        return (D, 3 * D)
    if s == SLOT_CTX_FC1 or s == SLOT_X_FC1:
        return (D, F)
    if s == SLOT_CTX_FC2 or s == SLOT_X_FC2:
        return (F, D)
    return (D, D)


def _sd35_slot_is_attn(slot: Int) -> Bool:
    var s = slot % SLOTS_PER_BLOCK
    return (
        s == SLOT_CTX_QKV or s == SLOT_CTX_PROJ
        or s == SLOT_X_QKV or s == SLOT_X_PROJ
    )


def _sd35_slot_targeted(slot: Int, targets: Int) -> Bool:
    if _sd35_slot_is_attn(slot):
        return targets >= SD35_LYCORIS_TGT_ATTN
    return targets >= SD35_LYCORIS_TGT_ALL


def _sd35_slot_factor(slot: Int, factor: Int, factor_attn: Int, factor_ff: Int) -> Int:
    if _sd35_slot_is_attn(slot):
        return factor_attn if factor_attn != 0 else factor
    return factor_ff if factor_ff != 0 else factor


def _sd35_prefix(bi: Int, slot: Int) -> String:
    var bp = String("transformer.joint_blocks.") + String(bi)
    if slot == SLOT_CTX_QKV:
        return bp + ".context_block.attn.qkv"
    if slot == SLOT_CTX_PROJ:
        return bp + ".context_block.attn.proj"
    if slot == SLOT_CTX_FC1:
        return bp + ".context_block.mlp.fc1"
    if slot == SLOT_CTX_FC2:
        return bp + ".context_block.mlp.fc2"
    if slot == SLOT_X_QKV:
        return bp + ".x_block.attn.qkv"
    if slot == SLOT_X_PROJ:
        return bp + ".x_block.attn.proj"
    if slot == SLOT_X_FC1:
        return bp + ".x_block.mlp.fc1"
    return bp + ".x_block.mlp.fc2"


struct SD35LoKrSet(Copyable, Movable):
    var ad: List[LoKrAdapter]
    var active: List[Bool]
    var depth: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoKrAdapter], var active: List[Bool],
        depth: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.depth = depth
        self.rank = rank


def empty_sd35_lokr_set() -> SD35LoKrSet:
    return SD35LoKrSet(List[LoKrAdapter](), List[Bool](), 0, 0)


def build_sd35_lokr_set(
    depth: Int, D: Int, F: Int,
    rank: Int, alpha: Float32,
    factor: Int, factor_attn: Int, factor_ff: Int,
    decompose_both: Bool, full_matrix: Bool,
    targets: Int, seed: UInt64,
) raises -> SD35LoKrSet:
    if targets < SD35_LYCORIS_TGT_ATTN or targets > 3:
        raise Error("build_sd35_lokr_set: targets must be 1(attn)|2(all)|3(all)")
    var ad = List[LoKrAdapter]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(depth):
        for slot in range(SLOTS_PER_BLOCK):
            if _sd35_slot_targeted(slot, targets):
                var dims = sd35_lycoris_slot_dims(slot, D, F)
                var f = _sd35_slot_factor(slot, factor, factor_attn, factor_ff)
                ad.append(new_lokr_adapter(
                    dims[0], dims[1], rank, alpha, f, s,
                    decompose_both, full_matrix,
                ))
                active.append(True)
            else:
                ad.append(_dummy_lokr())
                active.append(False)
            s += 1
    return SD35LoKrSet(ad^, active^, depth, rank)


def sd35_lokr_carrier_set(set: SD35LoKrSet, D: Int, F: Int) raises -> SD35LoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(lokr_carrier_adapter(set.ad[i]))
        else:
            var dims = sd35_lycoris_slot_dims(i % SLOTS_PER_BLOCK, D, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return SD35LoraSet(ad^, set.depth, set.rank)


def sd35_lokr_carrier_total_bytes(set: SD35LoKrSet, D: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = lokr_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = sd35_lycoris_slot_dims(i % SLOTS_PER_BLOCK, D, F)
            elems += dims[0] + dims[1]
    return elems * 2


struct SD35LoKrGrads(Movable):
    var g: List[LoKrGrads]

    def __init__(out self, var g: List[LoKrGrads]):
        self.g = g^


def sd35_lokr_chain_all(
    set: SD35LoKrSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> SD35LoKrGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("sd35_lokr_chain_all: grad list count mismatch")
    var g = List[LoKrGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(lokr_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_lokr_grads())
    return SD35LoKrGrads(g^)


def sd35_lokr_grad_norm(grads: SD35LoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _grads_sqsum(grads.g[i])
    return sqrt(s)


def sd35_lokr_clip_grads(mut grads: SD35LoKrGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _grads_scale(grads.g[i], clip_scale)


def sd35_lokr_adamw_step(
    mut set: SD35LoKrSet, grads: SD35LoKrGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            lokr_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def sd35_lokr_zero_leg_l1(set: SD35LoKrSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        if lo.w2_factored:
            for j in range(len(lo.w2b)):
                var v = Float64(lo.w2b[j].cast[DType.float32]())
                s += v if v >= 0.0 else -v
        else:
            for j in range(len(lo.w2)):
                var v = Float64(lo.w2[j].cast[DType.float32]())
                s += v if v >= 0.0 else -v
    return s


def save_sd35_lokr(set: SD35LoKrSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoKr]()
    for bi in range(set.depth):
        for slot in range(SLOTS_PER_BLOCK):
            var flat = bi * SLOTS_PER_BLOCK + slot
            if set.active[flat]:
                named.append(NamedLoKr(_sd35_prefix(bi, slot), set.ad[flat].copy()))
    return save_lokr_peft(named, path, ctx)


struct SD35LoHaSet(Copyable, Movable):
    var ad: List[LoHaAdapter]
    var active: List[Bool]
    var depth: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoHaAdapter], var active: List[Bool],
        depth: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.depth = depth
        self.rank = rank


def empty_sd35_loha_set() -> SD35LoHaSet:
    return SD35LoHaSet(List[LoHaAdapter](), List[Bool](), 0, 0)


def build_sd35_loha_set(
    depth: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
) raises -> SD35LoHaSet:
    if targets < SD35_LYCORIS_TGT_ATTN or targets > 3:
        raise Error("build_sd35_loha_set: targets must be 1(attn)|2(all)|3(all)")
    var ad = List[LoHaAdapter]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(depth):
        for slot in range(SLOTS_PER_BLOCK):
            if _sd35_slot_targeted(slot, targets):
                var dims = sd35_lycoris_slot_dims(slot, D, F)
                ad.append(new_loha_adapter(dims[0], dims[1], rank, alpha, s))
                active.append(True)
            else:
                ad.append(_dummy_loha())
                active.append(False)
            s += 1
    return SD35LoHaSet(ad^, active^, depth, rank)


def sd35_loha_carrier_set(set: SD35LoHaSet, D: Int, F: Int) raises -> SD35LoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(loha_carrier_adapter(set.ad[i]))
        else:
            var dims = sd35_lycoris_slot_dims(i % SLOTS_PER_BLOCK, D, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return SD35LoraSet(ad^, set.depth, set.rank)


def sd35_loha_carrier_total_bytes(set: SD35LoHaSet, D: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = loha_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = sd35_lycoris_slot_dims(i % SLOTS_PER_BLOCK, D, F)
            elems += dims[0] + dims[1]
    return elems * 2


struct SD35LoHaGrads(Movable):
    var g: List[LoHaGrads]

    def __init__(out self, var g: List[LoHaGrads]):
        self.g = g^


def sd35_loha_chain_all(
    set: SD35LoHaSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> SD35LoHaGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("sd35_loha_chain_all: grad list count mismatch")
    var g = List[LoHaGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(loha_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_loha_grads())
    return SD35LoHaGrads(g^)


def sd35_loha_grad_norm(grads: SD35LoHaGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _loha_grads_sqsum(grads.g[i])
    return sqrt(s)


def sd35_loha_clip_grads(mut grads: SD35LoHaGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _loha_grads_scale(grads.g[i], clip_scale)


def sd35_loha_adamw_step(
    mut set: SD35LoHaSet, grads: SD35LoHaGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            loha_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def sd35_loha_zero_leg_l1(set: SD35LoHaSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def save_sd35_loha(set: SD35LoHaSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoHa]()
    for bi in range(set.depth):
        for slot in range(SLOTS_PER_BLOCK):
            var flat = bi * SLOTS_PER_BLOCK + slot
            if set.active[flat]:
                named.append(NamedLoHa(_sd35_prefix(bi, slot), set.ad[flat].copy()))
    return save_loha_peft(named, path, ctx)
