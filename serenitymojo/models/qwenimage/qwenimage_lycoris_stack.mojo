# models/qwenimage/qwenimage_lycoris_stack.mojo -- LoKr/LoHa carrier dispatch.
#
# Qwen-Image trains a flat QwenLoraSet: num_double * 12 adapters in
# qwenimage_stack_lora.DBLSLOTS order. Additive LyCORIS families can be trained
# by materializing plain-LoRA carriers, running the existing stack, then chaining
# dA/dB back to the LyCORIS masters.

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
from serenitymojo.models.qwenimage.qwenimage_stack_lora import (
    QwenLoraSet, DBL_SLOTS,
)


comptime QWEN_LYCORIS_TGT_ATTN = 1
comptime QWEN_LYCORIS_TGT_ALL = 2


def qwen_lycoris_slot_dims(slot: Int, D: Int, F: Int) raises -> Tuple[Int, Int]:
    var s = slot % DBL_SLOTS
    if s == 4 or s == 10:
        return (D, F)       # img/txt ff_up
    if s == 5 or s == 11:
        return (F, D)       # img/txt ff_down
    return (D, D)           # q/k/v/out for img/txt


def _qwen_slot_is_attn(slot: Int) -> Bool:
    var s = slot % DBL_SLOTS
    return s <= 3 or (s >= 6 and s <= 9)


def _qwen_slot_targeted(slot: Int, targets: Int) -> Bool:
    if _qwen_slot_is_attn(slot):
        return targets >= QWEN_LYCORIS_TGT_ATTN
    return targets >= QWEN_LYCORIS_TGT_ALL


def _qwen_slot_factor(
    slot: Int, factor: Int, factor_attn: Int, factor_ff: Int
) -> Int:
    if _qwen_slot_is_attn(slot):
        return factor_attn if factor_attn != 0 else factor
    return factor_ff if factor_ff != 0 else factor


def _qwen_prefix(block_idx: Int, slot: Int) -> String:
    var b = String("transformer.transformer_blocks.") + String(block_idx)
    if slot == 0:
        return b + ".attn.to_q"
    if slot == 1:
        return b + ".attn.to_k"
    if slot == 2:
        return b + ".attn.to_v"
    if slot == 3:
        return b + ".attn.to_out.0"
    if slot == 4:
        return b + ".img_mlp.net.0.proj"
    if slot == 5:
        return b + ".img_mlp.net.2"
    if slot == 6:
        return b + ".attn.add_q_proj"
    if slot == 7:
        return b + ".attn.add_k_proj"
    if slot == 8:
        return b + ".attn.add_v_proj"
    if slot == 9:
        return b + ".attn.to_add_out"
    if slot == 10:
        return b + ".txt_mlp.net.0.proj"
    return b + ".txt_mlp.net.2"


struct QwenLoKrSet(Copyable, Movable):
    var ad: List[LoKrAdapter]
    var active: List[Bool]
    var num_double: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoKrAdapter], var active: List[Bool],
        num_double: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.num_double = num_double
        self.rank = rank


def empty_qwen_lokr_set() -> QwenLoKrSet:
    return QwenLoKrSet(List[LoKrAdapter](), List[Bool](), 0, 0)


def build_qwen_lokr_set(
    num_double: Int, D: Int, F: Int,
    rank: Int, alpha: Float32,
    factor: Int, factor_attn: Int, factor_ff: Int,
    decompose_both: Bool, full_matrix: Bool,
    targets: Int, seed: UInt64,
) raises -> QwenLoKrSet:
    if targets < QWEN_LYCORIS_TGT_ATTN or targets > 3:
        raise Error("build_qwen_lokr_set: targets must be 1(attn)|2(all)|3(all)")
    var ad = List[LoKrAdapter]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_double):
        for slot in range(DBL_SLOTS):
            if _qwen_slot_targeted(slot, targets):
                var dims = qwen_lycoris_slot_dims(slot, D, F)
                var f = _qwen_slot_factor(slot, factor, factor_attn, factor_ff)
                ad.append(new_lokr_adapter(
                    dims[0], dims[1], rank, alpha, f, s,
                    decompose_both, full_matrix,
                ))
                active.append(True)
            else:
                ad.append(_dummy_lokr())
                active.append(False)
            s += 1
    return QwenLoKrSet(ad^, active^, num_double, rank)


def qwen_lokr_carrier_set(set: QwenLoKrSet, D: Int, F: Int) raises -> QwenLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(lokr_carrier_adapter(set.ad[i]))
        else:
            var dims = qwen_lycoris_slot_dims(i % DBL_SLOTS, D, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return QwenLoraSet(ad^, set.num_double, set.rank)


def qwen_lokr_carrier_total_bytes(set: QwenLoKrSet, D: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = lokr_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = qwen_lycoris_slot_dims(i % DBL_SLOTS, D, F)
            elems += dims[0] + dims[1]
    return elems * 2


struct QwenLoKrGrads(Movable):
    var g: List[LoKrGrads]

    def __init__(out self, var g: List[LoKrGrads]):
        self.g = g^


def qwen_lokr_chain_all(
    set: QwenLoKrSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> QwenLoKrGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("qwen_lokr_chain_all: grad list count mismatch")
    var g = List[LoKrGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(lokr_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_lokr_grads())
    return QwenLoKrGrads(g^)


def qwen_lokr_grad_norm(grads: QwenLoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _grads_sqsum(grads.g[i])
    return sqrt(s)


def qwen_lokr_clip_grads(mut grads: QwenLoKrGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _grads_scale(grads.g[i], clip_scale)


def qwen_lokr_adamw_step(
    mut set: QwenLoKrSet, grads: QwenLoKrGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            lokr_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def qwen_lokr_zero_leg_l1(set: QwenLoKrSet) -> Float64:
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


def save_qwen_lokr(set: QwenLoKrSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoKr]()
    for bi in range(set.num_double):
        for slot in range(DBL_SLOTS):
            var flat = bi * DBL_SLOTS + slot
            if set.active[flat]:
                named.append(NamedLoKr(_qwen_prefix(bi, slot), set.ad[flat].copy()))
    return save_lokr_peft(named, path, ctx)


struct QwenLoHaSet(Copyable, Movable):
    var ad: List[LoHaAdapter]
    var active: List[Bool]
    var num_double: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoHaAdapter], var active: List[Bool],
        num_double: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.num_double = num_double
        self.rank = rank


def empty_qwen_loha_set() -> QwenLoHaSet:
    return QwenLoHaSet(List[LoHaAdapter](), List[Bool](), 0, 0)


def build_qwen_loha_set(
    num_double: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
) raises -> QwenLoHaSet:
    if targets < QWEN_LYCORIS_TGT_ATTN or targets > 3:
        raise Error("build_qwen_loha_set: targets must be 1(attn)|2(all)|3(all)")
    var ad = List[LoHaAdapter]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_double):
        for slot in range(DBL_SLOTS):
            if _qwen_slot_targeted(slot, targets):
                var dims = qwen_lycoris_slot_dims(slot, D, F)
                ad.append(new_loha_adapter(dims[0], dims[1], rank, alpha, s))
                active.append(True)
            else:
                ad.append(_dummy_loha())
                active.append(False)
            s += 1
    return QwenLoHaSet(ad^, active^, num_double, rank)


def qwen_loha_carrier_set(set: QwenLoHaSet, D: Int, F: Int) raises -> QwenLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(loha_carrier_adapter(set.ad[i]))
        else:
            var dims = qwen_lycoris_slot_dims(i % DBL_SLOTS, D, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return QwenLoraSet(ad^, set.num_double, set.rank)


def qwen_loha_carrier_total_bytes(set: QwenLoHaSet, D: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = loha_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = qwen_lycoris_slot_dims(i % DBL_SLOTS, D, F)
            elems += dims[0] + dims[1]
    return elems * 2


struct QwenLoHaGrads(Movable):
    var g: List[LoHaGrads]

    def __init__(out self, var g: List[LoHaGrads]):
        self.g = g^


def qwen_loha_chain_all(
    set: QwenLoHaSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> QwenLoHaGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("qwen_loha_chain_all: grad list count mismatch")
    var g = List[LoHaGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(loha_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_loha_grads())
    return QwenLoHaGrads(g^)


def qwen_loha_grad_norm(grads: QwenLoHaGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _loha_grads_sqsum(grads.g[i])
    return sqrt(s)


def qwen_loha_clip_grads(mut grads: QwenLoHaGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _loha_grads_scale(grads.g[i], clip_scale)


def qwen_loha_adamw_step(
    mut set: QwenLoHaSet, grads: QwenLoHaGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            loha_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def qwen_loha_zero_leg_l1(set: QwenLoHaSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def save_qwen_loha(set: QwenLoHaSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoHa]()
    for bi in range(set.num_double):
        for slot in range(DBL_SLOTS):
            var flat = bi * DBL_SLOTS + slot
            if set.active[flat]:
                named.append(NamedLoHa(_qwen_prefix(bi, slot), set.ad[flat].copy()))
    return save_loha_peft(named, path, ctx)
