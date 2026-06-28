# models/zimage/zimage_loha_stack.mojo — LoHa carrier dispatch for Z-Image
# (twin of zimage_lokr_stack.mojo). LoHa has no factorization variants, so the
# orchestration is simpler than LoKr: one LoHaAdapter per active slot, carrier
# r_eff=r², w2a is the zero-leg. Reuses the model-agnostic LoHa carrier core +
# the zimage slot geometry from zimage_lokr_stack.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.loha_adapter import (
    LoHaAdapter, LoHaGrads, new_loha_adapter, loha_adamw,
)
from serenitymojo.training.loha_stack import (
    loha_carrier_adapter, loha_carrier_r_eff, loha_chain_carrier_grads,
    _dummy_loha, _empty_loha_grads, _loha_grads_sqsum,
)
from serenitymojo.training.lokr_stack import _inactive_carrier
from serenitymojo.training.loha_save import NamedLoHa, save_loha_peft
from serenitymojo.models.zimage.lora_block import (
    ZIMAGE_SLOTS, SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_W1, SLOT_W3, SLOT_W2,
)
from serenitymojo.models.zimage.zimage_lokr_stack import (
    zimage_lokr_slot_dims, _zimage_slot_targeted,
)


struct ZImageLoHaSet(Copyable, Movable):
    var ad: List[LoHaAdapter]
    var active: List[Bool]
    var num_nr: Int
    var num_cr: Int
    var num_main: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoHaAdapter], var active: List[Bool],
        num_nr: Int, num_cr: Int, num_main: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.num_nr = num_nr
        self.num_cr = num_cr
        self.num_main = num_main
        self.rank = rank

    def num_blocks(self) -> Int:
        return self.num_nr + self.num_cr + self.num_main


def empty_zimage_loha_set() -> ZImageLoHaSet:
    return ZImageLoHaSet(List[LoHaAdapter](), List[Bool](), 0, 0, 0, 0)


def build_zimage_loha_set(
    num_nr: Int, num_cr: Int, num_main: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
) raises -> ZImageLoHaSet:
    if targets < 1 or targets > 2:
        raise Error("build_zimage_loha_set: targets must be 1(attn)|2(all)")
    var ad = List[LoHaAdapter]()
    var active = List[Bool]()
    var s = seed
    var total_blocks = num_nr + num_cr + num_main
    for _b in range(total_blocks):
        for slot in range(ZIMAGE_SLOTS):
            if _zimage_slot_targeted(slot, targets):
                var dims = zimage_lokr_slot_dims(slot, D, F)
                ad.append(new_loha_adapter(dims[0], dims[1], rank, alpha, s))
                active.append(True)
            else:
                ad.append(_dummy_loha())
                active.append(False)
            s += 1
    return ZImageLoHaSet(ad^, active^, num_nr, num_cr, num_main, rank)


def zimage_loha_carrier_lists(set: ZImageLoHaSet, D: Int, F: Int) raises -> List[LoraAdapter]:
    var out = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(loha_carrier_adapter(set.ad[i]))
        else:
            var dims = zimage_lokr_slot_dims(i % ZIMAGE_SLOTS, D, F)
            out.append(_inactive_carrier(dims[0], dims[1]))
    return out^


def zimage_loha_carrier_total_bytes(set: ZImageLoHaSet) -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = loha_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            elems += set.ad[i].in_f + set.ad[i].out_f
    return elems * 2


struct ZImageLoHaGrads(Movable):
    var g: List[LoHaGrads]

    def __init__(out self, var g: List[LoHaGrads]):
        self.g = g^


def zimage_loha_chain_all(
    set: ZImageLoHaSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> ZImageLoHaGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("zimage_loha_chain_all: grad list count mismatch")
    var g = List[LoHaGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(loha_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_loha_grads())
    return ZImageLoHaGrads(g^)


def zimage_loha_grad_norm(grads: ZImageLoHaGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _loha_grads_sqsum(grads.g[i])
    return sqrt(s)


def zimage_loha_adamw_step(
    mut set: ZImageLoHaSet, grads: ZImageLoHaGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            loha_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


# w2a is the LoHa zero-leg (starts EXACTLY 0; must be >0 after a real step).
def zimage_loha_zero_leg_l1(set: ZImageLoHaSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def _zimage_loha_prefix(set: ZImageLoHaSet, block_idx: Int, slot: Int) -> String:
    var sp: String
    if block_idx < set.num_nr:
        sp = String("noise_refiner.") + String(block_idx)
    elif block_idx < set.num_nr + set.num_cr:
        sp = String("context_refiner.") + String(block_idx - set.num_nr)
    else:
        sp = String("layers.") + String(block_idx - set.num_nr - set.num_cr)
    if slot == SLOT_Q:
        return sp + ".attention.to_q"
    elif slot == SLOT_K:
        return sp + ".attention.to_k"
    elif slot == SLOT_V:
        return sp + ".attention.to_v"
    elif slot == SLOT_O:
        return sp + ".attention.to_out.0"
    elif slot == SLOT_W1:
        return sp + ".feed_forward.w1"
    elif slot == SLOT_W3:
        return sp + ".feed_forward.w3"
    return sp + ".feed_forward.w2"


def save_zimage_loha(set: ZImageLoHaSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoHa]()
    for bi in range(set.num_blocks()):
        for slot in range(ZIMAGE_SLOTS):
            var flat = bi * ZIMAGE_SLOTS + slot
            if set.active[flat]:
                named.append(NamedLoHa(_zimage_loha_prefix(set, bi, slot), set.ad[flat].copy()))
    return save_loha_peft(named, path, ctx)
