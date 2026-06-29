# models/krea2/krea2_loha_stack.mojo -- LoHa carrier dispatch for krea2.
#
# krea2's training stack consumes one flat List[LoraAdapter] in the same 8-slot
# order as krea2_lokr_stack. LoHa factors into a small rank^2 carrier, so the
# existing krea2 LoRA forward/backward can train LoHa masters without stack
# changes.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.loha_adapter import (
    LoHaAdapter, LoHaGrads, new_loha_adapter, loha_adamw,
)
from serenitymojo.training.loha_stack import (
    loha_carrier_adapter, loha_carrier_r_eff, loha_chain_carrier_grads,
    _dummy_loha, _empty_loha_grads, _loha_grads_sqsum, _loha_grads_scale,
)
from serenitymojo.training.lokr_stack import _inactive_carrier
from serenitymojo.training.loha_save import NamedLoHa, save_loha_peft
from serenitymojo.models.krea2.krea2_lokr_stack import (
    KREA2_SLOTS, K2LOKR_TGT_ATTN, K2LOKR_TGT_ALL,
    krea2_lokr_slot_dims, _krea2_slot_targeted,
)


struct Krea2LoHaSet(Copyable, Movable):
    var ad: List[LoHaAdapter]
    var active: List[Bool]
    var num_blocks: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoHaAdapter], var active: List[Bool],
        num_blocks: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.num_blocks = num_blocks
        self.rank = rank


def empty_krea2_loha_set() -> Krea2LoHaSet:
    return Krea2LoHaSet(List[LoHaAdapter](), List[Bool](), 0, 0)


def build_krea2_loha_set(
    num_blocks: Int, D: Int, F: Int, qdim: Int, kvdim: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
) raises -> Krea2LoHaSet:
    if targets < K2LOKR_TGT_ATTN or targets > K2LOKR_TGT_ALL:
        raise Error("build_krea2_loha_set: targets must be 1(attn)|2(all)")
    var ad = List[LoHaAdapter]()
    var active = List[Bool]()
    var s = seed
    for _b in range(num_blocks):
        for slot in range(KREA2_SLOTS):
            if _krea2_slot_targeted(slot, targets):
                var dims = krea2_lokr_slot_dims(slot, D, F, qdim, kvdim)
                ad.append(new_loha_adapter(dims[0], dims[1], rank, alpha, s))
                active.append(True)
            else:
                ad.append(_dummy_loha())
                active.append(False)
            s += 1
    return Krea2LoHaSet(ad^, active^, num_blocks, rank)


def krea2_loha_carrier_lists(
    set: Krea2LoHaSet, D: Int, F: Int, qdim: Int, kvdim: Int
) raises -> List[LoraAdapter]:
    var out = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(loha_carrier_adapter(set.ad[i]))
        else:
            var dims = krea2_lokr_slot_dims(i % KREA2_SLOTS, D, F, qdim, kvdim)
            out.append(_inactive_carrier(dims[0], dims[1]))
    return out^


def krea2_loha_carrier_total_bytes(
    set: Krea2LoHaSet, D: Int, F: Int, qdim: Int, kvdim: Int
) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = loha_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = krea2_lokr_slot_dims(i % KREA2_SLOTS, D, F, qdim, kvdim)
            elems += dims[0] + dims[1]
    return elems * 2


struct Krea2LoHaGrads(Movable):
    var g: List[LoHaGrads]

    def __init__(out self, var g: List[LoHaGrads]):
        self.g = g^


def krea2_loha_chain_all(
    set: Krea2LoHaSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> Krea2LoHaGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("krea2_loha_chain_all: grad list count mismatch")
    var g = List[LoHaGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(loha_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_loha_grads())
    return Krea2LoHaGrads(g^)


def krea2_loha_grad_norm(grads: Krea2LoHaGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _loha_grads_sqsum(grads.g[i])
    return sqrt(s)


def krea2_loha_clip_grads(mut grads: Krea2LoHaGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _loha_grads_scale(grads.g[i], clip_scale)


def krea2_loha_adamw_step(
    mut set: Krea2LoHaSet, grads: Krea2LoHaGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            loha_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


# w2a is the LoHa zero-leg: zero at init, nonzero after a real master update.
def krea2_loha_zero_leg_l1(set: Krea2LoHaSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def _krea2_loha_prefix(bi: Int, slot: Int) raises -> String:
    var b = String("diffusion_model.blocks.") + String(bi)
    if slot == 0:
        return b + ".attn.wq"
    elif slot == 1:
        return b + ".attn.wk"
    elif slot == 2:
        return b + ".attn.wv"
    elif slot == 3:
        return b + ".attn.gate"
    elif slot == 4:
        return b + ".attn.wo"
    elif slot == 5:
        return b + ".mlp.gate"
    elif slot == 6:
        return b + ".mlp.up"
    elif slot == 7:
        return b + ".mlp.down"
    raise Error(String("_krea2_loha_prefix: bad slot ") + String(slot))


def save_krea2_loha(set: Krea2LoHaSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoHa]()
    for bi in range(set.num_blocks):
        for slot in range(KREA2_SLOTS):
            var flat = bi * KREA2_SLOTS + slot
            if set.active[flat]:
                named.append(NamedLoHa(_krea2_loha_prefix(bi, slot), set.ad[flat].copy()))
    return save_loha_peft(named, path, ctx)
