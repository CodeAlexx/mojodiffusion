# models/zimage/zimage_lokr_stack.mojo — LoKr carrier dispatch for the Z-Image
# trainer (mirrors training/lokr_stack.mojo's klein orchestration, adapted to
# zimage's FLAT geometry: 3 segments (nr|cr|main) × ZIMAGE_SLOTS=7 plain slots,
# no fused qkv, no dbl/sgl split). The carrier CORE (lokr_carrier_adapter /
# lokr_chain_carrier_grads) is model-agnostic and reused as-is — only the per-
# model SET orchestration is new. Carriers ride the EXISTING zimage_stack_lora
# forward/backward (ZImageLoraSet (a,b) path); no stack/kernel change.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lokr_adapter import (
    LoKrAdapter, LoKrGrads, new_lokr_adapter, lokr_adamw,
)
from serenitymojo.training.lokr_stack import (
    lokr_carrier_adapter, lokr_carrier_r_eff, lokr_chain_carrier_grads,
    _inactive_carrier, _dummy_lokr, _empty_lokr_grads, _grads_sqsum,
)
from serenitymojo.training.lokr_save import NamedLoKr, save_lokr_peft
from serenitymojo.models.zimage.lora_block import (
    ZIMAGE_SLOTS, SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_W1, SLOT_W3, SLOT_W2,
)

comptime ZLOKR_TGT_ATTN = 1     # to_q/k/v/out
comptime ZLOKR_TGT_ALL = 2      # + feed_forward.w1/w3/w2


# (in,out) for a zimage slot — mirrors zimage_stack_lora _slot_in/_slot_out.
def zimage_lokr_slot_dims(slot: Int, D: Int, F: Int) -> Tuple[Int, Int]:
    var in_f = F if slot == SLOT_W2 else D
    var out_f = F if (slot == SLOT_W1 or slot == SLOT_W3) else D
    return (in_f, out_f)


def _zimage_slot_targeted(slot: Int, targets: Int) -> Bool:
    var is_attn = slot == SLOT_Q or slot == SLOT_K or slot == SLOT_V or slot == SLOT_O
    if is_attn:
        return targets >= ZLOKR_TGT_ATTN
    return targets >= ZLOKR_TGT_ALL


struct ZImageLoKrSet(Copyable, Movable):
    var ad: List[LoKrAdapter]   # (num_nr+num_cr+num_main) * 7 (dummy when inactive)
    var active: List[Bool]
    var num_nr: Int
    var num_cr: Int
    var num_main: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoKrAdapter], var active: List[Bool],
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


def empty_zimage_lokr_set() -> ZImageLoKrSet:
    return ZImageLoKrSet(List[LoKrAdapter](), List[Bool](), 0, 0, 0, 0)


def build_zimage_lokr_set(
    num_nr: Int, num_cr: Int, num_main: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, factor: Int,
    decompose_both: Bool, full_matrix: Bool, targets: Int, seed: UInt64,
) raises -> ZImageLoKrSet:
    if targets < ZLOKR_TGT_ATTN or targets > ZLOKR_TGT_ALL:
        raise Error("build_zimage_lokr_set: targets must be 1(attn)|2(all)")
    var ad = List[LoKrAdapter]()
    var active = List[Bool]()
    var s = seed
    var total_blocks = num_nr + num_cr + num_main
    for _b in range(total_blocks):
        for slot in range(ZIMAGE_SLOTS):
            if _zimage_slot_targeted(slot, targets):
                var dims = zimage_lokr_slot_dims(slot, D, F)
                ad.append(new_lokr_adapter(
                    dims[0], dims[1], rank, alpha, factor, s, decompose_both, full_matrix
                ))
                active.append(True)
            else:
                ad.append(_dummy_lokr())
                active.append(False)
            s += 1
    return ZImageLoKrSet(ad^, active^, num_nr, num_cr, num_main, rank)


def zimage_lokr_carrier_lists(set: ZImageLoKrSet, D: Int, F: Int) raises -> List[LoraAdapter]:
    var out = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(lokr_carrier_adapter(set.ad[i]))
        else:
            var dims = zimage_lokr_slot_dims(i % ZIMAGE_SLOTS, D, F)
            out.append(_inactive_carrier(dims[0], dims[1]))
    return out^


def zimage_lokr_carrier_total_bytes(set: ZImageLoKrSet) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = lokr_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            elems += set.ad[i].in_f + set.ad[i].out_f
    return elems * 2


struct ZImageLoKrGrads(Movable):
    var g: List[LoKrGrads]

    def __init__(out self, var g: List[LoKrGrads]):
        self.g = g^


def zimage_lokr_chain_all(
    set: ZImageLoKrSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> ZImageLoKrGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("zimage_lokr_chain_all: grad list count mismatch")
    var g = List[LoKrGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(lokr_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_lokr_grads())
    return ZImageLoKrGrads(g^)


def zimage_lokr_grad_norm(grads: ZImageLoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _grads_sqsum(grads.g[i])
    return sqrt(s)


def zimage_lokr_adamw_step(
    mut set: ZImageLoKrSet, grads: ZImageLoKrGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            lokr_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


# w2-side zero leg (starts EXACTLY 0; must be >0 after a real step).
def zimage_lokr_zero_leg_l1(set: ZImageLoKrSet) -> Float64:
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


# ── save: provisional zimage lycoris key convention (mirrors the LoRA prefix
# scheme noise_refiner./context_refiner./layers. + slot suffix; the lokr keys
# lokr_w1[_a/_b]/lokr_w2[_a/_b]/.alpha come from save_lokr_peft). No inference-
# side lycoris loader exists yet, so this convention is being SET here. ──────────
def _zimage_lokr_prefix(set: ZImageLoKrSet, block_idx: Int, slot: Int) -> String:
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


def save_zimage_lokr(set: ZImageLoKrSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoKr]()
    for bi in range(set.num_blocks()):
        for slot in range(ZIMAGE_SLOTS):
            var flat = bi * ZIMAGE_SLOTS + slot
            if set.active[flat]:
                named.append(NamedLoKr(_zimage_lokr_prefix(set, bi, slot), set.ad[flat].copy()))
    return save_lokr_peft(named, path, ctx)
