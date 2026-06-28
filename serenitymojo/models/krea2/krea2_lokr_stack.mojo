# models/krea2/krea2_lokr_stack.mojo — LoKr carrier dispatch for the krea2
# trainer. krea2's LoRA seam is _build_host_lora -> List[LoraAdapter] (28 blocks
# × 8 plain slots: wq/wk/wv/gate/wo + mlp.gate/up/down), consumed by
# krea2_stack_lora_forward/backward_streamed — exactly the (a,b) form the carrier
# produces. So the model-agnostic LoKr carrier core ports cleanly (mirrors the
# zimage port); only this per-model SET orchestration is new. No stack change.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lokr_adapter import (
    LoKrAdapter, LoKrGrads, new_lokr_adapter, lokr_adamw,
)
from serenitymojo.training.lokr_stack import (
    lokr_carrier_adapter, lokr_carrier_r_eff, lokr_chain_carrier_grads,
    _inactive_carrier, _dummy_lokr, _empty_lokr_grads, _grads_sqsum, _grads_scale,
)
from serenitymojo.training.lokr_save import NamedLoKr, save_lokr_peft

comptime KREA2_SLOTS = 8     # wq wk wv gate wo | mlp_gate mlp_up mlp_down
comptime K2LOKR_TGT_ATTN = 1 # slots 0-4
comptime K2LOKR_TGT_ALL = 2  # + slots 5-7 (mlp)


# (in,out) for slot s. D=hidden(FEATURES), F=mlp(MLPDIM), qdim=HEADS·HEADDIM,
# kvdim=KVHEADS·HEADDIM. Mirrors _build_host_lora in train_krea2.mojo.
def krea2_lokr_slot_dims(s: Int, D: Int, F: Int, qdim: Int, kvdim: Int) raises -> Tuple[Int, Int]:
    if s == 0:
        return (D, qdim)        # wq
    elif s == 1 or s == 2:
        return (D, kvdim)       # wk, wv
    elif s == 3 or s == 4:
        return (D, D)           # gate, wo
    elif s == 5 or s == 6:
        return (D, F)           # mlp gate, up
    elif s == 7:
        return (F, D)           # mlp down
    raise Error(String("krea2_lokr_slot_dims: bad slot ") + String(s))


def _krea2_slot_targeted(s: Int, targets: Int) -> Bool:
    if s <= 4:
        return targets >= K2LOKR_TGT_ATTN   # attn (incl. gate/wo)
    return targets >= K2LOKR_TGT_ALL        # mlp


struct Krea2LoKrSet(Copyable, Movable):
    var ad: List[LoKrAdapter]   # num_blocks * 8 (dummy when inactive)
    var active: List[Bool]
    var num_blocks: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoKrAdapter], var active: List[Bool],
        num_blocks: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.num_blocks = num_blocks
        self.rank = rank


def empty_krea2_lokr_set() -> Krea2LoKrSet:
    return Krea2LoKrSet(List[LoKrAdapter](), List[Bool](), 0, 0)


def build_krea2_lokr_set(
    num_blocks: Int, D: Int, F: Int, qdim: Int, kvdim: Int,
    rank: Int, alpha: Float32, factor: Int,
    decompose_both: Bool, full_matrix: Bool, targets: Int, seed: UInt64,
) raises -> Krea2LoKrSet:
    if targets < K2LOKR_TGT_ATTN or targets > K2LOKR_TGT_ALL:
        raise Error("build_krea2_lokr_set: targets must be 1(attn)|2(all)")
    var ad = List[LoKrAdapter]()
    var active = List[Bool]()
    var s = seed
    for _b in range(num_blocks):
        for slot in range(KREA2_SLOTS):
            if _krea2_slot_targeted(slot, targets):
                var dims = krea2_lokr_slot_dims(slot, D, F, qdim, kvdim)
                ad.append(new_lokr_adapter(
                    dims[0], dims[1], rank, alpha, factor, s, decompose_both, full_matrix
                ))
                active.append(True)
            else:
                ad.append(_dummy_lokr())
                active.append(False)
            s += 1
    return Krea2LoKrSet(ad^, active^, num_blocks, rank)


def krea2_lokr_carrier_lists(
    set: Krea2LoKrSet, D: Int, F: Int, qdim: Int, kvdim: Int
) raises -> List[LoraAdapter]:
    var out = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(lokr_carrier_adapter(set.ad[i]))
        else:
            var dims = krea2_lokr_slot_dims(i % KREA2_SLOTS, D, F, qdim, kvdim)
            out.append(_inactive_carrier(dims[0], dims[1]))
    return out^


def krea2_lokr_carrier_total_bytes(set: Krea2LoKrSet) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = lokr_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            elems += set.ad[i].in_f + set.ad[i].out_f
    return elems * 2


struct Krea2LoKrGrads(Movable):
    var g: List[LoKrGrads]

    def __init__(out self, var g: List[LoKrGrads]):
        self.g = g^


def krea2_lokr_chain_all(
    set: Krea2LoKrSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> Krea2LoKrGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("krea2_lokr_chain_all: grad list count mismatch")
    var g = List[LoKrGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(lokr_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_lokr_grads())
    return Krea2LoKrGrads(g^)


def krea2_lokr_grad_norm(grads: Krea2LoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _grads_sqsum(grads.g[i])
    return sqrt(s)


def krea2_lokr_clip_grads(mut grads: Krea2LoKrGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _grads_scale(grads.g[i], clip_scale)


def krea2_lokr_adamw_step(
    mut set: Krea2LoKrSet, grads: Krea2LoKrGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            lokr_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def krea2_lokr_zero_leg_l1(set: Krea2LoKrSet) -> Float64:
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


# save: krea2 module prefix (diffusion_model.blocks.<bi>.attn.{wq,wk,wv,gate,wo}/
# .mlp.{gate,up,down}) + lokr keys from save_lokr_peft. PROVISIONAL lokr key
# convention (matches the plain-LoRA krea2 prefix; ai-toolkit krea2 LoKr keys to
# be confirmed against a real ai-toolkit krea2 LoKr save).
def _krea2_lokr_prefix(bi: Int, slot: Int) raises -> String:
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
    raise Error(String("_krea2_lokr_prefix: bad slot ") + String(slot))


def save_krea2_lokr(set: Krea2LoKrSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoKr]()
    for bi in range(set.num_blocks):
        for slot in range(KREA2_SLOTS):
            var flat = bi * KREA2_SLOTS + slot
            if set.active[flat]:
                named.append(NamedLoKr(_krea2_lokr_prefix(bi, slot), set.ad[flat].copy()))
    return save_lokr_peft(named, path, ctx)
