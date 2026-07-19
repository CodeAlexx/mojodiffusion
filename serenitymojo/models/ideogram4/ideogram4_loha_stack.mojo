# models/ideogram4/ideogram4_loha_stack.mojo — LoHa carrier dispatch for the
# ideogram4 trainer (twin of ideogram4_lokr_stack.mojo; mirrors
# zimage_loha_stack.mojo). LoHa has no factorization variants: one LoHaAdapter
# per active slot, carrier r_eff = rank², w2a is the zero-leg. Reuses the
# model-agnostic LoHa carrier core + ideogram4's slot geometry + the SAME
# host<->device bridge as the LoKr stack (ideogram4's LoRA is device-resident —
# see the ideogram4_lokr_stack.mojo header for why the bridge exists).

from std.collections import List
from std.math import sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.training.train_step import LoraAdapter as CarrierLoRA
from serenitymojo.training.loha_adapter import (
    LoHaAdapter, LoHaGrads, new_loha_adapter, loha_adamw,
)
from serenitymojo.training.loha_stack import (
    loha_carrier_adapter, loha_carrier_r_eff, loha_chain_carrier_grads,
    _dummy_loha, _empty_loha_grads, _loha_grads_sqsum, _loha_grads_scale,
)
from serenitymojo.training.loha_save import NamedLoHa, save_loha_peft
from serenitymojo.models.ideogram4.lora_module import LoraAdapter
from serenitymojo.models.ideogram4.block import (
    Ideogram4LoraSet, Ideogram4StackLoraGrads,
    I4_SLOT_QKV, I4_SLOT_O, I4_SLOT_W1, I4_SLOT_W2, I4_SLOT_W3, I4_SLOT_ADALN,
    I4_SLOTS_PER_BLOCK,
)
from serenitymojo.models.ideogram4.ideogram4_lokr_stack import (
    ideogram4_lokr_slot_dims, _ideogram4_slot_targeted,
)


comptime LArc = ArcPointer[LoraAdapter]

comptime I4LOKR_TGT_ATTN = 1     # slots qkv(0), o(1)
comptime I4LOKR_TGT_ALL = 2      # + w1(2)/w2(3)/w3(4)/adaln(5)


struct Ideogram4LoHaSet(Copyable, Movable):
    var ad: List[LoHaAdapter]   # n_layers * I4_SLOTS_PER_BLOCK (dummy when inactive)
    var active: List[Bool]
    var n_layers: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoHaAdapter], var active: List[Bool],
        n_layers: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.n_layers = n_layers
        self.rank = rank


def empty_ideogram4_loha_set() -> Ideogram4LoHaSet:
    return Ideogram4LoHaSet(List[LoHaAdapter](), List[Bool](), 0, 0)


def build_ideogram4_loha_set(
    n_layers: Int, H: Int, F: Int, A: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
) raises -> Ideogram4LoHaSet:
    if targets < I4LOKR_TGT_ATTN or targets > I4LOKR_TGT_ALL:
        raise Error("build_ideogram4_loha_set: targets must be 1(attn)|2(all)")
    var ad = List[LoHaAdapter]()
    var active = List[Bool]()
    var s = seed
    for _b in range(n_layers):
        for slot in range(I4_SLOTS_PER_BLOCK):
            if _ideogram4_slot_targeted(slot, targets):
                var dims = ideogram4_lokr_slot_dims(slot, H, F, A)
                ad.append(new_loha_adapter(dims[0], dims[1], rank, alpha, s))
                active.append(True)
            else:
                ad.append(_dummy_loha())
                active.append(False)
            s += 1
    return Ideogram4LoHaSet(ad^, active^, n_layers, rank)


def ideogram4_loha_carrier_total_bytes(set: Ideogram4LoHaSet, H: Int, F: Int, A: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = loha_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = ideogram4_lokr_slot_dims(i % I4_SLOTS_PER_BLOCK, H, F, A)
            elems += dims[0] + dims[1]
    return elems * 2


# ── host carrier -> device ideogram4 LoRA adapter (scale()==1.0) ─────────────
def _carrier_to_device(c: CarrierLoRA, ctx: DeviceContext) raises -> LArc:
    var a_sh = List[Int](); a_sh.append(c.rank); a_sh.append(c.in_f)   # [r_eff, in]
    var b_sh = List[Int](); b_sh.append(c.out_f); b_sh.append(c.rank)  # [out, r_eff]
    var a = Tensor.from_host_bf16(c.a.copy(), a_sh^, ctx)
    var b = Tensor.from_host_bf16(c.b.copy(), b_sh^, ctx)
    return LArc(LoraAdapter(a^, b^, c.rank, Float32(c.rank)))


def _inactive_device(in_f: Int, out_f: Int, ctx: DeviceContext) raises -> LArc:
    var a_sh = List[Int](); a_sh.append(1); a_sh.append(in_f)
    var b_sh = List[Int](); b_sh.append(out_f); b_sh.append(1)
    var a = zeros_device(a_sh^, STDtype.BF16, ctx)
    var b = zeros_device(b_sh^, STDtype.BF16, ctx)
    return LArc(LoraAdapter(a^, b^, 1, Float32(1.0)))


def ideogram4_loha_carrier_device_set(
    set: Ideogram4LoHaSet, H: Int, F: Int, A: Int, ctx: DeviceContext
) raises -> Ideogram4LoraSet:
    var ad = List[LArc]()
    for i in range(len(set.ad)):
        if set.active[i]:
            var c = loha_carrier_adapter(set.ad[i])
            ad.append(_carrier_to_device(c, ctx))
        else:
            var dims = ideogram4_lokr_slot_dims(i % I4_SLOTS_PER_BLOCK, H, F, A)
            ad.append(_inactive_device(dims[0], dims[1], ctx))
    return Ideogram4LoraSet(ad^, set.n_layers, set.rank)


struct Ideogram4LoHaGrads(Movable):
    var g: List[LoHaGrads]

    def __init__(out self, var g: List[LoHaGrads]):
        self.g = g^


def ideogram4_loha_chain_all(
    set: Ideogram4LoHaSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> Ideogram4LoHaGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("ideogram4_loha_chain_all: grad list count mismatch")
    var g = List[LoHaGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(loha_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_loha_grads())
    return Ideogram4LoHaGrads(g^)


def ideogram4_loha_chain_from_device(
    set: Ideogram4LoHaSet, grads: Ideogram4StackLoraGrads, ctx: DeviceContext
) raises -> Ideogram4LoHaGrads:
    if len(grads.d_a) != len(set.ad) or len(grads.d_b) != len(set.ad):
        raise Error("ideogram4_loha_chain_from_device: device grad count mismatch")
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(len(set.ad)):
        if set.active[i]:
            d_a.append(grads.d_a[i][].to_host(ctx))
            d_b.append(grads.d_b[i][].to_host(ctx))
        else:
            d_a.append(List[Float32]())
            d_b.append(List[Float32]())
    return ideogram4_loha_chain_all(set, d_a, d_b)


def ideogram4_loha_grad_norm(grads: Ideogram4LoHaGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _loha_grads_sqsum(grads.g[i])
    return sqrt(s)


def ideogram4_loha_clip_grads(mut grads: Ideogram4LoHaGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _loha_grads_scale(grads.g[i], clip_scale)


def ideogram4_loha_adamw_step(
    mut set: Ideogram4LoHaSet, grads: Ideogram4LoHaGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            loha_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


# w2a is the LoHa zero-leg (starts EXACTLY 0; must be >0 after a real step).
def ideogram4_loha_zero_leg_l1(set: Ideogram4LoHaSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def _ideogram4_loha_prefix(layer: Int, slot: Int) raises -> String:
    var b = String("transformer.layers.") + String(layer) + String(".")
    if slot == I4_SLOT_QKV:
        return b + "attention.qkv"
    elif slot == I4_SLOT_O:
        return b + "attention.o"
    elif slot == I4_SLOT_W1:
        return b + "feed_forward.w1"
    elif slot == I4_SLOT_W2:
        return b + "feed_forward.w2"
    elif slot == I4_SLOT_W3:
        return b + "feed_forward.w3"
    elif slot == I4_SLOT_ADALN:
        return b + "adaln_modulation"
    raise Error(String("_ideogram4_loha_prefix: bad slot ") + String(slot))


def save_ideogram4_loha(set: Ideogram4LoHaSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoHa]()
    for layer in range(set.n_layers):
        for slot in range(I4_SLOTS_PER_BLOCK):
            var flat = layer * I4_SLOTS_PER_BLOCK + slot
            if set.active[flat]:
                named.append(NamedLoHa(_ideogram4_loha_prefix(layer, slot), set.ad[flat].copy()))
    return save_loha_peft(named, path, ctx)
