# models/anima/anima_lycoris_stack.mojo -- LoKr/LoHa carrier dispatch for Anima.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
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
from serenitymojo.training.dora_adapter import (
    DoRAAdapter, DoRAGrads, new_dora_adapter, dora_adamw,
)
from serenitymojo.training.dora_stack import (
    dora_carrier_adapter, dora_chain_carrier_grads,
)
from serenitymojo.training.dora_save import NamedDoRA, save_dora_onetrainer
from serenitymojo.training.oft_stack import (
    oft_ot_carrier_adapter, oft_ot_chain_carrier_grads,
)
from serenitymojo.models.anima.lora_block import (
    ANIMA_SLOTS,
    SLOT_SA_Q, SLOT_SA_K, SLOT_SA_V, SLOT_SA_O,
    SLOT_CA_Q, SLOT_CA_K, SLOT_CA_V, SLOT_CA_O, SLOT_MLP1, SLOT_MLP2,
)
from serenitymojo.models.anima.anima_stack_lora import AnimaLoraSet


comptime ANIMA_LYCORIS_TGT_ATTN = 1
comptime ANIMA_LYCORIS_TGT_ALL = 2


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def anima_lycoris_slot_dims(slot: Int, D: Int, JOINT: Int, F: Int) raises -> Tuple[Int, Int]:
    var s = slot % ANIMA_SLOTS
    if s == SLOT_CA_K or s == SLOT_CA_V:
        return (JOINT, D)
    if s == SLOT_MLP1:
        return (D, F)
    if s == SLOT_MLP2:
        return (F, D)
    return (D, D)


def _anima_slot_is_attn(slot: Int) -> Bool:
    var s = slot % ANIMA_SLOTS
    return s != SLOT_MLP1 and s != SLOT_MLP2


def _anima_slot_targeted(slot: Int, targets: Int) -> Bool:
    if _anima_slot_is_attn(slot):
        return targets >= ANIMA_LYCORIS_TGT_ATTN
    return targets >= ANIMA_LYCORIS_TGT_ALL


def anima_full_delta_carrier_bytes_estimate(
    num_blocks: Int, D: Int, JOINT: Int, F: Int, targets: Int,
) raises -> Int:
    if targets < ANIMA_LYCORIS_TGT_ATTN or targets > ANIMA_LYCORIS_TGT_ALL:
        raise Error("anima_full_delta_carrier_bytes_estimate: targets must be 1(attn)|2(all)")
    var elems = 0
    for _bi in range(num_blocks):
        for slot in range(ANIMA_SLOTS):
            var dims = anima_lycoris_slot_dims(slot, D, JOINT, F)
            if _anima_slot_targeted(slot, targets):
                elems += dims[0] * dims[0] + dims[1] * dims[0]
            else:
                elems += dims[0] + dims[1]
    return elems * 2


def anima_full_delta_preflight(
    num_blocks: Int, D: Int, JOINT: Int, F: Int,
    targets: Int, budget_bytes: Int,
) raises -> Int:
    var b = anima_full_delta_carrier_bytes_estimate(num_blocks, D, JOINT, F, targets)
    if b > budget_bytes:
        raise Error(
            String("Anima full-delta carrier needs ") + String(b)
            + String(" bytes (> budget ") + String(budget_bytes)
            + String("). Use attention-only targets or the direct W_eff path.")
        )
    return b


def _anima_slot_factor(slot: Int, factor: Int, factor_attn: Int, factor_ff: Int) -> Int:
    if _anima_slot_is_attn(slot):
        return factor_attn if factor_attn != 0 else factor
    return factor_ff if factor_ff != 0 else factor


def _anima_ot_module(slot: Int) -> String:
    if slot == SLOT_SA_Q:
        return String("attn1.to_q")
    if slot == SLOT_SA_K:
        return String("attn1.to_k")
    if slot == SLOT_SA_V:
        return String("attn1.to_v")
    if slot == SLOT_SA_O:
        return String("attn1.to_out.0")
    if slot == SLOT_CA_Q:
        return String("attn2.to_q")
    if slot == SLOT_CA_K:
        return String("attn2.to_k")
    if slot == SLOT_CA_V:
        return String("attn2.to_v")
    if slot == SLOT_CA_O:
        return String("attn2.to_out.0")
    if slot == SLOT_MLP1:
        return String("ff.net.0.proj")
    return String("ff.net.2")


def _anima_prefix(block_idx: Int, slot: Int) -> String:
    return (
        String("transformer.transformer_blocks.") + String(block_idx)
        + String(".") + _anima_ot_module(slot)
    )


struct AnimaLoKrSet(Copyable, Movable):
    var ad: List[LoKrAdapter]
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


def empty_anima_lokr_set() -> AnimaLoKrSet:
    return AnimaLoKrSet(List[LoKrAdapter](), List[Bool](), 0, 0)


def build_anima_lokr_set(
    num_blocks: Int, D: Int, JOINT: Int, F: Int,
    rank: Int, alpha: Float32,
    factor: Int, factor_attn: Int, factor_ff: Int,
    decompose_both: Bool, full_matrix: Bool,
    targets: Int, seed: UInt64,
) raises -> AnimaLoKrSet:
    if targets < ANIMA_LYCORIS_TGT_ATTN or targets > ANIMA_LYCORIS_TGT_ALL:
        raise Error("build_anima_lokr_set: targets must be 1(attn)|2(all)")
    var ad = List[LoKrAdapter]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_blocks):
        for slot in range(ANIMA_SLOTS):
            if _anima_slot_targeted(slot, targets):
                var dims = anima_lycoris_slot_dims(slot, D, JOINT, F)
                var f = _anima_slot_factor(slot, factor, factor_attn, factor_ff)
                ad.append(new_lokr_adapter(
                    dims[0], dims[1], rank, alpha, f, s,
                    decompose_both, full_matrix,
                ))
                active.append(True)
            else:
                ad.append(_dummy_lokr())
                active.append(False)
            s += 1
    return AnimaLoKrSet(ad^, active^, num_blocks, rank)


def anima_lokr_carrier_set(set: AnimaLoKrSet, D: Int, JOINT: Int, F: Int) raises -> AnimaLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(lokr_carrier_adapter(set.ad[i]))
        else:
            var dims = anima_lycoris_slot_dims(i % ANIMA_SLOTS, D, JOINT, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return AnimaLoraSet(ad^, set.num_blocks, set.rank)


def anima_lokr_carrier_total_bytes(set: AnimaLoKrSet, D: Int, JOINT: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = lokr_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = anima_lycoris_slot_dims(i % ANIMA_SLOTS, D, JOINT, F)
            elems += dims[0] + dims[1]
    return elems * 2


struct AnimaLoKrGrads(Movable):
    var g: List[LoKrGrads]

    def __init__(out self, var g: List[LoKrGrads]):
        self.g = g^


def anima_lokr_chain_all(
    set: AnimaLoKrSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> AnimaLoKrGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("anima_lokr_chain_all: grad list count mismatch")
    var g = List[LoKrGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(lokr_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_lokr_grads())
    return AnimaLoKrGrads(g^)


def anima_lokr_grad_norm(grads: AnimaLoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _grads_sqsum(grads.g[i])
    return sqrt(s)


def anima_lokr_clip_grads(mut grads: AnimaLoKrGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _grads_scale(grads.g[i], clip_scale)


def anima_lokr_adamw_step(
    mut set: AnimaLoKrSet, grads: AnimaLoKrGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            lokr_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def anima_lokr_zero_leg_l1(set: AnimaLoKrSet) -> Float64:
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


def save_anima_lokr(set: AnimaLoKrSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoKr]()
    for bi in range(set.num_blocks):
        for slot in range(ANIMA_SLOTS):
            var flat = bi * ANIMA_SLOTS + slot
            if set.active[flat]:
                named.append(NamedLoKr(_anima_prefix(bi, slot), set.ad[flat].copy()))
    return save_lokr_peft(named, path, ctx)


struct AnimaLoHaSet(Copyable, Movable):
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


def empty_anima_loha_set() -> AnimaLoHaSet:
    return AnimaLoHaSet(List[LoHaAdapter](), List[Bool](), 0, 0)


def build_anima_loha_set(
    num_blocks: Int, D: Int, JOINT: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
) raises -> AnimaLoHaSet:
    if targets < ANIMA_LYCORIS_TGT_ATTN or targets > ANIMA_LYCORIS_TGT_ALL:
        raise Error("build_anima_loha_set: targets must be 1(attn)|2(all)")
    var ad = List[LoHaAdapter]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_blocks):
        for slot in range(ANIMA_SLOTS):
            if _anima_slot_targeted(slot, targets):
                var dims = anima_lycoris_slot_dims(slot, D, JOINT, F)
                ad.append(new_loha_adapter(dims[0], dims[1], rank, alpha, s))
                active.append(True)
            else:
                ad.append(_dummy_loha())
                active.append(False)
            s += 1
    return AnimaLoHaSet(ad^, active^, num_blocks, rank)


def anima_loha_carrier_set(set: AnimaLoHaSet, D: Int, JOINT: Int, F: Int) raises -> AnimaLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(loha_carrier_adapter(set.ad[i]))
        else:
            var dims = anima_lycoris_slot_dims(i % ANIMA_SLOTS, D, JOINT, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return AnimaLoraSet(ad^, set.num_blocks, set.rank)


def anima_loha_carrier_total_bytes(set: AnimaLoHaSet, D: Int, JOINT: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = loha_carrier_r_eff(set.ad[i])
            elems += r * set.ad[i].in_f + set.ad[i].out_f * r
        else:
            var dims = anima_lycoris_slot_dims(i % ANIMA_SLOTS, D, JOINT, F)
            elems += dims[0] + dims[1]
    return elems * 2


struct AnimaLoHaGrads(Movable):
    var g: List[LoHaGrads]

    def __init__(out self, var g: List[LoHaGrads]):
        self.g = g^


def anima_loha_chain_all(
    set: AnimaLoHaSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> AnimaLoHaGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("anima_loha_chain_all: grad list count mismatch")
    var g = List[LoHaGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(loha_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_loha_grads())
    return AnimaLoHaGrads(g^)


def anima_loha_grad_norm(grads: AnimaLoHaGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _loha_grads_sqsum(grads.g[i])
    return sqrt(s)


def anima_loha_clip_grads(mut grads: AnimaLoHaGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _loha_grads_scale(grads.g[i], clip_scale)


def anima_loha_adamw_step(
    mut set: AnimaLoHaSet, grads: AnimaLoHaGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            loha_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def anima_loha_zero_leg_l1(set: AnimaLoHaSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def save_anima_loha(set: AnimaLoHaSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoHa]()
    for bi in range(set.num_blocks):
        for slot in range(ANIMA_SLOTS):
            var flat = bi * ANIMA_SLOTS + slot
            if set.active[flat]:
                named.append(NamedLoHa(_anima_prefix(bi, slot), set.ad[flat].copy()))
    return save_loha_peft(named, path, ctx)


# ── DoRA full-delta carrier ──────────────────────────────────────────────────
struct AnimaDoRASet(Copyable, Movable):
    var ad: List[DoRAAdapter]
    var w: List[List[Float32]]
    var active: List[Bool]
    var num_blocks: Int
    var rank: Int

    def __init__(
        out self, var ad: List[DoRAAdapter], var w: List[List[Float32]],
        var active: List[Bool], num_blocks: Int, rank: Int,
    ):
        self.ad = ad^
        self.w = w^
        self.active = active^
        self.num_blocks = num_blocks
        self.rank = rank


def empty_anima_dora_set() raises -> AnimaDoRASet:
    return AnimaDoRASet(List[DoRAAdapter](), List[List[Float32]](), List[Bool](), 0, 0)


def _dummy_dora() raises -> DoRAAdapter:
    var w = List[Float32]()
    w.append(Float32(1.0))
    return new_dora_adapter(w^, 1, 1, 1, Float32(1.0), 0)


def _synth_w(out_f: Int, in_f: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(out_f * in_f):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * Float32(0.5))
    return out^


def _anima_weight_key(block_idx: Int, slot: Int) raises -> String:
    var bp = String("net.blocks.") + String(block_idx) + String(".")
    if slot == SLOT_SA_Q:
        return bp + String("self_attn.q_proj.weight")
    if slot == SLOT_SA_K:
        return bp + String("self_attn.k_proj.weight")
    if slot == SLOT_SA_V:
        return bp + String("self_attn.v_proj.weight")
    if slot == SLOT_SA_O:
        return bp + String("self_attn.output_proj.weight")
    if slot == SLOT_CA_Q:
        return bp + String("cross_attn.q_proj.weight")
    if slot == SLOT_CA_K:
        return bp + String("cross_attn.k_proj.weight")
    if slot == SLOT_CA_V:
        return bp + String("cross_attn.v_proj.weight")
    if slot == SLOT_CA_O:
        return bp + String("cross_attn.output_proj.weight")
    if slot == SLOT_MLP1:
        return bp + String("mlp.layer1.weight")
    if slot == SLOT_MLP2:
        return bp + String("mlp.layer2.weight")
    raise Error(String("_anima_weight_key: bad slot ") + String(slot))


def _read_anima_weight_f32(
    st: SafeTensors, block_idx: Int, slot: Int, D: Int, JOINT: Int, F: Int,
) raises -> List[Float32]:
    var key = _anima_weight_key(block_idx, slot)
    var dims = anima_lycoris_slot_dims(slot, D, JOINT, F)
    var info = st.tensor_info(key)
    if info.dtype != STDtype.BF16:
        raise Error(String("Anima LyCORIS base weight: expected BF16 for ") + key)
    if len(info.shape) != 2:
        raise Error(String("Anima LyCORIS base weight: expected 2D for ") + key)
    if Int(info.shape[0]) != dims[1] or Int(info.shape[1]) != dims[0]:
        raise Error(String("Anima LyCORIS base weight: shape mismatch for ") + key)
    var bytes = st.tensor_bytes(key)
    var bp = bytes.unsafe_ptr().bitcast[BFloat16]()
    var out = List[Float32]()
    for i in range(dims[0] * dims[1]):
        out.append(bp[i].cast[DType.float32]())
    return out^


def build_anima_dora_set(
    num_blocks: Int, D: Int, JOINT: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool = False,
) raises -> AnimaDoRASet:
    if targets < ANIMA_LYCORIS_TGT_ATTN or targets > ANIMA_LYCORIS_TGT_ALL:
        raise Error("build_anima_dora_set: targets must be 1(attn)|2(all)")
    var ad = List[DoRAAdapter]()
    var weights = List[List[Float32]]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_blocks):
        for slot in range(ANIMA_SLOTS):
            if _anima_slot_targeted(slot, targets):
                var dims = anima_lycoris_slot_dims(slot, D, JOINT, F)
                var w = _synth_w(dims[1], dims[0], s * 131 + 7)
                ad.append(new_dora_adapter(
                    w.copy(), dims[0], dims[1], rank, alpha, s,
                    Float32(1.0e-7), wd_on_out,
                ))
                weights.append(w^)
                active.append(True)
            else:
                ad.append(_dummy_dora())
                var w1 = List[Float32]()
                w1.append(Float32(1.0))
                weights.append(w1^)
                active.append(False)
            s += 1
    return AnimaDoRASet(ad^, weights^, active^, num_blocks, rank)


def build_anima_dora_set_from_checkpoint(
    st: SafeTensors, num_blocks: Int, D: Int, JOINT: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool = False,
) raises -> AnimaDoRASet:
    if targets < ANIMA_LYCORIS_TGT_ATTN or targets > ANIMA_LYCORIS_TGT_ALL:
        raise Error("build_anima_dora_set_from_checkpoint: targets must be 1(attn)|2(all)")
    var ad = List[DoRAAdapter]()
    var weights = List[List[Float32]]()
    var active = List[Bool]()
    var s = seed
    for bi in range(num_blocks):
        for slot in range(ANIMA_SLOTS):
            if _anima_slot_targeted(slot, targets):
                var dims = anima_lycoris_slot_dims(slot, D, JOINT, F)
                var w = _read_anima_weight_f32(st, bi, slot, D, JOINT, F)
                ad.append(new_dora_adapter(
                    w.copy(), dims[0], dims[1], rank, alpha, s,
                    Float32(1.0e-7), wd_on_out,
                ))
                weights.append(w^)
                active.append(True)
            else:
                ad.append(_dummy_dora())
                var w1 = List[Float32]()
                w1.append(Float32(1.0))
                weights.append(w1^)
                active.append(False)
            s += 1
    return AnimaDoRASet(ad^, weights^, active^, num_blocks, rank)


def anima_dora_carrier_set(set: AnimaDoRASet, D: Int, JOINT: Int, F: Int) raises -> AnimaLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ad.append(dora_carrier_adapter(set.ad[i], set.w[i]))
        else:
            var dims = anima_lycoris_slot_dims(i % ANIMA_SLOTS, D, JOINT, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return AnimaLoraSet(ad^, set.num_blocks, set.rank)


def anima_dora_carrier_total_bytes(set: AnimaDoRASet, D: Int, JOINT: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var inf = set.ad[i].in_f
            elems += inf * inf + set.ad[i].out_f * inf
        else:
            var dims = anima_lycoris_slot_dims(i % ANIMA_SLOTS, D, JOINT, F)
            elems += dims[0] + dims[1]
    return elems * 2


def anima_dora_preflight(set: AnimaDoRASet, D: Int, JOINT: Int, F: Int, budget_bytes: Int) raises:
    var b = anima_dora_carrier_total_bytes(set, D, JOINT, F)
    if b > budget_bytes:
        raise Error(
            String("Anima DoRA carrier needs ") + String(b)
            + String(" bytes (> budget ") + String(budget_bytes)
            + String("). Use attention-only targets or the direct W_eff path.")
        )


struct AnimaDoRAGrads(Movable):
    var g: List[DoRAGrads]

    def __init__(out self, var g: List[DoRAGrads]):
        self.g = g^


def _empty_dora_grads() -> DoRAGrads:
    return DoRAGrads(List[Float32](), List[Float32](), List[Float32](), List[Float32]())


def anima_dora_chain_all(
    set: AnimaDoRASet, d_a: List[List[Float32]], d_b: List[List[Float32]],
) raises -> AnimaDoRAGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("anima_dora_chain_all: grad list count mismatch")
    var out = List[DoRAGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            out.append(dora_chain_carrier_grads(set.ad[i], set.w[i], d_a[i], d_b[i]))
        else:
            out.append(_empty_dora_grads())
    return AnimaDoRAGrads(out^)


def _dora_grads_sqsum(g: DoRAGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.d_a)):
        s += Float64(g.d_a[i]) * Float64(g.d_a[i])
    for i in range(len(g.d_b)):
        s += Float64(g.d_b[i]) * Float64(g.d_b[i])
    for i in range(len(g.d_m)):
        s += Float64(g.d_m[i]) * Float64(g.d_m[i])
    return s


def anima_dora_grad_norm(grads: AnimaDoRAGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _dora_grads_sqsum(grads.g[i])
    return sqrt(s)


def _dora_grads_scale(mut g: DoRAGrads, scale: Float32):
    for i in range(len(g.d_a)):
        g.d_a[i] *= scale
    for i in range(len(g.d_b)):
        g.d_b[i] *= scale
    for i in range(len(g.d_m)):
        g.d_m[i] *= scale


def anima_dora_clip_grads(mut grads: AnimaDoRAGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _dora_grads_scale(grads.g[i], clip_scale)


def anima_dora_adamw_step(
    mut set: AnimaDoRASet, grads: AnimaDoRAGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            dora_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


def anima_dora_zero_leg_l1(set: AnimaDoRASet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref lo = set.ad[i]
        for j in range(len(lo.b)):
            var v = Float64(lo.b[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def save_anima_dora(set: AnimaDoRASet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedDoRA]()
    for bi in range(set.num_blocks):
        for slot in range(ANIMA_SLOTS):
            var flat = bi * ANIMA_SLOTS + slot
            if set.active[flat]:
                named.append(NamedDoRA(_anima_prefix(bi, slot), set.ad[flat].copy()))
    return save_dora_onetrainer(named, path, ctx)


# ── OneTrainer OFT full-delta carrier ────────────────────────────────────────
struct AnimaOFTSlot(Copyable, Movable):
    var vec: List[Float32]
    var w: List[Float32]
    var in_f: Int
    var out_f: Int
    var b: Int
    var r: Int
    var m: List[Float32]
    var v: List[Float32]

    def __init__(
        out self, var vec: List[Float32], var w: List[Float32],
        in_f: Int, out_f: Int, b: Int, r: Int,
        var m: List[Float32], var v: List[Float32],
    ):
        self.vec = vec^
        self.w = w^
        self.in_f = in_f
        self.out_f = out_f
        self.b = b
        self.r = r
        self.m = m^
        self.v = v^


struct AnimaOFTSet(Copyable, Movable):
    var ad: List[AnimaOFTSlot]
    var active: List[Bool]
    var num_blocks: Int
    var block_size: Int

    def __init__(
        out self, var ad: List[AnimaOFTSlot], var active: List[Bool],
        num_blocks: Int, block_size: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.num_blocks = num_blocks
        self.block_size = block_size


def empty_anima_oft_set() -> AnimaOFTSet:
    return AnimaOFTSet(List[AnimaOFTSlot](), List[Bool](), 0, 0)


def _dummy_oft_slot() -> AnimaOFTSlot:
    return AnimaOFTSlot(_zeros(1), _zeros(1), 1, 1, 1, 1, _zeros(1), _zeros(1))


def _make_oft_slot_with_w(
    var w: List[Float32], in_f: Int, out_f: Int, block_size: Int,
) raises -> AnimaOFTSlot:
    if in_f % block_size != 0:
        raise Error(String("Anima OFT: in_f ") + String(in_f) + String(" not divisible by block_size ") + String(block_size))
    if len(w) != out_f * in_f:
        raise Error("Anima OFT: base weight numel mismatch")
    var b = block_size
    var r = in_f // b
    var ne = b * (b - 1) // 2
    return AnimaOFTSlot(_zeros(r * ne), w^, in_f, out_f, b, r, _zeros(r * ne), _zeros(r * ne))


def build_anima_oft_set(
    num_blocks: Int, D: Int, JOINT: Int, F: Int,
    block_size: Int, targets: Int, seed: UInt64,
) raises -> AnimaOFTSet:
    if targets < ANIMA_LYCORIS_TGT_ATTN or targets > ANIMA_LYCORIS_TGT_ALL:
        raise Error("build_anima_oft_set: targets must be 1(attn)|2(all)")
    var ad = List[AnimaOFTSlot]()
    var active = List[Bool]()
    var s = seed
    for _bi in range(num_blocks):
        for slot in range(ANIMA_SLOTS):
            if _anima_slot_targeted(slot, targets):
                var dims = anima_lycoris_slot_dims(slot, D, JOINT, F)
                var w = _synth_w(dims[1], dims[0], s * 131 + 7)
                ad.append(_make_oft_slot_with_w(w^, dims[0], dims[1], block_size))
                active.append(True)
            else:
                ad.append(_dummy_oft_slot())
                active.append(False)
            s += 1
    return AnimaOFTSet(ad^, active^, num_blocks, block_size)


def build_anima_oft_set_from_checkpoint(
    st: SafeTensors, num_blocks: Int, D: Int, JOINT: Int, F: Int,
    block_size: Int, targets: Int,
) raises -> AnimaOFTSet:
    if targets < ANIMA_LYCORIS_TGT_ATTN or targets > ANIMA_LYCORIS_TGT_ALL:
        raise Error("build_anima_oft_set_from_checkpoint: targets must be 1(attn)|2(all)")
    var ad = List[AnimaOFTSlot]()
    var active = List[Bool]()
    for bi in range(num_blocks):
        for slot in range(ANIMA_SLOTS):
            if _anima_slot_targeted(slot, targets):
                var dims = anima_lycoris_slot_dims(slot, D, JOINT, F)
                var w = _read_anima_weight_f32(st, bi, slot, D, JOINT, F)
                ad.append(_make_oft_slot_with_w(w^, dims[0], dims[1], block_size))
                active.append(True)
            else:
                ad.append(_dummy_oft_slot())
                active.append(False)
    return AnimaOFTSet(ad^, active^, num_blocks, block_size)


def anima_oft_carrier_set(set: AnimaOFTSet, D: Int, JOINT: Int, F: Int) raises -> AnimaLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ref sl = set.ad[i]
            ad.append(oft_ot_carrier_adapter(sl.vec, sl.w, sl.in_f, sl.out_f, sl.b, sl.r))
        else:
            var dims = anima_lycoris_slot_dims(i % ANIMA_SLOTS, D, JOINT, F)
            ad.append(_inactive_carrier(dims[0], dims[1]))
    return AnimaLoraSet(ad^, set.num_blocks, set.block_size)


def anima_oft_carrier_total_bytes(set: AnimaOFTSet, D: Int, JOINT: Int, F: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var inf = set.ad[i].in_f
            elems += inf * inf + set.ad[i].out_f * inf
        else:
            var dims = anima_lycoris_slot_dims(i % ANIMA_SLOTS, D, JOINT, F)
            elems += dims[0] + dims[1]
    return elems * 2


def anima_oft_preflight(set: AnimaOFTSet, D: Int, JOINT: Int, F: Int, budget_bytes: Int) raises:
    var b = anima_oft_carrier_total_bytes(set, D, JOINT, F)
    if b > budget_bytes:
        raise Error(
            String("Anima OFT carrier needs ") + String(b)
            + String(" bytes (> budget ") + String(budget_bytes)
            + String("). Use attention-only targets or the direct W_eff path.")
        )


struct AnimaOFTGrads(Movable):
    var g: List[List[Float32]]

    def __init__(out self, var g: List[List[Float32]]):
        self.g = g^


def anima_oft_chain_all(
    set: AnimaOFTSet, d_a: List[List[Float32]], d_b: List[List[Float32]],
) raises -> AnimaOFTGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("anima_oft_chain_all: grad list count mismatch")
    var out = List[List[Float32]]()
    for i in range(len(set.ad)):
        if set.active[i]:
            ref sl = set.ad[i]
            out.append(oft_ot_chain_carrier_grads(
                sl.vec, sl.w, d_a[i], d_b[i], sl.in_f, sl.out_f, sl.b, sl.r,
            ))
        else:
            out.append(List[Float32]())
    return AnimaOFTGrads(out^)


def _vec_sqsum(g: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g)):
        s += Float64(g[i]) * Float64(g[i])
    return s


def anima_oft_grad_norm(grads: AnimaOFTGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _vec_sqsum(grads.g[i])
    return sqrt(s)


def _vec_scale(mut g: List[Float32], scale: Float32):
    for i in range(len(g)):
        g[i] *= scale


def anima_oft_clip_grads(mut grads: AnimaOFTGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _vec_scale(grads.g[i], clip_scale)


def _oft_vec_adamw(
    mut vec: List[Float32], d_vec: List[Float32],
    mut m: List[Float32], mut v: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    if len(vec) != len(d_vec) or len(vec) != len(m) or len(vec) != len(v):
        raise Error("Anima OFT AdamW: len mismatch")
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    for i in range(len(vec)):
        var gv = d_vec[i]
        var mi = beta1 * m[i] + (Float32(1.0) - beta1) * gv
        var vi = beta2 * v[i] + (Float32(1.0) - beta2) * gv * gv
        m[i] = mi
        v[i] = vi
        var pv = vec[i]
        if weight_decay > 0.0:
            pv = pv * (Float32(1.0) - lr * weight_decay)
        vec[i] = pv - lr * (mi / bc1) / (sqrt(vi / bc2) + eps)


def anima_oft_adamw_step(
    mut set: AnimaOFTSet, grads: AnimaOFTGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            _oft_vec_adamw(
                set.ad[i].vec, grads.g[i], set.ad[i].m, set.ad[i].v,
                t, lr, beta1, beta2, eps, weight_decay,
            )


def anima_oft_vec_l1(set: AnimaOFTSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        if not set.active[i]:
            continue
        ref sl = set.ad[i]
        for j in range(len(sl.vec)):
            var v = Float64(sl.vec[j])
            s += v if v >= 0.0 else -v
    return s


def _f32_2d(var values: List[Float32], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


def save_anima_oft(set: AnimaOFTSet, path: String, ctx: DeviceContext) raises -> Int:
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    var nmods = 0
    for bi in range(set.num_blocks):
        for slot in range(ANIMA_SLOTS):
            var flat = bi * ANIMA_SLOTS + slot
            if not set.active[flat]:
                continue
            ref sl = set.ad[flat]
            var ne = sl.b * (sl.b - 1) // 2
            names.append(_anima_prefix(bi, slot) + String(".oft_R.weight"))
            tensors.append(ArcPointer(_f32_2d(sl.vec.copy(), sl.r, ne, ctx)))
            nmods += 1
    if nmods == 0:
        raise Error("save_anima_oft: refusing to write an empty OFT file")
    save_safetensors(names, tensors, path, ctx)
    return nmods
