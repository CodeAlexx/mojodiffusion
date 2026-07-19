# models/ideogram4/ideogram4_lokr_stack.mojo — LoKr carrier dispatch for the
# ideogram4 trainer (mirrors zimage_lokr_stack.mojo / krea2_lokr_stack.mojo,
# adapted to ideogram4's geometry: IDEOGRAM4_NUM_LAYERS blocks × I4_SLOTS_PER_BLOCK
# (qkv/o/w1/w2/w3/adaln), no dbl/sgl split).
#
# ── WHAT IS DIFFERENT FROM ZIMAGE/KREA2 (the device bridge) ──────────────────
# zimage/krea2 consume the carrier in the HOST LoraAdapter form
# (training/train_step.LoraAdapter, host BF16 a/b) and their stack uploads it
# internally, returning HOST d_a/d_b (List[List[Float32]]). ideogram4's LoRA is
# DEVICE-resident: the trainer runs ideogram4_lora_train_compute_resident over an
# Ideogram4LoraSet (device lora_module.LoraAdapter) and gets DEVICE d_a/d_b
# tensors back. So this file wraps the SAME model-agnostic LoKr carrier core with
# a host<->device bridge:
#   synth : master LoKr --lokr_carrier_adapter--> host (a,b) --from_host_bf16-->
#           device Ideogram4LoraSet (scale folded into b_c, carrier scale 1.0 =>
#           the ideogram4 adapter is built rank=r_eff, alpha=r_eff => scale()==1)
#   grads : device grads.d_a[i]/.d_b[i] --to_host--> host F32 lists -->
#           lokr_chain_carrier_grads --> master grads
# The carrier CORE (lokr_carrier_adapter / lokr_chain_carrier_grads / lokr_adamw /
# lokr_perturbed_normal_init) is reused as-is; only ideogram4's SET geometry +
# the bridge + the save prefix are new. No stack/kernel change.

from std.collections import List
from std.math import sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.training.train_step import LoraAdapter as CarrierLoRA
from serenitymojo.training.lokr_adapter import (
    LoKrAdapter, LoKrGrads, new_lokr_adapter, lokr_adamw,
)
from serenitymojo.training.lokr_stack import (
    lokr_carrier_adapter, lokr_carrier_r_eff, lokr_chain_carrier_grads,
    _inactive_carrier, _dummy_lokr, _empty_lokr_grads, _grads_sqsum, _grads_scale,
    LOKR_CARRIER_MAX_DEVICE_BYTES,
)
from serenitymojo.training.lokr_save import NamedLoKr, save_lokr_peft
from serenitymojo.models.ideogram4.lora_module import LoraAdapter
from serenitymojo.models.ideogram4.block import (
    Ideogram4LoraSet, Ideogram4StackLoraGrads,
    I4_SLOT_QKV, I4_SLOT_O, I4_SLOT_W1, I4_SLOT_W2, I4_SLOT_W3, I4_SLOT_ADALN,
    I4_SLOTS_PER_BLOCK,
)
from serenitymojo.models.ideogram4.config import (
    IDEOGRAM4_ADALN_DIM, IDEOGRAM4_HIDDEN, IDEOGRAM4_INTERMEDIATE_SIZE,
    IDEOGRAM4_NUM_LAYERS,
)


comptime LArc = ArcPointer[LoraAdapter]

comptime I4LOKR_TGT_ATTN = 1     # slots qkv(0), o(1)
comptime I4LOKR_TGT_ALL = 2      # + w1(2)/w2(3)/w3(4)/adaln(5)


# (in_f, out_f) for one ideogram4 slot — MUST mirror build_ideogram4_lora_set
# (block.mojo): qkv (H,3H), o (H,H), w1 (H,F), w2 (F,H), w3 (H,F), adaln (A,4H).
def ideogram4_lokr_slot_dims(slot: Int, H: Int, F: Int, A: Int) raises -> Tuple[Int, Int]:
    if slot == I4_SLOT_QKV:
        return (H, 3 * H)
    elif slot == I4_SLOT_O:
        return (H, H)
    elif slot == I4_SLOT_W1:
        return (H, F)
    elif slot == I4_SLOT_W2:
        return (F, H)
    elif slot == I4_SLOT_W3:
        return (H, F)
    elif slot == I4_SLOT_ADALN:
        return (A, 4 * H)
    raise Error(String("ideogram4_lokr_slot_dims: bad slot ") + String(slot))


def _ideogram4_slot_targeted(slot: Int, targets: Int) -> Bool:
    var is_attn = slot == I4_SLOT_QKV or slot == I4_SLOT_O
    if is_attn:
        return targets >= I4LOKR_TGT_ATTN
    return targets >= I4LOKR_TGT_ALL


struct Ideogram4LoKrSet(Copyable, Movable):
    var ad: List[LoKrAdapter]   # n_layers * I4_SLOTS_PER_BLOCK (dummy when inactive)
    var active: List[Bool]
    var n_layers: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoKrAdapter], var active: List[Bool],
        n_layers: Int, rank: Int,
    ):
        self.ad = ad^
        self.active = active^
        self.n_layers = n_layers
        self.rank = rank


def empty_ideogram4_lokr_set() -> Ideogram4LoKrSet:
    return Ideogram4LoKrSet(List[LoKrAdapter](), List[Bool](), 0, 0)


def build_ideogram4_lokr_set(
    n_layers: Int, H: Int, F: Int, A: Int,
    rank: Int, alpha: Float32, factor: Int,
    decompose_both: Bool, full_matrix: Bool, targets: Int, seed: UInt64,
) raises -> Ideogram4LoKrSet:
    if targets < I4LOKR_TGT_ATTN or targets > I4LOKR_TGT_ALL:
        raise Error("build_ideogram4_lokr_set: targets must be 1(attn)|2(all)")
    var ad = List[LoKrAdapter]()
    var active = List[Bool]()
    var s = seed
    for _b in range(n_layers):
        for slot in range(I4_SLOTS_PER_BLOCK):
            if _ideogram4_slot_targeted(slot, targets):
                var dims = ideogram4_lokr_slot_dims(slot, H, F, A)
                ad.append(new_lokr_adapter(
                    dims[0], dims[1], rank, alpha, factor, s, decompose_both, full_matrix
                ))
                active.append(True)
            else:
                ad.append(_dummy_lokr())
                active.append(False)
            s += 1
    return Ideogram4LoKrSet(ad^, active^, n_layers, rank)


# bf16 device bytes of the whole carrier LoRA set (a_c + b_c per active slot;
# in+out for inactive rank-1 zero legs). Fail-loud preflight vs the shared budget.
def ideogram4_lokr_carrier_total_bytes(set: Ideogram4LoKrSet, H: Int, F: Int, A: Int) raises -> Int:
    var elems = 0
    for i in range(len(set.ad)):
        if set.active[i]:
            var r = lokr_carrier_r_eff(set.ad[i])
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
    # rank=r_eff, alpha=r_eff => scale() = alpha/rank = 1.0 (carrier scale folded
    # into b_c already, exactly like the zimage/krea2 device-free carriers).
    return LArc(LoraAdapter(a^, b^, c.rank, Float32(c.rank)))


# inactive slot: a rank-1 zero LoRA (delta==0 always; its grads are ignored).
def _inactive_device(in_f: Int, out_f: Int, ctx: DeviceContext) raises -> LArc:
    var a_sh = List[Int](); a_sh.append(1); a_sh.append(in_f)
    var b_sh = List[Int](); b_sh.append(out_f); b_sh.append(1)
    var a = zeros_device(a_sh^, STDtype.BF16, ctx)
    var b = zeros_device(b_sh^, STDtype.BF16, ctx)
    return LArc(LoraAdapter(a^, b^, 1, Float32(1.0)))


# Materialize the carrier as a fresh DEVICE Ideogram4LoraSet for one step's
# forward/backward. Flat order layer*I4_SLOTS_PER_BLOCK + slot matches the
# Ideogram4StackLoraGrads d_a/d_b order (block.mojo backward).
def ideogram4_lokr_carrier_device_set(
    set: Ideogram4LoKrSet, H: Int, F: Int, A: Int, ctx: DeviceContext
) raises -> Ideogram4LoraSet:
    var ad = List[LArc]()
    for i in range(len(set.ad)):
        if set.active[i]:
            var c = lokr_carrier_adapter(set.ad[i])
            ad.append(_carrier_to_device(c, ctx))
        else:
            var dims = ideogram4_lokr_slot_dims(i % I4_SLOTS_PER_BLOCK, H, F, A)
            ad.append(_inactive_device(dims[0], dims[1], ctx))
    return Ideogram4LoraSet(ad^, set.n_layers, set.rank)


struct Ideogram4LoKrGrads(Movable):
    var g: List[LoKrGrads]

    def __init__(out self, var g: List[LoKrGrads]):
        self.g = g^


# host-list chain (used by the compile/orchestration smoke and by the device
# helper below). d_a[i]/d_b[i] are the per-slot carrier grads pulled to host.
def ideogram4_lokr_chain_all(
    set: Ideogram4LoKrSet, d_a: List[List[Float32]], d_b: List[List[Float32]]
) raises -> Ideogram4LoKrGrads:
    if len(d_a) != len(set.ad) or len(d_b) != len(set.ad):
        raise Error("ideogram4_lokr_chain_all: grad list count mismatch")
    var g = List[LoKrGrads]()
    for i in range(len(set.ad)):
        if set.active[i]:
            g.append(lokr_chain_carrier_grads(set.ad[i], d_a[i], d_b[i]))
        else:
            g.append(_empty_lokr_grads())
    return Ideogram4LoKrGrads(g^)


# device grads -> host -> chain. Only active slots are read back (inactive slots
# contribute nothing and their device grads are discarded).
def ideogram4_lokr_chain_from_device(
    set: Ideogram4LoKrSet, grads: Ideogram4StackLoraGrads, ctx: DeviceContext
) raises -> Ideogram4LoKrGrads:
    if len(grads.d_a) != len(set.ad) or len(grads.d_b) != len(set.ad):
        raise Error("ideogram4_lokr_chain_from_device: device grad count mismatch")
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(len(set.ad)):
        if set.active[i]:
            d_a.append(grads.d_a[i][].to_host(ctx))
            d_b.append(grads.d_b[i][].to_host(ctx))
        else:
            d_a.append(List[Float32]())
            d_b.append(List[Float32]())
    return ideogram4_lokr_chain_all(set, d_a, d_b)


def ideogram4_lokr_grad_norm(grads: Ideogram4LoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(grads.g)):
        s += _grads_sqsum(grads.g[i])
    return sqrt(s)


def ideogram4_lokr_clip_grads(mut grads: Ideogram4LoKrGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(grads.g)):
        _grads_scale(grads.g[i], clip_scale)


def ideogram4_lokr_adamw_step(
    mut set: Ideogram4LoKrSet, grads: Ideogram4LoKrGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.ad)):
        if set.active[i]:
            lokr_adamw(set.ad[i], grads.g[i], t, lr, beta1, beta2, eps, weight_decay)


# w2-side zero leg (starts EXACTLY 0; must be >0 after a real step).
def ideogram4_lokr_zero_leg_l1(set: Ideogram4LoKrSet) -> Float64:
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


# ── save: provisional ideogram4 lokr key convention. Mirrors the plain-LoRA
# module prefix (transformer.layers.<L>.<suffix> — ideogram4LoraTargets.mojo)
# with the lokr_w1[_a/_b]/lokr_w2[_a/_b]/.alpha keys from save_lokr_peft. No
# inference-side lycoris loader exists yet, so this convention is being SET here
# (same posture as zimage/krea2). ────────────────────────────────────────────
def _ideogram4_lokr_prefix(layer: Int, slot: Int) raises -> String:
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
    raise Error(String("_ideogram4_lokr_prefix: bad slot ") + String(slot))


def save_ideogram4_lokr(set: Ideogram4LoKrSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoKr]()
    for layer in range(set.n_layers):
        for slot in range(I4_SLOTS_PER_BLOCK):
            var flat = layer * I4_SLOTS_PER_BLOCK + slot
            if set.active[flat]:
                named.append(NamedLoKr(_ideogram4_lokr_prefix(layer, slot), set.ad[flat].copy()))
    return save_lokr_peft(named, path, ctx)
