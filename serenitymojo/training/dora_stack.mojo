# training/dora_stack.mojo — DoRA via the shared (a,b) carrier (FULL materialized
# delta). DoRA replaces the effective weight W_eff = m·(W+ΔW)/‖·‖_detached, which
# is NOT low-rank, so its carrier is a FULL delta:
#       a = I_in [in,in]   b = ΔW_dora = (W_eff - W) [out,in]   r_eff = in
# The stack adds y += (x@a^T)@b^T = x@ΔW_dora^T to the frozen x@Wᵀ, giving
# x@W_effᵀ (== dora_forward). VRAM-bound (r_eff=in) → preflight fails loud at large
# in; mechanically identical to LoKr/LoHa otherwise. Re-materialized each step.
#
# Backward: the stack's d_b == ∂L/∂ΔW_dora == d_WP_dora (since W frozen). The chain
# reproduces dora_backward's decomposition tail (detached norm) → d_A/d_B/d_m;
# d_a (the frozen identity's grad) is discarded. Gated vs dora_backward in
# tests/dora_carrier_parity.mojo.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.dora_adapter import (
    DoRAAdapter, DoRAGrads, dora_effective_weight, new_dora_adapter, dora_adamw,
)
from serenitymojo.training.lokr_stack import (
    klein_lokr_slot_dims, _slot_targeted, _DBL_SLOTS, _SGL_SLOTS,
    LOKR_TGT_ALL, klein_lokr_prefix, _inactive_carrier,
    LOKR_CARRIER_MAX_DEVICE_BYTES,
)
from serenitymojo.training.dora_save import NamedDoRA, save_dora_peft


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _bf16_to_f32(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


def _matmul(a: List[Float32], ra: Int, ca: Int, b: List[Float32], rb: Int, cb: Int) raises -> List[Float32]:
    if ca != rb:
        raise Error("dora_stack _matmul: dim mismatch")
    var out = _zeros(ra * cb)
    for i in range(ra):
        for k in range(ca):
            var aik = a[i * ca + k]
            if aik == Float32(0.0):
                continue
            var brow = k * cb
            var orow = i * cb
            for j in range(cb):
                out[orow + j] = out[orow + j] + aik * b[brow + j]
    return out^


def _transpose(a: List[Float32], r: Int, c: Int) -> List[Float32]:
    var out = _zeros(r * c)
    for i in range(r):
        for j in range(c):
            out[j * r + i] = a[i * c + j]
    return out^


def dora_carrier_r_eff(d: DoRAAdapter) -> Int:
    return d.in_f


# Materialize the FULL-delta carrier (a=I_in, b=W_eff-W). Carrier scale == 1.
def dora_carrier_adapter(d: DoRAAdapter, w_orig: List[Float32]) raises -> LoraAdapter:
    var IN = d.in_f
    var OUT = d.out_f
    var eff = dora_effective_weight(w_orig, d)          # eff.wp_dora = W_eff [out,in]
    var a = _zeros(IN * IN)                              # I_in
    for i in range(IN):
        a[i * IN + i] = Float32(1.0)
    var b = _zeros(OUT * IN)                             # ΔW_dora = W_eff - W
    for i in range(OUT * IN):
        b[i] = eff.wp_dora[i] - w_orig[i]
    return LoraAdapter(
        a^, b^, IN, IN, OUT, Float32(1.0),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
    )


# Chain the carrier grads back to the DoRA masters. d_b_carrier == d_WP_dora;
# d_a_carrier (frozen I) is discarded. Reproduces dora_backward's decomposition
# tail (detached den), axis-aware (wd_on_out).
def dora_chain_carrier_grads(
    d: DoRAAdapter, w_orig: List[Float32],
    d_a_carrier: List[Float32], d_b_carrier: List[Float32],
) raises -> DoRAGrads:
    var IN = d.in_f
    var OUT = d.out_f
    var R = d.rank
    if len(d_b_carrier) != OUT * IN:
        raise Error("dora_chain_carrier_grads: d_b numel mismatch")
    var eff = dora_effective_weight(w_orig, d)           # wp, den
    var mlen = OUT if d.wd_on_out else IN
    var d_m = _zeros(mlen)
    var d_wp = _zeros(OUT * IN)
    for o in range(OUT):
        for i in range(IN):
            var idx = o * IN + i
            var k = o if d.wd_on_out else i
            var g = d_b_carrier[idx]                      # = d_WP_dora[o,i]
            d_m[k] = d_m[k] + g * eff.wp[idx] / eff.den[k]
            d_wp[idx] = g * d.m[k] / eff.den[k]
    var g_scaled = d_wp.copy()
    for i in range(len(g_scaled)):
        g_scaled[i] = g_scaled[i] * d.scale              # ΔW=(B@A)*scale
    var a = _bf16_to_f32(d.a)
    var b = _bf16_to_f32(d.b)
    var a_t = _transpose(a, R, IN)                        # [in,rank]
    var d_b = _matmul(g_scaled, OUT, IN, a_t, IN, R)      # d_B = g @ Aᵀ  [out,rank]
    var b_t = _transpose(b, OUT, R)                       # [rank,out]
    var d_a = _matmul(b_t, R, OUT, g_scaled, OUT, IN)     # d_A = Bᵀ @ g  [rank,in]
    return DoRAGrads(d_a^, d_b^, d_m^, List[Float32]())


# ══════════════════ klein orchestration (full-delta, w_orig per slot) ══════════
# DoRA needs the FROZEN base weight per targeted projection (the carrier
# re-materializes W_eff−W each step). The set stores w_orig per slot. VRAM-bound
# (r_eff=in) → preflight fails loud at klein scale; usable small/subset.

struct KleinDoRASet(Copyable, Movable):
    var dbl: List[DoRAAdapter]
    var dbl_w: List[List[Float32]]      # w_orig [out*in] per dbl slot
    var dbl_active: List[Bool]
    var sgl: List[DoRAAdapter]
    var sgl_w: List[List[Float32]]
    var sgl_active: List[Bool]
    var num_double: Int
    var num_single: Int
    var rank: Int

    def __init__(
        out self, var dbl: List[DoRAAdapter], var dbl_w: List[List[Float32]],
        var dbl_active: List[Bool], var sgl: List[DoRAAdapter],
        var sgl_w: List[List[Float32]], var sgl_active: List[Bool],
        num_double: Int, num_single: Int, rank: Int,
    ):
        self.dbl = dbl^
        self.dbl_w = dbl_w^
        self.dbl_active = dbl_active^
        self.sgl = sgl^
        self.sgl_w = sgl_w^
        self.sgl_active = sgl_active^
        self.num_double = num_double
        self.num_single = num_single
        self.rank = rank


def empty_klein_dora_set() raises -> KleinDoRASet:
    return KleinDoRASet(
        List[DoRAAdapter](), List[List[Float32]](), List[Bool](),
        List[DoRAAdapter](), List[List[Float32]](), List[Bool](), 0, 0, 0,
    )


def _dummy_dora() raises -> DoRAAdapter:
    var w = List[Float32]()
    w.append(Float32(1.0))
    return new_dora_adapter(w^, 1, 1, 1, Float32(1.0), 0)


# Deterministic synthetic w_orig for the ORCHESTRATION gate. The live trainer
# sources real frozen weights per slot (the follow-on); the carrier/chain math
# is w_orig-agnostic (already gated in dora_carrier_parity).
def _synth_w(out_f: Int, in_f: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(out_f * in_f):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * Float32(0.5))
    return out^


def build_klein_dora_set(
    num_double: Int, num_single: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
    wd_on_out: Bool = False,
) raises -> KleinDoRASet:
    if targets < 1 or targets > LOKR_TGT_ALL:
        raise Error("build_klein_dora_set: targets must be 1(attn)|2(attn+ff)|3(all)")
    var s = seed
    var dbl = List[DoRAAdapter]()
    var dbl_w = List[List[Float32]]()
    var dbl_active = List[Bool]()
    for _bi in range(num_double):
        for slot in range(_DBL_SLOTS):
            if _slot_targeted(True, slot, targets):
                var dims = klein_lokr_slot_dims(True, slot, D, F)
                var w = _synth_w(dims[1], dims[0], s * 131 + 7)
                dbl.append(new_dora_adapter(w.copy(), dims[0], dims[1], rank, alpha, s, Float32(1.0e-7), wd_on_out))
                dbl_w.append(w^)
                dbl_active.append(True)
            else:
                dbl.append(_dummy_dora())
                var w1 = List[Float32](); w1.append(Float32(1.0))
                dbl_w.append(w1^)
                dbl_active.append(False)
            s += 1
    var sgl = List[DoRAAdapter]()
    var sgl_w = List[List[Float32]]()
    var sgl_active = List[Bool]()
    for _bi in range(num_single):
        for slot in range(_SGL_SLOTS):
            if _slot_targeted(False, slot, targets):
                var dims = klein_lokr_slot_dims(False, slot, D, F)
                var w = _synth_w(dims[1], dims[0], s * 131 + 7)
                sgl.append(new_dora_adapter(w.copy(), dims[0], dims[1], rank, alpha, s, Float32(1.0e-7), wd_on_out))
                sgl_w.append(w^)
                sgl_active.append(True)
            else:
                sgl.append(_dummy_dora())
                var w1 = List[Float32](); w1.append(Float32(1.0))
                sgl_w.append(w1^)
                sgl_active.append(False)
            s += 1
    return KleinDoRASet(dbl^, dbl_w^, dbl_active^, sgl^, sgl_w^, sgl_active^, num_double, num_single, rank)


def klein_dora_carrier_lists(
    set: KleinDoRASet, D: Int, F: Int
) raises -> Tuple[List[LoraAdapter], List[LoraAdapter]]:
    var dbl = List[LoraAdapter]()
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            dbl.append(dora_carrier_adapter(set.dbl[i], set.dbl_w[i]))
        else:
            var dims = klein_lokr_slot_dims(True, i % _DBL_SLOTS, D, F)
            dbl.append(_inactive_carrier(dims[0], dims[1]))
    var sgl = List[LoraAdapter]()
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            sgl.append(dora_carrier_adapter(set.sgl[i], set.sgl_w[i]))
        else:
            var dims = klein_lokr_slot_dims(False, i % _SGL_SLOTS, D, F)
            sgl.append(_inactive_carrier(dims[0], dims[1]))
    return (dbl^, sgl^)


def klein_dora_carrier_total_bytes(set: KleinDoRASet) -> Int:
    var elems = 0
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            var inf = set.dbl[i].in_f
            elems += inf * inf + set.dbl[i].out_f * inf   # a=I + b=ΔW (r_eff=in)
        else:
            elems += 2
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            var inf = set.sgl[i].in_f
            elems += inf * inf + set.sgl[i].out_f * inf
        else:
            elems += 2
    return elems * 2


def klein_dora_preflight(set: KleinDoRASet) raises:
    var b = klein_dora_carrier_total_bytes(set)
    if b > LOKR_CARRIER_MAX_DEVICE_BYTES:
        raise Error(
            String("DoRA carrier (full delta r_eff=in) needs ") + String(b)
            + " bytes (> budget " + String(LOKR_CARRIER_MAX_DEVICE_BYTES)
            + "); use a small subset of targets/dims or the W_eff-substitution path."
        )


struct KleinDoRAGrads(Movable):
    var dbl: List[DoRAGrads]
    var sgl: List[DoRAGrads]

    def __init__(out self, var dbl: List[DoRAGrads], var sgl: List[DoRAGrads]):
        self.dbl = dbl^
        self.sgl = sgl^


def _empty_dora_grads() -> DoRAGrads:
    return DoRAGrads(List[Float32](), List[Float32](), List[Float32](), List[Float32]())


def klein_dora_chain_all(
    set: KleinDoRASet,
    dbl_d_a: List[List[Float32]], dbl_d_b: List[List[Float32]],
    sgl_d_a: List[List[Float32]], sgl_d_b: List[List[Float32]],
) raises -> KleinDoRAGrads:
    if len(dbl_d_a) != len(set.dbl) or len(sgl_d_a) != len(set.sgl):
        raise Error("klein_dora_chain_all: grad list count mismatch")
    var dbl = List[DoRAGrads]()
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            dbl.append(dora_chain_carrier_grads(set.dbl[i], set.dbl_w[i], dbl_d_a[i], dbl_d_b[i]))
        else:
            dbl.append(_empty_dora_grads())
    var sgl = List[DoRAGrads]()
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            sgl.append(dora_chain_carrier_grads(set.sgl[i], set.sgl_w[i], sgl_d_a[i], sgl_d_b[i]))
        else:
            sgl.append(_empty_dora_grads())
    return KleinDoRAGrads(dbl^, sgl^)


def klein_dora_adamw_step(
    mut set: KleinDoRASet, grads: KleinDoRAGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            dora_adamw(set.dbl[i], grads.dbl[i], t, lr, beta1, beta2, eps, weight_decay)
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            dora_adamw(set.sgl[i], grads.sgl[i], t, lr, beta1, beta2, eps, weight_decay)


# B (lora_up) is the DoRA zero-leg (0 at init; must be >0 after a real step).
def klein_dora_zero_leg_l1(set: KleinDoRASet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.dbl)):
        if not set.dbl_active[i]:
            continue
        ref lo = set.dbl[i]
        for j in range(len(lo.b)):
            var v = Float64(lo.b[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    for i in range(len(set.sgl)):
        if not set.sgl_active[i]:
            continue
        ref lo = set.sgl[i]
        for j in range(len(lo.b)):
            var v = Float64(lo.b[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def save_klein_dora(set: KleinDoRASet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedDoRA]()
    for bi in range(set.num_double):
        for slot in range(_DBL_SLOTS):
            var flat = bi * _DBL_SLOTS + slot
            if set.dbl_active[flat]:
                named.append(NamedDoRA(klein_lokr_prefix(True, bi, slot), set.dbl[flat].copy()))
    for bi in range(set.num_single):
        for slot in range(_SGL_SLOTS):
            var flat = bi * _SGL_SLOTS + slot
            if set.sgl_active[flat]:
                named.append(NamedDoRA(klein_lokr_prefix(False, bi, slot), set.sgl[flat].copy()))
    return save_dora_peft(named, path, ctx)
