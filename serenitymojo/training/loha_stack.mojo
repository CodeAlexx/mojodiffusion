# training/loha_stack.mojo — LoHa e2e TRAINING integration via the shared (a,b)
# carrier (mirrors lokr_stack.mojo).
#
# ── THE CARRIER REPRESENTATION (why no stack/kernel change is needed) ─────────
# The klein/model LoRA stack consumes adapters in the plain-LoRA form
#       y_delta = (x @ a_c^T) @ b_c^T          a_c:[R_eff,in]  b_c:[out,R_eff]
# i.e. ΔW_carrier^T = b_c @ a_c  [out,in].  A LoHa delta is
#       ΔW = (w1a@w1b) ⊙ (w2a@w2b) · scale     ΔW:[in,out]
# The Hadamard of two rank-R products has rank ≤ R², so it factors into a SMALL
# (a_c,b_c) carrier with R_eff = R² — NOT a VRAM-prohibitive full delta:
#   ΔW^T[o,i] = scale·Σ_{k,l} (w1b[k,o]·w2b[l,o])·(w1a[i,k]·w2a[i,l])
# ⇒ a_c[(k·R+l), i] = w1a[i,k]·w2a[i,l]            (scale-free)
#   b_c[o, (k·R+l)] = scale·w1b[k,o]·w2b[l,o]       (scale folded into b_c)
#   b_c @ a_c = ΔW^T  ⇒  y_delta = x @ ΔW  (matches loha_forward).
# L is bilinear in (a_c,b_c) so the stack's ∂L/∂a_c, ∂L/∂b_c chain EXACTLY to the
# four LoHa factor grads (loha_chain_carrier_grads, gated vs loha_backward in
# tests/loha_carrier_parity.mojo). Identity-at-init holds because w2a==0 ⇒ a_c==0
# ⇒ ΔW==0 (the LoHa zero-leg), exactly like the primitive.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.loha_adapter import (
    LoHaAdapter, LoHaGrads, new_loha_adapter, loha_adamw,
)
from serenitymojo.training.lokr_stack import (
    klein_lokr_slot_dims, _slot_targeted, _DBL_SLOTS, _SGL_SLOTS,
    LOKR_TGT_ALL, klein_lokr_prefix, _inactive_carrier,
)
from serenitymojo.training.loha_save import NamedLoHa, save_loha_peft


def _bf16_to_f32(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def loha_carrier_r_eff(lo: LoHaAdapter) -> Int:
    return lo.rank * lo.rank


# Materialize the (a_c, b_c) carrier for one LoHa master. Carrier scale == 1.0
# (the LoHa scale is folded into b_c). Optimizer moments are EMPTY (the master
# owns the optimizer state; the carrier is never stepped directly).
def loha_carrier_adapter(lo: LoHaAdapter) raises -> LoraAdapter:
    var IN = lo.in_f
    var OUT = lo.out_f
    var R = lo.rank
    var r_eff = R * R
    var sc = lo.scale
    var w1a = _bf16_to_f32(lo.w1a)   # [in,R]
    var w1b = _bf16_to_f32(lo.w1b)   # [R,out]
    var w2a = _bf16_to_f32(lo.w2a)   # [in,R]
    var w2b = _bf16_to_f32(lo.w2b)   # [R,out]
    var a = _zeros(r_eff * IN)       # [R²,in]
    var b = _zeros(OUT * r_eff)      # [out,R²]
    for k in range(R):
        for l in range(R):
            var rr = k * R + l
            for i in range(IN):
                a[rr * IN + i] = w1a[i * R + k] * w2a[i * R + l]
            for o in range(OUT):
                b[o * r_eff + rr] = sc * w1b[k * OUT + o] * w2b[l * OUT + o]
    return LoraAdapter(
        a^, b^, r_eff, IN, OUT, Float32(1.0),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
    )


# Chain the carrier (a_c,b_c) grads back to the four LoHa factor grads.
# From a_c[(k,l),i] = w1a[i,k]·w2a[i,l]:
#   d_w1a[i,k] = Σ_l d_a[(k,l),i]·w2a[i,l] ;  d_w2a[i,l] = Σ_k d_a[(k,l),i]·w1a[i,k]
# From b_c[o,(k,l)] = sc·w1b[k,o]·w2b[l,o]:
#   d_w1b[k,o] = sc·Σ_l d_b[o,(k,l)]·w2b[l,o] ; d_w2b[l,o] = sc·Σ_k d_b[o,(k,l)]·w1b[k,o]
# d_x is owned by the stack (the carrier's residual-stream grad) — empty here.
def loha_chain_carrier_grads(
    lo: LoHaAdapter, d_a: List[Float32], d_b: List[Float32]
) raises -> LoHaGrads:
    var IN = lo.in_f
    var OUT = lo.out_f
    var R = lo.rank
    var r_eff = R * R
    var sc = lo.scale
    if len(d_a) != r_eff * IN:
        raise Error("loha_chain_carrier_grads: d_a numel mismatch")
    if len(d_b) != OUT * r_eff:
        raise Error("loha_chain_carrier_grads: d_b numel mismatch")
    var w1a = _bf16_to_f32(lo.w1a)
    var w1b = _bf16_to_f32(lo.w1b)
    var w2a = _bf16_to_f32(lo.w2a)
    var w2b = _bf16_to_f32(lo.w2b)
    var d_w1a = _zeros(IN * R)
    var d_w1b = _zeros(R * OUT)
    var d_w2a = _zeros(IN * R)
    var d_w2b = _zeros(R * OUT)
    for k in range(R):
        for l in range(R):
            var rr = k * R + l
            for i in range(IN):
                var da = d_a[rr * IN + i]
                d_w1a[i * R + k] = d_w1a[i * R + k] + da * w2a[i * R + l]
                d_w2a[i * R + l] = d_w2a[i * R + l] + da * w1a[i * R + k]
            for o in range(OUT):
                var db = d_b[o * r_eff + rr]
                d_w1b[k * OUT + o] = d_w1b[k * OUT + o] + sc * db * w2b[l * OUT + o]
                d_w2b[l * OUT + o] = d_w2b[l * OUT + o] + sc * db * w1b[k * OUT + o]
    return LoHaGrads(d_w1a^, d_w1b^, d_w2a^, d_w2b^, List[Float32]())


# ══════════════════ klein orchestration (mirrors lokr_stack) ══════════════════
# LoHa has no factorization variants, so the per-slot set is simpler than LoKr:
# one LoHaAdapter per active (in,out) slot; w2a is the zero-leg (ΔW==0 at init).

struct KleinLoHaSet(Copyable, Movable):
    var dbl: List[LoHaAdapter]      # num_double * 12 (dummy 1x1 when inactive)
    var dbl_active: List[Bool]
    var sgl: List[LoHaAdapter]      # num_single * 2
    var sgl_active: List[Bool]
    var num_double: Int
    var num_single: Int
    var rank: Int

    def __init__(
        out self, var dbl: List[LoHaAdapter], var dbl_active: List[Bool],
        var sgl: List[LoHaAdapter], var sgl_active: List[Bool],
        num_double: Int, num_single: Int, rank: Int,
    ):
        self.dbl = dbl^
        self.dbl_active = dbl_active^
        self.sgl = sgl^
        self.sgl_active = sgl_active^
        self.num_double = num_double
        self.num_single = num_single
        self.rank = rank


def _dummy_loha() -> LoHaAdapter:
    return new_loha_adapter(1, 1, 1, Float32(1.0), 0)


def build_klein_loha_set(
    num_double: Int, num_single: Int, D: Int, F: Int,
    rank: Int, alpha: Float32, targets: Int, seed: UInt64,
) raises -> KleinLoHaSet:
    if targets < 1 or targets > LOKR_TGT_ALL:
        raise Error("build_klein_loha_set: targets must be 1(attn)|2(attn+ff)|3(all)")
    var s = seed
    var dbl = List[LoHaAdapter]()
    var dbl_active = List[Bool]()
    for _bi in range(num_double):
        for slot in range(_DBL_SLOTS):
            if _slot_targeted(True, slot, targets):
                var dims = klein_lokr_slot_dims(True, slot, D, F)
                dbl.append(new_loha_adapter(dims[0], dims[1], rank, alpha, s))
                dbl_active.append(True)
            else:
                dbl.append(_dummy_loha())
                dbl_active.append(False)
            s += 1
    var sgl = List[LoHaAdapter]()
    var sgl_active = List[Bool]()
    for _bi in range(num_single):
        for slot in range(_SGL_SLOTS):
            if _slot_targeted(False, slot, targets):
                var dims = klein_lokr_slot_dims(False, slot, D, F)
                sgl.append(new_loha_adapter(dims[0], dims[1], rank, alpha, s))
                sgl_active.append(True)
            else:
                sgl.append(_dummy_loha())
                sgl_active.append(False)
            s += 1
    return KleinLoHaSet(dbl^, dbl_active^, sgl^, sgl_active^, num_double, num_single, rank)


def klein_loha_carrier_lists(
    set: KleinLoHaSet, D: Int, F: Int
) raises -> Tuple[List[LoraAdapter], List[LoraAdapter]]:
    var dbl = List[LoraAdapter]()
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            dbl.append(loha_carrier_adapter(set.dbl[i]))
        else:
            var dims = klein_lokr_slot_dims(True, i % _DBL_SLOTS, D, F)
            dbl.append(_inactive_carrier(dims[0], dims[1]))
    var sgl = List[LoraAdapter]()
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            sgl.append(loha_carrier_adapter(set.sgl[i]))
        else:
            var dims = klein_lokr_slot_dims(False, i % _SGL_SLOTS, D, F)
            sgl.append(_inactive_carrier(dims[0], dims[1]))
    return (dbl^, sgl^)


def lokr_loha_carrier_total_bytes(set: KleinLoHaSet) -> Int:
    """Preflight: bf16 device bytes of the full LoHa carrier set (R_eff=R²)."""
    var elems = 0
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            var r = loha_carrier_r_eff(set.dbl[i])
            elems += r * set.dbl[i].in_f + set.dbl[i].out_f * r
        else:
            elems += set.dbl[i].in_f + set.dbl[i].out_f
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            var r = loha_carrier_r_eff(set.sgl[i])
            elems += r * set.sgl[i].in_f + set.sgl[i].out_f * r
        else:
            elems += set.sgl[i].in_f + set.sgl[i].out_f
    return elems * 2


struct KleinLoHaGrads(Movable):
    var dbl: List[LoHaGrads]
    var sgl: List[LoHaGrads]

    def __init__(out self, var dbl: List[LoHaGrads], var sgl: List[LoHaGrads]):
        self.dbl = dbl^
        self.sgl = sgl^


def _empty_loha_grads() -> LoHaGrads:
    return LoHaGrads(
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
        List[Float32](),
    )


def klein_loha_chain_all(
    set: KleinLoHaSet,
    dbl_d_a: List[List[Float32]], dbl_d_b: List[List[Float32]],
    sgl_d_a: List[List[Float32]], sgl_d_b: List[List[Float32]],
) raises -> KleinLoHaGrads:
    if len(dbl_d_a) != len(set.dbl) or len(sgl_d_a) != len(set.sgl):
        raise Error("klein_loha_chain_all: grad list count mismatch")
    var dbl = List[LoHaGrads]()
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            dbl.append(loha_chain_carrier_grads(set.dbl[i], dbl_d_a[i], dbl_d_b[i]))
        else:
            dbl.append(_empty_loha_grads())
    var sgl = List[LoHaGrads]()
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            sgl.append(loha_chain_carrier_grads(set.sgl[i], sgl_d_a[i], sgl_d_b[i]))
        else:
            sgl.append(_empty_loha_grads())
    return KleinLoHaGrads(dbl^, sgl^)


def klein_loha_adamw_step(
    mut set: KleinLoHaSet, g: KleinLoHaGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            loha_adamw(set.dbl[i], g.dbl[i], t, lr, beta1, beta2, eps, weight_decay)
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            loha_adamw(set.sgl[i], g.sgl[i], t, lr, beta1, beta2, eps, weight_decay)


# w2a is the LoHa zero-leg (starts EXACTLY 0; must be >0 after a real step).
def klein_loha_zero_leg_l1(set: KleinLoHaSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.dbl)):
        if not set.dbl_active[i]:
            continue
        ref lo = set.dbl[i]
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    for i in range(len(set.sgl)):
        if not set.sgl_active[i]:
            continue
        ref lo = set.sgl[i]
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


def klein_loha_trainable_l1(set: KleinLoHaSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.dbl)):
        if not set.dbl_active[i]:
            continue
        ref lo = set.dbl[i]
        for j in range(len(lo.w1a)):
            var v = Float64(lo.w1a[j].cast[DType.float32]()); s += v if v >= 0.0 else -v
        for j in range(len(lo.w1b)):
            var v = Float64(lo.w1b[j].cast[DType.float32]()); s += v if v >= 0.0 else -v
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]()); s += v if v >= 0.0 else -v
        for j in range(len(lo.w2b)):
            var v = Float64(lo.w2b[j].cast[DType.float32]()); s += v if v >= 0.0 else -v
    for i in range(len(set.sgl)):
        if not set.sgl_active[i]:
            continue
        ref lo = set.sgl[i]
        for j in range(len(lo.w1a)):
            var v = Float64(lo.w1a[j].cast[DType.float32]()); s += v if v >= 0.0 else -v
        for j in range(len(lo.w1b)):
            var v = Float64(lo.w1b[j].cast[DType.float32]()); s += v if v >= 0.0 else -v
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]()); s += v if v >= 0.0 else -v
        for j in range(len(lo.w2b)):
            var v = Float64(lo.w2b[j].cast[DType.float32]()); s += v if v >= 0.0 else -v
    return s


def save_klein_loha(set: KleinLoHaSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoHa]()
    for bi in range(set.num_double):
        for slot in range(_DBL_SLOTS):
            var flat = bi * _DBL_SLOTS + slot
            if set.dbl_active[flat]:
                named.append(NamedLoHa(klein_lokr_prefix(True, bi, slot), set.dbl[flat].copy()))
    for bi in range(set.num_single):
        for slot in range(_SGL_SLOTS):
            var flat = bi * _SGL_SLOTS + slot
            if set.sgl_active[flat]:
                named.append(NamedLoHa(klein_lokr_prefix(False, bi, slot), set.sgl[flat].copy()))
    return save_loha_peft(named, path, ctx)


# ── trainer-seam helpers (mirror lokr_stack: empty sentinel + grad norm/clip) ──
def empty_klein_loha_set() -> KleinLoHaSet:
    """Default-off placeholder (adapter_algo != 2)."""
    return KleinLoHaSet(
        List[LoHaAdapter](), List[Bool](), List[LoHaAdapter](), List[Bool](),
        0, 0, 0,
    )


def _loha_grads_sqsum(g: LoHaGrads) -> Float64:
    var s = Float64(0.0)
    for j in range(len(g.d_w1a)):
        s += Float64(g.d_w1a[j]) * Float64(g.d_w1a[j])
    for j in range(len(g.d_w1b)):
        s += Float64(g.d_w1b[j]) * Float64(g.d_w1b[j])
    for j in range(len(g.d_w2a)):
        s += Float64(g.d_w2a[j]) * Float64(g.d_w2a[j])
    for j in range(len(g.d_w2b)):
        s += Float64(g.d_w2b[j]) * Float64(g.d_w2b[j])
    return s


def klein_loha_grad_norm(g: KleinLoHaGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.dbl)):
        s += _loha_grads_sqsum(g.dbl[i])
    for i in range(len(g.sgl)):
        s += _loha_grads_sqsum(g.sgl[i])
    return sqrt(s)


def _loha_grads_scale(mut g: LoHaGrads, s: Float32):
    for j in range(len(g.d_w1a)):
        g.d_w1a[j] = g.d_w1a[j] * s
    for j in range(len(g.d_w1b)):
        g.d_w1b[j] = g.d_w1b[j] * s
    for j in range(len(g.d_w2a)):
        g.d_w2a[j] = g.d_w2a[j] * s
    for j in range(len(g.d_w2b)):
        g.d_w2b[j] = g.d_w2b[j] * s


def klein_loha_clip_grads(mut g: KleinLoHaGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(g.dbl)):
        _loha_grads_scale(g.dbl[i], clip_scale)
    for i in range(len(g.sgl)):
        _loha_grads_scale(g.sgl[i], clip_scale)
