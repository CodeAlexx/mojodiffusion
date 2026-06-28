# training/oft_stack.mojo — OneTrainer-OFT via the shared (a,b) carrier (FULL
# materialized delta), reusing the proven stack path (no klein-stack surgery).
# OFT W_eff[o,base+k] = Σ_c R_g[k,c]·W[o,base+c] (input-block rotation folded into
# the weight). The carrier is the full delta:
#       a = I_in [in,in]   b = ΔW_oft = W_eff - W [out,in]   r_eff = in
# y = x@Wᵀ + x@bᵀ = x@W_effᵀ (== oft_ot_forward). VRAM-bound (r_eff=in) → preflight
# fails loud at large in. Re-materialized each step.
#
# Backward: the stack's d_b == ∂L/∂ΔW_oft == d_W_eff. Chain per block:
#   d_R_g[k,c] = Σ_o d_W_eff[o,base+k]·W[o,base+c]  →  d_Q_g (Neumann bwd)  →  d_vec_g.
# Gated vs oft_ot_backward in tests/oft_carrier_parity.mojo.

from std.collections import List
from std.math import sqrt
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.oft_onetrainer import (
    oft_ot_skew, oft_ot_neumann_r, _neumann_backward,
)
from serenitymojo.training.lokr_stack import (
    klein_lokr_slot_dims, _slot_targeted, _DBL_SLOTS, _SGL_SLOTS,
    LOKR_TGT_ALL, LOKR_CARRIER_MAX_DEVICE_BYTES,
)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def oft_ot_carrier_r_eff(IN: Int) -> Int:
    return IN


# Materialize the FULL-delta carrier (a=I_in, b=W_eff-W). Carrier scale == 1.
def oft_ot_carrier_adapter(
    vec: List[Float32], w: List[Float32], IN: Int, OUT: Int, b: Int, r: Int
) raises -> LoraAdapter:
    var ne = b * (b - 1) // 2
    var a = _zeros(IN * IN)                              # I_in
    for i in range(IN):
        a[i * IN + i] = Float32(1.0)
    var bb = _zeros(OUT * IN)                            # ΔW_oft = W_eff - W
    for g in range(r):
        var vblk = List[Float32]()
        for t in range(ne):
            vblk.append(vec[g * ne + t])
        var q = oft_ot_skew(vblk, b)
        var rg = oft_ot_neumann_r(q, b)                 # [b,b]
        var base = g * b
        for o in range(OUT):
            for k in range(b):
                var weff = Float32(0.0)
                for c in range(b):
                    weff += rg[k * b + c] * w[o * IN + base + c]
                bb[o * IN + base + k] = weff - w[o * IN + base + k]
    return LoraAdapter(
        a^, bb^, IN, IN, OUT, Float32(1.0),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
    )


# Chain carrier grads → d_vec. d_b_carrier == d_W_eff; d_a_carrier discarded.
def oft_ot_chain_carrier_grads(
    vec: List[Float32], w: List[Float32],
    d_a_carrier: List[Float32], d_b_carrier: List[Float32],
    IN: Int, OUT: Int, b: Int, r: Int,
) raises -> List[Float32]:
    var ne = b * (b - 1) // 2
    if len(d_b_carrier) != OUT * IN:
        raise Error("oft_ot_chain_carrier_grads: d_b numel mismatch")
    var d_vec = _zeros(r * ne)
    for g in range(r):
        var base = g * b
        # d_R_g[k,c] = Σ_o d_W_eff[o,base+k]·W[o,base+c]
        var d_rg = _zeros(b * b)
        for k in range(b):
            for c in range(b):
                var acc = Float32(0.0)
                for o in range(OUT):
                    acc += d_b_carrier[o * IN + base + k] * w[o * IN + base + c]
                d_rg[k * b + c] = acc
        var vblk = List[Float32]()
        for t in range(ne):
            vblk.append(vec[g * ne + t])
        var q = oft_ot_skew(vblk, b)
        var dq = _neumann_backward(q, d_rg, b)
        var t = 0
        for i in range(b):
            for j in range(i + 1, b):
                d_vec[g * ne + t] = dq[i * b + j] - dq[j * b + i]
                t += 1
    return d_vec^


# ══════════════════ klein orchestration (full-delta, w + vec per slot) ═════════
# OFT trainable = the triu vec per slot (F32). NOTE: the OneTrainer-OFT triu-vec
# SAVE format is net-new (no inference loader exists for it) — save_klein_oft is
# a follow-on; this wave proves build→carrier→chain→step (vec moves off zero).

def _oft_vec_adamw(
    mut vec: List[Float32], d_vec: List[Float32],
    mut m: List[Float32], mut v: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    var n = len(vec)
    if len(d_vec) != n or len(m) != n or len(v) != n:
        raise Error("_oft_vec_adamw: len mismatch")
    var b1p = Float32(1.0); var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1; b2p *= beta2
    var bc1 = Float32(1.0) - b1p; var bc2 = Float32(1.0) - b2p
    for i in range(n):
        var gv = d_vec[i]
        var mi = beta1 * m[i] + (Float32(1.0) - beta1) * gv
        var vi = beta2 * v[i] + (Float32(1.0) - beta2) * gv * gv
        m[i] = mi; v[i] = vi
        var pv = vec[i]
        if weight_decay > 0.0:
            pv = pv * (Float32(1.0) - lr * weight_decay)
        vec[i] = pv - lr * (mi / bc1) / (sqrt(vi / bc2) + eps)


struct OFTSlot(Copyable, Movable):
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


struct KleinOFTSet(Copyable, Movable):
    var dbl: List[OFTSlot]
    var dbl_active: List[Bool]
    var sgl: List[OFTSlot]
    var sgl_active: List[Bool]
    var num_double: Int
    var num_single: Int

    def __init__(
        out self, var dbl: List[OFTSlot], var dbl_active: List[Bool],
        var sgl: List[OFTSlot], var sgl_active: List[Bool],
        num_double: Int, num_single: Int,
    ):
        self.dbl = dbl^
        self.dbl_active = dbl_active^
        self.sgl = sgl^
        self.sgl_active = sgl_active^
        self.num_double = num_double
        self.num_single = num_single


def _oft_synth_w(out_f: Int, in_f: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(out_f * in_f):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * Float32(0.5))
    return out^


def _dummy_oft_slot() -> OFTSlot:
    return OFTSlot(_zeros(1), _zeros(1), 1, 1, 1, 1, _zeros(1), _zeros(1))


def _make_oft_slot(in_f: Int, out_f: Int, block_size: Int, seed: UInt64) raises -> OFTSlot:
    if in_f % block_size != 0:
        raise Error(String("OFT: in_f ") + String(in_f) + " not divisible by block_size " + String(block_size))
    var b = block_size
    var r = in_f // b
    var ne = b * (b - 1) // 2
    return OFTSlot(
        _zeros(r * ne),                       # vec = 0 → R = I at init
        _oft_synth_w(out_f, in_f, seed),
        in_f, out_f, b, r,
        _zeros(r * ne), _zeros(r * ne),       # m, v
    )


def build_klein_oft_set(
    num_double: Int, num_single: Int, D: Int, F: Int,
    block_size: Int, targets: Int, seed: UInt64,
) raises -> KleinOFTSet:
    if targets < 1 or targets > LOKR_TGT_ALL:
        raise Error("build_klein_oft_set: targets must be 1(attn)|2(attn+ff)|3(all)")
    var s = seed
    var dbl = List[OFTSlot]()
    var dbl_active = List[Bool]()
    for _bi in range(num_double):
        for slot in range(_DBL_SLOTS):
            if _slot_targeted(True, slot, targets):
                var dims = klein_lokr_slot_dims(True, slot, D, F)
                dbl.append(_make_oft_slot(dims[0], dims[1], block_size, s * 131 + 7))
                dbl_active.append(True)
            else:
                dbl.append(_dummy_oft_slot())
                dbl_active.append(False)
            s += 1
    var sgl = List[OFTSlot]()
    var sgl_active = List[Bool]()
    for _bi in range(num_single):
        for slot in range(_SGL_SLOTS):
            if _slot_targeted(False, slot, targets):
                var dims = klein_lokr_slot_dims(False, slot, D, F)
                sgl.append(_make_oft_slot(dims[0], dims[1], block_size, s * 131 + 7))
                sgl_active.append(True)
            else:
                sgl.append(_dummy_oft_slot())
                sgl_active.append(False)
            s += 1
    return KleinOFTSet(dbl^, dbl_active^, sgl^, sgl_active^, num_double, num_single)


def klein_oft_carrier_lists(
    set: KleinOFTSet, D: Int, F: Int
) raises -> Tuple[List[LoraAdapter], List[LoraAdapter]]:
    var dbl = List[LoraAdapter]()
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            ref sl = set.dbl[i]
            dbl.append(oft_ot_carrier_adapter(sl.vec, sl.w, sl.in_f, sl.out_f, sl.b, sl.r))
        else:
            var dims = klein_lokr_slot_dims(True, i % _DBL_SLOTS, D, F)
            dbl.append(_inactive_oft_carrier(dims[0], dims[1]))
    var sgl = List[LoraAdapter]()
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            ref sl = set.sgl[i]
            sgl.append(oft_ot_carrier_adapter(sl.vec, sl.w, sl.in_f, sl.out_f, sl.b, sl.r))
        else:
            var dims = klein_lokr_slot_dims(False, i % _SGL_SLOTS, D, F)
            sgl.append(_inactive_oft_carrier(dims[0], dims[1]))
    return (dbl^, sgl^)


def _inactive_oft_carrier(in_f: Int, out_f: Int) raises -> LoraAdapter:
    return LoraAdapter(
        _zeros(in_f), _zeros(out_f), 1, in_f, out_f, Float32(0.0),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
    )


def klein_oft_carrier_total_bytes(set: KleinOFTSet) -> Int:
    var elems = 0
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            var inf = set.dbl[i].in_f
            elems += inf * inf + set.dbl[i].out_f * inf
        else:
            elems += 2
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            var inf = set.sgl[i].in_f
            elems += inf * inf + set.sgl[i].out_f * inf
        else:
            elems += 2
    return elems * 2


def klein_oft_preflight(set: KleinOFTSet) raises:
    var b = klein_oft_carrier_total_bytes(set)
    if b > LOKR_CARRIER_MAX_DEVICE_BYTES:
        raise Error(
            String("OFT carrier (full delta r_eff=in) needs ") + String(b)
            + " bytes (> budget " + String(LOKR_CARRIER_MAX_DEVICE_BYTES) + ")."
        )


struct KleinOFTGrads(Movable):
    var dbl: List[List[Float32]]   # d_vec per dbl slot
    var sgl: List[List[Float32]]

    def __init__(out self, var dbl: List[List[Float32]], var sgl: List[List[Float32]]):
        self.dbl = dbl^
        self.sgl = sgl^


def klein_oft_chain_all(
    set: KleinOFTSet,
    dbl_d_a: List[List[Float32]], dbl_d_b: List[List[Float32]],
    sgl_d_a: List[List[Float32]], sgl_d_b: List[List[Float32]],
) raises -> KleinOFTGrads:
    var dbl = List[List[Float32]]()
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            ref sl = set.dbl[i]
            dbl.append(oft_ot_chain_carrier_grads(sl.vec, sl.w, dbl_d_a[i], dbl_d_b[i], sl.in_f, sl.out_f, sl.b, sl.r))
        else:
            dbl.append(List[Float32]())
    var sgl = List[List[Float32]]()
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            ref sl = set.sgl[i]
            sgl.append(oft_ot_chain_carrier_grads(sl.vec, sl.w, sgl_d_a[i], sgl_d_b[i], sl.in_f, sl.out_f, sl.b, sl.r))
        else:
            sgl.append(List[Float32]())
    return KleinOFTGrads(dbl^, sgl^)


def klein_oft_adamw_step(
    mut set: KleinOFTSet, grads: KleinOFTGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            _oft_vec_adamw(set.dbl[i].vec, grads.dbl[i], set.dbl[i].m, set.dbl[i].v,
                           t, lr, beta1, beta2, eps, weight_decay)
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            _oft_vec_adamw(set.sgl[i].vec, grads.sgl[i], set.sgl[i].m, set.sgl[i].v,
                           t, lr, beta1, beta2, eps, weight_decay)


# vec starts EXACTLY 0 (R=I); must be >0 after a real step.
def klein_oft_vec_l1(set: KleinOFTSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.dbl)):
        if not set.dbl_active[i]:
            continue
        ref sl = set.dbl[i]
        for j in range(len(sl.vec)):
            var x = Float64(sl.vec[j])
            s += x if x >= 0.0 else -x
    for i in range(len(set.sgl)):
        if not set.sgl_active[i]:
            continue
        ref sl = set.sgl[i]
        for j in range(len(sl.vec)):
            var x = Float64(sl.vec[j])
            s += x if x >= 0.0 else -x
    return s
