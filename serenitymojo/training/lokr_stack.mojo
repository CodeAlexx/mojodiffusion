# training/lokr_stack.mojo — LoKr e2e TRAINING integration for the Klein
# LoRA-stack trainer (T2.G, 2026-06-11). SimpleTuner-parity LoKr: the upstream
# (pip lycoris_lora 3.4.0) LokrModule semantics that SimpleTuner's
# `--lora_type=lycoris` + `lycoris_config.json {"algo":"lokr", ...}` path
# exposes, trained through the EXISTING klein stack forward/backward.
#
# ── THE CARRIER REPRESENTATION (why no stack/kernel changes are needed) ──────
# The klein stack consumes adapters in the plain-LoRA form
#       ΔW_oi = b @ a · scale          a:[R,in]  b:[out,R]
# and returns ∂L/∂a, ∂L/∂b per slot. A LoKr delta is a Kronecker product
#       ΔW_oi = kron(W1, W2) · scale   W1:[out_l,in_m]  W2:[out_k,in_n]
# and the Kronecker mixed-product identity kron(A,B)@kron(C,D) = kron(AC, BD)
# factors EVERY upstream LoKr variant into exactly one (b_c, a_c) pair:
#
#   L1 both-full   (use_w1 & use_w2; the SimpleTuner full_matrix=true default):
#       b_c = kron(W1, W2)·scale  [out, in]      a_c = I_in           R_eff = in
#   L2 W1 full + W2 factored (upstream default, decompose_both=false):
#       b_c = kron(W1, w2a)·scale [out, in_m·r]  a_c = kron(I_im,w2b) R_eff = in_m·r
#   L3 both factored (decompose_both=true and rank small enough):
#       b_c = kron(w1a, w2a)·scale [out, r·r]    a_c = kron(w1b,w2b)  R_eff = r·r
#   (W1 factored + W2 full is IMPOSSIBLE upstream: the W2-full condition
#    rank >= max(out_k,in_n)/2 with out_k>=out_l, in_n>=in_m contradicts the
#    W1-factor condition rank < max(out_l,in_m)/2; full_matrix also explicitly
#    blocks W1 factoring in lokr.py:156.)
#
# Each LoKr master parameter appears in EXACTLY ONE carrier, and L is bilinear
# in (a_c, b_c), so the stack's ∂L/∂a_c, ∂L/∂b_c chain EXACTLY to the master
# grads (lokr_chain_carrier_grads below — gated numerically against the
# upstream-parity-gated lokr_backward primitive in tests/lokr_st_parity.mojo).
# The masters then take the host AdamW step (lokr_adamw, cfg betas/eps/wd) and
# the carriers are re-materialized + re-uploaded for the next step.
#
# COST NOTE (measured math, klein9b D=4096 F=12288): with an explicit factor f,
# factorization(in,f)=(f, in/f) so R_eff = f·rank — tiny (f=4, rank=16 → 64).
# factor=-1 (auto) gives in_m≈sqrt(in) → R_eff≈64·rank, and full_matrix gives
# R_eff = in (a full-shape dense delta per projection: the single-stream fused
# projections alone are ~17 GB bf16) — the preflight below fails loud with the
# computed bytes when the carrier set cannot fit the device budget.
#
# ── SimpleTuner target preset mapping (LycorisNetwork.apply_preset) ──────────
# ST klein/Flux2 targets (documentation/LYCORIS.md:62-99): Flux2Attention,
# Flux2FeedForward, Flux2ParallelSelfAttention. Klein slot equivalence:
#   Flux2Attention             = double slots 0-3 (img q/k/v/out) + 6-9 (txt)
#   Flux2FeedForward           = double slots 4,5,10,11 (ff_in/ff_out both streams)
#   Flux2ParallelSelfAttention = single slots 0,1 (fused qkv_mlp + out)
# lokr_targets: 1=attn, 2=attn+ff (ST's generic default preset
# ["Attention","FeedForward"]), 3=all (ST's recommended Flux2 set).
# module_algo_map per-class factor = lokr_factor_attn/_ff/_single (0=inherit).
#
# Mojo 0.26.x: `def` + raises; no implicit Tensor copies; host List math.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lokr_adapter import (
    LoKrAdapter, LoKrGrads, new_lokr_adapter, lokr_adamw,
    lokr_perturbed_normal_init,
)
from serenitymojo.training.lokr_save import NamedLoKr, save_lokr_peft


comptime LOKR_TGT_ATTN = 1
comptime LOKR_TGT_ATTN_FF = 2
comptime LOKR_TGT_ALL = 3

# Klein slot scheme (klein_stack_lora.mojo contract).
comptime _DBL_SLOTS = 12
comptime _SGL_SLOTS = 2

# Device budget for the carrier set (bf16 params; grads are per-block
# transients on the hand-chain backward path). Fail-loud preflight.
comptime LOKR_CARRIER_MAX_DEVICE_BYTES = 10 * 1024 * 1024 * 1024


def _zeros_f32(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _bf16_to_f32(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


# ── the master set: one LoKr adapter per TARGETED klein slot ─────────────────
struct KleinLoKrSet(Copyable, Movable):
    var dbl: List[LoKrAdapter]      # num_double * 12 (dummy 1x1 when inactive)
    var dbl_active: List[Bool]
    var sgl: List[LoKrAdapter]      # num_single * 2
    var sgl_active: List[Bool]
    var num_double: Int
    var num_single: Int
    var rank: Int

    def __init__(
        out self, var dbl: List[LoKrAdapter], var dbl_active: List[Bool],
        var sgl: List[LoKrAdapter], var sgl_active: List[Bool],
        num_double: Int, num_single: Int, rank: Int,
    ):
        self.dbl = dbl^
        self.dbl_active = dbl_active^
        self.sgl = sgl^
        self.sgl_active = sgl_active^
        self.num_double = num_double
        self.num_single = num_single
        self.rank = rank


def empty_klein_lokr_set() -> KleinLoKrSet:
    """Default-off placeholder (adapter_algo != 4)."""
    return KleinLoKrSet(
        List[LoKrAdapter](), List[Bool](), List[LoKrAdapter](), List[Bool](),
        0, 0, 0,
    )


def klein_lokr_slot_dims(kind_double: Bool, slot: Int, D: Int, F: Int) raises -> Tuple[Int, Int]:
    """(in_f, out_f) per klein slot — MUST mirror build_klein_lora_set."""
    if kind_double:
        if slot == 4 or slot == 10:
            return (D, 2 * F)      # ff_in
        if slot == 5 or slot == 11:
            return (F, D)          # ff_out
        return (D, D)              # q/k/v/out both streams
    if slot == 0:
        return (D, 3 * D + 2 * F)  # fused qkv_mlp
    return (D + F, D)              # single to_out


def _slot_is_attn(kind_double: Bool, slot: Int) -> Bool:
    if not kind_double:
        return False
    return slot <= 3 or (slot >= 6 and slot <= 9)


def _slot_targeted(kind_double: Bool, slot: Int, targets: Int) -> Bool:
    if kind_double:
        if _slot_is_attn(kind_double, slot):
            return True                      # attn targeted at every level
        return targets >= LOKR_TGT_ATTN_FF   # ff slots
    return targets >= LOKR_TGT_ALL           # single (parallel attn) slots


def _slot_factor(
    kind_double: Bool, slot: Int,
    factor: Int, factor_attn: Int, factor_ff: Int, factor_single: Int,
) -> Int:
    """module_algo_map equivalent: per-class factor override, 0 = inherit."""
    if kind_double:
        if _slot_is_attn(kind_double, slot):
            return factor_attn if factor_attn != 0 else factor
        return factor_ff if factor_ff != 0 else factor
    return factor_single if factor_single != 0 else factor


def _dummy_lokr() raises -> LoKrAdapter:
    # inactive-slot placeholder (1x1, never trained, never saved)
    return new_lokr_adapter(1, 1, 1, Float32(1.0), -1, UInt64(1))


def build_klein_lokr_set(
    num_double: Int, num_single: Int, D: Int, F: Int,
    rank: Int, alpha: Float32,
    factor: Int, factor_attn: Int, factor_ff: Int, factor_single: Int,
    decompose_both: Bool, full_matrix: Bool, targets: Int, seed: UInt64,
) raises -> KleinLoKrSet:
    if targets < LOKR_TGT_ATTN or targets > LOKR_TGT_ALL:
        raise Error("build_klein_lokr_set: lokr_targets must be 1(attn)|2(attn+ff)|3(all)")
    var s = seed
    var dbl = List[LoKrAdapter]()
    var dbl_active = List[Bool]()
    for _bi in range(num_double):
        for slot in range(_DBL_SLOTS):
            if _slot_targeted(True, slot, targets):
                var dims = klein_lokr_slot_dims(True, slot, D, F)
                var f = _slot_factor(True, slot, factor, factor_attn, factor_ff, factor_single)
                dbl.append(new_lokr_adapter(
                    dims[0], dims[1], rank, alpha, f, s, decompose_both, full_matrix
                ))
                dbl_active.append(True)
            else:
                dbl.append(_dummy_lokr())
                dbl_active.append(False)
            s += 1
    var sgl = List[LoKrAdapter]()
    var sgl_active = List[Bool]()
    for _bi in range(num_single):
        for slot in range(_SGL_SLOTS):
            if _slot_targeted(False, slot, targets):
                var dims = klein_lokr_slot_dims(False, slot, D, F)
                var f = _slot_factor(False, slot, factor, factor_attn, factor_ff, factor_single)
                sgl.append(new_lokr_adapter(
                    dims[0], dims[1], rank, alpha, f, s, decompose_both, full_matrix
                ))
                sgl_active.append(True)
            else:
                sgl.append(_dummy_lokr())
                sgl_active.append(False)
            s += 1
    return KleinLoKrSet(dbl^, dbl_active^, sgl^, sgl_active^, num_double, num_single, rank)


# ── carrier materialization ──────────────────────────────────────────────────
def lokr_carrier_r_eff(lo: LoKrAdapter) raises -> Int:
    if (not lo.w1_factored) and (not lo.w2_factored):
        return lo.in_f                  # L1: identity a-leg
    if (not lo.w1_factored) and lo.w2_factored:
        return lo.in_m * lo.rank        # L2
    if lo.w1_factored and lo.w2_factored:
        return lo.rank * lo.rank        # L3
    raise Error("lokr carrier: W1-factored + W2-full is impossible upstream")


def lokr_carrier_adapter(lo: LoKrAdapter) raises -> LoraAdapter:
    """Materialize the (a_c, b_c) carrier pair for one LoKr master.
    Carrier scale is ALWAYS 1.0 — the LoKr scale (incl. the both-full forced
    scale=1 upstream quirk, already folded into lo.scale by the LoKrAdapter
    ctor) is folded into b_c. Optimizer moments are EMPTY (the masters own the
    optimizer state; klein's fused AdamW is never run on carriers)."""
    var OL = lo.out_l; var OK = lo.out_k; var IM = lo.in_m; var INn = lo.in_n
    var R = lo.rank
    var IN = lo.in_f; var OUT = lo.out_f
    var r_eff = lokr_carrier_r_eff(lo)
    var a = _zeros_f32(r_eff * IN)
    var b = _zeros_f32(OUT * r_eff)
    var sc = lo.scale

    if (not lo.w1_factored) and (not lo.w2_factored):
        # L1: b = kron(W1, W2)·scale  [out,in] ; a = I_in
        var w1d = _bf16_to_f32(lo.w1)
        var w2d = _bf16_to_f32(lo.w2)
        for c in range(IN):
            a[c * IN + c] = Float32(1.0)
        for l in range(OL):
            for c in range(IM):
                var w1lc = w1d[l * IM + c] * sc
                if w1lc == Float32(0.0):
                    continue
                for k in range(OK):
                    var row = (l * OK + k) * IN
                    var w2row = k * INn
                    var col0 = c * INn
                    for n in range(INn):
                        b[row + col0 + n] = w1lc * w2d[w2row + n]
    elif (not lo.w1_factored) and lo.w2_factored:
        # L2: b = kron(W1, w2a)·scale [out, IM·R] ; a = kron(I_IM, w2b)
        var w1d = _bf16_to_f32(lo.w1)
        var w2a = _bf16_to_f32(lo.w2a)
        var w2b = _bf16_to_f32(lo.w2b)
        for l in range(OL):
            for c in range(IM):
                var w1lc = w1d[l * IM + c] * sc
                for k in range(OK):
                    var row = (l * OK + k) * r_eff
                    var col0 = c * R
                    var w2arow = k * R
                    for j in range(R):
                        b[row + col0 + j] = w1lc * w2a[w2arow + j]
        for c in range(IM):
            for j in range(R):
                var row = (c * R + j) * IN
                var col0 = c * INn
                var w2brow = j * INn
                for n in range(INn):
                    a[row + col0 + n] = w2b[w2brow + n]
    elif lo.w1_factored and lo.w2_factored:
        # L3: b = kron(w1a, w2a)·scale [out, R·R] ; a = kron(w1b, w2b)
        var w1a = _bf16_to_f32(lo.w1a)
        var w1b = _bf16_to_f32(lo.w1b)
        var w2a = _bf16_to_f32(lo.w2a)
        var w2b = _bf16_to_f32(lo.w2b)
        for l in range(OL):
            for i in range(R):
                var w1ali = w1a[l * R + i] * sc
                for k in range(OK):
                    var row = (l * OK + k) * r_eff
                    var col0 = i * R
                    var w2arow = k * R
                    for j in range(R):
                        b[row + col0 + j] = w1ali * w2a[w2arow + j]
        for i in range(R):
            for c in range(IM):
                var w1bic = w1b[i * IM + c]
                for j in range(R):
                    var row = (i * R + j) * IN
                    var col0 = c * INn
                    var w2brow = j * INn
                    for n in range(INn):
                        a[row + col0 + n] = w1bic * w2b[w2brow + n]
    else:
        raise Error("lokr carrier: W1-factored + W2-full is impossible upstream")

    return LoraAdapter(
        a^, b^, r_eff, IN, OUT, Float32(1.0),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
    )


def _inactive_carrier(in_f: Int, out_f: Int) raises -> LoraAdapter:
    # rank-1 zero adapter, scale 0: contributes nothing, grads ignored.
    return LoraAdapter(
        _zeros_f32(in_f), _zeros_f32(out_f), 1, in_f, out_f, Float32(0.0),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
    )


def klein_lokr_carrier_lists(
    set: KleinLoKrSet, D: Int, F: Int
) raises -> Tuple[List[LoraAdapter], List[LoraAdapter]]:
    """Full dbl+sgl carrier adapter lists in klein flat slot order."""
    var dbl = List[LoraAdapter]()
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            dbl.append(lokr_carrier_adapter(set.dbl[i]))
        else:
            var dims = klein_lokr_slot_dims(True, i % _DBL_SLOTS, D, F)
            dbl.append(_inactive_carrier(dims[0], dims[1]))
    var sgl = List[LoraAdapter]()
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            sgl.append(lokr_carrier_adapter(set.sgl[i]))
        else:
            var dims = klein_lokr_slot_dims(False, i % _SGL_SLOTS, D, F)
            sgl.append(_inactive_carrier(dims[0], dims[1]))
    return (dbl^, sgl^)


def lokr_carrier_total_bytes(set: KleinLoKrSet) raises -> Int:
    """Preflight: bf16 device bytes of the full carrier set (a_c + b_c)."""
    var elems = 0
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            var r = lokr_carrier_r_eff(set.dbl[i])
            elems += r * set.dbl[i].in_f + set.dbl[i].out_f * r
        else:
            elems += set.dbl[i].in_f + set.dbl[i].out_f
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            var r = lokr_carrier_r_eff(set.sgl[i])
            elems += r * set.sgl[i].in_f + set.sgl[i].out_f * r
        else:
            elems += set.sgl[i].in_f + set.sgl[i].out_f
    return elems * 2


# ── carrier-grad → master-grad chaining (the exact bilinear chain rule) ──────
def lokr_chain_carrier_grads(
    lo: LoKrAdapter, d_a: List[Float32], d_b: List[Float32]
) raises -> LoKrGrads:
    var OL = lo.out_l; var OK = lo.out_k; var IM = lo.in_m; var INn = lo.in_n
    var R = lo.rank
    var IN = lo.in_f
    var r_eff = lokr_carrier_r_eff(lo)
    if len(d_a) != r_eff * IN:
        raise Error("lokr_chain_carrier_grads: d_a numel mismatch")
    if len(d_b) != lo.out_f * r_eff:
        raise Error("lokr_chain_carrier_grads: d_b numel mismatch")
    var sc = lo.scale

    var d_w1 = List[Float32]()
    var d_w1a = List[Float32]()
    var d_w1b = List[Float32]()
    var d_w2 = List[Float32]()
    var d_w2a = List[Float32]()
    var d_w2b = List[Float32]()

    if (not lo.w1_factored) and (not lo.w2_factored):
        # L1: d_b == ∂L/∂ΔW. d_W1[l,c] = Σ_{k,n} d_b·W2[k,n]·sc ;
        #     d_W2[k,n] = Σ_{l,c} d_b·W1[l,c]·sc.  (d_a is the frozen identity's
        #     grad — discarded, like upstream where I is not a parameter.)
        var w1d = _bf16_to_f32(lo.w1)
        var w2d = _bf16_to_f32(lo.w2)
        d_w1 = _zeros_f32(OL * IM)
        d_w2 = _zeros_f32(OK * INn)
        for l in range(OL):
            for c in range(IM):
                var w1lc = w1d[l * IM + c] * sc
                var acc = Float32(0.0)
                for k in range(OK):
                    var row = (l * OK + k) * IN
                    var col0 = c * INn
                    var w2row = k * INn
                    for n in range(INn):
                        var g = d_b[row + col0 + n]
                        acc += g * w2d[w2row + n]
                        d_w2[w2row + n] = d_w2[w2row + n] + g * w1lc
                d_w1[l * IM + c] = acc * sc
    elif (not lo.w1_factored) and lo.w2_factored:
        # L2
        var w1d = _bf16_to_f32(lo.w1)
        var w2a = _bf16_to_f32(lo.w2a)
        d_w1 = _zeros_f32(OL * IM)
        d_w2a = _zeros_f32(OK * R)
        d_w2b = _zeros_f32(R * INn)
        for l in range(OL):
            for c in range(IM):
                var w1lc = w1d[l * IM + c] * sc
                var acc = Float32(0.0)
                for k in range(OK):
                    var row = (l * OK + k) * r_eff
                    var col0 = c * R
                    var w2arow = k * R
                    for j in range(R):
                        var g = d_b[row + col0 + j]
                        acc += g * w2a[w2arow + j]
                        d_w2a[w2arow + j] = d_w2a[w2arow + j] + g * w1lc
                d_w1[l * IM + c] = acc * sc
        for c in range(IM):
            for j in range(R):
                var row = (c * R + j) * IN
                var col0 = c * INn
                var dst = j * INn
                for n in range(INn):
                    d_w2b[dst + n] = d_w2b[dst + n] + d_a[row + col0 + n]
    elif lo.w1_factored and lo.w2_factored:
        # L3
        var w1a = _bf16_to_f32(lo.w1a)
        var w1b = _bf16_to_f32(lo.w1b)
        var w2a = _bf16_to_f32(lo.w2a)
        var w2b = _bf16_to_f32(lo.w2b)
        d_w1a = _zeros_f32(OL * R)
        d_w1b = _zeros_f32(R * IM)
        d_w2a = _zeros_f32(OK * R)
        d_w2b = _zeros_f32(R * INn)
        for l in range(OL):
            for i in range(R):
                var w1ali = w1a[l * R + i] * sc
                var acc = Float32(0.0)
                for k in range(OK):
                    var row = (l * OK + k) * r_eff
                    var col0 = i * R
                    var w2arow = k * R
                    for j in range(R):
                        var g = d_b[row + col0 + j]
                        acc += g * w2a[w2arow + j]
                        d_w2a[w2arow + j] = d_w2a[w2arow + j] + g * w1ali
                d_w1a[l * R + i] = acc * sc
        for i in range(R):
            for c in range(IM):
                var w1bic = w1b[i * IM + c]
                var acc = Float32(0.0)
                for j in range(R):
                    var row = (i * R + j) * IN
                    var col0 = c * INn
                    var w2brow = j * INn
                    for n in range(INn):
                        var g = d_a[row + col0 + n]
                        acc += g * w2b[w2brow + n]
                        d_w2b[w2brow + n] = d_w2b[w2brow + n] + g * w1bic
                d_w1b[i * IM + c] = acc
    else:
        raise Error("lokr_chain_carrier_grads: impossible factor combo")

    return LoKrGrads(d_w1^, d_w1a^, d_w1b^, d_w2^, d_w2a^, d_w2b^, List[Float32]())


# ── master grad utilities (global-norm clip + AdamW over the whole set) ──────
def _grads_sqsum(g: LoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.d_w1)):
        s += Float64(g.d_w1[i]) * Float64(g.d_w1[i])
    for i in range(len(g.d_w1a)):
        s += Float64(g.d_w1a[i]) * Float64(g.d_w1a[i])
    for i in range(len(g.d_w1b)):
        s += Float64(g.d_w1b[i]) * Float64(g.d_w1b[i])
    for i in range(len(g.d_w2)):
        s += Float64(g.d_w2[i]) * Float64(g.d_w2[i])
    for i in range(len(g.d_w2a)):
        s += Float64(g.d_w2a[i]) * Float64(g.d_w2a[i])
    for i in range(len(g.d_w2b)):
        s += Float64(g.d_w2b[i]) * Float64(g.d_w2b[i])
    return s


def _grads_scale(mut g: LoKrGrads, s: Float32):
    for i in range(len(g.d_w1)):
        g.d_w1[i] *= s
    for i in range(len(g.d_w1a)):
        g.d_w1a[i] *= s
    for i in range(len(g.d_w1b)):
        g.d_w1b[i] *= s
    for i in range(len(g.d_w2)):
        g.d_w2[i] *= s
    for i in range(len(g.d_w2a)):
        g.d_w2a[i] *= s
    for i in range(len(g.d_w2b)):
        g.d_w2b[i] *= s


struct KleinLoKrGrads(Movable):
    var dbl: List[LoKrGrads]   # parallel to set.dbl (empty grads when inactive)
    var sgl: List[LoKrGrads]

    def __init__(out self, var dbl: List[LoKrGrads], var sgl: List[LoKrGrads]):
        self.dbl = dbl^
        self.sgl = sgl^


def _empty_lokr_grads() -> LoKrGrads:
    return LoKrGrads(
        List[Float32](), List[Float32](), List[Float32](),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
    )


def klein_lokr_chain_all(
    set: KleinLoKrSet,
    dbl_d_a: List[List[Float32]], dbl_d_b: List[List[Float32]],
    sgl_d_a: List[List[Float32]], sgl_d_b: List[List[Float32]],
) raises -> KleinLoKrGrads:
    if len(dbl_d_a) != len(set.dbl) or len(sgl_d_a) != len(set.sgl):
        raise Error("klein_lokr_chain_all: grad list count mismatch")
    var dbl = List[LoKrGrads]()
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            dbl.append(lokr_chain_carrier_grads(set.dbl[i], dbl_d_a[i], dbl_d_b[i]))
        else:
            dbl.append(_empty_lokr_grads())
    var sgl = List[LoKrGrads]()
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            sgl.append(lokr_chain_carrier_grads(set.sgl[i], sgl_d_a[i], sgl_d_b[i]))
        else:
            sgl.append(_empty_lokr_grads())
    return KleinLoKrGrads(dbl^, sgl^)


def klein_lokr_grad_norm(g: KleinLoKrGrads) -> Float64:
    var s = Float64(0.0)
    for i in range(len(g.dbl)):
        s += _grads_sqsum(g.dbl[i])
    for i in range(len(g.sgl)):
        s += _grads_sqsum(g.sgl[i])
    return sqrt(s)


def klein_lokr_clip_grads(mut g: KleinLoKrGrads, clip_scale: Float32):
    if clip_scale == Float32(1.0):
        return
    for i in range(len(g.dbl)):
        _grads_scale(g.dbl[i], clip_scale)
    for i in range(len(g.sgl)):
        _grads_scale(g.sgl[i], clip_scale)


def klein_lokr_adamw_step(
    mut set: KleinLoKrSet, g: KleinLoKrGrads, t: Int, lr: Float32,
    beta1: Float32, beta2: Float32, eps: Float32, weight_decay: Float32,
) raises:
    for i in range(len(set.dbl)):
        if set.dbl_active[i]:
            lokr_adamw(set.dbl[i], g.dbl[i], t, lr, beta1, beta2, eps, weight_decay)
    for i in range(len(set.sgl)):
        if set.sgl_active[i]:
            lokr_adamw(set.sgl[i], g.sgl[i], t, lr, beta1, beta2, eps, weight_decay)


# ── moving-factor evidence for the train-smoke gate ──────────────────────────
def klein_lokr_trainable_l1(set: KleinLoKrSet) -> Float64:
    """L1 over every TRAINABLE LoKr factor tensor (the smoke prints its growth:
    the zero legs must move off zero for training to be real)."""
    var s = Float64(0.0)
    for i in range(len(set.dbl)):
        if not set.dbl_active[i]:
            continue
        ref lo = set.dbl[i]
        for j in range(len(lo.w1)):
            var v = Float64(lo.w1[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
        for j in range(len(lo.w1a)):
            var v = Float64(lo.w1a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
        for j in range(len(lo.w1b)):
            var v = Float64(lo.w1b[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
        for j in range(len(lo.w2)):
            var v = Float64(lo.w2[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
        for j in range(len(lo.w2b)):
            var v = Float64(lo.w2b[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    for i in range(len(set.sgl)):
        if not set.sgl_active[i]:
            continue
        ref lo = set.sgl[i]
        for j in range(len(lo.w1)):
            var v = Float64(lo.w1[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
        for j in range(len(lo.w1a)):
            var v = Float64(lo.w1a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
        for j in range(len(lo.w1b)):
            var v = Float64(lo.w1b[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
        for j in range(len(lo.w2)):
            var v = Float64(lo.w2[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
        for j in range(len(lo.w2a)):
            var v = Float64(lo.w2a[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
        for j in range(len(lo.w2b)):
            var v = Float64(lo.w2b[j].cast[DType.float32]())
            s += v if v >= 0.0 else -v
    return s


# Sum |w2-side zero-leg| only (starts EXACTLY 0; must be >0 after training).
def klein_lokr_zero_leg_l1(set: KleinLoKrSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.dbl)):
        if not set.dbl_active[i]:
            continue
        ref lo = set.dbl[i]
        if lo.w2_factored:
            for j in range(len(lo.w2b)):
                var v = Float64(lo.w2b[j].cast[DType.float32]())
                s += v if v >= 0.0 else -v
        else:
            for j in range(len(lo.w2)):
                var v = Float64(lo.w2[j].cast[DType.float32]())
                s += v if v >= 0.0 else -v
    for i in range(len(set.sgl)):
        if not set.sgl_active[i]:
            continue
        ref lo = set.sgl[i]
        if lo.w2_factored:
            for j in range(len(lo.w2b)):
                var v = Float64(lo.w2b[j].cast[DType.float32]())
                s += v if v >= 0.0 else -v
        else:
            for j in range(len(lo.w2)):
                var v = Float64(lo.w2[j].cast[DType.float32]())
                s += v if v >= 0.0 else -v
    return s


# ── save: upstream LyCORIS wrapper naming for the diffusers Flux2 transformer ─
# create_lycoris names each module f"lycoris_{module_path}".replace(".","_")
# (wrapper.py:152/382) where module_path is relative to the WRAPPED module —
# SimpleTuner wraps model.get_trained_component() = the transformer itself, so
# klein's "transformer.transformer_blocks.0.attn.to_q" maps to
# "lycoris_transformer_blocks_0_attn_to_q".
def klein_lokr_prefix(kind_double: Bool, block_idx: Int, slot: Int) -> String:
    if kind_double:
        var b = String("lycoris_transformer_blocks_") + String(block_idx)
        if slot == 0:
            return b + "_attn_to_q"
        elif slot == 1:
            return b + "_attn_to_k"
        elif slot == 2:
            return b + "_attn_to_v"
        elif slot == 3:
            return b + "_attn_to_out_0"
        elif slot == 4:
            return b + "_ff_linear_in"
        elif slot == 5:
            return b + "_ff_linear_out"
        elif slot == 6:
            return b + "_attn_add_q_proj"
        elif slot == 7:
            return b + "_attn_add_k_proj"
        elif slot == 8:
            return b + "_attn_add_v_proj"
        elif slot == 9:
            return b + "_attn_to_add_out"
        elif slot == 10:
            return b + "_ff_context_linear_in"
        return b + "_ff_context_linear_out"
    var s = String("lycoris_single_transformer_blocks_") + String(block_idx)
    if slot == 0:
        return s + "_attn_to_qkv_mlp_proj"
    return s + "_attn_to_out"


def save_klein_lokr(set: KleinLoKrSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLoKr]()
    for bi in range(set.num_double):
        for slot in range(_DBL_SLOTS):
            var flat = bi * _DBL_SLOTS + slot
            if set.dbl_active[flat]:
                named.append(NamedLoKr(
                    klein_lokr_prefix(True, bi, slot), set.dbl[flat].copy()
                ))
    for bi in range(set.num_single):
        for slot in range(_SGL_SLOTS):
            var flat = bi * _SGL_SLOTS + slot
            if set.sgl_active[flat]:
                named.append(NamedLoKr(
                    klein_lokr_prefix(False, bi, slot), set.sgl[flat].copy()
                ))
    return save_lokr_peft(named, path, ctx)


# ── org-weight stats for --init_lokr_norm (perturbed-normal init) ────────────
@fieldwise_init
struct LokrOrgStats(Copyable, Movable):
    var norm: Float64
    var mean: Float64
    var std: Float64    # unbiased (n-1), torch.std default


def _stats_over_bf16_rows(
    st: SafeTensors, key: String, row0: Int, row1: Int
) raises -> LokrOrgStats:
    var info = st.tensor_info(key)
    if info.dtype != STDtype.BF16:
        raise Error(String("lokr org stats: expected BF16 for ") + key)
    if len(info.shape) != 2:
        raise Error(String("lokr org stats: expected 2D for ") + key)
    var rows = info.shape[0]
    var cols = info.shape[1]
    if row1 > rows or row0 < 0 or row0 >= row1:
        raise Error(String("lokr org stats: bad row slice for ") + key)
    var bytes = st.tensor_bytes(key)
    var bp = bytes.unsafe_ptr().bitcast[BFloat16]()
    var n = (row1 - row0) * cols
    var base = row0 * cols
    var s = Float64(0.0)
    var ss = Float64(0.0)
    for i in range(n):
        var v = Float64(bp[base + i].cast[DType.float32]())
        s += v
        ss += v * v
    var mean = s / Float64(n)
    var varu = (ss - Float64(n) * mean * mean) / Float64(n - 1) if n > 1 else Float64(0.0)
    if varu < 0.0:
        varu = 0.0
    return LokrOrgStats(sqrt(ss), mean, sqrt(varu))


def _read_bf16_rows_f32(
    st: SafeTensors, key: String, row0: Int, row1: Int
) raises -> List[Float32]:
    var info = st.tensor_info(key)
    if info.dtype != STDtype.BF16:
        raise Error(String("klein base weight: expected BF16 for ") + key)
    if len(info.shape) != 2:
        raise Error(String("klein base weight: expected 2D for ") + key)
    var rows = info.shape[0]
    var cols = info.shape[1]
    if row1 > rows or row0 < 0 or row0 >= row1:
        raise Error(String("klein base weight: bad row slice for ") + key)
    var bytes = st.tensor_bytes(key)
    var bp = bytes.unsafe_ptr().bitcast[BFloat16]()
    var n = (row1 - row0) * cols
    var base = row0 * cols
    var out = List[Float32]()
    for i in range(n):
        out.append(bp[base + i].cast[DType.float32]())
    return out^


def klein_lokr_base_weight_f32(
    st: SafeTensors, kind_double: Bool, block_idx: Int, slot: Int, D: Int, F: Int
) raises -> List[Float32]:
    """Read one frozen Klein projection weight as [out,in] F32 host values.

    This mirrors klein_lokr_org_stats' key and fused-row mapping and is used by
    full-delta carriers that need W_orig (DoRA/OFT). The checkpoint storage
    boundary remains BF16; F32 here is host-side adapter math.
    """
    if kind_double:
        var stream = String("img") if slot <= 5 else String("txt")
        var s = slot if slot <= 5 else slot - 6
        var b = String("double_blocks.") + String(block_idx) + "." + stream
        if s <= 2:
            return _read_bf16_rows_f32(st, b + "_attn.qkv.weight", s * D, (s + 1) * D)
        if s == 3:
            return _read_bf16_rows_f32(st, b + "_attn.proj.weight", 0, D)
        if s == 4:
            return _read_bf16_rows_f32(st, b + "_mlp.0.weight", 0, 2 * F)
        return _read_bf16_rows_f32(st, b + "_mlp.2.weight", 0, D)
    var sb = String("single_blocks.") + String(block_idx)
    if slot == 0:
        return _read_bf16_rows_f32(st, sb + ".linear1.weight", 0, 3 * D + 2 * F)
    return _read_bf16_rows_f32(st, sb + ".linear2.weight", 0, D)


def klein_lokr_org_stats(
    st: SafeTensors, kind_double: Bool, block_idx: Int, slot: Int, D: Int, F: Int
) raises -> LokrOrgStats:
    """Per-slot org-weight stats from the BFL klein checkpoint layout
    (double_blocks.{i}.{img|txt}_attn.qkv fused rows / .proj / mlp.0 / mlp.2;
    single_blocks.{i}.linear1/linear2)."""
    if kind_double:
        var stream = String("img") if slot <= 5 else String("txt")
        var s = slot if slot <= 5 else slot - 6
        var b = String("double_blocks.") + String(block_idx) + "." + stream
        if s <= 2:
            # q/k/v rows of the fused qkv [3D, D]
            return _stats_over_bf16_rows(st, b + "_attn.qkv.weight", s * D, (s + 1) * D)
        if s == 3:
            return _stats_over_bf16_rows(st, b + "_attn.proj.weight", 0, D)
        if s == 4:
            return _stats_over_bf16_rows(st, b + "_mlp.0.weight", 0, 2 * F)
        return _stats_over_bf16_rows(st, b + "_mlp.2.weight", 0, D)
    var sb = String("single_blocks.") + String(block_idx)
    if slot == 0:
        return _stats_over_bf16_rows(st, sb + ".linear1.weight", 0, 3 * D + 2 * F)
    return _stats_over_bf16_rows(st, sb + ".linear2.weight", 0, D)


def klein_lokr_apply_perturbed_init(
    mut set: KleinLoKrSet, st: SafeTensors, D: Int, F: Int,
    scale: Float64, seed: UInt64,
) raises:
    """SimpleTuner --init_lokr_norm over the whole set (trainer.py:2757-2761).
    Requires every active module both-full (full_matrix), like upstream."""
    var s = seed
    for bi in range(set.num_double):
        for slot in range(_DBL_SLOTS):
            var flat = bi * _DBL_SLOTS + slot
            if set.dbl_active[flat]:
                var stt = klein_lokr_org_stats(st, True, bi, slot, D, F)
                lokr_perturbed_normal_init(set.dbl[flat], stt.norm, stt.mean, stt.std, scale, s)
            s += 1
    for bi in range(set.num_single):
        for slot in range(_SGL_SLOTS):
            var flat = bi * _SGL_SLOTS + slot
            if set.sgl_active[flat]:
                var stt = klein_lokr_org_stats(st, False, bi, slot, D, F)
                lokr_perturbed_normal_init(set.sgl[flat], stt.norm, stt.mean, stt.std, scale, s)
            s += 1
